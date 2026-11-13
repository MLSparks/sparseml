"""sparkrun's guarantees for an agent loop you already have.

`Agent` runs the loop for you. `Harness` wraps the loop you already have, whether it's
LangGraph, CrewAI, the OpenAI Agents SDK, your own while-loop, or a plain script:

    from sparkrun import Harness, verify

    with Harness("refund-1042", "Refund order 1042", checks=[refund_issued]) as h:
        refund = h.tool(issue_refund, side_effect=True, needs_approval=True)  # once, after approval
        lookup = h.tool(get_order)
        llm = h.model(my_model)            # optional: cost and token accounting
        ...                                # your framework calls refund(...), lookup(...), llm.complete(...)
        result = h.done(answer)            # independent checks decide: verified or failed

What you get, without changing how your agent decides what to do:
  * side-effect tools run at most once per run, across crashes and restarts
  * approvals: needs_approval tools, or a policy gate, pause the run until a person approves
  * independent checks on done(): your agent's claim is not the verdict
  * a crash-safe journal in the same format as Agent, so reports, exports and approvals work
  * cost per verified run, from the models you wrap
"""
from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .journal import Journal, stable_hash
from .models import Model, Usage
from .runner import CRITICAL, Result, _short, _State, approve_pending
from .verify import Check, Verdict, run_checks
from dataclasses import asdict


class PendingApproval(Exception):
    """Raised by a wrapped tool that needs a person's approval first. Approve with
    Harness.approve() (or `python -m sparkrun approve runs/<run_id>`), then run again."""

    def __init__(self, tool: str, args: dict, note: str, key: str):
        super().__init__(note)
        self.tool, self.args, self.note, self.key = tool, args, note, key


class Harness:
    def __init__(self, run_id: str, goal: str | None = None, *, checks: Iterable[Check] = (),
                 runs_dir: str | Path = "runs", gate: Callable[[str, dict], "str | None"] | None = None,
                 durability: str = "side_effects", on_event: Callable[[dict], Any] | None = None):
        self.run_id = run_id
        self.checks = list(checks)
        self.gate = gate
        self.on_event = on_event
        self.journal = Journal(Path(runs_dir) / run_id / "journal.jsonl", durability)
        if not self.journal.events:
            if not goal:
                raise ValueError("a goal is required to start a new run")
            self.journal.append("run_started", goal=goal, models=[], tools=[], checks=len(self.checks), harness=True)
        self.state = _State.from_journal(self.journal)
        self._values: dict[str, Any] = {}
        for e in self.journal.events:
            if e["kind"] == "tool_result" and e.get("side_effect") and e["ok"]:
                self._values[e["key"]] = e.get("value", e["result"])
        if self.state.steps or self.state.pending or self._values:
            self._emit("resumed")

    # context manager -------------------------------------------------------

    def __enter__(self) -> "Harness":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        try:
            if exc_type is PendingApproval and not self.state.finished:
                self._emit("paused", status=self.state.pending["reason"] if self.state.pending else "needs_approval")
                return True  # paused, not crashed: check h.status
            return False
        finally:
            self.journal.close()

    # wrapping -----------------------------------------------------------------

    def tool(self, fn: Callable | None = None, *, side_effect: bool = False, needs_approval: bool = False,
             name: str | None = None):
        """Wrap a function so calls are journaled; side effects run at most once per run."""

        def wrap(f: Callable) -> Callable:
            tname = name or getattr(f, "__name__", "tool")
            sig = inspect.signature(f)

            @functools.wraps(f)
            def call(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                a = json.loads(json.dumps(dict(bound.arguments), default=str))
                key = stable_hash([tname, a])
                if side_effect and key in self._values:
                    self._emit("tool_replayed", key=key, tool=tname, args=a, result=_short(self._values[key]))
                    return self._values[key]
                if side_effect and key in self.state.started:
                    if key not in self.state.approved:
                        note = f"{tname} started before a crash and may already have happened. Check, then approve to run it again."
                        self._emit("approval_requested", key=key, tool=tname, args=a, reason="needs_reconcile", note=note)
                        raise PendingApproval(tname, a, note, key)
                gated = self.gate(tname, a) if self.gate and key not in self.state.approved else None
                if (needs_approval or gated) and key not in self.state.approved:
                    note = gated or f"Approve {tname}({json.dumps(a)[:120]}) to continue."
                    self._emit("approval_requested", key=key, tool=tname, args=a, reason="needs_approval", note=note)
                    raise PendingApproval(tname, a, note, key)
                self._emit("tool_started", key=key, tool=tname, args=a, side_effect=side_effect)
                try:
                    out = f(*args, **kwargs)
                except Exception as e:
                    self._emit("tool_result", key=key, tool=tname, args=a, side_effect=side_effect, ok=False,
                               result=f"{type(e).__name__}: {e}", attempts=1)
                    raise
                value = json.loads(json.dumps(out, default=str))
                self._emit("tool_result", key=key, tool=tname, args=a, side_effect=side_effect, ok=True,
                           result=_short(out), value=value, attempts=1)
                if side_effect:
                    self._values[key] = value
                return out

            return call

        return wrap(fn) if fn is not None else wrap

    def model(self, model: Model, *, level: int = 0) -> Model:
        """Wrap a sparkrun Model so every call is counted (tokens, cost, escalation level)."""
        h = self

        class Recorded:
            name = model.name

            def complete(self, messages: list[dict], max_tokens: int = 512) -> tuple[str, Usage]:
                text, usage = model.complete(messages, max_tokens)
                h.model_call(model.name, reply=text, tokens_in=usage.prompt_tokens,
                             tokens_out=usage.completion_tokens, cost_usd=usage.cost_usd, level=level)
                return text, usage

        return Recorded()

    def model_call(self, model: str, *, reply: str = "", tokens_in: int = 0, tokens_out: int = 0,
                   cost_usd: float = 0.0, level: int = 0) -> None:
        """Record a model call made outside sparkrun (any SDK), for cost per verified run."""
        self._emit("model_call", step=self.state.steps + 1, model=model, level=level, reply=reply[:4000],
                   tokens=tokens_in + tokens_out, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd)

    def escalate(self, model: str, reason: str) -> None:
        """Record that your loop moved to a stronger model (shows up in reports and diffs)."""
        self._emit("escalated", level=self.state.level + 1, model=model, reason=reason)

    def note(self, text: str) -> None:
        self._emit("remember", text=_short(text, 300))

    # outcome -------------------------------------------------------------------

    def check(self, answer: str, **ctx: Any) -> Verdict | None:
        """Run the independent checks on an answer without ending the run, so your loop can
        retry (for example on a stronger model, after escalate()). None if there are no checks."""
        self._emit("claimed_done", answer=str(answer))
        if not self.checks:
            return None
        verdict = run_checks(self.checks, {"answer": str(answer), "run_id": self.run_id,
                                           "results": dict(self._values), **ctx})
        self._emit("verified", passed=verdict.passed, evidence=[asdict(e) for e in verdict.evidence])
        return verdict

    def done(self, answer: str, **ctx: Any) -> Result:
        """Your agent says it's finished. Independent checks decide whether it's verified."""
        if self.state.finished:
            return self.result
        verdict = self.check(answer, **ctx)
        if verdict is None:
            return self._finish("unverified", "no checks configured")
        return self._finish("verified" if verdict.passed else "failed",
                            "all checks passed" if verdict.passed else "checks failed")

    def fail(self, reason: str) -> Result:
        return self._finish("failed", reason)

    def approve(self, key: str | None = None) -> dict:
        approved = approve_pending(self.journal.path, key)
        self.journal = Journal(self.journal.path, self.journal.durability)
        self.state = _State.from_journal(self.journal)
        return approved

    @property
    def status(self) -> str:
        if self.state.finished:
            return self.state.finished["status"]
        if self.state.pending and self.state.pending["key"] not in self.state.approved:
            return self.state.pending["reason"]
        return "in_progress"

    @property
    def result(self) -> Result:
        s = self.state
        reason = s.finished.get("reason", "") if s.finished else ""
        return Result(self.run_id, self.status, s.answer, s.verdict, s.steps, round(s.cost, 6), s.tokens,
                      s.escalations, s.replayed, reason, s.pending, str(self.journal.path))

    # internals -----------------------------------------------------------------

    def _finish(self, status: str, reason: str) -> Result:
        self._emit("finished", status=status, reason=reason, answer=self.state.answer)
        return self.result

    def _emit(self, kind: str, **data: Any) -> None:
        critical = kind in CRITICAL or (kind in ("tool_started", "tool_result") and data.get("side_effect", False))
        event = self.journal.append(kind, critical=critical, **data)
        self.state.apply(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:
                pass
