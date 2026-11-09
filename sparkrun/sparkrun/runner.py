"""The run loop: durable, verified, budgeted agent execution for small models.

Each step: render compact task memory -> ask the current model for ONE JSON action ->
check policy -> run the tool (side effects at most once) -> record the result.
When the model says "done", independent checks decide. If a small model keeps
failing (bad output, failed verification, repeating itself), the run escalates to
the next model in the list. Every change is an event in the journal, so a run that
crashes resumes where it stopped.
"""
from __future__ import annotations

import ast
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .journal import Journal, stable_hash
from .memory import TaskState
from .models import Model, PermanentError
from .tools import Policy, Tool, TransientError
from .verify import Check, Evidence, Verdict, run_checks

SYSTEM_PROMPT = """You are a careful agent working toward a GOAL. Reply with exactly ONE JSON object and nothing else.
To use a tool: {{"tool": "<name>", "args": {{...}}, "remember": "<optional fact worth keeping>"}}
When the goal is fully achieved: {{"done": "<one-line summary of what you did>"}}
Your work is checked independently after you say done, so only say done when it is true.
Never repeat an action that already succeeded.{instructions}

Tools:
{tools}"""

TERMINAL = {"verified", "unverified", "failed", "budget_exceeded", "stuck"}
# Events that must survive a power cut: side-effect boundaries, approvals and outcomes.
CRITICAL = {"approval_requested", "approved", "paused", "finished"}


# --------------------------------------------------------------------------- parsing

class ParseError(ValueError):
    pass


def parse_action(text: str) -> dict:
    """Pull one action out of a model reply.

    Small models wrap JSON in prose or code fences, use single quotes, or rename keys.
    All of that is accepted; anything else is a ParseError the model is told about.
    """
    blob = _first_object(text)
    if blob is None:
        raise ParseError("no JSON object found")
    obj: Any = None
    for loader in (json.loads, ast.literal_eval):
        try:
            obj = loader(blob)
            break
        except Exception:
            continue
    if not isinstance(obj, dict):
        raise ParseError("the JSON object could not be parsed")
    remember = obj.get("remember")
    for key in ("done", "final", "answer", "finish"):
        if key in obj:
            return {"done": str(obj[key]), "remember": remember}
    name = obj.get("tool") or obj.get("action") or obj.get("name")
    args = obj.get("args", obj.get("arguments", obj.get("input", {})))
    if not isinstance(name, str) or not name:
        raise ParseError('expected "tool" or "done"')
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ParseError('"args" must be an object')
    return {"tool": name, "args": args, "remember": remember}


def _first_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, quote, escaped = 0, "", False
        for i in range(start, len(text)):
            c = text[i]
            if quote:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == quote:
                    quote = ""
            elif c in "\"'":
                quote = c
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- state

def _short(value: Any, limit: int = 600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def _call_str(tool: str, args: dict) -> str:
    return _short(f"{tool}({json.dumps(args, ensure_ascii=False, default=str)})", 200)


@dataclass
class _State:
    """Everything about a run, rebuilt by replaying its journal events in order."""

    task: TaskState = field(default_factory=TaskState)
    steps: int = 0
    cost: float = 0.0
    tokens: int = 0
    level: int = 0
    escalations: int = 0
    resumes: int = 0
    bad_replies: int = 0
    verify_failures: int = 0
    replayed: int = 0
    done_results: dict = field(default_factory=dict)
    started: dict = field(default_factory=dict)
    approved: set = field(default_factory=set)
    pending: dict | None = None
    recent_keys: list = field(default_factory=list)
    answer: str | None = None
    verdict: Verdict | None = None
    finished: dict | None = None

    @classmethod
    def from_journal(cls, journal: Journal) -> "_State":
        s = cls()
        for event in journal:
            s.apply(event)
        return s

    def apply(self, e: dict) -> None:
        k = e["kind"]
        if k == "run_started":
            self.task.goal = e["goal"]
        elif k == "resumed":
            self.resumes += 1
        elif k == "model_call":
            self.steps += 1
            self.cost += e.get("cost_usd", 0.0)
            self.tokens += e.get("tokens", 0)
        elif k == "bad_reply":
            self.bad_replies += 1
            self.task.add_step(f"Your last reply was rejected ({e['error']}). Reply with ONE JSON object.")
        elif k == "feedback":
            self.task.add_step(e["text"])
        elif k == "remember":
            self.task.add_fact(e["text"])
        elif k == "escalated":
            self.level = e["level"]
            self.escalations += 1
            self.bad_replies = 0
        elif k == "tool_started":
            self.started[e["key"]] = e
            self.recent_keys.append(e["key"])
        elif k == "tool_replayed":
            self.replayed += 1
            self.recent_keys.append(e["key"])
            self.task.add_step(f"{_call_str(e['tool'], e['args'])} already succeeded earlier, so it was "
                                    f"not run again. Result: {e['result']}")
        elif k == "tool_result":
            self.started.pop(e["key"], None)
            self.bad_replies = 0
            if e["side_effect"] and e["ok"]:
                self.done_results[e["key"]] = e["result"]
            status = "OK" if e["ok"] else "ERROR"
            self.task.add_step(f"{_call_str(e['tool'], e['args'])} -> {status}: {e['result']}")
            if self.pending and self.pending["key"] == e["key"]:
                self.pending = None
        elif k == "approval_requested":
            self.pending = {key: e[key] for key in ("key", "tool", "args", "reason", "note") if key in e}
        elif k == "approved":
            self.approved.add(e["key"])
        elif k == "claimed_done":
            self.answer = e["answer"]
            self.bad_replies = 0
        elif k == "verified":
            self.verdict = Verdict(e["passed"], [Evidence(**ev) for ev in e["evidence"]])
            if e["passed"]:
                self.task.open_items = []
            else:
                self.verify_failures += 1
                self.task.open_items = [f"{f.check}: {f.detail}" for f in self.verdict.failures]
                self.task.add_step("You said done, but independent checks failed. "
                                        "Fix everything under MUST FIX, then say done again.")
        elif k == "finished":
            self.finished = e

    def is_repeating(self, key: str, limit: int) -> bool:
        window = self.recent_keys[-2 * limit:]
        return window.count(key) >= limit - 1


# --------------------------------------------------------------------------- results

@dataclass
class Result:
    run_id: str
    status: str
    """verified | unverified | failed | budget_exceeded | stuck |
    needs_approval | needs_reconcile | model_unavailable (the last three can be resumed)"""
    answer: str | None
    verdict: Verdict | None
    steps: int
    cost_usd: float
    tokens: int
    escalations: int
    duplicates_prevented: int
    reason: str = ""
    pending: dict | None = None
    journal: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "verified"


def approve_pending(journal_path: str | Path, key: str | None = None) -> dict:
    """Approve the action a paused run is waiting on. Returns what was approved."""
    j = Journal(journal_path)
    s = _State.from_journal(j)
    if key is None:
        if not s.pending:
            raise ValueError("this run is not waiting for approval")
        key = s.pending["key"]
    j.append("approved", key=key, critical=True)
    j.close()
    return s.pending or {"key": key}


# --------------------------------------------------------------------------- agent

class Agent:
    """A reliable agent loop.

    models:  one model, or a list from cheapest to strongest. The run starts on the first
             and escalates only when it fails.
    tools:   Tool objects (see @tool).
    checks:  independent verifiers. Without checks a finished run is "unverified".
    """

    def __init__(self, models: Model | Sequence[Model], tools: Iterable[Tool], *,
                 checks: Iterable[Check] = (), policy: Policy | None = None,
                 runs_dir: str | Path = "runs", context_chars: int = 6000,
                 max_tokens: int = 384, instructions: str = "", durability: str = "side_effects",
                 on_event: Callable[[dict], Any] | None = None):
        self.models = list(models) if isinstance(models, (list, tuple)) else [models]
        if not self.models:
            raise ValueError("at least one model is required")
        self.tools = {t.name: t for t in tools}
        self.checks = list(checks)
        self.policy = policy or Policy()
        self.runs_dir = Path(runs_dir)
        self.context_chars = context_chars
        self.max_tokens = max_tokens
        if durability not in Journal.DURABILITY:
            raise ValueError(f"durability must be one of {Journal.DURABILITY}")
        self.durability = durability
        self.on_event = on_event
        """Called with every journal event as it is written (mirror to a bus, a UI, metrics)."""
        extra = f"\n{instructions.strip()}" if instructions.strip() else ""
        specs = "\n".join(t.spec() for t in self.tools.values()) or "(none)"
        self.system = SYSTEM_PROMPT.format(instructions=extra, tools=specs)

    # public API -------------------------------------------------------------

    def run(self, goal: str | None = None, *, run_id: str | None = None) -> Result:
        """Start a run, or resume one by passing its run_id."""
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        j = Journal(self.runs_dir / run_id / "journal.jsonl", self.durability)
        try:
            return self._start(j, run_id, goal)
        finally:
            j.close()

    def run_many(self, goals: Iterable[str], *, workers: int = 8) -> list[Result]:
        """Run independent goals in parallel. Agents spend most of their time waiting on
        model calls, so threads give near-linear speedups. Models must be thread-safe
        (OpenAICompatible and ScriptedModel are)."""
        from .team import parallel_map
        return parallel_map(lambda g: self.run(g), goals, workers)

    def _start(self, j: Journal, run_id: str, goal: str | None) -> Result:
        if not j.events:
            if not goal:
                raise ValueError("a goal is required to start a new run")
            j.append("run_started", goal=goal, models=[m.name for m in self.models],
                     tools=sorted(self.tools), checks=len(self.checks))
        s = _State.from_journal(j)
        if s.finished:
            return self._result(j, run_id, s)
        if s.steps or s.pending:
            self._emit(j, s, "resumed")
        if not s.pending:  # a side effect started before a crash may or may not have happened
            for key, ev in list(s.started.items()):
                if ev.get("side_effect"):
                    self._emit(j, s, "approval_requested", key=key, tool=ev["tool"], args=ev["args"],
                               reason="needs_reconcile",
                               note="Started before a crash and may already have happened. "
                                    "Check, then approve to run it again.")
                    break
        return self._loop(j, run_id, s)

    def approve(self, run_id: str, key: str | None = None) -> dict:
        return approve_pending(self.runs_dir / run_id / "journal.jsonl", key)

    # loop -------------------------------------------------------------------

    def _loop(self, j: Journal, run_id: str, s: _State) -> Result:
        p = self.policy
        t0 = time.monotonic()
        while True:
            if s.pending:
                if s.pending["key"] not in s.approved:
                    self._emit(j, s, "paused", status=s.pending["reason"])
                    return self._result(j, run_id, s, s.pending["reason"], s.pending.get("note", ""))
                self._execute(j, s, s.pending["tool"], s.pending["args"], s.pending["key"])
                continue

            if s.steps >= p.max_steps:
                return self._finish(j, run_id, s, "budget_exceeded", f"reached {p.max_steps} steps")
            if p.max_cost_usd is not None and s.cost >= p.max_cost_usd:
                return self._finish(j, run_id, s, "budget_exceeded", f"spent ${s.cost:.4f}")
            if p.max_seconds is not None and time.monotonic() - t0 >= p.max_seconds:
                return self._finish(j, run_id, s, "budget_exceeded", f"ran {p.max_seconds}s")

            model = self.models[s.level]
            messages = [{"role": "system", "content": self.system},
                        {"role": "user", "content": s.task.render(self.context_chars)}]
            try:
                text, usage = self._call(model, messages)
            except PermanentError as e:
                return self._finish(j, run_id, s, "failed", f"model error: {e}")
            except TransientError as e:
                self._emit(j, s, "paused", status="model_unavailable", reason=str(e))
                return self._result(j, run_id, s, "model_unavailable", str(e))
            self._emit(j, s, "model_call", step=s.steps + 1, model=model.name, level=s.level,
                       reply=text[:4000], tokens=usage.tokens, cost_usd=usage.cost_usd)

            try:
                action = parse_action(text)
            except ParseError as e:
                self._emit(j, s, "bad_reply", error=str(e), reply=text[:300])
                if s.bad_replies >= 2:
                    self._escalate(j, s, "two invalid replies in a row")
                continue
            if action.get("remember"):
                self._emit(j, s, "remember", text=_short(action["remember"], 300))

            if "done" in action:
                result = self._on_done(j, run_id, s, action["done"])
                if result:
                    return result
                continue

            name, args = action["tool"], action["args"]
            tool = self.tools.get(name)
            if tool is None or (p.allowed_tools is not None and name not in p.allowed_tools):
                allowed = sorted(self.tools if p.allowed_tools is None else p.allowed_tools & set(self.tools))
                self._emit(j, s, "feedback", text=f"Tool '{name}' is not available. Use one of: {', '.join(allowed)}.")
                continue
            problem = tool.check_args(args)
            if problem:
                self._emit(j, s, "feedback", text=problem)
                continue

            key = stable_hash([name, args])
            if s.is_repeating(key, p.repeat_limit):
                if s.level < len(self.models) - 1:
                    self._escalate(j, s, f"repeated {name} with the same arguments")
                    self._emit(j, s, "feedback", text=f"You keep calling {_call_str(name, args)}. "
                                                       "It will not help. Try a different approach.")
                    continue
                return self._finish(j, run_id, s, "stuck", f"kept repeating {_call_str(name, args)}")
            if tool.side_effect and key in s.done_results:
                self._emit(j, s, "tool_replayed", key=key, tool=name, args=args, result=s.done_results[key])
                continue
            gated = p.gate(name, args) if p.gate and key not in s.approved else None
            if (tool.needs_approval or gated) and key not in s.approved:
                self._emit(j, s, "approval_requested", key=key, tool=name, args=args, reason="needs_approval",
                           note=gated or f"Approve {_call_str(name, args)} to continue.")
                continue
            self._execute(j, s, name, args, key)

    def _on_done(self, j: Journal, run_id: str, s: _State, answer: str) -> Result | None:
        self._emit(j, s, "claimed_done", answer=answer)
        if not self.checks:
            return self._finish(j, run_id, s, "unverified", "no checks configured")
        verdict = run_checks(self.checks, {"answer": answer, "run_id": run_id, "results": dict(s.done_results)})
        self._emit(j, s, "verified", passed=verdict.passed, evidence=[asdict(e) for e in verdict.evidence])
        if verdict.passed:
            return self._finish(j, run_id, s, "verified", "all checks passed")
        if s.verify_failures >= self.policy.max_verify_failures:
            return self._finish(j, run_id, s, "failed", f"checks failed {s.verify_failures} times")
        self._escalate(j, s, "verification failed")
        return None

    def _execute(self, j: Journal, s: _State, name: str, args: dict, key: str) -> None:
        tool = self.tools[name]
        self._emit(j, s, "tool_started", key=key, tool=name, args=args, side_effect=tool.side_effect)
        attempt = 0
        while True:
            try:
                out, ok = tool(**args), True
                break
            except TransientError as e:
                if attempt < self.policy.max_tool_retries:
                    attempt += 1
                    time.sleep(min(self.policy.backoff_s * 2 ** attempt, 30))
                    continue
                out, ok = f"TransientError: {e}", False
                break
            except Exception as e:
                out, ok = f"{type(e).__name__}: {e}", False
                break
        self._emit(j, s, "tool_result", key=key, tool=name, args=args, side_effect=tool.side_effect,
                   ok=ok, result=_short(out), attempts=attempt + 1)

    def _call(self, model: Model, messages: list[dict]):
        for attempt in range(self.policy.max_model_retries + 1):
            try:
                return model.complete(messages, max_tokens=self.max_tokens)
            except TransientError:
                if attempt == self.policy.max_model_retries:
                    raise
                time.sleep(min(self.policy.backoff_s * 2 ** attempt, 30))
        raise AssertionError("unreachable")

    # helpers ----------------------------------------------------------------

    def _emit(self, j: Journal, s: _State, kind: str, **data: Any) -> None:
        critical = kind in CRITICAL or (kind in ("tool_started", "tool_result") and data.get("side_effect", False))
        event = j.append(kind, critical=critical, **data)
        s.apply(event)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # an observer must never break a run
                pass

    def _escalate(self, j: Journal, s: _State, reason: str) -> None:
        if s.level < len(self.models) - 1:
            self._emit(j, s, "escalated", level=s.level + 1, model=self.models[s.level + 1].name, reason=reason)

    def _finish(self, j: Journal, run_id: str, s: _State, status: str, reason: str) -> Result:
        self._emit(j, s, "finished", status=status, reason=reason, answer=s.answer)
        return self._result(j, run_id, s)

    def _result(self, j: Journal, run_id: str, s: _State, status: str | None = None, reason: str = "") -> Result:
        if s.finished:
            status, reason = s.finished["status"], s.finished.get("reason", "")
        return Result(run_id, status or "in_progress", s.answer, s.verdict, s.steps, round(s.cost, 6),
                      s.tokens, s.escalations, s.replayed, reason, s.pending, str(j.path))
