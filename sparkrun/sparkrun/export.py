"""Export sparkrun runs to the rest of the Agent Reliability Stack.

    to_agentdiff(journal, level=None)   AgentDiff trajectory (schema v1). Diff a failed run against a
                                        verified one, or a small model's attempt against the stronger
                                        model it escalated to (escalation_pair).
    to_omniscope(journals, deps)        Omniscope trace events: the causal DAG across a team's agents,
                                        ready for its RiskEngine or POST /api/v1/traces.
    to_trace_viz(journal)               agent-trace-viz Trace dict: actions, observations, cost, outcome.

Exporters only read journal files, so they work on any past run, and they need none of
the target tools installed. Nothing is invented: every field comes from a journal event,
and fields a journal can't know are left empty.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .journal import Journal

Source = "str | Path | Journal | Sequence[dict]"

_SEARCH_WORDS = ("search", "find", "lookup", "query")
_RETRIEVE_WORDS = ("retrieve", "fetch", "rank")
_READ_WORDS = ("read", "open", "get_")


def _events(src) -> list[dict]:
    if isinstance(src, Journal):
        return list(src.events)
    if isinstance(src, (str, Path)):
        p = Path(src)
        return list(Journal(p / "journal.jsonl" if p.is_dir() else p).events)
    return list(src)


def _json(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)


def _step_type(tool: str) -> str:
    t = tool.lower()
    if any(w in t for w in _SEARCH_WORDS):
        return "search"
    if any(w in t for w in _RETRIEVE_WORDS):
        return "retrieve"
    if any(t.startswith(w) for w in _READ_WORDS):
        return "read"
    return "tool_call"


def _run_id(src) -> str:
    if isinstance(src, (str, Path)):
        p = Path(src)
        return (p if p.is_dir() else p.parent).name
    return "run"


def levels(src) -> list[int]:
    """Model levels a run used, in order (0 = the cheapest model)."""
    seen: list[int] = []
    for e in _events(src):
        if e["kind"] == "model_call" and e.get("level", 0) not in seen:
            seen.append(e.get("level", 0))
    return seen


# --------------------------------------------------------------------------- AgentDiff

def to_agentdiff(src, *, level: int | None = None, task_id: str | None = None,
                 expected: str | None = None, agent_name: str | None = None) -> dict:
    """One run (or the part of it a given model level handled) as an AgentDiff trajectory.

    With level=k, only events while the run was on its k-th model are included, and
    success means that model's own attempt passed verification.
    """
    events = _events(src)
    start = next((e for e in events if e["kind"] == "run_started"), None)
    if start is None:
        raise ValueError("not a sparkrun journal: no run_started event")
    goal, tools = start["goal"], start.get("tools", [])

    cur = 0
    picked: list[dict] = []
    for e in events:
        if e["kind"] == "escalated":
            cur = e["level"]
            continue
        if e["kind"] == "model_call":
            cur = e.get("level", cur)
        if level is None or cur == level:
            picked.append(e)

    models = [e["model"] for e in picked if e["kind"] == "model_call"]
    steps: list[dict] = []
    last_t = picked[0]["t"] if picked else start["t"]
    pending = {"tokens": 0, "in": None, "out": None}
    answer, success, status = "", False, "unfinished"

    def add(**step):
        nonlocal last_t
        t = step.pop("_t")
        step.setdefault("tokens", 0)
        step["latency_s"] = round(max(0.0, t - last_t), 4)
        last_t = t
        steps.append(step)

    for e in picked:
        k = e["kind"]
        if k == "model_call":
            pending = {"tokens": e.get("tokens", 0), "in": e.get("tokens_in"), "out": e.get("tokens_out")}
        elif k in ("tool_result", "tool_replayed"):
            step = {"type": _step_type(e["tool"]), "name": e["tool"], "input": _json(e["args"]),
                    "output": ("[replayed, not run again] " if k == "tool_replayed" else "") + _json(e["result"]),
                    "tokens": pending["tokens"], "_t": e["t"]}
            if pending["in"] is not None:
                step["input_tokens"], step["output_tokens"] = pending["in"], pending["out"]
            if k == "tool_result" and not e["ok"]:
                step["quality"], step["note"] = "bad", "tool returned an error"
            pending = {"tokens": 0, "in": None, "out": None}
            add(**step)
        elif k == "bad_reply":
            add(type="reason", name="invalid_reply", input="", output=e.get("reply", ""), quality="bad",
                note=e["error"], tokens=pending["tokens"], _t=e["t"])
            pending = {"tokens": 0, "in": None, "out": None}
        elif k == "feedback":
            add(type="reason", name="feedback", input="", output=e["text"], _t=e["t"])
        elif k == "claimed_done":
            answer = e["answer"]
            add(type="reason", name="claimed_done", input="", output=answer, tokens=pending["tokens"], _t=e["t"])
            pending = {"tokens": 0, "in": None, "out": None}
        elif k == "verified":
            fails = [f"{ev['check']}: {ev['detail']}" for ev in e["evidence"] if not ev["passed"]]
            add(type="reason", name="verification", input="independent checks",
                output="all checks passed" if e["passed"] else "; ".join(fails),
                quality="good" if e["passed"] else "bad", _t=e["t"])
            success = bool(e["passed"])
        elif k == "approval_requested":
            add(type="plan", name="awaiting_approval", input=_json(e.get("args", {})), output=e.get("note", ""), _t=e["t"])
        elif k == "finished":
            status = e["status"]

    if level is None:
        success = status == "verified"
    if not answer and steps and steps[-1]["name"] == "claimed_done":
        answer = steps[-1]["output"]
    if steps and steps[-1]["name"] == "claimed_done":
        steps.pop()  # becomes the answer step
    steps.append({"type": "answer", "name": "answer", "input": "", "output": answer or f"(no answer: {status})",
                  "tokens": 0, "latency_s": 0.0})
    for i, s in enumerate(steps):
        s["index"] = i

    tokens_in = sum(e.get("tokens_in") or 0 for e in picked if e["kind"] == "model_call")
    tokens_out = sum(e.get("tokens_out") or 0 for e in picked if e["kind"] == "model_call")
    run_id = _run_id(src)
    return {
        "schema_version": 1,
        "trace_id": f"sparkrun-{run_id}" + ("" if level is None else f"-L{level}"),
        "started_at": start["t"],
        "agent": {"name": agent_name or (models[0] if models else "sparkrun-agent"),
                  "model": models[0] if models else "unknown", "version": "sparkrun"},
        "task": {"id": task_id or run_id, "prompt": goal, "expected": expected},
        "outcome": {"success": bool(success), "answer": answer, "score": 1.0 if success else 0.0},
        "totals": {"input_tokens": tokens_in, "output_tokens": tokens_out,
                   "cost_usd": round(sum(e.get("cost_usd", 0.0) for e in picked if e["kind"] == "model_call"), 6),
                   "latency_s": round((picked[-1]["t"] - picked[0]["t"]) if picked else 0.0, 4)},
        "tools": [{"name": t} for t in tools],
        "steps": steps,
    }


def escalation_pair(src, *, task_id: str | None = None) -> tuple[dict, dict]:
    """The cheapest model's attempt and the final model's attempt on the same task, as two
    AgentDiff trajectories. Every escalation is a natural A/B pair: same goal, same tools."""
    lv = levels(src)
    if len(lv) < 2:
        raise ValueError("this run never escalated, so there is no pair to compare")
    tid = task_id or _run_id(src)
    return to_agentdiff(src, level=lv[0], task_id=tid), to_agentdiff(src, level=lv[-1], task_id=tid)


# --------------------------------------------------------------------------- Omniscope

_OMNI_TYPES = {"run_started": "agent_start", "finished": "agent_end", "model_call": "llm_call",
               "tool_result": "tool_call", "tool_replayed": "tool_call", "verified": "judge_evaluation",
               "escalated": "agent_decision", "approval_requested": "agent_decision", "approved": "agent_decision",
               "bad_reply": "error", "paused": "system_event"}


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def to_omniscope(journals: Mapping[str, object], *, deps: Mapping[str, Sequence[str]] | None = None,
                 trace_id: str = "sparkrun-trace") -> list[dict]:
    """Omniscope events for one agent or a whole team.

    journals: {agent_name: journal path or events}. deps: {agent_name: [agents it waited on]},
    so the first event of an agent is causally linked to the last event of each dependency.
    """
    deps = deps or {}
    per_agent: dict[str, list[dict]] = {}
    for agent, src in journals.items():
        out, prev, prev_t = [], None, None
        for e in _events(src):
            etype = _OMNI_TYPES.get(e["kind"])
            if etype is None:
                continue
            eid = f"{trace_id}:{agent}:{e['seq']}"
            ev = {"event_id": eid, "timestamp": _iso(e["t"]), "trace_id": trace_id, "span_id": eid,
                  "parent_span_id": prev, "event_type": etype, "agent_id": agent, "agent_name": agent,
                  "model_name": e.get("model"), "framework": "sparkrun",
                  "input_tokens": int(e.get("tokens_in") or 0), "output_tokens": int(e.get("tokens_out") or 0),
                  "latency_ms": round((e["t"] - prev_t) * 1000, 3) if prev_t is not None else 0.0,
                  "cost_usd": e.get("cost_usd"), "confidence_score": None, "input_preview": "",
                  "output_preview": "", "tool_name": None, "tool_input": {}, "tool_output": "",
                  "tool_success": True, "error_message": None, "causal_parents": [prev] if prev else [],
                  "data_dependencies": [], "tags": {"sparkrun_kind": e["kind"]}, "metadata": {}}
            k = e["kind"]
            if k == "run_started":
                ev["input_preview"] = e["goal"][:2000]
            elif k == "model_call":
                ev["output_preview"] = e.get("reply", "")[:2000]
            elif k in ("tool_result", "tool_replayed"):
                ev.update(tool_name=e["tool"], tool_input=e["args"], tool_output=_json(e["result"])[:2000],
                          tool_success=bool(e.get("ok", True)))
                if k == "tool_replayed":
                    ev["tags"]["replayed"] = "true"
            elif k == "verified":
                ev["confidence_score"] = 1.0 if e["passed"] else 0.0
                ev["output_preview"] = "; ".join(f"{x['check']}: {x['detail']}" for x in e["evidence"])[:2000]
            elif k == "escalated":
                ev["output_preview"] = f"escalated to {e.get('model')}: {e.get('reason')}"
            elif k == "approval_requested":
                ev["output_preview"] = e.get("note", "")
                ev["tags"]["needs_human"] = "true"
            elif k == "bad_reply":
                ev["error_message"] = e["error"]
            elif k == "finished":
                ev["output_preview"] = f"{e['status']}: {e.get('reason', '')}"
                ev["tags"]["status"] = e["status"]
            out.append(ev)
            prev, prev_t = eid, e["t"]
        per_agent[agent] = out
    for agent, evs in per_agent.items():
        if evs:
            parents = [per_agent[d][-1]["event_id"] for d in deps.get(agent, ()) if per_agent.get(d)]
            evs[0]["causal_parents"] = parents
            evs[0]["data_dependencies"] = parents
    return sorted((e for evs in per_agent.values() for e in evs), key=lambda e: e["timestamp"])


def team_omniscope(team, team_id: str) -> list[dict]:
    """Omniscope events for a sparkrun Team run, with the team's dependencies as causal edges."""
    journals = {name: team.runs_dir / f"{team_id}.{name}" / "journal.jsonl" for name in team.roles
                if (team.runs_dir / f"{team_id}.{name}" / "journal.jsonl").exists()}
    deps = {name: list(r.after) for name, r in team.roles.items()}
    return to_omniscope(journals, deps=deps, trace_id=team_id)


# --------------------------------------------------------------------------- agent-trace-viz

def to_trace_viz(src, *, benchmark: str = "sparkrun", strategy: str | None = None,
                 task_id: str | None = None) -> dict:
    """One run as an agent-trace-viz Trace dict (load it with agent_trace_viz.schema.Trace.from_dict)."""
    events = _events(src)
    start = next(e for e in events if e["kind"] == "run_started")
    fin = next((e for e in reversed(events) if e["kind"] == "finished"), None)
    steps, cost, last_reply, last_note = [], 0.0, "", None
    for e in events:
        if e["kind"] == "model_call":
            cost += e.get("cost_usd", 0.0)
            last_reply = e.get("reply", "")
        elif e["kind"] == "remember":
            last_note = e["text"]
        elif e["kind"] in ("tool_result", "tool_replayed"):
            ok = e.get("ok", True)
            steps.append({"idx": len(steps), "state_summary": last_reply[:300],
                          "action": {"tool": e["tool"], "args": e["args"], "cost": 0.0 if e["kind"] == "tool_replayed" else 1.0},
                          "observation": {"result": e["result"] if ok else None, "error": None if ok else e["result"]},
                          "hypothesis": last_note})
            last_note = None
    run_id = _run_id(src)
    status = fin["status"] if fin else "unfinished"
    return {"trace_id": f"sparkrun-{run_id}", "benchmark": benchmark,
            "strategy": strategy or " → ".join(dict.fromkeys(e["model"] for e in events if e["kind"] == "model_call")),
            "task_id": task_id or run_id, "steps": steps, "final_answer": fin.get("answer") if fin else None,
            "succeeded": status == "verified", "total_cost": round(cost, 6),
            "elapsed_ms": round((events[-1]["t"] - events[0]["t"]) * 1000, 3) if events else 0.0,
            "scores": {"verified": 1.0 if status == "verified" else 0.0,
                       "escalations": float(sum(e["kind"] == "escalated" for e in events)),
                       "duplicates_prevented": float(sum(e["kind"] == "tool_replayed" for e in events))},
            "parent_trace_id": None, "user_id": None, "session_id": None}


def write_json(obj, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return p
