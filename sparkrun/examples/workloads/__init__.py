"""Example workloads: the same infrastructure applied to different kinds of agents.

Each module exposes TITLE, DOMAIN, STYLE (how it plugs into sparkrun), SUMMARY, and
run(runs_dir, wrap=None) -> list of journal paths. measure() reads those journals the same
way for every workload, so the numbers are comparable across domains.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # examples/ (trip.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # sparkrun package

from sparkrun import Journal  # noqa: E402

from . import coding, research, support, travel  # noqa: E402
from .common import PRICES  # noqa: E402

ALL = [travel, coding, support, research]


def measure(journals: list[Path]) -> dict:
    """Comparable numbers for any set of sparkrun journals (Agent, Team or Harness)."""
    calls, caught, effects, statuses = [], [], [], []
    approvals = replays = 0
    for path in journals:
        events = list(Journal(path).events)
        agent = path.parent.name.split(".")[-1]
        said = None
        for e in events:
            k = e["kind"]
            if k == "model_call":
                calls.append(e)
            elif k == "claimed_done":
                said = e["answer"]
            elif k == "verified" and not e["passed"]:
                caught.append({"agent": agent, "said": said,
                               "why": "; ".join(f"{x['check']}: {x['detail']}" for x in e["evidence"] if not x["passed"])})
            elif k == "verified" and e["passed"] and caught and caught[-1]["agent"] == agent and "delivered" not in caught[-1]:
                caught[-1]["delivered"] = said
            elif k == "tool_result" and e.get("side_effect") and e["ok"]:
                effects.append({"agent": agent, "tool": e["tool"], "result": e["result"][:160]})
            elif k == "tool_replayed":
                replays += 1
            elif k == "approved":
                approvals += 1
            elif k == "finished":
                statuses.append(e["status"])
    cost = round(sum(e.get("cost_usd", 0.0) for e in calls), 6)
    strong_only = round(len(calls) * PRICES["strong"], 6)
    return {"status": "verified" if statuses and all(s == "verified" for s in statuses) else (statuses[-1] if statuses else "unfinished"),
            "agents": len(journals), "model_calls": len(calls),
            "cheap_calls": sum(1 for e in calls if e.get("level", 0) == 0),
            "cost_usd": cost, "strong_only_usd": strong_only,
            "savings_x": round(strong_only / cost, 1) if cost else None,
            "caught": caught, "side_effects": effects, "duplicates_prevented": replays, "approvals": approvals}
