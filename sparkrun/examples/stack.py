"""The Agent Reliability Stack, end to end, on one multi-agent travel run.

    python examples/stack.py --out stack-data                 # sparkrun alone (exports still written)
    python examples/stack.py --out stack-data --with-tools    # also run AgentDiff, Omniscope, ARES

1. Run    sparkrun runs a six-agent travel team (flights, hotels, guide, ranker, activities,
          booking). Cheap models first; escalation only when a check fails; booking waits
          for approval.
2. Trace  the run exports to an Omniscope causal DAG (scored by Omniscope's own RiskEngine)
          and to agent-trace-viz traces.
3. Diff   every escalation is a natural A/B pair (same task, cheap vs strong model). AgentDiff
          compares each pair and names the decisive step.
4. Eval   ARES groundedness and F1 are the guide's checks; the ARES leaderboard (if given)
          adds the quality-vs-cost frontier of 55 agentic-RAG configurations.

Writes stack.json (everything the Stack Explorer shows) plus the exports and AgentDiff
report pages. Tools that are not installed are skipped and marked as such; nothing is faked.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import trip  # noqa: E402
from sparkrun import Agent, CachedModel, Journal, Policy, ScriptedModel, __version__, compute, export, tool, verify  # noqa: E402

# --------------------------------------------------------------------------- helpers

def has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def compact(events: list[dict]) -> list[dict]:
    keep = ("kind", "seq", "t", "model", "level", "tool", "args", "ok", "result", "reason", "passed", "evidence",
            "status", "answer", "note", "error", "text", "tokens", "cost_usd", "reply")
    out = []
    for e in events:
        if e["kind"] in ("resumed", "tool_started", "paused"):
            continue
        c = {k: e[k] for k in keep if k in e}
        if "reply" in c:
            c["reply"] = c["reply"][:240]
        if isinstance(c.get("result"), str):
            c["result"] = c["result"][:400]
        out.append(c)
    return out


def lean_bench(steps: int = 400) -> dict:
    @tool
    def noop(i: int):
        """Do nothing."""
        return i

    script = [json.dumps({"tool": "noop", "args": {"i": i}}) for i in range(steps)] + [json.dumps({"done": "ok"})]
    with tempfile.TemporaryDirectory() as d:
        a = Agent(ScriptedModel(script), [noop], runs_dir=d, policy=Policy(max_steps=steps + 5, repeat_limit=10 ** 6))
        t0 = time.perf_counter()
        a.run("bench")
        per_step = (time.perf_counter() - t0) / steps * 1000

        def slow(messages):
            time.sleep(0.05)
            n = messages[1]["content"].count("- noop(")
            return json.dumps({"tool": "noop", "args": {"i": n}}) if n < 2 else json.dumps({"done": "ok"})

        par = Agent(ScriptedModel(slow), [noop], runs_dir=Path(d) / "p")
        t1 = time.perf_counter()
        par.run_many([f"t{i}" for i in range(32)], workers=1)
        seq = time.perf_counter() - t1
        t2 = time.perf_counter()
        par.run_many([f"u{i}" for i in range(32)], workers=32)
        many = time.perf_counter() - t2
    return {"overhead_ms_per_step": round(per_step, 3), "steps": steps,
            "parallel": {"tasks": 32, "model_latency_ms": 50, "sequential_s": round(seq, 2), "workers_32_s": round(many, 2)}}


def agentdiff_compare(a: Path, b: Path, out: Path, html: Path) -> dict | None:
    if not has("agentdiff"):
        return None
    p = subprocess.run([sys.executable, "-m", "agentdiff", "compare", str(a), str(b), "-o", str(out), "--html", str(html)],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not out.exists():
        print(p.stdout[-2000:], p.stderr[-2000:], file=sys.stderr)
        return None
    rep = json.loads(out.read_text())
    return {"verdict_card": rep.get("verdict_card", {}).get("lines", []),
            "divergences": [{k: d.get(k) for k in ("rank", "a_index", "b_index", "kind", "summary")}
                            for d in rep.get("divergences", [])][:6],
            "alignment": [{k: x.get(k) for k in ("a_index", "b_index", "op", "similarity")} for x in rep.get("alignment", [])],
            "attribution": rep.get("attribution"), "report_html": html.name}


def omniscope_risk(events: list[dict]) -> dict | None:
    if not (has("omniscope") and has("pydantic") and has("httpx")):
        return None
    from omniscope.server.models import TraceEvent
    from omniscope.server.risk import RiskEngine
    scores = RiskEngine().compute_all([TraceEvent(**e) for e in events])
    return {a: {k: v for k, v in s.model_dump().items() if k != "agent_id"} for a, s in scores.items()}


def ares_leaderboard(path: Path | None) -> list[dict] | None:
    if not path or not path.exists():
        return None
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({"name": r["name"], "phase": r["phase"], "f1": float(r["f1"]), "em": float(r["em"]),
                         "recall": float(r["recall"]), "grounding": float(r["grounding"]),
                         "latency_s": float(r["latency"]), "tokens": float(r["tokens"]), "prompt": r["prompt_variant"],
                         "k": int(r["retrieval_k"]), "decompose": r["decompose"] == "True", "reflect": r["reflect"] == "True"})
    return rows


def cost_story(team, first, again, first_s, again_s, first_hits, repeat_hits, journals) -> dict:
    """What the run cost, against sending every call to the strong model."""
    calls = [e for p in journals.values() for e in Journal(p).events if e["kind"] == "model_call"]
    cheap = sum(1 for e in calls if e.get("level", 0) == 0)
    strong_only = round(len(calls) * trip.PRICES["strong"], 6)
    return {"model_calls": len(calls), "cheap_calls": cheap, "first_run_usd": first.cost_usd,
            "strong_only_usd": strong_only, "savings_x": round(strong_only / first.cost_usd, 1) if first.cost_usd else None,
            "prices": trip.PRICES, "first_run_s": round(first_s, 3),
            "repeat": {"status": again.status, "usd": again.cost_usd, "seconds": round(again_s, 3),
                       "cache_hits": repeat_hits, "calls": again.steps},
            "first_cache_hits": first_hits}


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stack-data")
    ap.add_argument("--options", type=int, default=600)
    ap.add_argument("--with-tools", action="store_true", help="run AgentDiff/Omniscope/ARES if installed")
    ap.add_argument("--ares-leaderboard", type=Path, help="path to ares/results/leaderboard.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = Path(tempfile.mkdtemp(prefix="sparkrun-stack-"))

    # 1. Run ---------------------------------------------------------------
    req = trip.TripRequest(options=args.options)
    cache: dict = {}
    wrappers: list = []

    def cached(m):
        wrappers.append(CachedModel(m, cache))
        return wrappers[-1]

    def run_trip(runs_dir: Path):
        tm = trip.build_team(req, runs_dir, gate=trip.spending_gate, guide=True, wrap=cached)
        t = time.perf_counter()
        res, n = tm.run("trip"), 0
        while res.status == "needs_approval" and n < 4:
            tm.approve("trip", "booking")
            n += 1
            res = tm.run("trip")
        return tm, res, n, time.perf_counter() - t

    team, result, approvals, team_s = run_trip(runs)
    first_hits = sum(w.hits for w in wrappers)
    again_team, again, _, again_s = run_trip(runs / "repeat")  # the same trip requested again
    repeat_hits = sum(w.hits for w in wrappers) - first_hits
    print(f"travel team: {result.status}")

    journals = {name: runs / f"trip.{name}" / "journal.jsonl" for name in team.roles}
    b = team.board
    opt = b["options"][0]

    # 2. Trace ---------------------------------------------------------------
    omni = export.team_omniscope(team, "trip")
    traces = [export.to_trace_viz(p, benchmark="travel", task_id=f"trip.{n}") for n, p in journals.items()]
    export.write_json(omni, out / "omniscope-events.json")
    export.write_json({"traces": traces}, out / "agent-trace-viz.json")
    risk = omniscope_risk(omni) if args.with_tools else None

    # 3. Diff ----------------------------------------------------------------
    pairs = []
    for title, src, tid in [("Guide: answering from memory vs from the travel guide", journals["guide"], "travel-guide"),
                            ("Activities: claiming done vs actually planning", journals["activities"], "plan-activities")]:
        a, bb = export.escalation_pair(src, task_id=tid)
        slug = tid
        pa, pb = export.write_json(a, out / f"agentdiff-{slug}-a.json"), export.write_json(bb, out / f"agentdiff-{slug}-b.json")
        rep = agentdiff_compare(pa, pb, out / f"agentdiff-{slug}-report.json", out / f"agentdiff-{slug}.html") if args.with_tools else None
        pairs.append({"title": title, "task": tid, "a": a, "b": bb, "report": rep})

    # 4. Eval ----------------------------------------------------------------
    board_keep = {"option": opt, "bookings": b.get("bookings", []), "activities": b.get("activities", []),
                  "total": b.get("total"), "rank_stats": b.get("rank_stats"), "request": b.get("request"),
                  "options": b.get("options", [])[:5], "flights_found": len(b.get("flights", [])),
                  "hotels_found": len(b.get("hotels", []))}
    bundle = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "compute_backend": compute.backend(), "sparkrun": __version__,
                        "tools": {"agentdiff": bool(pairs[0]["report"]), "omniscope": risk is not None,
                                  "agent_trace_viz": has("agent_trace_viz"), "ares": verify.ARES_SOURCE}},
        "team": {"status": result.status, "seconds": round(team_s, 3), "approvals": approvals,
                 "steps": result.steps, "cost_usd": result.cost_usd,
                 "roles": [{"name": n, "after": list(r.after), "status": result.roles[n].status,
                            "steps": result.roles[n].steps, "cost_usd": result.roles[n].cost_usd,
                            "escalations": result.roles[n].escalations,
                            "models": list(dict.fromkeys(e["model"] for e in Journal(journals[n]).events if e["kind"] == "model_call")),
                            "events": compact(Journal(journals[n]).events)} for n, r in team.roles.items()],
                 "verdict": [vars(e) for e in result.verdict.evidence] if result.verdict else [],
                 "board": board_keep},
        "guide": {"question": trip.GUIDE_QUESTIONS[req.destination][0], "gold": trip.GUIDE_QUESTIONS[req.destination][1],
                  "answer": b.get("guide_answer"), "corpus": [{"title": t, "text": x} for t, x in trip.GUIDES[req.destination]]},
        "omniscope": {"events": [{k: e[k] for k in ("event_id", "agent_id", "event_type", "causal_parents", "timestamp",
                                                   "model_name", "tool_name", "output_preview", "confidence_score", "tags")}
                                 for e in omni], "risk": risk},
        "trace_viz": traces,
        "agentdiff": pairs,
        "ares": {"source": verify.ARES_SOURCE, "leaderboard": ares_leaderboard(args.ares_leaderboard)},
        "lean": {**lean_bench(), "cost": cost_story(team, result, again, team_s, again_s, first_hits, repeat_hits, journals)},
    }
    export.write_json(bundle, out / "stack.json")
    print(f"wrote {out / 'stack.json'} ({(out / 'stack.json').stat().st_size / 1024:.0f} KB)")
    print("tools:", bundle["environment"]["tools"])
    print("lean:", {k: v for k, v in bundle["lean"].items()})


if __name__ == "__main__":
    main()
