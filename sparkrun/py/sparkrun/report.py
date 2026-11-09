"""Roll up runs into the numbers that matter: verified success rate and cost per verified task."""
from __future__ import annotations

from pathlib import Path

from .journal import Journal
from .runner import _State


def summarize(runs_dir: str | Path = "runs") -> dict:
    rows = []
    for path in sorted(Path(runs_dir).glob("*/journal.jsonl")):
        s = _State.from_journal(Journal(path))
        if s.finished:
            status = s.finished["status"]
        elif s.pending:
            status = s.pending["reason"]
        else:
            status = "in_progress"
        rows.append({"run": path.parent.name, "status": status, "steps": s.steps,
                     "cost_usd": round(s.cost, 6), "escalations": s.escalations,
                     "duplicates_prevented": s.replayed, "resumes": s.resumes, "goal": s.task.goal})
    verified = [r for r in rows if r["status"] == "verified"]
    total_cost = sum(r["cost_usd"] for r in rows)
    return {
        "runs": rows,
        "total": len(rows),
        "verified": len(verified),
        "success_rate": len(verified) / len(rows) if rows else 0.0,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_verified_usd": round(total_cost / len(verified), 6) if verified else None,
        "duplicates_prevented": sum(r["duplicates_prevented"] for r in rows),
        "escalations": sum(r["escalations"] for r in rows),
        "resumes": sum(r["resumes"] for r in rows),
    }


def format_report(summary: dict) -> str:
    cpv = summary["cost_per_verified_usd"]
    lines = [
        f"Runs: {summary['total']}   verified: {summary['verified']}   "
        f"success rate: {summary['success_rate']:.0%}",
        f"Total cost: ${summary['total_cost_usd']:.4f}   cost per verified task: "
        + (f"${cpv:.4f}" if cpv is not None else "n/a"),
        f"Duplicate side effects prevented: {summary['duplicates_prevented']}   "
        f"escalations: {summary['escalations']}   resumes: {summary['resumes']}",
        "",
        f"{'run':<28} {'status':<17} {'steps':>5} {'cost':>9}  goal",
    ]
    for r in summary["runs"]:
        lines.append(f"{r['run'][:28]:<28} {r['status']:<17} {r['steps']:>5} ${r['cost_usd']:>8.4f}  {r['goal'][:60]}")
    return "\n".join(lines)
