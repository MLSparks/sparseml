"""Command line: python -m sparkrun {report,show,approve}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .journal import Journal
from .report import format_report, summarize
from .runner import approve_pending


def _show(run_dir: Path) -> None:
    for e in Journal(run_dir / "journal.jsonl"):
        k = e["kind"]
        if k == "model_call":
            detail = f"{e['model']}: {e['reply'][:120]!r}"
        elif k in ("tool_started", "tool_replayed", "approval_requested"):
            detail = f"{e['tool']}({json.dumps(e['args'])[:100]})"
        elif k == "tool_result":
            detail = f"{e['tool']} -> {'ok' if e['ok'] else 'ERROR'}: {str(e['result'])[:100]}"
        elif k == "verified":
            detail = ("PASSED" if e["passed"] else "FAILED") + "; " + "; ".join(
                f"{ev['check']}: {'ok' if ev['passed'] else ev['detail'][:80]}" for ev in e["evidence"])
        else:
            detail = ", ".join(f"{a}={str(b)[:80]}" for a, b in e.items() if a not in ("seq", "t", "kind"))
        print(f"{e['seq']:>4}  {k:<18} {detail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sparkrun", description="Inspect and control sparkrun agent runs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="success rate and cost per verified task")
    r.add_argument("runs_dir", nargs="?", default="runs")
    r.add_argument("--json", action="store_true", help="print JSON instead of a table")
    sh = sub.add_parser("show", help="print one run's journal")
    sh.add_argument("run_dir")
    ap_ = sub.add_parser("approve", help="approve the action a paused run is waiting on")
    ap_.add_argument("run_dir")
    args = ap.parse_args(argv)

    if args.cmd == "report":
        summary = summarize(args.runs_dir)
        print(json.dumps(summary, indent=2) if args.json else format_report(summary))
    elif args.cmd == "show":
        _show(Path(args.run_dir))
    else:
        try:
            approved = approve_pending(Path(args.run_dir) / "journal.jsonl")
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        print(f"Approved {approved.get('tool', '')}{json.dumps(approved.get('args', {}))}. "
              f"Run the agent again with run_id={Path(args.run_dir).name!r} to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
