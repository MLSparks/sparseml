"""sparkrun demo.

Offline (no model needed):        python examples/demo.py
With a free local small model:    ollama pull qwen2.5:3b && python examples/demo.py --ollama qwen2.5:3b

Scene 1: a small model falsely claims it finished. The independent check catches it,
         the run escalates to a stronger model, and only a real, tested result counts.
Scene 2: the process crashes right after sending an invoice. On resume the model asks to
         send it again; sparkrun replays the recorded result instead of sending twice.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkrun import Agent, OpenAICompatible, Policy, ScriptedModel, format_report, summarize, tool, verify


def workspace_tools(work: Path):
    def inside(path: str) -> Path:
        p = (work / path).resolve()
        if work.resolve() not in p.parents and p != work.resolve():
            raise ValueError("path is outside the workspace")
        return p

    @tool(side_effect=True)
    def write_file(path: str, content: str):
        """Create or overwrite a file in the workspace."""
        inside(path).write_text(content)
        return f"wrote {len(content)} characters to {path}"

    @tool
    def read_file(path: str):
        """Read a file from the workspace."""
        return inside(path).read_text()

    @tool
    def list_files():
        """List files in the workspace."""
        return sorted(p.name for p in work.iterdir())

    return [write_file, read_file, list_files]


def scene_false_claim(runs: Path, work: Path, models) -> None:
    print("\nScene 1: catch a false 'done' claim")
    goal = "Create hello.py in the workspace. Running `python hello.py` must print: hello from sparkrun"
    checks = [verify.command([sys.executable, "hello.py"], cwd=work, expect_output="hello from sparkrun",
                             name="python hello.py prints the greeting")]
    r = Agent(models, workspace_tools(work), checks=checks, runs_dir=runs,
              policy=Policy(max_steps=12, backoff_s=0.2)).run(goal)
    for v in verdict_lines(r):
        print("  check:", v)
    print(f"  status={r.status}  steps={r.steps}  escalations={r.escalations}  cost=${r.cost_usd:.4f}")


def verdict_lines(result):
    from sparkrun import Journal
    out = []
    for e in Journal(result.journal).of("verified"):
        out += [f"{'PASS' if ev['passed'] else 'FAIL'} {ev['check']} ({ev['detail'][:60]})" for ev in e["evidence"]]
    return out


def scene_crash_resume(runs: Path) -> None:
    print("\nScene 2: crash after a side effect, then resume")
    sent: list[str] = []

    @tool(side_effect=True)
    def send_invoice(customer: str, amount: int):
        """Email an invoice to a customer."""
        sent.append(f"{customer}:{amount}")
        return f"invoice for ${amount} sent to {customer}"

    check = verify.predicate(lambda c: (sent == ["acme:120"], f"invoices sent: {sent}"), "exactly one invoice")
    invoice = json.dumps({"tool": "send_invoice", "args": {"customer": "acme", "amount": 120}})
    before = ScriptedModel([invoice, RuntimeError("simulated crash")], name="small-model", usd_per_call=0.0002)
    try:
        Agent(before, [send_invoice], checks=[check], runs_dir=runs).run("Invoice acme $120", run_id="invoice-acme")
    except RuntimeError as e:
        print(f"  process died: {e}  (invoices sent so far: {len(sent)})")
    after = ScriptedModel([invoice, json.dumps({"done": "Invoice sent"})], name="small-model", usd_per_call=0.0002)
    r = Agent(after, [send_invoice], checks=[check], runs_dir=runs).run(run_id="invoice-acme")
    print(f"  resumed: status={r.status}  invoices sent={len(sent)}  duplicates prevented={r.duplicates_prevented}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", metavar="MODEL", help="use a local Ollama model, e.g. qwen2.5:3b")
    ap.add_argument("--keep", action="store_true", help="keep the runs folder and print its path")
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="sparkrun-demo-"))
    runs, work = root / "runs", root / "work"
    work.mkdir()

    if args.ollama:
        models = [OpenAICompatible("http://localhost:11434/v1", args.ollama)]
    else:
        small = ScriptedModel(['Sure, done! {"done": "Created hello.py"}'], name="small-model", usd_per_call=0.0002)
        big = ScriptedModel([
            json.dumps({"tool": "write_file", "args": {"path": "hello.py",
                                                       "content": "print('hello from sparkrun')\n"}}),
            json.dumps({"done": "Wrote hello.py, which prints the greeting"}),
        ], name="stronger-model", usd_per_call=0.003)
        models = [small, big]

    scene_false_claim(runs, work, models)
    scene_crash_resume(runs)
    print("\nReport\n" + format_report(summarize(runs)))
    if args.keep:
        print(f"\nRuns kept in {runs}. Inspect one with: python -m sparkrun show {runs}/<run>")
    else:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
