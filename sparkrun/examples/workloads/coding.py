"""Coding: fix a failing test in a real workspace (sparkrun Agent, verified by running the tests)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from sparkrun import Agent, Policy, tool, verify

from .common import did, stand_in

TITLE = "Fix a failing test"
DOMAIN = "Software"
STYLE = "sparkrun Agent: file tools, verified by actually running the test suite"
SUMMARY = "A coding agent fixes an off-by-one bug. Done means the tests pass, not that the model says so."

BUGGY = "def average(xs):\n    return sum(xs) / (len(xs) - 1)\n"
FIXED = "def average(xs):\n    if not xs:\n        raise ValueError('average of an empty list')\n    return sum(xs) / len(xs)\n"
TESTS = """import unittest
from calc import average

class TestAverage(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(average([2, 4, 6]), 4)

    def test_single(self):
        self.assertEqual(average([5]), 5)

if __name__ == "__main__":
    unittest.main()
"""


def run(runs_dir: Path, wrap=None) -> list[Path]:
    ws = Path(tempfile.mkdtemp(prefix="sparkrun-coding-"))
    (ws / "calc.py").write_text(BUGGY)
    (ws / "test_calc.py").write_text(TESTS)

    @tool
    def read_file(path: str):
        """Read a file in the repository."""
        return (ws / path).read_text()

    @tool(side_effect=True)
    def write_file(path: str, content: str):
        """Overwrite a file in the repository."""
        (ws / path).write_text(content)
        return f"wrote {len(content)} characters to {path}"

    def lazy(prompt):  # a cheap model that claims the fix without making it
        return {"done": "Fixed the off-by-one in average()"}

    def careful(prompt):
        if not did(prompt, "read_file"):
            return {"tool": "read_file", "args": {"path": "calc.py"}}
        if not did(prompt, "write_file"):
            return {"tool": "write_file", "args": {"path": "calc.py", "content": FIXED}}
        return {"done": "average() divided by len - 1; now divides by len and rejects empty lists"}

    models = [stand_in("coder-small", lazy), stand_in("coder-strong", careful, "strong")]
    models = [wrap(m) for m in models] if wrap else models
    tests = verify.command([sys.executable, "-m", "unittest", "-q"], cwd=ws, name="test suite passes")
    Agent(models, [read_file, write_file], checks=[tests], runs_dir=runs_dir,
          policy=Policy(max_steps=10, backoff_s=0)).run("Make the failing tests in this repository pass", run_id="coding")
    return [runs_dir / "coding" / "journal.jsonl"]
