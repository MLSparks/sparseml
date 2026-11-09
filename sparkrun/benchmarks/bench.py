"""Measure sparkrun's own overhead (model time excluded) and multi-run throughput.

    python benchmarks/bench.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkrun import Agent, Policy, ScriptedModel, tool


@tool
def lookup(i: int):
    """Look something up."""
    return f"value {i}"


def script(n: int):
    return [json.dumps({"tool": "lookup", "args": {"i": i}}) for i in range(n)] + [json.dumps({"done": "ok"})]


def one_run(runs: Path, steps: int, durability: str | None) -> float:
    kw = {} if durability is None else {"durability": durability}
    agent = Agent(ScriptedModel(script(steps)), [lookup], runs_dir=runs,
                  policy=Policy(max_steps=steps + 5, repeat_limit=10**6), **kw)
    t0 = time.perf_counter()
    r = agent.run("benchmark")
    assert r.status == "unverified", r.status
    return time.perf_counter() - t0


def parallel(runs: Path, tasks: int, workers: int, latency_s: float) -> float:
    """Each task: 3 model calls that each take latency_s, like a real small model."""
    def slow(messages):
        time.sleep(latency_s)
        n = sum(1 for m in messages[1]["content"].splitlines() if m.startswith("- lookup"))
        return json.dumps({"tool": "lookup", "args": {"i": n}}) if n < 2 else json.dumps({"done": "ok"})

    agent = Agent(ScriptedModel(slow), [lookup], runs_dir=runs)
    t0 = time.perf_counter()
    results = agent.run_many([f"task {i}" for i in range(tasks)], workers=workers)
    assert all(r.status == "unverified" and r.steps == 3 for r in results), [r.status for r in results]
    return time.perf_counter() - t0


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        if "--baseline" in sys.argv:
            t = one_run(d / "base", steps, None)
            print(f"default: {t / steps * 1e3:.3f} ms/step, {steps / t:,.0f} steps/s")
            return
        print(f"Overhead per step, {steps}-step run (model time excluded):")
        for mode in ("always", "side_effects", "never"):
            t = one_run(d / mode, steps, mode)
            print(f"  durability={mode:<13} {t / steps * 1e3:7.3f} ms/step   {steps / t:>9,.0f} steps/s")
        print("\nThroughput, 64 tasks x 3 model calls of 50 ms each:")
        for workers in (1, 16, 64):
            t = parallel(d / f"par{workers}", 64, workers, 0.05)
            print(f"  workers={workers:<3} {t:6.2f}s   {64 / t:6.1f} tasks/s")


if __name__ == "__main__":
    main()
