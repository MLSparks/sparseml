"""Learn which agent setup works best, measured by verified success per dollar.

Keep a few variants (a prompt wording, a model list, a context budget), let each run
pick one, and record the verified outcome. Better variants get picked more often;
nothing is kept that did not pass independent checks.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence


class Variants:
    def __init__(self, names: Sequence[str], *, path: str | Path = "runs/variants.json",
                 epsilon: float = 0.2, cost_weight: float = 1.0, seed: int | None = None):
        if not names:
            raise ValueError("at least one variant is required")
        self.names = list(names)
        self.path = Path(path)
        self.epsilon = epsilon
        self.cost_weight = cost_weight
        self.rng = random.Random(seed)
        self.stats = {n: {"runs": 0, "verified": 0, "cost_usd": 0.0} for n in self.names}
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            for n, st in saved.items():
                if n in self.stats:
                    self.stats[n].update(st)

    def score(self, name: str) -> float:
        st = self.stats[name]
        if st["runs"] == 0:
            return float("inf")  # try everything at least once
        return st["verified"] / st["runs"] - self.cost_weight * st["cost_usd"] / st["runs"]

    def choose(self) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.names)
        return max(self.names, key=self.score)

    def record(self, name: str, result) -> None:
        st = self.stats[name]
        st["runs"] += 1
        st["verified"] += int(result.status == "verified")
        st["cost_usd"] += float(result.cost_usd)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.stats, indent=2))

    def table(self) -> list[dict]:
        rows = []
        for n in self.names:
            st = self.stats[n]
            runs = st["runs"] or 1
            rows.append({"variant": n, "runs": st["runs"], "verified_rate": st["verified"] / runs,
                         "avg_cost_usd": st["cost_usd"] / runs})
        return sorted(rows, key=lambda r: (-r["verified_rate"], r["avg_cost_usd"]))
