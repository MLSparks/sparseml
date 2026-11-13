"""Shared bits for the example workloads: stand-in models and their prices."""
from __future__ import annotations

import json

from sparkrun import ScriptedModel

PRICES = {"small": 0.0002, "strong": 0.003}  # stand-in cost per call, in dollars


def stand_in(name: str, decide, tier: str = "small") -> ScriptedModel:
    """A deterministic policy that reads the prompt like a model would and returns one action."""
    return ScriptedModel(lambda messages: json.dumps(decide(messages[1]["content"])), name=name,
                         usd_per_call=PRICES[tier])


def did(prompt: str, tool: str) -> bool:
    """True if the task memory shows this tool already succeeded."""
    return f"{tool}(" in prompt and "-> OK" in prompt.split(f"{tool}(", 1)[1].split("\n", 1)[0]
