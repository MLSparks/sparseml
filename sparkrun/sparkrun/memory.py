"""Task memory that survives compaction.

Instead of resending a growing transcript, every model call gets a compact view of
the task: the goal, what must still be fixed, facts worth keeping, and the newest
steps that fit the budget. Small models with short context windows stay on track,
and the prompt size stays flat however long the run goes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_ITEM_CHARS = 600


@dataclass
class TaskState:
    goal: str = ""
    facts: list[str] = field(default_factory=list)
    open_items: list[str] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)

    def render(self, budget_chars: int = 6000) -> str:
        goal = f"GOAL:\n{self.goal}"
        must_fix = ("MUST FIX BEFORE DONE:\n" + "\n".join(f"- {_clip(x)}" for x in self.open_items)
                    if self.open_items else "")
        left = max(0, budget_chars - len(goal) - len(must_fix) - 120)
        steps, dropped = _newest_that_fit(self.recent, left * 2 // 3)
        facts, _ = _newest_that_fit(self.facts, left - sum(len(x) + 3 for x in steps))

        parts = [goal]
        if facts:
            parts.append("REMEMBERED:\n" + "\n".join(f"- {x}" for x in facts))
        if must_fix:
            parts.append(must_fix)
        if steps:
            note = f" ({dropped} older steps omitted)" if dropped else ""
            parts.append(f"STEPS SO FAR{note}:\n" + "\n".join(f"- {x}" for x in steps))
        else:
            parts.append("STEPS SO FAR: none yet.")
        parts.append("What is the next action? Reply with one JSON object.")
        return "\n\n".join(parts)


def _clip(text: str, limit: int = MAX_ITEM_CHARS) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _newest_that_fit(items: list[str], budget: int) -> tuple[list[str], int]:
    out: list[str] = []
    used = 0
    for item in reversed(items):
        clipped = _clip(item)
        if used + len(clipped) + 3 > budget:
            break
        out.append(clipped)
        used += len(clipped) + 3
    out.reverse()
    return out, len(items) - len(out)
