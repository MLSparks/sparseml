"""Travel: a six-agent team plans and books a trip (sparkrun Team)."""
from __future__ import annotations

from pathlib import Path

import trip

TITLE = "Plan and book a trip"
DOMAIN = "Travel"
STYLE = "sparkrun Team: six agents in parallel waves, GPU ranking, approvals"
SUMMARY = "Flights, hotels and a guide search in parallel; a ranker scores every option; a person approves each booking."


def run(runs_dir: Path, wrap=None) -> list[Path]:
    team = trip.build_team(trip.TripRequest(options=600), runs_dir, gate=trip.spending_gate, guide=True, wrap=wrap)
    result, n = team.run("trip"), 0
    while result.status == "needs_approval" and n < 4:  # a person approves each booking
        team.approve("trip", "booking")
        n += 1
        result = team.run("trip")
    return [runs_dir / f"trip.{name}" / "journal.jsonl" for name in team.roles]
