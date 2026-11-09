"""A minimal multi-agent team on top of sparkrun.

Each Role is a specialist sparkrun Agent with its own tools, checks and models. Roles
share a blackboard (a plain dict their tools read and write), run in waves set by their
dependencies (roles in the same wave run in parallel), and inherit every sparkrun
guarantee: verified results, crash-safe resume, side effects at most once, approvals.
After the last role, team-level checks verify the combined outcome.

Run the same team_id again to resume: finished roles return instantly from their
journals, the blackboard is restored, and only unfinished work continues.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .runner import Agent, Result
from .verify import Check, Verdict, run_checks

OK = {"verified", "unverified"}


@dataclass
class Role:
    name: str
    agent: Agent
    goal: str | Callable[[dict], str]
    """A fixed goal, or a function of the blackboard (to pass earlier roles' results along)."""
    after: Sequence[str] = ()


@dataclass
class TeamResult:
    team_id: str
    status: str
    """verified | unverified | failed | needs_approval | needs_reconcile | model_unavailable | ...
    (the first role that did not finish cleanly decides it)"""
    roles: dict[str, Result] = field(default_factory=dict)
    board: dict = field(default_factory=dict)
    verdict: Verdict | None = None
    waiting_on: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "verified"

    @property
    def cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.roles.values()), 6)

    @property
    def steps(self) -> int:
        return sum(r.steps for r in self.roles.values())


class Team:
    def __init__(self, roles: Iterable[Role], *, board: dict | None = None, checks: Iterable[Check] = (),
                 runs_dir: str | Path = "runs", workers: int = 8,
                 on_event: Callable[[str, dict], Any] | None = None):
        self.roles = {r.name: r for r in roles}
        self.board = board if board is not None else {}
        self.checks = list(checks)
        self.runs_dir = Path(runs_dir)
        self.workers = workers
        self.on_event = on_event
        self._lock = threading.Lock()
        self._board_file: Path | None = None
        for r in self.roles.values():
            unknown = set(r.after) - set(self.roles)
            if unknown:
                raise ValueError(f"role {r.name!r} depends on unknown roles {sorted(unknown)}")
            r.agent.on_event = (lambda name: lambda e: self._event(name, e))(r.name)
            r.agent.runs_dir = self.runs_dir  # journals live with the team, for run() and approve()
        self.waves = self._waves()

    def _event(self, role: str, event: dict) -> None:
        # Tools write to the blackboard before their result is journaled, so saving here keeps
        # the board in step with the journals: a resumed role never skips a write it relied on.
        if event["kind"] in ("tool_result", "finished") and self._board_file is not None:
            self._save(self._board_file)
        if self.on_event is not None:
            self.on_event(role, event)

    def _waves(self) -> list[list[str]]:
        done: set[str] = set()
        waves = []
        while len(done) < len(self.roles):
            ready = [n for n, r in self.roles.items() if n not in done and set(r.after) <= done]
            if not ready:
                raise ValueError("roles have a dependency cycle")
            waves.append(ready)
            done |= set(ready)
        return waves

    def run(self, team_id: str, *, stop_before: Iterable[str] = ()) -> TeamResult:
        """Run (or resume) the team. stop_before pauses before those roles, for example to
        hand a heavy step to another runtime, then call run() again to continue."""
        stop = set(stop_before)
        board_file = self._board_file = self.runs_dir / f"{team_id}.board.json"
        if board_file.exists():
            self.board.update(json.loads(board_file.read_text()))
        result = TeamResult(team_id, "in_progress", board=self.board)

        def one(name: str) -> tuple[str, Result]:
            role = self.roles[name]
            goal = role.goal(self.board) if callable(role.goal) else role.goal
            return name, role.agent.run(goal, run_id=f"{team_id}.{name}")

        for wave in self.waves:
            todo = [n for n in wave if n not in stop]
            with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(todo) or 1))) as pool:
                for name, res in pool.map(one, todo):
                    result.roles[name] = res
            self._save(board_file)
            bad = [n for n in wave if n in result.roles and result.roles[n].status not in OK]
            if bad:
                result.status, result.waiting_on = result.roles[bad[0]].status, bad[0]
                return result
            if len(todo) < len(wave):
                result.status, result.waiting_on = "paused", next(n for n in wave if n in stop)
                return result

        if not self.checks:
            result.status = "unverified"
            return result
        result.verdict = run_checks(self.checks, {"board": self.board, "roles": result.roles})
        result.status = "verified" if result.verdict.passed else "failed"
        return result

    def approve(self, team_id: str, role: str, key: str | None = None) -> dict:
        """Approve what a paused role is waiting on; then call run(team_id) to continue."""
        return self.roles[role].agent.approve(f"{team_id}.{role}", key)

    def _save(self, path: Path) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(self.board, default=str))
            os.replace(tmp, path)
