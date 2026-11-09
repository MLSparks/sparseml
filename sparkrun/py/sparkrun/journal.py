"""Append-only, crash-safe event journal.

Every state change in a run is written here first, so a run can be rebuilt
exactly after a crash, audited afterwards, and never repeats a side effect.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator


def stable_hash(obj: Any) -> str:
    """Short, deterministic hash of any JSON-able value (used as an idempotency key)."""
    data = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


class Journal:
    """A JSON-lines file of events. Each append is flushed and fsynced before returning."""

    DURABILITY = ("side_effects", "always", "never")

    def __init__(self, path: str | os.PathLike, durability: str = "side_effects"):
        """durability decides when an event is forced to disk with fsync:

        side_effects  (default) only events marked critical: around side effects, approvals,
                      pauses and the end of a run. Every event is still written and flushed at
                      once, so a process crash loses nothing; a power cut can lose at most the
                      latest non-critical events, which only means re-asking the model.
        always        every event. Safest, slowest.
        never         no fsync. Fastest; survives process crashes, not power loss.
        """
        if durability not in self.DURABILITY:
            raise ValueError(f"durability must be one of {self.DURABILITY}")
        self.path = Path(path)
        self.durability = durability
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self._f = None
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        torn = False
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.events.append(json.loads(line))
                except json.JSONDecodeError:
                    torn = True  # a crash cut the last write short
                    break
        if torn:  # drop the torn tail so later appends stay readable
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for event in self.events:
                    f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)

    def append(self, kind: str, *, critical: bool = False, **data: Any) -> dict:
        """Write one event. critical=True forces it to disk under the default durability."""
        event = {"seq": len(self.events), "t": round(time.time(), 3), "kind": kind, **data}
        line = json.dumps(event, default=str, ensure_ascii=False)
        if self._f is None:
            self._f = self.path.open("a", encoding="utf-8")
        self._f.write(line + "\n")
        self._f.flush()  # in the OS from here on: a process crash can't lose it
        if self.durability == "always" or (critical and self.durability == "side_effects"):
            os.fsync(self._f.fileno())
        self.events.append(event)
        return event

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]

    def __iter__(self) -> Iterator[dict]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)
