"""Independent outcome checks.

A run is only "verified" when checks that do not trust the model pass: a test suite
exits 0, a file holds the expected content, an endpoint answers, a record exists.
Each check returns Evidence, which is stored in the journal.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union


@dataclass
class Evidence:
    check: str
    passed: bool
    detail: str = ""


@dataclass
class Verdict:
    passed: bool
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def failures(self) -> list[Evidence]:
        return [e for e in self.evidence if not e.passed]


Check = Callable[[dict], Union[Evidence, bool]]


def run_checks(checks: Iterable[Check], ctx: dict) -> Verdict:
    evidence: list[Evidence] = []
    for check in checks:
        label = getattr(check, "check_name", getattr(check, "__name__", "check"))
        try:
            ev = check(ctx)
            if not isinstance(ev, Evidence):
                ev = Evidence(label, bool(ev))
        except Exception as e:  # a broken check must fail closed, not crash the run
            ev = Evidence(label, False, f"check crashed: {type(e).__name__}: {e}")
        evidence.append(ev)
    return Verdict(bool(evidence) and all(e.passed for e in evidence), evidence)


def _named(fn: Callable, label: str) -> Callable:
    fn.check_name = label  # type: ignore[attr-defined]
    return fn


def command(cmd: str | Sequence[str], *, cwd: str | Path | None = None, timeout: float = 300,
            expect_output: str | None = None, name: str | None = None) -> Check:
    """Passes when the command exits 0 (and prints expect_output, if given)."""
    argv = shlex.split(cmd) if isinstance(cmd, str) else [str(a) for a in cmd]
    label = name or f"command: {' '.join(argv)}"

    def check(ctx: dict) -> Evidence:
        try:
            p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            return Evidence(label, False, str(e))
        except subprocess.TimeoutExpired:
            return Evidence(label, False, f"timed out after {timeout}s")
        out = (p.stdout + p.stderr).strip()
        if p.returncode != 0:
            return Evidence(label, False, f"exit code {p.returncode}: {out[-800:]}")
        if expect_output is not None and expect_output not in out:
            return Evidence(label, False, f"output did not contain {expect_output!r}: {out[-800:]}")
        return Evidence(label, True, f"exit code 0: {out[-200:]}")

    return _named(check, label)


def file_contains(path: str | Path, text: str | None = None, *, regex: str | None = None,
                  name: str | None = None) -> Check:
    """Passes when the file exists and contains the text and/or matches the regex."""
    label = name or f"file: {path}"

    def check(ctx: dict) -> Evidence:
        p = Path(path)
        if not p.is_file():
            return Evidence(label, False, "file does not exist")
        body = p.read_text(errors="replace")
        if text is not None and text not in body:
            return Evidence(label, False, f"missing text {text!r}")
        if regex is not None and not re.search(regex, body):
            return Evidence(label, False, f"no match for /{regex}/")
        return Evidence(label, True, f"{len(body)} characters")

    return _named(check, label)


def http_ok(url: str, *, status: int = 200, contains: str | None = None, timeout: float = 20,
            name: str | None = None) -> Check:
    """Passes when the URL answers with the expected status (and body text, if given)."""
    label = name or f"http: {url}"

    def check(ctx: dict) -> Evidence:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                code, body = r.status, r.read(200_000).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            code, body = e.code, ""
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return Evidence(label, False, f"unreachable: {e}")
        if code != status:
            return Evidence(label, False, f"status {code}, expected {status}")
        if contains is not None and contains not in body:
            return Evidence(label, False, f"body did not contain {contains!r}")
        return Evidence(label, True, f"status {code}")

    return _named(check, label)


def predicate(fn: Callable[[dict], Any], name: str) -> Check:
    """Wrap your own check. fn(ctx) returns a bool or (bool, detail)."""

    def check(ctx: dict) -> Evidence:
        out = fn(ctx)
        ok, detail = out if isinstance(out, tuple) else (bool(out), "")
        return Evidence(name, bool(ok), str(detail))

    return _named(check, name)
