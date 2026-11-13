"""Research: answer from internal documents, graded by ARES groundedness (sparkrun Agent)."""
from __future__ import annotations

import re
from pathlib import Path

from sparkrun import Agent, Policy, tool, verify

from .common import did, stand_in

TITLE = "Answer from internal docs"
DOMAIN = "Knowledge work"
STYLE = "sparkrun Agent: retrieval, graded by ARES groundedness and F1"
SUMMARY = "An analyst agent answers a policy question. The answer must be supported by the documents it retrieved."

DOCS = [
    ("Refund window", "Customers can request a refund within 30 days of delivery."),
    ("Refund approvals", "Refunds over $100 are approved by a support lead before they are issued."),
    ("Shipping", "Standard shipping takes 3 to 5 business days within the country."),
    ("Exchanges", "Exchanges are free for the same item in a different size or color."),
    ("Warranty", "Electronics carry a one-year limited warranty from the date of purchase."),
]
QUESTION = "How long do customers have to request a refund, and who approves refunds over $100?"
GOLD = "30 days, a support lead"


def run(runs_dir: Path, wrap=None) -> list[Path]:
    evidence: list[str] = []

    @tool
    def search_docs(query: str, k: int = 2):
        """Search the internal documentation."""
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        hits = sorted(DOCS, key=lambda d: -len(words & set(re.findall(r"[a-z0-9]+", " ".join(d).lower()))))[:int(k)]
        evidence.extend(text for _, text in hits)
        return [{"title": t, "text": x} for t, x in hits]

    def from_memory(prompt):
        return {"done": "Customers have 60 days, and the CEO approves large refunds."}

    def with_docs(prompt):
        if not did(prompt, "search_docs"):
            return {"tool": "search_docs", "args": {"query": "refund request days approved over $100", "k": 2}}
        return {"done": "Within 30 days of delivery; refunds over $100 are approved by a support lead."}

    models = [stand_in("analyst-small", from_memory), stand_in("analyst-strong", with_docs, "strong")]
    models = [wrap(m) for m in models] if wrap else models
    checks = [verify.grounded(lambda c: " ".join(evidence), min_score=0.8), verify.answer_f1(GOLD, min_score=0.4)]
    Agent(models, [search_docs], checks=checks, runs_dir=runs_dir, policy=Policy(max_steps=8, backoff_s=0)).run(
        QUESTION, run_id="research")
    return [runs_dir / "research" / "journal.jsonl"]
