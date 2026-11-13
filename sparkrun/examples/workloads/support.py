"""Support: refund a damaged item, using your own agent loop (sparkrun Harness).

The loop below stands in for any framework (LangGraph, CrewAI, the OpenAI Agents SDK, your own
code). sparkrun doesn't drive it; the Harness only wraps its tools and models, so refunds happen
at most once, large refunds wait for a person, and nothing reaches the customer until checks pass.
"""
from __future__ import annotations

from pathlib import Path

from sparkrun import Harness, ScriptedModel, verify

from .common import PRICES

TITLE = "Refund a damaged item"
DOMAIN = "Customer support"
STYLE = "sparkrun Harness: your own agent loop, wrapped; no rewrite"
SUMMARY = "A support bot handles a refund request. The refund happens once, and the reply goes out only after checks pass."

ORDER = {"id": "A-1042", "customer": "customer@example.com",
         "items": [{"sku": "mug", "price": 45, "damaged": True}, {"sku": "lamp", "price": 135, "damaged": False}]}


def run(runs_dir: Path, wrap=None) -> list[Path]:
    ledger, outbox = [], []

    def lookup_order(order_id: str):
        return ORDER

    def issue_refund(order_id: str, amount: int):
        ledger.append((order_id, amount))
        return {"order": order_id, "amount": amount, "ref": f"RF-{order_id}"}

    def send_reply(to: str, text: str):
        outbox.append((to, text))
        return "sent"

    owed = sum(i["price"] for i in ORDER["items"] if i["damaged"])
    checks = [verify.predicate(lambda c: (ledger == [(ORDER["id"], owed)], f"refunds issued: {ledger or 'none'}"),
                               f"exactly one refund of ${owed} for the damaged item"),
              verify.predicate(lambda c: (f"${owed}" in c["answer"], "reply states the refunded amount"),
                               "reply matches what was refunded")]
    policy = lambda tool, args: "refunds over $100 need a support lead" if tool == "issue_refund" and args["amount"] > 100 else None

    def models():
        small = ScriptedModel(["Sorry about that! I've refunded your order, it's on its way."], name="support-small",
                              usd_per_call=PRICES["small"])
        strong = ScriptedModel([f"Refund ${owed} for the damaged mug only.",
                                f"We've refunded ${owed} for the damaged mug (ref RF-{ORDER['id']}). Sorry about that!"],
                               name="support-strong", usd_per_call=PRICES["strong"])
        return [wrap(m) for m in (small, strong)] if wrap else [small, strong]

    with Harness("support", "Customer: my mug arrived broken (order A-1042).", checks=checks,
                 runs_dir=runs_dir, gate=policy) as h:
        # ---- your existing agent loop, unchanged except for the h.tool / h.model wrappers ----
        lookup, refund, reply = h.tool(lookup_order), h.tool(issue_refund, side_effect=True), h.tool(send_reply, side_effect=True)
        small, strong = (h.model(m, level=i) for i, m in enumerate(models()))
        draft, _ = small.complete([{"role": "user", "content": "Customer says the mug is broken. Draft a reply."}])
        if not h.check(draft).passed:                    # a claim, not a refund: caught
            h.escalate("support-strong", "checks failed: nothing was refunded")
            order = lookup(ORDER["id"])
            strong.complete([{"role": "user", "content": f"Order: {order}. What should we refund?"}])
            refund(ORDER["id"], owed)
            draft, _ = strong.complete([{"role": "user", "content": "Draft the reply."}])
        result = h.done(draft)
        if result.ok:                                    # only verified replies reach the customer
            reply(ORDER["customer"], draft)
    return [runs_dir / "support" / "journal.jsonl"]
