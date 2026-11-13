"""Harness: sparkrun's guarantees for an agent loop you already have."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkrun import Harness, PendingApproval, ScriptedModel, export, summarize, verify


class Crash(BaseException):
    pass


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name)
        self.refunds = []

    def tearDown(self):
        self.tmp.cleanup()

    def refund(self, order: str, amount: int):
        self.refunds.append((order, amount))
        return {"order": order, "amount": amount, "ref": f"R-{order}"}

    def one_refund(self):
        return verify.predicate(lambda c: (len(self.refunds) == 1, f"{len(self.refunds)} refunds"), "exactly one refund")

    def h(self, run_id="r1", goal="Refund order 7", **kw):
        kw.setdefault("checks", [self.one_refund()])
        return Harness(run_id, goal, runs_dir=self.runs, **kw)


class TestHarness(Base):
    def test_plain_loop_verified_with_cost(self):
        with self.h() as h:
            refund = h.tool(self.refund, side_effect=True)
            llm = h.model(ScriptedModel(["refund it"], name="small", usd_per_call=0.001))
            llm.complete([{"role": "user", "content": "what now?"}])
            out = refund("7", 40)
            r = h.done("Refunded $40")
        self.assertEqual(out["ref"], "R-7")
        self.assertEqual((r.status, r.steps, r.cost_usd), ("verified", 1, 0.001))

    def test_side_effect_once_across_restarts(self):
        with self.h() as h:
            h.tool(self.refund, side_effect=True)("7", 40)
        with self.h() as h:  # a new process re-runs the same code
            out = h.tool(self.refund, side_effect=True)(order="7", amount=40)
            r = h.done("Refunded")
        self.assertEqual(self.refunds, [("7", 40)])
        self.assertEqual(out, {"order": "7", "amount": 40, "ref": "R-7"})  # original value, not a string
        self.assertEqual((r.status, r.duplicates_prevented), ("verified", 1))

    def test_approval_pauses_then_runs_once(self):
        with self.h() as h:
            h.tool(self.refund, side_effect=True, needs_approval=True)("7", 40)
            self.fail("should have paused")
        self.assertEqual((h.status, self.refunds), ("needs_approval", []))
        h.approve()
        with self.h() as h:
            h.tool(self.refund, side_effect=True, needs_approval=True)("7", 40)
            r = h.done("Refunded")
        self.assertEqual((r.status, self.refunds), ("verified", [("7", 40)]))

    def test_policy_gate(self):
        gate = lambda tool, args: "refunds over $100 need a person" if args.get("amount", 0) > 100 else None
        with self.h(gate=gate) as h:
            refund = h.tool(self.refund, side_effect=True)
            refund("7", 40)
            with self.assertRaises(PendingApproval) as cm:
                refund("8", 500)
        self.assertIn("over $100", cm.exception.note)
        self.assertEqual(self.refunds, [("7", 40)])

    def test_crash_during_side_effect_needs_reconcile(self):
        def flaky(order: str):
            self.refunds.append(order)
            raise Crash()

        with self.assertRaises(Crash):
            with self.h() as h:
                h.tool(flaky, side_effect=True, name="refund")("7")
        with self.h() as h:
            h.tool(flaky, side_effect=True, name="refund")("7")
        self.assertEqual(h.status, "needs_reconcile")
        self.assertEqual(self.refunds, ["7"])

    def test_false_claim_fails_checks(self):
        with self.h() as h:
            r = h.done("All refunded!")
            again = h.done("All refunded!")
        self.assertEqual((r.status, again.status), ("failed", "failed"))
        self.assertIn("0 refunds", r.verdict.failures[0].detail)

    def test_check_then_retry_on_a_stronger_model(self):
        with self.h() as h:
            v = h.check("Refunded!")             # cheap model's claim: nothing was refunded
            self.assertFalse(v.passed)
            h.escalate("strong", "verification failed")
            h.tool(self.refund, side_effect=True)("7", 40)
            r = h.done("Refunded $40")
        self.assertEqual((r.status, r.escalations), ("verified", 1))
        self.assertEqual([e["passed"] for e in h.journal.events if e["kind"] == "verified"], [False, True])

    def test_tool_errors_are_recorded_and_raised(self):
        def broken(x: int):
            raise ValueError("nope")

        with self.h(checks=[]) as h:
            with self.assertRaises(ValueError):
                h.tool(broken)(1)
            r = h.done("gave up")
        self.assertEqual(r.status, "unverified")
        self.assertTrue(any(e["kind"] == "tool_result" and not e["ok"] for e in h.journal.events))

    def test_exports_and_report_work_on_harness_runs(self):
        with self.h() as h:
            h.model_call("small", reply="claimed done", tokens_in=10, tokens_out=2, cost_usd=0.0002)
            h.done("done")  # fails: no refund
        with self.h(run_id="r2") as h:
            h.escalate("strong", "verification failed")
            h.model_call("strong", reply="refund", tokens_in=12, tokens_out=3, cost_usd=0.003, level=1)
            h.tool(self.refund, side_effect=True)("7", 40)
            h.done("Refunded")
        t = export.to_agentdiff(self.runs / "r2" / "journal.jsonl")
        self.assertEqual(t["steps"][-1]["type"], "answer")
        self.assertTrue(t["outcome"]["success"])
        s = summarize(self.runs)
        self.assertEqual((s["total"], s["verified"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
