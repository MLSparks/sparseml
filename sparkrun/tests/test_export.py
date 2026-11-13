"""Exporters and ARES checks. When the real tools are importable (the stack CI job installs
them), outputs are also validated by each tool's own loader."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import trip  # examples/trip.py
from sparkrun import export, verify

STEP_TYPES = {"plan", "search", "retrieve", "read", "tool_call", "reason", "answer"}


def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.runs = Path(cls.tmp.name)
        cls.team = trip.build_team(trip.TripRequest(options=80), cls.runs, gate=trip.spending_gate)
        r = cls.team.run("trip")
        for _ in range(2):
            cls.team.approve("trip", "booking")
            r = cls.team.run("trip")
        assert r.status == "verified", r.status
        cls.activities = cls.runs / "trip.activities" / "journal.jsonl"
        cls.booking = cls.runs / "trip.booking" / "journal.jsonl"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestAgentDiff(Base):
    def check_shape(self, t):
        self.assertEqual(t["schema_version"], 1)
        self.assertEqual(t["steps"][-1]["type"], "answer")
        self.assertEqual([s["index"] for s in t["steps"]], list(range(len(t["steps"]))))
        self.assertTrue(all(s["type"] in STEP_TYPES for s in t["steps"]))
        for k in ("input_tokens", "output_tokens", "cost_usd", "latency_s"):
            self.assertIsInstance(t["totals"][k], (int, float))

    def test_whole_run(self):
        t = export.to_agentdiff(self.booking)
        self.check_shape(t)
        self.assertTrue(t["outcome"]["success"])
        names = [s["name"] for s in t["steps"]]
        self.assertIn("awaiting_approval", names)
        self.assertEqual(names.count("book_flight"), 1)

    def test_escalation_pair(self):
        self.assertEqual(export.levels(self.activities), [0, 1])
        a, b = export.escalation_pair(self.activities, task_id="plan")
        self.check_shape(a), self.check_shape(b)
        self.assertEqual((a["outcome"]["success"], b["outcome"]["success"]), (False, True))
        self.assertEqual(a["task"]["id"], b["task"]["id"])
        self.assertNotIn("plan_activities", [s["name"] for s in a["steps"]])
        self.assertIn("plan_activities", [s["name"] for s in b["steps"]])

    def test_no_escalation_no_pair(self):
        with self.assertRaises(ValueError):
            export.escalation_pair(self.booking)

    @unittest.skipUnless(has("agentdiff"), "agentdiff not installed")
    def test_agentdiff_accepts_it(self):
        from agentdiff.trace import Trajectory
        for t in (*export.escalation_pair(self.activities), export.to_agentdiff(self.booking)):
            Trajectory.from_dict(t)


class TestOmniscope(Base):
    def test_causal_edges_follow_team_dependencies(self):
        events = export.team_omniscope(self.team, "trip")
        by_agent = {}
        for e in events:
            by_agent.setdefault(e["agent_id"], []).append(e)
        self.assertEqual(set(by_agent), {"flights", "hotels", "rank", "activities", "booking"})
        first_rank = by_agent["rank"][0]
        self.assertEqual(set(first_rank["causal_parents"]),
                         {by_agent["flights"][-1]["event_id"], by_agent["hotels"][-1]["event_id"]})
        self.assertEqual(by_agent["flights"][0]["causal_parents"], [])
        self.assertTrue(any(e["event_type"] == "judge_evaluation" and e["confidence_score"] == 0.0
                            for e in by_agent["activities"]))
        self.assertTrue(any(e["tags"].get("needs_human") for e in by_agent["booking"]))

    @unittest.skipUnless(has("omniscope") and has("pydantic") and has("httpx"), "omniscope not installed")
    def test_omniscope_risk_engine_reads_it(self):
        from omniscope.server.models import TraceEvent
        from omniscope.server.risk import RiskEngine
        risk = RiskEngine().compute_all([TraceEvent(**e) for e in export.team_omniscope(self.team, "trip")])
        self.assertEqual(set(risk), {"flights", "hotels", "rank", "activities", "booking"})
        worst = max(risk, key=lambda a: risk[a].overall_failure_risk)
        self.assertEqual(worst, "activities")  # the agent whose cheap model faked success


class TestTraceViz(Base):
    def test_shape(self):
        t = export.to_trace_viz(self.activities)
        self.assertTrue(t["succeeded"])
        self.assertEqual(t["scores"]["escalations"], 1.0)
        self.assertEqual([s["action"]["tool"] for s in t["steps"]], ["plan_activities"])

    @unittest.skipUnless(has("agent_trace_viz"), "agent-trace-viz not installed")
    def test_trace_viz_accepts_it(self):
        from agent_trace_viz.schema import Trace
        tr = Trace.from_dict(export.to_trace_viz(self.booking))
        self.assertEqual([s.action.tool for s in tr.steps], ["book_flight", "book_hotel"])


class TestAres(unittest.TestCase):
    def test_grounded(self):
        ctx = lambda c: "The Golden Gate Bridge opened to traffic in 1937 in San Francisco."
        self.assertTrue(verify.grounded(ctx)({"answer": "opened in 1937"}).passed)
        bad = verify.grounded(ctx)({"answer": "it opened in 1952 after long delays"})
        self.assertFalse(bad.passed)
        self.assertIn("groundedness", bad.detail)
        self.assertFalse(verify.grounded(lambda c: "")({"answer": "x"}).passed)

    def test_command_output_uses_relative_paths(self):
        import tempfile as _t
        with _t.TemporaryDirectory() as d:
            ev = verify.command([sys.executable, "-c", "import os; print(os.getcwd() + '/x.py failed'); raise SystemExit(1)"], cwd=d)({})
        self.assertFalse(ev.passed)
        self.assertIn("./x.py failed", ev.detail)
        self.assertNotIn(d, ev.detail)

    def test_answer_f1(self):
        self.assertTrue(verify.answer_f1("1937")({"answer": "In 1937."}).passed)
        self.assertFalse(verify.answer_f1("1937")({"answer": "1952"}).passed)

    @unittest.skipUnless(has("ares.metrics"), "ares not installed")
    def test_matches_real_ares(self):
        from ares.metrics import f1, grounding
        from sparkrun.verify import _ares_f1, _ares_grounding
        for a, b in [("It opened in 1937", "The bridge opened in 1937."), ("The answer is Paris!", "paris"), ("", "x")]:
            self.assertAlmostEqual(f1(a, b), _ares_f1(a, b))
            self.assertAlmostEqual(grounding(a, b), _ares_grounding(a, b))


if __name__ == "__main__":
    unittest.main()
