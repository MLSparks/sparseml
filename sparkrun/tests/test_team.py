import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from sparkrun import Agent, Policy, Role, ScriptedModel, Team, compute, tool, verify

import trip  # examples/trip.py


def act(t, **a):
    return json.dumps({"tool": t, "args": a})


DONE = json.dumps({"done": "ok"})


class TestCompute(unittest.TestCase):
    def data(self, n=60, m=80, seed=3):
        r = random.Random(seed)
        return ([r.randint(100, 900) for _ in range(n)], [r.random() * 10 for _ in range(n)],
                [r.randint(50, 400) for _ in range(m)], [r.random() * 10 for _ in range(m)])

    def test_backends_agree(self):
        ac, as_, bc, bs = self.data()
        ref = compute.best_pairs(ac, as_, bc, bs, 1500, b_mult=3, k=10, use="python")
        self.assertEqual(len(ref), 10)
        for i, j, s, c in ref:
            self.assertLessEqual(c, 1500)
            self.assertAlmostEqual(s, as_[i] + bs[j])
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")
        got = compute.best_pairs(ac, as_, bc, bs, 1500, b_mult=3, k=10, use="numpy")
        self.assertEqual([(i, j) for i, j, _, _ in got], [(i, j) for i, j, _, _ in ref])

    def test_nothing_fits(self):
        ac, as_, bc, bs = self.data()
        self.assertEqual(compute.best_pairs(ac, as_, bc, bs, 10, use="python"), [])
        self.assertEqual(compute.best_pairs(ac, as_, bc, bs, 10), [])

    def test_backend_is_known(self):
        self.assertIn(compute.backend(), {"cupy", "torch-cuda", "numpy", "python"})


class TestTeam(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_waves_parallel_and_board(self):
        board, events = {}, []

        @tool
        def put(key: str, value: str):
            """Write to the board."""
            board[key] = value
            return "ok"

        def agent(key):
            return Agent(ScriptedModel([act("put", key=key, value=key.upper()), DONE]), [put],
                         checks=[verify.predicate(lambda c, k=key: k in board, f"{key} written")],
                         policy=Policy(backoff_s=0))

        team = Team([Role("a", agent("a"), "write a"), Role("b", agent("b"), "write b"),
                     Role("c", agent("c"), lambda b: f"combine {b['a']} and {b['b']}", after=("a", "b"))],
                    board=board, runs_dir=self.runs,
                    checks=[verify.predicate(lambda c: c["board"].keys() >= {"a", "b", "c"}, "all written")],
                    on_event=lambda role, e: events.append((role, e["kind"])))
        self.assertEqual(team.waves, [["a", "b"], ["c"]])
        r = team.run("t1")
        self.assertEqual(r.status, "verified")
        self.assertEqual(set(r.roles), {"a", "b", "c"})
        self.assertIn(("c", "finished"), events)
        self.assertTrue((self.runs / "t1.board.json").exists())

    def test_cycle_and_unknown_dependency_rejected(self):
        a = Agent(ScriptedModel([]), [])
        with self.assertRaises(ValueError):
            Team([Role("x", a, "g", after=("y",)), Role("y", a, "g", after=("x",))])
        with self.assertRaises(ValueError):
            Team([Role("x", a, "g", after=("nope",))])

    def test_failed_role_stops_the_team(self):
        bad = Agent(ScriptedModel([DONE]), [], checks=[verify.predicate(lambda c: False, "never")],
                    policy=Policy(max_verify_failures=1, backoff_s=0))
        later = Agent(ScriptedModel([DONE]), [])
        r = Team([Role("bad", bad, "g"), Role("later", later, "g", after=("bad",))], runs_dir=self.runs).run("t2")
        self.assertEqual((r.status, r.waiting_on), ("failed", "bad"))
        self.assertNotIn("later", r.roles)

    def test_stop_before_then_continue(self):
        board = {}
        first = Agent(ScriptedModel([DONE]), [])
        second = Agent(ScriptedModel([DONE]), [])
        team = Team([Role("one", first, "g"), Role("two", second, "g", after=("one",))], board=board, runs_dir=self.runs)
        r = team.run("t3", stop_before=["two"])
        self.assertEqual((r.status, r.waiting_on), ("paused", "two"))
        r = team.run("t3")
        self.assertEqual(r.status, "unverified")


class TestTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_trip_with_approval_and_resume_in_a_new_process(self):
        req = trip.TripRequest(options=120)
        team = trip.build_team(req, self.runs, gate=trip.spending_gate)
        r = team.run("trip")
        self.assertEqual((r.status, r.waiting_on), ("needs_approval", "booking"))
        self.assertEqual(r.roles["activities"].escalations, 1)  # lazy small model was caught
        self.assertNotIn("bookings", team.board)

        for _ in range(2):  # approve flight, then hotel; rebuild the team each time like a new process
            team = trip.build_team(req, self.runs, gate=trip.spending_gate)
            team.approve("trip", "booking")
            r = team.run("trip")
        self.assertEqual(r.status, "verified", r.verdict)
        self.assertEqual(sorted(b["type"] for b in team.board["bookings"]), ["flight", "hotel"])
        self.assertTrue(all(ev.passed for ev in r.verdict.evidence))

        again = trip.build_team(req, self.runs, gate=trip.spending_gate).run("trip")  # finished: nothing re-runs
        self.assertEqual(again.status, "verified")
        self.assertEqual(len(again.board["bookings"]), 2)

    def test_precomputed_rank_is_used(self):
        req = trip.TripRequest(options=50)
        board = {}
        team = trip.build_team(req, self.runs, board=board, gate=trip.spending_gate)
        r = team.run("t", stop_before=["rank"])
        self.assertEqual(r.status, "paused")
        board["accelerated_rank"] = {"pairs": [[0, 0, 1.0, 100.0]], "backend": "webgpu", "ms": 1.5}
        team.run("t")
        self.assertEqual(board["rank_stats"]["backend"], "webgpu")
        self.assertEqual(board["options"][0]["flight"]["id"], board["flights"][0]["id"])


if __name__ == "__main__":
    unittest.main()
