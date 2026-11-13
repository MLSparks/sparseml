"""Every example workload verifies on the same infrastructure, with its cheap-model mistake caught."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import workloads
from sparkrun import CachedModel


class TestWorkloads(unittest.TestCase):
    def test_each_workload(self):
        for w in workloads.ALL:
            with self.subTest(w.DOMAIN), tempfile.TemporaryDirectory() as d:
                m = workloads.measure(w.run(Path(d)))
                self.assertEqual(m["status"], "verified")
                self.assertGreaterEqual(len(m["caught"]), 1)
                self.assertTrue(all(c.get("delivered") for c in m["caught"]))
                self.assertLess(m["cost_usd"], m["strong_only_usd"])
                tools = [e["tool"] for e in m["side_effects"]]
                self.assertEqual(len(tools), len(set(tools)))  # every side effect exactly once

    def test_repeat_is_nearly_free(self):
        cache = {}
        wrap = lambda m: CachedModel(m, cache)
        for w in workloads.ALL:
            with self.subTest(w.DOMAIN), tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                first = workloads.measure(w.run(Path(a), wrap=wrap))
                again = workloads.measure(w.run(Path(b), wrap=wrap))
                self.assertEqual(again["status"], "verified")
                self.assertLess(again["cost_usd"], first["cost_usd"] * 0.2)


if __name__ == "__main__":
    unittest.main()
