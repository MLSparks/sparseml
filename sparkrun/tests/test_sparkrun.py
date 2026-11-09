import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparkrun import (Agent, Journal, Policy, ScriptedModel, TaskState, TransientError, Variants,
                      parse_action, summarize, tool, verify)
from sparkrun.models import PermanentError

FAST = Policy(backoff_s=0)


def act(tool_name, **args):
    return json.dumps({"tool": tool_name, "args": args})


def done(text="finished"):
    return json.dumps({"done": text})


class Crash(BaseException):
    """Simulates the process dying (not caught by the agent)."""


class Env:
    """A tiny world with one side effect (sending) and one read."""

    def __init__(self):
        self.sent = []
        self.flaky_failures = 0

        @tool(side_effect=True)
        def send(to: str, body: str):
            """Send a message."""
            self.sent.append((to, body))
            return f"sent to {to}"

        @tool
        def lookup(name: str):
            """Look up a contact."""
            return {"alice": "alice@example.com"}.get(name, "unknown")

        @tool
        def flaky():
            """Fails twice, then works."""
            if self.flaky_failures < 2:
                self.flaky_failures += 1
                raise TransientError("timeout")
            return "ok"

        @tool(needs_approval=True, side_effect=True)
        def pay(amount: int):
            """Pay money."""
            self.sent.append(("pay", amount))
            return f"paid {amount}"

        self.tools = [send, lookup, flaky, pay]
        self.sent_once = verify.predicate(lambda ctx: (len(self.sent) == 1, f"{len(self.sent)} sent"), "sent once")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name) / "runs"
        self.env = Env()

    def tearDown(self):
        self.tmp.cleanup()

    def agent(self, models, checks=None, policy=FAST, **kw):
        return Agent(models, self.env.tools, checks=[self.env.sent_once] if checks is None else checks,
                     policy=policy, runs_dir=self.runs, **kw)


class TestParse(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_action('{"tool": "a", "args": {"x": 1}}')["args"], {"x": 1})

    def test_prose_and_fences(self):
        a = parse_action('Sure! I will do it.\n```json\n{"tool": "lookup", "args": {"name": "it\'s"}}\n```')
        self.assertEqual(a["tool"], "lookup")

    def test_single_quotes_and_aliases(self):
        a = parse_action("{'action': 'lookup', 'arguments': {'name': 'alice'}}")
        self.assertEqual((a["tool"], a["args"]), ("lookup", {"name": "alice"}))

    def test_done_aliases(self):
        self.assertEqual(parse_action('{"final": "ok"}')["done"], "ok")

    def test_garbage(self):
        from sparkrun import ParseError
        with self.assertRaises(ParseError):
            parse_action("I think we're done here")


class TestRuns(Base):
    def test_verified_path(self):
        r = self.agent(ScriptedModel([act("send", to="alice", body="hi"), done()])).run("send alice hi")
        self.assertEqual(r.status, "verified")
        self.assertEqual(len(self.env.sent), 1)
        self.assertEqual(r.steps, 2)

    def test_false_done_claim_is_caught_and_escalated(self):
        small = ScriptedModel([done("all sent!")], name="small")
        big = ScriptedModel([act("send", to="alice", body="hi"), done()], name="big")
        r = self.agent([small, big]).run("send alice hi")
        self.assertEqual(r.status, "verified")
        self.assertEqual(r.escalations, 1)
        j = Journal(r.journal)
        self.assertEqual([e["passed"] for e in j.of("verified")], [False, True])

    def test_no_checks_means_unverified(self):
        r = self.agent(ScriptedModel([done()]), checks=[]).run("anything")
        self.assertEqual(r.status, "unverified")

    def test_crash_and_resume_never_repeats_side_effect(self):
        first = ScriptedModel([act("send", to="alice", body="hi"), RuntimeError("power cut")])
        with self.assertRaises(RuntimeError):
            self.agent(first).run("send alice hi", run_id="r1")
        second = ScriptedModel([act("send", to="alice", body="hi"), done()])
        r = self.agent(second).run(run_id="r1")
        self.assertEqual(r.status, "verified")
        self.assertEqual(len(self.env.sent), 1)
        self.assertEqual(r.duplicates_prevented, 1)

    def test_crash_mid_side_effect_requires_reconcile(self):
        env = self.env

        @tool(side_effect=True)
        def send(to: str, body: str):
            """Send, then crash before the result is recorded."""
            env.sent.append((to, body))
            raise Crash()

        self.env.tools[0] = send
        with self.assertRaises(Crash):
            self.agent(ScriptedModel([act("send", to="alice", body="hi")])).run("send", run_id="r2")
        r = self.agent(ScriptedModel([done()])).run(run_id="r2")
        self.assertEqual(r.status, "needs_reconcile")
        self.assertEqual(r.pending["tool"], "send")
        self.assertEqual(len(env.sent), 1)

    def test_approval_pause_and_resume(self):
        a = self.agent(ScriptedModel([act("pay", amount=5)]), checks=[
            verify.predicate(lambda c: ("pay", 5) in self.env.sent, "paid")])
        r = a.run("pay 5", run_id="r3")
        self.assertEqual(r.status, "needs_approval")
        self.assertEqual(self.env.sent, [])
        a.approve("r3")
        r = self.agent(ScriptedModel([done()]), checks=[
            verify.predicate(lambda c: ("pay", 5) in self.env.sent, "paid")]).run(run_id="r3")
        self.assertEqual(r.status, "verified")
        self.assertEqual(self.env.sent, [("pay", 5)])

    def test_stuck_loop_escalates_then_stops(self):
        loop = [act("lookup", name="bob")] * 10
        r = self.agent([ScriptedModel(loop, name="s"), ScriptedModel(loop, name="b")]).run("find bob")
        self.assertEqual(r.status, "stuck")
        self.assertEqual(r.escalations, 1)
        self.assertLess(r.steps, 10)

    def test_step_budget(self):
        r = self.agent(ScriptedModel([act("lookup", name=str(i)) for i in range(10)]),
                       policy=Policy(max_steps=3, backoff_s=0)).run("loop")
        self.assertEqual(r.status, "budget_exceeded")
        self.assertEqual(r.steps, 3)

    def test_cost_budget(self):
        r = self.agent(ScriptedModel([act("lookup", name=str(i)) for i in range(10)], usd_per_call=0.01),
                       policy=Policy(max_cost_usd=0.025, backoff_s=0)).run("loop")
        self.assertEqual(r.status, "budget_exceeded")
        self.assertEqual(r.steps, 3)

    def test_bad_output_feedback_then_escalation(self):
        small = ScriptedModel(["hmm", "not json either"], name="small")
        big = ScriptedModel([act("send", to="a", body="b"), done()], name="big")
        r = self.agent([small, big]).run("send")
        self.assertEqual(r.status, "verified")
        self.assertEqual(r.escalations, 1)

    def test_unknown_tool_and_bad_args_get_feedback(self):
        m = ScriptedModel([act("nope"), act("send", to="a"), act("send", to="a", body="b"), done()])
        r = self.agent(m).run("send")
        self.assertEqual(r.status, "verified")
        user_msgs = Journal(r.journal).of("feedback")
        self.assertEqual(len(user_msgs), 2)

    def test_transient_tool_errors_are_retried(self):
        r = self.agent(ScriptedModel([act("flaky"), done()]), checks=[
            verify.predicate(lambda c: self.env.flaky_failures == 2, "retried")]).run("flaky")
        self.assertEqual(r.status, "verified")
        self.assertEqual(Journal(r.journal).of("tool_result")[0]["attempts"], 3)

    def test_transient_model_errors_are_retried(self):
        m = ScriptedModel([TransientError("429"), act("send", to="a", body="b"), done()])
        self.assertEqual(self.agent(m).run("send").status, "verified")

    def test_model_unavailable_pauses_and_resumes(self):
        down = ScriptedModel([TransientError("down")] * 10)
        r = self.agent(down, policy=Policy(max_model_retries=1, backoff_s=0)).run("send", run_id="r4")
        self.assertEqual(r.status, "model_unavailable")
        r = self.agent(ScriptedModel([act("send", to="a", body="b"), done()])).run(run_id="r4")
        self.assertEqual(r.status, "verified")

    def test_permanent_model_error_fails(self):
        r = self.agent(ScriptedModel([PermanentError("no such model")])).run("x")
        self.assertEqual(r.status, "failed")

    def test_finished_run_is_not_rerun(self):
        a = self.agent(ScriptedModel([act("send", to="a", body="b"), done()]))
        a.run("send", run_id="r5")
        r = self.agent(ScriptedModel([])).run(run_id="r5")
        self.assertEqual(r.status, "verified")
        self.assertEqual(len(self.env.sent), 1)

    def test_prompt_stays_small_on_long_runs(self):
        seen = []

        def model(messages):
            seen.append(len(messages[1]["content"]))
            return act("lookup", name=str(len(seen))) if len(seen) < 60 else done()

        r = self.agent(ScriptedModel(model), checks=[], context_chars=2000,
                       policy=Policy(max_steps=100, backoff_s=0)).run("long task " + "x" * 50)
        self.assertEqual(r.status, "unverified")
        self.assertLessEqual(max(seen), 2000)

    def test_remember_survives_compaction(self):
        prompts = []

        def model(messages):
            prompts.append(messages[1]["content"])
            n = len(prompts)
            if n == 1:
                return json.dumps({"tool": "lookup", "args": {"name": "0"}, "remember": "KEY DECISION: use EU region"})
            return act("lookup", name=str(n)) if n < 40 else done()

        self.agent(ScriptedModel(model), checks=[], context_chars=1500,
                   policy=Policy(max_steps=100, backoff_s=0)).run("long task")
        self.assertIn("KEY DECISION: use EU region", prompts[-1])
        self.assertIn("older steps omitted", prompts[-1])

    def test_report(self):
        self.agent(ScriptedModel([act("send", to="a", body="b"), done()], usd_per_call=0.01)).run("one")
        self.agent(ScriptedModel([done()], usd_per_call=0.01),
                   checks=[verify.predicate(lambda c: False, "never passes")],
                   policy=Policy(max_verify_failures=1, backoff_s=0)).run("two")
        s = summarize(self.runs)
        self.assertEqual((s["total"], s["verified"]), (2, 1))
        self.assertAlmostEqual(s["cost_per_verified_usd"], 0.03)


class TestEfficiency(Base):
    def test_run_many_runs_in_parallel(self):
        import time as _t

        def slow(messages):
            _t.sleep(0.05)
            return done() if "send(" in messages[1]["content"] else act("send", to="a", body="b")

        agent = Agent(ScriptedModel(slow), self.env.tools, runs_dir=self.runs, policy=FAST)
        t0 = _t.perf_counter()
        results = agent.run_many([f"task {i}" for i in range(20)], workers=20)
        elapsed = _t.perf_counter() - t0
        self.assertEqual([r.status for r in results], ["unverified"] * 20)
        self.assertEqual(len({r.run_id for r in results}), 20)
        self.assertLess(elapsed, 1.0)  # sequential would take at least 20 * 2 * 0.05 = 2s

    def test_durability_modes_all_resume_correctly(self):
        for mode in ("always", "side_effects", "never"):
            env = Env()
            run_id = f"d-{mode}"
            with self.assertRaises(RuntimeError):
                Agent(ScriptedModel([act("send", to="a", body="b"), RuntimeError("crash")]), env.tools,
                      checks=[env.sent_once], policy=FAST, runs_dir=self.runs, durability=mode).run("x", run_id=run_id)
            r = Agent(ScriptedModel([act("send", to="a", body="b"), done()]), env.tools,
                      checks=[env.sent_once], policy=FAST, runs_dir=self.runs, durability=mode).run(run_id=run_id)
            self.assertEqual((mode, r.status, len(env.sent)), (mode, "verified", 1))

    def test_bad_durability_rejected(self):
        with self.assertRaises(ValueError):
            Agent(ScriptedModel([]), [], durability="sometimes")

    def test_critical_events_are_fsynced(self):
        import os as _os
        from unittest import mock
        with mock.patch.object(_os, "fsync", wraps=_os.fsync) as fsync:
            self.agent(ScriptedModel([act("lookup", name="x"), act("send", to="a", body="b"), done()])).run("x")
        # side-effect start + result, and the final outcome; the lookup and model replies are not fsynced
        self.assertEqual(fsync.call_count, 3)


class TestCachedModel(Base):
    def test_repeat_task_costs_nothing(self):
        from sparkrun import CachedModel
        cache = {}
        script = lambda msgs: act("send", to="a", body="b") if "send(" not in msgs[1]["content"] else done()
        first = self.agent(CachedModel(ScriptedModel(script, usd_per_call=0.01), cache)).run("send it")
        self.assertEqual((first.status, first.cost_usd), ("verified", 0.02))
        inner = ScriptedModel(script, usd_per_call=0.01)
        cached = CachedModel(inner, cache)
        self.env.sent.clear()
        again = self.agent(cached).run("send it")
        self.assertEqual(again.status, "verified")
        self.assertEqual(again.cost_usd, 0.0)
        self.assertEqual((cached.hits, cached.misses, inner.calls), (2, 0, 0))

    def test_cache_on_disk_survives_a_new_process(self):
        from sparkrun import CachedModel
        d = Path(self.tmp.name) / "cache"
        m1 = CachedModel(ScriptedModel(["hello"]), d)
        m1.complete([{"role": "user", "content": "hi"}])
        m2 = CachedModel(ScriptedModel([]), d)  # an empty script would fail if called
        self.assertEqual(m2.complete([{"role": "user", "content": "hi"}])[0], "hello")
        self.assertEqual(m2.hits, 1)

    def test_different_prompts_are_not_confused(self):
        from sparkrun import CachedModel
        m = CachedModel(ScriptedModel(["one", "two"]), {})
        self.assertEqual(m.complete([{"role": "user", "content": "a"}])[0], "one")
        self.assertEqual(m.complete([{"role": "user", "content": "b"}])[0], "two")
        self.assertEqual(m.misses, 2)


class TestOpenAICompatible(unittest.TestCase):
    """Runs the HTTP adapter against a stub server shaped like Ollama / llama.cpp / vLLM."""

    def setUp(self):
        import http.server
        import threading

        replies = [act("lookup", name="alice"), done("found alice")]
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body, self.headers.get("Authorization")))
                if len(outer.requests) == 1:
                    self.send_response(503)
                    self.end_headers()
                    return
                out = {"choices": [{"message": {"content": replies.pop(0)}}],
                       "usage": {"prompt_tokens": 1000, "completion_tokens": 20}}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.shutdown()
        self.tmp.cleanup()

    def test_run_against_server(self):
        import os
        from sparkrun import OpenAICompatible
        os.environ["SPARKRUN_TEST_KEY"] = "secret"
        port = self.server.server_address[1]
        m = OpenAICompatible(f"http://127.0.0.1:{port}/v1", "qwen2.5:3b", api_key_env="SPARKRUN_TEST_KEY",
                             usd_per_mtok_in=0.10, usd_per_mtok_out=0.40)
        env = Env()
        r = Agent(m, env.tools, checks=[verify.predicate(lambda c: "alice" in c["answer"], "mentions alice")],
                  policy=FAST, runs_dir=Path(self.tmp.name)).run("find alice")
        self.assertEqual(r.status, "verified")
        self.assertEqual(self.requests[0][0], "/v1/chat/completions")
        self.assertEqual(self.requests[1][1]["model"], "qwen2.5:3b")
        self.assertEqual(self.requests[1][2], "Bearer secret")
        self.assertEqual(len(self.requests), 3)  # one 503, retried, then two good calls
        self.assertAlmostEqual(r.cost_usd, 2 * (1000 * 0.10 + 20 * 0.40) / 1e6)


class TestPieces(unittest.TestCase):
    def test_torn_journal_line_is_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "j.jsonl"
            j = Journal(p)
            j.append("a")
            with p.open("a") as f:
                f.write('{"seq": 1, "kind": "b"')  # crash mid-write
            j2 = Journal(p)
            self.assertEqual(len(j2), 1)
            j2.append("c")
            self.assertEqual([e["kind"] for e in Journal(p)], ["a", "c"])

    def test_verify_command(self):
        ok = verify.command([sys.executable, "-c", "print('hello')"], expect_output="hello")({})
        bad = verify.command([sys.executable, "-c", "import sys; sys.exit(3)"])({})
        self.assertTrue(ok.passed)
        self.assertFalse(bad.passed)
        self.assertIn("exit code 3", bad.detail)

    def test_crashing_check_fails_closed(self):
        from sparkrun.verify import run_checks
        v = run_checks([verify.predicate(lambda c: 1 / 0, "boom")], {})
        self.assertFalse(v.passed)
        self.assertIn("ZeroDivisionError", v.evidence[0].detail)

    def test_memory_keeps_goal_and_must_fix(self):
        t = TaskState(goal="G", open_items=["tests fail"], recent=[f"step {i} " + "y" * 200 for i in range(100)])
        out = t.render(1500)
        self.assertTrue(out.startswith("GOAL:\nG"))
        self.assertIn("tests fail", out)
        self.assertLessEqual(len(out), 1500)

    def test_variants_prefer_verified_and_cheap(self):
        from types import SimpleNamespace as R
        with tempfile.TemporaryDirectory() as d:
            v = Variants(["a", "b"], path=Path(d) / "v.json", epsilon=0, seed=1)
            for _ in range(3):
                v.record("a", R(status="failed", cost_usd=0.01))
                v.record("b", R(status="verified", cost_usd=0.01))
            self.assertEqual(v.choose(), "b")
            self.assertEqual(Variants(["a", "b"], path=Path(d) / "v.json", epsilon=0).choose(), "b")


if __name__ == "__main__":
    unittest.main()
