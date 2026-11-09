# sparkrun

**Lean, verified, crash-safe execution for AI agents on small models.**

An agent that is right 99% of the time per step finishes a 69-step task only about
half the time (0.99^69 ≈ 0.50). Long tasks fail because errors compound: the agent
says it's done when it isn't, repeats an action, loses track after its context is
trimmed, or dies halfway through. sparkrun is a small runtime that wraps any agent
loop and answers three questions for every run:

1. **Did it do the right thing?** Independent checks decide, not the model's own claim.
2. **Can it prove it?** Every step and every piece of evidence is in an append-only journal.
3. **Can it do better next time?** It tracks cost per *verified* task, escalates from a
   cheap model to a stronger one only when needed, and learns which setup works best.

Plain Python standard library, no dependencies. It runs on free local small models
(Ollama, llama.cpp, vLLM) or any OpenAI-compatible API.

## Install

```bash
pip install "agentsparks-sparkrun @ git+https://github.com/MLSparks/sparseml#subdirectory=sparkrun"
```

Or download the wheel from the [latest `sparkrun-v*` release](https://github.com/MLSparks/sparseml/releases).
The package installs as `agentsparks-sparkrun` and imports as `sparkrun`. Docs: https://mlsparks.github.io/sparseml/sparkrun/

## Quick start

```bash
cd sparkrun
python examples/demo.py                                   # offline, no model needed
ollama pull qwen2.5:3b && python examples/demo.py --ollama qwen2.5:3b   # a real local small model
python -m unittest discover -s tests                      # 34 tests
```

```python
from sparkrun import Agent, OpenAICompatible, Policy, tool, verify

@tool(side_effect=True)                 # changes the world, so it runs at most once per run
def write_file(path: str, content: str):
    """Create or overwrite a file."""
    open(path, "w").write(content)
    return "ok"

small = OpenAICompatible("http://localhost:11434/v1", "qwen2.5:3b")           # free, local
strong = OpenAICompatible("https://api.example.com/v1", "big-model",            # only if needed
                          api_key_env="EXAMPLE_API_KEY", usd_per_mtok_in=1.0, usd_per_mtok_out=4.0)

agent = Agent([small, strong], [write_file],
              checks=[verify.command("pytest -q", cwd="project")],          # proof, not opinion
              policy=Policy(max_steps=40, max_cost_usd=0.50))
result = agent.run("Make the failing test in project/ pass")
print(result.status, result.cost_usd, result.verdict.failures if result.verdict else None)
```

## Multi-agent teams

`Team` runs specialist agents that share a blackboard. Roles run in waves set by their dependencies (the same
wave runs in parallel), every role keeps sparkrun's guarantees, and team-level checks verify the combined result.
Run the same `team_id` again to resume after a crash or an approval.

```python
from sparkrun import Role, Team

team = Team([
    Role("flights", flights_agent, "Find flights"),
    Role("hotels", hotels_agent, "Find hotels"),
    Role("rank", rank_agent, "Rank combinations", after=("flights", "hotels")),
    Role("booking", booking_agent, lambda board: f"Book {board['options'][0]}", after=("rank",)),
], board=board, checks=[fits_budget, dates_line_up])
result = team.run("trip-42")          # needs_approval while booking waits for a person
team.approve("trip-42", "booking"); team.run("trip-42")
```

`Policy(gate=...)` lets an outside policy pause any tool call for approval, and `on_event` mirrors every event to a
bus or UI. The [Sparks bridge](https://github.com/MLSparks/Auto-startup-) uses both to run sparkrun teams under
Sparks guardrails.

`python examples/trip.py` runs a five-agent trip booking: flights and hotels search in parallel, ranking scores
every combination, a cheap activities model is caught claiming false success, and booking waits for approval.

## GPU only where it pays

`sparkrun.compute.best_pairs()` does the one heavy step agents hit, scoring every pair from two large candidate
sets under a budget. It uses a GPU through CuPy or PyTorch with CUDA when installed, otherwise NumPy, otherwise
plain Python, with the same result every way. Everything else stays on the CPU, where it's cheap.

## How a run works

```
 goal ──► compact task memory ──► model (cheapest first) ──► ONE JSON action
              ▲                                                  │
              │                       policy: allowed? budget? repeating? needs approval?
              │                                                  │
              │                       tool runs (side effects at most once, retries on timeouts)
              │                                                  │
              └──────────── result recorded in the journal ◄─────┘

 model says "done" ──► independent checks ──► pass: verified
                                          └─► fail: failures become MUST FIX items,
                                              escalate to a stronger model, try again
```

| Problem in long agent runs | What sparkrun does | Where |
|---|---|---|
| Agent claims success incorrectly | Only independent checks (`command`, `file_contains`, `http_ok`, `predicate`) can mark a run verified. Failed checks are fed back as "must fix". | `verify.py` |
| Agent repeats actions | Side-effect tools get an idempotency key; a repeat returns the recorded result instead of running again. | `runner.py` |
| Process crashes mid-task | Every change is journaled and fsynced. `agent.run(run_id=...)` rebuilds the run and continues without re-paying for finished steps. | `journal.py` |
| Crash *during* a side effect | sparkrun can't know whether it happened, so the run pauses as `needs_reconcile` until a person approves a retry. | `runner.py` |
| Agent gets stuck in a loop | The same call repeated within a short window escalates to a stronger model, then stops as `stuck`. | `runner.py` |
| Context gets too long or is trimmed | Each call sends a compact view: goal, must-fix items, remembered facts, newest steps. The prompt stays under `context_chars` however long the run. | `memory.py` |
| Small models emit messy JSON | The parser accepts prose, code fences, single quotes and renamed keys, and tells the model what was wrong otherwise. | `runner.py` |
| Risky actions | `needs_approval=True` pauses the run until `approve`; `allowed_tools`, step, cost and time budgets are hard limits. | `tools.py` |
| Cost is measured per token, not per outcome | `python -m sparkrun report` gives success rate and cost per verified task. | `report.py` |
| Agents don't improve | `Variants` picks between setups by verified success per dollar. | `evolve.py` |

## Why it's lean

- **Small model first.** Pass models cheapest to strongest. Escalation happens only after
  invalid replies, a failed verification, or a loop, so most steps can run on a free local model.
- **Flat prompt size.** Compact task memory replaces the growing transcript, which suits
  4k-context small models and keeps cost per step constant.
- **No infrastructure.** The journal is a JSON-lines file, so there's no database, queue or
  server to run. It works anywhere Python 3.10+ runs: a laptop, a $0 CI runner, a container.

## Performance

sparkrun's own work is small next to a model call. Measured with `python benchmarks/bench.py`
on a 4-core cloud container (your numbers will differ):

| | Before tuning | Now |
|---|---|---|
| Overhead per step, default durability | ~1.0 ms | **~0.1 ms** (≈10x less) |
| 64 tasks × 3 model calls of 50 ms | 9.8 s sequential | **0.2 s** with `run_many(workers=64)` (≈49x faster) |

- **Durability where it matters.** Every event is written and flushed immediately, so a process crash loses
  nothing. Only critical events (side-effect start and result, approvals, pauses, the final outcome) are
  forced to disk with `fsync`. Choose `Agent(..., durability="always")` to fsync everything, or `"never"` for
  scratch runs.
- **Parallel runs.** `agent.run_many(goals, workers=16)` runs independent tasks on threads. Agents mostly wait
  on the model, so throughput grows almost linearly with workers until the model server is saturated.
- **Prefix caching friendly.** The system prompt (instructions and tool list) is identical on every call, so
  local servers like llama.cpp, vLLM and Ollama can reuse its cached computation; only the compact task view changes.
- **Flat prompts.** Task memory is trimmed once when it's added and rendered within `context_chars`, so cost
  per step stays constant on long runs.

## Run statuses

| Status | Meaning | Resumable |
|---|---|---|
| `verified` | All checks passed | done |
| `unverified` | Model said done, but no checks were configured | done |
| `failed` | Checks kept failing, or the model returned a permanent error | done |
| `budget_exceeded` | Step, cost or time limit reached | done |
| `stuck` | Kept repeating itself on the strongest model | done |
| `needs_approval` | Waiting for a person to approve an action | `approve`, then `run(run_id=...)` |
| `needs_reconcile` | A side effect may or may not have happened before a crash | `approve` to retry, then `run(run_id=...)` |
| `model_unavailable` | Model server unreachable after retries | `run(run_id=...)` |

## Command line

```bash
python -m sparkrun report runs          # success rate, cost per verified task, duplicates prevented
python -m sparkrun show runs/<run_id>   # every event in one run
python -m sparkrun approve runs/<run_id>
```

## Limits

- Verification is only as strong as the checks you write. With no checks, a run ends `unverified`, never `verified`.
- "At most once" covers side effects that went through sparkrun's tools. If a tool talks to an outside
  system, give that system an idempotency key too where it supports one.
- Variant learning is a simple epsilon-greedy choice, not a full experiment framework.

## Roadmap

1. **Verification library:** ready-made checks for coding agents (tests, lint, diff scope), browser agents
   (page state) and business workflows (record exists, email delivered).
2. **Failure benchmark:** collect and label real agent failures, then measure task success rate, recovery
   rate, duplicate actions and cost per verified task against plain agent loops.
3. **Framework adapters:** wrap LangGraph, CrewAI and custom loops so their tool calls go through the journal.
4. **Simulation:** fault injection (timeouts, crashes, bad tool output) to train and test agents before production.
5. **Sparse local models:** pair with pruned and quantized models so the cheapest tier runs fast on CPU.
