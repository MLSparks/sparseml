# AgentSparks

**One agent framework. Any agent. A fraction of the cost.**

Run agents on it, compose them into teams, or wrap the loop you already have in any framework. Cheap models
go first and a stronger model is paid for only when an independent check catches a cheap one failing. Every
result is verified before it's used, side effects happen exactly once, repeat work comes from cache, and every
failure is traced and explained.

Measured on four different workloads (travel, coding, customer support, internal-docs research): all four
verified, **5 cheap-model mistakes caught** before anyone saw them, **$0.0358** in model calls versus $0.0750
on the strong model only, and **$0.0002** to run all four again.

**See it:** https://mlsparks.github.io/sparseml/stack/

## What's here

```
sparkrun/        the lean agent runtime (plain Python, no dependencies)
  sparkrun/      runner (Agent), team (Team), harness (Harness: your own loop), journal, memory,
                 models (+ CachedModel), verify (+ ARES checks),
                 compute (GPU ranking), export (AgentDiff, Omniscope, agent-trace-viz)
  examples/      workloads/ (travel, coding, support, research), trip.py, stack.py (the Stack page's data), demo.py
  tests/         69 tests; the CI "stack" job runs them against every tool, nothing skipped
site/            every web page, built by site/build.py and deployed by .github/workflows/pages.yml
  index.html     AgentSparks home and job board
  stack/         Stack Explorer: what the traveler got, then the tools under the hood
  sparkrun/      docs, playground and travel demo (sparkrun runs in the browser)
deploy/vast/     serve the small models with vLLM on the cheapest suitable Vast.ai GPU
```

## Under the hood

sparkrun runs the agents. Four of our open-source tools check its work, each through a plain export,
so none of them is a dependency:

| Stage | Tool | What it adds |
|---|---|---|
| Run | [sparkrun](sparkrun/) | verified results, crash-safe resume, side effects once, cheap model first |
| Trace | [Omniscope](https://github.com/sugeerth/simple_agent_tracer), [agent-trace-viz](https://github.com/sugeerth/agent-trace-viz) | causal graph across agents; risk scoring |
| Diff | [AgentDiff](https://github.com/sugeerth/Agenttracing) | the decisive step where a cheap model went wrong |
| Evaluate | [ARES](https://github.com/sugeerth/ares) | groundedness and F1 as checks; quality vs cost |

## Quick start

```bash
pip install "agentsparks-sparkrun @ git+https://github.com/MLSparks/sparseml#subdirectory=sparkrun"
cd sparkrun
python examples/trip.py --approve        # the travel team, offline
python -m unittest discover -s tests     # 57 tests
python examples/stack.py --out stack-data --with-tools   # needs the four tools on PYTHONPATH
```

## License

Apache-2.0. See [LICENSE](LICENSE).
