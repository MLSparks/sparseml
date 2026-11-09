# AgentSparks

**Multi-agent travel planning that costs pennies and checks every claim.**

A team of small, cheap agents plans and books a trip. Every answer and booking is verified before the
traveler sees it, a stronger model is paid for only when a cheap one is caught failing, and repeat requests
are served from cache. In the reference run, a six-agent trip cost **$0.0142** in model calls, **3.2× less**
than running every call on the strong model; both cheap-model mistakes were caught and fixed, and the same
trip requested again cost **$0.0002**.

**See it:** https://mlsparks.github.io/sparseml/stack/

## What's here

```
sparkrun/        the lean agent runtime (plain Python, no dependencies)
  sparkrun/      runner, team, journal, memory, models (+ CachedModel), verify (+ ARES checks),
                 compute (GPU ranking), export (AgentDiff, Omniscope, agent-trace-viz)
  examples/      trip.py (the travel team), stack.py (the run behind the Stack page), demo.py
  tests/         57 tests; the CI "stack" job runs them against every tool, nothing skipped
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
