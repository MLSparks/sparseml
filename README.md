# AgentSparks

This repository holds two things:

| Folder | What it is |
|---|---|
| [`sparkrun/`](sparkrun/) | **sparkrun**: lean, verified, crash-safe execution for AI agents on small models. Plain Python, no dependencies. |
| [`agentsparks-site/`](agentsparks-site/) | The AgentSparks website, with the job board and Google Sheet applications. |

## Links

- Website: https://mlsparks.github.io/sparseml/
- sparkrun docs: https://mlsparks.github.io/sparseml/sparkrun/
- sparkrun playground (runs in your browser): https://mlsparks.github.io/sparseml/sparkrun/play/

## sparkrun quick start

```bash
pip install "agentsparks-sparkrun @ git+https://github.com/MLSparks/sparseml#subdirectory=sparkrun"
cd sparkrun && python examples/demo.py && python -m unittest discover -s tests
```

## How the site is published

The live site is served by GitHub Pages from the `gh-pages` branch. `agentsparks-site/index.html` is the source
for the home page; copy it to `gh-pages` to publish changes.

## License

Apache-2.0. See [LICENSE](LICENSE).
