# Cheap GPU inference on Vast.ai

Run sparkrun's small models on a rented GPU instead of a per-token API. The recipe is one
script: rent the cheapest reliable 24 GB GPU on Vast.ai and serve a small instruct model with
vLLM, which speaks the OpenAI API that sparkrun already uses.

```bash
pip install vastai && vastai set api-key <YOUR_KEY>
./deploy/vast/launch.sh                       # Qwen2.5-3B-Instruct (AWQ), prefix caching on
vastai show instances                         # note the public IP and the port mapped to 8000
```

```python
from sparkrun import Agent, CachedModel, OpenAICompatible

gpu = OpenAICompatible("http://<ip>:<port>/v1", "Qwen/Qwen2.5-3B-Instruct-AWQ")
agent = Agent([CachedModel(gpu, "cache/"), strong_api_model], tools, checks=checks)
```

Why this is cheap:

- **Small model first.** sparkrun escalates to an expensive model only when a check fails.
  In the travel demo, 11 of 15 model calls stayed on the small model.
- **Prefix caching.** sparkrun's system prompt is identical on every call, so vLLM's
  `--enable-prefix-caching` reuses its computation and only the compact task view is new.
- **Reply caching.** `CachedModel` answers repeated prompts from a cache for $0.
- **Quantized weights.** AWQ 4-bit fits a 3B model in a few GB, leaving room for large batches.
- **GPU for ranking too.** On the same machine, `pip install cupy-cuda12x` makes
  `sparkrun.compute` rank candidates on the GPU.

Cost per token on a rented GPU is the hourly price divided by the tokens served per hour,
so it depends on traffic. Measure it on your own workload before committing.

Status: written against the vastai CLI and the `vllm/vllm-openai:v0.6.3` image, but not yet run
end to end from this repository. It rents real hardware, so it spends money from your account.
