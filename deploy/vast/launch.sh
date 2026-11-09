#!/usr/bin/env bash
# Rent the cheapest suitable Vast.ai GPU and serve a small model with vLLM for sparkrun.
#
#   pip install vastai && vastai set api-key <YOUR_KEY>      # once
#   ./launch.sh                    # picks the cheapest reliable 24 GB GPU and starts vLLM
#   ./launch.sh Qwen/Qwen2.5-7B-Instruct-AWQ                 # or another model
#
# This spends money on your Vast.ai account (billed per second while the instance runs).
# Stop it with:  vastai destroy instance <id>
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
IMAGE="vllm/vllm-openai:v0.6.3"
QUERY='gpu_ram>=24 num_gpus=1 reliability>0.98 rentable=true inet_down>200 cuda_vers>=12.1'

offer=$(vastai search offers "$QUERY" -o 'dph+' --raw | python3 -c 'import json,sys; o=json.load(sys.stdin)[0]; print(o["id"], o["gpu_name"], round(o["dph_total"],3))')
read -r OFFER_ID GPU PRICE <<<"$offer"
echo "Cheapest match: $GPU at \$$PRICE/hour (offer $OFFER_ID)"

vastai create instance "$OFFER_ID" --image "$IMAGE" --disk 40 \
  --env '-p 8000:8000' \
  --args --model "$MODEL" --quantization awq --enable-prefix-caching \
         --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000

cat <<MSG
Started. When it is running, find its public address with:
  vastai show instances
and point sparkrun at it:
  OpenAICompatible("http://<ip>:<port>/v1", "$MODEL", usd_per_mtok_in=..., usd_per_mtok_out=...)
MSG
