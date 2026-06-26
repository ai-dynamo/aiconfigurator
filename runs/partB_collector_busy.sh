#!/bin/bash
# Part B: collector GPU-busy (Sum kernel dur) per ctx size, 1-GPU sharded tp4/ep4,
# REAL MoE (no --moe-noop), latency-source=gpu (launch-gap-free backbone), nsys on.
# One vLLM launch sweeps all sizes. max-num-batched-tokens raised to fit 3696.
set -e
cd /workspace/repo/aiconfigurator
export HF_HOME=/workspace/models/hf_home
export VLLM_LOGGING_LEVEL=WARNING
uv run --active python -m collector.layerwise.vllm.collect \
  --models "Qwen/Qwen3.6-35B-A3B" --tp-sizes 4 --ep-sizes 4 \
  --phases ctx --ctx-new-tokens 128,256,512,1024,2048,3696 --ctx-past-kv 0 \
  --max-num-batched-tokens 4096 \
  --nsys-capture full --latency-source gpu \
  --ctx-warmup-runs 2 --ctx-measured-runs 6 --ctx-repeat-aggregation stable_p25 \
  --run-dir runs/partB_busy
echo "=== PART B DONE ==="
