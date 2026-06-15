#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-V4-Pro"}
export HF_TOKEN=${HF_TOKEN:-"None"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"deepseek-ai/DeepSeek-V4-Pro"}
export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"


python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed "s/^/[Frontend] /" &

PREFILL_GPU=4
PREFILL_WORKERS=1
for ((w=0; w<PREFILL_WORKERS; w++)); do
  BASE=$(( w * PREFILL_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+PREFILL_GPU-1)))
  ( CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m dynamo.trtllm \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --extra-engine-args "/workspace/engine_configs/prefill_config.yaml" \
    --disaggregation-mode prefill \
 2>&1 | sed "s/^/[Prefill $w] /" ) &
done

wait
