#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export HF_TOKEN=${HF_TOKEN:-"None"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"


python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed "s/^/[Frontend] /" &

AGG_GPU=1
AGG_WORKERS=8
AGG_GPU_OFFSET=0
for ((w=0; w<AGG_WORKERS; w++)); do
  BASE=$(( AGG_GPU_OFFSET + w * AGG_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+AGG_GPU-1)))
  ( CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m dynamo.trtllm \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --extra-engine-args "/workspace/engine_configs/agg_config.yaml" \
 2>&1 | sed "s/^/[Worker $w] /" ) &
done
wait
