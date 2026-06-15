#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-V4-Pro"}
export HF_TOKEN=${HF_TOKEN:-"None"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"deepseek-ai/DeepSeek-V4-Pro"}
export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

FRONTEND_SYSTEM_PORT=${FRONTEND_SYSTEM_PORT:-8080}
AGG_SYSTEM_PORT=${AGG_SYSTEM_PORT:-8081}
PREFILL_WORKERS=1
DECODE_WORKERS=1
PREFILL_SYSTEM_PORT_BASE=${PREFILL_SYSTEM_PORT_BASE:-8082}
DECODE_SYSTEM_PORT_BASE=${DECODE_SYSTEM_PORT_BASE:-$((PREFILL_SYSTEM_PORT_BASE + PREFILL_WORKERS))}

OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed "s/^/[Frontend] /" &

AGG_GPU=8
AGG_WORKERS=1
AGG_GPU_OFFSET=0
for ((w=0; w<AGG_WORKERS; w++)); do
  BASE=$(( AGG_GPU_OFFSET + w * AGG_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+AGG_GPU-1)))
  WORKER_IDX=$(( w + 1 ))
  SYSTEM_PORT=$(( AGG_SYSTEM_PORT + w ))
  WORKER_NAME="dynamo-worker"
  if (( AGG_WORKERS > 1 )); then
    WORKER_NAME="${WORKER_NAME}-${WORKER_IDX}"
  fi
  ( CUDA_VISIBLE_DEVICES=$GPU_LIST \
  OTEL_SERVICE_NAME="${WORKER_NAME}" \
  DYN_SYSTEM_PORT="${SYSTEM_PORT}" \
  python3 -m dynamo.sglang \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 --max-running-requests 512 --max-prefill-tokens 5500 --enable-mixed-chunk --expert-parallel-size 1 --cuda-graph-bs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 16 20 24 28 32 32 40 48 56 64 64 80 96 112 128 128 160 192 224 256 256 320 384 448 512 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 --moe-runner-backend deepep_moe --trust-remote-code --disable-flashinfer-autotune --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --host "0.0.0.0" \
    --enable-metrics 2>&1 | sed "s/^/[Worker $w] /" ) &
done
wait
