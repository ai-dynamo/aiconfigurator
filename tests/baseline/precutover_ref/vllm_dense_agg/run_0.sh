#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export HF_TOKEN=${HF_TOKEN:-"None"}
python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed "s/^/[Frontend] /" &

AGG_GPU=1
AGG_WORKERS=8
AGG_GPU_OFFSET=0
AGG_SYSTEM_PORT_BASE=${AGG_SYSTEM_PORT_BASE:-${DYN_SYSTEM_PORT1:-8081}}
AGG_EVENT_PORT_BASE=${AGG_EVENT_PORT_BASE:-${DYN_VLLM_KV_EVENT_PORT:-20081}}
AGG_SIDE_CHANNEL_PORT_BASE=${AGG_SIDE_CHANNEL_PORT_BASE:-${VLLM_NIXL_SIDE_CHANNEL_PORT:-20097}}
for ((w=0; w<AGG_WORKERS; w++)); do
  BASE=$(( AGG_GPU_OFFSET + w * AGG_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+AGG_GPU-1)))
  SYSTEM_PORT=$(( AGG_SYSTEM_PORT_BASE + w ))
  if [[ $w -eq 1 && -n "${DYN_SYSTEM_PORT2:-}" ]]; then
    SYSTEM_PORT=$DYN_SYSTEM_PORT2
  fi
  EVENT_PORT=$(( AGG_EVENT_PORT_BASE + w ))
  SIDE_CHANNEL_PORT=$(( AGG_SIDE_CHANNEL_PORT_BASE + w ))
  ( DYN_SYSTEM_PORT=$SYSTEM_PORT \
    DYN_VLLM_KV_EVENT_PORT=$EVENT_PORT \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m dynamo.vllm \
    --model "$MODEL_PATH" \
\
    --tensor-parallel-size "1" --pipeline-parallel-size "1" --data-parallel-size "1" --max-model-len "6500" --max-num-seqs "512" --max-num-batched-tokens 6012 2>&1 | sed "s/^/[Worker $w] /" ) &
done
wait
