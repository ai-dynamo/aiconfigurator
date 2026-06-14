#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-30B-A3B"}
export HF_TOKEN=${HF_TOKEN:-"None"}
python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed "s/^/[Frontend] /" &

PREFILL_WORKERS=1
DECODE_WORKERS=7
DECODE_SYSTEM_PORT_BASE=${DECODE_SYSTEM_PORT_BASE:-${DYN_SYSTEM_PORT1:-8081}}
PREFILL_SYSTEM_PORT_BASE=${PREFILL_SYSTEM_PORT_BASE:-${DYN_SYSTEM_PORT2:-$((DECODE_SYSTEM_PORT_BASE + DECODE_WORKERS))}}
PREFILL_EVENT_PORT_BASE=${PREFILL_EVENT_PORT_BASE:-${DYN_VLLM_KV_EVENT_PORT:-20081}}
PREFILL_SIDE_CHANNEL_PORT_BASE=${PREFILL_SIDE_CHANNEL_PORT_BASE:-${VLLM_NIXL_SIDE_CHANNEL_PORT:-20097}}
DECODE_EVENT_PORT_BASE=${DECODE_EVENT_PORT_BASE:-$((PREFILL_EVENT_PORT_BASE + PREFILL_WORKERS))}
DECODE_SIDE_CHANNEL_PORT_BASE=${DECODE_SIDE_CHANNEL_PORT_BASE:-$((PREFILL_SIDE_CHANNEL_PORT_BASE + PREFILL_WORKERS))}

PREFILL_GPU=1
for ((w=0; w<PREFILL_WORKERS; w++)); do
  BASE=$(( w * PREFILL_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+PREFILL_GPU-1)))
  SYSTEM_PORT=$(( PREFILL_SYSTEM_PORT_BASE + w ))
  EVENT_PORT=$(( PREFILL_EVENT_PORT_BASE + w ))
  SIDE_CHANNEL_PORT=$(( PREFILL_SIDE_CHANNEL_PORT_BASE + w ))
  ( DYN_SYSTEM_PORT=$SYSTEM_PORT \
  DYN_VLLM_KV_EVENT_PORT=$EVENT_PORT \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
  CUDA_VISIBLE_DEVICES=$GPU_LIST \
    python3 -m dynamo.vllm \
      --model "$MODEL_PATH" \
\
      --tensor-parallel-size "1" --pipeline-parallel-size "1" --data-parallel-size "1" --max-num-seqs "128" --max-num-batched-tokens 5500 --is-prefill-worker --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' 2>&1 | sed "s/^/[Prefill $w] /" ) &
done

DECODE_GPU=1
DECODE_GPU_OFFSET=1
for ((w=0; w<DECODE_WORKERS; w++)); do
  BASE=$(( DECODE_GPU_OFFSET + w * DECODE_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+DECODE_GPU-1)))
  SYSTEM_PORT=$(( DECODE_SYSTEM_PORT_BASE + w ))
  EVENT_PORT=$(( DECODE_EVENT_PORT_BASE + w ))
  SIDE_CHANNEL_PORT=$(( DECODE_SIDE_CHANNEL_PORT_BASE + w ))
  ( DYN_SYSTEM_PORT=$SYSTEM_PORT \
  DYN_VLLM_KV_EVENT_PORT=$EVENT_PORT \
  VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
  CUDA_VISIBLE_DEVICES=$GPU_LIST \
    python3 -m dynamo.vllm \
      --model "$MODEL_PATH" \
\
      --tensor-parallel-size "1" --pipeline-parallel-size "1" --data-parallel-size "1" --max-num-seqs "512" --max-num-batched-tokens 512 --is-decode-worker --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' 2>&1 | sed "s/^/[Decode $w] /" ) &
done
wait
