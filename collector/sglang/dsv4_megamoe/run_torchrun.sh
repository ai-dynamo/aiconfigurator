#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SYSTEM_NAME="${SYSTEM_NAME:-gb200}"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"
EP_SIZE="${EP_SIZE:-8}"
NNODES="${NNODES:-}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MODEL_CONFIG="${MODEL_CONFIG:-dsv4_pro}"
OUTPUT_PATH="${OUTPUT_PATH:-${AIC_ROOT}/collector/sglang/dsv4_megamoe/results}"
DISTRIBUTIONS="${DISTRIBUTIONS:-balanced,power_law_1.01,power_law_1.2,power_law_sampled_1.9}"
SOURCE_POLICY="${SOURCE_POLICY:-random}"
ROUTING_SEED="${ROUTING_SEED:-0}"
ROUTING_SEEDS="${ROUTING_SEEDS:-}"
ROUTING_DUMP_ROOT="${ROUTING_DUMP_ROOT:-}"
ROUTING_DUMP_LAYER="${ROUTING_DUMP_LAYER:-bottleneck}"
ROUTING_DUMP_LAYERS="${ROUTING_DUMP_LAYERS:-}"
ROUTING_DUMP_WEIGHT_POLICY="${ROUTING_DUMP_WEIGHT_POLICY:-uniform}"
PHASES="${PHASES:-context,generation}"
PREFILL_TOKENS="${PREFILL_TOKENS:-1024,2048,4096,8192,16384,32768}"
DECODE_TOKENS="${DECODE_TOKENS:-1,2,4,8,16,32,64,128,256,512}"
PRE_DISPATCH="${PRE_DISPATCH:-sglang_jit}"
INCLUDE_ROUTED_SCALE="${INCLUDE_ROUTED_SCALE:-1}"
RENORMALIZE_TOPK_WEIGHTS="${RENORMALIZE_TOPK_WEIGHTS:-1}"
NUM_WARMUP="${NUM_WARMUP:-5}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
NUM_MAX_TOKENS_PER_RANK="${NUM_MAX_TOKENS_PER_RANK:-0}"
CAP_POLICY="${CAP_POLICY:-fixed}"
WRITE_DEBUG_OUTPUT="${WRITE_DEBUG_OUTPUT:-${AIC_DSV4_MEGAMOE_DEBUG:-0}}"
PERF_FILE="${PERF_FILE:-dsv4_megamoe_module_perf.txt}"
AIC_WAIT_FOR_ALL_NODES="${AIC_WAIT_FOR_ALL_NODES:-0}"
AIC_WAIT_FOR_ALL_NODES_MAX_ATTEMPTS="${AIC_WAIT_FOR_ALL_NODES_MAX_ATTEMPTS:-60}"

MAX_CASE_TOKENS=0
PHASES_COMPACT="${PHASES//[[:space:]]/}"
if [[ ",${PHASES_COMPACT}," == *",context,"* ]]; then
  for token in ${PREFILL_TOKENS//,/ }; do
    (( token > MAX_CASE_TOKENS )) && MAX_CASE_TOKENS="${token}"
  done
fi
if [[ ",${PHASES_COMPACT}," == *",generation,"* ]]; then
  for token in ${DECODE_TOKENS//,/ }; do
    (( token > MAX_CASE_TOKENS )) && MAX_CASE_TOKENS="${token}"
  done
fi
if (( NUM_MAX_TOKENS_PER_RANK <= 0 )); then
  NUM_MAX_TOKENS_PER_RANK="${MAX_CASE_TOKENS}"
fi
if (( NUM_MAX_TOKENS_PER_RANK <= 0 )); then
  echo "NUM_MAX_TOKENS_PER_RANK could not be inferred from PHASES=${PHASES}" >&2
  exit 1
fi

if [[ -z "${GPUS_PER_NODE}" ]]; then
  case "${SYSTEM_NAME^^}" in
    B200|B200_SXM|B300|B300_SXM) GPUS_PER_NODE=8 ;;
    GB200|GB300) GPUS_PER_NODE=4 ;;
    *) echo "GPUS_PER_NODE must be set for SYSTEM_NAME=${SYSTEM_NAME}" >&2; exit 1 ;;
  esac
fi

if [[ -z "${NNODES}" ]]; then
  NNODES=$(( (EP_SIZE + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
fi

if (( EP_SIZE != NNODES * GPUS_PER_NODE )); then
  echo "EP_SIZE must equal NNODES * GPUS_PER_NODE for one-rank-per-GPU collection." >&2
  echo "EP_SIZE=${EP_SIZE} NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
  exit 1
fi

if (( NNODES > 1 )) && [[ "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
  echo "MASTER_ADDR must be reachable from all nodes for multi-node collection." >&2
  echo "MASTER_ADDR=${MASTER_ADDR} NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE}" >&2
  echo "Set MASTER_ADDR to the rank-0 host or service DNS name." >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"
SGLANG_PYTHONPATHS=()
for candidate in /workspace/sglang/python /sgl-workspace/sglang/python; do
  if [[ -d "${candidate}" ]]; then
    SGLANG_PYTHONPATHS+=("${candidate}")
  fi
done
if ((${#SGLANG_PYTHONPATHS[@]})); then
  SGLANG_PYTHONPATH="$(IFS=:; echo "${SGLANG_PYTHONPATHS[*]}")"
  export PYTHONPATH="${AIC_ROOT}:${SGLANG_PYTHONPATH}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${AIC_ROOT}:${PYTHONPATH:-}"
fi
export MASTER_ADDR MASTER_PORT
export AIC_SYSTEM_NAME="${SYSTEM_NAME}"
export GPUS_PER_NODE
export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE="${SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE:-1}"
export SGLANG_OPT_FIX_HASH_MEGA_MOE="${SGLANG_OPT_FIX_HASH_MEGA_MOE:-1}"
export SGLANG_OPT_FIX_MEGA_MOE_MEMORY="${SGLANG_OPT_FIX_MEGA_MOE_MEMORY:-1}"
export SGLANG_OPT_FIX_NEXTN_MEGA_MOE="${SGLANG_OPT_FIX_NEXTN_MEGA_MOE:-1}"
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK="${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK:-${NUM_MAX_TOKENS_PER_RANK}}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-0}"

echo "[dsv4-megamoe] force MegaMoE env: SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=${SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE} SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=${SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK} SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"

if [[ "${AIC_WAIT_FOR_ALL_NODES}" == "1" && "${NNODES}" -gt 1 ]]; then
  MASTER_HOST="${MASTER_ADDR%%.*}"
  MASTER_DOMAIN="${MASTER_ADDR#*.}"
  JOB_PREFIX="${MASTER_HOST%-0}"
  if [[ -z "${JOB_PREFIX}" || "${JOB_PREFIX}" == "${MASTER_HOST}" ]]; then
    echo "AIC_WAIT_FOR_ALL_NODES=1 expects MASTER_ADDR like <job>-0.<job>.<namespace>.svc.cluster.local, got ${MASTER_ADDR}" >&2
    exit 1
  fi
  echo "[dsv4-megamoe] waiting for ${NNODES} indexed job pod DNS records before torchrun"
  for ((idx = 0; idx < NNODES; idx++)); do
    peer_host="${JOB_PREFIX}-${idx}.${MASTER_DOMAIN}"
    attempts=0
    until python3 - "${peer_host}" <<'PY'
import socket
import sys

try:
    socket.getaddrinfo(sys.argv[1], None)
except OSError:
    sys.exit(1)
PY
    do
      attempts=$((attempts + 1))
      if (( attempts >= AIC_WAIT_FOR_ALL_NODES_MAX_ATTEMPTS )); then
        echo "[dsv4-megamoe] timed out waiting for peer ${peer_host}" >&2
        echo "MASTER_ADDR=${MASTER_ADDR} NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} attempts=${attempts}" >&2
        exit 1
      fi
      echo "[dsv4-megamoe] waiting for peer ${peer_host}"
      sleep 10
    done
  done
  echo "[dsv4-megamoe] all indexed job pod DNS records are resolvable"
fi

if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN=(torchrun)
else
  TORCHRUN=(python3 -m torch.distributed.run)
fi

RUN_CMD=(
  "${TORCHRUN[@]}"
  --nnodes="${NNODES}" \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${AIC_ROOT}/collector/sglang/collect_dsv4_megamoe.py" \
  --model-config "${MODEL_CONFIG}" \
  --system-name "${SYSTEM_NAME}" \
  --gpus-per-node "${GPUS_PER_NODE}" \
  --phases "${PHASES}" \
  --prefill-tokens "${PREFILL_TOKENS}" \
  --decode-tokens "${DECODE_TOKENS}" \
  --distributions "${DISTRIBUTIONS}" \
  --source-policy "${SOURCE_POLICY}" \
  --routing-seed "${ROUTING_SEED}" \
  --routing-seeds "${ROUTING_SEEDS}" \
  --routing-dump-root "${ROUTING_DUMP_ROOT}" \
  --routing-dump-layer "${ROUTING_DUMP_LAYER}" \
  --routing-dump-layers "${ROUTING_DUMP_LAYERS}" \
  --routing-dump-weight-policy "${ROUTING_DUMP_WEIGHT_POLICY}" \
  --num-max-tokens-per-rank "${NUM_MAX_TOKENS_PER_RANK}" \
  --cap-policy "${CAP_POLICY}" \
  --pre-dispatch "${PRE_DISPATCH}" \
  --include-routed-scale "${INCLUDE_ROUTED_SCALE}" \
  --renormalize-topk-weights "${RENORMALIZE_TOPK_WEIGHTS}" \
  --num-warmup "${NUM_WARMUP}" \
  --num-iterations "${NUM_ITERATIONS}" \
  --output-path "${OUTPUT_PATH}" \
  --write-debug-output "${WRITE_DEBUG_OUTPUT}" \
  --perf-file "${PERF_FILE}"
)

if [[ "${AIC_NSYS_PROFILE:-0}" == "1" ]]; then
  if ! command -v nsys >/dev/null 2>&1 && [[ -d /nsys-tools ]]; then
    for nsys_bin in /nsys-tools/NsightSystems-cli-*/target-linux-sbsa-armv8/nsys /nsys-tools/NsightSystems-cli-*/target-linux-x64/nsys; do
      if [[ -x "${nsys_bin}" ]]; then
        export PATH="$(dirname "${nsys_bin}"):${PATH}"
        break
      fi
    done
  fi
  if ! command -v nsys >/dev/null 2>&1; then
    echo "AIC_NSYS_PROFILE=1 but nsys is not available in PATH" >&2
    exit 1
  fi
  NSYS_OUTPUT_DIR="${AIC_NSYS_OUTPUT_DIR:-${OUTPUT_PATH}/nsys}"
  mkdir -p "${NSYS_OUTPUT_DIR}"
  NSYS_OUTPUT="${NSYS_OUTPUT_DIR}/node${NODE_RANK}"
  echo "[dsv4-megamoe] nsys profile output=${NSYS_OUTPUT}.nsys-rep"
  exec nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --output="${NSYS_OUTPUT}" \
    "${RUN_CMD[@]}"
fi

exec "${RUN_CMD[@]}"
