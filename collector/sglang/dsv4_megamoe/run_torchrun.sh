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
PHASES="${PHASES:-context,generation}"
PREFILL_TOKENS="${PREFILL_TOKENS:-1024,2048,4096,8192,16384,32768}"
DECODE_TOKENS="${DECODE_TOKENS:-1,2,4,8,16,32,64,128,256,512}"
PRE_DISPATCH="${PRE_DISPATCH:-sglang_jit}"
INCLUDE_ROUTED_SCALE="${INCLUDE_ROUTED_SCALE:-1}"
RENORMALIZE_TOPK_WEIGHTS="${RENORMALIZE_TOPK_WEIGHTS:-1}"
NUM_WARMUP="${NUM_WARMUP:-5}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
NUM_MAX_TOKENS_PER_RANK="${NUM_MAX_TOKENS_PER_RANK:-}"
CAP_POLICY="${CAP_POLICY:-case_tokens}"
PERF_FILE="${PERF_FILE:-dsv4_megamoe_module_perf.txt}"
AIC_WAIT_FOR_ALL_NODES="${AIC_WAIT_FOR_ALL_NODES:-0}"
AIC_WAIT_FOR_ALL_NODES_MAX_ATTEMPTS="${AIC_WAIT_FOR_ALL_NODES_MAX_ATTEMPTS:-60}"

if ! [[ "${NUM_MAX_TOKENS_PER_RANK}" =~ ^[0-9]+$ ]] || (( NUM_MAX_TOKENS_PER_RANK <= 0 )); then
  echo "NUM_MAX_TOKENS_PER_RANK must be set to a positive integer by the renderer or runner." >&2
  exit 1
fi
case "${CAP_POLICY}" in
  fixed|case_tokens) ;;
  *)
    echo "CAP_POLICY must be fixed or case_tokens." >&2
    exit 1
    ;;
esac

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
_write_rank0_env() {
  {
    env | sort \
      | grep -E '^(AIC_|CUDA_|NCCL_|SGLANG_|TORCH_|RANK=|WORLD_SIZE=|LOCAL_RANK=|NODE_RANK=|NNODES=|MASTER_ADDR=|MASTER_PORT=|MODEL_CONFIG=|SYSTEM_NAME=|GPUS_PER_NODE=|EP_SIZE=|OUTPUT_PATH=|PERF_FILE=|PHASES=|PREFILL_TOKENS=|DECODE_TOKENS=|DISTRIBUTIONS=|SOURCE_POLICY=|PRE_DISPATCH=|NUM_MAX_TOKENS_PER_RANK=|CAP_POLICY=)' \
      | grep -Evi '(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|PRIVATE|SSH|KEY)' \
      || true
  } >"${OUTPUT_PATH}/rank0_env.txt"
}
if [[ -z "${SGLANG_VERSION:-}" ]]; then
  SGLANG_VERSION="$(python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("sglang"))
except PackageNotFoundError:
    print("unknown")
PY
)"
fi
export SGLANG_VERSION
if [[ "${NODE_RANK}" == "0" ]]; then
  printf '%s\n' "${SGLANG_VERSION}" >"${OUTPUT_PATH}/sglang_version.txt"
  _write_rank0_env
fi

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
  --num-max-tokens-per-rank "${NUM_MAX_TOKENS_PER_RANK}" \
  --cap-policy "${CAP_POLICY}" \
  --pre-dispatch "${PRE_DISPATCH}" \
  --include-routed-scale "${INCLUDE_ROUTED_SCALE}" \
  --renormalize-topk-weights "${RENORMALIZE_TOPK_WEIGHTS}" \
  --num-warmup "${NUM_WARMUP}" \
  --num-iterations "${NUM_ITERATIONS}" \
  --output-path "${OUTPUT_PATH}" \
  --perf-file "${PERF_FILE}" \
  --sglang-version "${SGLANG_VERSION}"
)

exec "${RUN_CMD[@]}"
