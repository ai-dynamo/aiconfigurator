#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

: "${MODEL_ID:?}"
: "${MODEL_REVISION:?}"
: "${DIST_INIT_ADDR:?}"
: "${SERVER_PORT:?}"
: "${TP_SIZE:?}"
: "${NNODES:?}"
: "${ARTIFACT_DIR:?}"

export HF_HOME=/hfcache
export HF_HUB_CACHE=/hfcache
export HUGGINGFACE_HUB_CACHE=/hfcache
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_OFFLINE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
# FlashInfer, DeepGEMM, and TorchInductor use file locks while compiling missing
# kernels.  The default remains job-local.  Large-model campaigns may mount a
# persistent COMPILE_CACHE_ROOT and supply a stable COMPILE_CACHE_KEY; separate
# SLURM_PROCID directories prevent independent nodes from replacing each
# other's lock files while still allowing dependent jobs to reuse the cache.
if [[ -n "${COMPILE_CACHE_ROOT:-}" ]]; then
    compile_cache_key="${COMPILE_CACHE_KEY:-${MODEL_ID}-${MODEL_REVISION}-tp${TP_SIZE}}"
    compile_cache_key="${compile_cache_key//\//--}"
    compile_cache_key="${compile_cache_key//:/-}"
    compile_cache_dir="$COMPILE_CACHE_ROOT/$compile_cache_key/node-$SLURM_PROCID"
else
    compile_cache_dir="/tmp/sglang-compile-${SLURM_JOB_ID}/node-$SLURM_PROCID"
fi
export FLASHINFER_WORKSPACE_BASE="$compile_cache_dir/flashinfer"
export DG_JIT_CACHE_DIR="$compile_cache_dir/deep-gemm"
export TORCHINDUCTOR_CACHE_DIR="$compile_cache_dir/torchinductor"
# Bound per-rank compilation parallelism so a site can size scheduler CPU
# allocations without inheriting an unbounded framework default.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$DG_JIT_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$ARTIFACT_DIR/raw"
mkdir -p "$ARTIFACT_DIR/raw"

OBSERVATION_SOURCE="${OBSERVATION_SOURCE:-recorder}"
if [[ "$OBSERVATION_SOURCE" == "response_routed_experts" ]]; then
    if [[ "${ENABLE_RETURN_ROUTED_EXPERTS:-0}" != "1" ]]; then
        echo "response_routed_experts observation requires ENABLE_RETURN_ROUTED_EXPERTS=1" >&2
        exit 12
    fi
    python3 /campaign/patch_sglang_hash_topk_capturer.py \
        --report "$ARTIFACT_DIR/hash-topk-capturer-patch-rank-$SLURM_PROCID.json"
fi

if [[ "${FLASHINFER_REPLAY_RECORDER:-0}" == "1" ]]; then
    python3 /campaign/patch_sglang_flashinfer_replay_recorder.py \
        --report "$ARTIFACT_DIR/flashinfer-replay-patch-rank-$SLURM_PROCID.json"
fi

MODEL_PATH="$(python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id=os.environ["MODEL_ID"],
        revision=os.environ["MODEL_REVISION"],
        cache_dir="/hfcache",
        token=False,
        local_files_only=True,
    )
)
PY
)"
test -f "$MODEL_PATH/config.json"

SERVER_ARGS=(
    --model-path "$MODEL_PATH"
    --tp-size "$TP_SIZE"
    --nnodes "$NNODES"
    --node-rank "$SLURM_PROCID"
    --dist-init-addr "$DIST_INIT_ADDR"
    --trust-remote-code
    --host 0.0.0.0
    --port "$SERVER_PORT"
    --context-length "${CONTEXT_LENGTH:-4352}"
    --max-running-requests 4
    --max-total-tokens "${MAX_TOTAL_TOKENS:-8192}"
    --max-prefill-tokens "${MAX_PREFILL_TOKENS:-4096}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-4096}"
    --mem-fraction-static 0.85
    --watchdog-timeout 3600
    --moe-runner-backend "${MOE_RUNNER_BACKEND:-flashinfer_trtllm_routed}"
    --disable-cuda-graph
    --disable-overlap-schedule
    --disable-shared-experts-fusion
    --disable-radix-cache
)

if [[ "$OBSERVATION_SOURCE" == "recorder" ]]; then
    SERVER_ARGS+=(
        --expert-distribution-recorder-mode stat
        --expert-distribution-recorder-buffer-size -1
    )
elif [[ "$OBSERVATION_SOURCE" == "response_routed_experts" ]]; then
    SERVER_ARGS+=(--enable-return-routed-experts)
else
    echo "unsupported OBSERVATION_SOURCE=$OBSERVATION_SOURCE" >&2
    exit 13
fi

if [[ "${ENABLE_DETERMINISTIC_INFERENCE:-0}" == "1" ]]; then
    SERVER_ARGS+=(--enable-deterministic-inference)
fi
if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
    SERVER_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi

if [[ "$SLURM_PROCID" -eq 0 ]]; then
    printf '%s\n' "${SERVER_ARGS[@]}" \
        | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))' \
        > "$ARTIFACT_DIR/server_args.json"
    printf '%s\n' "$MODEL_PATH" > "$ARTIFACT_DIR/model_path.txt"
    python3 - "$ARTIFACT_DIR/runtime_environment.json" <<'PY'
import json
import os
import sys

names = (
    "FLASHINFER_WORKSPACE_BASE",
    "DG_JIT_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "COMPILE_CACHE_ROOT",
    "COMPILE_CACHE_KEY",
    "SGLANG_DSV4_FP4_EXPERTS",
    "SGLANG_OPT_FP8_WO_A_GEMM",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
    "FLASHINFER_REPLAY_RECORDER",
    "ENABLE_RETURN_ROUTED_EXPERTS",
    "OBSERVATION_SOURCE",
)
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({name: os.environ[name] for name in names if name in os.environ}, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
fi

exec python3 -m sglang.launch_server "${SERVER_ARGS[@]}"
