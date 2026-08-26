#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

: "${ARTIFACT_DIR:?}"
: "${MODEL_ID:?}"
: "${MODEL_REVISION:?}"
: "${NUM_LAYERS:?}"
: "${NUM_EXPERTS:?}"
: "${TOP_K:?}"
: "${MOE_LAYER_IDS:?}"
: "${IMAGE_REFERENCE:?}"
: "${IMAGE_SHA256:?}"
: "${COLLECTOR_CODE_SHA256:?}"
: "${SERVER_PORT:?}"

export HF_HOME=/hfcache
export HF_HUB_ENABLE_HXET=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export CUDA_VISIBLE_DEVICES=0
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$ARTIFACT_DIR/raw"
mkdir -p "$ARTIFACT_DIR/raw"

OBSERVATION_SOURCE="${OBSERVATION_SOURCE:-recorder}"
if [[ "$OBSERVATION_SOURCE" == "response_routed_experts" ]]; then
    if [[ "${ENABLE_RETURN_ROUTED_EXPERTS:-0}" != "1" ]]; then
        echo "response_routed_experts observation requires ENABLE_RETURN_ROUTED_EXPERTS=1" >&2
        exit 12
    fi
    python3 /campaign/patch_sglang_hash_topk_capturer.py \
        --report "$ARTIFACT_DIR/hash-topk-capturer-patch.json"
fi

if [[ "${FLASHINFER_RECORDER_BRIDGE:-0}" == "1" ]]; then
    python3 /campaign/patch_sglang_flashinfer_recorder.py \
        --report "$ARTIFACT_DIR/flashinfer_recorder_bridge.json"
fi

MODEL_PATH="$(python3 - 2>"$ARTIFACT_DIR/model_download.log" <<'PY'
import os
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id=os.environ["MODEL_ID"],
        revision=os.environ["MODEL_REVISION"],
        cache_dir="/hfcache",
        token=False,
    )
)
PY
)"
export MODEL_PATH
test -f "$MODEL_PATH/config.json"

SERVER_ARGS=(
    --model-path "$MODEL_PATH"
    --tp-size 1
    --trust-remote-code
    --host 127.0.0.1
    --port "$SERVER_PORT"
    --context-length 4352
    --max-total-tokens 8192
    --max-prefill-tokens 4096
    --chunked-prefill-size 4096
    --mem-fraction-static 0.70
    --watchdog-timeout 1800
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
if [[ -n "${MOE_RUNNER_BACKEND:-}" ]]; then
    SERVER_ARGS+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
fi
SERVER_ARGS_JSON="$(printf '%s\n' "${SERVER_ARGS[@]}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))')"
export SERVER_ARGS_JSON

python3 -m sglang.launch_server "${SERVER_ARGS[@]}" >"$ARTIFACT_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup_server() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || return 0
            sleep 1
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT

ready=0
for attempt in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1; then
        ready=1
        echo "[$(date -u +%FT%TZ)] server healthy after $((attempt * 5)) seconds"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        tail -200 "$ARTIFACT_DIR/server.log" >&2 || true
        exit 20
    fi
    sleep 5
done
if [[ "$ready" -ne 1 ]]; then
    tail -200 "$ARTIFACT_DIR/server.log" >&2 || true
    exit 21
fi

curl -fsS "http://127.0.0.1:$SERVER_PORT/server_info" >"$ARTIFACT_DIR/server_info.json"
nvidia-smi -L >"$ARTIFACT_DIR/nvidia_smi_L.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv,noheader \
    >"$ARTIFACT_DIR/gpu_info.csv"

python3 /campaign/sglang_driver.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --base-url "http://127.0.0.1:$SERVER_PORT" \
    --model-id "$MODEL_ID" \
    --model-revision "$MODEL_REVISION" \
    --tokenizer-path "$MODEL_PATH" \
    --num-layers "$NUM_LAYERS" \
    --num-experts "$NUM_EXPERTS" \
    --top-k "$TOP_K" \
    --moe-layer-ids "$MOE_LAYER_IDS" \
    --replication-factor 1 \
    --observation-source "$OBSERVATION_SOURCE" \
    --expected-framework-version 0.5.14 \
    --image-reference "$IMAGE_REFERENCE" \
    --image-sha256 "$IMAGE_SHA256" \
    --collector-code-sha256 "$COLLECTOR_CODE_SHA256" \
    --server-args-json "$SERVER_ARGS_JSON" \
    --routing-observation-method "${ROUTING_OBSERVATION_METHOD:-sglang_standard_topk}" \
    --tokens-per-shard "${TOKENS_PER_SHARD:-65536}" \
    --isl-min "${ISL_MIN:-128}" \
    --isl-max "${ISL_MAX:-4096}" \
    --repeat-count "${REPEAT_COUNT:-2}" \
    --request-timeout "${REQUEST_TIMEOUT:-600}" \
    --min-shard-pearson "${MIN_SHARD_PEARSON:-0.95}" \
    --max-shard-jsd "${MAX_SHARD_JSD:-0.01}" \
    --repeat-validation-mode "${REPEAT_VALIDATION_MODE:-exact}" \
    --min-repeat-pearson "${MIN_REPEAT_PEARSON:-0.999}" \
    --max-repeat-jsd "${MAX_REPEAT_JSD:-0.001}"

cleanup_server
trap - EXIT
