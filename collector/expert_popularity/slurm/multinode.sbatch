#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=expert-pop-multinode
#SBATCH --time=06:00:00
#SBATCH --output=expert-popularity-%j.out
#SBATCH --error=expert-popularity-%j.err

set -Eeuo pipefail

: "${MODEL_ID:?submit with MODEL_ID}"
: "${MODEL_REVISION:?submit with MODEL_REVISION}"
: "${MODEL_SLUG:?submit with MODEL_SLUG}"
: "${NUM_LAYERS:?submit with NUM_LAYERS}"
: "${NUM_EXPERTS:?submit with NUM_EXPERTS}"
: "${TOP_K:?submit with TOP_K}"
: "${MOE_LAYER_IDS:?submit with MOE_LAYER_IDS}"
: "${CAMPAIGN_ROOT:?submit with CAMPAIGN_ROOT}"
: "${IMAGE_SQSH:?submit with IMAGE_SQSH}"
: "${IMAGE_REFERENCE:?submit with immutable IMAGE_REFERENCE}"
: "${HF_CACHE:?submit with HF_CACHE}"
: "${GPUS_PER_NODE:?submit with GPUS_PER_NODE}"

if [[ ! "$IMAGE_REFERENCE" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "IMAGE_REFERENCE must contain an immutable manifest digest" >&2
    exit 17
fi

CANONICAL_MODEL_ID="${CANONICAL_MODEL_ID:-$MODEL_ID}"
CANONICAL_MODEL_REVISION="${CANONICAL_MODEL_REVISION:-$MODEL_REVISION}"

SHARED_CODE="${CAMPAIGN_CODE_SOURCE:-$CAMPAIGN_ROOT/code}"
COMPILE_CACHE_HOST_DIR="${COMPILE_CACHE_HOST_DIR:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-4}}"
RUN_DIR="$CAMPAIGN_ROOT/$SLURM_JOB_ID-$MODEL_SLUG"
SERVER_PORT="${SERVER_PORT_OVERRIDE:-$((20000 + SLURM_JOB_ID % 10000))}"
DIST_PORT="${DIST_PORT_OVERRIDE:-$((40000 + SLURM_JOB_ID % 10000))}"
NNODES="$SLURM_NNODES"
if [[ ! "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "GPUS_PER_NODE must be a positive integer" >&2
    exit 18
fi
TP_SIZE=$((NNODES * GPUS_PER_NODE))

CAMPAIGN_CODE="$RUN_DIR/campaign"
mkdir -p "$CAMPAIGN_CODE"
cp "$0" "$RUN_DIR/job.sbatch"
cp \
    "$SHARED_CODE/sglang_driver.py" \
    "$SHARED_CODE/model_config.py" \
    "$SHARED_CODE/response_capture.py" \
    "$SHARED_CODE/patch_sglang_hash_topk_capturer.py" \
    "$SHARED_CODE/patch_sglang_flashinfer_replay_recorder.py" \
    "$SHARED_CODE/run_sglang_multinode_node.sh" \
    "$CAMPAIGN_CODE/"
test -s "$IMAGE_SQSH"
test -d "$HF_CACHE"
CONTAINER_MOUNTS="$HF_CACHE:/hfcache,$CAMPAIGN_CODE:/campaign,$RUN_DIR:/artifacts"
if [[ -n "$COMPILE_CACHE_HOST_DIR" ]]; then
    mkdir -p "$COMPILE_CACHE_HOST_DIR"
    CONTAINER_MOUNTS+=",$COMPILE_CACHE_HOST_DIR:/compile-cache"
    export COMPILE_CACHE_ROOT=/compile-cache
fi
IMAGE_SHA256="$(sha256sum "$IMAGE_SQSH" | awk '{print $1}')"
COLLECTOR_CODE_SHA256="$(sha256sum "$CAMPAIGN_CODE"/* | sort -k2 | sha256sum | awk '{print $1}')"
HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
DIST_INIT_ADDR="$HEAD_NODE:$DIST_PORT"
BASE_URL="http://$HEAD_NODE:$SERVER_PORT"
export IMAGE_REFERENCE IMAGE_SHA256 COLLECTOR_CODE_SHA256 HEAD_NODE DIST_INIT_ADDR BASE_URL SERVER_PORT NNODES TP_SIZE

job_status=FAIL
server_step_pid=
finalize_job() {
    local rc=$?
    if [[ -n "$server_step_pid" ]] && kill -0 "$server_step_pid" 2>/dev/null; then
        kill "$server_step_pid" 2>/dev/null || true
        wait "$server_step_pid" 2>/dev/null || true
    fi
    if [[ $rc -eq 0 ]]; then
        job_status=PASS
    fi
    python3 - "$RUN_DIR/job_status.json" "$job_status" "$rc" <<'PY' || true
import json
import sys
from datetime import datetime, timezone

path, status, exit_code = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        {"status": status, "exit_code": int(exit_code), "finished_at": datetime.now(timezone.utc).isoformat()},
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
}
trap finalize_job EXIT

srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
    nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv,noheader \
    > "$RUN_DIR/gpu_info.csv"
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
    bash -lc 'echo "$(hostname) $(cat /proc/sys/kernel/random/boot_id)"' \
    > "$RUN_DIR/node_info.txt"

export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0

srun \
    --nodes="$NNODES" \
    --ntasks="$NNODES" \
    --ntasks-per-node=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --container-image="$IMAGE_SQSH" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir=/tmp \
    --export=ALL,ARTIFACT_DIR=/artifacts,HF_HOME=/hfcache \
    --output="$RUN_DIR/server-%N.out" \
    --error="$RUN_DIR/server-%N.err" \
    bash /campaign/run_sglang_multinode_node.sh &
server_step_pid=$!

ready=0
for attempt in $(seq 1 540); do
    if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
        ready=1
        echo "[$(date -u +%FT%TZ)] server healthy after $((attempt * 10)) seconds"
        break
    fi
    if ! kill -0 "$server_step_pid" 2>/dev/null; then
        tail -120 "$RUN_DIR"/server-*.err >&2 || true
        tail -120 "$RUN_DIR"/server-*.out >&2 || true
        exit 20
    fi
    sleep 10
done
if [[ "$ready" -ne 1 ]]; then
    tail -120 "$RUN_DIR"/server-*.err >&2 || true
    tail -120 "$RUN_DIR"/server-*.out >&2 || true
    exit 21
fi

curl -fsS "$BASE_URL/server_info" > "$RUN_DIR/server_info.json"
SERVER_ARGS_JSON="$(cat "$RUN_DIR/server_args.json")"
RUNTIME_ENVIRONMENT_JSON="$(cat "$RUN_DIR/runtime_environment.json")"
MODEL_PATH="$(cat "$RUN_DIR/model_path.txt")"
ROUTING_EQUIVALENCE_EVIDENCE_JSON="{}"
if [[ -n "${ROUTING_EQUIVALENCE_EVIDENCE_B64:-}" && -n "${ROUTING_EQUIVALENCE_EVIDENCE_PATH:-}" ]]; then
    echo "set only one of ROUTING_EQUIVALENCE_EVIDENCE_B64 or ROUTING_EQUIVALENCE_EVIDENCE_PATH" >&2
    exit 22
elif [[ -n "${ROUTING_EQUIVALENCE_EVIDENCE_B64:-}" ]]; then
    ROUTING_EQUIVALENCE_EVIDENCE_JSON="$(printf '%s' "$ROUTING_EQUIVALENCE_EVIDENCE_B64" | base64 --decode)"
elif [[ -n "${ROUTING_EQUIVALENCE_EVIDENCE_PATH:-}" ]]; then
    test -s "$ROUTING_EQUIVALENCE_EVIDENCE_PATH"
    ROUTING_EQUIVALENCE_EVIDENCE_JSON="$(cat "$ROUTING_EQUIVALENCE_EVIDENCE_PATH")"
fi
export SERVER_ARGS_JSON RUNTIME_ENVIRONMENT_JSON MODEL_PATH ROUTING_EQUIVALENCE_EVIDENCE_JSON

srun \
    --overlap \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="$HEAD_NODE" \
    --cpus-per-task=4 \
    --container-image="$IMAGE_SQSH" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir=/tmp \
    --export=ALL,ARTIFACT_DIR=/artifacts,HF_HOME=/hfcache \
    python3 /campaign/sglang_driver.py \
        --artifact-dir /artifacts \
        --base-url "$BASE_URL" \
        --model-id "$CANONICAL_MODEL_ID" \
        --model-revision "$CANONICAL_MODEL_REVISION" \
        --checkpoint-model-id "$MODEL_ID" \
        --checkpoint-model-revision "$MODEL_REVISION" \
        --tokenizer-path "$MODEL_PATH" \
        --num-layers "$NUM_LAYERS" \
        --num-experts "$NUM_EXPERTS" \
        --top-k "$TOP_K" \
        --moe-layer-ids "$MOE_LAYER_IDS" \
        --replication-factor "$TP_SIZE" \
        --observation-source "${OBSERVATION_SOURCE:-recorder}" \
        --expected-framework-version 0.5.14 \
        --image-reference "$IMAGE_REFERENCE" \
        --image-sha256 "$IMAGE_SHA256" \
        --collector-code-sha256 "$COLLECTOR_CODE_SHA256" \
        --server-args-json "$SERVER_ARGS_JSON" \
        --runtime-environment-json "$RUNTIME_ENVIRONMENT_JSON" \
        --routing-observation-method "${ROUTING_OBSERVATION_METHOD:-sglang_explicit_topk_flashinfer_trtllm_routed}" \
        --routing-equivalence-evidence-json "$ROUTING_EQUIVALENCE_EVIDENCE_JSON" \
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

python3 - "$RUN_DIR/collection_result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if result.get("status") != "PASS":
    raise SystemExit(f"collection did not pass: {result}")
print("EXPERT_POPULARITY_MULTINODE_COLLECTION_PASS")
PY

kill "$server_step_pid" 2>/dev/null || true
wait "$server_step_pid" 2>/dev/null || true
server_step_pid=
echo "[$(date -u +%FT%TZ)] PASS artifacts=$RUN_DIR"
