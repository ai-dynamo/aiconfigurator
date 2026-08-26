#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=expert-popularity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
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

if [[ ! "$IMAGE_REFERENCE" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "IMAGE_REFERENCE must contain an immutable manifest digest" >&2
    exit 18
fi

SHARED_CODE="${CAMPAIGN_CODE_SOURCE:-$CAMPAIGN_ROOT/code}"
RUN_DIR="$CAMPAIGN_ROOT/$SLURM_JOB_ID-$MODEL_SLUG"
SERVER_PORT="${SERVER_PORT_OVERRIDE:-$((20000 + SLURM_JOB_ID % 10000))}"

CAMPAIGN_CODE="$RUN_DIR/campaign"
mkdir -p "$CAMPAIGN_CODE"
cp "$0" "$RUN_DIR/job.sbatch"
cp \
    "$SHARED_CODE/sglang_driver.py" \
    "$SHARED_CODE/model_config.py" \
    "$SHARED_CODE/response_capture.py" \
    "$SHARED_CODE/patch_sglang_hash_topk_capturer.py" \
    "$SHARED_CODE/patch_sglang_flashinfer_recorder.py" \
    "$SHARED_CODE/run_sglang_container.sh" \
    "$CAMPAIGN_CODE/"
test -s "$IMAGE_SQSH"
test -d "$HF_CACHE"
IMAGE_SHA256="$(sha256sum "$IMAGE_SQSH" | awk '{print $1}')"
COLLECTOR_CODE_SHA256="$(sha256sum "$CAMPAIGN_CODE"/* | sort -k2 | sha256sum | awk '{print $1}')"
export IMAGE_REFERENCE IMAGE_SHA256 COLLECTOR_CODE_SHA256 SERVER_PORT

job_status=FAIL
finalize_job() {
    local rc=$?
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
        {
            "status": status,
            "exit_code": int(exit_code),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
}
trap finalize_job EXIT

echo "[$(date -u +%FT%TZ)] job=$SLURM_JOB_ID node=$SLURMD_NODENAME model=$MODEL_ID"

srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="$IMAGE_SQSH" \
    --container-mounts="$HF_CACHE:/hfcache,$CAMPAIGN_CODE:/campaign,$RUN_DIR:/artifacts" \
    --container-workdir=/tmp \
    --export=ALL,ARTIFACT_DIR=/artifacts,HF_HOME=/hfcache \
    bash /campaign/run_sglang_container.sh

python3 - "$RUN_DIR/collection_result.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if result.get("status") != "PASS":
    raise SystemExit(f"collection did not pass: {result}")
print("EXPERT_POPULARITY_COLLECTION_PASS")
PY

echo "[$(date -u +%FT%TZ)] PASS artifacts=$RUN_DIR"
