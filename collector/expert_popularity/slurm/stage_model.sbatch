#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=expert-pop-stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=expert-popularity-stage-%j.out
#SBATCH --error=expert-popularity-stage-%j.err

set -Eeuo pipefail

: "${MODEL_ID:?submit with MODEL_ID}"
: "${MODEL_REVISION:?submit with MODEL_REVISION}"
: "${MODEL_SLUG:?submit with MODEL_SLUG}"
: "${CAMPAIGN_ROOT:?submit with CAMPAIGN_ROOT}"
: "${IMAGE_SQSH:?submit with IMAGE_SQSH}"
: "${HF_CACHE:?submit with HF_CACHE}"

STAGE_DIR="$CAMPAIGN_ROOT/stage-$SLURM_JOB_ID-$MODEL_SLUG"
mkdir -p "$STAGE_DIR"
cp "$0" "$STAGE_DIR/job.sbatch"

srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="$IMAGE_SQSH" \
    --container-mounts="$HF_CACHE:/hfcache,$STAGE_DIR:/artifacts" \
    --container-workdir=/tmp \
    --export=ALL,HF_HOME=/hfcache,HF_HUB_CACHE=/hfcache,HUGGINGFACE_HUB_CACHE=/hfcache \
    python3 - <<'PY'
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
revision = os.environ["MODEL_REVISION"]
for attempt in range(1, 9):
    try:
        path = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir="/hfcache",
                token=False,
            )
        )
        break
    except Exception as error:
        message = str(error)
        if "429 Too Many Requests" not in message or attempt == 8:
            raise
        match = re.search(r"Retry after (\d+) seconds", message)
        retry_after = int(match.group(1)) if match else 300
        delay = retry_after + 15
        print(
            f"Hugging Face rate limit for {model_id}; retry {attempt}/8 in {delay} seconds",
            flush=True,
        )
        time.sleep(delay)
config = path / "config.json"
if not config.is_file():
    raise RuntimeError(f"missing {config}")
resolved = path.resolve()
if resolved.name != revision:
    raise RuntimeError(f"snapshot resolved to {resolved.name}, expected {revision}")
files = sorted(item for item in path.rglob("*") if item.is_file())
payload = {
    "status": "PASS",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_id": model_id,
    "model_revision": revision,
    "snapshot_path": str(path),
    "resolved_snapshot_path": str(resolved),
    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "file_count": len(files),
    "logical_size_bytes": sum(item.stat().st_size for item in files),
}
Path("/artifacts/stage_result.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY

echo "[$(date -u +%FT%TZ)] MODEL_STAGE_PASS model=$MODEL_ID revision=$MODEL_REVISION"
