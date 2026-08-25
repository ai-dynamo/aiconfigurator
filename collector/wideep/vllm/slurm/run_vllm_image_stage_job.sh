#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

safe_existing_path() {
    local label=$1
    local raw=$2
    local resolved
    resolved=$(realpath -e -- "${raw}") || die "${label} does not exist: ${raw}"
    case "${resolved}" in
        /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "${label} uses prohibited storage: ${resolved}" ;;
    esac
    printf '%s\n' "${resolved}"
}

for required_name in SYSTEM CAMPAIGN_ROOT IMAGE_DIGEST IMAGE_ARCH CONTAINER_IMAGE SLURM_JOB_ID; do
    [[ -n "${!required_name:-}" ]] || die "missing ${required_name}"
done
case "${IMAGE_ARCH}" in arm64|amd64) ;; *) die "bad image architecture ${IMAGE_ARCH}" ;; esac
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid image digest"
[[ "${CONTAINER_IMAGE}" == *@"${IMAGE_DIGEST}" ]] || die "container ref is not digest pinned"

campaign_root=$(safe_existing_path "campaign root" "${CAMPAIGN_ROOT}")
digest_value=${IMAGE_DIGEST#sha256:}
image_dir="${campaign_root}/images"
job_dir="${image_dir}/${SYSTEM}/job_${SLURM_JOB_ID}"
mkdir -p "${job_dir}"
job_dir=$(safe_existing_path "image staging evidence" "${job_dir}")
final_image="${image_dir}/vllm_v024_${IMAGE_ARCH}_${digest_value}.sqsh"
[[ ! -e "${final_image}" ]] || die "refusing to overwrite staged image ${final_image}"
temporary_image="${image_dir}/.vllm_v024_${IMAGE_ARCH}_${digest_value}.sqsh.tmp.${SLURM_JOB_ID}"
[[ ! -e "${temporary_image}" ]] || die "stale temporary image ${temporary_image}"

export ENROOT_CACHE_PATH="/tmp/aic-enroot-image-stage-${SLURM_JOB_ID}"
export ENROOT_MAX_CONNECTIONS=1
export ENROOT_TRANSFER_RETRIES=8
mkdir -p "${ENROOT_CACHE_PATH}"
safe_existing_path "image layer cache" "${ENROOT_CACHE_PATH}" >/dev/null

export AIC_IMAGE_STAGE_EVIDENCE="${job_dir}/runtime.json"
srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-save="${temporary_image}" \
    --container-mounts="${job_dir}:${job_dir}" \
    bash -lc 'python3 - "${AIC_IMAGE_STAGE_EVIDENCE}" <<'"'"'PY'"'"'
import json
import sys
from importlib.metadata import version
from pathlib import Path

import deep_ep
import torch

assert hasattr(deep_ep, "Buffer")
assert hasattr(deep_ep, "ElasticBuffer")
Path(sys.argv[1]).write_text(json.dumps({
    "vllm": version("vllm"),
    "torch": version("torch"),
    "cuda": torch.version.cuda,
    "deep_ep_import": str(Path(deep_ep.__file__).resolve()),
}, indent=2, sort_keys=True) + "\n")
PY'

temporary_image=$(safe_existing_path "temporary staged image" "${temporary_image}")
unsquashfs -s "${temporary_image}" >/dev/null || die "saved image is not a valid squashfs"
sqsh_sha256=$(sha256sum "${temporary_image}" | awk '{print $1}')
image_meta="${final_image}.meta.json"
temporary_meta="${image_meta}.tmp.${SLURM_JOB_ID}"
[[ ! -e "${image_meta}" && ! -e "${temporary_meta}" ]] || die "refusing to overwrite staged image metadata"
python3 - "${job_dir}/image_meta.json" "${SYSTEM}" "${IMAGE_ARCH}" "${CONTAINER_IMAGE}" \
    "${IMAGE_DIGEST}" "${sqsh_sha256}" "${final_image}" "${AIC_IMAGE_STAGE_EVIDENCE}" <<'PY'
import json
import sys
from datetime import date
from pathlib import Path

output, system, arch, source_image, digest, sqsh_sha, image, runtime_path = sys.argv[1:]
runtime = json.loads(Path(runtime_path).read_text())
if runtime["vllm"] != "0.24.0":
    raise SystemExit(f"unexpected vLLM version: {runtime['vllm']}")
Path(output).write_text(json.dumps({
    "schema_version": 1,
    "system": system,
    "architecture": arch,
    "source_image": source_image,
    "source_image_digest": digest,
    "sqsh_sha256": sqsh_sha,
    "image": image,
    "runtime": runtime,
    "staged_at": date.today().isoformat(),
}, indent=2, sort_keys=True) + "\n")
PY
cp "${job_dir}/image_meta.json" "${temporary_meta}"
mv "${temporary_image}" "${final_image}"
mv "${temporary_meta}" "${image_meta}"
final_image=$(safe_existing_path "published staged image" "${final_image}")
image_meta=$(safe_existing_path "published staged image metadata" "${image_meta}")
[[ "$(sha256sum "${final_image}" | awk '{print $1}')" == "${sqsh_sha256}" ]] || die "staged image checksum mismatch"
touch "${job_dir}/SUCCESS"
echo "Published ${IMAGE_DIGEST} as ${final_image} (${sqsh_sha256})"
