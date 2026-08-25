#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
VLLM_COMMIT=ee0da84ab9e04ac7610e28580af62c365e898389
NVSHMEM_VERSION=3.3.24
CUDA_ARCHES='10.0a 10.3a'

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_env() {
    local name=$1
    [[ -n "${!name:-}" ]] || die "required environment variable ${name} is empty"
}

safe_existing_path() {
    local label=$1
    local raw=$2
    local resolved
    resolved=$(realpath -e -- "${raw}") || die "${label} does not exist: ${raw}"
    case "${resolved}" in
        /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "${label} uses prohibited shared storage: ${resolved}" ;;
    esac
    printf '%s\n' "${resolved}"
}

require_env SYSTEM
require_env CAMPAIGN_ROOT
require_env VLLM_SOURCE_ROOT
require_env CONTAINER_IMAGE
require_env IMAGE_DIGEST
require_env SLURM_JOB_ID

case "${SYSTEM}" in gb300|b300_sxm) ;; *) die "unsupported SM103 system ${SYSTEM}" ;; esac
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid image digest"
[[ "${CONTAINER_IMAGE}" == *@"${IMAGE_DIGEST}" ]] || die "container image does not use IMAGE_DIGEST"

campaign_root=$(safe_existing_path "campaign root" "${CAMPAIGN_ROOT}")
vllm_source_root=$(safe_existing_path "vLLM source" "${VLLM_SOURCE_ROOT}")
[[ -z "$(git -C "${vllm_source_root}" status --porcelain)" ]] || die "vLLM source checkout is dirty"
[[ "$(git -C "${vllm_source_root}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] || die "vLLM source commit mismatch"

staging_root="/tmp/aic-deepep-sm103-${SLURM_JOB_ID}"
[[ "${staging_root}" == /tmp/aic-deepep-sm103-* ]] || die "unsafe staging root"
mkdir -p "${staging_root}"
staging_root=$(safe_existing_path "overlay staging" "${staging_root}")
export ENROOT_CACHE_PATH="/tmp/aic-enroot-overlay-cache-${SLURM_JOB_ID}"
mkdir -p "${ENROOT_CACHE_PATH}"
safe_existing_path "container cache" "${ENROOT_CACHE_PATH}" >/dev/null

export AIC_OVERLAY_STAGING="${staging_root}"
export AIC_VLLM_SOURCE_ROOT="${vllm_source_root}"
export AIC_DEEPEP_COMMIT="${DEEPEP_COMMIT}"
export AIC_NVSHMEM_VERSION="${NVSHMEM_VERSION}"
export AIC_CUDA_ARCHES="${CUDA_ARCHES}"

srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${vllm_source_root}:${vllm_source_root},${staging_root}:${staging_root}" \
    bash -lc '
        set -euo pipefail
        export TORCH_CUDA_ARCH_LIST="${AIC_CUDA_ARCHES}"
        "${AIC_VLLM_SOURCE_ROOT}/tools/ep_kernels/install_python_libraries.sh" \
            --workspace "${AIC_OVERLAY_STAGING}/workspace" \
            --mode wheel \
            --deepep-ref "${AIC_DEEPEP_COMMIT}" \
            --nvshmem-ver "${AIC_NVSHMEM_VERSION}"
        mapfile -t wheels < <(find "${AIC_OVERLAY_STAGING}/workspace/dist" -maxdepth 1 -type f -name "*.whl" -print)
        [[ "${#wheels[@]}" == 1 ]]
        python3 -m pip install --no-deps --target "${AIC_OVERLAY_STAGING}/import-test" "${wheels[0]}" >/dev/null
        PYTHONPATH="${AIC_OVERLAY_STAGING}/import-test" python3 -c \
            "import deep_ep; assert hasattr(deep_ep, \"Buffer\"); assert hasattr(deep_ep, \"ElasticBuffer\")"
        basename "${wheels[0]}" > "${AIC_OVERLAY_STAGING}/wheel_name.txt"
    '

wheel_name_path=$(safe_existing_path "SM103 wheel-name marker" "${staging_root}/wheel_name.txt")
wheel_name=$(<"${wheel_name_path}")
[[ -n "${wheel_name}" && "${wheel_name}" == "$(basename -- "${wheel_name}")" && "${wheel_name}" == *.whl ]] || die \
    "unsafe wheel name ${wheel_name}"
wheel_path=$(safe_existing_path "SM103 overlay wheel" "${staging_root}/workspace/dist/${wheel_name}")
wheel_sha256=$(sha256sum "${wheel_path}" | awk '{print $1}')
gpu_identity=$(srun --nodes=1 --ntasks=1 nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader,nounits | head -n 1)

publish_dir="${campaign_root}/overlays/${SYSTEM}/job_${SLURM_JOB_ID}"
mkdir -p "${publish_dir}"
publish_dir=$(safe_existing_path "overlay publish directory" "${publish_dir}")
cp "${wheel_path}" "${publish_dir}/${wheel_name}"
python3 - "${publish_dir}/build_meta.json" "${wheel_name}" "${wheel_sha256}" "${gpu_identity}" \
    "${SYSTEM}" "${IMAGE_DIGEST}" "${VLLM_COMMIT}" "${DEEPEP_COMMIT}" \
    "${NVSHMEM_VERSION}" "${CUDA_ARCHES}" <<'PY'
import json
import sys
from datetime import date
from pathlib import Path

output, wheel_name, wheel_sha, gpu, system, image, vllm, deepep, nvshmem, arches = sys.argv[1:]
payload = {
    "schema_version": 1,
    "system": system,
    "image_digest": image,
    "vllm_source_commit": vllm,
    "deep_ep_source_commit": deepep,
    "nvshmem": nvshmem,
    "cuda_arches": arches,
    "wheel": wheel_name,
    "wheel_sha256": wheel_sha,
    "build_gpu": gpu,
    "built_at": date.today().isoformat(),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
[[ "$(sha256sum "${publish_dir}/${wheel_name}" | awk '{print $1}')" == "${wheel_sha256}" ]] || die \
    "published wheel checksum mismatch"
touch "${publish_dir}/SUCCESS"
echo "Published ${SYSTEM} SM103 overlay ${wheel_sha256} to ${publish_dir}"
