#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build one immutable DeepEP runtime overlay. Despite the historical filename,
# this runner now covers legacy 8-rank, patched legacy 4-rank, and EPv2 ABIs.

set -euo pipefail

LEGACY_DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c
V2_DEEPEP_COMMIT=b306af06afd412c88e51e71802951606e40b7358
VLLM_COMMIT=ee0da84ab9e04ac7610e28580af62c365e898389
NVSHMEM_VERSION=3.3.24
V2_NCCL_VERSION=2.30.4

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
require_env OVERLAY_PROFILE
require_env CUDA_ARCHES
require_env CAMPAIGN_ROOT
require_env REPO_DIR
require_env VLLM_SOURCE_ROOT
require_env CONTAINER_IMAGE
require_env IMAGE_DIGEST
require_env SLURM_JOB_ID

case "${SYSTEM}" in gb200|gb300|b200_sxm|b300_sxm|h100_sxm|h200_sxm) ;; *) die "unsupported system ${SYSTEM}" ;; esac
case "${OVERLAY_PROFILE}" in
    legacy-nvl8)
        deepep_commit=${LEGACY_DEEPEP_COMMIT}
        expected_api=Buffer
        scaleup_ranks=8
        ;;
    legacy-nvl4)
        deepep_commit=${LEGACY_DEEPEP_COMMIT}
        expected_api=Buffer
        scaleup_ranks=4
        ;;
    v2)
        deepep_commit=${V2_DEEPEP_COMMIT}
        expected_api=ElasticBuffer
        scaleup_ranks=auto
        ;;
    *) die "unsupported overlay profile ${OVERLAY_PROFILE}" ;;
esac

[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid image digest"
if [[ "${CONTAINER_IMAGE}" == /* ]]; then
    container_image=$(safe_existing_path "container image" "${CONTAINER_IMAGE}")
    unsquashfs -s "${container_image}" >/dev/null || die "container image is not a valid squashfs"
    container_image_meta=$(safe_existing_path "container image metadata" "${container_image}.meta.json")
    python3 - "${container_image}" "${container_image_meta}" "${IMAGE_DIGEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

image, metadata, expected_digest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
payload = json.loads(metadata.read_text())
if payload.get("source_image_digest") != expected_digest:
    raise SystemExit("local container source digest mismatch")
observed = hashlib.sha256()
with image.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        observed.update(chunk)
if payload.get("sqsh_sha256") != observed.hexdigest():
    raise SystemExit("local container squashfs checksum mismatch")
PY
else
    container_image=${CONTAINER_IMAGE}
    [[ "${container_image}" == *@"${IMAGE_DIGEST}" ]] || die "container image does not use IMAGE_DIGEST"
fi

campaign_root=$(safe_existing_path "campaign root" "${CAMPAIGN_ROOT}")
repo_dir=$(safe_existing_path "repository" "${REPO_DIR}")
vllm_source_root=$(safe_existing_path "vLLM source" "${VLLM_SOURCE_ROOT}")
[[ -z "$(git -C "${repo_dir}" status --porcelain)" ]] || die "repository checkout is dirty"
[[ -z "$(git -C "${vllm_source_root}" status --porcelain)" ]] || die "vLLM source checkout is dirty"
[[ "$(git -C "${vllm_source_root}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] || die "vLLM source commit mismatch"

legacy_patch=""
legacy_patch_sha256=""
if [[ "${OVERLAY_PROFILE}" == legacy-nvl4 ]]; then
    legacy_patch=$(safe_existing_path \
        "legacy four-rank topology patch" \
        "${repo_dir}/collector/wideep/vllm/patches/deepep_73b_nvl4.patch")
    legacy_patch_sha256=$(sha256sum "${legacy_patch}" | awk '{print $1}')
fi

staging_root="/tmp/aic-deepep-overlay-${SLURM_JOB_ID}"
[[ "${staging_root}" == /tmp/aic-deepep-overlay-* ]] || die "unsafe staging root"
mkdir -p "${staging_root}"
staging_root=$(safe_existing_path "overlay staging" "${staging_root}")
export ENROOT_CACHE_PATH="/tmp/aic-enroot-overlay-cache-${SLURM_JOB_ID}"
mkdir -p "${ENROOT_CACHE_PATH}"
safe_existing_path "container cache" "${ENROOT_CACHE_PATH}" >/dev/null

export AIC_OVERLAY_STAGING="${staging_root}"
export AIC_REPO_DIR="${repo_dir}"
export AIC_VLLM_SOURCE_ROOT="${vllm_source_root}"
export AIC_DEEPEP_COMMIT="${deepep_commit}"
export AIC_OVERLAY_PROFILE="${OVERLAY_PROFILE}"
export AIC_LEGACY_PATCH="${legacy_patch}"
export AIC_NVSHMEM_VERSION="${NVSHMEM_VERSION}"
export AIC_NCCL_VERSION="${V2_NCCL_VERSION}"
export AIC_CUDA_ARCHES="${CUDA_ARCHES}"
export AIC_EXPECTED_API="${expected_api}"

# The official vLLM child image intentionally does not ship the git CLI.
# Resolve and attest the DeepEP checkout on the host, then mount the exact
# checkout into the build container through the job-local staging directory.
workspace="${staging_root}/workspace"
mkdir -p "${workspace}"
git clone --filter=blob:none https://github.com/deepseek-ai/DeepEP "${workspace}/DeepEP"
git -C "${workspace}/DeepEP" checkout --detach "${deepep_commit}"
[[ "$(git -C "${workspace}/DeepEP" rev-parse HEAD)" == "${deepep_commit}" ]] || die "DeepEP source commit mismatch"
if [[ "${OVERLAY_PROFILE}" == legacy-nvl4 ]]; then
    git -C "${workspace}/DeepEP" apply --check --unidiff-zero "${legacy_patch}"
    git -C "${workspace}/DeepEP" apply --unidiff-zero "${legacy_patch}"
fi

srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${container_image}" \
    --container-mounts="${repo_dir}:${repo_dir},${vllm_source_root}:${vllm_source_root},${staging_root}:${staging_root}" \
    bash -lc '
        set -euo pipefail
        workspace="${AIC_OVERLAY_STAGING}/workspace"
        mkdir -p "${workspace}"
        export TORCH_CUDA_ARCH_LIST="${AIC_CUDA_ARCHES}"

        # The vLLM 0.24.0 child image carries its CUDA 13 development headers
        # alongside the pip-installed CUDA libraries instead of under
        # /usr/local/cuda/include. Keep the build in the digest-pinned runtime
        # image and explicitly expose those bundled headers to both the host
        # compiler and nvcc.
        mapfile -t bundled_cuda_includes < <(
            find /usr/local/lib/python* -type d -path "*/nvidia/cu13/include" -print
        )
        [[ "${#bundled_cuda_includes[@]}" == 1 ]]
        bundled_cuda_include="${bundled_cuda_includes[0]}"
        bundled_cuda_lib="$(dirname "${bundled_cuda_include}")/lib"
        [[ -f "${bundled_cuda_include}/cusparse.h" ]]
        [[ -f "${bundled_cuda_include}/nvrtc.h" ]]
        [[ -d "${bundled_cuda_lib}" ]]
        export CPATH="${bundled_cuda_include}:${CPATH:-}"
        export LIBRARY_PATH="${bundled_cuda_lib}:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="${bundled_cuda_lib}:${LD_LIBRARY_PATH:-}"
        printf "%s\n" "${bundled_cuda_include}" > "${AIC_OVERLAY_STAGING}/cuda_devel_root.txt"

        if [[ "${AIC_OVERLAY_PROFILE}" == v2 ]]; then
            mkdir -p "${AIC_OVERLAY_STAGING}/deps"
            python3 -m pip download --no-deps --dest "${AIC_OVERLAY_STAGING}/deps" \
                "nvidia-nccl-cu13==${AIC_NCCL_VERSION}" >/dev/null
            mapfile -t nccl_wheels < <(find "${AIC_OVERLAY_STAGING}/deps" -maxdepth 1 -type f -name "nvidia_nccl_cu13-*.whl" -print)
            [[ "${#nccl_wheels[@]}" == 1 ]]
            python3 -m pip install --no-deps --target "${AIC_OVERLAY_STAGING}/build-nccl" "${nccl_wheels[0]}" >/dev/null
            export PYTHONPATH="${AIC_OVERLAY_STAGING}/build-nccl:${PYTHONPATH:-}"
            export EP_NCCL_ROOT_DIR="${AIC_OVERLAY_STAGING}/build-nccl/nvidia/nccl"
            export LD_LIBRARY_PATH="${EP_NCCL_ROOT_DIR}/lib:${LD_LIBRARY_PATH:-}"
            basename "${nccl_wheels[0]}" > "${AIC_OVERLAY_STAGING}/nccl_wheel_name.txt"
        fi

        "${AIC_VLLM_SOURCE_ROOT}/tools/ep_kernels/install_python_libraries.sh" \
            --workspace "${workspace}" \
            --mode wheel \
            --deepep-ref "${AIC_DEEPEP_COMMIT}" \
            --nvshmem-ver "${AIC_NVSHMEM_VERSION}"
        mapfile -t deep_ep_wheels < <(find "${workspace}/dist" -maxdepth 1 -type f -name "*.whl" -print)
        [[ "${#deep_ep_wheels[@]}" == 1 ]]

        import_dir="${AIC_OVERLAY_STAGING}/import-test"
        mkdir -p "${import_dir}"
        if [[ "${AIC_OVERLAY_PROFILE}" == v2 ]]; then
            python3 -m pip install --no-deps --target "${import_dir}" "${nccl_wheels[0]}" >/dev/null
            export LD_LIBRARY_PATH="${import_dir}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
        fi
        python3 -m pip install --no-deps --target "${import_dir}" "${deep_ep_wheels[0]}" >/dev/null
        PYTHONPATH="${import_dir}" python3 - "${AIC_EXPECTED_API}" "${AIC_OVERLAY_STAGING}/runtime.json" <<'"'"'PY'"'"'
import json
import sys
from importlib.metadata import version
from pathlib import Path

import deep_ep

api, output = sys.argv[1:]
if not hasattr(deep_ep, api):
    raise SystemExit(f"DeepEP overlay is missing {api}")
payload = {
    "deep_ep": version("deep_ep"),
    "deep_ep_api": api,
    "deep_ep_import": str(Path(deep_ep.__file__).resolve()),
}
if api == "ElasticBuffer":
    payload["nccl"] = version("nvidia-nccl-cu13")
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
        basename "${deep_ep_wheels[0]}" > "${AIC_OVERLAY_STAGING}/deep_ep_wheel_name.txt"
    '

deep_ep_wheel_name=$(<"$(safe_existing_path "DeepEP wheel-name marker" "${staging_root}/deep_ep_wheel_name.txt")")
[[ -n "${deep_ep_wheel_name}" && "${deep_ep_wheel_name}" == "$(basename -- "${deep_ep_wheel_name}")" && \
   "${deep_ep_wheel_name}" == *.whl ]] || die "unsafe DeepEP wheel name ${deep_ep_wheel_name}"
deep_ep_wheel=$(safe_existing_path "DeepEP overlay wheel" "${staging_root}/workspace/dist/${deep_ep_wheel_name}")
deep_ep_wheel_sha256=$(sha256sum "${deep_ep_wheel}" | awk '{print $1}')

nccl_wheel_name=""
nccl_wheel_sha256=""
if [[ "${OVERLAY_PROFILE}" == v2 ]]; then
    nccl_wheel_name=$(<"$(safe_existing_path "NCCL wheel-name marker" "${staging_root}/nccl_wheel_name.txt")")
    [[ -n "${nccl_wheel_name}" && "${nccl_wheel_name}" == "$(basename -- "${nccl_wheel_name}")" && \
       "${nccl_wheel_name}" == *.whl ]] || die "unsafe NCCL wheel name ${nccl_wheel_name}"
    nccl_wheel=$(safe_existing_path "NCCL overlay wheel" "${staging_root}/deps/${nccl_wheel_name}")
    nccl_wheel_sha256=$(sha256sum "${nccl_wheel}" | awk '{print $1}')
fi

gpu_identity=$(srun --nodes=1 --ntasks=1 \
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader,nounits | head -n 1)
architecture=$(uname -m)
publish_dir="${campaign_root}/overlays/${architecture}/${OVERLAY_PROFILE}/job_${SLURM_JOB_ID}"
[[ ! -e "${publish_dir}" ]] || die "refusing to overwrite overlay directory ${publish_dir}"
mkdir -p "${publish_dir}"
publish_dir=$(safe_existing_path "overlay publish directory" "${publish_dir}")
cp "${deep_ep_wheel}" "${publish_dir}/${deep_ep_wheel_name}"
if [[ "${OVERLAY_PROFILE}" == v2 ]]; then
    cp "${nccl_wheel}" "${publish_dir}/${nccl_wheel_name}"
fi
cp "${staging_root}/runtime.json" "${publish_dir}/runtime.json"
cp "${staging_root}/cuda_devel_root.txt" "${publish_dir}/cuda_devel_root.txt"

python3 - "${publish_dir}/build_meta.json" "${SYSTEM}" "${architecture}" "${OVERLAY_PROFILE}" \
    "${scaleup_ranks}" "${IMAGE_DIGEST}" "${VLLM_COMMIT}" "${deepep_commit}" \
    "${legacy_patch_sha256}" "${NVSHMEM_VERSION}" "${V2_NCCL_VERSION}" "${CUDA_ARCHES}" \
    "${deep_ep_wheel_name}" "${deep_ep_wheel_sha256}" "${nccl_wheel_name}" "${nccl_wheel_sha256}" \
    "${gpu_identity}" <<'PY'
import json
import sys
from datetime import date
from pathlib import Path

(
    output, system, architecture, profile, scaleup_ranks, image, vllm, deepep,
    patch_sha, nvshmem, nccl, arches, deep_ep_wheel, deep_ep_sha,
    nccl_wheel, nccl_sha, gpu,
) = sys.argv[1:]
payload = {
    "schema_version": 2,
    "system": system,
    "architecture": architecture,
    "profile": profile,
    "deep_ep_scaleup_ranks": scaleup_ranks,
    "image_digest": image,
    "vllm_source_commit": vllm,
    "deep_ep_source_commit": deepep,
    "deep_ep_patch_sha256": patch_sha,
    "nvshmem": nvshmem,
    "nccl": nccl if profile == "v2" else "",
    "cuda_arches": arches,
    "deep_ep_wheel": deep_ep_wheel,
    "deep_ep_wheel_sha256": deep_ep_sha,
    "nccl_wheel": nccl_wheel,
    "nccl_wheel_sha256": nccl_sha,
    "build_gpu": gpu,
    "built_at": date.today().isoformat(),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

[[ "$(sha256sum "${publish_dir}/${deep_ep_wheel_name}" | awk '{print $1}')" == "${deep_ep_wheel_sha256}" ]] || \
    die "published DeepEP wheel checksum mismatch"
if [[ "${OVERLAY_PROFILE}" == v2 ]]; then
    [[ "$(sha256sum "${publish_dir}/${nccl_wheel_name}" | awk '{print $1}')" == "${nccl_wheel_sha256}" ]] || \
        die "published NCCL wheel checksum mismatch"
fi
touch "${publish_dir}/SUCCESS"
echo "Published ${OVERLAY_PROFILE} overlay ${deep_ep_wheel_sha256} to ${publish_dir}"
