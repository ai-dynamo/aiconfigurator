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
PYARROW_VERSION=24.0.0

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
require_env IMAGE_INDEX_DIGEST
require_env IMAGE_DIGEST
require_env IMAGE_VARIANT
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

[[ "${IMAGE_INDEX_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid image index digest"
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid image child digest"
case "${IMAGE_VARIANT}" in linux/amd64|linux/arm64) ;; *) die "invalid image variant" ;; esac
[[ "${CONTAINER_IMAGE}" == /* ]] || die "overlay build requires a locally staged image"
container_image=$(safe_existing_path "container image" "${CONTAINER_IMAGE}")
unsquashfs -s "${container_image}" >/dev/null || die "container image is not a valid squashfs"
container_image_meta=$(safe_existing_path "container image metadata" "${container_image}.meta.json")
safe_existing_path "container image completion marker" "${container_image}.SUCCESS" >/dev/null
python3 - "${container_image}" "${container_image_meta}" "${IMAGE_INDEX_DIGEST}" \
    "${IMAGE_DIGEST}" "${IMAGE_VARIANT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

image, metadata = map(Path, sys.argv[1:3])
expected_index, expected_child, expected_variant = sys.argv[3:]
payload = json.loads(metadata.read_text())
checks = {
    "schema_version": 2,
    "configured_image": f"vllm/vllm-openai:v0.24.0@{expected_index}",
    "configured_image_digest": expected_index,
    "observed_image_digest": expected_child,
    "image_variant": expected_variant,
}
for key, expected in checks.items():
    if payload.get(key) != expected:
        raise SystemExit(f"local container {key} mismatch")
observed = hashlib.sha256()
with image.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        observed.update(chunk)
if payload.get("sqsh_sha256") != observed.hexdigest():
    raise SystemExit("local container squashfs checksum mismatch")
PY

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
export AIC_PYARROW_VERSION="${PYARROW_VERSION}"
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

# Both supported DeepEP revisions encode build-directory library paths in the
# extension RUNPATH. Remove those source-level rpaths while retaining explicit
# link directories, then attest the exact rewritten setup.py in the overlay.
python3 - "${workspace}/DeepEP/setup.py" "${OVERLAY_PROFILE}" \
    "${staging_root}/deep_ep_setup_sha256.txt" <<'PY'
import hashlib
import sys
from pathlib import Path

path, profile, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
source = path.read_text()
legacy = ", f'-Wl,-rpath,{nvshmem_dir}/lib'"
v2_nvshmem = ", f'-Wl,-rpath,{nvshmem_root_dir}/lib'"
v2_nccl = "extra_link_args.extend([f'-l:libnccl.so', f'-Wl,-rpath,{nccl_root_dir}/lib'])"
v2_nccl_portable = "library_dirs.extend([f'{nccl_root_dir}/lib'])\n    extra_link_args.extend([f'-l:libnccl.so'])"

if profile.startswith("legacy-"):
    if source.count(legacy) != 1 or v2_nvshmem in source or v2_nccl in source:
        raise SystemExit("unexpected legacy DeepEP linker configuration")
    source = source.replace(legacy, "")
elif profile == "v2":
    if source.count(v2_nvshmem) != 1 or source.count(v2_nccl) != 1 or legacy in source:
        raise SystemExit("unexpected DeepEP V2 linker configuration")
    source = source.replace(v2_nvshmem, "").replace(v2_nccl, v2_nccl_portable)
else:
    raise SystemExit(f"unsupported overlay profile {profile}")

path.write_text(source)
output.write_text(hashlib.sha256(source.encode()).hexdigest() + "\n")
PY

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

        mapfile -t base_nvshmem_libs < <(
            find /usr/local/lib/python* -type d -path "*/nvidia/nvshmem/lib" -print
        )
        [[ "${#base_nvshmem_libs[@]}" == 1 ]]
        base_nvshmem_lib="${base_nvshmem_libs[0]}"
        [[ -f "${base_nvshmem_lib}/libnvshmem_host.so.3" ]]
        base_ld_library_path="${base_nvshmem_lib}:${LD_LIBRARY_PATH:-}"

        mkdir -p "${AIC_OVERLAY_STAGING}/deps"
        python3 -m pip download --no-deps --dest "${AIC_OVERLAY_STAGING}/deps" \
            "pyarrow==${AIC_PYARROW_VERSION}" >/dev/null
        mapfile -t pyarrow_wheels < <(
            find "${AIC_OVERLAY_STAGING}/deps" -maxdepth 1 -type f -name "pyarrow-*.whl" -print
        )
        [[ "${#pyarrow_wheels[@]}" == 1 ]]
        basename "${pyarrow_wheels[0]}" > "${AIC_OVERLAY_STAGING}/pyarrow_wheel_name.txt"

        if [[ "${AIC_OVERLAY_PROFILE}" == v2 ]]; then
            python3 -m pip download --no-deps --dest "${AIC_OVERLAY_STAGING}/deps" \
                "nvidia-nccl-cu13==${AIC_NCCL_VERSION}" >/dev/null
            mapfile -t nccl_wheels < <(find "${AIC_OVERLAY_STAGING}/deps" -maxdepth 1 -type f -name "nvidia_nccl_cu13-*.whl" -print)
            [[ "${#nccl_wheels[@]}" == 1 ]]
            python3 -m pip install --no-deps --target "${AIC_OVERLAY_STAGING}/build-nccl" "${nccl_wheels[0]}" >/dev/null
            export PYTHONPATH="${AIC_OVERLAY_STAGING}/build-nccl:${PYTHONPATH:-}"
            build_nccl_root="${AIC_OVERLAY_STAGING}/build-nccl/nvidia/nccl"
            export LD_LIBRARY_PATH="${build_nccl_root}/lib:${LD_LIBRARY_PATH:-}"
            unset EP_NCCL_ROOT_DIR
            basename "${nccl_wheels[0]}" > "${AIC_OVERLAY_STAGING}/nccl_wheel_name.txt"
        fi

        "${AIC_VLLM_SOURCE_ROOT}/tools/ep_kernels/install_python_libraries.sh" \
            --workspace "${workspace}" \
            --mode wheel \
            --deepep-ref "${AIC_DEEPEP_COMMIT}" \
            --nvshmem-ver "${AIC_NVSHMEM_VERSION}"
        mapfile -t deep_ep_wheels < <(find "${workspace}/dist" -maxdepth 1 -type f -name "*.whl" -print)
        [[ "${#deep_ep_wheels[@]}" == 1 ]]

        elf_audit_dir="${AIC_OVERLAY_STAGING}/elf-audit"
        mkdir -p "${elf_audit_dir}"
        python3 - "${deep_ep_wheels[0]}" "${elf_audit_dir}" <<'"'"'PY'"'"'
import sys
import zipfile
from pathlib import Path

wheel, output = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(wheel) as archive:
    shared_objects = [name for name in archive.namelist() if name.endswith(".so")]
    if len(shared_objects) != 1:
        raise SystemExit(f"expected one DeepEP shared object, found {shared_objects}")
    archive.extract(shared_objects[0], output)
PY
        mapfile -t deep_ep_shared_objects < <(find "${elf_audit_dir}" -type f -name "*.so" -print)
        [[ "${#deep_ep_shared_objects[@]}" == 1 ]]
        readelf -d "${deep_ep_shared_objects[0]}" > "${AIC_OVERLAY_STAGING}/elf_dynamic.txt"
        if grep -Eq "\((RPATH|RUNPATH)\)" "${AIC_OVERLAY_STAGING}/elf_dynamic.txt"; then
            echo "DeepEP wheel contains a non-portable RPATH/RUNPATH" >&2
            exit 1
        fi

        import_dir="${AIC_OVERLAY_STAGING}/import-test"
        mkdir -p "${import_dir}"
        python3 -m pip install --no-deps --target "${import_dir}" "${pyarrow_wheels[0]}" >/dev/null
        if [[ "${AIC_OVERLAY_PROFILE}" == v2 ]]; then
            python3 -m pip install --no-deps --target "${import_dir}" "${nccl_wheels[0]}" >/dev/null
            export LD_LIBRARY_PATH="${import_dir}/nvidia/nccl/lib:${base_ld_library_path}"
        else
            export LD_LIBRARY_PATH="${base_ld_library_path}"
        fi
        python3 -m pip install --no-deps --target "${import_dir}" "${deep_ep_wheels[0]}" >/dev/null
        PYTHONPATH="${import_dir}" python3 - "${AIC_EXPECTED_API}" "${AIC_OVERLAY_STAGING}/runtime.json" <<'"'"'PY'"'"'
import json
import sys
from importlib.metadata import version
from pathlib import Path

import deep_ep
import pyarrow

api, output = sys.argv[1:]
if not hasattr(deep_ep, api):
    raise SystemExit(f"DeepEP overlay is missing {api}")
payload = {
    "deep_ep": version("deep_ep"),
    "deep_ep_api": api,
    "deep_ep_import": str(Path(deep_ep.__file__).resolve()),
    "pyarrow": pyarrow.__version__,
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
deep_ep_setup_sha256=$(<"$(safe_existing_path \
    "rewritten DeepEP setup SHA" "${staging_root}/deep_ep_setup_sha256.txt")")
[[ "${deep_ep_setup_sha256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid rewritten DeepEP setup SHA"

pyarrow_wheel_name=$(<"$(safe_existing_path "pyarrow wheel-name marker" "${staging_root}/pyarrow_wheel_name.txt")")
[[ -n "${pyarrow_wheel_name}" && "${pyarrow_wheel_name}" == "$(basename -- "${pyarrow_wheel_name}")" && \
   "${pyarrow_wheel_name}" == *.whl ]] || die "unsafe pyarrow wheel name ${pyarrow_wheel_name}"
pyarrow_wheel=$(safe_existing_path "pyarrow runtime wheel" "${staging_root}/deps/${pyarrow_wheel_name}")
pyarrow_wheel_sha256=$(sha256sum "${pyarrow_wheel}" | awk '{print $1}')

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
cp "${pyarrow_wheel}" "${publish_dir}/${pyarrow_wheel_name}"
if [[ "${OVERLAY_PROFILE}" == v2 ]]; then
    cp "${nccl_wheel}" "${publish_dir}/${nccl_wheel_name}"
fi
cp "${staging_root}/runtime.json" "${publish_dir}/runtime.json"
cp "${staging_root}/cuda_devel_root.txt" "${publish_dir}/cuda_devel_root.txt"
cp "${staging_root}/elf_dynamic.txt" "${publish_dir}/elf_dynamic.txt"

python3 - "${publish_dir}/build_meta.json" "${SYSTEM}" "${architecture}" "${OVERLAY_PROFILE}" \
    "${scaleup_ranks}" "${IMAGE_DIGEST}" "${VLLM_COMMIT}" "${deepep_commit}" \
    "${legacy_patch_sha256}" "${NVSHMEM_VERSION}" "${V2_NCCL_VERSION}" "${CUDA_ARCHES}" \
    "${deep_ep_wheel_name}" "${deep_ep_wheel_sha256}" "${nccl_wheel_name}" "${nccl_wheel_sha256}" \
    "${PYARROW_VERSION}" "${pyarrow_wheel_name}" "${pyarrow_wheel_sha256}" "${gpu_identity}" \
    "${deep_ep_setup_sha256}" <<'PY'
import json
import sys
from datetime import date
from pathlib import Path

(
    output, system, architecture, profile, scaleup_ranks, image, vllm, deepep,
    patch_sha, nvshmem, nccl, arches, deep_ep_wheel, deep_ep_sha,
    nccl_wheel, nccl_sha, pyarrow, pyarrow_wheel, pyarrow_sha, gpu, setup_sha,
) = sys.argv[1:]
payload = {
    "schema_version": 3,
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
    "deep_ep_setup_sha256": setup_sha,
    "nccl_wheel": nccl_wheel,
    "nccl_wheel_sha256": nccl_sha,
    "pyarrow": pyarrow,
    "pyarrow_wheel": pyarrow_wheel,
    "pyarrow_wheel_sha256": pyarrow_sha,
    "build_gpu": gpu,
    "built_at": date.today().isoformat(),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

[[ "$(sha256sum "${publish_dir}/${deep_ep_wheel_name}" | awk '{print $1}')" == "${deep_ep_wheel_sha256}" ]] || \
    die "published DeepEP wheel checksum mismatch"
[[ "$(sha256sum "${publish_dir}/${pyarrow_wheel_name}" | awk '{print $1}')" == "${pyarrow_wheel_sha256}" ]] || \
    die "published pyarrow wheel checksum mismatch"
if [[ "${OVERLAY_PROFILE}" == v2 ]]; then
    [[ "$(sha256sum "${publish_dir}/${nccl_wheel_name}" | awk '{print $1}')" == "${nccl_wheel_sha256}" ]] || \
        die "published NCCL wheel checksum mismatch"
fi
touch "${publish_dir}/SUCCESS"
echo "Published ${OVERLAY_PROFILE} overlay ${deep_ep_wheel_sha256} to ${publish_dir}"
