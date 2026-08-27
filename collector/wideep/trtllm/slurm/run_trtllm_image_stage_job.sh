#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
TRT_SOURCE_COMMIT=14efb6ac673c0cbe828e1206cc5c7d5748d05ffa
DEEPEP_COMMIT=5be51b228a7c82dbdb213ea58e77bffd12b38af8
NVSHMEM_VERSION=3.2.5-1
NVSHMEM_ARCHIVE_SHA256=eb2c8fb3b7084c2db86bd9fd905387909f1dfd483e7b45f7b3c3d5fcf5374b5a
TRANSFORMERS_REQUIREMENT=transformers==4.57.3

die() { echo "ERROR: $*" >&2; exit 1; }
safe_existing_path() {
    local resolved; resolved=$(realpath -e -- "$2") || die "$1 does not exist: $2"
    case "${resolved}" in /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "$1 uses prohibited storage" ;; esac
    printf '%s\n' "${resolved}"
}
for name in SYSTEM CAMPAIGN_ROOT IMAGE_ARCH CUDA_ARCHES IMAGE_INDEX_DIGEST CONTAINER_IMAGE SLURM_JOB_ID; do
    [[ -n "${!name:-}" ]] || die "missing ${name}"
done
case "${IMAGE_ARCH}" in arm64|amd64) ;; *) die "invalid IMAGE_ARCH" ;; esac
[[ "${IMAGE_INDEX_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid index digest"
[[ "${CONTAINER_IMAGE}" == "nvcr.io#nvidia/tensorrt-llm/release:${IMAGE_INDEX_DIGEST}" ]] || die "image must use configured index"
campaign_root=$(safe_existing_path "campaign root" "${CAMPAIGN_ROOT}")
SEED_IMAGE=${SEED_IMAGE:-}
SEED_IMAGE_META=${SEED_IMAGE_META:-}
SEED_WHEEL_DIR=${SEED_WHEEL_DIR:-}
[[ -z "${SEED_WHEEL_DIR}" || -n "${SEED_IMAGE}" ]] || die "seed wheel requires seed image"
case "${IMAGE_ARCH}" in
    arm64) EXPECTED_IMAGE_DIGEST=sha256:2202825c5950b4925e1add7d458228c9ad3368671789856f24d8947b4defd21c ;;
    amd64) EXPECTED_IMAGE_DIGEST=sha256:9b3b4dfb811caa9420fa99a6f958155f6a1f727ffc2b5a5c2d9d2ce51fdc323d ;;
esac

if [[ -z "${SEED_IMAGE}" ]]; then
read -r IMAGE_DIGEST < <(python3 - "${IMAGE_ARCH}" "${IMAGE_INDEX_DIGEST}" <<'PY'
import json, sys, urllib.parse, urllib.request
arch, index = sys.argv[1:]
query = urllib.parse.urlencode({"service": "nvcr.io", "scope": "repository:nvidia/tensorrt-llm/release:pull"})
with urllib.request.urlopen(f"https://nvcr.io/proxy_auth?{query}") as response:
    token = json.load(response)["token"]
request = urllib.request.Request(
    f"https://nvcr.io/v2/nvidia/tensorrt-llm/release/manifests/{index}",
    headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.docker.distribution.manifest.list.v2+json"},
)
with urllib.request.urlopen(request) as response:
    if response.headers.get("Docker-Content-Digest") != index:
        raise SystemExit("configured TRT-LLM index digest mismatch")
    manifest = json.load(response)
matches = [m["digest"] for m in manifest.get("manifests", []) if m.get("platform", {}).get("os") == "linux" and m.get("platform", {}).get("architecture") == arch]
if len(matches) != 1:
    raise SystemExit(f"ambiguous {arch} image children: {matches}")
print(matches[0])
PY
) || die "failed to resolve platform child"
else
    IMAGE_DIGEST=${EXPECTED_IMAGE_DIGEST}
fi
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid child digest"
[[ "${IMAGE_DIGEST}" == "${EXPECTED_IMAGE_DIGEST}" ]] || die "platform child digest mismatch"

image_dir="${campaign_root}/images/trtllm/${SYSTEM}"
artifact_dir="${campaign_root}/runtime/trtllm/${SYSTEM}"
job_root="/tmp/aic-trtllm-stage-${SLURM_JOB_ID}"
mkdir -p "${image_dir}" "${artifact_dir}" "${job_root}"
job_root=$(safe_existing_path "job staging" "${job_root}")
digest_value=${IMAGE_DIGEST#sha256:}
final_image="${image_dir}/trtllm_rc20_${IMAGE_ARCH}_${digest_value}.sqsh"
final_meta="${final_image}.meta.json"
final_wheel_dir="${artifact_dir}/wheel_${TRT_SOURCE_COMMIT}_${CUDA_ARCHES//[^0-9A-Za-z]/_}"
[[ ! -e "${final_image}" && ! -e "${final_meta}" && ! -e "${final_wheel_dir}" ]] || die "refusing to overwrite staged runtime"

temporary_image="${job_root}/runtime.sqsh"
seed_provenance="${job_root}/seed_provenance.json"
repo_root=$(cd "$(dirname "$0")/../../../.." && pwd)
if [[ -n "${SEED_IMAGE}" ]]; then
    seed_image=$(safe_existing_path "seed image" "${SEED_IMAGE}")
    seed_image_meta=$(safe_existing_path "seed image metadata" "${SEED_IMAGE_META}")
    seed_args=(--image "${seed_image}" --image-meta "${seed_image_meta}" --target-system "${SYSTEM}" --output "${seed_provenance}")
    if [[ -n "${SEED_WHEEL_DIR}" ]]; then
        seed_wheel_dir=$(safe_existing_path "seed wheel directory" "${SEED_WHEEL_DIR}")
        seed_args+=(--wheel-dir "${seed_wheel_dir}")
    fi
    PYTHONPATH="${repo_root}" python3 -m collector.wideep.trtllm.runtime_artifacts "${seed_args[@]}" \
        || die "seed runtime validation failed"
    cp --reflink=auto -- "${seed_image}" "${temporary_image}"
else
    printf 'null\n' > "${seed_provenance}"
    export ENROOT_CACHE_PATH="${job_root}/enroot-cache" ENROOT_MAX_CONNECTIONS=1 ENROOT_TRANSFER_RETRIES=8
    mkdir -p "${ENROOT_CACHE_PATH}"
enroot_library_dir="${job_root}/enroot-library"
mkdir -p "${enroot_library_dir}"
cp -a /usr/lib/enroot/. "${enroot_library_dir}/"
python3 - "${enroot_library_dir}/docker.sh" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
source, count = re.subn(r"\.manifests\[\](?!\?)", ".manifests[]?", source)
if count == 0 and ".manifests[]?" not in source:
    raise SystemExit("unexpected Enroot manifest-list parser")
path.write_text(source)
PY
ENROOT_LIBRARY_PATH="${enroot_library_dir}" enroot import --arch="$([[ "${IMAGE_ARCH}" == amd64 ]] && echo x86_64 || echo aarch64)" \
    --output="${temporary_image}" "docker://${CONTAINER_IMAGE}"
fi
unsquashfs -s "${temporary_image}" >/dev/null || die "invalid staged squashfs"

if [[ -n "${SEED_WHEEL_DIR}" ]]; then
    publish_wheel_dir="${job_root}/publish-wheel"
    cp -a -- "${seed_wheel_dir}" "${publish_wheel_dir}"
    rm -f -- "${publish_wheel_dir}/SUCCESS"
    python3 - "${publish_wheel_dir}/build_meta.json" "${seed_provenance}" "${SYSTEM}" <<'PY'
import json
import sys
from datetime import date
from pathlib import Path

meta_path, seed_path, system = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
meta = json.loads(meta_path.read_text(encoding="utf-8"))
meta["system"] = system
meta["seed_provenance"] = json.loads(seed_path.read_text(encoding="utf-8"))
meta["staged_at"] = date.today().isoformat()
meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    cp -- "${publish_wheel_dir}/build_meta.json" "${job_root}/build_meta.json"
    cp -- "${job_root}/build_meta.json" "${final_meta}"
    mv -- "${temporary_image}" "${final_image}"
    touch "${publish_wheel_dir}/SUCCESS"
    mv -- "${publish_wheel_dir}" "${final_wheel_dir}"
    echo "Published seeded TRT-LLM runtime ${final_image} and ${final_wheel_dir}"
    exit 0
fi

source_root="${job_root}/TensorRT-LLM"
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/TensorRT-LLM.git "${source_root}"
git -C "${source_root}" checkout --detach "${TRT_SOURCE_COMMIT}"
git -C "${source_root}" submodule update --init --recursive
[[ "$(git -C "${source_root}" rev-parse HEAD)" == "${TRT_SOURCE_COMMIT}" ]] || die "source commit mismatch"
grep -Fq "\"git_tag\": \"${DEEPEP_COMMIT}\"" "${source_root}/3rdparty/fetch_content.json" || die "DeepEP pin mismatch"
[[ "$(sha256sum "${source_root}/cpp/tensorrt_llm/deep_ep/nvshmem_src_${NVSHMEM_VERSION}.txz" | awk '{print $1}')" == "${NVSHMEM_ARCHIVE_SHA256}" ]] || die "NVSHMEM archive mismatch"

wheel_staging="${job_root}/wheel"
dependency_staging="${job_root}/dependencies"
validation_root="${job_root}/validation"
mkdir -p "${wheel_staging}" "${dependency_staging}" "${validation_root}"
export AIC_SOURCE_ROOT="${source_root}" AIC_WHEEL_STAGING="${wheel_staging}" AIC_CUDA_ARCHES="${CUDA_ARCHES}"
export AIC_DEPENDENCY_STAGING="${dependency_staging}" AIC_VALIDATION_ROOT="${validation_root}"
export AIC_TRANSFORMERS_REQUIREMENT="${TRANSFORMERS_REQUIREMENT}"
srun --mpi=pmix --nodes=1 --ntasks=1 --container-image="${temporary_image}" \
    --container-mounts="${source_root}:${source_root},${wheel_staging}:${wheel_staging},${dependency_staging}:${dependency_staging},${validation_root}:${validation_root}" \
    --container-workdir="${source_root}" bash -lc \
    'set -euo pipefail; python3 scripts/build_wheel.py --clean --no-venv --cuda_architectures "${AIC_CUDA_ARCHES}" --dist_dir "${AIC_WHEEL_STAGING}"; python3 -m pip download --only-binary=:all: --dest "${AIC_DEPENDENCY_STAGING}" "${AIC_TRANSFORMERS_REQUIREMENT}"; python3 -m pip install --no-index --find-links "${AIC_DEPENDENCY_STAGING}" --target "${AIC_VALIDATION_ROOT}" "${AIC_TRANSFORMERS_REQUIREMENT}"; python3 -m pip install --no-deps --target "${AIC_VALIDATION_ROOT}" "${AIC_WHEEL_STAGING}"/tensorrt_llm-*.whl; PYTHONPATH="${AIC_VALIDATION_ROOT}" python3 - <<'"'"'PY'"'"'
import tensorrt_llm
assert tensorrt_llm.__version__ == "1.3.0rc11", tensorrt_llm.__version__
import transformers
assert transformers.__version__ == "4.57.3", transformers.__version__
from tensorrt_llm._torch.modules.fused_moe.communication import CommunicationFactory
assert hasattr(CommunicationFactory, "_create_forced_method")
PY'
mapfile -t wheels < <(find "${wheel_staging}" -maxdepth 1 -type f -name 'tensorrt_llm-*.whl' -print)
[[ "${#wheels[@]}" == 1 ]] || die "expected exactly one TRT-LLM wheel"
mapfile -t dependency_wheels < <(find "${dependency_staging}" -maxdepth 1 -type f -name '*.whl' -print | sort)
[[ "${#dependency_wheels[@]}" -gt 0 ]] || die "dependency wheelhouse is empty"
wheel_sha=$(sha256sum "${wheels[0]}" | awk '{print $1}')
sqsh_sha=$(sha256sum "${temporary_image}" | awk '{print $1}')
mkdir -p "${final_wheel_dir}/dependencies"
cp -- "${wheels[0]}" "${final_wheel_dir}/"
cp -- "${dependency_wheels[@]}" "${final_wheel_dir}/dependencies/"
wheel_name=$(basename "${wheels[0]}")
python3 - "${job_root}/build_meta.json" "${SYSTEM}" "${IMAGE_ARCH}" "${CUDA_ARCHES}" \
    "${IMAGE_INDEX_DIGEST}" "${IMAGE_DIGEST}" "${sqsh_sha}" "${wheel_name}" "${wheel_sha}" \
    "${dependency_staging}" "${TRANSFORMERS_REQUIREMENT}" "${seed_provenance}" <<'PY'
import hashlib, json, sys
from datetime import date
from pathlib import Path
out, system, arch, cuda_arches, index, child, sqsh_sha, wheel, wheel_sha, dependency_dir, transformers_requirement, seed_path = sys.argv[1:]
dependency_wheels = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(Path(dependency_dir).glob("*.whl"))
}
if not dependency_wheels:
    raise SystemExit("dependency wheelhouse is empty")
payload = {
  "schema_version": 1, "system": system, "architecture": arch, "image_variant": f"linux/{arch}",
  "configured_image": f"nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20@{index}",
  "configured_image_digest": index, "observed_image_digest": child, "sqsh_sha256": sqsh_sha,
  "trtllm_version": "1.3.0rc11", "source_commit": "14efb6ac673c0cbe828e1206cc5c7d5748d05ffa",
  "deep_ep": "5be51b228a7c82dbdb213ea58e77bffd12b38af8", "nvshmem": "3.2.5-1",
  "nvshmem_archive_sha256": "eb2c8fb3b7084c2db86bd9fd905387909f1dfd483e7b45f7b3c3d5fcf5374b5a",
  "cuda_arches": cuda_arches, "wheel": wheel, "wheel_sha256": wheel_sha,
  "python_requirements": [transformers_requirement], "dependency_wheels": dependency_wheels,
  "staged_at": date.today().isoformat(),
}
seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
if seed is not None:
    payload["seed_provenance"] = seed
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
cp -- "${job_root}/build_meta.json" "${final_wheel_dir}/build_meta.json"
cp -- "${job_root}/build_meta.json" "${final_meta}"
mv "${temporary_image}" "${final_image}"
touch "${final_wheel_dir}/SUCCESS"
echo "Published TRT-LLM image ${final_image} and rc11 wheel ${final_wheel_dir}/${wheel_name} (${wheel_sha})"
