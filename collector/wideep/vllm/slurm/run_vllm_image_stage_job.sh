#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

DEEPEP_COMMIT=73b6ea4a439ba03a695563f9fd242c8e4b02b37c

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
if [[ "${CONTAINER_IMAGE}" == *@"${IMAGE_DIGEST}" ]]; then
    image_reference_mode=digest
elif [[ "${CONTAINER_IMAGE}" == "docker.io#vllm/vllm-openai:${IMAGE_DIGEST}" ]]; then
    # Enroot 3.4 does not accept Docker's @digest spelling, but its tag parser
    # accepts ':' and sends this exact value to the registry manifest endpoint.
    # This remains a digest request; it is not a mutable registry tag.
    image_reference_mode=enroot-3.4-digest
elif [[ "${CONTAINER_IMAGE}" == "docker.io#vllm/vllm-openai:v0.24.0" ]]; then
    # Enroot before 4.0 parses tag@digest as registry credentials. Resolve the
    # multi-arch tag immediately before import and fail closed unless its
    # selected platform manifest is the locked digest.
    python3 - "${IMAGE_ARCH}" "${IMAGE_DIGEST}" <<'PY'
import json
import sys
import urllib.parse
import urllib.request

architecture, expected_digest = sys.argv[1:]
query = urllib.parse.urlencode({
    "service": "registry.docker.io",
    "scope": "repository:vllm/vllm-openai:pull",
})
with urllib.request.urlopen(f"https://auth.docker.io/token?{query}") as response:
    token = json.load(response)["token"]
request = urllib.request.Request(
    "https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/v0.24.0",
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": ",".join((
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        )),
    },
)
with urllib.request.urlopen(request) as response:
    manifest = json.load(response)
matches = [
    entry["digest"]
    for entry in manifest.get("manifests", [])
    if entry.get("platform", {}).get("os") == "linux"
    and entry.get("platform", {}).get("architecture") == architecture
]
if matches != [expected_digest]:
    raise SystemExit(
        f"vLLM 0.24.0 {architecture} registry digest mismatch: "
        f"expected {[expected_digest]}, observed {matches}"
    )
PY
    image_reference_mode=verified-tag
else
    die "container ref is neither digest pinned nor the verified vLLM 0.24.0 legacy-Pyxis tag"
fi

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

runtime_container_image=${CONTAINER_IMAGE}
container_save_args=(--container-save="${temporary_image}")
if [[ "${image_reference_mode}" == enroot-3.4-digest ]]; then
    # Enroot 3.4 assumes every manifest response is a multi-arch index and
    # errors on a digest-selected single manifest. Patch a job-local copy only;
    # never alter the cluster installation or shared Enroot configuration.
    enroot_library_dir="/tmp/aic-enroot-library-${SLURM_JOB_ID}"
    [[ ! -e "${enroot_library_dir}" ]] || die "stale job-local Enroot library ${enroot_library_dir}"
    mkdir -p "${enroot_library_dir}"
    cp -a /usr/lib/enroot/. "${enroot_library_dir}/"
    python3 - "${enroot_library_dir}/docker.sh" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
needle = ".manifests[]"
replacement = ".manifests[]?"
if source.count(needle) != 1 or replacement in source:
    raise SystemExit("unexpected Enroot docker manifest-list parser")
    path.write_text(source.replace(needle, replacement))
PY
    case "${IMAGE_ARCH}" in
        amd64) enroot_arch=x86_64 ;;
        arm64) enroot_arch=aarch64 ;;
    esac
    ENROOT_LIBRARY_PATH="${enroot_library_dir}" enroot import \
        --arch="${enroot_arch}" \
        --output="${temporary_image}" \
        "docker://${CONTAINER_IMAGE}"
    runtime_container_image=$(safe_existing_path "temporary staged image" "${temporary_image}")
    container_save_args=()
fi

export AIC_IMAGE_STAGE_EVIDENCE="${job_dir}/runtime.json"
srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${runtime_container_image}" \
    "${container_save_args[@]}" \
    --container-mounts="${job_dir}:${job_dir}" \
    bash -lc 'python3 - "${AIC_IMAGE_STAGE_EVIDENCE}" <<'"'"'PY'"'"'
import json
import sys
from importlib.metadata import version
from pathlib import Path

import deep_ep
import torch

assert hasattr(deep_ep, "Buffer")
Path(sys.argv[1]).write_text(json.dumps({
    "vllm": version("vllm"),
    "torch": version("torch"),
    "cuda": torch.version.cuda,
    "deep_ep": version("deep_ep"),
    "deep_ep_v2_available": hasattr(deep_ep, "ElasticBuffer"),
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
    "${IMAGE_DIGEST}" "${sqsh_sha256}" "${final_image}" "${AIC_IMAGE_STAGE_EVIDENCE}" \
    "${DEEPEP_COMMIT}" "${image_reference_mode}" <<'PY'
import json
import sys
from datetime import date
from pathlib import Path

(
    output, system, arch, source_image, digest, sqsh_sha, image, runtime_path,
    deepep_commit, image_reference_mode,
) = sys.argv[1:]
runtime = json.loads(Path(runtime_path).read_text())
if runtime["vllm"] != "0.24.0":
    raise SystemExit(f"unexpected vLLM version: {runtime['vllm']}")
if runtime["deep_ep"] != f"1.2.1+{deepep_commit[:7]}":
    raise SystemExit(f"unexpected DeepEP build: {runtime['deep_ep']}")
Path(output).write_text(json.dumps({
    "schema_version": 1,
    "system": system,
    "architecture": arch,
    "source_image": source_image,
    "source_image_digest": digest,
    "deep_ep_source_commit": deepep_commit,
    "sqsh_sha256": sqsh_sha,
    "image": image,
    "image_reference_mode": image_reference_mode,
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
