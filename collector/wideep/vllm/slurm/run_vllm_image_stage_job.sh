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

for required_name in SYSTEM CAMPAIGN_ROOT IMAGE_INDEX_DIGEST IMAGE_ARCH CONTAINER_IMAGE SLURM_JOB_ID; do
    [[ -n "${!required_name:-}" ]] || die "missing ${required_name}"
done
case "${IMAGE_ARCH}" in arm64|amd64) ;; *) die "bad image architecture ${IMAGE_ARCH}" ;; esac
[[ "${IMAGE_INDEX_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid image index digest"
[[ "${CONTAINER_IMAGE}" == "registry-1.docker.io#vllm/vllm-openai:${IMAGE_INDEX_DIGEST}" ]] || \
    die "container reference must use the configured multi-arch index digest"

# Resolve the platform child from the configured immutable index at execution
# time. The child is observed evidence, never a second configured image pin.
IMAGE_DIGEST=$(python3 - "${IMAGE_ARCH}" "${IMAGE_INDEX_DIGEST}" <<'PY'
import json
import sys
import urllib.parse
import urllib.request

architecture, expected_index_digest = sys.argv[1:]
query = urllib.parse.urlencode({
    "service": "registry.docker.io",
    "scope": "repository:vllm/vllm-openai:pull",
})
with urllib.request.urlopen(f"https://auth.docker.io/token?{query}") as response:
    token = json.load(response)["token"]
request = urllib.request.Request(
    f"https://registry-1.docker.io/v2/vllm/vllm-openai/manifests/{expected_index_digest}",
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": ",".join((
            "application/vnd.oci.image.index.v1+json",
            "application/vnd.docker.distribution.manifest.list.v2+json",
        )),
    },
)
with urllib.request.urlopen(request) as response:
    observed_index_digest = response.headers.get("Docker-Content-Digest")
    manifest = json.load(response)
if observed_index_digest != expected_index_digest:
    raise SystemExit(
        f"vLLM image index digest mismatch: expected {expected_index_digest}, observed {observed_index_digest}"
    )
matches = [
    entry["digest"]
    for entry in manifest.get("manifests", [])
    if entry.get("platform", {}).get("os") == "linux"
    and entry.get("platform", {}).get("architecture") == architecture
]
if len(matches) != 1:
    raise SystemExit(
        f"vLLM 0.24.0 index has ambiguous {architecture} children: {matches}"
    )
print(matches[0])
PY
) || die "failed to resolve platform child from configured image index"
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid observed child digest"
image_reference_mode=enroot-3.4-index-digest

campaign_root=$(safe_existing_path "campaign root" "${CAMPAIGN_ROOT}")
digest_value=${IMAGE_DIGEST#sha256:}
image_dir="${campaign_root}/images"
job_dir="${image_dir}/${SYSTEM}/job_${SLURM_JOB_ID}"
mkdir -p "${job_dir}"
job_dir=$(safe_existing_path "image staging evidence" "${job_dir}")
final_image="${image_dir}/vllm_v024_${IMAGE_ARCH}_${digest_value}.sqsh"
image_meta="${final_image}.meta.json"
if [[ -f "${final_image}" && -f "${image_meta}" ]]; then
    python3 - "${final_image}" "${image_meta}" "${SYSTEM}" "${IMAGE_INDEX_DIGEST}" "${IMAGE_DIGEST}" <<'PY'
import hashlib, json, sys
from pathlib import Path
image, meta_path = map(Path, sys.argv[1:3])
system, index, child = sys.argv[3:]
meta = json.loads(meta_path.read_text(encoding="utf-8"))
expected = {"schema_version": 2, "system": system, "configured_image_digest": index,
            "observed_image_digest": child}
for key, value in expected.items():
    if meta.get(key) != value:
        raise SystemExit(f"existing staged image {key} mismatch")
digest = hashlib.sha256()
with image.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != meta.get("sqsh_sha256"):
    raise SystemExit("existing staged image checksum mismatch")
PY
    touch "${job_dir}/SUCCESS"
    echo "Reused checksum-verified staged image ${final_image}"
    exit 0
fi
[[ ! -e "${final_image}" && ! -e "${image_meta}" ]] || die "partial staged image set requires operator cleanup"
temporary_image="${image_dir}/.vllm_v024_${IMAGE_ARCH}_${digest_value}.sqsh.tmp.${SLURM_JOB_ID}"
[[ ! -e "${temporary_image}" ]] || die "stale temporary image ${temporary_image}"

export ENROOT_CACHE_PATH="/tmp/aic-enroot-image-stage-${SLURM_JOB_ID}"
export ENROOT_MAX_CONNECTIONS=1
export ENROOT_TRANSFER_RETRIES=8
mkdir -p "${ENROOT_CACHE_PATH}"
safe_existing_path "image layer cache" "${ENROOT_CACHE_PATH}" >/dev/null

enroot_library_dir="/tmp/aic-enroot-library-${SLURM_JOB_ID}"
[[ ! -e "${enroot_library_dir}" ]] || die "stale job-local Enroot library ${enroot_library_dir}"
mkdir -p "${enroot_library_dir}"
cp -a /usr/lib/enroot/. "${enroot_library_dir}/"
python3 - "${enroot_library_dir}/docker.sh" "${enroot_library_dir}/common.sh" <<'PY'
import re
import sys
from pathlib import Path

docker_path, common_path = map(Path, sys.argv[1:])
source = docker_path.read_text()
needle = ".manifests[]"
replacement = ".manifests[]?"
source, replacement_count = re.subn(r"\.manifests\[\](?!\?)", replacement, source)
if replacement_count == 0 and replacement not in source:
    raise SystemExit("unexpected Enroot docker manifest-list parser")
docker_path.write_text(source)

# Cluster Enroot suppresses jq's actual parser error. The job-local copy keeps
# stderr so a registry or parser regression has actionable evidence.
source = common_path.read_text()
needle = 'if ! jq "$@" 2> /dev/null; then'
replacement = 'if ! tee "${AIC_ENROOT_JSON_DEBUG_FILE:-/dev/null}" | jq "$@"; then'
if replacement not in source:
    if source.count(needle) != 1:
        raise SystemExit("unexpected Enroot jq wrapper")
    source = source.replace(needle, replacement)
common_path.write_text(source)
PY
case "${IMAGE_ARCH}" in
    amd64) enroot_arch=x86_64 ;;
    arm64) enroot_arch=aarch64 ;;
esac
export AIC_ENROOT_JSON_DEBUG_FILE="${job_dir}/registry_manifest_response.json"
ENROOT_LIBRARY_PATH="${enroot_library_dir}" enroot import \
    --arch="${enroot_arch}" \
    --output="${temporary_image}" \
    "docker://${CONTAINER_IMAGE}"
runtime_container_image=$(safe_existing_path "temporary staged image" "${temporary_image}")

export AIC_IMAGE_STAGE_EVIDENCE="${job_dir}/runtime.json"
srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="${runtime_container_image}" \
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
temporary_meta="${image_meta}.tmp.${SLURM_JOB_ID}"
[[ ! -e "${image_meta}" && ! -e "${temporary_meta}" ]] || die "refusing to overwrite staged image metadata"
configured_image="vllm/vllm-openai:v0.24.0@${IMAGE_INDEX_DIGEST}"
python3 - "${job_dir}/image_meta.json" "${SYSTEM}" "${IMAGE_ARCH}" "${configured_image}" \
    "${IMAGE_INDEX_DIGEST}" "${IMAGE_DIGEST}" "${sqsh_sha256}" "${final_image}" "${AIC_IMAGE_STAGE_EVIDENCE}" \
    "${DEEPEP_COMMIT}" "${image_reference_mode}" <<'PY'
import json
import sys
from datetime import date
from pathlib import Path

(
    output, system, arch, configured_image, index_digest, child_digest, sqsh_sha, image, runtime_path,
    deepep_commit, image_reference_mode,
) = sys.argv[1:]
runtime = json.loads(Path(runtime_path).read_text())
if runtime["vllm"] != "0.24.0":
    raise SystemExit(f"unexpected vLLM version: {runtime['vllm']}")
if runtime["deep_ep"] != f"1.2.1+{deepep_commit[:7]}":
    raise SystemExit(f"unexpected DeepEP build: {runtime['deep_ep']}")
Path(output).write_text(json.dumps({
    "schema_version": 2,
    "system": system,
    "architecture": arch,
    "image_variant": f"linux/{arch}",
    "configured_image": configured_image,
    "configured_image_digest": index_digest,
    "observed_image_digest": child_digest,
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
echo "Published ${IMAGE_INDEX_DIGEST} ${IMAGE_ARCH} child ${IMAGE_DIGEST} as ${final_image} (${sqsh_sha256})"
