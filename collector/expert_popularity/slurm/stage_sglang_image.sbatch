#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=expert-pop-image
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=expert-popularity-image-%j.out
#SBATCH --error=expert-popularity-image-%j.err

set -Eeuo pipefail

: "${IMAGE_REFERENCE:?submit with an immutable docker://...@sha256: reference}"
: "${IMPORT_REFERENCE:?submit with the enroot-compatible pinned tag used for import}"
: "${IMAGE_OUTPUT_PATH:?submit with IMAGE_OUTPUT_PATH}"
EXPECTED_IMAGE_SHA256="${EXPECTED_IMAGE_SHA256:-}"
if [[ ! "$IMAGE_REFERENCE" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "IMAGE_REFERENCE must contain an immutable manifest digest: $IMAGE_REFERENCE" >&2
    exit 19
fi
if [[ -n "$EXPECTED_IMAGE_SHA256" && ! "$EXPECTED_IMAGE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "EXPECTED_IMAGE_SHA256 must be a lowercase sha256: $EXPECTED_IMAGE_SHA256" >&2
    exit 21
fi

IMAGE_PATH="$IMAGE_OUTPUT_PATH"
IMAGE_DIR="$(dirname "$IMAGE_PATH")"
IMAGE_BASENAME="$(basename "$IMAGE_PATH")"
if [[ "$IMAGE_BASENAME" == "." || "$IMAGE_BASENAME" == "/" ]]; then
    echo "IMAGE_OUTPUT_PATH must name a file" >&2
    exit 18
fi
IMAGE_SOURCE_PATH="$IMAGE_PATH.source"
IMAGE_SHA256_PATH="$IMAGE_PATH.sha256"
TEMP_PATH="$IMAGE_DIR/.$IMAGE_BASENAME.$SLURM_JOB_ID.tmp"
mkdir -p "$IMAGE_DIR"
cleanup_temp() {
    rm -f "$TEMP_PATH"
}
trap cleanup_temp EXIT

resolve_import_digest() {
    python3 - "$IMPORT_REFERENCE" <<'PY'
import json
import re
import sys
import urllib.parse
import urllib.request

reference = sys.argv[1]
match = re.fullmatch(r"docker://(?:registry-1[.]docker[.]io/|docker[.]io/)?(.+):([^/:@]+)", reference)
if match is None:
    raise SystemExit(f"unsupported Docker Hub import reference: {reference}")
repository, tag = match.groups()
query = urllib.parse.urlencode(
    {"service": "registry.docker.io", "scope": f"repository:{repository}:pull"}
)
with urllib.request.urlopen(f"https://auth.docker.io/token?{query}", timeout=60) as response:
    token = json.load(response)["token"]
request = urllib.request.Request(
    f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
    method="HEAD",
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": ", ".join(
            (
                "application/vnd.oci.image.index.v1+json",
                "application/vnd.docker.distribution.manifest.list.v2+json",
                "application/vnd.oci.image.manifest.v1+json",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
        ),
    },
)
with urllib.request.urlopen(request, timeout=60) as response:
    digest = response.headers.get("Docker-Content-Digest")
if not digest:
    raise SystemExit("registry response did not include Docker-Content-Digest")
print(digest)
PY
}

expected_manifest_digest="${IMAGE_REFERENCE##*@}"
resolved_manifest_digest="$(resolve_import_digest)"
if [[ "$resolved_manifest_digest" != "$expected_manifest_digest" ]]; then
    echo "import tag digest mismatch: expected=$expected_manifest_digest actual=$resolved_manifest_digest" >&2
    exit 24
fi

if [[ -s "$IMAGE_PATH" ]]; then
    if [[ ! -s "$IMAGE_SOURCE_PATH" ]] || [[ "$(<"$IMAGE_SOURCE_PATH")" != "$IMAGE_REFERENCE" ]]; then
        echo "existing image has no matching immutable source identity" >&2
        exit 20
    fi
    if [[ ! -s "$IMAGE_SHA256_PATH" ]]; then
        echo "existing image has no stored sqsh sha256" >&2
        exit 25
    fi
    actual_sha256="$(sha256sum "$IMAGE_PATH" | awk '{print $1}')"
    if [[ "$actual_sha256" != "$(<"$IMAGE_SHA256_PATH")" ]]; then
        echo "existing image no longer matches its stored sqsh sha256" >&2
        exit 26
    fi
    if [[ -n "$EXPECTED_IMAGE_SHA256" && "$actual_sha256" != "$EXPECTED_IMAGE_SHA256" ]]; then
        echo "existing image sha256 mismatch: expected=$EXPECTED_IMAGE_SHA256 actual=$actual_sha256" >&2
        exit 22
    fi
    echo "[$(date -u +%FT%TZ)] IMAGE_STAGE_ALREADY_PASS path=$IMAGE_PATH sha256=$actual_sha256 source=$IMAGE_REFERENCE"
    exit 0
fi

command -v enroot >/dev/null
enroot import --output "$TEMP_PATH" "$IMPORT_REFERENCE"
resolved_manifest_digest_after="$(resolve_import_digest)"
if [[ "$resolved_manifest_digest_after" != "$expected_manifest_digest" ]]; then
    echo "import tag changed during import: expected=$expected_manifest_digest actual=$resolved_manifest_digest_after" >&2
    exit 27
fi
actual_sha256="$(sha256sum "$TEMP_PATH" | awk '{print $1}')"
if [[ -n "$EXPECTED_IMAGE_SHA256" && "$actual_sha256" != "$EXPECTED_IMAGE_SHA256" ]]; then
    echo "imported image sha256 mismatch: expected=$EXPECTED_IMAGE_SHA256 actual=$actual_sha256 import=$IMPORT_REFERENCE" >&2
    exit 23
fi
mv "$TEMP_PATH" "$IMAGE_PATH"
printf '%s\n' "$IMAGE_REFERENCE" > "$IMAGE_SOURCE_PATH"
printf '%s\n' "$actual_sha256" > "$IMAGE_SHA256_PATH"
echo "[$(date -u +%FT%TZ)] IMAGE_STAGE_PASS path=$IMAGE_PATH sha256=$actual_sha256 source=$IMAGE_REFERENCE"
