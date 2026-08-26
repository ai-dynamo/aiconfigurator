#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

system=""
profile=""
campaign_root=""
repo_dir=""
vllm_source_root=""
container_image=""
cuda_arches=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system) system=$2; shift 2 ;;
        --profile) profile=$2; shift 2 ;;
        --campaign-root) campaign_root=$2; shift 2 ;;
        --repo-dir) repo_dir=$2; shift 2 ;;
        --vllm-source-root) vllm_source_root=$2; shift 2 ;;
        --container-image) container_image=$2; shift 2 ;;
        --cuda-arches) cuda_arches=$2; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --system SYSTEM --profile legacy-nvl8|legacy-nvl4|v2 --campaign-root PATH --repo-dir PATH --vllm-source-root PATH --container-image PATH [--cuda-arches LIST]"
            exit 0
            ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ -n "${system}" && -n "${profile}" && -n "${campaign_root}" && -n "${repo_dir}" && \
   -n "${vllm_source_root}" ]] || die "missing required argument"
case "${profile}" in legacy-nvl8|legacy-nvl4|v2) ;; *) die "bad --profile ${profile}" ;; esac

image_index_digest=sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
case "${system}" in
    gb200)
        account=coreai_comparch_inferencex; partition=batch; qos=normal
        image_arch=arm64; cuda_arches=${cuda_arches:-"10.0a 10.3a"}
        [[ "${profile}" != legacy-nvl8 ]] || die "gb200 overlay builds use legacy-nvl4"
        ;;
    gb300)
        account=blackwell; partition=gb300nvl72_preprod; qos=normal
        image_arch=arm64; cuda_arches=${cuda_arches:-"10.0a 10.3a"}
        [[ "${profile}" != legacy-nvl8 ]] || die "gb300 overlay builds use legacy-nvl4"
        ;;
    h100_sxm)
        account=dl_frameworks; partition=dgxh100; qos=normal
        image_arch=amd64; cuda_arches=${cuda_arches:-"9.0 10.0a 10.3a"}
        [[ "${profile}" != legacy-nvl4 ]] || die "H100 overlay builds use legacy-nvl8"
        ;;
    h200_sxm)
        account=dl_frameworks; partition=dgxh200; qos=normal
        image_arch=amd64; cuda_arches=${cuda_arches:-"9.0 10.0a 10.3a"}
        [[ "${profile}" != legacy-nvl4 ]] || die "H200 overlay builds use legacy-nvl8"
        ;;
    b200_sxm)
        account=beta-users_fallback
        partition='b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb'
        qos=batch-short; image_arch=amd64
        cuda_arches=${cuda_arches:-"9.0 10.0a 10.3a"}
        [[ "${profile}" != legacy-nvl4 ]] || die "B200 overlay builds use legacy-nvl8"
        ;;
    b300_sxm)
        account=beta-users_b300
        partition='b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb'
        qos=batch-short; image_arch=amd64
        cuda_arches=${cuda_arches:-"9.0 10.0a 10.3a"}
        [[ "${profile}" != legacy-nvl4 ]] || die "B300 overlay builds use legacy-nvl8"
        ;;
    *) die "unsupported system ${system}" ;;
esac

campaign_root=$(realpath -e "${campaign_root}")
repo_dir=$(realpath -e "${repo_dir}")
vllm_source_root=$(realpath -e "${vllm_source_root}")
for checked_path in "${campaign_root}" "${repo_dir}" "${vllm_source_root}"; do
    case "${checked_path}" in
        /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited shared storage path ${checked_path}" ;;
    esac
done
[[ "${container_image}" == /* ]] || die "--container-image must be a locally staged squashfs path"
container_image=$(realpath -e -- "${container_image}")
case "${container_image}" in
    /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited container image path ${container_image}" ;;
esac
container_image_meta=$(realpath -e -- "${container_image}.meta.json")
read -r image_digest image_variant < <(
    python3 - "${container_image_meta}" "${image_index_digest}" "${image_arch}" <<'PY'
import json
import sys
from pathlib import Path

meta_path, expected_index, expected_arch = sys.argv[1:]
payload = json.loads(Path(meta_path).read_text())
if payload.get("schema_version") != 2:
    raise SystemExit("staged image metadata schema mismatch")
if payload.get("configured_image") != f"vllm/vllm-openai:v0.24.0@{expected_index}":
    raise SystemExit("staged configured image mismatch")
if payload.get("configured_image_digest") != expected_index:
    raise SystemExit("staged image index digest mismatch")
if payload.get("image_variant") != f"linux/{expected_arch}":
    raise SystemExit("staged image architecture mismatch")
print(payload["observed_image_digest"], payload["image_variant"])
PY
) || die "staged image metadata validation failed"

script_dir=$(cd "$(dirname "$0")" && pwd)
payload=$(realpath -e "${script_dir}/run_deepep_sm103_overlay_job.sh")
log_dir="${campaign_root}/slurm_logs/${system}/overlay"
mkdir -p "${log_dir}"
log_dir=$(realpath -e "${log_dir}")

export SYSTEM="${system}"
export OVERLAY_PROFILE="${profile}"
export CUDA_ARCHES="${cuda_arches}"
export CAMPAIGN_ROOT="${campaign_root}"
export REPO_DIR="${repo_dir}"
export VLLM_SOURCE_ROOT="${vllm_source_root}"
export CONTAINER_IMAGE="${container_image}"
export IMAGE_INDEX_DIGEST="${image_index_digest}"
export IMAGE_DIGEST="${image_digest}"
export IMAGE_VARIANT="${image_variant}"

# Overlay compilation emits no benchmark data. One typed GPU is sufficient for
# the post-build CUDA/ABI check; benchmark jobs remain exclusive-node jobs.
job_id=$(sbatch \
    --parsable \
    --job-name="aic-v024-${system}-${profile}-overlay" \
    --account="${account}" \
    --partition="${partition}" \
    --qos="${qos}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    --switches=1 \
    --time=04:00:00 \
    --output="${log_dir}/${profile}_%j.out" \
    --error="${log_dir}/${profile}_%j.err" \
    --export=ALL \
    "${payload}")
echo "submitted ${job_id}: ${system} ${profile} DeepEP overlay build"
