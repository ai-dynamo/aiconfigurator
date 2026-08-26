#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

system=""
campaign_root=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system) system=$2; shift 2 ;;
        --campaign-root) campaign_root=$2; shift 2 ;;
        -h|--help) echo "Usage: $0 --system gb200|gb300|b200_sxm|b300_sxm|h100_sxm|h200_sxm --campaign-root PATH"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
[[ -n "${system}" && -n "${campaign_root}" ]] || die "missing required argument"

image_index_digest=sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
case "${system}" in
    gb200)
        account=coreai_comparch_inferencex; partition=batch; qos=normal
        image_arch=arm64
        ;;
    gb300)
        account=blackwell; partition=gb300nvl72_preprod; qos=normal
        image_arch=arm64
        ;;
    h100_sxm)
        account=dl_frameworks; partition=dgxh100; qos=normal
        image_arch=amd64
        ;;
    h200_sxm)
        account=dl_frameworks; partition=dgxh200; qos=normal
        image_arch=amd64
        ;;
    b200_sxm)
        account=beta-users_fallback
        partition='b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb'
        qos=batch-short; image_arch=amd64
        ;;
    b300_sxm)
        account=beta-users_b300
        partition='b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb'
        qos=batch-short; image_arch=amd64
        ;;
    *) die "unsupported image-staging system ${system}" ;;
esac

campaign_root=$(realpath -e "${campaign_root}")
case "${campaign_root}" in
    /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited campaign path ${campaign_root}" ;;
esac
script_dir=$(cd "$(dirname "$0")" && pwd)
payload=$(realpath -e "${script_dir}/run_vllm_image_stage_job.sh")
log_dir="${campaign_root}/slurm_logs/${system}/image_stage"
mkdir -p "${log_dir}"
log_dir=$(realpath -e "${log_dir}")

export SYSTEM="${system}" CAMPAIGN_ROOT="${campaign_root}"
export IMAGE_INDEX_DIGEST="${image_index_digest}" IMAGE_ARCH="${image_arch}"
export CONTAINER_IMAGE="registry-1.docker.io#vllm/vllm-openai:${image_index_digest}"

# Image staging emits no benchmark data and only imports the runtime for ABI
# attestation. Keep scarce full nodes available for exclusive benchmark jobs.
job_id=$(sbatch \
    --parsable \
    --job-name="aic-v024-${system}-image-stage" \
    --account="${account}" \
    --partition="${partition}" \
    --qos="${qos}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    --switches=1 \
    --time=02:00:00 \
    --output="${log_dir}/image_stage_%j.out" \
    --error="${log_dir}/image_stage_%j.err" \
    --export=ALL \
    "${payload}")
echo "submitted ${job_id}: ${system} ${image_arch} index-pinned image staging"
