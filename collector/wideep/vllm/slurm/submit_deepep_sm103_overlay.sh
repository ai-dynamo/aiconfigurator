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
vllm_source_root=""
container_image=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system) system=$2; shift 2 ;;
        --campaign-root) campaign_root=$2; shift 2 ;;
        --vllm-source-root) vllm_source_root=$2; shift 2 ;;
        --container-image) container_image=$2; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --system gb300|b300_sxm --campaign-root PATH --vllm-source-root PATH [--container-image PATH]"
            exit 0
            ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ -n "${system}" && -n "${campaign_root}" && -n "${vllm_source_root}" ]] || die "missing required argument"

case "${system}" in
    gb300)
        account=blackwell
        partition=gb300nvl72_preprod
        qos=normal
        gpus_per_node=4
        image_digest=sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b
        ;;
    b300_sxm)
        account=beta-users_b300
        partition='b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb'
        qos=batch-short
        gpus_per_node=8
        image_digest=sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8
        ;;
    *) die "SM103 overlay build supports only gb300 or b300_sxm" ;;
esac

campaign_root=$(realpath -e "${campaign_root}")
vllm_source_root=$(realpath -e "${vllm_source_root}")
for checked_path in "${campaign_root}" "${vllm_source_root}"; do
    case "${checked_path}" in
        /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited shared storage path ${checked_path}" ;;
    esac
done
if [[ -z "${container_image}" ]]; then
    container_image="vllm/vllm-openai:v0.24.0@${image_digest}"
else
    container_image=$(realpath -e -- "${container_image}")
    case "${container_image}" in
        /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited container image path ${container_image}" ;;
    esac
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
payload=$(realpath -e "${script_dir}/run_deepep_sm103_overlay_job.sh")
log_dir="${campaign_root}/slurm_logs/${system}/overlay"
mkdir -p "${log_dir}"
log_dir=$(realpath -e "${log_dir}")

export SYSTEM="${system}"
export CAMPAIGN_ROOT="${campaign_root}"
export VLLM_SOURCE_ROOT="${vllm_source_root}"
export CONTAINER_IMAGE="${container_image}"
export IMAGE_DIGEST="${image_digest}"

job_id=$(sbatch \
    --parsable \
    --job-name="aic-v024-${system}-sm103-overlay" \
    --account="${account}" \
    --partition="${partition}" \
    --qos="${qos}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node="${gpus_per_node}" \
    --exclusive \
    --switches=1 \
    --time=04:00:00 \
    --output="${log_dir}/overlay_%j.out" \
    --error="${log_dir}/overlay_%j.err" \
    --export=ALL \
    "${payload}")
echo "submitted ${job_id}: ${system} DeepEP SM103 overlay build"
