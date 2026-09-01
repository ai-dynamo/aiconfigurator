#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

system=""; campaign_root=""; seed_image=""; seed_wheel_dir=""
account_override=""; partition_override=""; qos_override=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system) system=$2; shift 2 ;;
        --campaign-root) campaign_root=$2; shift 2 ;;
        --account) account_override=$2; shift 2 ;;
        --partition) partition_override=$2; shift 2 ;;
        --qos) qos_override=$2; shift 2 ;;
        --seed-image) seed_image=$2; shift 2 ;;
        --seed-wheel-dir) seed_wheel_dir=$2; shift 2 ;;
        -h|--help) echo "Usage: $0 --system SYSTEM --campaign-root PATH [--seed-image SQSH [--seed-wheel-dir DIR]] [--account ACCOUNT --partition PARTITION --qos QOS]"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
[[ -n "${system}" && -n "${campaign_root}" ]] || die "missing required argument"

case "${system}" in
    gb200) account=coreai_comparch_inferencex; partition=batch; qos=normal; time_limit=04:00:00; image_arch=arm64; cuda_arches='100a-real'; gpu_args=(--gpus-per-node=4) ;;
    gb300) account=blackwell; partition=gb300nvl72_preprod; qos=normal; time_limit=04:00:00; image_arch=arm64; cuda_arches='103a-real'; gpu_args=(--gpus-per-node=4) ;;
    h100_sxm) account=dl_frameworks; partition=dgxh100; qos=normal; time_limit=06:00:00; image_arch=amd64; cuda_arches='90-real'; gpu_args=(--gpus=1) ;;
    h200_sxm) account=dl_frameworks; partition=dgxh200; qos=normal; time_limit=06:00:00; image_arch=amd64; cuda_arches='90-real'; gpu_args=(--gpus=1) ;;
    b200_sxm) account=beta-users_fallback; partition='b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb'; qos=batch-short; time_limit=04:00:00; image_arch=amd64; cuda_arches='100a-real'; gpu_args=(--gpus=1) ;;
    b300_sxm) account=beta-users_b300; partition='b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb'; qos=batch-short; time_limit=04:00:00; image_arch=amd64; cuda_arches='103a-real'; gpu_args=(--gpus=1) ;;
    *) die "unsupported system ${system}" ;;
esac
[[ -z "${account_override}" ]] || account=${account_override}
[[ -z "${partition_override}" ]] || partition=${partition_override}
[[ -z "${qos_override}" ]] || qos=${qos_override}

campaign_root=$(realpath -e -- "${campaign_root}")
case "${campaign_root}" in /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited campaign path" ;; esac
[[ -z "${seed_wheel_dir}" || -n "${seed_image}" ]] || die "--seed-wheel-dir requires --seed-image"
if [[ -n "${seed_image}" ]]; then
    seed_image=$(realpath -e -- "${seed_image}") || die "seed image does not exist"
    seed_image_meta=$(realpath -e -- "${seed_image}.meta.json") || die "seed image metadata does not exist"
    case "${seed_image}" in /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "seed image uses prohibited storage" ;; esac
else
    seed_image_meta=""
fi
if [[ -n "${seed_wheel_dir}" ]]; then
    seed_wheel_dir=$(realpath -e -- "${seed_wheel_dir}") || die "seed wheel directory does not exist"
    case "${seed_wheel_dir}" in /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "seed wheel directory uses prohibited storage" ;; esac
fi
script_dir=$(cd "$(dirname "$0")" && pwd)
payload=$(realpath -e "${script_dir}/run_trtllm_image_stage_job.sh")
repo_root=$(realpath -e "${script_dir}/../../../..")
log_dir="${campaign_root}/slurm_logs/${system}/trtllm_image_stage"
mkdir -p "${log_dir}" "${campaign_root}/images/trtllm/${system}" "${campaign_root}/runtime/trtllm/${system}"

export SYSTEM="${system}" CAMPAIGN_ROOT="${campaign_root}" IMAGE_ARCH="${image_arch}" CUDA_ARCHES="${cuda_arches}"
export IMAGE_INDEX_DIGEST=sha256:1532b38814b3faf2affdb5ef01ca91468685d314ffb7e8926a0567595355ed88
export CONTAINER_IMAGE="nvcr.io#nvidia/tensorrt-llm/release:${IMAGE_INDEX_DIGEST}"
export SEED_IMAGE="${seed_image}" SEED_IMAGE_META="${seed_image_meta}" SEED_WHEEL_DIR="${seed_wheel_dir}"
export AIC_REPO_DIR="${repo_root}"

job_id=$(sbatch --parsable --job-name="aic-trt-a2a-${system}-stage" \
    --account="${account}" --partition="${partition}" --qos="${qos}" \
    --nodes=1 --ntasks=1 "${gpu_args[@]}" --switches=1 --time="${time_limit}" \
    --output="${log_dir}/stage_%j.out" --error="${log_dir}/stage_%j.err" --export=ALL "${payload}")
echo "submitted ${job_id}: ${system} TRT-LLM rc11 source-wheel/image stage"
