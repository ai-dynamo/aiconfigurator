#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

system=""; run_kind=""; backend=""; campaign_root=""; repo_dir=""; container_image=""; wheel_dir=""
afterok_job=""; afterok_stage_job=""; approved_nodelist=""; fabric_approval_id=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system) system=$2; shift 2 ;;
        --run-kind) run_kind=$2; shift 2 ;;
        --backend) backend=$2; shift 2 ;;
        --campaign-root) campaign_root=$2; shift 2 ;;
        --repo-dir) repo_dir=$2; shift 2 ;;
        --container-image) container_image=$2; shift 2 ;;
        --wheel-dir) wheel_dir=$2; shift 2 ;;
        --afterok-job) afterok_job=$2; shift 2 ;;
        --afterok-stage-job) afterok_stage_job=$2; shift 2 ;;
        --approved-nodelist) approved_nodelist=$2; shift 2 ;;
        --fabric-approval-id) fabric_approval_id=$2; shift 2 ;;
        -h|--help) echo "Usage: $0 --system SYSTEM --run-kind canary|full --backend trtllm_deepep_ht|trtllm_deepep_ll --campaign-root PATH --repo-dir PATH --container-image SQSH --wheel-dir PATH [--afterok-stage-job ID | --afterok-job ID]"; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done
[[ -n "${system}" && -n "${run_kind}" && -n "${backend}" && -n "${campaign_root}" && -n "${repo_dir}" && -n "${container_image}" && -n "${wheel_dir}" ]] || die "missing required argument"
case "${run_kind}" in canary|full) ;; *) die "run kind must be canary or full" ;; esac
case "${backend}" in trtllm_deepep_ht|trtllm_deepep_ll) ;; *) die "unsupported backend ${backend}" ;; esac
if [[ -n "${afterok_job}" ]]; then
    [[ "${run_kind}" == full && "${afterok_job}" =~ ^[0-9]+$ ]] || die "--afterok-job requires a numeric full-job dependency"
fi
if [[ -n "${afterok_stage_job}" ]]; then
    [[ "${run_kind}" == canary && "${afterok_stage_job}" =~ ^[0-9]+$ ]] || die "--afterok-stage-job requires a numeric canary dependency"
fi
[[ -z "${afterok_job}" || -z "${afterok_stage_job}" ]] || die "only one dependency kind may be specified"

case "${system}" in
    gb200) account=coreai_comparch_inferencex; partition=batch; qos=normal; gpus_per_node=4; image_arch=arm64; cuda_arches=100a_real ;;
    gb300) account=blackwell; partition=gb300nvl72_preprod; qos=normal; gpus_per_node=4; image_arch=arm64; cuda_arches=103a_real ;;
    h100_sxm) account=dl_frameworks; partition=dgxh100; qos=normal; gpus_per_node=8; image_arch=amd64; cuda_arches=90_real ;;
    h200_sxm) account=dl_frameworks; partition=dgxh200; qos=normal; gpus_per_node=8; image_arch=amd64; cuda_arches=90_real ;;
    b200_sxm) account=beta-users_fallback; partition='b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb'; qos=batch-short; gpus_per_node=8; image_arch=amd64; cuda_arches=100a_real ;;
    b300_sxm) account=beta-users_b300; partition='b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb'; qos=batch-short; gpus_per_node=8; image_arch=amd64; cuda_arches=103a_real ;;
    *) die "unsupported system ${system}" ;;
esac

for variable in campaign_root repo_dir; do
    value=$(realpath -e -- "${!variable}") || die "${variable} does not exist"
    case "${value}" in /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "${variable} uses prohibited storage" ;; esac
    printf -v "${variable}" '%s' "${value}"
done
if [[ "${image_arch}" == arm64 ]]; then
    image_child=2202825c5950b4925e1add7d458228c9ad3368671789856f24d8947b4defd21c
else
    image_child=9b3b4dfb811caa9420fa99a6f958155f6a1f727ffc2b5a5c2d9d2ce51fdc323d
fi
expected_image="${campaign_root}/images/trtllm/${system}/trtllm_rc20_${image_arch}_${image_child}.sqsh"
expected_wheel_dir="${campaign_root}/runtime/trtllm/${system}/wheel_14efb6ac673c0cbe828e1206cc5c7d5748d05ffa_${cuda_arches}"
container_image=$(realpath -m -- "${container_image}")
wheel_dir=$(realpath -m -- "${wheel_dir}")
case "${container_image}" in /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "container_image uses prohibited storage" ;; esac
case "${wheel_dir}" in /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "wheel_dir uses prohibited storage" ;; esac
[[ "${container_image}" == "${expected_image}" && "${wheel_dir}" == "${expected_wheel_dir}" ]] || die "runtime paths do not match the attested stage outputs"

runtime_complete=false
if [[ -f "${container_image}" && -f "${container_image}.meta.json" && -f "${wheel_dir}/build_meta.json" && -f "${wheel_dir}/SUCCESS" ]]; then
    runtime_complete=true
fi
deferred_runtime=false
if [[ "${run_kind}" == canary && -n "${afterok_stage_job}" ]] || [[ "${run_kind}" == full && -n "${afterok_job}" ]]; then
    deferred_runtime=true
fi
if [[ "${runtime_complete}" != true ]]; then
    [[ "${deferred_runtime}" == true ]] || die "staged runtime is incomplete and no transitive afterok gate was supplied"
    for parent in "$(dirname "${container_image}")" "$(dirname "${wheel_dir}")"; do
        resolved_parent=$(realpath -e -- "${parent}") || die "future runtime parent does not exist: ${parent}"
        [[ "${resolved_parent}" == "${campaign_root}"/* ]] || die "future runtime parent escapes campaign root"
    done
fi
if [[ "${system}" == b200_sxm || "${system}" == b300_sxm ]]; then
    [[ -n "${approved_nodelist}" && -n "${fabric_approval_id}" ]] || die "${system} submission requires infra-approved nodelist and approval ID"
fi
if [[ "${run_kind}" == full && -z "${afterok_job}" ]]; then
    compgen -G "${campaign_root}/${system}/trtllm/canary/1n/${backend}/job_*/SUCCESS" >/dev/null || die "no successful ${backend} canary found"
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
payload=$(realpath -e "${script_dir}/run_trtllm_moe_a2a_job.sh")
log_dir="${campaign_root}/slurm_logs/${system}/trtllm"
mkdir -p "${log_dir}"
export SYSTEM="${system}" GPUS_PER_NODE="${gpus_per_node}" BACKEND="${backend}" RUN_KIND="${run_kind}"
export CAMPAIGN_ROOT="${campaign_root}" REPO_DIR="${repo_dir}" CONTAINER_IMAGE="${container_image}" WHEEL_DIR="${wheel_dir}"
export IMAGE_ARCH="${image_arch}" AIC_APPROVED_NODELIST="${approved_nodelist}" AIC_FABRIC_APPROVAL_ID="${fabric_approval_id}"
args=(--parsable --job-name="aic-trt-a2a-${system}-${backend}-${run_kind}" --account="${account}" --partition="${partition}" --qos="${qos}"
    --nodes=1 --ntasks="${gpus_per_node}" --ntasks-per-node="${gpus_per_node}" --gpus-per-node="${gpus_per_node}"
    --exclusive --switches=1 --time=06:00:00 --output="${log_dir}/${backend}_${run_kind}_%j.out" --error="${log_dir}/${backend}_${run_kind}_%j.err" --export=ALL)
[[ -z "${approved_nodelist}" ]] || args+=(--nodelist="${approved_nodelist}")
[[ -z "${afterok_stage_job}" ]] || args+=(--dependency="afterok:${afterok_stage_job}")
[[ -z "${afterok_job}" ]] || args+=(--dependency="afterok:${afterok_job}")
job_id=$(sbatch "${args[@]}" "${payload}")
echo "submitted ${job_id}: ${system} 1-node EP${gpus_per_node} ${backend} ${run_kind}"
