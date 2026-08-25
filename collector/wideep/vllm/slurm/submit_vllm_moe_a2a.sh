#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_vllm_moe_a2a.sh --system SYSTEM --run-kind canary|full \
  --campaign-root PATH --repo-dir PATH --vllm-source-root PATH [OPTIONS]

Options:
  --nodes 2|4|all             Canary accepts only 2; full defaults to all.
  --backends LIST             Default: deepep_ht,deepep_ll,deepep_v2.
  --container-image REF       Digest-pinned image; defaults by architecture.
  --image-digest DIGEST       Observed child digest; defaults by architecture.
  --overlay-wheel PATH        Required for GB300/B300 (SM103 DeepEP wheel).
  --approved-nodelist LIST    Required for B200/B300.
  --fabric-approval-id ID     Required for B200/B300; copied into provenance.
  --allow-full-without-canary Diagnostic escape hatch; formal operators should not use it.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

system=""
run_kind=""
campaign_root=""
repo_dir=""
vllm_source_root=""
nodes=""
backends="deepep_ht,deepep_ll,deepep_v2"
container_image=""
image_digest=""
overlay_wheel=""
approved_nodelist=""
fabric_approval_id=""
allow_full_without_canary=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system) system=$2; shift 2 ;;
        --run-kind) run_kind=$2; shift 2 ;;
        --campaign-root) campaign_root=$2; shift 2 ;;
        --repo-dir) repo_dir=$2; shift 2 ;;
        --vllm-source-root) vllm_source_root=$2; shift 2 ;;
        --nodes) nodes=$2; shift 2 ;;
        --backends) backends=$2; shift 2 ;;
        --container-image) container_image=$2; shift 2 ;;
        --image-digest) image_digest=$2; shift 2 ;;
        --overlay-wheel) overlay_wheel=$2; shift 2 ;;
        --approved-nodelist) approved_nodelist=$2; shift 2 ;;
        --fabric-approval-id) fabric_approval_id=$2; shift 2 ;;
        --allow-full-without-canary) allow_full_without_canary=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ -n "${system}" && -n "${run_kind}" && -n "${campaign_root}" && \
   -n "${repo_dir}" && -n "${vllm_source_root}" ]] || { usage; exit 2; }
case "${run_kind}" in
    canary) [[ -z "${nodes}" || "${nodes}" == 2 ]] || die "canary is always 2-node"; nodes=2 ;;
    full) nodes=${nodes:-all}; [[ "${nodes}" == 2 || "${nodes}" == 4 || "${nodes}" == all ]] || die "bad --nodes" ;;
    *) die "--run-kind must be canary or full" ;;
esac

arm64_digest="sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b"
amd64_digest="sha256:f9de5cd9fa907fbf6dbba691eb7db095d48ad58ea283e3eba7142f9a91e186e8"

case "${system}" in
    gb200)
        account=coreai_comparch_inferencex; partition=batch; qos=normal; gpus_per_node=4; time_limit=04:00:00
        image_digest=${image_digest:-${arm64_digest}}
        ;;
    gb300)
        account=blackwell; partition=gb300nvl72_preprod; qos=normal; gpus_per_node=4; time_limit=04:00:00
        image_digest=${image_digest:-${arm64_digest}}
        ;;
    h100_sxm)
        account=dl_frameworks; partition=dgxh100; qos=normal; gpus_per_node=8; time_limit=04:00:00
        image_digest=${image_digest:-${amd64_digest}}
        ;;
    h200_sxm)
        account=dl_frameworks; partition=dgxh200; qos=normal; gpus_per_node=8; time_limit=04:00:00
        image_digest=${image_digest:-${amd64_digest}}
        ;;
    b200_sxm)
        account=beta-users_fallback
        partition='b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb'
        qos=batch-short; gpus_per_node=8; time_limit=04:00:00
        image_digest=${image_digest:-${amd64_digest}}
        [[ -n "${approved_nodelist}" && -n "${fabric_approval_id}" ]] || die \
            "B200 formal/canary submission requires infra-approved nodelist and approval ID"
        [[ "${nodes}" != all ]] || die "B200 needs a distinct approved nodelist for each node count"
        ;;
    b300_sxm)
        account=beta-users_b300
        partition='b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb'
        qos=batch-short; gpus_per_node=8; time_limit=04:00:00
        image_digest=${image_digest:-${amd64_digest}}
        [[ -n "${approved_nodelist}" && -n "${fabric_approval_id}" ]] || die \
            "B300 formal/canary submission requires infra-approved nodelist and approval ID"
        [[ "${nodes}" != all ]] || die "B300 needs a distinct approved nodelist for each node count"
        ;;
    *) die "unsupported system ${system}" ;;
esac

container_image=${container_image:-"vllm/vllm-openai:v0.24.0@${image_digest}"}
[[ "${container_image}" == *@"${image_digest}" ]] || die "container reference and observed child digest differ"

if [[ "${system}" == gb300 || "${system}" == b300_sxm ]]; then
    [[ -f "${overlay_wheel}" ]] || die "SM103 system requires --overlay-wheel"
    overlay_sha256=$(sha256sum "${overlay_wheel}" | awk '{print $1}')
    runtime_abi_json=$(printf \
        '{"build_mode":"official-v0.24.0-image","torch":"2.11.0","cuda":"13.0.2","deep_ep":"73b6ea4a439ba03a695563f9fd242c8e4b02b37c","nvshmem":"3.3.24","deep_ep_overlay_wheel_sha256":"%s","deep_ep_cuda_arches":"10.0a 10.3a"}' \
        "${overlay_sha256}")
else
    runtime_abi_json='{"build_mode":"official-v0.24.0-image","torch":"2.11.0","cuda":"13.0.2","deep_ep":"73b6ea4a439ba03a695563f9fd242c8e4b02b37c","nvshmem":"3.3.24"}'
fi

script_dir=$(cd "$(dirname "$0")" && pwd)
payload=$(realpath -e "${script_dir}/run_vllm_moe_a2a_job.sh")
campaign_root=$(realpath -e "${campaign_root}")
repo_dir=$(realpath -e "${repo_dir}")
vllm_source_root=$(realpath -e "${vllm_source_root}")
for checked_path in "${campaign_root}" "${repo_dir}" "${vllm_source_root}"; do
    case "${checked_path}" in
        /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited shared storage path ${checked_path}" ;;
    esac
done
log_dir="${campaign_root}/slurm_logs/${system}"
mkdir -p "${log_dir}"
log_dir=$(realpath -e "${log_dir}")

IFS=',' read -r -a backend_values <<< "${backends}"
if [[ "${nodes}" == all ]]; then
    node_values=(2 4)
else
    node_values=("${nodes}")
fi

for node_num in "${node_values[@]}"; do
    for backend in "${backend_values[@]}"; do
        case "${backend}" in deepep_ht|deepep_ll|deepep_v2) ;; *) die "bad backend ${backend}" ;; esac
        if [[ "${run_kind}" == full && "${allow_full_without_canary}" != true ]]; then
            compgen -G "${campaign_root}/${system}/canary/2n/${backend}/job_*/SUCCESS" >/dev/null || die \
                "no successful 2-node ${backend} canary found for ${system}"
        fi
        world_size=$((node_num * gpus_per_node))
        job_name="aic-v024-${system}-${node_num}n-${backend}-${run_kind}"
        export SYSTEM="${system}" NODE_NUM="${node_num}" GPUS_PER_NODE="${gpus_per_node}"
        export BACKEND="${backend}" RUN_KIND="${run_kind}" CAMPAIGN_ROOT="${campaign_root}"
        export REPO_DIR="${repo_dir}" VLLM_SOURCE_ROOT="${vllm_source_root}"
        export CONTAINER_IMAGE="${container_image}" IMAGE_DIGEST="${image_digest}"
        export RUNTIME_ABI_JSON="${runtime_abi_json}"
        export DEEP_EP_OVERLAY_WHEEL="${overlay_wheel}"
        export AIC_APPROVED_NODELIST="${approved_nodelist}" AIC_FABRIC_APPROVAL_ID="${fabric_approval_id}"

        sbatch_args=(
            --parsable
            --job-name="${job_name}"
            --account="${account}"
            --partition="${partition}"
            --qos="${qos}"
            --nodes="${node_num}"
            --ntasks="${world_size}"
            --ntasks-per-node="${gpus_per_node}"
            --gpus-per-node="${gpus_per_node}"
            --exclusive
            --switches=1
            --time="${time_limit}"
            --output="${log_dir}/${job_name}_%j.out"
            --error="${log_dir}/${job_name}_%j.err"
            --export=ALL
        )
        if [[ -n "${approved_nodelist}" ]]; then
            sbatch_args+=(--nodelist="${approved_nodelist}")
        fi
        job_id=$(sbatch "${sbatch_args[@]}" "${payload}")
        echo "submitted ${job_id}: ${system} ${node_num}-node ${backend} ${run_kind}"
    done
done
