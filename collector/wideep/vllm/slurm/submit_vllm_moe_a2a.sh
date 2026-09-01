#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_vllm_moe_a2a.sh --system SYSTEM --run-kind canary|full \
  --campaign-root PATH --repo-dir PATH --vllm-source-root PATH [OPTIONS]

Options:
  --nodes 1                   Formal campaign is single-node only (default: 1).
  --backends LIST             Default: the formal backend set for SYSTEM.
  --container-image PATH      Required image staged from the pinned index.
  --image-index-digest DIGEST Configured multi-arch index digest.
  --legacy-overlay-dir PATH   Required for GB200/GB300 and B300 legacy jobs.
  --v2-overlay-dir PATH       Required for every deepep_v2 job.
  --approved-nodelist LIST    Required for B200/B300.
  --fabric-approval-id ID     Required for B200/B300; copied into provenance.
  --partition-override NAME   Submit to an explicitly selected compatible Slurm partition.
  --afterok-job BACKEND=ID    Gate one full backend on its own canary; repeat per backend.
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
backends=""
container_image=""
image_index_digest="sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
image_digest=""
image_variant=""
legacy_overlay_dir=""
v2_overlay_dir=""
approved_nodelist=""
fabric_approval_id=""
partition_override=""
afterok_specs=()
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
        --image-index-digest) image_index_digest=$2; shift 2 ;;
        --legacy-overlay-dir) legacy_overlay_dir=$2; shift 2 ;;
        --v2-overlay-dir) v2_overlay_dir=$2; shift 2 ;;
        --approved-nodelist) approved_nodelist=$2; shift 2 ;;
        --fabric-approval-id) fabric_approval_id=$2; shift 2 ;;
        --partition-override) partition_override=$2; shift 2 ;;
        --afterok-job)
            [[ "$2" =~ ^(deepep_ht|deepep_ll|deepep_v2)=([0-9]+)$ ]] || \
                die "--afterok-job must be BACKEND=JOB_ID"
            afterok_specs+=("$2")
            shift 2
            ;;
        --allow-full-without-canary) allow_full_without_canary=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
done

[[ -n "${system}" && -n "${run_kind}" && -n "${campaign_root}" && \
   -n "${repo_dir}" && -n "${vllm_source_root}" ]] || { usage; exit 2; }
case "${run_kind}" in
    canary|full)
        nodes=${nodes:-1}
        [[ "${nodes}" == 1 ]] || die "the formal vLLM campaign supports only --nodes 1"
        ;;
    *) die "--run-kind must be canary or full" ;;
esac
if [[ "${#afterok_specs[@]}" -gt 0 ]]; then
    [[ "${run_kind}" == full ]] || die "--afterok-job is valid only for full jobs"
fi

case "${system}" in
    h100_sxm) formal_backends="deepep_ht,deepep_ll,deepep_v2" ;;
    gb200|gb300|b200_sxm|b300_sxm|h200_sxm) formal_backends="deepep_ht,deepep_ll" ;;
    *) die "unsupported system ${system}" ;;
esac
backends=${backends:-${formal_backends}}
IFS=',' read -r -a backend_values <<< "${backends}"
[[ "${#backend_values[@]}" -gt 0 ]] || die "--backends must select at least one backend"
for backend_index in "${!backend_values[@]}"; do
    backend=${backend_values[${backend_index}]}
    case "${backend}" in deepep_ht|deepep_ll|deepep_v2) ;; *) die "bad backend ${backend}" ;; esac
    case ",${formal_backends}," in
        *,"${backend}",*) ;;
        *) die "${backend} is not a formal backend for ${system}; supported: ${formal_backends}" ;;
    esac
    for earlier_index in "${!backend_values[@]}"; do
        [[ "${earlier_index}" -ge "${backend_index}" ]] && break
        [[ "${backend_values[${earlier_index}]}" != "${backend}" ]] || die "duplicate selected backend ${backend}"
    done
done
if [[ "${#afterok_specs[@]}" -gt 0 ]]; then
    for spec_index in "${!afterok_specs[@]}"; do
        afterok_spec=${afterok_specs[${spec_index}]}
        spec_backend=${afterok_spec%%=*}
        spec_job=${afterok_spec#*=}
        selected=false
        for backend in "${backend_values[@]}"; do
            [[ "${backend}" != "${spec_backend}" ]] || selected=true
        done
        [[ "${selected}" == true ]] || die "--afterok-job backend ${spec_backend} is not selected"
        for earlier_index in "${!afterok_specs[@]}"; do
            [[ "${earlier_index}" -ge "${spec_index}" ]] && break
            earlier_spec=${afterok_specs[${earlier_index}]}
            [[ "${earlier_spec%%=*}" != "${spec_backend}" ]] || die "duplicate --afterok-job for ${spec_backend}"
            [[ "${earlier_spec#*=}" != "${spec_job}" ]] || die "Slurm canary job ${spec_job} is bound to multiple backends"
        done
    done
    for backend in "${backend_values[@]}"; do
        mapped=false
        for afterok_spec in "${afterok_specs[@]}"; do
            [[ "${afterok_spec%%=*}" != "${backend}" ]] || mapped=true
        done
        [[ "${mapped}" == true ]] || die "missing --afterok-job for selected backend ${backend}"
    done
fi

case "${system}" in
    gb200)
        account=coreai_comparch_inferencex; partition=batch; qos=normal; gpus_per_node=4; time_limit=04:00:00
        image_arch=arm64
        ;;
    gb300)
        account=blackwell; partition=gb300nvl72_preprod; qos=normal; gpus_per_node=4; time_limit=04:00:00
        image_arch=arm64
        ;;
    h100_sxm)
        account=dl_frameworks; partition=dgxh100; qos=normal; gpus_per_node=8; time_limit=04:00:00
        image_arch=amd64
        ;;
    h200_sxm)
        account=dl_frameworks; partition=dgxh200; qos=normal; gpus_per_node=8; time_limit=04:00:00
        image_arch=amd64
        ;;
    b200_sxm)
        account=beta-users_fallback
        partition='b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb'
        qos=batch-short; gpus_per_node=8; time_limit=04:00:00
        image_arch=amd64
        ;;
    b300_sxm)
        account=beta-users_b300
        partition='b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb'
        qos=batch-short; gpus_per_node=8; time_limit=04:00:00
        image_arch=amd64
        ;;
    *) die "unsupported system ${system}" ;;
esac
if [[ -n "${partition_override}" ]]; then
    [[ "${partition_override}" =~ ^[A-Za-z0-9_.@+-]+$ ]] || die "invalid Slurm partition override"
    partition=${partition_override}
fi

[[ "${image_index_digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid image index digest"
[[ "${container_image}" == /* ]] || die "--container-image must be a locally staged squashfs path"
container_image=$(realpath -e -- "${container_image}")
case "${container_image}" in
    /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited container image path ${container_image}" ;;
esac
container_image_meta=$(realpath -e -- "${container_image}.meta.json")
container_image_success=$(realpath -e -- "${container_image}.SUCCESS") || \
    die "staged image completion marker does not exist"
read -r image_digest image_variant < <(
    python3 - "${container_image_meta}" "${image_index_digest}" "${image_arch}" <<'PY'
import json
import sys
from pathlib import Path

meta_path, expected_index, expected_arch = sys.argv[1:]
payload = json.loads(Path(meta_path).read_text())
checks = {
    "schema_version": 2,
    "configured_image": f"vllm/vllm-openai:v0.24.0@{expected_index}",
    "configured_image_digest": expected_index,
    "image_variant": f"linux/{expected_arch}",
}
for key, expected in checks.items():
    if payload.get(key) != expected:
        raise SystemExit(f"staged image {key} mismatch: {payload.get(key)!r} != {expected!r}")
child = payload.get("observed_image_digest", "")
if not child.startswith("sha256:") or len(child) != 71:
    raise SystemExit(f"staged image has invalid observed child digest: {child!r}")
print(child, payload["image_variant"])
PY
) || die "staged image metadata validation failed"

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

node_values=(1)

# Validate the complete requested backend set before the first sbatch.  Keep
# the resolved overlays in backend-indexed maps so the submission loop cannot
# discover a missing later overlay after an earlier backend has been queued.
declare -A overlay_dirs
declare -A runtime_abi_json_by_backend
if [[ "${system}" == b200_sxm || "${system}" == b300_sxm ]]; then
    [[ -n "${approved_nodelist}" && -n "${fabric_approval_id}" ]] || die \
        "${system} submission requires infra-approved nodelist and approval ID"
fi
for backend in "${backend_values[@]}"; do
    if [[ "${backend}" == deepep_v2 ]]; then
        [[ -d "${v2_overlay_dir}" ]] || die "deepep_v2 requires --v2-overlay-dir"
        overlay_dirs[${backend}]=$(realpath -e -- "${v2_overlay_dir}")
        runtime_abi_json_by_backend[${backend}]='{"build_mode":"official-v0.24.0-image+deepep-v2-overlay","torch":"2.11.0","cuda":"13.0.2","deep_ep":"b306af06afd412c88e51e71802951606e40b7358","nvshmem":"3.3.24","deep_ep_api":"ElasticBuffer","nccl":"2.30.4","deep_ep_topology_source":"nccl_lsa"}'
    else
        [[ -d "${legacy_overlay_dir}" ]] || die \
            "${system} ${backend} requires --legacy-overlay-dir"
        overlay_dirs[${backend}]=$(realpath -e -- "${legacy_overlay_dir}")
        runtime_abi_json_by_backend[${backend}]='{"build_mode":"official-v0.24.0-image","torch":"2.11.0","cuda":"13.0.2","deep_ep":"73b6ea4a439ba03a695563f9fd242c8e4b02b37c","nvshmem":"3.3.24","deep_ep_api":"Buffer"}'
    fi
    case "${overlay_dirs[${backend}]}" in
        /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*) die "prohibited overlay path ${overlay_dirs[${backend}]}" ;;
    esac
    if [[ "${run_kind}" == full && "${#afterok_specs[@]}" -eq 0 && "${allow_full_without_canary}" != true ]]; then
        compgen -G "${campaign_root}/${system}/canary/1n/${backend}/job_*/SUCCESS" >/dev/null || die \
            "no successful 1-node ${backend} canary found for ${system}"
    fi
done

for node_num in "${node_values[@]}"; do
    for backend in "${backend_values[@]}"; do
        overlay_dir=${overlay_dirs[${backend}]}
        runtime_abi_json=${runtime_abi_json_by_backend[${backend}]}
        afterok_job=""
        if [[ "${#afterok_specs[@]}" -gt 0 ]]; then
            for afterok_spec in "${afterok_specs[@]}"; do
                [[ "${afterok_spec%%=*}" != "${backend}" ]] || afterok_job=${afterok_spec#*=}
            done
        fi
        world_size=$((node_num * gpus_per_node))
        job_name="aic-v024-${system}-${node_num}n-${backend}-${run_kind}"
        export SYSTEM="${system}" NODE_NUM="${node_num}" GPUS_PER_NODE="${gpus_per_node}"
        export BACKEND="${backend}" RUN_KIND="${run_kind}" CAMPAIGN_ROOT="${campaign_root}"
        export REPO_DIR="${repo_dir}" VLLM_SOURCE_ROOT="${vllm_source_root}"
        export CONTAINER_IMAGE="${container_image}" IMAGE_INDEX_DIGEST="${image_index_digest}"
        export IMAGE_DIGEST="${image_digest}" IMAGE_VARIANT="${image_variant}"
        export RUNTIME_ABI_JSON="${runtime_abi_json}"
        export DEEP_EP_OVERLAY_DIR="${overlay_dir}"
        export AIC_APPROVED_NODELIST="${approved_nodelist}" AIC_FABRIC_APPROVAL_ID="${fabric_approval_id}"
        export AIC_CANARY_JOB_ID="${afterok_job}"

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
        if [[ -n "${afterok_job}" ]]; then
            sbatch_args+=(--dependency="afterok:${afterok_job}")
        fi
        job_id=$(sbatch "${sbatch_args[@]}" "${payload}")
        echo "submitted ${job_id}: ${system} ${node_num}-node ${backend} ${run_kind}"
    done
done
