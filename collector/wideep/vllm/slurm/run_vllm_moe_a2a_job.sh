#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Slurm job payload for one system/node-count/backend.  Formal artifacts are
# finalized on job-local /tmp first, checksummed, then copied to the compliant
# campaign root.  Every path is canonicalized and shared CIFS is rejected.

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_env() {
    local name=$1
    [[ -n "${!name:-}" ]] || die "required environment variable ${name} is empty"
}

safe_existing_path() {
    local label=$1
    local raw=$2
    local resolved
    resolved=$(realpath -e -- "${raw}") || die "${label} does not exist: ${raw}"
    case "${resolved}" in
        /mnt/cifs|/mnt/cifs/*|/mnt/nvdl|/mnt/nvdl/*)
            die "${label} resolves to prohibited shared storage: ${resolved}"
            ;;
    esac
    printf '%s\n' "${resolved}"
}

safe_future_path() {
    local label=$1
    local raw=$2
    local parent
    local resolved_parent
    parent=$(dirname -- "${raw}")
    mkdir -p -- "${parent}"
    resolved_parent=$(safe_existing_path "${label} parent" "${parent}")
    printf '%s/%s\n' "${resolved_parent}" "$(basename -- "${raw}")"
}

require_env SYSTEM
require_env NODE_NUM
require_env GPUS_PER_NODE
require_env BACKEND
require_env RUN_KIND
require_env CAMPAIGN_ROOT
require_env REPO_DIR
require_env VLLM_SOURCE_ROOT
require_env CONTAINER_IMAGE
require_env IMAGE_DIGEST
require_env RUNTIME_ABI_JSON
require_env SLURM_JOB_ID
require_env SLURM_NODELIST

case "${BACKEND}" in
    deepep_ht|deepep_ll|deepep_v2) ;;
    *) die "unsupported backend ${BACKEND}" ;;
esac
case "${RUN_KIND}" in
    canary|full) ;;
    *) die "RUN_KIND must be canary or full" ;;
esac
case "${NODE_NUM}" in
    2|4) ;;
    *) die "NODE_NUM must be 2 or 4" ;;
esac

case "${SYSTEM}" in
    gb200|gb300)
        [[ "${GPUS_PER_NODE}" == 4 ]] || die "${SYSTEM} requires 4 GPUs/node"
        expected_ep=$((NODE_NUM * 4))
        cross_node_nvlink_capable=true
        topology_mode=native
        if [[ "${SYSTEM}" == gb200 ]]; then
            expected_gpu_token=GB200
            expected_compute_capability=10.0
        else
            expected_gpu_token=GB300
            expected_compute_capability=10.3
        fi
        ;;
    b200_sxm|b300_sxm|h100_sxm|h200_sxm)
        [[ "${GPUS_PER_NODE}" == 8 ]] || die "${SYSTEM} requires 8 GPUs/node"
        expected_ep=$((NODE_NUM * 8))
        cross_node_nvlink_capable=false
        if [[ "${SYSTEM}" == b200_sxm || "${SYSTEM}" == b300_sxm ]]; then
            topology_mode=approved_nodelist
        else
            topology_mode=native
        fi
        case "${SYSTEM}" in
            b200_sxm) expected_gpu_token=B200; expected_compute_capability=10.0 ;;
            b300_sxm) expected_gpu_token=B300; expected_compute_capability=10.3 ;;
            h100_sxm) expected_gpu_token=H100; expected_compute_capability=9.0 ;;
            h200_sxm) expected_gpu_token=H200; expected_compute_capability=9.0 ;;
        esac
        ;;
    *) die "unsupported system ${SYSTEM}" ;;
esac

[[ "${SLURM_NTASKS:-}" == "${expected_ep}" ]] || die \
    "SLURM_NTASKS=${SLURM_NTASKS:-unset}, expected ${expected_ep}"

repo_dir=$(safe_existing_path "repository" "${REPO_DIR}")
vllm_source_root=$(safe_existing_path "vLLM source" "${VLLM_SOURCE_ROOT}")
campaign_root=$(safe_existing_path "campaign root" "${CAMPAIGN_ROOT}")
[[ -z "$(git -C "${repo_dir}" status --porcelain)" ]] || die "repository checkout is dirty"
[[ -z "$(git -C "${vllm_source_root}" status --porcelain)" ]] || die "vLLM source checkout is dirty"
[[ "$(git -C "${vllm_source_root}" rev-parse HEAD)" == \
    "ee0da84ab9e04ac7610e28580af62c365e898389" ]] || die "vLLM source commit mismatch"

if [[ "${CONTAINER_IMAGE}" == /* ]]; then
    container_image=$(safe_existing_path "container image" "${CONTAINER_IMAGE}")
    container_image_meta=$(safe_existing_path "container image metadata" "${container_image}.meta.json")
    python3 - "${container_image}" "${container_image_meta}" "${IMAGE_DIGEST}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

image, metadata, expected_digest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
payload = json.loads(metadata.read_text())
if payload.get("source_image_digest") != expected_digest:
    raise SystemExit("local container source digest mismatch")
observed = hashlib.sha256()
with image.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        observed.update(chunk)
if payload.get("sqsh_sha256") != observed.hexdigest():
    raise SystemExit("local container squashfs checksum mismatch")
PY
else
    container_image=${CONTAINER_IMAGE}
    [[ "${container_image}" == *@sha256:* ]] || die "registry image must be digest pinned"
    [[ "${container_image}" == *@"${IMAGE_DIGEST}" ]] || die "container image does not use IMAGE_DIGEST"
fi
[[ "${IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid IMAGE_DIGEST"

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_NODELIST}" | sort -u)
[[ "${#allocated_nodes[@]}" == "${NODE_NUM}" ]] || die \
    "allocation has ${#allocated_nodes[@]} nodes, expected ${NODE_NUM}"

fabric_identity=""
if [[ "${topology_mode}" == approved_nodelist ]]; then
    require_env AIC_APPROVED_NODELIST
    require_env AIC_FABRIC_APPROVAL_ID
    mapfile -t approved_nodes < <(scontrol show hostnames "${AIC_APPROVED_NODELIST}" | sort -u)
    [[ "${#approved_nodes[@]}" == "${NODE_NUM}" ]] || die \
        "approved nodelist has ${#approved_nodes[@]} nodes, expected ${NODE_NUM}"
    [[ "$(printf '%s\n' "${allocated_nodes[@]}")" == "$(printf '%s\n' "${approved_nodes[@]}")" ]] || die \
        "allocation differs from infra-approved nodelist"
    fabric_identity="approval:${AIC_FABRIC_APPROVAL_ID}"
else
    topology_matches=()
    first_allocated_node=${allocated_nodes[0]}
    while IFS= read -r topology_line; do
        [[ "${topology_line}" == BlockName=* || \
           ("${topology_line}" == SwitchName=* && "${topology_line}" == *"Level=0"*) ]] || continue
        topology_nodes=${topology_line#*Nodes=}
        topology_nodes=${topology_nodes%% *}
        [[ -n "${topology_nodes}" ]] || continue
        # Expanding every block in a large Slurm topology takes minutes and
        # hammers the controller. Select textual candidates by their literal
        # nodelist prefix, then prove membership by expanding only those
        # candidates through Slurm's authoritative hostlist parser.
        prefix_matches=false
        IFS=',' read -r -a topology_segments <<< "${topology_nodes}"
        for topology_segment in "${topology_segments[@]}"; do
            topology_prefix=${topology_segment%%\[*}
            if [[ "${first_allocated_node}" == "${topology_prefix}"* ]]; then
                prefix_matches=true
                break
            fi
        done
        [[ "${prefix_matches}" == true ]] || continue
        mapfile -t expanded_topology_nodes < <(scontrol show hostnames "${topology_nodes}" 2>/dev/null | sort -u)
        all_present=true
        for allocated_node in "${allocated_nodes[@]}"; do
            if ! printf '%s\n' "${expanded_topology_nodes[@]}" | grep -Fxq -- "${allocated_node}"; then
                all_present=false
                break
            fi
        done
        if [[ "${all_present}" == true ]]; then
            topology_matches+=("${topology_line%% *}")
        fi
    done < <(scontrol show topology)
    [[ "${#topology_matches[@]}" == 1 ]] || die \
        "allocation is not contained in exactly one authoritative leaf/block: ${topology_matches[*]:-none}"
    fabric_identity=${topology_matches[0]}
fi

rdma_counts=$(srun --nodes="${NODE_NUM}" --ntasks="${NODE_NUM}" --ntasks-per-node=1 \
    bash -lc 'find /sys/class/infiniband -mindepth 1 -maxdepth 1 -type l 2>/dev/null | wc -l')
while read -r rdma_count; do
    [[ "${rdma_count}" =~ ^[0-9]+$ && "${rdma_count}" -gt 0 ]] || die \
        "at least one allocated node has no observed RDMA device: ${rdma_counts}"
done <<< "${rdma_counts}"
rdma_device_count_min=$(printf '%s\n' "${rdma_counts}" | sort -n | head -n 1)

gpu_inventory=$(srun --nodes="${NODE_NUM}" --ntasks="${NODE_NUM}" --ntasks-per-node=1 \
    bash -lc 'nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader,nounits | sed "s/^/$(hostname)|/"')
mapfile -t gpu_inventory_lines <<< "${gpu_inventory}"
[[ "${#gpu_inventory_lines[@]}" == "${expected_ep}" ]] || die \
    "GPU inventory has ${#gpu_inventory_lines[@]} rows, expected ${expected_ep}: ${gpu_inventory}"
gpu_names_seen=()
driver_versions_seen=()
compute_capabilities_seen=()
for inventory_line in "${gpu_inventory_lines[@]}"; do
    IFS='|' read -r inventory_host inventory_fields <<< "${inventory_line}"
    IFS=',' read -r inventory_gpu inventory_driver inventory_capability <<< "${inventory_fields}"
    inventory_gpu=$(xargs <<< "${inventory_gpu:-}")
    inventory_driver=$(xargs <<< "${inventory_driver:-}")
    inventory_capability=$(xargs <<< "${inventory_capability:-}")
    [[ -n "${inventory_host}" && -n "${inventory_gpu}" && -n "${inventory_driver}" && \
       -n "${inventory_capability}" ]] || die "malformed GPU inventory row: ${inventory_line}"
    [[ "${inventory_gpu^^}" == *"${expected_gpu_token}"* ]] || die \
        "${inventory_host} has ${inventory_gpu}, expected ${expected_gpu_token} for ${SYSTEM}"
    [[ "${inventory_capability}" == "${expected_compute_capability}" ]] || die \
        "${inventory_host} has compute capability ${inventory_capability}, expected ${expected_compute_capability}"
    gpu_names_seen+=("${inventory_gpu}")
    driver_versions_seen+=("${inventory_driver}")
    compute_capabilities_seen+=("${inventory_capability}")
done
mapfile -t gpu_names < <(printf '%s\n' "${gpu_names_seen[@]}" | sort -u)
mapfile -t driver_versions < <(printf '%s\n' "${driver_versions_seen[@]}" | sort -u)
mapfile -t compute_capabilities < <(printf '%s\n' "${compute_capabilities_seen[@]}" | sort -u)
[[ "${#gpu_names[@]}" == 1 && "${#driver_versions[@]}" == 1 && \
   "${#compute_capabilities[@]}" == 1 ]] || die "heterogeneous GPU inventory: ${gpu_inventory}"
gpu_name=${gpu_names[0]}
driver_version=${driver_versions[0]}
compute_capability=${compute_capabilities[0]}
nvlink_topology_sha256=$(
    srun --nodes="${NODE_NUM}" --ntasks="${NODE_NUM}" --ntasks-per-node=1 bash -lc \
        'hostname; nvidia-smi topo -m' | sha256sum | awk '{print $1}'
)

staging_root="/tmp/aic-vllm-a2a-${SLURM_JOB_ID}"
[[ "${staging_root}" == /tmp/aic-vllm-a2a-* ]] || die "unsafe staging root ${staging_root}"
mkdir -p -- "${staging_root}"
staging_root=$(safe_existing_path "job staging" "${staging_root}")
output_dir="${staging_root}/${SYSTEM}/${RUN_KIND}/${NODE_NUM}n/${BACKEND}"
srun --nodes="${NODE_NUM}" --ntasks="${NODE_NUM}" --ntasks-per-node=1 mkdir -p -- "${output_dir}"
output_dir=$(safe_existing_path "job output" "${output_dir}")

export ENROOT_CACHE_PATH="/tmp/aic-enroot-cache-${SLURM_JOB_ID}"
[[ "${ENROOT_CACHE_PATH}" == /tmp/aic-enroot-cache-* ]] || die "unsafe container cache path"
srun --nodes="${NODE_NUM}" --ntasks="${NODE_NUM}" --ntasks-per-node=1 mkdir -p -- "${ENROOT_CACHE_PATH}"
safe_existing_path "container cache" "${ENROOT_CACHE_PATH}" >/dev/null

runtime_abi_json=$(
    python3 - "${RUNTIME_ABI_JSON}" "${SYSTEM}" "${fabric_identity}" "${gpu_name}" \
        "${driver_version}" "${compute_capability}" "${rdma_device_count_min}" \
        "${cross_node_nvlink_capable}" "${nvlink_topology_sha256}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
payload.update(
    {
        "system": sys.argv[2],
        "fabric_identity": sys.argv[3],
        "gpu_name": sys.argv[4],
        "driver": sys.argv[5],
        "compute_capability": sys.argv[6],
        "rdma_device_count_min": sys.argv[7],
        "cross_node_nvlink_capable": sys.argv[8],
        "nvlink_topology_sha256": sys.argv[9],
        "slurm_topology_verified": "true",
    }
)
print(json.dumps(payload, separators=(",", ":")))
PY
)

master_addr=${allocated_nodes[0]}
container_mounts="${repo_dir}:${repo_dir},${vllm_source_root}:${vllm_source_root},${staging_root}:${staging_root}"
if [[ "${SYSTEM}" == gb300 || "${SYSTEM}" == b300_sxm ]]; then
    require_env DEEP_EP_OVERLAY_WHEEL
    overlay_wheel=$(safe_existing_path "DeepEP overlay wheel" "${DEEP_EP_OVERLAY_WHEEL}")
    overlay_sha256=$(sha256sum "${overlay_wheel}" | awk '{print $1}')
    read -r attested_overlay_sha256 attested_cuda_arches < <(
        python3 - "${runtime_abi_json}" <<'PY'
import json
import sys

abi = json.loads(sys.argv[1])
print(abi.get("deep_ep_overlay_wheel_sha256", ""), abi.get("deep_ep_cuda_arches", ""))
PY
    )
    [[ "${attested_overlay_sha256}" == "${overlay_sha256}" ]] || die \
        "SM103 overlay wheel SHA256 differs from the runtime attestation"
    [[ "${attested_cuda_arches}" == "10.0a 10.3a" ]] || die \
        "SM103 runtime ABI must attest exact CUDA arches 10.0a 10.3a"
    container_mounts+=",${overlay_wheel}:${overlay_wheel}"
    export AIC_OVERLAY_WHEEL="${overlay_wheel}"
    container_command='overlay_dir="${AIC_STAGING_ROOT}/overlay-${SLURM_PROCID}"; mkdir -p "${overlay_dir}"; python3 -m pip install --no-deps --target "${overlay_dir}" "${AIC_OVERLAY_WHEEL}" >/dev/null; export PYTHONPATH="${overlay_dir}:${AIC_REPO_DIR}:${PYTHONPATH:-}";'
else
    container_command='export PYTHONPATH="${AIC_REPO_DIR}:${PYTHONPATH:-}";'
fi

canary_flag=""
if [[ "${RUN_KIND}" == canary ]]; then
    canary_flag="--canary"
fi

export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL=1
export AIC_STAGING_ROOT="${staging_root}"
export AIC_REPO_DIR="${repo_dir}"
export AIC_OUTPUT_DIR="${output_dir}"
export AIC_VLLM_SOURCE_ROOT="${vllm_source_root}"
export AIC_IMAGE_DIGEST="${IMAGE_DIGEST}"
export AIC_RUNTIME_ABI_JSON="${runtime_abi_json}"
export AIC_GPUS_PER_NODE="${GPUS_PER_NODE}"
export AIC_BACKEND="${BACKEND}"
export AIC_CANARY_FLAG="${canary_flag}"
container_command+=' python3 -m collector.wideep.vllm.collect_moe_a2a --gpus-per-node "${AIC_GPUS_PER_NODE}" --backends "${AIC_BACKEND}" --output-path "${AIC_OUTPUT_DIR}" --vllm-source-root "${AIC_VLLM_SOURCE_ROOT}" --image-digest "${AIC_IMAGE_DIGEST}" --runtime-abi-json "${AIC_RUNTIME_ABI_JSON}" ${AIC_CANARY_FLAG}'

srun \
    --nodes="${NODE_NUM}" \
    --ntasks="${expected_ep}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --mpi=pmix \
    --container-image="${container_image}" \
    --container-mounts="${container_mounts}" \
    --container-workdir="${repo_dir}" \
    bash -lc "${container_command}"

parquet_path="${output_dir}/moe_a2a_perf.parquet"
sidecar_path="${output_dir}/collection_meta.yaml"
[[ -f "${parquet_path}" && -f "${sidecar_path}" ]] || die "collector did not finalize both formal artifacts"
if compgen -G "${output_dir}/errors_moe_a2a_vllm.rank*.json" >/dev/null; then
    die "formal job produced classified failure records"
fi

python3 - "${parquet_path}" "${sidecar_path}" "${output_dir}/artifact_checksums.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

paths = [Path(value) for value in sys.argv[1:3]]
checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
Path(sys.argv[3]).write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
PY

campaign_job_dir=$(safe_future_path \
    "campaign job output" \
    "${campaign_root}/${SYSTEM}/${RUN_KIND}/${NODE_NUM}n/${BACKEND}/job_${SLURM_JOB_ID}")
mkdir -p -- "${campaign_job_dir}"
campaign_job_dir=$(safe_existing_path "campaign job output" "${campaign_job_dir}")
cp -a -- "${output_dir}/." "${campaign_job_dir}/"
touch "${campaign_job_dir}/SUCCESS"
echo "Published validated ${SYSTEM} ${NODE_NUM}-node ${BACKEND} ${RUN_KIND} artifacts to ${campaign_job_dir}"
