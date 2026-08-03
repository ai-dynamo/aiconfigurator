#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# MoE all-to-all (DeepEP HT + LL) benchmark - submit multiple parallel jobs.
# Each world size is its own Slurm job with its own output directory: the
# collector finalizes moe_a2a_perf.parquet and attests a world-specific
# collection_meta.yaml per run, so jobs cannot share one file across worlds.
# Within a job, every case appends to that job's single moe_a2a_perf.txt via
# helper.log_perf's lockfile.
#
# Usage:
#   bash submit_moe_a2a.sh                                  # default: 8,16,32,48,64,72 GPUs, 4 GPUs/node
#   bash submit_moe_a2a.sh --gpu-list 8,16                  # only 8 and 16 GPUs
#   bash submit_moe_a2a.sh --gpus-per-node 8 --gpu-list 16  # 8-GPU nodes

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [OPTIONS]

Options:
  --gpu-list <list>        Comma-separated GPU counts to benchmark; each must be
                           a multiple of --gpus-per-node (default: 8,16,32,48,64,72)
  --gpus-per-node <n>      GPUs per node for node/task calculation
                           (default: \${GPUS_PER_NODE:-4})
  --modes <list>           DeepEP kernel families to collect
                           (default: deepep_ht,deepep_ll)
  -h, --help               Show this help message

Environment variables:
  GPUS_PER_NODE            Override default GPUs per node (default: 4)
  CONTAINER_IMAGE          Container image (default: the manifest wideep_sglang pin)
  CONTAINER_MOUNTS         Container mount paths
  ACCOUNT                  Slurm account
  PARTITION                Slurm partition

Examples:
  bash $(basename "$0")
  bash $(basename "$0") --gpu-list 8,16
  bash $(basename "$0") --gpus-per-node 8 --gpu-list 16,32 --modes deepep_ht
EOF
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# Default image = the collector/framework_manifest.yaml `wideep_sglang` pin
# (images.grace_blackwell — this launcher targets the GB200 partition below).
# The value is duplicated here because login nodes cannot be assumed to have
# the repo's Python deps; it is kept in sync with the manifest by
# tests/unit/collector/test_network_layout.py::test_submit_moe_a2a_default_image_matches_manifest.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-deepseek-v4-grace-blackwell}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${REPO_DIR}:${REPO_DIR}}"
ACCOUNT="${ACCOUNT:-coreai_tritoninference_triton3}"
PARTITION="${PARTITION:-gb200}"

# Defaults
GPU_LIST="8,16,32,48,64,72"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MODES="deepep_ht,deepep_ll"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-list)       GPU_LIST="$2";      shift 2 ;;
        --gpus-per-node)  GPUS_PER_NODE="$2"; shift 2 ;;
        --modes)          MODES="$2";         shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "Error: Unknown option: $1"; echo ""; usage; exit 1 ;;
    esac
done

# Validate GPUS_PER_NODE is a positive integer
if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: GPUS_PER_NODE must be a positive integer, got '${GPUS_PER_NODE}'"
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/errors" "${SCRIPT_DIR}/results"

# Convert comma-separated list to array
IFS=',' read -ra GPU_COUNTS <<< "${GPU_LIST}"

echo "=========================================="
echo "MoE all-to-all (DeepEP) benchmark [${MODES}]"
echo "Submitting parallel jobs for: ${GPU_LIST} GPUs"
echo "Container image: ${CONTAINER_IMAGE}"
echo "Results: ${SCRIPT_DIR}/results/moe_a2a_<N>gpu/"
echo "=========================================="

for NUM_GPUS in "${GPU_COUNTS[@]}"; do
    # The collector derives node_num = WORLD_SIZE // gpus_per_node and raises
    # on a non-integral node count, so reject bad worlds before submission.
    if [ $((NUM_GPUS % GPUS_PER_NODE)) -ne 0 ]; then
        echo "Error: ${NUM_GPUS} GPUs is not a multiple of ${GPUS_PER_NODE} GPUs/node"
        exit 1
    fi
    NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
    TASKS_PER_NODE=${GPUS_PER_NODE}

    JOB_NAME="${ACCOUNT}-moe_a2a.${NUM_GPUS}gpu"
    OUTPUT_DIR="${SCRIPT_DIR}/results/moe_a2a_${NUM_GPUS}gpu"
    mkdir -p "${OUTPUT_DIR}"

    echo "Submitting: ${JOB_NAME} (${NUM_NODES} nodes, ${NUM_GPUS} GPUs)"

    # A 72-GPU world is a full GB200 NVL72 rack: pin the allocation into one
    # segment so cross-rack links never carry the measured traffic.
    if [ "${NUM_GPUS}" -eq 72 ]; then
        SBATCH_EXTRA_ARGS="--segment ${NUM_NODES}"
    else
        SBATCH_EXTRA_ARGS=""
    fi

    sbatch \
        --job-name="${JOB_NAME}" \
        --nodes=${NUM_NODES} \
        --ntasks=${NUM_GPUS} \
        --ntasks-per-node=${TASKS_PER_NODE} \
        --account=${ACCOUNT} \
        --partition=${PARTITION} \
        --output="${SCRIPT_DIR}/logs/${JOB_NAME}_%j.out" \
        --error="${SCRIPT_DIR}/errors/${JOB_NAME}_%j.err" \
        ${SBATCH_EXTRA_ARGS} \
        --wrap="export MASTER_ADDR=\"\$(scontrol show hostname \"\$SLURM_NODELIST\" | head -n 1)\"; \
                export MASTER_PORT=\"29500\"; \
                export PYTHONPATH=\"${REPO_DIR}\"; \
                srun \
                    --container-image=\"${CONTAINER_IMAGE}\" \
                    --container-mounts=\"${CONTAINER_MOUNTS}\" \
                    --mpi=pmix \
                    -- python -m collector.wideep.sglang.collect_moe_a2a \
                        --gpus-per-node \"${TASKS_PER_NODE}\" \
                        --modes \"${MODES}\" \
                        --output-path \"${OUTPUT_DIR}\""
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Check status: squeue -u \$USER"
echo "Results: ${SCRIPT_DIR}/results/moe_a2a_<N>gpu/moe_a2a_perf.parquet (+ collection_meta.yaml)"
echo "=========================================="
