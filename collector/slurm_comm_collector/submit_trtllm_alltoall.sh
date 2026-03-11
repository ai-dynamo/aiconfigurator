#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# TensorRT-LLM MoE AlltoAll Benchmark - Submit multiple parallel jobs
# All results append to the same output file
#
# Usage:
#   bash submit_trtllm_alltoall.sh                                                # default: NVLinkTwoSided, 2,4,8,16,32,48,64,72 GPUs, 4 GPUs/node
#   bash submit_trtllm_alltoall.sh --kernel-source NVLinkOneSided --gpu-list 2,4  # NVLinkOneSided, 2 and 4 GPUs
#   bash submit_trtllm_alltoall.sh --gpu-list 4,8,16                              # NVLinkTwoSided, 4,8,16 GPUs

usage() {
    cat <<EOF
Usage: bash $(basename "$0") [OPTIONS]

Options:
  --kernel-source <name>   Communication strategy: NVLinkTwoSided or NVLinkOneSided
                           (default: NVLinkTwoSided)
  --gpu-list <list>        Comma-separated GPU counts to benchmark
                           (default: 2,4,8,16,32,48,64,72)
  --gpus-per-node <n>      GPUs per node for node/task calculation
                           (default: \${GPUS_PER_NODE:-4})
  -h, --help               Show this help message

Environment variables:
  GPUS_PER_NODE            Override default GPUs per node (default: 4)
  CONTAINER_IMAGE          Container image to use
  CONTAINER_MOUNTS         Container mount paths
  ACCOUNT                  Slurm account
  PARTITION                Slurm partition

Examples:
  bash $(basename "$0")
  bash $(basename "$0") --kernel-source NVLinkOneSided --gpu-list 2,4
  bash $(basename "$0") --gpus-per-node 8 --gpu-list 8,16,32
EOF
}

SCRIPT_DIR="${HOME}/repo/aiconfigurator/collector/slurm_comm_collector"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}/repo/aiconfigurator:${HOME}/repo/aiconfigurator}"
ACCOUNT="${ACCOUNT:-coreai_tritoninference_triton3}"
PARTITION="${PARTITION:-gb200}"

COLLECTOR_SCRIPT="${SCRIPT_DIR}/collect_trtllm_alltoall.py"
OUTPUT_FILE="${SCRIPT_DIR}/results/trtllm_alltoall_perf.txt"

# Defaults
KERNEL_SOURCE="NVLinkTwoSided"
GPU_LIST="2,4,8,16,32,48,64,72"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel-source)  KERNEL_SOURCE="$2"; shift 2 ;;
        --gpu-list)       GPU_LIST="$2";      shift 2 ;;
        --gpus-per-node)  GPUS_PER_NODE="$2"; shift 2 ;;
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
echo "TensorRT-LLM MoE AlltoAll Benchmark [${KERNEL_SOURCE}]"
echo "Submitting parallel jobs for: ${GPU_LIST} GPUs"
echo "Output: ${OUTPUT_FILE}"
echo "=========================================="

for NUM_GPUS in "${GPU_COUNTS[@]}"; do
    # Calculate nodes needed
    if [ ${NUM_GPUS} -le ${GPUS_PER_NODE} ]; then
        NUM_NODES=1
        TASKS_PER_NODE=${NUM_GPUS}
    else
        NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
        TASKS_PER_NODE=${GPUS_PER_NODE}
    fi
    
    JOB_NAME="${ACCOUNT}-alltoall-${KERNEL_SOURCE}.${NUM_GPUS}gpu"
    
    echo "Submitting: ${JOB_NAME} (${NUM_NODES} nodes, ${NUM_GPUS} GPUs)"
    
    # get full rack
    if [ "${NUM_GPUS}" -eq 72 ]; then
        SBATCH_EXTRA_ARGS="--segment 18"
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
                srun \
                    --container-image=\"${CONTAINER_IMAGE}\" \
                    --container-mounts=\"${CONTAINER_MOUNTS}\" \
                    --mpi=pmix \
                    -- python \"${COLLECTOR_SCRIPT}\" --kernel-source \"${KERNEL_SOURCE}\" --output \"${OUTPUT_FILE}\""
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Check status: squeue -u \$USER"
echo "Results: ${OUTPUT_FILE}"
echo "=========================================="
