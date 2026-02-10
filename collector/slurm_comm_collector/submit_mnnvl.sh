#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# WideEP AlltoAll (MNNVL) Benchmark - Submit multiple parallel jobs for 2, 4, 8, 16 GPUs
# All results append to the same output file
#
# Usage: bash submit_slurm.sh

SCRIPT_DIR="${HOME}/repo/aiconfigurator/collector/slurm_comm_collector"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc5}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}/repo/aiconfigurator:${HOME}/repo/aiconfigurator}"
ACCOUNT="${ACCOUNT:-coreai_tritoninference_triton3}"
PARTITION="${PARTITION:-gb200}"

COLLECTOR_SCRIPT="${SCRIPT_DIR}/collect_trtllm_alltoall.py"
OUTPUT_FILE="${SCRIPT_DIR}/results/wideep_alltoall_perf.txt"

mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/errors" "${SCRIPT_DIR}/results"

echo "=========================================="
echo "WideEP AlltoAll (MNNVL) Multi-Scale Benchmark"
echo "Submitting parallel jobs for: 2, 4, 8, 16, 32, 48, 64, 72 GPUs"
echo "Output: ${OUTPUT_FILE}"
echo "=========================================="

# GPU configurations to test
GPU_COUNTS=(2 4 8 16 32 48 64 72)
GPUS_PER_NODE=4

for NUM_GPUS in "${GPU_COUNTS[@]}"; do
    # Calculate nodes needed
    if [ ${NUM_GPUS} -le ${GPUS_PER_NODE} ]; then
        NUM_NODES=1
        TASKS_PER_NODE=${NUM_GPUS}
    else
        NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
        TASKS_PER_NODE=${GPUS_PER_NODE}
    fi
    
    JOB_NAME="${ACCOUNT}-wideep.${NUM_GPUS}gpu"
    
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
        --wrap="export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n 1); \
                export MASTER_PORT=29500; \
                srun \
                    --container-image=${CONTAINER_IMAGE} \
                    --container-mounts=${CONTAINER_MOUNTS} \
                    --mpi=pmix \
                    -- python ${COLLECTOR_SCRIPT} --output ${OUTPUT_FILE}"
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Check status: squeue -u \$USER"
echo "Results: ${OUTPUT_FILE}"
echo "=========================================="
