#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: ./collect_comm.sh [backend1 backend2 ...]
# Examples:
#   ./collect_comm.sh              # Run all available backends
#   ./collect_comm.sh trtllm       # Run only trtllm
#   ./collect_comm.sh trtllm sglang # Run trtllm and sglang

# NCCL
num_gpus_nccl=(2 4 8)
nccl_ops=("all_gather" "alltoall" "reduce_scatter" "all_reduce")
dtypes=("half" "int8")

for n in "${num_gpus_nccl[@]}"; do
    for op in "${nccl_ops[@]}"; do
        for dtype in "${dtypes[@]}"; do
            python3 collect_nccl.py -n "$n" -NCCL "$op" --dtype "$dtype"
        done
    done
done

# Backend-specific AllReduce (CUDA Graph based)
num_gpus_allreduce=(2 4 8)

# If command line arguments provided, use them; otherwise use all available backends
if [ $# -gt 0 ]; then
    backends=("$@")
    echo "Running specified backends: ${backends[*]}"
else
    backends=("trtllm" "sglang" "vllm_v1")
    echo "Running all available backends: ${backends[*]}"
fi

for backend in "${backends[@]}"; do
    # Check if collect_all_reduce.py exists for this backend
    if [ -f "${backend}/collect_all_reduce.py" ]; then
        echo ""
        echo "====================================="
        echo "Running AllReduce benchmark for backend: ${backend}"
        echo "====================================="
        
        for n in "${num_gpus_allreduce[@]}"; do
            echo "Running ${backend} AllReduce benchmark with $n GPUs using CUDA Graph method"
            mpirun -n "$n" --allow-run-as-root python3 "${backend}/collect_all_reduce.py" \
                --perf-filename "custom_allreduce_perf.txt"
        done
        
        echo ""
    else
        echo ""
        echo "⚠️  Skipping ${backend}: collect_all_reduce.py not found in ${backend}/"
        echo ""
    fi
done

echo "====================================="
echo "Communication benchmarks completed!"
echo "====================================="
