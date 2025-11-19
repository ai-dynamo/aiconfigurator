#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Default backend
all_reduce_backend="trtllm"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all_reduce_backend)
            all_reduce_backend="$2"
            if [[ "$all_reduce_backend" != "trtllm" && "$all_reduce_backend" != "vllm" ]]; then
                echo "Error: --all_reduce_backend must be either 'trtllm' or 'vllm'"
                echo "Usage: $0 [--all_reduce_backend trtllm|vllm]"
                exit 1
            fi
            shift 2
            ;;
        --measure_power)
            measure_power=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--all_reduce_backend trtllm|vllm] [--measure_power]"
            echo ""
            echo "Options:"
            echo "  --all_reduce_backend  Backend for AllReduce benchmark (default: trtllm)"
            echo "                        Choices: trtllm, vllm"
            echo "  --measure_power       Measure power consumption if given"
            echo "  -h, --help            Show this help message and exit"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Usage: $0 [--all_reduce_backend trtllm|vllm] [--measure_power]"
            exit 1
            ;;
    esac
done

echo "Running benchmarks with all_reduce_backend: $all_reduce_backend"
echo "================================================"

# NCCL
num_gpus_nccl=(2 4 8)
nccl_ops=("all_gather" "alltoall" "reduce_scatter" "all_reduce")
dtypes=("half" "int8")

for n in "${num_gpus_nccl[@]}"; do
    for op in "${nccl_ops[@]}"; do
        for dtype in "${dtypes[@]}"; do
            python3 collect_nccl.py -n "$n" -NCCL "$op" --dtype "$dtype" ${measure_power:+--measure_power}
        done
    done
done

echo "Running AllReduce Benchmarks with $all_reduce_backend backend..."
num_gpus_allreduce=(2 4 8)

if [[ "$all_reduce_backend" == "trtllm" ]]; then
    # TRTLLM allreduce (CUDA Graph based)
    for n in "${num_gpus_allreduce[@]}"; do
        echo "Running TRTLLM AllReduce benchmark with $n GPUs using CUDA Graph method"
        mpirun -n "$n" --allow-run-as-root python3 collect_all_reduce.py \
            --perf-filename "custom_allreduce_perf.txt" ${measure_power:+--measure_power}
    done
elif [[ "$all_reduce_backend" == "vllm" ]]; then
    # VLLM allreduce implementation
    for n in "${num_gpus_allreduce[@]}"; do
        echo "Running VLLM AllReduce benchmark with $n GPUs"
        torchrun --nproc_per_node=$n collect_all_reduce.py --backend vllm \
            --perf-filename "custom_allreduce_perf.txt" ${measure_power:+--measure_power}
    done
fi

echo ""
echo "All benchmarks completed!"
