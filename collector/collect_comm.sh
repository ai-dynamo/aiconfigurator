#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Default backend
all_reduce_backend="trtllm"
measure_power=false
power_test_duration=1.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all_reduce_backend)
            all_reduce_backend="$2"
            if [[ "$all_reduce_backend" != "trtllm" && "$all_reduce_backend" != "vllm" ]]; then
                echo "Error: --all_reduce_backend must be either 'trtllm' or 'vllm'"
                echo "Usage: $0 [OPTIONS]"
                exit 1
            fi
            shift 2
            ;;
        --measure_power)
            measure_power=true
            shift
            ;;
        --power_test_duration)
            power_test_duration="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all_reduce_backend  Backend for AllReduce benchmark (default: trtllm)"
            echo "                        Choices: trtllm, vllm"
            echo "  --measure_power       Enable power monitoring during execution"
            echo "  --power_test_duration Minimum test duration for power measurement in seconds (default: 1.0)"
            echo "  -h, --help           Show this help message and exit"
            echo ""
            echo "Examples:"
            echo "  $0 --all_reduce_backend trtllm"
            echo "  $0 --measure_power --power_test_duration 2.0"
            echo "  $0 --all_reduce_backend vllm --measure_power"
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Run '$0 --help' for more information"
            exit 1
            ;;
    esac
done

echo "Running benchmarks with all_reduce_backend: $all_reduce_backend"
if [[ "$measure_power" == "true" ]]; then
    echo "Power monitoring: ENABLED (duration: ${power_test_duration}s)"
else
    echo "Power monitoring: DISABLED"
fi
echo "================================================"

# NCCL
num_gpus_nccl=(2 4 8)
nccl_ops=("all_gather" "alltoall" "reduce_scatter" "all_reduce")
dtypes=("half" "int8")

for n in "${num_gpus_nccl[@]}"; do
    for op in "${nccl_ops[@]}"; do
        for dtype in "${dtypes[@]}"; do
            if [[ "$measure_power" == "true" ]]; then
                python3 collect_nccl.py -n "$n" -NCCL "$op" --dtype "$dtype" \
                    --measure_power --power_test_duration_sec "$power_test_duration"
            else
                python3 collect_nccl.py -n "$n" -NCCL "$op" --dtype "$dtype"
            fi
        done
    done
done

echo "Running AllReduce Benchmarks with $all_reduce_backend backend..."
num_gpus_allreduce=(2 4 8)

if [[ "$all_reduce_backend" == "trtllm" ]]; then
    # TRTLLM allreduce (CUDA Graph based)
    for n in "${num_gpus_allreduce[@]}"; do
        echo "Running TRTLLM AllReduce benchmark with $n GPUs using CUDA Graph method"
        if [[ "$measure_power" == "true" ]]; then
            mpirun -n "$n" --allow-run-as-root python3 collect_all_reduce.py \
                --perf-filename "custom_allreduce_perf.txt" \
                --measure_power --power_test_duration_sec "$power_test_duration"
        else
            mpirun -n "$n" --allow-run-as-root python3 collect_all_reduce.py \
                --perf-filename "custom_allreduce_perf.txt"
        fi
    done
elif [[ "$all_reduce_backend" == "vllm" ]]; then
    # VLLM allreduce implementation
    for n in "${num_gpus_allreduce[@]}"; do
        echo "Running VLLM AllReduce benchmark with $n GPUs"
        if [[ "$measure_power" == "true" ]]; then
            torchrun --nproc_per_node=$n collect_all_reduce.py --backend vllm \
                --perf-filename "custom_allreduce_perf.txt" \
                --measure_power --power_test_duration_sec "$power_test_duration"
        else
            torchrun --nproc_per_node=$n collect_all_reduce.py --backend vllm \
                --perf-filename "custom_allreduce_perf.txt"
        fi
    done
fi

echo ""
echo "All benchmarks completed!"
