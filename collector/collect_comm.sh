#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Default backend
all_reduce_backend="trtllm"
device="cuda"
measure_power=false
power_test_duration=1.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all_reduce_backend)
            all_reduce_backend="$2"
            if [[ "$all_reduce_backend" != "trtllm" && "$all_reduce_backend" != "vllm" && "$all_reduce_backend" != "sglang" ]]; then
                echo "Error: --all_reduce_backend must be 'trtllm', 'vllm', or 'sglang'"
                echo "Usage: $0 [OPTIONS]"
                exit 1
            fi
            shift 2
            ;;
        --device)
            device="$2"
            if [[ "$device" != "cuda" && "$device" != "xpu" ]]; then
                echo "Error: --device must be 'cuda' or 'xpu'"
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
            echo "  --all_reduce_backend       Backend for AllReduce benchmark (default: trtllm)"
            echo "                             Choices: trtllm, vllm, sglang"
            echo "  --measure_power            Enable power monitoring during execution"
            echo "  --power_test_duration      Minimum test duration for power measurement in seconds (default: 1.0)"
            echo "  -h, --help                 Show this help message and exit"
            echo ""
            echo "Examples:"
            echo "  $0 --all_reduce_backend trtllm"
            echo "  $0 --measure_power --power_test_duration 2.0"
            echo "  $0 --all_reduce_backend vllm --measure_power"
            echo "  $0 --all_reduce_backend sglang"
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

if [[ "$device" == "cuda" ]]; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
elif [[ "$device" == "xpu" ]]; then
    GPU_COUNT=$(sycl-ls | grep '\[level_zero:gpu\]' | wc -l)
fi

echo "Found $GPU_COUNT GPUs."
if [ "$GPU_COUNT" -ge 8 ]; then
    gpu_count_list=(2 4 8)
elif [ "$GPU_COUNT" -ge 4 ]; then
    gpu_count_list=(2 4)
elif [ "$GPU_COUNT" -ge 2 ]; then
    gpu_count_list=(2)
else
    echo "Error: single GPU detected."
    exit 1
fi

# NCCL
if [[ "$device" == "cuda" ]]; then
    nccl_ops=("all_gather" "alltoall" "reduce_scatter" "all_reduce")
    dtypes=("half" "int8")

    for n in "${gpu_count_list[@]}"; do
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
else
    echo "Skipping NCCL benchmarks because device is $device"
fi

echo "Running AllReduce Benchmarks with $all_reduce_backend backend..."

if [[ "$all_reduce_backend" == "trtllm" ]]; then
    # TRTLLM allreduce (CUDA Graph based)
    for n in "${gpu_count_list[@]}"; do
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
    for n in "${gpu_count_list[@]}"; do
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
elif [[ "$all_reduce_backend" == "sglang" ]]; then
    # SGLang allreduce implementation
    for n in "${gpu_count_list[@]}"; do
        echo "Running SGLang AllReduce benchmark with $n GPUs"
        if [[ "$measure_power" == "true" ]]; then
            torchrun --nproc_per_node=$n collect_all_reduce.py --backend sglang \
                --perf-filename "custom_allreduce_perf.txt" \
                --measure_power --power_test_duration_sec "$power_test_duration"
        else
            torchrun --nproc_per_node=$n collect_all_reduce.py --backend sglang \
                --perf-filename "custom_allreduce_perf.txt"
        fi
    done
fi

echo ""
echo "All benchmarks completed!"
