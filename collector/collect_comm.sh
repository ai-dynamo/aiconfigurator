#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NCCL
num_gpus_nccl=(2 4 8)
nccl_ops=("all_gather" "alltoall" "reduce_scatter" "all_reduce")
dtypes=("half" "int8")

for n in "${num_gpus_nccl[@]}"; do
    for op in "${nccl_ops[@]}"; do
        for dtype in "${dtypes[@]}"; do
            python3 collect_NCCL.py -n "$n" -NCCL "$op" --dtype "$dtype"
        done
    done
done

# TRTLLM allreduce
num_gpus_trtllm=(2 4 8)
for n in "${num_gpus_trtllm[@]}"; do
    mpirun -n "$n" --allow-run-as-root python collect_all_reduce.py
done
