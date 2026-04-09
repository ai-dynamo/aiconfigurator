#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH -N 4
#SBATCH --gpus 16
#SBATCH --ntasks-per-node=4
#SBATCH -o log_rccl/4node16gpu.out
#SBATCH -e error_rccl/4node16gpu.err
#SBATCH -J 4node16gpu

export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME=enp81s0f1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export UCX_NET_DEVICES=benic1p1:1,benic2p1:1,benic3p1:1,benic4p1:1,benic5p1:1,benic6p1:1,benic7p1:1,benic8p1:1
# export NCCL_ALGO=ring

srun -l \
    --ntasks 16 --ntasks-per-node 4 \
    --container-image=lmsysorg/sglang:v0.5.9-rocm720-mi35x \
    --container-mounts=/dev:/dev,${HOME}:${HOME},/apps/theresa:/workspace \
    --container-workdir=/workspace \
    --export=ALL \
    --mpi=pmix bash -c "/workspace/rccl-tests/build/all_reduce_perf -b 256 -e 8g -d half -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /workspace/rccl-tests/build/all_reduce_perf -b 256 -e 8g -d int8 -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /workspace/rccl-tests/build/alltoall_perf -b 256 -e 8g -d half -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /workspace/rccl-tests/build/alltoall_perf -b 256 -e 8g -d int8 -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /workspace/rccl-tests/build/reduce_scatter_perf -b 256 -e 8g -d half -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /workspace/rccl-tests/build/reduce_scatter_perf -b 256 -e 8g -d int8 -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /workspace/rccl-tests/build/all_gather_perf -b 256 -e 8g -d half -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /workspace/rccl-tests/build/all_gather_perf -b 256 -e 8g -d int8 -f2 -g 1 -w 40 -a 1 -n 60 -c 0"
