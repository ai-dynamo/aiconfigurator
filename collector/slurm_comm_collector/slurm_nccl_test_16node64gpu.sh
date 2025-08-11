#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH -N 16
#SBATCH --gpus 64
#SBATCH --ntasks-per-node=4
#SBATCH -o log_nccl/16node64gpu.out
#SBATCH -e error_nccl/16node64gpu.err
#SBATCH -J 16node64gpu

export NCCL_DEBUG=ERROR
export NCCL_NET_GDR_C2C=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_NVLS_ENABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_MIN_CTAS=16
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
# export NCCL_ALGO=ring

srun -l \
    --ntasks 64 --ntasks-per-node 4 \
    --container-image=/path/to/trtllm_aarch64_release_v0.20.sqsh \
    --container-mounts=/dev:/dev,${HOME}:${HOME},/shared_data/kimi:/kimi \
    --container-workdir=/kimi \
    --export=ALL \
    --mpi=pmix bash -c "/path/to/nccl/nccl-tests/build/all_reduce_perf -b 256 -e 8g -d half -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /path/to/nccl/nccl-tests/build/all_reduce_perf -b 256 -e 8g -d int8 -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /path/to/nccl/nccl-tests/build/alltoall_perf -b 256 -e 8g -d half -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /path/to/nccl/nccl-tests/build/alltoall_perf -b 256 -e 8g -d int8 -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /path/to/nccl/nccl-tests/build/reduce_scatter_perf -b 256 -e 8g -d half -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /path/to/nccl/nccl-tests/build/reduce_scatter_perf -b 256 -e 8g -d int8 -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /path/to/nccl/nccl-tests/build/all_gather_perf -b 256 -e 8g -d half -f2 -g 1 -w 40 -a 1 -n 60 -c 0; \
                        /path/to/nccl/nccl-tests/build/all_gather_perf -b 256 -e 8g -d int8 -f2 -g 1 -w 40 -a 1 -n 60 -c 0"
