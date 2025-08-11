#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH -N 1
#SBATCH --gpus 2
#SBATCH --ntasks-per-node=2
#SBATCH -o log_slurm_py/trtllm-bench.out
#SBATCH -e log_slurm_py/trtllm-bench.err
#SBATCH -J slurm_py

export NCCL_DEBUG=ERROR
export OMPI_MCA_rmaps_base_oversubscribe=true
export NCCL_NET_GDR_C2C=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_SOCKET_IFNAME=bond0
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
export NCCL_NVLS_ENABLE=1
export TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED=0

srun -l \
    --ntasks 2 --ntasks-per-node 2 \
    --container-image=/path/to/trtllm_aarch64_release_v1.0.0rc2.sqsh \
    --container-mounts=/dev:/dev,${HOME}:${HOME},/path/to/:/kimi \
    --export=ALL \
    --mpi=pmix python /path/to/collect_allreduce.py --mca btl tcp,self --mca btl_tcp_if_include bond0
