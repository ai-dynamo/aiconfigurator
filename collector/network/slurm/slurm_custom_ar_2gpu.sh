#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH -o log_slurm_py/trtllm-bench-2gpu.out
#SBATCH -e log_slurm_py/trtllm-bench-2gpu.err
#SBATCH -J slurm_py_2gpu

export NCCL_DEBUG=ERROR
export OMPI_MCA_rmaps_base_oversubscribe=true
export NCCL_NET_GDR_C2C=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_SOCKET_IFNAME=bond0
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=bond0
export NCCL_NVLS_ENABLE=1

srun -l \
    --ntasks 2 --ntasks-per-node 2 \
    --gpus-per-task=1 --gpu-bind=single:1 \
    --container-image=/path/to/trtllm_aarch64_release_v1.3.0rc20.sqsh \
    --container-mounts=/dev:/dev,/path/to/aiconfigurator:/workspace/aiconfigurator \
    --export=ALL \
    --mpi=pmix python /workspace/aiconfigurator/collector/network/slurm/collect_allreduce.py
