#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Multi-node custom allreduce sweep: TP=16 across 4 nodes (GB200/GB300: 4 GPUs/node,
# so TP>4 always spans nodes). Env template mirrors slurm_nccl_test_4node16gpu.sh;
# adjust NCCL_IB_HCA / NCCL_SOCKET_IFNAME / container-image to the cluster.
#SBATCH -N 4
#SBATCH --gpus 16
#SBATCH --ntasks-per-node=4
#SBATCH -o log_slurm_py/trtllm-bench-16gpu.out
#SBATCH -e log_slurm_py/trtllm-bench-16gpu.err
#SBATCH -J slurm_py_16gpu

export NCCL_DEBUG=ERROR
export OMPI_MCA_rmaps_base_oversubscribe=true
export NCCL_NET_GDR_C2C=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
export NCCL_SOCKET_IFNAME=bond0
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1
# Open MPI MCA settings belong in the environment: trailing `--mca` args land on
# the python command (collect_allreduce.py does not parse argv) and are ignored.
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=bond0
# Cross-node MNNVL allreduce (same recipe as the NCCL multi-node sweep)
export NCCL_NVLS_ENABLE=1
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_HOST_ENABLE=0
export TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED=0

# collect_allreduce.py reads SLURM_NTASKS / SLURM_NTASKS_PER_NODE / SLURM_LOCALID;
# the per-task rank comes from RANK if set, else SLURM_PROCID (what srun exports).
srun -l \
    --ntasks 16 --ntasks-per-node 4 \
    --container-image=/path/to/trtllm_aarch64_release_v1.0.0rc2.sqsh \
    --container-mounts=/dev:/dev,${HOME}:${HOME},/path/to/:/kimi \
    --export=ALL \
    --mpi=pmix python /path/to/collect_allreduce.py
