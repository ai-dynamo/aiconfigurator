# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AllReduce Performance Collector

This script uses CUDA Graph based benchmarking for AllReduce operations,
providing efficient and accurate performance measurements.

Usage:
    # With MPI
    mpirun -n 4 python collect_all_reduce.py

    # With SLURM
    python collect_all_reduce.py --use-slurm

    # Custom range and output file
    python collect_all_reduce.py --range "128,1000000,2" --perf-filename "my_perf.txt"
"""

import os
from argparse import ArgumentParser

# isort: off
import torch

# isort: on
import tensorrt_llm as tllm
try:
    from cuda import cudart
except:
    from cuda.bindings import runtime as cudart
from tensorrt_llm import Mapping
from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp
from tensorrt_llm._torch.distributed import AllReduceParams as TorchAllReduceParams
from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST, mpi_comm
from tensorrt_llm.functional import AllReduceStrategy

from helper import log_perf


def get_input_shape_and_comm_size(size, token_dim=4096):
    """Convert size to appropriate input shape for AllReduce operations"""
    if size <= token_dim:
        return [1, size]
    else:
        num_token = size // token_dim
        return [num_token, token_dim]


def allreduce_benchmark(
    dtype: str,
    test_range: str = "128,1073741824,2",
    use_slurm: bool = False,
    perf_filename: str = "custom_allreduce_perf.txt",
):
    """
    CUDA Graph based AllReduce benchmark method
    """
    # Setup distributed environment
    if use_slurm:
        # Use SLURM environment variables
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["RANK"])
        gpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        # Use MPI
        world_size = tllm.mpi_world_size()
        rank = tllm.mpi_rank()
        local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)
        local_rank = local_comm.Get_rank()
        gpus_per_node = local_comm.Get_size()

    if world_size == 1:
        raise RuntimeError("Benchmark must run with world_size > 1")

    torch.cuda.set_device(local_rank)
    cudart.cudaSetDevice(local_rank)
    mapping = Mapping(world_size=world_size, rank=rank, gpus_per_node=gpus_per_node, tp_size=world_size)

    # Parse test range
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)

    # AllReduce parameters
    all_reduce_params = TorchAllReduceParams(
        strategy=AllReduceStrategy.AUTO,
        fusion_op=AllReduceFusionOp.NONE,
        residual=None,
        norm_weight=None,
        scale=None,
        bias=None,
        eps=1e-6,
    )

    # Benchmark parameters
    repeat_n = 5
    num_warmups = 3
    num_runs = 20

    size = min_size
    while size < max_size:
        input_shape = get_input_shape_and_comm_size(size)
        input_tensor = torch.ones(input_shape, dtype=torch_dtype, device="cuda")

        op_list = []
        for i in range(repeat_n):
            allreduce = AllReduce(mapping=mapping).cuda()
            allreduce(input_tensor, all_reduce_params=all_reduce_params)  # dry run to init
            op_list.append(allreduce)

        # Capture CUDA Graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for op in op_list:
                op(input_tensor, all_reduce_params=all_reduce_params)

        # Warmup and timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        for i in range(num_warmups):
            g.replay()
        torch.cuda.synchronize()

        start_event.record()
        for i in range(num_runs):
            g.replay()
        end_event.record()
        torch.cuda.synchronize()

        latency = start_event.elapsed_time(end_event) / num_runs / repeat_n

        if rank == 0 and local_rank == 0:
            print(f"Size: {size}, Latency: {latency:.4f} ms")

            # Get TensorRT-LLM version
            trtllm_version = tllm.__version__ if hasattr(tllm, "__version__") else "unknown"

            # Use log_perf directly, similar to collect_nccl.py
            log_perf(
                item_list=[
                    {
                        "allreduce_dtype": dtype,
                        "num_gpus": world_size,
                        "message_size": size,  # element count, not bytes
                        "latency": latency,
                    }
                ],
                framework="TRTLLM",
                version=trtllm_version,
                device_name=torch.cuda.get_device_name(),
                op_name="all_reduce",
                kernel_source="TRTLLM",
                perf_filename=perf_filename,
            )

        size *= ratio


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", "-t", default="float16")
    parser.add_argument(
        "--range",
        "-r",
        default="128,1073741824,2",  # 128B to 1024MB
        help="min_size,max_size,multiplicative_ratio",
    )
    parser.add_argument("--use-slurm", action="store_true", help="Use SLURM environment variables instead of MPI")
    parser.add_argument(
        "--perf-filename",
        "-f",
        default="custom_allreduce_perf.txt",
        help="Output performance file name",
    )
    args = parser.parse_args()

    allreduce_benchmark(args.dtype, args.range, args.use_slurm, args.perf_filename)
