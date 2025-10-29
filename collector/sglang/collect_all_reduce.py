# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AllReduce Performance Collector for SGLang

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
import sys
from argparse import ArgumentParser

# Add parent directory to path for helper import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pkg_resources
import torch
import torch.distributed as dist
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce

try:
    from helper import log_perf
except ModuleNotFoundError:
    from collector.helper import log_perf


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
    CUDA Graph based AllReduce benchmark method using SGLang's CustomAllreduce
    """
    # Setup distributed environment
    if use_slurm:
        # Use SLURM environment variables
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        # Use MPI environment variables
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", 0)))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", 0)))

    if world_size == 1:
        raise RuntimeError("Benchmark must run with world_size > 1")

    # Initialize SGLang distributed environment
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        # Use SGLang's init_distributed_environment
        if use_slurm:
            dist_init_method = "env://"
        else:
            # For MPI, use tcp with a port
            dist_init_method = "tcp://127.0.0.1:29500"

        init_distributed_environment(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=dist_init_method,
        )

    # Parse test range
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Benchmark parameters
    repeat_n = 5
    num_warmups = 3
    num_runs = 20

    size = min_size
    while size < max_size:
        input_shape = get_input_shape_and_comm_size(size)
        input_tensor = torch.ones(input_shape, dtype=torch_dtype, device="cuda")

        # Create CustomAllreduce instances for repeated measurement
        op_list = []
        for _ in range(repeat_n):
            custom_ar = CustomAllreduce(
                group=dist.group.WORLD,
                device=torch.cuda.current_device(),
            )
            # Dry run to initialize
            custom_ar(input_tensor)
            op_list.append(custom_ar)

        # Capture CUDA Graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for op in op_list:
                op(input_tensor)

        # Warmup and timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        for _ in range(num_warmups):
            g.replay()
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(num_runs):
            g.replay()
        end_event.record()
        torch.cuda.synchronize()

        latency = start_event.elapsed_time(end_event) / num_runs / repeat_n

        if rank == 0:
            print(f"Size: {size}, Latency: {latency:.4f} ms")

            # Get SGLang version
            try:
                sglang_version = pkg_resources.get_distribution("sglang").version
            except Exception:
                sglang_version = "unknown"

            # Use log_perf to save results
            log_perf(
                item_list=[
                    {
                        "allreduce_dtype": dtype,
                        "num_gpus": world_size,
                        "message_size": size,  # element count, not bytes
                        "latency": latency,
                    }
                ],
                framework="SGLang",
                version=sglang_version,
                device_name=torch.cuda.get_device_name(),
                op_name="all_reduce",
                kernel_source="CUDA_Graph",
                perf_filename=perf_filename,
            )

        size *= ratio

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dtype", "-t", default="float16", help="Data type for AllReduce (float16, float32, bfloat16)")
    parser.add_argument(
        "--range",
        "-r",
        default="128,1073741824,2",  # 128 elements to 1024M elements
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
