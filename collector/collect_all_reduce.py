# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gneeral-purpose AllReduce Performance Collector

Suppported Backends:
    TensorRT-LLM
    vLLM

This script uses CUDA Graph based benchmarking for AllReduce operations,
supporting both TensorRT-LLM and vLLM backends.

Usage:
    # With MPI for TensorRT-LLM
    mpirun -n 4 python collect_all_reduce.py --backend trtllm

    # With vLLM (requires appropriate environment setup)
    torchrun --nproc_per_node=8 collect_all_reduce.py --backend vllm

    # With SLURM
    python collect_all_reduce.py --use-slurm

    # Custom range and output file
    python collect_all_reduce.py --range "128,1000000,2" --perf-filename "my_perf.txt"
"""

import os
import sys
from argparse import ArgumentParser
from typing import Optional

# isort: off
import torch

# isort: on


def get_input_shape_and_comm_size(size, token_dim=4096):
    """Convert size to appropriate input shape for AllReduce operations"""
    if size <= token_dim:
        return [1, size]
    else:
        num_token = size // token_dim
        return [num_token, token_dim]


def import_trtllm():
    """Import TensorRT-LLM modules"""
    try:
        import tensorrt_llm as tllm
        from cuda import cudart
        from tensorrt_llm import Mapping
        from tensorrt_llm._torch.distributed import AllReduce, AllReduceFusionOp
        from tensorrt_llm._torch.distributed import AllReduceParams as TorchAllReduceParams
        from tensorrt_llm._utils import OMPI_COMM_TYPE_HOST, mpi_comm
        from tensorrt_llm.functional import AllReduceStrategy

        return {
            "tllm": tllm,
            "cudart": cudart,
            "Mapping": Mapping,
            "AllReduce": AllReduce,
            "AllReduceFusionOp": AllReduceFusionOp,
            "TorchAllReduceParams": TorchAllReduceParams,
            "OMPI_COMM_TYPE_HOST": OMPI_COMM_TYPE_HOST,
            "mpi_comm": mpi_comm,
            "AllReduceStrategy": AllReduceStrategy,
        }
    except ImportError as e:
        print(f"Failed to import TensorRT-LLM modules: {e}")
        print("Please ensure TensorRT-LLM is installed and PYTHONPATH is set correctly")
        sys.exit(1)


def benchmark_trtllm_allreduce(
    dtype: str,
    test_range: str,
    world_size: int,
    rank: int,
    use_slurm: bool,
    perf_filename: str,
    measure_power: bool = False,
):
    """Benchmark TensorRT-LLM AllReduce implementation"""
    trtllm_mods = import_trtllm()
    tllm = trtllm_mods["tllm"]

    if use_slurm:
        gpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        local_comm = trtllm_mods["mpi_comm"]().Split_type(split_type=trtllm_mods["OMPI_COMM_TYPE_HOST"])
        local_rank = local_comm.Get_rank()
        gpus_per_node = local_comm.Get_size()

    torch.cuda.set_device(local_rank)
    trtllm_mods["cudart"].cudaSetDevice(local_rank)
    mapping = trtllm_mods["Mapping"](world_size=world_size, rank=rank, gpus_per_node=gpus_per_node, tp_size=world_size)

    # Initialize NVML power monitor if power measurement is enabled
    power_monitor = None
    if measure_power:
        try:
            from nvml_power_monitor import NVMLPowerMonitor

            power_monitor = NVMLPowerMonitor(gpu_indices=[local_rank])
            if rank == 0:
                print("NVML power monitoring enabled on all ranks")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to initialize NVML power monitor: {e}")
            raise  # Fail if power measurement requested but NVML unavailable

    # Parse test range
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]
    torch_dtype = tllm._utils.str_dtype_to_torch(dtype)

    # AllReduce parameters
    all_reduce_params = trtllm_mods["TorchAllReduceParams"](
        strategy=trtllm_mods["AllReduceStrategy"].AUTO,
        fusion_op=trtllm_mods["AllReduceFusionOp"].NONE,
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

    from helper import log_perf

    size = min_size
    while size < max_size:
        input_shape = get_input_shape_and_comm_size(size)
        input_tensor = torch.ones(input_shape, dtype=torch_dtype, device="cuda")

        op_list = []
        for i in range(repeat_n):
            allreduce = trtllm_mods["AllReduce"](mapping=mapping).cuda()
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

        # Power measurement for AllReduce operations
        if measure_power and power_monitor is not None:
            power_monitor.begin_window("allreduce", sync_execution=True)
            start_event.record()
            for i in range(num_runs):
                g.replay()
            end_event.record()
            measurement = power_monitor.end_window("allreduce", sync_execution=True)
            torch.cuda.synchronize()
            # Calculate average power across this rank
            power_rank = measurement.gpu_energy[local_rank] / measurement.time
        else:
            start_event.record()
            for i in range(num_runs):
                g.replay()
            end_event.record()
            torch.cuda.synchronize()
            power_rank = None

        latency = start_event.elapsed_time(end_event) / num_runs / repeat_n

        # Collect power data from all ranks if measuring
        if measure_power and power_rank is not None:
            # Gather power from all ranks to rank 0
            power_list = [None] * world_size if rank == 0 else None
            torch.distributed.all_gather_object(power_list if rank == 0 else [], power_rank)
            if rank == 0:
                # Average power across all GPUs
                avg_power = sum(power_list) / len(power_list)
            else:
                avg_power = None
        else:
            avg_power = None

        if rank == 0 and local_rank == 0:
            power_str = f", Power: {avg_power:.2f} W" if avg_power is not None else ""
            print(f"[TensorRT-LLM] Size: {size}, Latency: {latency:.4f} ms{power_str}")

            # Get TensorRT-LLM version
            trtllm_version = tllm.__version__ if hasattr(tllm, "__version__") else "unknown"

            # Build result item
            item = {
                "allreduce_dtype": dtype,
                "num_gpus": world_size,
                "message_size": size,
                "latency": latency,
                "implementation": "trtllm",
            }

            if avg_power is not None:
                item["power"] = avg_power
                item["compute_bound"] = 0  # Communication is always memory/bandwidth-bound

            log_perf(
                item_list=[item],
                framework="TRTLLM",
                version=trtllm_version,
                device_name=torch.cuda.get_device_name(),
                op_name="all_reduce",
                kernel_source="TRTLLM",
                perf_filename=perf_filename,
            )

        size *= ratio


def setup_vllm_distributed(world_size, rank, use_slurm):
    """Setup vLLM distributed environment"""
    try:
        from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            graph_capture,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.utils import get_open_port

        vllm_mods = {
            "tensor_model_parallel_all_reduce": tensor_model_parallel_all_reduce,
            "init_distributed_environment": init_distributed_environment,
            "initialize_model_parallel": initialize_model_parallel,
            "graph_capture": graph_capture,
            "destroy_model_parallel": destroy_model_parallel,
            "get_open_port": get_open_port,
        }
    except ImportError as e:
        print(f"Failed to import vLLM modules: {e}")
        print("Please ensure vLLM is installed and PYTHONPATH is set correctly")
        sys.exit(1)

    if use_slurm:
        # Use SLURM environment variables
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    else:
        # For non-SLURM, assume single node or use environment variables
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    # Set CUDA device
    torch.cuda.set_device(local_rank)

    # Initialize distributed environment
    if not torch.distributed.is_initialized():
        # Get master address and port
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")

        # Construct the init method string
        distributed_init_method = f"tcp://{master_addr}:{master_port}"

        print("Setting up distributed environment:")
        print(f"  Init method: {distributed_init_method}")
        print(f"  World size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local rank: {local_rank}")

        try:
            vllm_mods["init_distributed_environment"](
                world_size=world_size,
                rank=rank,
                distributed_init_method=distributed_init_method,
                local_rank=local_rank,
                backend="nccl",
            )
        except Exception as e:
            print(f"\nERROR: Failed to initialize distributed environment: {e}")
            raise

    # Initialize model parallel groups
    vllm_mods["initialize_model_parallel"](tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1)

    return vllm_mods, local_rank


def benchmark_vllm_allreduce(
    dtype: str,
    test_range: str,
    world_size: int,
    rank: int,
    use_slurm: bool,
    perf_filename: str,
    measure_power: bool = False,
):
    """Benchmark vLLM custom AllReduce backend"""
    vllm_mods, local_rank = setup_vllm_distributed(world_size, rank, use_slurm)

    # Initialize NVML power monitor if power measurement is enabled
    power_monitor = None
    if measure_power:
        try:
            from nvml_power_monitor import NVMLPowerMonitor

            power_monitor = NVMLPowerMonitor(gpu_indices=[local_rank])
            if rank == 0:
                print("NVML power monitoring enabled on all ranks")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to initialize NVML power monitor: {e}")
            raise  # Fail if power measurement requested but NVML unavailable

    # Parse test range
    min_size, max_size, ratio = [int(i) for i in test_range.split(",")]

    # Map dtype string to torch dtype
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Benchmark parameters
    repeat_n = 5
    num_warmups = 3
    num_runs = 20

    from helper import log_perf

    # Warmup communication
    warmup_tensor = torch.ones(1, dtype=torch_dtype, device="cuda")
    _ = vllm_mods["tensor_model_parallel_all_reduce"](warmup_tensor)
    torch.cuda.synchronize()

    size = min_size
    while size < max_size:
        input_shape = get_input_shape_and_comm_size(size)

        # Test both graph capture and eager mode
        for use_graph in [True, False]:
            mode_str = "graph" if use_graph else "eager"

            if use_graph:
                # Graph capture mode
                with vllm_mods["graph_capture"](device=torch.cuda.current_device()) as graph_capture_context:
                    # Create input tensors
                    input_tensors = []
                    for _ in range(repeat_n):
                        inp = torch.ones(input_shape, dtype=torch_dtype, device="cuda")
                        input_tensors.append(inp)

                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()

                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        outputs = []
                        for inp in input_tensors:
                            out = vllm_mods["tensor_model_parallel_all_reduce"](inp)
                            outputs.append(out)

                # Warmup and timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                for i in range(num_warmups):
                    graph.replay()
                torch.cuda.synchronize()

                # Power measurement for graph mode
                if measure_power and power_monitor is not None:
                    power_monitor.begin_window("allreduce_graph", sync_execution=True)
                    start_event.record()
                    for i in range(num_runs):
                        graph.replay()
                    end_event.record()
                    measurement = power_monitor.end_window("allreduce_graph", sync_execution=True)
                    torch.cuda.synchronize()
                    power_rank = measurement.gpu_energy[local_rank] / measurement.time
                else:
                    start_event.record()
                    for i in range(num_runs):
                        graph.replay()
                    end_event.record()
                    torch.cuda.synchronize()
                    power_rank = None

            else:
                # Eager mode
                input_tensor = torch.ones(input_shape, dtype=torch_dtype, device="cuda")

                # Warmup
                torch.cuda.synchronize()
                for _ in range(num_warmups):
                    for _ in range(repeat_n):
                        _ = vllm_mods["tensor_model_parallel_all_reduce"](input_tensor.clone())
                torch.cuda.synchronize()

                # Timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Power measurement for eager mode
                if measure_power and power_monitor is not None:
                    power_monitor.begin_window("allreduce_eager", sync_execution=True)
                    start_event.record()
                    for _ in range(num_runs):
                        for _ in range(repeat_n):
                            _ = vllm_mods["tensor_model_parallel_all_reduce"](input_tensor.clone())
                    end_event.record()
                    measurement = power_monitor.end_window("allreduce_eager", sync_execution=True)
                    torch.cuda.synchronize()
                    power_rank = measurement.gpu_energy[local_rank] / measurement.time
                else:
                    start_event.record()
                    for _ in range(num_runs):
                        for _ in range(repeat_n):
                            _ = vllm_mods["tensor_model_parallel_all_reduce"](input_tensor.clone())
                    end_event.record()
                    torch.cuda.synchronize()
                    power_rank = None

            latency = start_event.elapsed_time(end_event) / num_runs / repeat_n

            # Collect power data from all ranks if measuring
            if measure_power and power_rank is not None:
                # Gather power from all ranks to rank 0
                power_list = [None] * world_size if rank == 0 else None
                torch.distributed.all_gather_object(power_list if rank == 0 else [], power_rank)
                if rank == 0:
                    # Average power across all GPUs
                    avg_power = sum(power_list) / len(power_list)
                else:
                    avg_power = None
            else:
                avg_power = None

            if rank == 0:
                power_str = f", Power: {avg_power:.2f} W" if avg_power is not None else ""
                print(f"[vLLM-{mode_str}] Size: {size}, Latency: {latency:.4f} ms{power_str}")

                # Get vLLM version
                try:
                    import vllm

                    vllm_version = vllm.__version__ if hasattr(vllm, "__version__") else "unknown"
                except:
                    vllm_version = "unknown"

                # Build result item
                item = {
                    "allreduce_dtype": dtype,
                    "num_gpus": world_size,
                    "message_size": size,
                    "latency": latency,
                    "backend": f"vllm_{mode_str}",
                }

                if avg_power is not None:
                    item["power"] = avg_power
                    item["compute_bound"] = 0  # Communication is always memory/bandwidth-bound

                log_perf(
                    item_list=[item],
                    framework="vLLM",
                    version=vllm_version,
                    device_name=torch.cuda.get_device_name(),
                    op_name="all_reduce",
                    kernel_source=f"vLLM_custom_{mode_str}",
                    perf_filename=perf_filename,
                )

        size *= ratio

    # Cleanup vLLM distributed environment
    vllm_mods["destroy_model_parallel"]()


def allreduce_benchmark(
    backend: str,
    dtype: str,
    test_range: str = "128,1073741824,2",
    use_slurm: bool = False,
    perf_filename: str = "custom_allreduce_perf.txt",
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    measure_power: bool = False,
):
    """
    CUDA Graph based AllReduce benchmark method supporting multiple backends with optional power measurement
    """
    # Setup distributed environment based on backend
    if backend == "trtllm":
        # TensorRT-LLM uses MPI by default
        tllm_mods = import_trtllm()
        tllm = tllm_mods["tllm"]

        if use_slurm:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ["RANK"])
        else:
            world_size = tllm.mpi_world_size()
            rank = tllm.mpi_rank()

        if world_size == 1:
            raise RuntimeError("Benchmark must run with world_size > 1")

        benchmark_trtllm_allreduce(dtype, test_range, world_size, rank, use_slurm, perf_filename, measure_power)

    elif backend == "vllm":
        if use_slurm:
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
        else:
            # Check if running under torchrun (it sets these env vars)
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                world_size = int(os.environ["WORLD_SIZE"])
                rank = int(os.environ["RANK"])
                print(f"Detected torchrun environment: world_size={world_size}, rank={rank}")
            else:
                # Use provided values or environment variables
                if world_size is None:
                    world_size = int(os.environ.get("WORLD_SIZE", "1"))
                if rank is None:
                    rank = int(os.environ.get("RANK", "0"))

        if world_size == 1:
            raise RuntimeError("Benchmark must run with world_size > 1")

        benchmark_vllm_allreduce(dtype, test_range, world_size, rank, use_slurm, perf_filename, measure_power)
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--backend", "-b", choices=["trtllm", "vllm"], default="trtllm", help="AllReduce backend to benchmark"
    )
    parser.add_argument("--dtype", "-t", default="float16")
    parser.add_argument(
        "--range",
        "-r",
        default="128,1073741824,2",  # 128B to 1024MB
        help="min_size,max_size,multiplicative_ratio",
    )
    parser.add_argument("--use-slurm", action="store_true", help="Use SLURM environment variables")
    parser.add_argument(
        "--perf-filename",
        "-f",
        default="custom_allreduce_perf.txt",
        help="Output performance file name",
    )
    # Additional arguments for vLLM when not using MPI/SLURM
    parser.add_argument("--world-size", default=8, type=int, help="World size for distributed setup (vLLM)")
    parser.add_argument("--rank", default=0, type=int, help="Rank for distributed setup (vLLM)")
    parser.add_argument(
        "--measure_power",
        action="store_true",
        default=False,
        help="Enable power measurement during AllReduce benchmark",
    )

    args = parser.parse_args()

    allreduce_benchmark(
        args.backend,
        args.dtype,
        args.range,
        args.use_slurm,
        args.perf_filename,
        args.world_size,
        args.rank,
        args.measure_power,
    )
