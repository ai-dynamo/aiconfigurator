# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT-LLM WideEP NVLinkTwoSided All-to-All Communication Benchmark

This script collects All-to-All communication performance data for WideEP MoE
using NVLinkTwoSided communication strategy.

Supports only balanced token distribution workloads.

Usage (Slurm):
    srun --ntasks 8 --ntasks-per-node 4 --mpi=pmix python collect_trtllm_alltoall.py
"""

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist

# Add parent directory to path for helper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper import log_perf


class TokenDistribution(Enum):
    """Token distribution strategies for expert selection."""

    BALANCED = "balanced"  # Uniform distribution across experts


class MoEDtype(Enum):
    """Supported MoE data types for All-to-All communication."""

    FLOAT16 = "float16"  # BFloat16/Float16
    FP8 = "fp8"  # FP8 E4M3
    NVFP4 = "nvfp4"  # NVFP4 with scale factors


# Token distribution configurations
DEFAULT_DISTRIBUTIONS = [
    TokenDistribution.BALANCED,
]

# Supported MoE data types
DEFAULT_MOE_DTYPES = [
    MoEDtype.FLOAT16,
    MoEDtype.FP8,
    MoEDtype.NVFP4,
]


@dataclass
class AlltoallTestCase:
    """Test case configuration for All-to-All benchmark."""

    num_tokens: int
    hidden_size: int
    num_experts: int
    top_k: int
    ep_size: int
    moe_dtype: MoEDtype = MoEDtype.FLOAT16
    distribution: TokenDistribution = TokenDistribution.BALANCED
    description: str = ""

    def __post_init__(self):
        """Generate description if not provided."""
        if not self.description:
            self.description = (
                f"tokens={self.num_tokens}, hidden={self.hidden_size}, "
                f"experts={self.num_experts}, topk={self.top_k}, "
                f"dtype={self.moe_dtype.value}, dist={self.distribution.value}"
            )


def get_default_test_cases(ep_size: int) -> list[AlltoallTestCase]:
    """
    Generate default test cases for All-to-All benchmark.

    Args:
        ep_size: Expert Parallelism size (number of GPUs)

    Returns:
        List of test cases covering different token counts, model configs, dtypes, and distributions
    """
    test_cases = []

    # Token counts to test (covering prefill and decode scenarios)
    token_counts = [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        256,
        320,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        20480,
        32768,
        65536,
    ]

    # Model configurations (hidden_size, num_experts, top_k)
    model_configs = [
        # DeepSeek-V3 style
        (7168, 256, 8),
    ]

    for num_tokens in token_counts:
        for hidden_size, num_experts, top_k in model_configs:
            # Skip if num_experts < ep_size
            if num_experts < ep_size:
                continue

            for moe_dtype in DEFAULT_MOE_DTYPES:
                for distribution in DEFAULT_DISTRIBUTIONS:
                    test_cases.append(
                        AlltoallTestCase(
                            num_tokens=num_tokens,
                            hidden_size=hidden_size,
                            num_experts=num_experts,
                            top_k=top_k,
                            ep_size=ep_size,
                            moe_dtype=moe_dtype,
                            distribution=distribution,
                        )
                    )

    return test_cases


def init_distributed():
    """
    Initialize distributed environment using Slurm with srun --mpi=pmix.

    MNNVL requires MPI for symmetric memory management.

    Returns:
        Tuple of (rank, world_size, device)
    """
    from tensorrt_llm._utils import mpi_comm

    # Get MPI communicator (srun --mpi=pmix)
    comm = mpi_comm()
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Get local rank from Slurm environment
    if "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
    elif "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        gpus_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
        local_rank = rank % gpus_per_node

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Also set via cuda bindings for consistency
    try:
        from cuda import cudart
    except ImportError:
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            cudart = None
    if cudart is not None:
        cudart.cudaSetDevice(local_rank)

    # Initialize NCCL process group for barriers
    if world_size > 1 and not dist.is_initialized():
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", device_id=device)

    print(
        f"Rank {rank} initialized with MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}"
    )

    return rank, world_size, device


def check_mnnvl_support() -> bool:
    """
    Check if MNNVL (Multi-Node NVLink) is supported on current hardware.

    Returns:
        True if MNNVL is supported
    """
    try:
        from tensorrt_llm._mnnvl_utils import MnnvlMemory

        MnnvlMemory.initialize()
        return MnnvlMemory.supports_mnnvl()
    except Exception as e:
        print(f"MNNVL support check failed: {e}")
        return False


def create_mapping(rank: int, world_size: int):
    """
    Create TensorRT-LLM Mapping for MoE EP.

    Args:
        rank: Current rank
        world_size: Total number of ranks

    Returns:
        Mapping object
    """
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=4,
        tp_size=world_size,  # Must satisfy: tp_size * pp_size == world_size
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=world_size,
    )

    return mapping


def generate_balanced_expert_ids(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate balanced expert IDs for testing.

    Each token selects top_k experts, distributed evenly across all experts.

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        top_k: Number of experts per token
        device: Target device

    Returns:
        Expert IDs tensor of shape [num_tokens, top_k]
    """
    expert_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32, device=device)

    for i in range(num_tokens):
        # Distribute tokens evenly across experts
        base_expert = (i * top_k) % num_experts
        for k in range(top_k):
            expert_ids[i, k] = (base_expert + k) % num_experts

    return expert_ids


def generate_expert_ids(
    test_case: AlltoallTestCase,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate expert IDs based on test case distribution configuration.

    Args:
        test_case: Test case with distribution settings
        device: Target device

    Returns:
        Expert IDs tensor of shape [num_tokens, top_k]
    """
    if test_case.distribution == TokenDistribution.BALANCED:
        return generate_balanced_expert_ids(
            test_case.num_tokens,
            test_case.num_experts,
            test_case.top_k,
            device,
        )
    else:
        raise ValueError(f"Unknown distribution: {test_case.distribution}")


def get_dispatch_data_size_bytes(num_tokens: int, hidden_size: int, moe_dtype: MoEDtype) -> int:
    """
    Calculate data size in bytes for AlltoAll dispatch based on MoE dtype.

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension size
        moe_dtype: MoE data type

    Returns:
        Data size in bytes
    """
    if moe_dtype == MoEDtype.FLOAT16:
        # bfloat16: 2 bytes per element
        return num_tokens * hidden_size * 2
    elif moe_dtype == MoEDtype.FP8:
        # fp8: 1 byte per element
        return num_tokens * hidden_size * 1
    elif moe_dtype == MoEDtype.NVFP4:
        # NVFP4: 0.5 bytes per element (2 values packed in 1 byte) + scale factors
        # Scale factors: 1 scale per 16 elements, stored as uint8
        data_bytes = num_tokens * (hidden_size // 2)  # packed FP4
        sf_bytes = num_tokens * (hidden_size // 16)  # scale factors
        return data_bytes + sf_bytes
    else:
        return num_tokens * hidden_size * 2


def get_combine_data_size_bytes(num_tokens: int, hidden_size: int) -> int:
    """
    Calculate data size in bytes for AlltoAll combine.

    Combine always uses bfloat16 (expert output precision).

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension size

    Returns:
        Data size in bytes
    """
    # Combine always uses bfloat16: 2 bytes per element
    return num_tokens * hidden_size * 2


def calculate_bandwidth_gbps(data_size_bytes: int, latency_ms: float) -> float:
    """
    Calculate bandwidth in GB/s.

    Args:
        data_size_bytes: Data size in bytes
        latency_ms: Latency in milliseconds

    Returns:
        Bandwidth in GB/s
    """
    if latency_ms <= 0:
        return 0.0
    # Convert: bytes / ms -> GB/s
    # bytes / ms = bytes * 1000 / s = KB/s * 1000 = MB/s
    # GB/s = bytes / ms / 1e6
    return data_size_bytes / latency_ms / 1e6


def prepare_test_data(
    test_case: AlltoallTestCase,
    device: torch.device,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Prepare test data based on MoE dtype.

    Args:
        test_case: Test case configuration
        device: CUDA device

    Returns:
        Tuple of (hidden_states, hidden_states_sf, token_selected_slots, token_final_scales)
        - hidden_states_sf is scale factor for NVFP4, None otherwise
    """
    num_tokens = test_case.num_tokens
    hidden_size = test_case.hidden_size
    top_k = test_case.top_k
    moe_dtype = test_case.moe_dtype

    # Generate expert IDs
    token_selected_slots = generate_expert_ids(test_case, device)
    token_final_scales = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device) / top_k

    # Generate hidden states based on dtype
    hidden_states_sf = None

    if moe_dtype == MoEDtype.FLOAT16:
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    elif moe_dtype == MoEDtype.FP8:
        # FP8: generate in bfloat16 then cast
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        hidden_states = hidden_states.to(torch.float8_e4m3fn)
    elif moe_dtype == MoEDtype.NVFP4:
        # NVFP4: use uint8 for quantized data + scale factors
        # hidden_size/2 because we pack 2 FP4 values per uint8
        hidden_states = torch.randint(0, 255, (num_tokens, hidden_size // 2), dtype=torch.uint8, device=device)
        # Scale factors: hidden_size/16 (one scale per 16 elements)
        hidden_states_sf = torch.randint(0, 255, (num_tokens, hidden_size // 16), dtype=torch.uint8, device=device)
    else:
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    return hidden_states, hidden_states_sf, token_selected_slots, token_final_scales


@dataclass
class AlltoallBenchmarkResult:
    """Benchmark results for each operation."""

    prepare_latency_ms: float
    dispatch_latency_ms: float
    combine_latency_ms: float
    combine_low_precision_latency_ms: float


def benchmark_nvlink_two_sided_alltoall(
    test_case: AlltoallTestCase,
    mapping,
    device: torch.device,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> AlltoallBenchmarkResult:
    """
    Benchmark NVLinkTwoSided All-to-All communication.

    Args:
        test_case: Test case configuration
        mapping: TensorRT-LLM Mapping
        device: CUDA device
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        AlltoallBenchmarkResult containing latencies for each operation
    """
    from tensorrt_llm._mnnvl_utils import MnnvlMoe

    # Get workspaces
    alltoall_workspace = MnnvlMoe.get_moe_workspaces(mapping)
    alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(mapping)

    num_tokens = test_case.num_tokens
    hidden_size = test_case.hidden_size
    num_experts = test_case.num_experts
    top_k = test_case.top_k
    ep_size = test_case.ep_size
    ep_rank = mapping.moe_ep_rank
    moe_dtype = test_case.moe_dtype

    # Number of slots (same as num_experts for simple case)
    num_slots = num_experts

    # Prepare test data
    hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = prepare_test_data(test_case, device)

    # All rank token counts
    all_rank_num_tokens = [num_tokens] * ep_size
    all_rank_max_num_tokens = max(all_rank_num_tokens)

    # Collect timing results
    all_prepare_times = []
    all_dispatch_times = []
    all_combine_times = []
    all_combine_low_precision_times = []

    # ============================================================================
    # Benchmark: alltoall_prepare
    # ============================================================================
    def prepare_func():
        return MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
            token_selected_slots,
            None,  # expert_statics (optional for EPLB)
            alltoall_prepare_workspace,
            all_rank_max_num_tokens,
            ep_rank,
            ep_size,
            num_experts,
            num_slots,
            top_k,
        )

    # Warmup
    for _ in range(num_warmup):
        alltoall_info, _ = prepare_func()
    torch.cuda.synchronize()

    # Benchmark prepare
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_event.record()
        alltoall_info, _ = prepare_func()
        end_event.record()
        end_event.synchronize()
        all_prepare_times.append(start_event.elapsed_time(end_event))

    # ============================================================================
    # Benchmark: alltoall_dispatch (All-to-All send)
    # ============================================================================
    def dispatch_func():
        return MnnvlMoe.mnnvl_moe_alltoallv(
            [hidden_states, hidden_states_sf, token_selected_slots, token_final_scales],
            alltoall_info,
            alltoall_workspace,
            ep_rank,
            ep_size,
        )

    # Warmup
    for _ in range(num_warmup):
        dispatched = dispatch_func()
    torch.cuda.synchronize()

    # Benchmark dispatch
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_event.record()
        dispatched = dispatch_func()
        end_event.record()
        end_event.synchronize()
        all_dispatch_times.append(start_event.elapsed_time(end_event))

    # Get dispatched hidden states for combine benchmark
    recv_hidden_states = dispatched[0]

    # Simulate MoE output: combine always operates on bfloat16 expert output
    moe_output = torch.randn(recv_hidden_states.shape[0], hidden_size, dtype=torch.bfloat16, device=device)

    # ============================================================================
    # Benchmark: alltoall_combine (do_reduce=False, use_low_precision_combine=False)
    # ============================================================================
    def combine_func():
        return MnnvlMoe.mnnvl_moe_alltoallv_combine(
            moe_output,
            alltoall_info,
            alltoall_workspace,
            ep_rank=ep_rank,
            ep_size=ep_size,
            top_k=top_k,
            token_count=num_tokens,
            use_low_precision_combine=False,
            do_reduce=False,
        )

    # Warmup
    for _ in range(num_warmup):
        combine_func()
    torch.cuda.synchronize()

    # Benchmark combine
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start_event.record()
        combine_func()
        end_event.record()
        end_event.synchronize()
        all_combine_times.append(start_event.elapsed_time(end_event))

    # ============================================================================
    # Benchmark: alltoall_combine_low_precision (do_reduce=False, use_low_precision_combine=True)
    # Only benchmark for NVFP4 dtype as low_precision_combine is most relevant for it
    # ============================================================================
    if moe_dtype == MoEDtype.NVFP4:

        def combine_low_precision_func():
            return MnnvlMoe.mnnvl_moe_alltoallv_combine(
                moe_output,
                alltoall_info,
                alltoall_workspace,
                ep_rank=ep_rank,
                ep_size=ep_size,
                top_k=top_k,
                token_count=num_tokens,
                use_low_precision_combine=True,
                do_reduce=False,
            )

        # Warmup
        for _ in range(num_warmup):
            try:
                combine_low_precision_func()
            except Exception:
                # Low precision combine may not be supported
                pass
        torch.cuda.synchronize()

        # Benchmark combine with low precision
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_event.record()
            try:
                combine_low_precision_func()
                end_event.record()
                end_event.synchronize()
                all_combine_low_precision_times.append(start_event.elapsed_time(end_event))
            except Exception:
                # Low precision combine may fail for certain configurations
                end_event.record()
                end_event.synchronize()

    # Calculate average latencies
    prepare_latency = sum(all_prepare_times) / len(all_prepare_times) if all_prepare_times else 0.0
    dispatch_latency = sum(all_dispatch_times) / len(all_dispatch_times) if all_dispatch_times else 0.0
    combine_latency = sum(all_combine_times) / len(all_combine_times) if all_combine_times else 0.0
    combine_low_precision_latency = (
        sum(all_combine_low_precision_times) / len(all_combine_low_precision_times)
        if all_combine_low_precision_times
        else 0.0
    )

    return AlltoallBenchmarkResult(
        prepare_latency_ms=prepare_latency,
        dispatch_latency_ms=dispatch_latency,
        combine_latency_ms=combine_latency,
        combine_low_precision_latency_ms=combine_low_precision_latency,
    )


def log_alltoall_perf(
    test_case: AlltoallTestCase,
    op_name: str,
    latency_ms: float,
    framework: str,
    version: str,
    device_name: str,
    perf_filename: str,
):
    """
    Log All-to-All performance data in moe_perf.txt compatible format.

    Args:
        test_case: Test case configuration
        op_name: Operation name (alltoall_prepare, alltoall_dispatch, alltoall_combine, alltoall_combine_low_precision)
        latency_ms: Latency in milliseconds
        framework: Framework name (e.g., "TRTLLM")
        version: Framework version
        device_name: GPU device name
        perf_filename: Output file path
    """
    distribution_str = test_case.distribution.value

    log_perf(
        item_list=[
            {
                "moe_dtype": test_case.moe_dtype.value,
                "num_tokens": test_case.num_tokens,
                "hidden_size": test_case.hidden_size,
                "topk": test_case.top_k,
                "num_experts": test_case.num_experts,
                "moe_ep_size": test_case.ep_size,
                "distribution": distribution_str,
                "latency": latency_ms,
            }
        ],
        framework=framework,
        version=version,
        device_name=device_name,
        op_name=op_name,
        kernel_source="MnnvlMoe",
        perf_filename=perf_filename,
    )


def run_benchmark(
    rank: int,
    world_size: int,
    device: torch.device,
    output_file: str = "wideep_alltoall_perf.txt",
    num_warmup: int = 3,
    num_iterations: int = 10,
):
    """
    Run All-to-All benchmark and log results.

    Args:
        rank: Current rank
        world_size: Total number of ranks
        device: CUDA device
        output_file: Output file path
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
    """
    import tensorrt_llm

    # Check MNNVL support
    if not check_mnnvl_support():
        if rank == 0:
            print("ERROR: MNNVL (NVLink) not supported on this hardware.")
            print("NVLinkTwoSided requires full NVLink connectivity between GPUs.")
        return

    # Create mapping
    mapping = create_mapping(rank, world_size)

    # Get test cases
    test_cases = get_default_test_cases(world_size)

    framework = "TRTLLM"
    version = tensorrt_llm.__version__
    device_name = torch.cuda.get_device_name(device)

    if rank == 0:
        print(f"\n{'=' * 70}")
        print("TensorRT-LLM WideEP NVLinkTwoSided All-to-All Benchmark")
        print(f"{'=' * 70}")
        print(f"EP size: {world_size}")
        print(f"Device: {device_name}")
        print(f"TensorRT-LLM version: {version}")
        print(f"Number of test cases: {len(test_cases)}")
        print(f"MoE dtypes: {[d.value for d in DEFAULT_MOE_DTYPES]}")
        print(f"{'=' * 70}\n")

    # Run benchmarks
    for idx, test_case in enumerate(test_cases):
        try:
            if rank == 0:
                print(f"[{idx + 1}/{len(test_cases)}] {test_case.description}")

            # Synchronize before benchmark
            if world_size > 1:
                dist.barrier()

            result = benchmark_nvlink_two_sided_alltoall(
                test_case,
                mapping,
                device,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )

            # Log results (only rank 0)
            if rank == 0:
                # Calculate data sizes and bandwidths
                dispatch_data_size = get_dispatch_data_size_bytes(
                    test_case.num_tokens, test_case.hidden_size, test_case.moe_dtype
                )
                combine_data_size = get_combine_data_size_bytes(test_case.num_tokens, test_case.hidden_size)

                dispatch_bw = calculate_bandwidth_gbps(dispatch_data_size, result.dispatch_latency_ms)
                combine_bw = calculate_bandwidth_gbps(combine_data_size, result.combine_latency_ms)
                combine_lp_bw = calculate_bandwidth_gbps(combine_data_size, result.combine_low_precision_latency_ms)

                print(f"  Prepare: {result.prepare_latency_ms:.3f} ms")
                dispatch_kb = dispatch_data_size / 1024
                print(f"  Dispatch: {result.dispatch_latency_ms:.3f} ms ({dispatch_bw:.2f} GB/s, {dispatch_kb:.1f} KB)")
                combine_kb = combine_data_size / 1024
                print(f"  Combine: {result.combine_latency_ms:.3f} ms ({combine_bw:.2f} GB/s, {combine_kb:.1f} KB)")
                if result.combine_low_precision_latency_ms > 0:
                    lp_ms = result.combine_low_precision_latency_ms
                    print(f"  Combine (low precision): {lp_ms:.3f} ms ({combine_lp_bw:.2f} GB/s)")

                # Log each operation separately
                log_alltoall_perf(
                    test_case,
                    "alltoall_prepare",
                    result.prepare_latency_ms,
                    framework,
                    version,
                    device_name,
                    output_file,
                )
                log_alltoall_perf(
                    test_case,
                    "alltoall_dispatch",
                    result.dispatch_latency_ms,
                    framework,
                    version,
                    device_name,
                    output_file,
                )
                log_alltoall_perf(
                    test_case,
                    "alltoall_combine",
                    result.combine_latency_ms,
                    framework,
                    version,
                    device_name,
                    output_file,
                )
                if result.combine_low_precision_latency_ms > 0:
                    log_alltoall_perf(
                        test_case,
                        "alltoall_combine_low_precision",
                        result.combine_low_precision_latency_ms,
                        framework,
                        version,
                        device_name,
                        output_file,
                    )

        except Exception as e:
            if rank == 0:
                print(f"  ERROR: {e}")
            continue

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Benchmark completed. Results saved to: {output_file}")
        print(f"{'=' * 70}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WideEP NVLinkTwoSided All-to-All Communication Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="wideep_alltoall_perf.txt",
        help="Output file path for performance results (default: wideep_alltoall_perf.txt)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    rank, world_size, device = init_distributed()

    if world_size < 2:
        print("ERROR: This benchmark requires at least 2 GPUs.")
        print("Usage: srun --ntasks N --mpi=pmix python collect_trtllm_alltoall.py")
        return

    if rank == 0:
        print(f"Running with {world_size} GPUs")

    try:
        run_benchmark(
            rank,
            world_size,
            device,
            output_file=args.output,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
