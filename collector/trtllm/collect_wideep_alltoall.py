# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT-LLM WideEP NVLinkTwoSided All-to-All Communication Benchmark

This script collects All-to-All communication performance data for WideEP MoE
using NVLinkTwoSided communication strategy.

Supports both balanced and power-law token distributions to simulate real-world
workloads where some experts receive more tokens than others.

IMPORTANT: This script MUST be run with mpirun because TensorRT-LLM's MNNVL
(Multi-Node NVLink) uses MPI for symmetric memory management.

Usage:
    # Run with mpirun (REQUIRED):
    mpirun -np 4 python collect_wideep_alltoall.py

    # For multi-node:
    mpirun -np 8 -hostfile hostfile python collect_wideep_alltoall.py
"""

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

# Add parent directory to path for helper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from helper import benchmark_with_power, log_perf, sample_power_law
except ModuleNotFoundError:
    from helper import benchmark_with_power, log_perf, sample_power_law


class TokenDistribution(Enum):
    """Token distribution strategies for expert selection."""

    BALANCED = "balanced"  # Uniform distribution across experts
    POWER_LAW = "power_law"  # Skewed distribution (some experts get more tokens)


class MoEDtype(Enum):
    """Supported MoE data types for All-to-All communication."""

    FLOAT16 = "float16"  # BFloat16/Float16
    FP8 = "fp8"  # FP8 E4M3
    NVFP4 = "nvfp4"  # NVFP4 with scale factors


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
    power_law_alpha: Optional[float] = None  # Only used when distribution is POWER_LAW
    description: str = ""

    def __post_init__(self):
        """Generate description if not provided."""
        if not self.description:
            dist_str = self.distribution.value
            if self.distribution == TokenDistribution.POWER_LAW and self.power_law_alpha:
                dist_str = f"power_law_α={self.power_law_alpha}"
            self.description = (
                f"tokens={self.num_tokens}, hidden={self.hidden_size}, "
                f"experts={self.num_experts}, topk={self.top_k}, "
                f"dtype={self.moe_dtype.value}, dist={dist_str}"
            )


# Token distribution configurations: (distribution_type, power_law_alpha)
# - balanced: uniform distribution across all experts
# - power_law with α=1.01: slight imbalance
# - power_law with α=1.2: moderate imbalance (realistic workload)
DEFAULT_DISTRIBUTIONS = [
    (TokenDistribution.BALANCED, None),
    # (TokenDistribution.POWER_LAW, 1.01),
    # (TokenDistribution.POWER_LAW, 1.2),
]

# Supported MoE data types
DEFAULT_MOE_DTYPES = [
    MoEDtype.FLOAT16,
    MoEDtype.FP8,
    MoEDtype.NVFP4,
]


def get_torch_dtype(moe_dtype: MoEDtype) -> torch.dtype:
    """Convert MoEDtype to torch.dtype for hidden states."""
    if moe_dtype == MoEDtype.FLOAT16:
        return torch.bfloat16
    elif moe_dtype == MoEDtype.FP8:
        return torch.float8_e4m3fn
    elif moe_dtype == MoEDtype.NVFP4:
        # NVFP4 uses uint8 for storage, but hidden states are bfloat16 before quantization
        return torch.bfloat16
    else:
        return torch.bfloat16


def get_default_test_cases(ep_size: int) -> List[AlltoallTestCase]:
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
        1, 2, 4, 8, 16, 32, 48, 64, 80, 96,
        128, 160, 192, 256, 320, 384, 512, 768,
        1024, 1536, 2048, 3072, 4096, 6144,
        8192, 12288, 16384, 20480, 32768, 65536
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
                for distribution, alpha in DEFAULT_DISTRIBUTIONS:
                    test_cases.append(
                        AlltoallTestCase(
                            num_tokens=num_tokens,
                            hidden_size=hidden_size,
                            num_experts=num_experts,
                            top_k=top_k,
                            ep_size=ep_size,
                            moe_dtype=moe_dtype,
                            distribution=distribution,
                            power_law_alpha=alpha,
                        )
                    )

    return test_cases


def init_distributed():
    """
    Initialize distributed environment using MPI.

    MNNVL requires MPI for symmetric memory management, so this script
    must be launched with mpirun.

    Returns:
        Tuple of (rank, world_size, device)
    """
    # Import MPI from TensorRT-LLM's utilities
    from tensorrt_llm._utils import mpi_comm

    # Get MPI communicator
    comm = mpi_comm()
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Calculate local rank (for multi-node setups)
    # Use OMPI environment variable if available, otherwise use rank % num_gpus
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        # Assume GPUs are assigned sequentially
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize NCCL process group for barriers (optional but useful for synchronization)
    if world_size > 1 and not dist.is_initialized():
        # Set environment variables for torch.distributed
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl")

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
    ep_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate balanced expert IDs for testing.

    Each token selects top_k experts, distributed evenly across all experts.

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        top_k: Number of experts per token
        ep_size: Expert Parallelism size
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


def generate_power_law_expert_ids(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    ep_size: int,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate power-law distributed expert IDs for testing imbalanced workloads.

    Some experts ("hot" experts) will receive significantly more tokens than others,
    simulating real-world scenarios where certain experts specialize in common patterns.

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        top_k: Number of experts per token
        ep_size: Expert Parallelism size
        alpha: Power-law exponent (higher = more imbalanced)
               - α ≈ 1.0: nearly uniform
               - α = 1.2: moderate skew
               - α > 1.5: heavy skew
        device: Target device

    Returns:
        Expert IDs tensor of shape [num_tokens, top_k]
    """
    # Generate power-law token counts per expert
    if num_tokens * top_k > num_experts:
        tokens_per_expert = sample_power_law(num_experts, alpha, 1, num_tokens * 0.8)
    else:
        tokens_per_expert = sample_power_law(num_experts, alpha, 0.01, 2)

    # Normalize to match target total
    target_sum = num_tokens * top_k
    tokens_per_expert = tokens_per_expert / tokens_per_expert.sum() * target_sum
    tokens_per_expert = torch.round(tokens_per_expert).to(torch.int64)

    # Adjust to exactly match target sum
    current_sum = tokens_per_expert.sum().item()
    delta = int(target_sum - current_sum)
    if delta != 0:
        sorted_indices = torch.argsort(tokens_per_expert, descending=True)
        if delta > 0:
            for i in range(delta):
                expert_idx = sorted_indices[i % len(sorted_indices)]
                tokens_per_expert[expert_idx] += 1
        else:
            for i in range(-delta):
                expert_idx = sorted_indices[-(i % len(sorted_indices)) - 1]
                if tokens_per_expert[expert_idx] > 0:
                    tokens_per_expert[expert_idx] -= 1
                else:
                    tokens_per_expert[torch.argmax(tokens_per_expert)] -= 1

    # Ensure the max-load EP rank is rank 0 for consistent benchmarking
    # This simulates measuring the "worst case" rank
    experts_per_rank = num_experts // ep_size
    if experts_per_rank > 0:
        rank_loads = tokens_per_expert.view(ep_size, experts_per_rank).sum(dim=1)
        max_rank = torch.argmax(rank_loads).item()
        if max_rank != 0:
            # Swap experts between rank 0 and max_rank
            reshaped = tokens_per_expert.view(ep_size, experts_per_rank)
            reshaped[0], reshaped[max_rank] = reshaped[max_rank].clone(), reshaped[0].clone()
            tokens_per_expert = reshaped.view(-1)

    # Build expert assignments for each token
    # Each token needs top_k DIFFERENT experts
    expert_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32, device=device)

    # Create a pool of expert assignments based on power-law counts
    # Each expert appears in the pool according to its token count
    expert_pool = []
    for expert_id in range(num_experts):
        count = int(tokens_per_expert[expert_id].item())
        expert_pool.extend([expert_id] * count)

    # Shuffle the pool to distribute experts randomly across tokens
    import random
    random.shuffle(expert_pool)

    # Assign experts to tokens, ensuring each token gets top_k different experts
    pool_idx = 0
    for token_idx in range(num_tokens):
        assigned_experts = set()
        k_idx = 0
        attempts = 0
        max_attempts = num_experts * 2  # Prevent infinite loop

        while k_idx < top_k and attempts < max_attempts:
            if pool_idx >= len(expert_pool):
                # Pool exhausted, wrap around
                pool_idx = 0
                random.shuffle(expert_pool)

            expert_id = expert_pool[pool_idx]
            pool_idx += 1

            # Only assign if this expert hasn't been assigned to this token yet
            if expert_id not in assigned_experts:
                expert_ids[token_idx, k_idx] = expert_id
                assigned_experts.add(expert_id)
                k_idx += 1
            attempts += 1

        # If we couldn't fill all top_k slots (shouldn't happen normally),
        # fill remaining with random different experts
        if k_idx < top_k:
            available = [e for e in range(num_experts) if e not in assigned_experts]
            for remaining_idx in range(k_idx, top_k):
                if available:
                    expert_ids[token_idx, remaining_idx] = available.pop(0)
                else:
                    # Fallback: just use any expert
                    expert_ids[token_idx, remaining_idx] = remaining_idx % num_experts

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
            test_case.ep_size,
            device,
        )
    elif test_case.distribution == TokenDistribution.POWER_LAW:
        alpha = test_case.power_law_alpha or 1.2  # Default alpha
        return generate_power_law_expert_ids(
            test_case.num_tokens,
            test_case.num_experts,
            test_case.top_k,
            test_case.ep_size,
            alpha,
            device,
        )
    else:
        raise ValueError(f"Unknown distribution: {test_case.distribution}")


def prepare_test_data(
    test_case: AlltoallTestCase,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
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
    num_distribution_samples: int = 5,
) -> AlltoallBenchmarkResult:
    """
    Benchmark NVLinkTwoSided All-to-All communication.

    For power-law distributions, multiple samples are taken to reduce variance
    from the random distribution generation.

    Args:
        test_case: Test case configuration
        mapping: TensorRT-LLM Mapping
        device: CUDA device
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        num_distribution_samples: Number of distribution samples for power-law (default: 5)

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

    # For power-law, we sample multiple distributions to get stable results
    is_power_law = test_case.distribution == TokenDistribution.POWER_LAW
    num_samples = num_distribution_samples if is_power_law else 1

    # Pre-generate multiple test data samples for power-law
    test_data_samples = [prepare_test_data(test_case, device) for _ in range(num_samples)]

    # All rank token counts (same for balanced case)
    all_rank_num_tokens = [num_tokens] * ep_size
    all_rank_max_num_tokens = max(all_rank_num_tokens)

    # Collect results across all distribution samples
    all_prepare_times = []
    all_dispatch_times = []
    all_combine_times = []
    all_combine_low_precision_times = []

    for sample_idx, (hidden_states, hidden_states_sf, token_selected_slots, token_final_scales) in enumerate(test_data_samples):
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

        # Warmup (only on first sample)
        if sample_idx == 0:
            for _ in range(num_warmup):
                alltoall_info, _ = prepare_func()
            torch.cuda.synchronize()

        # Benchmark prepare
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            alltoall_info, _ = prepare_func()
            end.record()
            end.synchronize()
            all_prepare_times.append(start.elapsed_time(end))

        # ============================================================================
        # Benchmark: alltoall_dispatch (All-to-All send)
        # ============================================================================
        def dispatch_func():
            return MnnvlMoe.mnnvl_moe_alltoallv(
                [hidden_states.clone(), hidden_states_sf.clone() if hidden_states_sf is not None else None,
                 token_selected_slots.clone(), token_final_scales.clone()],
                alltoall_info,
                alltoall_workspace,
                ep_rank,
                ep_size,
            )

        # Warmup (only on first sample)
        if sample_idx == 0:
            for _ in range(num_warmup):
                dispatched = dispatch_func()
            torch.cuda.synchronize()

        # Benchmark dispatch
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dispatched = dispatch_func()
            end.record()
            end.synchronize()
            all_dispatch_times.append(start.elapsed_time(end))

        # Get dispatched hidden states for combine benchmark
        recv_hidden_states = dispatched[0]

        # Simulate MoE output (same shape as received hidden states, but always bfloat16 for combine)
        # Combine always operates on expert output which is bfloat16
        if recv_hidden_states.dtype == torch.uint8:
            # For NVFP4, expert output is bfloat16
            moe_output = torch.randn(recv_hidden_states.shape[0], hidden_size, dtype=torch.bfloat16, device=device)
        elif recv_hidden_states.dtype == torch.float8_e4m3fn:
            # For FP8, expert output is bfloat16
            moe_output = torch.randn(recv_hidden_states.shape[0], hidden_size, dtype=torch.bfloat16, device=device)
        else:
            moe_output = torch.randn_like(recv_hidden_states)

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

        # Warmup (only on first sample)
        if sample_idx == 0:
            for _ in range(num_warmup):
                combined = combine_func()
            torch.cuda.synchronize()

        # Benchmark combine
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            combined = combine_func()
            end.record()
            end.synchronize()
            all_combine_times.append(start.elapsed_time(end))

        # ============================================================================
        # Benchmark: alltoall_combine_low_precision (do_reduce=False, use_low_precision_combine=True)
        # Note: low_precision_combine only works with bfloat16/float16 output
        # ============================================================================
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

        # Warmup (only on first sample)
        if sample_idx == 0:
            for _ in range(num_warmup):
                try:
                    combined_lp = combine_low_precision_func()
                except Exception:
                    # Low precision combine may not be supported for all dtypes
                    pass
            torch.cuda.synchronize()

        # Benchmark combine with low precision
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                combined_lp = combine_low_precision_func()
                end.record()
                end.synchronize()
                all_combine_low_precision_times.append(start.elapsed_time(end))
            except Exception:
                # Low precision combine may fail for certain configurations
                end.record()
                end.synchronize()

    # Calculate average latencies across all samples
    prepare_latency = sum(all_prepare_times) / len(all_prepare_times) if all_prepare_times else 0.0
    dispatch_latency = sum(all_dispatch_times) / len(all_dispatch_times) if all_dispatch_times else 0.0
    combine_latency = sum(all_combine_times) / len(all_combine_times) if all_combine_times else 0.0
    combine_low_precision_latency = (
        sum(all_combine_low_precision_times) / len(all_combine_low_precision_times)
        if all_combine_low_precision_times else 0.0
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
    # Format distribution string
    if test_case.distribution == TokenDistribution.POWER_LAW:
        distribution_str = f"power_law_{test_case.power_law_alpha}"
    else:
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
):
    """
    Run All-to-All benchmark and log results.

    Args:
        rank: Current rank
        world_size: Total number of ranks
        device: CUDA device
        output_file: Output file path
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
        print(f"TensorRT-LLM WideEP NVLinkTwoSided All-to-All Benchmark")
        print(f"{'=' * 70}")
        print(f"World size: {world_size}")
        print(f"Device: {device_name}")
        print(f"TensorRT-LLM version: {version}")
        print(f"Number of test cases: {len(test_cases)}")
        print(f"MoE dtypes: {[d.value for d in DEFAULT_MOE_DTYPES]}")
        print(f"Distributions: balanced, power_law (α=1.01, α=1.2)")
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
                num_warmup=3,
                num_iterations=10,
            )

            # Log results (only rank 0)
            if rank == 0:
                print(f"  Prepare: {result.prepare_latency_ms:.3f} ms")
                print(f"  Dispatch: {result.dispatch_latency_ms:.3f} ms")
                print(f"  Combine: {result.combine_latency_ms:.3f} ms")
                print(f"  Combine (low precision): {result.combine_low_precision_latency_ms:.3f} ms")

                # Log each operation separately
                log_alltoall_perf(
                    test_case, "alltoall_prepare", result.prepare_latency_ms,
                    framework, version, device_name, output_file
                )
                log_alltoall_perf(
                    test_case, "alltoall_dispatch", result.dispatch_latency_ms,
                    framework, version, device_name, output_file
                )
                log_alltoall_perf(
                    test_case, "alltoall_combine", result.combine_latency_ms,
                    framework, version, device_name, output_file
                )
                if result.combine_low_precision_latency_ms > 0:
                    log_alltoall_perf(
                        test_case, "alltoall_combine_low_precision", result.combine_low_precision_latency_ms,
                        framework, version, device_name, output_file
                    )

        except Exception as e:
            if rank == 0:
                print(f"  ERROR: {e}")
            continue

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Benchmark completed. Results saved to: {output_file}")
        print(f"{'=' * 70}")


def get_wideep_alltoall_test_cases():
    """
    Returns test cases for collect.py framework integration.

    Returns:
        List of [ep_size, output_filename] pairs
    """
    return [
        [2, "wideep_alltoall_perf.txt"],
        [4, "wideep_alltoall_perf.txt"],
        [8, "wideep_alltoall_perf.txt"],
        [16, "wideep_alltoall_perf.txt"],
        [32, "wideep_alltoall_perf.txt"],
        [64, "wideep_alltoall_perf.txt"],
        [72, "wideep_alltoall_perf.txt"],  # NVL72 configuration
    ]


def run_wideep_alltoall(ep_size: int, perf_filename: str, device: str = "cuda:0"):
    """
    Entry point for collect.py framework.

    Note: This function requires multi-GPU setup with mpirun (NOT torchrun).
    MNNVL uses MPI for symmetric memory management.
    """
    print(f"WideEP All-to-All benchmark requires multi-GPU setup with MPI.")
    print(f"Please run with: mpirun -np {ep_size} python {__file__}")


def main():
    """Main entry point."""
    rank, world_size, device = init_distributed()

    if world_size < 2:
        print("ERROR: This benchmark requires at least 2 GPUs.")
        print("IMPORTANT: Must use mpirun (NOT torchrun) because MNNVL uses MPI.")
        print("Usage: mpirun -np N python collect_wideep_alltoall.py")
        return

    try:
        run_benchmark(rank, world_size, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
