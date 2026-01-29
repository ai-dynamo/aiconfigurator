# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT-LLM WideEP NVLinkTwoSided All-to-All Communication Benchmark

This script collects All-to-All communication performance data for WideEP MoE
using NVLinkTwoSided communication strategy.

Supports both balanced and power-law token distributions to simulate real-world
workloads where some experts receive more tokens than others.

Usage:
    # Run with torchrun for multi-GPU:
    torchrun --nproc_per_node=4 collect_wideep_alltoall.py

    # Or with mpirun:
    mpirun -np 4 python collect_wideep_alltoall.py
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


@dataclass
class AlltoallTestCase:
    """Test case configuration for All-to-All benchmark."""

    num_tokens: int
    hidden_size: int
    num_experts: int
    top_k: int
    ep_size: int
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
                f"experts={self.num_experts}, topk={self.top_k}, dist={dist_str}"
            )


# Token distribution configurations: (distribution_type, power_law_alpha)
# - balanced: uniform distribution across all experts
# - power_law with α=1.01: slight imbalance
# - power_law with α=1.2: moderate imbalance (realistic workload)
DEFAULT_DISTRIBUTIONS = [
    (TokenDistribution.BALANCED, None),
    (TokenDistribution.POWER_LAW, 1.01),
    (TokenDistribution.POWER_LAW, 1.2),
]


def get_default_test_cases(ep_size: int) -> List[AlltoallTestCase]:
    """
    Generate default test cases for All-to-All benchmark.

    Args:
        ep_size: Expert Parallelism size (number of GPUs)

    Returns:
        List of test cases covering different token counts, model configs, and distributions
    """
    test_cases = []

    # Token counts to test (covering prefill and decode scenarios)
    token_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

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

            for distribution, alpha in DEFAULT_DISTRIBUTIONS:
                test_cases.append(
                    AlltoallTestCase(
                        num_tokens=num_tokens,
                        hidden_size=hidden_size,
                        num_experts=num_experts,
                        top_k=top_k,
                        ep_size=ep_size,
                        distribution=distribution,
                        power_law_alpha=alpha,
                    )
                )

    return test_cases


def init_distributed():
    """
    Initialize distributed environment.

    Returns:
        Tuple of (rank, world_size, device)
    """
    # Try torchrun environment first
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    # Try MPI environment
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if world_size > 1 and not dist.is_initialized():
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
        tp_size=world_size,
        pp_size=1,
    )
    # Set MoE EP parameters
    mapping.moe_ep_size = world_size
    mapping.moe_ep_rank = rank
    mapping.moe_tp_size = 1
    mapping.moe_tp_rank = 0

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

    # Build expert assignments based on token counts
    expert_assignments = []
    _, sorted_indices = torch.sort(tokens_per_expert, descending=True)
    for expert_id in sorted_indices.tolist():
        count = int(tokens_per_expert[expert_id].item())
        expert_assignments.extend([expert_id] * count)

    # Reshape to [num_tokens, top_k]
    expert_assignments = torch.tensor(expert_assignments, dtype=torch.int32, device=device)
    expert_ids = expert_assignments.reshape(top_k, num_tokens).T.contiguous()

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


def benchmark_nvlink_two_sided_alltoall(
    test_case: AlltoallTestCase,
    mapping,
    device: torch.device,
    num_warmup: int = 3,
    num_iterations: int = 10,
    num_distribution_samples: int = 5,
) -> Tuple[float, float, float]:
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
        Tuple of (prepare_latency_ms, dispatch_latency_ms, combine_latency_ms)
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

    # Number of slots (same as num_experts for simple case)
    num_slots = num_experts

    # For power-law, we sample multiple distributions to get stable results
    is_power_law = test_case.distribution == TokenDistribution.POWER_LAW
    num_samples = num_distribution_samples if is_power_law else 1

    # Generate test data
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # Pre-generate multiple expert ID distributions for power-law
    expert_id_samples = [generate_expert_ids(test_case, device) for _ in range(num_samples)]

    token_final_scales = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device) / top_k

    # All rank token counts (same for balanced case)
    all_rank_num_tokens = [num_tokens] * ep_size
    all_rank_max_num_tokens = max(all_rank_num_tokens)

    # Collect results across all distribution samples
    all_prepare_times = []
    all_dispatch_times = []
    all_combine_times = []

    for sample_idx, token_selected_slots in enumerate(expert_id_samples):
        # ============================================================================
        # Benchmark: prepare_dispatch
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
        # Benchmark: dispatch (All-to-All send)
        # ============================================================================
        def dispatch_func():
            return MnnvlMoe.mnnvl_moe_alltoallv(
                [hidden_states.clone(), None, token_selected_slots.clone(), token_final_scales.clone()],
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

        # ============================================================================
        # Benchmark: combine (All-to-All receive/reduce)
        # ============================================================================
        # Simulate MoE output (same shape as received hidden states)
        moe_output = torch.randn_like(recv_hidden_states)

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
                do_reduce=True,
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

    # Calculate average latencies across all samples
    prepare_latency = sum(all_prepare_times) / len(all_prepare_times)
    dispatch_latency = sum(all_dispatch_times) / len(all_dispatch_times)
    combine_latency = sum(all_combine_times) / len(all_combine_times)

    return prepare_latency, dispatch_latency, combine_latency


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

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"TensorRT-LLM WideEP NVLinkTwoSided All-to-All Benchmark")
        print(f"{'=' * 70}")
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"TensorRT-LLM version: {tensorrt_llm.__version__}")
        print(f"Number of test cases: {len(test_cases)}")
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

            prepare_latency, dispatch_latency, combine_latency = benchmark_nvlink_two_sided_alltoall(
                test_case,
                mapping,
                device,
                num_warmup=3,
                num_iterations=10,
            )

            # Log results (only rank 0)
            if rank == 0:
                total_latency = prepare_latency + dispatch_latency + combine_latency

                # Calculate bandwidth
                # Dispatch: each rank sends num_tokens * hidden_size * sizeof(bf16) bytes
                data_size_bytes = test_case.num_tokens * test_case.hidden_size * 2  # bf16 = 2 bytes
                dispatch_bw_gbps = (data_size_bytes / 1e9) / (dispatch_latency / 1000)  # GB/s
                combine_bw_gbps = (data_size_bytes / 1e9) / (combine_latency / 1000)

                print(f"  Prepare: {prepare_latency:.3f} ms")
                print(f"  Dispatch: {dispatch_latency:.3f} ms ({dispatch_bw_gbps:.2f} GB/s)")
                print(f"  Combine: {combine_latency:.3f} ms ({combine_bw_gbps:.2f} GB/s)")
                print(f"  Total: {total_latency:.3f} ms")

                # Format distribution string for logging
                if test_case.distribution == TokenDistribution.POWER_LAW:
                    distribution_str = f"power_law_{test_case.power_law_alpha}"
                else:
                    distribution_str = test_case.distribution.value

                # Log to file
                log_perf(
                    item_list=[
                        {
                            "num_tokens": test_case.num_tokens,
                            "hidden_size": test_case.hidden_size,
                            "num_experts": test_case.num_experts,
                            "top_k": test_case.top_k,
                            "ep_size": test_case.ep_size,
                            "distribution": distribution_str,
                            "prepare_latency_ms": prepare_latency,
                            "dispatch_latency_ms": dispatch_latency,
                            "combine_latency_ms": combine_latency,
                            "total_latency_ms": total_latency,
                            "dispatch_bw_gbps": dispatch_bw_gbps,
                            "combine_bw_gbps": combine_bw_gbps,
                        }
                    ],
                    framework="TRTLLM",
                    version=tensorrt_llm.__version__,
                    device_name=torch.cuda.get_device_name(device),
                    op_name="wideep_alltoall",
                    kernel_source="nvlink_two_sided",
                    perf_filename=output_file,
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

    Note: This function requires multi-GPU setup with torchrun/mpirun.
    When called from collect.py, it should spawn subprocess with proper
    distributed environment.
    """
    print(f"WideEP All-to-All benchmark requires multi-GPU setup.")
    print(f"Please run with: torchrun --nproc_per_node={ep_size} {__file__}")


def main():
    """Main entry point."""
    rank, world_size, device = init_distributed()

    if world_size < 2:
        print("ERROR: This benchmark requires at least 2 GPUs.")
        print("Usage: torchrun --nproc_per_node=N collect_wideep_alltoall.py")
        return

    try:
        run_benchmark(rank, world_size, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
