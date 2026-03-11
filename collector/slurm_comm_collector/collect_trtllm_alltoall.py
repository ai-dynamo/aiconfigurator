# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT-LLM MoE NVLink All-to-All Communication Benchmark (NVFP4 Only)

Unified benchmark for two NVLink-based All-to-All communication strategies:

  --kernel-source NVLinkTwoSided
      WideEPMoE backend.
      Phases: prepare + dispatch + combine [+ combine_low_precision].
      Supports multi-node.

  --kernel-source NVLinkOneSided
      CutlassMoE backend.
      Phases: dispatch + combine (no prepare).
      Single-node only.

Output: trtllm_alltoall_perf.txt

Usage (Slurm):
    srun --ntasks 4 --ntasks-per-node 4 --mpi=pmix \\
        python collect_trtllm_alltoall.py --kernel-source NVLinkOneSided

    srun --ntasks 8 --ntasks-per-node 4 --mpi=pmix \\
        python collect_trtllm_alltoall.py --kernel-source NVLinkTwoSided
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

from helper import benchmark_with_power, log_perf

KERNEL_SOURCE_TWO_SIDED = "NVLinkTwoSided"
KERNEL_SOURCE_ONE_SIDED = "NVLinkOneSided"
VALID_KERNEL_SOURCES = [KERNEL_SOURCE_TWO_SIDED, KERNEL_SOURCE_ONE_SIDED]


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
            # Skip if num_experts < ep_size or not evenly divisible
            if num_experts < ep_size or num_experts % ep_size != 0:
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


def create_mapping(rank: int, world_size: int, gpus_per_node: int):
    """
    Create TensorRT-LLM Mapping for MoE EP.

    Args:
        rank: Current rank
        world_size: Total number of ranks
        gpus_per_node: Number of GPUs per node

    Returns:
        Mapping object
    """
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=gpus_per_node,
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

    Distributes tokens across ranks and experts in a round-robin pattern that
    achieves balance at three levels:
      1. Rank-level: each rank receives the same number of token-expert pairs.
      2. Expert-level: each expert within a rank receives equal tokens.

    Example (ep_size=16, top_k=8):
      - 2 rank groups: [0-7] and [8-15]
      - token 0 → ranks [0-7],  expert offset 0
      - token 1 → ranks [8-15], expert offset 0
      - token 2 → ranks [0-7],  expert offset 1
      - token 3 → ranks [8-15], expert offset 1
      - ...

    Args:
        num_tokens: Number of tokens
        num_experts: Total number of experts
        top_k: Number of experts per token
        ep_size: Expert Parallelism size (number of GPUs)
        device: Target device

    Returns:
        Expert IDs tensor of shape [num_tokens, top_k]
    """
    experts_per_rank = num_experts // ep_size
    expert_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32, device=device)

    if ep_size >= top_k:
        # WideEP: group ranks into sets of top_k consecutive ranks
        num_rank_groups = ep_size // top_k
        for i in range(num_tokens):
            group = i % num_rank_groups
            expert_offset = (i // num_rank_groups) % experts_per_rank
            for k in range(top_k):
                target_rank = group * top_k + k
                expert_ids[i, k] = target_rank * experts_per_rank + expert_offset
    else:
        # Small EP (ep_size < top_k): each token sends to all ranks,
        # multiple experts per rank per token
        for i in range(num_tokens):
            for k in range(top_k):
                target_rank = k % ep_size
                intra_rank_idx = k // ep_size
                expert_offset = (i + intra_rank_idx) % experts_per_rank
                expert_ids[i, k] = target_rank * experts_per_rank + expert_offset

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
    else:
        raise ValueError(f"Unknown distribution: {test_case.distribution}")


def get_dispatch_data_size_bytes(
    num_tokens: int, hidden_size: int, top_k: int, moe_dtype: MoEDtype, ep_size: int
) -> int:
    """
    Calculate NVLink dispatch data volume per rank (remote traffic only).

    Hidden state is sent once per remote rank, not per expert slot.
    With balanced distribution a token reaches min(top_k, ep_size) distinct ranks;
    one of them is local, so remote_ranks = min(top_k, ep_size) - 1.

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension size
        top_k: Number of experts per token
        moe_dtype: MoE data type
        ep_size: Expert parallelism size (number of GPUs)

    Returns:
        Remote data size in bytes
    """
    if moe_dtype == MoEDtype.FLOAT16:
        per_token = hidden_size * 2
    elif moe_dtype == MoEDtype.FP8:
        per_token = hidden_size * 1
    elif moe_dtype == MoEDtype.NVFP4:
        per_token = (hidden_size // 2) + (hidden_size // 16)
    else:
        per_token = hidden_size * 2
    remote_ranks = min(top_k, ep_size) - 1
    return num_tokens * remote_ranks * per_token


def get_combine_data_size_bytes(num_tokens: int, hidden_size: int, top_k: int, ep_size: int) -> int:
    """
    Calculate NVLink combine data volume per rank (remote traffic only).

    Combine gathers expert outputs back. Hidden state is transferred once per
    remote rank, not per expert. Combine always uses bfloat16.

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension size
        top_k: Number of experts per token
        ep_size: Expert parallelism size (number of GPUs)

    Returns:
        Remote data size in bytes
    """
    remote_ranks = min(top_k, ep_size) - 1
    return num_tokens * remote_ranks * hidden_size * 2


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
    """
    Benchmark results for each All-to-All operation.

    NVLinkTwoSided populates all four fields.
    NVLinkOneSided only populates dispatch and combine (prepare and combine_lp stay 0).
    """

    dispatch_latency_ms: float
    combine_latency_ms: float
    prepare_latency_ms: float = 0.0
    combine_low_precision_latency_ms: float = 0.0


# ============================================================================
# NVLinkTwoSided benchmark  (WideEPMoE backend)
# ============================================================================
def benchmark_nvlink_two_sided(
    test_case: AlltoallTestCase,
    mapping,
    device: torch.device,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> AlltoallBenchmarkResult:
    """
    Benchmark NVLinkTwoSided All-to-All communication.
    Benchmarks four phases: prepare, dispatch, combine, combine_low_precision.

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

    # ============================================================================
    # Helper: benchmark using shared CUDA Graph infrastructure from helper.py.
    # Reuses benchmark_with_power which handles graph capture, fallback to eager
    # execution, warmup, timing, and optional power monitoring.
    # ============================================================================
    def _benchmark_op(func, label):
        """Benchmark a single operation using shared benchmark_with_power."""
        with benchmark_with_power(
            device=device,
            kernel_func=func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            allow_graph_fail=True,
        ) as results:
            latency = results["latency_ms"]
            if ep_rank == 0:
                mode = "CUDA Graph" if results["used_cuda_graph"] else "Eager"
                print(f"    [{label}] {mode} timing: {latency:.3f} ms")
        return latency

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

    prepare_latency = _benchmark_op(prepare_func, "prepare")
    # Run prepare once more to get valid alltoall_info for dispatch/combine
    alltoall_info, _ = prepare_func()
    torch.cuda.synchronize()

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

    dispatch_latency = _benchmark_op(dispatch_func, "dispatch")
    # Run dispatch once more to get valid output for combine
    dispatched = dispatch_func()
    torch.cuda.synchronize()

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

    combine_latency = _benchmark_op(combine_func, "combine")

    # ============================================================================
    # Benchmark: alltoall_combine_low_precision (do_reduce=False, use_low_precision_combine=True)
    # Only benchmark for NVFP4 dtype as low_precision_combine is most relevant for it
    # ============================================================================
    combine_low_precision_latency = 0.0
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

        try:
            # Test if low precision combine is supported
            combine_low_precision_func()
            torch.cuda.synchronize()
            combine_low_precision_latency = _benchmark_op(combine_low_precision_func, "combine_lp")
        except Exception:
            combine_low_precision_latency = 0.0

    return AlltoallBenchmarkResult(
        prepare_latency_ms=prepare_latency,
        dispatch_latency_ms=dispatch_latency,
        combine_latency_ms=combine_latency,
        combine_low_precision_latency_ms=combine_low_precision_latency,
    )


# ============================================================================
# NVLinkOneSided benchmark  (CutlassMoE backend)
# ============================================================================
def benchmark_nvlink_one_sided(
    test_case: AlltoallTestCase,
    mapping,
    device: torch.device,
    max_num_tokens: int,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> AlltoallBenchmarkResult:
    """
    Benchmark NVLinkOneSided All-to-All communication.

    Uses torch.ops.trtllm.moe_a2a_dispatch / moe_a2a_combine C++ ops directly
    (bypassing the Python MoeAlltoAll state machine) so that both dispatch and
    combine can be captured into CUDA Graphs.

    Args:
        test_case: Test case configuration
        mapping: TensorRT-LLM Mapping
        device: CUDA device
        max_num_tokens: Maximum number of tokens across all test cases.
            MoeAlltoAll workspace is a process-level singleton sized by this value.
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations

    Returns:
        AlltoallBenchmarkResult containing dispatch and combine latencies
    """
    from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll

    ep_rank = mapping.moe_ep_rank
    num_slots = test_case.num_experts
    act_dtype = torch.bfloat16

    # Calculate workspace size and create MoeAlltoAll (for workspace initialization)
    workspace_size = MoeAlltoAll.calculate_required_workspace_size(
        test_case.ep_size,
        test_case.top_k,
        max_num_tokens,
        test_case.hidden_size,
        act_dtype,
    )
    moe_a2a = MoeAlltoAll(
        mapping=mapping,
        max_num_tokens=max_num_tokens,
        top_k=test_case.top_k,
        num_experts=num_slots,
        workspace_size_per_rank=workspace_size,
    )

    # Extract workspace and metainfo for direct C++ op calls
    workspace = moe_a2a.workspace
    metainfo = moe_a2a.metainfo

    # Prepare test data
    hidden_states, hidden_states_sf, token_selected_slots, token_final_scales = prepare_test_data(test_case, device)

    runtime_max_tokens_per_rank = test_case.num_tokens  # balanced: all ranks have same count

    # Build payloads list matching CutlassFusedMoE.forward_chunk() convention:
    #   payloads = [hidden_states, (hidden_states_sf if NVFP4), token_selected_slots, token_final_scales]
    payloads = [hidden_states]
    if hidden_states_sf is not None:
        payloads.append(hidden_states_sf)
    payloads.append(token_selected_slots)
    payloads.append(token_final_scales)

    # ------------------------------------------------------------------
    # One bootstrap dispatch
    # ------------------------------------------------------------------
    _, combine_payload_offset = torch.ops.trtllm.moe_a2a_dispatch(
        token_selected_slots,
        payloads,
        workspace,
        metainfo,
        runtime_max_tokens_per_rank,
        ep_rank,
        test_case.ep_size,
        test_case.top_k,
        num_slots,
    )
    combine_payload_offset = int(combine_payload_offset)
    torch.cuda.synchronize()

    # Pre-fill the combine payload region with mock MoE output (bfloat16).
    combine_payload = torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace,
        ep_rank,
        test_case.ep_size,
        runtime_max_tokens_per_rank,
        combine_payload_offset,
        torch.bfloat16,
        test_case.hidden_size,
    )
    combine_payload.copy_(torch.randn_like(combine_payload))
    torch.cuda.synchronize()

    # ========================================================================
    # Helper: benchmark using shared CUDA Graph infrastructure from helper.py.
    # ========================================================================
    def _benchmark_op(func, label):
        """Benchmark a single operation using shared benchmark_with_power."""
        with benchmark_with_power(
            device=device,
            kernel_func=func,
            num_warmups=num_warmup,
            num_runs=num_iterations,
            allow_graph_fail=True,
        ) as results:
            latency = results["latency_ms"]
            if ep_rank == 0:
                mode = "CUDA Graph" if results["used_cuda_graph"] else "Eager"
                print(f"    [{label}] {mode} timing: {latency:.3f} ms")
        return latency

    # ========================================================================
    # Benchmark: dispatch
    # ========================================================================
    def dispatch_op():
        torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_slots,
            payloads,
            workspace,
            metainfo,
            runtime_max_tokens_per_rank,
            ep_rank,
            test_case.ep_size,
            test_case.top_k,
            num_slots,
        )

    dispatch_latency = _benchmark_op(dispatch_op, "dispatch")

    # ========================================================================
    # Benchmark: combine
    # ========================================================================
    moe_output = combine_payload.view(
        test_case.ep_size,
        runtime_max_tokens_per_rank,
        test_case.hidden_size,
    )

    def combine_op():
        torch.ops.trtllm.moe_a2a_combine(
            moe_output,
            test_case.num_tokens,
            workspace,
            metainfo,
            runtime_max_tokens_per_rank,
            ep_rank,
            test_case.ep_size,
            test_case.top_k,
            combine_payload_offset,
            True,  # payload_in_workspace: MoE output was written into workspace via get_combine_payload_tensor
        )

    combine_latency = _benchmark_op(combine_op, "combine")

    return AlltoallBenchmarkResult(
        dispatch_latency_ms=dispatch_latency,
        combine_latency_ms=combine_latency,
    )


# ============================================================================
# Performance logging
# ============================================================================
def log_alltoall_perf(
    test_case: AlltoallTestCase,
    op_name: str,
    latency_ms: float,
    framework: str,
    version: str,
    device_name: str,
    kernel_source: str,
    perf_filename: str,
):
    """
    Log All-to-All performance data in perf.txt compatible format.

    Args:
        test_case: Test case configuration
        op_name: Operation name (alltoall_prepare, alltoall_dispatch, alltoall_combine,
                 alltoall_combine_low_precision)
        latency_ms: Latency in milliseconds
        framework: Framework name (e.g., "TRTLLM")
        version: Framework version
        device_name: GPU device name
        kernel_source: Communication strategy (NVLinkTwoSided or NVLinkOneSided)
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
        kernel_source=kernel_source,
        perf_filename=perf_filename,
    )


def run_benchmark(
    rank: int,
    world_size: int,
    device: torch.device,
    kernel_source: str,
    output_file: str = "trtllm_alltoall_perf.txt",
    num_warmup: int = 3,
    num_iterations: int = 10,
):
    """
    Run All-to-All benchmark and log results.

    Args:
        rank: Current rank
        world_size: Total number of ranks
        device: CUDA device
        kernel_source: Communication strategy (NVLinkTwoSided or NVLinkOneSided)
        output_file: Output file path
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
    """
    import tensorrt_llm

    # Check MNNVL support
    if not check_mnnvl_support():
        if rank == 0:
            print("ERROR: MNNVL (NVLink) not supported on this hardware.")
            print("Both NVLinkTwoSided and NVLinkOneSided require NVLink connectivity.")
        return

    # Create mapping
    gpus_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", torch.cuda.device_count()))
    mapping = create_mapping(rank, world_size, gpus_per_node)

    # Get test cases
    test_cases = get_default_test_cases(world_size)

    framework = "TRTLLM"
    version = tensorrt_llm.__version__
    device_name = torch.cuda.get_device_name(device)

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"TensorRT-LLM MoE All-to-All Benchmark  [{kernel_source}]")
        print(f"{'=' * 70}")
        print(f"EP size: {world_size}")
        print(f"Device: {device_name}")
        print(f"TensorRT-LLM version: {version}")
        print(f"Number of test cases: {len(test_cases)}")
        print(f"MoE dtypes: {[d.value for d in DEFAULT_MOE_DTYPES]}")
        print(f"Output: {output_file}")
        print(f"{'=' * 70}\n")

    # For NVLinkOneSided, workspace is a process-level singleton sized by max_num_tokens
    max_num_tokens = max(tc.num_tokens for tc in test_cases) if test_cases else 0

    # Run benchmarks
    for idx, test_case in enumerate(test_cases):
        if rank == 0:
            print(f"[{idx + 1}/{len(test_cases)}] {test_case.description}")

        # Synchronize before benchmark
        if world_size > 1:
            dist.barrier()

        try:
            if kernel_source == KERNEL_SOURCE_TWO_SIDED:
                result = benchmark_nvlink_two_sided(
                    test_case,
                    mapping,
                    device,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                )
            elif kernel_source == KERNEL_SOURCE_ONE_SIDED:
                result = benchmark_nvlink_one_sided(
                    test_case,
                    mapping,
                    device,
                    max_num_tokens,
                    num_warmup=num_warmup,
                    num_iterations=num_iterations,
                )
            else:
                raise ValueError(f"Unknown kernel_source '{kernel_source}', expected one of {VALID_KERNEL_SOURCES}")

            # Log results (only rank 0)
            if rank == 0:
                # Calculate data sizes and bandwidths
                dispatch_data_size = get_dispatch_data_size_bytes(
                    test_case.num_tokens,
                    test_case.hidden_size,
                    test_case.top_k,
                    test_case.moe_dtype,
                    test_case.ep_size,
                )
                combine_data_size = get_combine_data_size_bytes(
                    test_case.num_tokens,
                    test_case.hidden_size,
                    test_case.top_k,
                    test_case.ep_size,
                )

                if result.prepare_latency_ms > 0:
                    print(f"  Prepare:  {result.prepare_latency_ms:.3f} ms")

                dispatch_bw = calculate_bandwidth_gbps(dispatch_data_size, result.dispatch_latency_ms)
                dispatch_kb = dispatch_data_size / 1024
                print(f"  Dispatch: {result.dispatch_latency_ms:.3f} ms ({dispatch_bw:.2f} GB/s, {dispatch_kb:.1f} KB)")

                combine_bw = calculate_bandwidth_gbps(combine_data_size, result.combine_latency_ms)
                combine_kb = combine_data_size / 1024
                print(f"  Combine:  {result.combine_latency_ms:.3f} ms ({combine_bw:.2f} GB/s, {combine_kb:.1f} KB)")

                if result.combine_low_precision_latency_ms > 0:
                    combine_lp_bw = calculate_bandwidth_gbps(
                        combine_data_size,
                        result.combine_low_precision_latency_ms,
                    )
                    print(
                        f"  Combine (low precision): {result.combine_low_precision_latency_ms:.3f} ms "
                        f"({combine_lp_bw:.2f} GB/s)"
                    )

                # Log each operation separately
                if result.prepare_latency_ms > 0:
                    log_alltoall_perf(
                        test_case,
                        "alltoall_prepare",
                        result.prepare_latency_ms,
                        framework,
                        version,
                        device_name,
                        kernel_source,
                        output_file,
                    )
                log_alltoall_perf(
                    test_case,
                    "alltoall_dispatch",
                    result.dispatch_latency_ms,
                    framework,
                    version,
                    device_name,
                    kernel_source,
                    output_file,
                )
                log_alltoall_perf(
                    test_case,
                    "alltoall_combine",
                    result.combine_latency_ms,
                    framework,
                    version,
                    device_name,
                    kernel_source,
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
                        kernel_source,
                        output_file,
                    )

        except Exception as e:
            if rank == 0:
                import traceback

                print(f"  ERROR: {e}")
                traceback.print_exc()

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Benchmark completed. Results saved to: {output_file}")
        print(f"{'=' * 70}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TensorRT-LLM MoE NVLink All-to-All Communication Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kernel-source",
        type=str,
        default=KERNEL_SOURCE_TWO_SIDED,
        choices=VALID_KERNEL_SOURCES,
        help=f"Communication strategy (default: {KERNEL_SOURCE_TWO_SIDED})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="trtllm_alltoall_perf.txt",
        help="Output file path for performance results (default: trtllm_alltoall_perf.txt)",
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
        print("Usage: srun --ntasks N --mpi=pmix python collect_trtllm_alltoall.py --kernel-source ...")
        return

    if rank == 0:
        print(f"Running {args.kernel_source} with {world_size} GPUs")

    try:
        run_benchmark(
            rank,
            world_size,
            device,
            kernel_source=args.kernel_source,
            output_file=args.output,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
