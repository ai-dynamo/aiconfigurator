# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM MoE NVLink All-to-All benchmark -> unified ``moe_a2a_perf``.

Benchmarks two NVLink-based All-to-All communication strategies:

  --kernel-source NVLinkTwoSided
      WideEPMoE backend (MnnvlMoe).
      Phases: prepare + dispatch + combine [+ combine_low_precision].
      Supports multi-node.

  --kernel-source NVLinkOneSided
      CutlassMoE backend (torch.ops.trtllm.moe_a2a_*).
      Phases: dispatch + combine (no prepare).
      Single-node only.

Rows are emitted in the unified ``moe_a2a`` schema consumed by
``aiconfigurator_core.sdk.operations.moe_comm.load_moe_a2a_data`` — the same
table (and CSV header) the sglang DeepEP collector
(``collector/wideep/sglang/collect_moe_a2a.py``) writes, whose row builder
and sidecar finalizer this module shares. The column mapping mirrors the
SDK's legacy adapter ``_adapt_legacy_trtllm_alltoall`` EXACTLY, so a
new-schema row and an adapted legacy row for the same measurement land on the
same store key with the same leaf value:

* ``kernel_source`` -> ``comm_backend``: ``NVLinkTwoSided -> nvlink_two_sided``,
  ``NVLinkOneSided -> nvlink_one_sided`` (``_LEGACY_TRTLLM_KERNEL_TO_BACKEND``).
* ``op_name`` -> ``phase``/``comm_dtype``: ``alltoall_prepare -> prepare``,
  ``alltoall_dispatch -> dispatch``, ``alltoall_combine -> combine`` (each
  keyed by the run's ``moe_dtype``), and
  ``alltoall_combine_low_precision -> combine`` keyed ``"fp4"``
  (``_LEGACY_TRTLLM_OP_TO_PHASE_DTYPE``).
* ``ep_size = world_size``; ``node_num = world_size // gpus_per_node`` from
  the launcher environment — explicit and asserted integral, replacing the
  legacy loader's fabricated ``max(1, ep_size // 4)`` derivation.
* ``sms = 0`` — "legacy alltoall rows carry no SM budget" (the adapter keys
  every trtllm row under sms 0).

UNITS: the measured per-phase latency is milliseconds
(``benchmark_with_power``), but the unified ``latency`` column is
MICROSECONDS — ``load_moe_a2a_data`` divides the column by 1000
(moe_comm.py:393 "collector records us; leaves are ms"), whereas the LEGACY
``trtllm_alltoall_perf`` rows were already ms and are adapted raw
(``_adapt_legacy_trtllm_alltoall`` docstring: "no us->ms conversion"). The
writer therefore emits ``latency_ms * 1000`` so the loader's leaf equals the
adapted-legacy leaf for the same measurement. The trtllm measurement has no
transmit/notify split (the adapter leaf carries only ``latency``), so —
exactly like the DeepEP LL precedent (moe_comm.py:232-234) — the whole
latency is recorded as ``transmit_us`` with ``notify_us = 0.0``; the loader
reads only their sum.

Provenance: this is a STANDALONE collector
(``provenance.STANDALONE_COLLECTOR_MODULES``) — it owns its case plan, its
classified failure log (``errors_trtllm_alltoall.rank<N>.json``), its parquet
finalization and its ``collection_meta.yaml`` sidecar, written by rank 0
post-finalize. The recorded runtime is the installed tensorrt_llm version,
gated against the manifest ``trtllm`` pin.

Emission order (D5): cases are expanded ascending on every non-token key axis
(``comm_dtype, hidden_size, topk, num_experts, num_tokens``; ``comm_backend``
and the world are fixed per run, ``sms`` is a constant 0), and within a case
rows are emitted ascending on ``(phase, comm_dtype)`` — so for an NVFP4
two-sided case: combine/fp4, combine/nvfp4, dispatch/nvfp4, prepare/nvfp4.

Whole-branch flag (pre-existing, unchanged here): the token-count sweep and
model shape grid are internal constants rather than declared
``cases/base_ops`` axes — the same undeclared-axis flag already ledgered for
the family's other standalone collectors.

Usage (see ``submit_trtllm_alltoall.sh``)::

    srun --ntasks 8 --ntasks-per-node 4 --mpi=pmix \\
        python collect_trtllm_alltoall.py --kernel-source NVLinkTwoSided \\
        --gpus-per-node 4 --output-path results/moe_a2a_NVLinkTwoSided.8gpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# Standalone entry point: `python collector/network/slurm/collect_trtllm_alltoall.py`
# must be able to import the `collector` package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collector.framework_manifest import get_collector_runtime
from collector.helper import benchmark_with_power, finalize_perf_files, log_perf
from collector.registry_types import PerfFile
from collector.wideep.sglang.collect_moe_a2a import (
    MoeA2AShape,
    _build_moe_a2a_row,
    write_moe_a2a_sidecar,
)

MODULE_NAME = "collector.network.slurm.collect_trtllm_alltoall"
#: The unified comm table this module writes (shared with the sglang DeepEP
#: collector — same file name, same frozen header).
OP_NAME = "moe_a2a"

#: Identity recorded in the perf-row prefix columns. The framework version is
#: the INSTALLED tensorrt_llm, gated against this manifest pin.
FRAMEWORK = "TRTLLM"
MANIFEST_FRAMEWORK = "trtllm"

KERNEL_SOURCE_TWO_SIDED = "NVLinkTwoSided"
KERNEL_SOURCE_ONE_SIDED = "NVLinkOneSided"
VALID_KERNEL_SOURCES = [KERNEL_SOURCE_TWO_SIDED, KERNEL_SOURCE_ONE_SIDED]

#: ``kernel_source`` -> ``comm_backend``. MUST mirror the SDK adapter's
#: ``_LEGACY_TRTLLM_KERNEL_TO_BACKEND`` (aic-core .../sdk/operations/
#: moe_comm.py) so new-schema rows key identically to adapted legacy rows.
KERNEL_SOURCE_TO_COMM_BACKEND = {
    KERNEL_SOURCE_TWO_SIDED: "nvlink_two_sided",
    KERNEL_SOURCE_ONE_SIDED: "nvlink_one_sided",
}

#: op_name -> (phase, comm_dtype); ``None`` means the case's ``moe_dtype``
#: passes through. MUST mirror the SDK adapter's
#: ``_LEGACY_TRTLLM_OP_TO_PHASE_DTYPE``: prepare/dispatch/standard combine are
#: keyed by the run dtype (standard-combine payload is physically bf16 but is
#: keyed by run dtype so every legacy leaf maps 1:1); the low-precision
#: combine kernel keys as "fp4".
OP_TO_PHASE_DTYPE = {
    "alltoall_prepare": ("prepare", None),
    "alltoall_dispatch": ("dispatch", None),
    "alltoall_combine": ("combine", None),
    "alltoall_combine_low_precision": ("combine", "fp4"),
}

#: The SDK's legacy trtllm adapter keys every alltoall row under sms 0
#: ("legacy alltoall rows carry no SM budget"); the new rows key identically.
ALLTOALL_SMS = 0

#: Classified failure log, rank-scoped (the output dir may be shared storage;
#: one file would be corrupted by concurrent writers).
ERRORS_FILENAME_TEMPLATE = "errors_trtllm_alltoall.rank{rank}.json"


class TrtllmAlltoallDeclarationError(RuntimeError):
    """A declared input (world layout, runtime pin, plan) does not resolve."""


class TrtllmAlltoallBenchmarkError(RuntimeError):
    """A queued case failed to execute. Classified failure record, never a skip."""


class TokenDistribution(Enum):
    """Token distribution strategies for expert selection."""

    BALANCED = "balanced"  # Uniform distribution across experts


class MoEDtype(Enum):
    """Supported MoE data types for All-to-All communication."""

    BFLOAT16 = "bfloat16"  # BFloat16
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
    moe_dtype: MoEDtype = MoEDtype.BFLOAT16
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

    def sort_key(self) -> tuple:
        """D5: ascending on every varying non-token key axis, token last.

        The consumer's store keys (below the fixed ``comm_backend``/world/
        ``sms`` levels for one run): ``comm_dtype`` sits above the shape axes
        ``hidden_size -> topk -> num_experts``, with ``num_tokens`` as the
        interpolated leaf axis.
        """
        return (
            self.moe_dtype.value,
            self.hidden_size,
            self.top_k,
            self.num_experts,
            self.num_tokens,
        )


def get_default_test_cases(ep_size: int) -> list[AlltoallTestCase]:
    """Generate the test-case plan for this world, in D5 sort order.

    The token sweep and model shapes are internal constants (pre-existing;
    declared-axes migration is a ledgered whole-branch flag). Drops are the
    op's universal math — ``num_experts % ep_size == 0`` (experts shard across
    EP ranks) — counted and logged, never silent; a zero-case expansion raises.
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

    dropped_ep = 0
    for hidden_size, num_experts, top_k in model_configs:
        # Experts must shard evenly across EP ranks.
        if num_experts < ep_size or num_experts % ep_size != 0:
            dropped_ep += 1
            continue
        for num_tokens in token_counts:
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

    test_cases.sort(key=AlltoallTestCase.sort_key)
    print(
        f"trtllm_alltoall: {len(test_cases)} cases for ep_size={ep_size} from "
        f"{len(model_configs)} shapes x {len(token_counts)} tokens x "
        f"{len(DEFAULT_MOE_DTYPES)} dtypes "
        f"(dropped: {dropped_ep} shapes with num_experts % ep_size != 0)",
        flush=True,
    )
    if not test_cases:
        raise TrtllmAlltoallDeclarationError(
            f"trtllm_alltoall expanded to zero cases for ep_size={ep_size}: all "
            f"{len(model_configs)} shapes were dropped (num_experts % ep_size != 0). "
            "Collecting nothing is never a clean completion — fix the world layout or the shapes."
        )
    return test_cases


# ---------------------------------------------------------------------------
# Distributed identity — from the launcher, never from filenames
# ---------------------------------------------------------------------------


def resolve_gpus_per_node(arg_value: int | None, env: dict[str, str]) -> int:
    """The GPUs-per-node divisor: explicit ``--gpus-per-node``, else Slurm's
    ``SLURM_NTASKS_PER_NODE`` (one task per GPU under this launcher), else
    raise. ``node_num`` is a persisted key column derived from this value, so
    guessing it (device count, a system table, a stale default) would
    silently mislabel every collected row.
    """
    if arg_value is not None:
        if arg_value <= 0:
            raise TrtllmAlltoallDeclarationError(f"--gpus-per-node must be positive, got {arg_value}")
        return arg_value
    if "SLURM_NTASKS_PER_NODE" in env:
        return int(env["SLURM_NTASKS_PER_NODE"])
    raise TrtllmAlltoallDeclarationError(
        "cannot derive gpus_per_node: pass --gpus-per-node explicitly (SLURM_NTASKS_PER_NODE is unset); "
        "node_num = world_size // gpus_per_node is a persisted key column"
    )


def derive_node_num(world_size: int, gpus_per_node: int) -> int:
    """``node_num = world_size // gpus_per_node``, asserted integral.

    ``gpus_per_node`` is the tasks-per-node the launcher actually placed (a
    sub-node job on a 4-GPU node runs with ``--ntasks-per-node`` = the GPU
    count, so the division is exact there too). This replaces the legacy
    loader's fabricated ``max(1, ep_size // 4)`` — which this derivation
    reproduces for the shipped GB200 NVL4 fleet — with the launcher's
    declared layout. No visible-device cross-check: a sub-node allocation
    legitimately sees more CUDA devices than it runs tasks.
    """
    if gpus_per_node <= 0:
        raise TrtllmAlltoallDeclarationError(f"gpus_per_node must be positive, got {gpus_per_node}")
    if world_size % gpus_per_node != 0:
        raise TrtllmAlltoallDeclarationError(
            f"WORLD_SIZE={world_size} is not an integral number of nodes at gpus_per_node={gpus_per_node}; "
            "moe_a2a rows persist node_num = world_size // gpus_per_node"
        )
    return world_size // gpus_per_node


def init_distributed():
    """
    Initialize distributed environment using Slurm with srun --mpi=pmix.

    MNNVL requires MPI for symmetric memory management.

    Returns:
        Tuple of (rank, world_size, device)
    """
    import torch
    import torch.distributed as dist
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


def require_mnnvl_support() -> None:
    """Raise unless MNNVL (Multi-Node NVLink) is supported on this hardware.

    Both kernel sources require NVLink connectivity; running anyway would
    benchmark nothing. Execute-or-raise — never a silent early return.
    """
    from tensorrt_llm._mnnvl_utils import MnnvlMemory

    try:
        MnnvlMemory.initialize()
        supported = MnnvlMemory.supports_mnnvl()
    except Exception as error:
        raise TrtllmAlltoallBenchmarkError(f"MNNVL support probe failed: {error}") from error
    if not supported:
        raise TrtllmAlltoallBenchmarkError(
            "MNNVL (NVLink) is not supported on this hardware; both NVLinkTwoSided and "
            "NVLinkOneSided require NVLink connectivity"
        )


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
    device,
):
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
    import torch

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
    device,
):
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
    if moe_dtype == MoEDtype.BFLOAT16:
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
    device,
):
    """
    Prepare test data based on MoE dtype.

    Args:
        test_case: Test case configuration
        device: CUDA device

    Returns:
        Tuple of (hidden_states, hidden_states_sf, token_selected_slots, token_final_scales)
        - hidden_states_sf is scale factor for NVFP4, None otherwise
    """
    import torch

    num_tokens = test_case.num_tokens
    hidden_size = test_case.hidden_size
    top_k = test_case.top_k
    moe_dtype = test_case.moe_dtype

    # Generate expert IDs
    token_selected_slots = generate_expert_ids(test_case, device)
    token_final_scales = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device) / top_k

    # Generate hidden states based on dtype
    hidden_states_sf = None

    if moe_dtype == MoEDtype.BFLOAT16:
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

    NVLinkTwoSided populates all four latency fields.
    NVLinkOneSided only populates dispatch and combine (prepare and combine_lp
    stay 0, meaning "phase not run" — no row is emitted for them).

    ``combine_low_precision_error`` carries a failed low-precision-combine
    probe out to the run loop so it lands in the classified failure log —
    a missing fp4 row must be explained, never a silent drop.
    """

    dispatch_latency_ms: float
    combine_latency_ms: float
    prepare_latency_ms: float = 0.0
    combine_low_precision_latency_ms: float = 0.0
    combine_low_precision_error: Optional[BaseException] = field(default=None, compare=False)


def _benchmark_op(func, label, device, ep_rank, num_warmup, num_iterations):
    """Benchmark a single operation using shared benchmark_with_power.

    ``measure_power=False``: the moe_a2a table emits NO power column BY RULING
    (Task 4 review, see ``collect_moe_a2a._power_columns``): the ruling is per
    TABLE — HT/LL and trtllm writers share one file and one header, and
    ``helper.log_perf`` writes the header from the first row — so this writer
    must not sample or emit power either.
    """
    with benchmark_with_power(
        device=device,
        kernel_func=func,
        num_warmups=num_warmup,
        num_runs=num_iterations,
        measure_power=False,
        allow_graph_fail=True,
    ) as results:
        latency = results["latency_ms"]
        if ep_rank == 0:
            mode = "CUDA Graph" if results["used_cuda_graph"] else "Eager"
            print(f"    [{label}] {mode} timing: {latency:.3f} ms")
    return latency


# ============================================================================
# NVLinkTwoSided benchmark  (WideEPMoE backend)
# ============================================================================
def benchmark_nvlink_two_sided(
    test_case: AlltoallTestCase,
    mapping,
    device,
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
    import torch
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

    prepare_latency = _benchmark_op(prepare_func, "prepare", device, ep_rank, num_warmup, num_iterations)
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

    dispatch_latency = _benchmark_op(dispatch_func, "dispatch", device, ep_rank, num_warmup, num_iterations)
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

    combine_latency = _benchmark_op(combine_func, "combine", device, ep_rank, num_warmup, num_iterations)

    # ============================================================================
    # Benchmark: alltoall_combine_low_precision (do_reduce=False, use_low_precision_combine=True)
    # Only benchmark for NVFP4 dtype as low_precision_combine is most relevant for it
    # ============================================================================
    combine_low_precision_latency = 0.0
    combine_low_precision_error: BaseException | None = None
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
            combine_low_precision_func()
            torch.cuda.synchronize()
            combine_low_precision_latency = _benchmark_op(
                combine_low_precision_func, "combine_lp", device, ep_rank, num_warmup, num_iterations
            )
        except Exception as error:
            # The fp4 row is dropped, but never silently: the error rides the
            # result out to the run loop's classified failure record.
            combine_low_precision_error = error

    return AlltoallBenchmarkResult(
        prepare_latency_ms=prepare_latency,
        dispatch_latency_ms=dispatch_latency,
        combine_latency_ms=combine_latency,
        combine_low_precision_latency_ms=combine_low_precision_latency,
        combine_low_precision_error=combine_low_precision_error,
    )


# ============================================================================
# NVLinkOneSided benchmark  (CutlassMoE backend)
# ============================================================================
def benchmark_nvlink_one_sided(
    test_case: AlltoallTestCase,
    mapping,
    device,
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
    import torch
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

    dispatch_latency = _benchmark_op(dispatch_op, "dispatch", device, ep_rank, num_warmup, num_iterations)

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

    combine_latency = _benchmark_op(combine_op, "combine", device, ep_rank, num_warmup, num_iterations)

    return AlltoallBenchmarkResult(
        dispatch_latency_ms=dispatch_latency,
        combine_latency_ms=combine_latency,
    )


# ============================================================================
# Unified-schema row construction
# ============================================================================


def result_measurements(result: AlltoallBenchmarkResult) -> list[tuple[str, float]]:
    """The ``(op_name, latency_ms)`` measurements a result actually ran.

    A 0.0 latency means "phase not run" (one-sided has no prepare; the
    low-precision combine only runs for NVFP4 and its failure is separately
    classified), never a measured zero.
    """
    measurements = []
    if result.prepare_latency_ms > 0:
        measurements.append(("alltoall_prepare", result.prepare_latency_ms))
    measurements.append(("alltoall_dispatch", result.dispatch_latency_ms))
    measurements.append(("alltoall_combine", result.combine_latency_ms))
    if result.combine_low_precision_latency_ms > 0:
        measurements.append(("alltoall_combine_low_precision", result.combine_low_precision_latency_ms))
    return measurements


def build_unified_rows(
    test_case: AlltoallTestCase,
    result: AlltoallBenchmarkResult,
    *,
    kernel_source: str,
    node_num: int,
) -> list[dict]:
    """The case's unified ``moe_a2a`` rows, in D5 emission order.

    Shares ``_build_moe_a2a_row`` with the sglang DeepEP collector — one frozen
    header for the whole table. Mapping and units per the module docstring:
    the SDK adapter's kernel_source/op_name maps mirrored exactly, latency
    emitted in MICROSECONDS (``latency_ms * 1000``) because
    ``load_moe_a2a_data`` divides by 1000 while the adapted legacy ms rows are
    stored raw; the whole latency is ``transmit_us`` with ``notify_us = 0.0``
    (no transmit/notify split exists in this measurement — the DeepEP LL
    precedent).

    Rows sort ascending on ``(phase, comm_dtype)``, the two store levels that
    vary within a case (``comm_backend``/world/shape/tokens are fixed here and
    ascending across cases via ``AlltoallTestCase.sort_key``).
    """
    comm_backend = KERNEL_SOURCE_TO_COMM_BACKEND[kernel_source]
    shape = MoeA2AShape(
        hidden_size=test_case.hidden_size,
        topk=test_case.top_k,
        num_experts=test_case.num_experts,
    )
    keyed = []
    for op_name, latency_ms in result_measurements(result):
        phase, comm_dtype = OP_TO_PHASE_DTYPE[op_name]
        if comm_dtype is None:
            comm_dtype = test_case.moe_dtype.value
        keyed.append((phase, comm_dtype, latency_ms))
    keyed.sort(key=lambda item: (item[0], item[1]))

    return [
        _build_moe_a2a_row(
            comm_backend=comm_backend,
            phase=phase,
            comm_dtype=comm_dtype,
            ep_size=test_case.ep_size,
            node_num=node_num,
            shape=shape,
            num_tokens=test_case.num_tokens,
            sms=ALLTOALL_SMS,
            transmit_us=latency_ms * 1000.0,
            notify_us=0.0,
        )
        for phase, comm_dtype, latency_ms in keyed
    ]


def case_plan_ids(cases: list[AlltoallTestCase], *, kernel_source: str, node_num: int) -> list[str]:
    """Stable case identifiers for ``provenance.case_plan_hash``.

    ``kernel_source`` and the world (``ep_size``/``node_num``) are part of the
    identity: the same grid collected under another strategy or on another
    world is a different attested plan.
    """
    ids = []
    for case in cases:
        payload = {
            "distribution": case.distribution.value,
            "ep_size": case.ep_size,
            "hidden_size": case.hidden_size,
            "kernel_source": kernel_source,
            "moe_dtype": case.moe_dtype.value,
            "node_num": node_num,
            "num_experts": case.num_experts,
            "num_tokens": case.num_tokens,
            "topk": case.top_k,
        }
        ids.append(f"{MODULE_NAME}:run_case:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return ids


# ============================================================================
# Provenance
# ============================================================================


def resolve_runtime_meta(installed_version: str) -> dict:
    """The sidecar ``runtime`` block, from the manifest ``trtllm`` pin.

    The INSTALLED version is what actually produced the data, so it is what
    is recorded — but it must equal the pin, or the collected rows would be
    attributed to a runtime that never ran them.
    """
    from packaging.version import InvalidVersion, Version

    runtime = get_collector_runtime(MANIFEST_FRAMEWORK)
    try:
        installed_public = Version(installed_version).public
    except InvalidVersion as error:
        raise TrtllmAlltoallDeclarationError(f"invalid installed tensorrt_llm version {installed_version!r}") from error
    if installed_public != Version(runtime.version).public:
        raise TrtllmAlltoallDeclarationError(
            f"trtllm alltoall collection requires tensorrt_llm {runtime.version} (manifest trtllm pin), "
            f"found {installed_version}; use {runtime.image()}"
        )
    image, sep, digest = runtime.image().partition("@")
    meta = {"framework": runtime.framework, "version": installed_version, "image": image}
    if sep:
        meta["image_digest"] = digest
    return meta


def record_failure(
    output_dir: Path,
    case: AlltoallTestCase,
    error: BaseException,
    *,
    rank: int,
    kernel_source: str,
    node_num: int,
    op_name: str | None = None,
) -> None:
    """Append one classified failure record (rank-scoped ``errors_*.json``)."""
    record = {
        "module": MODULE_NAME,
        "op": op_name or OP_NAME,
        "classification": "unexpected",
        "error_type": type(error).__name__,
        "error": str(error),
        "rank": rank,
        "case": {
            "kernel_source": kernel_source,
            "moe_dtype": case.moe_dtype.value,
            "distribution": case.distribution.value,
            "ep_size": case.ep_size,
            "node_num": node_num,
            "hidden_size": case.hidden_size,
            "topk": case.top_k,
            "num_experts": case.num_experts,
            "num_tokens": case.num_tokens,
        },
    }
    path = output_dir / ERRORS_FILENAME_TEMPLATE.format(rank=rank)
    existing = json.loads(path.read_text()) if path.exists() else []
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2))
    print(f"  FAILED {op_name or 'case'} ({case.description}): {error}", flush=True)


# ============================================================================
# Run loop
# ============================================================================


def run_benchmark(
    rank: int,
    world_size: int,
    device,
    *,
    kernel_source: str,
    gpus_per_node: int,
    output_dir: Path,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> None:
    """Run the All-to-All benchmark, emit unified rows, finalize + attest.

    Rank 0 owns the perf file, the parquet finalization and the
    ``collection_meta.yaml`` sidecar; every rank records its own classified
    failures. After each case the ranks ``all_reduce(MAX)`` their failure
    flags so they agree whether the case produced data and the next collective
    stays in lockstep.
    """
    import tensorrt_llm
    import torch
    import torch.distributed as dist

    require_mnnvl_support()

    node_num = derive_node_num(world_size, gpus_per_node)
    mapping = create_mapping(rank, world_size, gpus_per_node)
    test_cases = get_default_test_cases(world_size)
    case_ids = case_plan_ids(test_cases, kernel_source=kernel_source, node_num=node_num)

    version = tensorrt_llm.__version__
    runtime_meta = resolve_runtime_meta(version)
    device_name = torch.cuda.get_device_name(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    perf_path = str(output_dir / PerfFile.MOE_A2A.value)

    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"TensorRT-LLM MoE All-to-All Benchmark  [{kernel_source}]")
        print(f"{'=' * 70}")
        print(f"EP size: {world_size} ({node_num} node(s) x {gpus_per_node} GPUs)")
        print(f"Device: {device_name}")
        print(f"TensorRT-LLM version: {version}")
        print(f"Number of test cases: {len(test_cases)}")
        print(f"MoE dtypes: {[d.value for d in DEFAULT_MOE_DTYPES]}")
        print(f"Output: {perf_path}")
        print(f"{'=' * 70}\n")

    # For NVLinkOneSided, workspace is a process-level singleton sized by max_num_tokens
    max_num_tokens = max(tc.num_tokens for tc in test_cases)

    failure_count = 0
    for idx, test_case in enumerate(test_cases):
        if rank == 0:
            print(f"[{idx + 1}/{len(test_cases)}] {test_case.description}")

        # Synchronize before benchmark
        if world_size > 1:
            dist.barrier()

        failed = 0
        lp_failed = 0
        result = None
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
        except Exception as error:
            failed = 1
            record_failure(output_dir, test_case, error, rank=rank, kernel_source=kernel_source, node_num=node_num)
        else:
            if result.combine_low_precision_error is not None:
                lp_failed = 1
                record_failure(
                    output_dir,
                    test_case,
                    result.combine_low_precision_error,
                    rank=rank,
                    kernel_source=kernel_source,
                    node_num=node_num,
                    op_name="alltoall_combine_low_precision",
                )

        # Every rank must agree whether this case produced data, or the next
        # collective desyncs; the fp4 leg's flag rides along so a
        # peer-rank-only failure still suppresses the row and is counted.
        if world_size > 1:
            agreement = torch.tensor([failed, lp_failed], dtype=torch.int32, device=device)
            dist.all_reduce(agreement, op=dist.ReduceOp.MAX)
            failed, lp_failed = (int(value) for value in agreement.tolist())
        if failed:
            failure_count += 1
            continue
        if lp_failed:
            failure_count += 1
            result.combine_low_precision_latency_ms = 0.0

        if rank == 0:
            _print_case_summary(test_case, result)
            for row in build_unified_rows(test_case, result, kernel_source=kernel_source, node_num=node_num):
                log_perf(
                    item_list=[row],
                    framework=FRAMEWORK,
                    version=version,
                    device_name=device_name,
                    op_name=OP_NAME,
                    kernel_source=kernel_source,
                    perf_filename=perf_path,
                )

    if rank == 0:
        converted = finalize_perf_files([perf_path])
        if not converted:
            raise TrtllmAlltoallBenchmarkError(
                f"trtllm alltoall produced no rows: all {failure_count}/{len(test_cases)} cases failed. "
                f"See {output_dir / ERRORS_FILENAME_TEMPLATE.format(rank='*')} — a whole-family failure "
                "is a collector problem to fix, not a partial collection to publish."
            )
        [parquet_path] = converted
        meta_path = write_moe_a2a_sidecar(
            output_dir,
            runtime_meta=runtime_meta,
            case_ids=case_ids,
            parquet_path=parquet_path,
            failure_count=failure_count,
            module_name=MODULE_NAME,
        )
        print(f"\n{'=' * 70}")
        print(f"Benchmark completed. Wrote {parquet_path} and {meta_path} ({failure_count} classified failures)")
        print(f"{'=' * 70}")


def _print_case_summary(test_case: AlltoallTestCase, result: AlltoallBenchmarkResult) -> None:
    """Console bandwidth summary (derived quantities, not persisted)."""
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
        combine_lp_bw = calculate_bandwidth_gbps(combine_data_size, result.combine_low_precision_latency_ms)
        print(f"  Combine (low precision): {result.combine_low_precision_latency_ms:.3f} ms ({combine_lp_bw:.2f} GB/s)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TensorRT-LLM MoE NVLink All-to-All Communication Benchmark (unified moe_a2a schema)",
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
        "--gpus-per-node",
        type=int,
        default=None,
        help="GPUs per node in this allocation; node_num = world_size // gpus-per-node (a persisted "
        "key column). Defaults to SLURM_NTASKS_PER_NODE; raises when neither is available.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default=os.getcwd(),
        help="Output DIRECTORY for moe_a2a_perf parquet + collection_meta.yaml (default: cwd)",
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
    return parser.parse_args(argv)


def main():
    """Main entry point."""
    args = parse_args()

    import torch.distributed as dist

    gpus_per_node = resolve_gpus_per_node(args.gpus_per_node, dict(os.environ))
    rank, world_size, device = init_distributed()

    if world_size < 2:
        raise TrtllmAlltoallDeclarationError(
            "this benchmark requires at least 2 GPUs; launch with "
            "srun --ntasks N --mpi=pmix python collect_trtllm_alltoall.py ..."
        )

    if rank == 0:
        print(f"Running {args.kernel_source} with {world_size} GPUs")

    try:
        run_benchmark(
            rank,
            world_size,
            device,
            kernel_source=args.kernel_source,
            gpus_per_node=gpus_per_node,
            output_dir=Path(args.output_path),
            num_warmup=args.warmup,
            num_iterations=args.iterations,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
