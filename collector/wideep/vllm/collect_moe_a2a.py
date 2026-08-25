# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone multi-node vLLM DeepEP serving-parity collector.

Launch one rank per local GPU under a multi-node torch/Slurm world.  The
collector calls vLLM's prepare/finalize implementations directly and allocates
no model weights.  vLLM imports are intentionally confined to
``VllmBenchmarkAdapter`` so population and persistence can be tested on CPU.

The adapter mirrors vLLM v0.24.0 commit ``ee0da84``:

* HT buffer/default SMS: ``all2all.py:169-171,218-257`` and constructor call
  ``all2all_utils.py:159-169``.
* LL capacity/buffer and constructor call:
  ``all2all.py:286-343`` and ``all2all_utils.py:171-204``.
* v2 ElasticBuffer, theoretical SMS, and constructor call:
  ``all2all.py:1022-1083`` and ``all2all_utils.py:205-233``.

Each queued case either returns two rows or a classified failure.  A failed
write fails the run; an all-failed run is never finalized.  Classified
per-case failures remain observable data and do not demote a finalized table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collector import provenance
from collector.framework_manifest import get_collector_runtime
from collector.helper import finalize_perf_files, log_perf, stale_output_artifacts
from collector.registry_types import PerfFile
from collector.wideep.backend_contracts import contract_for
from collector.wideep.distributed_lifecycle import StageAgreement, agree_stage, raise_for_stage
from collector.wideep.sglang.collect_moe_a2a import (
    DistIdentity,
    MoeA2ADeclarationError,
    MoeA2AShape,
    PhaseTiming,
    _build_moe_a2a_row,
    derive_dist_identity,
    get_moe_a2a_workload_grid,
)
from collector.wideep.sglang.collect_moe_a2a import (
    _git_collector_ref as _unattested_git_collector_ref,
)

MODULE_NAME = "collector.wideep.vllm.collect_moe_a2a"
OP_NAME = "moe_a2a"
FRAMEWORK = "vLLM"
KERNEL_SOURCE = "deepep"
COMM_DTYPE = "default"
PHASES = ("combine", "dispatch")
BACKENDS = ("deepep_ht", "deepep_ll", "deepep_v2")
BACKEND_CONTRACTS = {backend: contract_for("vllm", backend) for backend in BACKENDS}
SUPPORTED_WORLD_SIZES = (8, 16, 32)
SUPPORTED_NODE_COUNTS = (2, 4)
HT_SMS = 20
LL_SMS = 0
HT_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024
DEEPEP_LL_SUPPORTED_HIDDEN_SIZES = frozenset({2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192})
TARGET_VLLM_SOURCE_COMMIT = "ee0da84ab9e04ac7610e28580af62c365e898389"
LEGACY_DEEPEP_COMMIT = "73b6ea4a439ba03a695563f9fd242c8e4b02b37c"
V2_DEEPEP_COMMIT = "b306af06afd412c88e51e71802951606e40b7358"
LEGACY_NVL4_PATCH = _REPO_ROOT / "collector" / "wideep" / "vllm" / "patches" / "deepep_73b_nvl4.patch"
ERRORS_FILENAME_TEMPLATE = "errors_moe_a2a_vllm.rank{rank}.json"


class VllmMoeA2AError(RuntimeError):
    """Base class for classified vLLM collector failures."""


class VllmMoeA2ADeclarationError(MoeA2ADeclarationError, VllmMoeA2AError):
    """The declared population, runtime, or world layout is invalid."""


class VllmMoeA2ABenchmarkError(VllmMoeA2AError):
    """A queued benchmark or persistence operation failed."""


class VllmMoeA2APeerError(VllmMoeA2AError):
    """Another distributed rank failed the same queued case."""


@dataclass(frozen=True, order=True)
class VllmMoeA2ACase:
    """One persisted physical point before its two communication-phase rows."""

    comm_backend: str
    inference_phase: str
    shape: MoeA2AShape
    num_tokens: int
    sms: int | None
    capacity: int

    def persisted_key(self, *, ep_size: int, node_num: int, sms: int | None = None) -> tuple[Any, ...]:
        resolved_sms = self.sms if sms is None else sms
        if resolved_sms is None:
            raise VllmMoeA2ADeclarationError("deepep_v2 persisted key requires live theoretical SMS")
        return (
            self.comm_backend,
            COMM_DTYPE,
            ep_size,
            node_num,
            self.shape.hidden_size,
            self.shape.topk,
            self.shape.num_experts,
            self.num_tokens,
            resolved_sms,
        )

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.comm_backend,
            -1 if self.sms is None else self.sms,
            self.shape.hidden_size,
            self.shape.topk,
            self.shape.num_experts,
            self.num_tokens,
        )


@dataclass(frozen=True)
class BenchmarkResult:
    """Adapter result for one full dispatch/combine round-trip."""

    timings: dict[str, PhaseTiming]
    sms: int
    capacity: int


class BenchmarkAdapter(Protocol):
    """Injectable boundary between pure orchestration and the GPU runtime."""

    def benchmark(self, case: VllmMoeA2ACase) -> BenchmarkResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CaseFailure:
    case: VllmMoeA2ACase
    error_type: str
    error: str


@dataclass(frozen=True)
class CollectionResult:
    rows: list[dict[str, Any]]
    failures: list[CaseFailure]
    resolved_cases: list[VllmMoeA2ACase]


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise VllmMoeA2ADeclarationError(f"token capacity must be positive, got {value}")
    return 1 << (value - 1).bit_length()


def get_vllm_moe_a2a_shapes(
    *,
    required_expert_parallel_size: int | None = None,
    supported_hidden_sizes: frozenset[int] | None = None,
) -> list[MoeA2AShape]:
    """Resolve correlated WideEP shapes through the vLLM case population."""
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model

    recipes = get_common_moe_test_cases(
        backend="vllm",
        required_expert_parallel_size=required_expert_parallel_size,
        supported_hidden_sizes=supported_hidden_sizes,
    )
    shapes = {
        MoeA2AShape(int(recipe.hidden_size), int(recipe.topk), int(recipe.num_experts))
        for recipe in recipes
        if is_wideep_moe_model(recipe.model_name)
    }
    ordered = sorted(shapes)
    if not ordered:
        raise VllmMoeA2ADeclarationError(
            "backend='vllm' resolved zero declared WideEP MoE shapes; empty collection is not success"
        )
    return ordered


def build_case_plan(
    *,
    shapes: list[MoeA2AShape],
    grid: dict[str, list[int]],
    world_size: int,
    node_num: int,
    backends: tuple[str, ...] = BACKENDS,
) -> list[VllmMoeA2ACase]:
    """Build deterministic multi-node cases with serving-owned policies.

    v2 uses the union of the two declared token axes.  Tokens on the LL axis
    use generation/cudagraph mode; remaining HT-only tokens use context/eager
    mode.  This gives every persisted v2 key exactly one invocation identity.
    """
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise VllmMoeA2ADeclarationError(f"vLLM moe_a2a supports world sizes {SUPPORTED_WORLD_SIZES}, got {world_size}")
    if node_num not in SUPPORTED_NODE_COUNTS:
        raise VllmMoeA2ADeclarationError(f"vLLM moe_a2a supports node counts {SUPPORTED_NODE_COUNTS}, got {node_num}")
    if world_size % node_num:
        raise VllmMoeA2ADeclarationError(f"world_size={world_size} is not divisible by node_num={node_num}")
    unknown = sorted(set(backends) - set(BACKENDS))
    if unknown:
        raise VllmMoeA2ADeclarationError(f"unsupported DeepEP backend(s): {unknown}")
    if not backends:
        raise VllmMoeA2ADeclarationError("no DeepEP backends selected")

    ht_tokens = _validated_axis(grid, "ht_token_counts")
    ll_tokens = _validated_axis(grid, "ll_token_counts")
    invalid_shapes = [shape for shape in shapes if shape.num_experts % world_size != 0]
    if invalid_shapes:
        raise VllmMoeA2ADeclarationError(
            f"declared vLLM moe_a2a shapes are not divisible by world_size={world_size}: "
            f"{invalid_shapes}; request the EP constraint from get_vllm_moe_a2a_shapes "
            "instead of filtering a generated plan"
        )
    unsupported_ll_shapes = [
        shape
        for shape in shapes
        if "deepep_ll" in backends and shape.hidden_size not in DEEPEP_LL_SUPPORTED_HIDDEN_SIZES
    ]
    if unsupported_ll_shapes:
        raise VllmMoeA2ADeclarationError(
            "declared vLLM DeepEP-LL shapes use hidden sizes outside the pinned kernel capability "
            f"{sorted(DEEPEP_LL_SUPPORTED_HIDDEN_SIZES)}: {unsupported_ll_shapes}; request the hidden-size "
            "constraint from get_vllm_moe_a2a_shapes instead of filtering a generated plan"
        )

    cases: list[VllmMoeA2ACase] = []
    for shape in shapes:
        if "deepep_ht" in backends:
            cases.extend(
                VllmMoeA2ACase("deepep_ht", "context", shape, tokens, HT_SMS, HT_BUFFER_SIZE_BYTES)
                for tokens in ht_tokens
            )
        if "deepep_ll" in backends:
            cases.extend(
                VllmMoeA2ACase("deepep_ll", "generation", shape, tokens, LL_SMS, tokens) for tokens in ll_tokens
            )
        if "deepep_v2" in backends:
            for tokens in sorted(set(ht_tokens) | set(ll_tokens)):
                phase = "generation" if tokens in ll_tokens else "context"
                cases.append(
                    VllmMoeA2ACase(
                        "deepep_v2",
                        phase,
                        shape,
                        tokens,
                        None,
                        _next_power_of_two(tokens),
                    )
                )

    cases.sort(key=VllmMoeA2ACase.sort_key)
    if not cases:
        raise VllmMoeA2ADeclarationError(
            f"moe_a2a expanded to zero cases: {len(shapes)} shapes, world_size={world_size}, node_num={node_num}"
        )
    invocation_ids = [
        (
            case.comm_backend,
            case.inference_phase,
            case.shape,
            case.num_tokens,
            case.sms,
            case.capacity,
        )
        for case in cases
    ]
    if len(invocation_ids) != len(set(invocation_ids)):
        raise VllmMoeA2ADeclarationError("duplicate vLLM DeepEP benchmark invocation identity")
    static_keys = [case.persisted_key(ep_size=world_size, node_num=node_num) for case in cases if case.sms is not None]
    if len(static_keys) != len(set(static_keys)):
        raise VllmMoeA2ADeclarationError("duplicate vLLM DeepEP persisted key")
    print(
        f"moe_a2a vllm: {len(cases)} cases from {len(shapes)} shapes for world_size={world_size}, node_num={node_num}",
        flush=True,
    )
    return cases


def _validated_axis(grid: dict[str, list[int]], name: str) -> list[int]:
    values = grid.get(name)
    if not isinstance(values, list) or not values:
        raise VllmMoeA2ADeclarationError(f"{name} must be a non-empty list")
    parsed = [int(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise VllmMoeA2ADeclarationError(f"{name} must contain only positive values")
    if len(parsed) != len(set(parsed)):
        raise VllmMoeA2ADeclarationError(f"{name} contains duplicate token counts")
    return sorted(parsed)


def case_plan_ids(cases: list[VllmMoeA2ACase], *, world_size: int, node_num: int) -> list[str]:
    ids = []
    for case in cases:
        payload = {
            "capacity": case.capacity,
            "comm_backend": case.comm_backend,
            "ep_size": world_size,
            "hidden_size": case.shape.hidden_size,
            "inference_phase": case.inference_phase,
            "node_num": node_num,
            "num_experts": case.shape.num_experts,
            "num_tokens": case.num_tokens,
            "sms": case.sms,
            "topk": case.shape.topk,
        }
        ids.append(f"{MODULE_NAME}:benchmark:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return ids


def select_canary_cases(cases: list[VllmMoeA2ACase]) -> list[VllmMoeA2ACase]:
    """Keep one smallest case for every serving backend/inference phase."""
    selected: dict[tuple[str, str], VllmMoeA2ACase] = {}
    for case in cases:
        selected.setdefault((case.comm_backend, case.inference_phase), case)
    return sorted(selected.values(), key=VllmMoeA2ACase.sort_key)


def collect_with_adapter(
    cases: list[VllmMoeA2ACase],
    *,
    adapter: BenchmarkAdapter,
    world_size: int,
    node_num: int,
    failure_agreement: Callable[[bool], bool] = bool,
    stage_agreement: StageAgreement | None = None,
) -> CollectionResult:
    """Pure collection loop used by GPU-free tests and the torchrun entrypoint."""
    rows: list[dict[str, Any]] = []
    failures: list[CaseFailure] = []
    resolved_cases: list[VllmMoeA2ACase] = []
    resolved_keys: set[tuple[Any, ...]] = set()
    agreement = stage_agreement or (
        lambda stage, failed: failure_agreement(failed) if stage.endswith(":benchmark") else failed
    )
    prepare_error: BaseException | None = None
    try:
        prepare = getattr(adapter, "prepare", None)
        if prepare is not None:
            prepare(cases)
    except BaseException as error:
        prepare_error = error
    raise_for_stage(
        agree_stage(
            "adapter_prepare",
            prepare_error,
            agreement=agreement,
            peer_error_type=VllmMoeA2APeerError,
        )
    )
    for case_index, case in enumerate(cases):
        case_rows: list[dict[str, Any]] = []
        resolved: VllmMoeA2ACase | None = None
        local_error: BaseException | None = None
        try:
            result = adapter.benchmark(case)
            contract = BACKEND_CONTRACTS[case.comm_backend]
            if contract.kernel_source != KERNEL_SOURCE:
                raise VllmMoeA2ABenchmarkError(
                    f"{case.comm_backend} contract kernel source is {contract.kernel_source!r}"
                )
            expected_sms = {"deepep_ht": HT_SMS, "deepep_ll": LL_SMS}.get(case.comm_backend)
            if expected_sms is not None and result.sms != expected_sms:
                raise VllmMoeA2ABenchmarkError(
                    f"{case.comm_backend} returned sms={result.sms}, expected {expected_sms}"
                )
            if result.capacity != case.capacity:
                raise VllmMoeA2ABenchmarkError(
                    f"{case.comm_backend} returned capacity={result.capacity}, expected {case.capacity}"
                )
            if set(result.timings) != set(PHASES):
                raise VllmMoeA2ABenchmarkError(
                    f"{case.comm_backend} returned phases {sorted(result.timings)}, expected {list(PHASES)}"
                )
            resolved = replace(case, sms=result.sms)
            key = resolved.persisted_key(ep_size=world_size, node_num=node_num)
            if key in resolved_keys:
                raise VllmMoeA2ADeclarationError(f"duplicate resolved persisted key: {key}")
            for phase in PHASES:
                timing = result.timings[phase]
                case_rows.append(
                    _build_moe_a2a_row(
                        comm_backend=case.comm_backend,
                        phase=phase,
                        ep_size=world_size,
                        node_num=node_num,
                        shape=case.shape,
                        num_tokens=case.num_tokens,
                        sms=result.sms,
                        transmit_us=timing.transmit_us,
                        notify_us=timing.notify_us,
                        comm_dtype=COMM_DTYPE,
                    )
                )
        except Exception as error:
            local_error = error

        # Agree before accepting this case's rows. A rank-local post-kernel
        # validation failure must not let rank 0 publish the same physical
        # key as successfully collected.
        outcome = agree_stage(
            f"case:{case_index}:benchmark",
            local_error,
            agreement=agreement,
            peer_error_type=VllmMoeA2APeerError,
        )
        if outcome.failed:
            assert outcome.error is not None
            error = outcome.error
            failures.append(CaseFailure(case, type(error).__name__, str(error)))
            continue

        assert resolved is not None
        resolved_keys.add(resolved.persisted_key(ep_size=world_size, node_num=node_num))
        resolved_cases.append(resolved)
        rows.extend(case_rows)
    close_error: BaseException | None = None
    try:
        adapter.close()
    except BaseException as error:
        close_error = error
    raise_for_stage(
        agree_stage(
            "adapter_close",
            close_error,
            agreement=agreement,
            peer_error_type=VllmMoeA2APeerError,
        )
    )
    return CollectionResult(rows, failures, resolved_cases)


def _init_process_groups(identity: DistIdentity):
    """Initialize separate benchmark and CPU failure-agreement groups."""
    import torch
    import torch.distributed as dist

    if identity.node_num not in SUPPORTED_NODE_COUNTS or identity.world_size not in SUPPORTED_WORLD_SIZES:
        raise VllmMoeA2ADeclarationError(
            f"vLLM collector requires nodes {SUPPORTED_NODE_COUNTS} and world size "
            f"{SUPPORTED_WORLD_SIZES}; got nodes={identity.node_num}, world={identity.world_size}"
        )
    torch.cuda.set_device(identity.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{identity.master_addr}:{identity.master_port}",
            world_size=identity.world_size,
            rank=identity.rank,
            device_id=torch.device(f"cuda:{identity.local_rank}"),
        )
    ranks = list(range(identity.world_size))
    benchmark_group = dist.new_group(ranks, backend="nccl")
    agreement_group = dist.new_group(ranks, backend="gloo")
    return benchmark_group, agreement_group


class VllmBenchmarkAdapter:
    """vLLM-specific invocation adapter for v0.24.0 commit ``ee0da84``."""

    def __init__(
        self,
        group,
        identity: DistIdentity,
        *,
        allow_mnnvl: bool = True,
        disable_nvlink: bool = False,
        warmups: int = 3,
        runs: int = 10,
    ):
        self.group = group
        self.identity = identity
        self.allow_mnnvl = allow_mnnvl
        self.disable_nvlink = disable_nvlink
        self.warmups = warmups
        self.runs = runs
        self._buffer = None
        self._buffer_key: tuple[Any, ...] | None = None
        self._ll_rdma_bytes: int | None = None
        self._ll_num_qps_per_rank: int | None = None
        self._v2_max_tokens: dict[tuple[Any, ...], int] = {}
        self.runtime_capability: dict[str, str] | None = None

    def prepare(self, cases: list[VllmMoeA2ACase]) -> None:
        """Size reusable DeepEP buffers once from the declared case plan.

        A serving worker owns a long-lived DeepEP buffer. Recreating NVSHMEM
        for every benchmark key leaks/invalidates GDR resources on real
        multi-node runs, so the collector follows that lifecycle too.
        """
        if not cases:
            raise VllmMoeA2ADeclarationError("cannot prepare an empty vLLM moe_a2a case plan")
        backends = {case.comm_backend for case in cases}
        if len(backends) != 1:
            raise VllmMoeA2ADeclarationError(
                f"one adapter may prepare exactly one DeepEP backend, found {sorted(backends)}"
            )
        backend = next(iter(backends))
        if backend == "deepep_ll":
            import deep_ep

            self._ll_rdma_bytes = max(
                deep_ep.Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank=case.capacity,
                    hidden=case.shape.hidden_size,
                    num_ranks=self.identity.world_size,
                    num_experts=case.shape.num_experts,
                )
                for case in cases
            )
            self._ll_num_qps_per_rank = max(case.shape.num_experts // self.identity.world_size for case in cases)
        elif backend == "deepep_v2":
            for case in cases:
                key = self._runtime_buffer_key(case)
                self._v2_max_tokens[key] = max(self._v2_max_tokens.get(key, 0), case.capacity)

    def close(self) -> None:
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
            self._buffer_key = None

    def benchmark(self, case: VllmMoeA2ACase) -> BenchmarkResult:
        """Run exact public prepare/finalize calls with synthetic activations."""
        torch, prepare_finalize, quant_config, reduce_impl = self._make_runtime(case)
        torch.manual_seed(17 + self.identity.rank)
        tokens = torch.randn(
            (case.num_tokens, case.shape.hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        topk_ids = torch.stack(
            [torch.randperm(case.shape.num_experts, device="cuda")[: case.shape.topk] for _ in range(case.num_tokens)]
        ).to(torch.int64)
        topk_weights = torch.full(
            topk_ids.shape,
            1.0 / case.shape.topk,
            device="cuda",
            dtype=torch.float32,
        )

        def one_round() -> tuple[float, float]:
            with self._forward_context(case):
                if case.comm_backend == "deepep_ll":
                    self._buffer.clean_low_latency_buffer(
                        case.capacity,
                        case.shape.hidden_size,
                        case.shape.num_experts,
                    )
                start_dispatch = torch.cuda.Event(enable_timing=True)
                end_dispatch = torch.cuda.Event(enable_timing=True)
                start_dispatch.record()
                prepared = prepare_finalize.prepare(
                    tokens,
                    topk_weights,
                    topk_ids,
                    case.shape.num_experts,
                    None,
                    False,
                    quant_config,
                )
                end_dispatch.record()
                expert_x = prepared[0]
                # Stand in for expert output without charging that materialization
                # to the communication combine phase.
                expert_output = expert_x.clone()
                output = torch.empty_like(tokens)
                start_combine = torch.cuda.Event(enable_timing=True)
                end_combine = torch.cuda.Event(enable_timing=True)
                start_combine.record()
                prepare_finalize.finalize(
                    output,
                    expert_output,
                    topk_weights,
                    topk_ids,
                    False,
                    reduce_impl,
                )
                end_combine.record()
                torch.cuda.synchronize()
            return start_dispatch.elapsed_time(end_dispatch), start_combine.elapsed_time(end_combine)

        for _ in range(self.warmups):
            one_round()
        dispatch_ms = combine_ms = 0.0
        for _ in range(self.runs):
            dispatch, combine = one_round()
            dispatch_ms += dispatch
            combine_ms += combine
        sms = case.sms
        if case.comm_backend == "deepep_v2":
            sms = int(
                self._buffer.get_theoretical_num_sms(
                    num_experts=case.shape.num_experts,
                    num_topk=case.shape.topk,
                )
            )
        assert sms is not None
        return BenchmarkResult(
            timings={
                "dispatch": PhaseTiming(dispatch_ms * 1000.0 / self.runs, 0.0),
                "combine": PhaseTiming(combine_ms * 1000.0 / self.runs, 0.0),
            },
            sms=sms,
            capacity=case.capacity,
        )

    def _forward_context(self, case: VllmMoeA2ACase):
        """Mirror the pinned serving model-forward scope around MoE calls."""
        import torch
        from vllm.config import ParallelConfig, VllmConfig
        from vllm.forward_context import set_forward_context

        # vLLM ee0da84 wraps model execution in set_forward_context
        # (vllm/v1/worker/gpu/model_runner.py:1271-1280). DeepEP v2 decode
        # reads DPMetadata to bound its receive allocation
        # (prepare_finalize/deepep_v2.py:121-140). Supply the serving DP/EP
        # identity and its already-coordinated per-rank token vector so this
        # standalone adapter does not fall back to a false DP=1 context.
        parallel_config = ParallelConfig(
            data_parallel_size=self.identity.world_size,
            data_parallel_size_local=self.identity.gpus_per_node,
            data_parallel_rank=self.identity.rank,
            data_parallel_rank_local=self.identity.local_rank,
            is_moe_model=True,
            enable_expert_parallel=True,
        )
        num_tokens_across_dp = torch.full(
            (self.identity.world_size,),
            case.num_tokens,
            dtype=torch.int32,
            device="cpu",
        )
        return set_forward_context(
            None,
            VllmConfig(parallel_config=parallel_config),
            num_tokens=case.num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
        )

    @staticmethod
    def _runtime_buffer_key(case: VllmMoeA2ACase) -> tuple[Any, ...]:
        if case.comm_backend == "deepep_v2":
            # ElasticBuffer fixes hidden/top-k at construction. Token capacity
            # is overprovisioned per signature by prepare().
            return (case.comm_backend, case.shape.hidden_size, case.shape.topk)
        # Legacy HT/LL buffers can serve every declared shape when their byte
        # allocation and QP count are sized to the maximum plan requirement.
        return (case.comm_backend,)

    def _make_runtime(self, case: VllmMoeA2ACase):
        import deep_ep
        import torch
        from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
            DeepEPHTPrepareAndFinalize,
        )
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
            DeepEPLLPrepareAndFinalize,
        )
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceDelegate,
            TopKWeightAndReduceNoOP,
        )

        local_experts = case.shape.num_experts // self.identity.world_size
        rank_offset = self.identity.rank * local_experts
        quant_config = FusedMoEQuantConfig.make(None)
        buffer_key = self._runtime_buffer_key(case)
        if self._buffer is not None and self._buffer_key != buffer_key:
            self.close()
        if case.comm_backend == "deepep_ht":
            if self._buffer is None:
                is_internode = self.identity.node_num > 1
                self._buffer = deep_ep.Buffer(
                    group=self.group,
                    num_nvl_bytes=case.capacity,
                    num_rdma_bytes=case.capacity if is_internode else 0,
                    low_latency_mode=False,
                    num_qps_per_rank=HT_SMS // 2 if is_internode else 1,
                    explicitly_destroy=True,
                )
                self._buffer_key = buffer_key
                num_scaleout_ranks = int(self._buffer.runtime.get_num_rdma_ranks())
                self._record_capability(
                    backend=case.comm_backend,
                    topology_source="legacy_compile_time",
                    num_scaleout_ranks=num_scaleout_ranks,
                    num_scaleup_ranks=self.identity.world_size // num_scaleout_ranks,
                )
            deep_ep.Buffer.set_num_sms(HT_SMS)
            prepare_finalize = DeepEPHTPrepareAndFinalize(
                self._buffer,
                num_dispatchers=self.identity.world_size,
                dp_size=self.identity.world_size,
                rank_expert_offset=rank_offset,
            )
            reduce_impl = TopKWeightAndReduceNoOP()
        elif case.comm_backend == "deepep_ll":
            if self._buffer is None:
                rdma_bytes = self._ll_rdma_bytes
                if rdma_bytes is None:
                    rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                        num_max_dispatch_tokens_per_rank=case.capacity,
                        hidden=case.shape.hidden_size,
                        num_ranks=self.identity.world_size,
                        num_experts=case.shape.num_experts,
                    )
                num_qps_per_rank = self._ll_num_qps_per_rank or local_experts
                self._buffer = deep_ep.Buffer(
                    group=self.group,
                    num_nvl_bytes=HT_BUFFER_SIZE_BYTES,
                    num_rdma_bytes=rdma_bytes,
                    low_latency_mode=True,
                    num_qps_per_rank=num_qps_per_rank,
                    allow_nvlink_for_low_latency_mode=not self.disable_nvlink,
                    explicitly_destroy=True,
                    allow_mnnvl=self.allow_mnnvl,
                )
                self._buffer_key = buffer_key
                num_scaleout_ranks = int(self._buffer.runtime.get_num_rdma_ranks())
                self._record_capability(
                    backend=case.comm_backend,
                    topology_source="legacy_compile_time",
                    num_scaleout_ranks=num_scaleout_ranks,
                    num_scaleup_ranks=self.identity.world_size // num_scaleout_ranks,
                )
            prepare_finalize = DeepEPLLPrepareAndFinalize(
                self._buffer,
                max_tokens_per_rank=case.capacity,
                num_dispatchers=self.identity.world_size,
                use_fp8_dispatch=False,
            )
            reduce_impl = TopKWeightAndReduceDelegate()
        elif case.comm_backend == "deepep_v2":
            from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
                DeepEPV2PrepareAndFinalize,
            )

            if self._buffer is None:
                self._buffer = deep_ep.ElasticBuffer(
                    group=self.group,
                    num_max_tokens_per_rank=self._v2_max_tokens.get(buffer_key, case.capacity),
                    hidden=case.shape.hidden_size,
                    num_topk=case.shape.topk,
                    use_fp8_dispatch=False,
                    allow_hybrid_mode=True,
                    prefer_overlap_with_compute=False,
                    allow_multiple_reduction=False,
                    explicitly_destroy=True,
                )
                self._buffer_key = buffer_key
                num_scaleout_ranks, num_scaleup_ranks = self._buffer.get_logical_domain_size()
                num_rdma_ranks, num_nvlink_ranks = self._buffer.get_physical_domain_size()
                self._record_capability(
                    backend=case.comm_backend,
                    topology_source="nccl_lsa",
                    num_scaleout_ranks=num_scaleout_ranks,
                    num_scaleup_ranks=num_scaleup_ranks,
                    num_rdma_ranks=num_rdma_ranks,
                    num_nvlink_ranks=num_nvlink_ranks,
                )
            prepare_finalize = DeepEPV2PrepareAndFinalize(
                buffer=self._buffer,
                num_dispatchers=self.identity.world_size,
                dp_size=self.identity.world_size,
                rank_expert_offset=rank_offset,
                num_experts=case.shape.num_experts,
                num_topk=case.shape.topk,
                use_fp8_dispatch=False,
                use_cudagraph=case.inference_phase == "generation",
            )
            reduce_impl = TopKWeightAndReduceNoOP()
        else:  # defensive: population rejects this before execution
            raise VllmMoeA2ADeclarationError(f"unsupported backend {case.comm_backend!r}")
        return torch, prepare_finalize, quant_config, reduce_impl

    def _record_capability(self, *, backend: str, topology_source: str, **domains: int) -> None:
        capability = {
            "backend": backend,
            "topology_source": topology_source,
            **{key: str(value) for key, value in domains.items()},
        }
        if int(capability["num_scaleout_ranks"]) * int(capability["num_scaleup_ranks"]) != self.identity.world_size:
            raise VllmMoeA2ABenchmarkError(
                f"DeepEP reported an invalid logical topology: {capability}, world={self.identity.world_size}"
            )
        if self.runtime_capability is not None and self.runtime_capability != capability:
            raise VllmMoeA2ABenchmarkError(
                f"DeepEP topology changed between cases: {self.runtime_capability} != {capability}"
            )
        self.runtime_capability = capability


def attest_vllm_runtime(
    *,
    source_root: str | Path,
    backend: str,
    observed_abi: dict[str, str],
    observed_image_digest: str,
    installed_version_getter=distribution_version,
    source_commit_getter=None,
    live_abi_getter=None,
) -> dict[str, Any]:
    """Attest the live installed version, backend ABI, and source checkout."""
    source_root = Path(source_root)
    if source_commit_getter is None:
        source_commit_getter = _git_head
    if live_abi_getter is None:
        live_abi_getter = _observe_live_backend_abi
    if backend not in BACKENDS:
        raise VllmMoeA2ADeclarationError(f"unsupported runtime backend {backend!r}")
    installed_version = installed_version_getter("vllm")
    source_commit = source_commit_getter(source_root)
    runtime = get_collector_runtime("vllm", workload="wideep")
    from packaging.version import InvalidVersion, Version

    try:
        version_matches = Version(installed_version).public == Version(runtime.version).public
    except InvalidVersion as error:
        raise VllmMoeA2ADeclarationError(f"invalid installed vLLM version {installed_version!r}") from error
    if not version_matches:
        raise VllmMoeA2ADeclarationError(
            f"wideep_vllm requires package version {runtime.version}, found {installed_version}"
        )
    if runtime.source_commit != TARGET_VLLM_SOURCE_COMMIT or source_commit != runtime.source_commit:
        raise VllmMoeA2ADeclarationError(
            f"vLLM source must be {TARGET_VLLM_SOURCE_COMMIT}, found {source_commit!r} at {source_root}"
        )
    required_abi = runtime.abi_for_backend(backend)
    system = observed_abi.get("system")
    if backend in ("deepep_ht", "deepep_ll"):
        expected_scaleup_ranks = "4" if system in ("gb200", "gb300") else "8"
        required_abi["deep_ep_scaleup_ranks"] = expected_scaleup_ranks
        if expected_scaleup_ranks == "4":
            required_abi["deep_ep_patch_sha256"] = _file_sha256(LEGACY_NVL4_PATCH)
    else:
        required_abi["deep_ep_topology_source"] = "nccl_lsa"
    mismatched_abi = {
        key: {"expected": expected, "observed": observed_abi.get(key)}
        for key, expected in required_abi.items()
        if observed_abi.get(key) != expected
    }
    if mismatched_abi:
        raise VllmMoeA2ADeclarationError(
            f"wideep_vllm ABI mismatch: {mismatched_abi}; full observed ABI={observed_abi}"
        )
    overlay_sha = observed_abi.get("deep_ep_overlay_wheel_sha256")
    overlay_required = backend == "deepep_v2" or bool(observed_abi.get("deep_ep_patch_sha256"))
    overlay_invalid = bool(overlay_sha) and (
        len(overlay_sha) != 64 or any(char not in "0123456789abcdef" for char in overlay_sha)
    )
    if overlay_invalid or (overlay_required and not overlay_sha):
        raise VllmMoeA2ADeclarationError(
            f"{backend} requires an attested DeepEP overlay wheel SHA256; observed ABI={observed_abi}"
        )
    live_abi = live_abi_getter(backend)
    live_mismatches = {
        key: {"expected": expected, "observed": live_abi.get(key)}
        for key, expected in (
            ("torch", required_abi["torch"]),
            ("deep_ep_api", required_abi["deep_ep_api"]),
        )
        if live_abi.get(key) != expected
    }
    expected_deepep = required_abi["deep_ep"]
    if overlay_sha:
        # The exact overlay is identified by the verified wheel SHA and build
        # metadata. Patched legacy wheels deliberately report ``+local``.
        pass
    elif expected_deepep[:7] not in live_abi.get("deep_ep_distribution", ""):
        live_mismatches["deep_ep_distribution"] = {
            "expected": f"*+{expected_deepep[:7]}",
            "observed": live_abi.get("deep_ep_distribution"),
        }
    if backend == "deepep_v2" and live_abi.get("nccl") != required_abi["nccl"]:
        live_mismatches["nccl"] = {"expected": required_abi["nccl"], "observed": live_abi.get("nccl")}
    if live_mismatches:
        raise VllmMoeA2ADeclarationError(f"wideep_vllm live ABI mismatch: {live_mismatches}; full live ABI={live_abi}")
    matching_images = [
        (variant, image.partition("@"))
        for variant, image in runtime.images.items()
        if image.partition("@")[1] and image.partition("@")[2] == observed_image_digest
    ]
    if not matching_images:
        expected_digests = sorted(image.partition("@")[2] for image in runtime.images.values() if "@" in image)
        raise VllmMoeA2ADeclarationError(
            f"wideep_vllm image digest mismatch: expected one of {expected_digests!r}, found {observed_image_digest!r}"
        )
    image_variant, (image, _, digest) = matching_images[0]
    return {
        "framework": runtime.framework,
        "version": installed_version,
        "image": image,
        "image_variant": image_variant,
        "image_digest": digest,
        "source_commit": source_commit,
        "abi": observed_abi,
        "live_abi": live_abi,
    }


def _observe_live_backend_abi(backend: str) -> dict[str, str]:
    """Observe fields that must come from the process actually loading DeepEP."""

    import deep_ep
    import torch
    from packaging.version import Version

    api = "ElasticBuffer" if backend == "deepep_v2" else "Buffer"
    nccl_version = ""
    if backend == "deepep_v2":
        try:
            nccl_version = distribution_version("nvidia-nccl-cu13")
        except Exception as error:
            raise VllmMoeA2ADeclarationError("deepep_v2 cannot observe nvidia-nccl-cu13") from error
    return {
        "torch": Version(distribution_version("torch")).public,
        "torch_cuda": str(torch.version.cuda),
        "deep_ep_distribution": distribution_version("deep_ep"),
        "deep_ep_api": api if hasattr(deep_ep, api) else "missing",
        "deep_ep_import": str(Path(deep_ep.__file__).resolve()),
        "nccl": nccl_version,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_collector_ref(repo_root: Path) -> str:
    """Return the host-attested collector SHA, rejecting observable drift."""

    declared = os.environ.get("AIC_COLLECTOR_REF", "")
    if not declared:
        return _unattested_git_collector_ref(repo_root)
    if len(declared) != 40 or any(character not in "0123456789abcdef" for character in declared.lower()):
        raise VllmMoeA2ADeclarationError(f"invalid AIC_COLLECTOR_REF {declared!r}")
    observed = _unattested_git_collector_ref(repo_root)
    if observed != "unknown" and observed.lower() != declared.lower():
        raise VllmMoeA2ADeclarationError(
            f"host-attested collector ref {declared} does not match mounted checkout {observed}"
        )
    return declared.lower()


def _activate_v2_rdma_rate_tool(expected_rate_gbps: str) -> None:
    """Expose the host-attested ibstat wrapper to DeepEP V2 inside Python."""

    tool_root_raw = os.environ.get("AIC_IBSTAT_TOOL_ROOT", "")
    loader = os.environ.get("AIC_IBSTAT_LOADER_BASENAME", "")
    if not tool_root_raw or not loader:
        raise VllmMoeA2ADeclarationError("deepep_v2 requires staged host ibstat attestation")
    tool_root = Path(tool_root_raw).resolve(strict=True)
    if not str(tool_root).startswith("/tmp/aic-vllm-a2a-") or tool_root.name != "host-rdma-tools":
        raise VllmMoeA2ADeclarationError(f"unsafe AIC_IBSTAT_TOOL_ROOT {tool_root}")
    if not re.fullmatch(r"ld[^/]*\.so(?:\.[0-9]+)*", loader):
        raise VllmMoeA2ADeclarationError(f"invalid AIC_IBSTAT_LOADER_BASENAME {loader!r}")
    for required in (
        tool_root / "bin" / "ibstat.real",
        tool_root / "lib" / loader,
    ):
        if not required.is_file():
            raise VllmMoeA2ADeclarationError(f"missing staged ibstat runtime file {required}")

    wrapper_dir = _REPO_ROOT / "collector" / "wideep" / "vllm" / "slurm" / "host_tools"
    wrapper = wrapper_dir / "ibstat"
    if not wrapper.is_file():
        raise VllmMoeA2ADeclarationError(f"missing attested ibstat wrapper {wrapper}")
    os.environ["PATH"] = f"{wrapper_dir}:{os.environ.get('PATH', '')}"
    try:
        result = subprocess.run(
            ["ibstat", "mlx5_0"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VllmMoeA2ADeclarationError(f"staged ibstat runtime check failed: {error}") from error
    rates = re.findall(r"^\s*Rate:\s*([0-9]+)\s*$", result.stdout, re.MULTILINE)
    if len(rates) != 1 or rates[0] != expected_rate_gbps:
        raise VllmMoeA2ADeclarationError(
            f"staged ibstat rate mismatch: expected {expected_rate_gbps!r}, observed {rates!r}"
        )


def _git_head(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as git_error:
        # `git` is unusable in two indistinguishable-to-the-caller ways: the
        # binary may be absent (FileNotFoundError), or present but unable to
        # read the tree and exiting non-zero -- a staged checkout carries
        # `.git/HEAD` without the object store `rev-parse` needs.  Both resolve
        # from the checkout metadata staged next to the source.
        try:
            head = (path / ".git" / "HEAD").read_text().strip()
            if head.startswith("ref: "):
                head = (path / ".git" / head.removeprefix("ref: ")).read_text().strip()
            if len(head) != 40 or any(character not in "0123456789abcdef" for character in head.lower()):
                raise ValueError(f"invalid HEAD {head!r}")
            return head
        except (OSError, ValueError) as error:
            raise VllmMoeA2ADeclarationError(
                f"cannot attest vLLM source commit at {path}: {git_error}; "
                f"reading {path / '.git' / 'HEAD'} also failed: {error}"
            ) from error


def _write_rows(
    rows: list[dict[str, Any]],
    *,
    perf_path: Path,
    runtime_meta: dict[str, str],
    device_name: str,
) -> None:
    if perf_path.exists():
        raise VllmMoeA2ABenchmarkError(f"stale staging file exists at {perf_path}; resume/merge must be explicit")
    for row in rows:
        if not log_perf(
            item_list=[row],
            framework=FRAMEWORK,
            version=runtime_meta["version"],
            device_name=device_name,
            op_name=OP_NAME,
            kernel_source=KERNEL_SOURCE,
            perf_filename=str(perf_path),
        ):
            raise VllmMoeA2ABenchmarkError(f"write loss: log_perf rejected row key {_row_key(row)}")
    with perf_path.open(newline="") as handle:
        persisted = list(csv.DictReader(handle))
    if len(persisted) != len(rows):
        raise VllmMoeA2ABenchmarkError(
            f"write loss: emitted {len(rows)} rows but staging file contains {len(persisted)}"
        )
    keys = [_row_key(row) for row in persisted]
    if len(keys) != len(set(keys)):
        raise VllmMoeA2ABenchmarkError("staging file contains duplicate full moe_a2a keys")


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[name]
        for name in (
            "comm_backend",
            "phase",
            "comm_dtype",
            "ep_size",
            "node_num",
            "hidden_size",
            "topk",
            "num_experts",
            "num_tokens",
            "sms",
        )
    )


def _write_failures(output_dir: Path, failures: list[CaseFailure], identity: DistIdentity) -> None:
    records = []
    for failure in failures:
        records.append(
            {
                "module": MODULE_NAME,
                "op": OP_NAME,
                "classification": "unexpected",
                "error_type": failure.error_type,
                "error": failure.error,
                "rank": identity.rank,
                "case": {
                    "comm_backend": failure.case.comm_backend,
                    "inference_phase": failure.case.inference_phase,
                    "ep_size": identity.world_size,
                    "node_num": identity.node_num,
                    "hidden_size": failure.case.shape.hidden_size,
                    "topk": failure.case.shape.topk,
                    "num_experts": failure.case.shape.num_experts,
                    "num_tokens": failure.case.num_tokens,
                    "sms": failure.case.sms,
                    "capacity": failure.case.capacity,
                },
            }
        )
    if records:
        path = output_dir / ERRORS_FILENAME_TEMPLATE.format(rank=identity.rank)
        path.write_text(json.dumps(records, indent=2))


def _write_rank_error(output_dir: Path, error: BaseException, identity: DistIdentity, *, stage: str) -> Path:
    """Record fatal non-case failures, such as DeepEP teardown errors."""
    path = output_dir / ERRORS_FILENAME_TEMPLATE.format(rank=identity.rank)
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(
        {
            "module": MODULE_NAME,
            "op": OP_NAME,
            "classification": "unexpected",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "rank": identity.rank,
            "case": None,
        }
    )
    path.write_text(json.dumps(records, indent=2))
    return path


def _write_sidecar(
    output_dir: Path,
    *,
    runtime_meta: dict[str, str],
    case_ids: list[str],
    parquet_path: Path,
    failure_count: int,
) -> Path:
    import pyarrow.parquet as pq

    if not case_ids:
        raise VllmMoeA2ADeclarationError("refusing to attest an empty case plan")
    closures = provenance.load_closures(_REPO_ROOT / "collector" / "hash_closures.yaml")
    table = {
        "collector_ref": _git_collector_ref(_REPO_ROOT),
        "collector_hash": provenance.collector_hash(MODULE_NAME, _REPO_ROOT, closures),
        "case_plan_hash": provenance.case_plan_hash(case_ids),
        "collected_at": date.today().isoformat(),
        "rows": pq.read_metadata(parquet_path).num_rows,
        "status": provenance.derive_table_status(
            unresolved_failed_count=failure_count,
            had_module_failure=False,
        ),
    }
    return provenance.write_collection_meta(
        output_dir,
        runtime_meta,
        {Path(PerfFile.MOE_A2A.value).stem: table},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--output-path", default=os.getcwd())
    parser.add_argument(
        "--vllm-source-root",
        default=os.environ.get("VLLM_SOURCE_ROOT", str(_REPO_ROOT / "libaries.tmpfile" / "vllm")),
    )
    parser.add_argument("--image-digest")
    parser.add_argument("--runtime-abi-json")
    parser.add_argument(
        "--allow-mnnvl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow MNNVL in the low-latency DeepEP buffer (publishable default: enabled)",
    )
    parser.add_argument(
        "--disable-nvlink",
        action="store_true",
        help="disable NVLink in the low-latency DeepEP buffer (diagnostic only)",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--world-size", type=int)
    return parser.parse_args(argv)


def _parse_backends(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    unknown = sorted(set(values) - set(BACKENDS))
    if unknown or not values:
        raise VllmMoeA2ADeclarationError(f"invalid --backends {values}; expected a non-empty subset of {BACKENDS}")
    return values


def transport_is_default(*, allow_mnnvl: bool, disable_nvlink: bool) -> bool:
    """Only one transport flag combination may finalize publishable rows."""

    return allow_mnnvl and not disable_nvlink


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env = dict(os.environ)
    if args.plan_only and args.world_size is not None:
        env["WORLD_SIZE"] = str(args.world_size)
    identity = derive_dist_identity(env, gpus_per_node=args.gpus_per_node)
    backends = _parse_backends(args.backends)
    cases = []
    for backend in backends:
        supported_hidden_sizes = DEEPEP_LL_SUPPORTED_HIDDEN_SIZES if backend == "deepep_ll" else None
        cases.extend(
            build_case_plan(
                shapes=get_vllm_moe_a2a_shapes(
                    required_expert_parallel_size=identity.ep_size,
                    supported_hidden_sizes=supported_hidden_sizes,
                ),
                grid=get_moe_a2a_workload_grid(),
                world_size=identity.world_size,
                node_num=identity.node_num,
                backends=(backend,),
            )
        )
    cases.sort(key=VllmMoeA2ACase.sort_key)
    if args.canary:
        cases = select_canary_cases(cases)
    ids = case_plan_ids(cases, world_size=identity.world_size, node_num=identity.node_num)
    if args.plan_only:
        print(json.dumps({"cases": len(cases), "case_plan_hash": provenance.case_plan_hash(ids)}, indent=2))
        return

    if len(backends) != 1:
        raise VllmMoeA2ADeclarationError(
            "measured runs require exactly one backend because HT/LL and V2 use distinct DeepEP ABIs"
        )

    import torch
    import torch.distributed as dist

    identity = derive_dist_identity(
        dict(os.environ),
        gpus_per_node=args.gpus_per_node,
        visible_device_count=torch.cuda.device_count(),
    )
    if not args.image_digest or not args.runtime_abi_json:
        raise VllmMoeA2ADeclarationError("--image-digest and --runtime-abi-json are required for a measured run")
    try:
        observed_abi = json.loads(args.runtime_abi_json)
    except json.JSONDecodeError as error:
        raise VllmMoeA2ADeclarationError(f"invalid --runtime-abi-json: {error}") from error
    if not isinstance(observed_abi, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in observed_abi.items()
    ):
        raise VllmMoeA2ADeclarationError("--runtime-abi-json must be a JSON object of string fields")
    if backends[0] == "deepep_v2":
        _activate_v2_rdma_rate_tool(observed_abi.get("ibstat_mlx5_0_rate_gbps", ""))
    runtime_meta = attest_vllm_runtime(
        source_root=args.vllm_source_root,
        backend=backends[0],
        observed_abi=observed_abi,
        observed_image_digest=args.image_digest,
    )
    runtime_meta["transport"] = {
        "allow_mnnvl": args.allow_mnnvl,
        "allow_nvlink": not args.disable_nvlink,
        "failure_agreement": "gloo_cpu",
    }
    group, agreement_group = _init_process_groups(identity)
    output_dir = Path(args.output_path)
    print(
        f"[vllm moe_a2a] host={socket.gethostname()} rank={identity.rank}/{identity.world_size} "
        f"source={runtime_meta['source_commit']}",
        flush=True,
    )
    run_failed = False
    try:

        def agree(stage: str, failed: bool) -> bool:
            # Keep lifecycle agreement independent of the CUDA context and
            # benchmark NCCL communicator. A failed DeepEP kernel can poison
            # both; the CPU/Gloo group must still propagate the original stage
            # failure to every rank.
            failure = torch.tensor([int(failed)], device="cpu", dtype=torch.int64)
            dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=agreement_group)
            return bool(failure.item())

        preflight_error: BaseException | None = None
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            stale = stale_output_artifacts(output_dir, PerfFile.MOE_A2A.value)
            if stale:
                raise VllmMoeA2ABenchmarkError(
                    f"vLLM moe_a2a refuses stale output artifacts in {output_dir}: {', '.join(stale)}"
                )
        except BaseException as error:
            preflight_error = error
        raise_for_stage(
            agree_stage(
                "preflight",
                preflight_error,
                agreement=agree,
                peer_error_type=VllmMoeA2APeerError,
            )
        )

        adapter = VllmBenchmarkAdapter(
            group,
            identity,
            allow_mnnvl=args.allow_mnnvl,
            disable_nvlink=args.disable_nvlink,
        )
        result = collect_with_adapter(
            cases,
            adapter=adapter,
            world_size=identity.world_size,
            node_num=identity.node_num,
            stage_agreement=agree,
        )

        failure_record_error: BaseException | None = None
        try:
            _write_failures(output_dir, result.failures, identity)
        except BaseException as error:
            failure_record_error = error
        raise_for_stage(
            agree_stage(
                "failure_records",
                failure_record_error,
                agreement=agree,
                peer_error_type=VllmMoeA2APeerError,
            )
        )
        if identity.rank == 0:
            for failure in result.failures:
                print(
                    f"[vllm moe_a2a] case failure {failure.case.comm_backend} "
                    f"tokens={failure.case.num_tokens}: {failure.error_type}: {failure.error}",
                    file=sys.stderr,
                    flush=True,
                )

        failure_count = len(result.failures)
        if failure_count == len(cases):
            raise VllmMoeA2ABenchmarkError(
                f"all {len(cases)} cases failed; no parquet or complete sidecar will be written"
            )
        if adapter.runtime_capability is None:
            raise VllmMoeA2ABenchmarkError("DeepEP did not report a runtime topology capability")
        runtime_meta["backend_capability"] = adapter.runtime_capability

        perf_path = output_dir / PerfFile.MOE_A2A.value
        row_write_error: BaseException | None = None
        if identity.rank == 0:
            try:
                _write_rows(
                    result.rows,
                    perf_path=perf_path,
                    runtime_meta=runtime_meta,
                    device_name=torch.cuda.get_device_name(identity.local_rank),
                )
            except BaseException as error:
                row_write_error = error
        raise_for_stage(
            agree_stage(
                "row_write",
                row_write_error,
                agreement=agree,
                peer_error_type=VllmMoeA2APeerError,
            )
        )

        publishable = "deepep_ll" not in backends or transport_is_default(
            allow_mnnvl=args.allow_mnnvl,
            disable_nvlink=args.disable_nvlink,
        )
        parquet_path: Path | None = None
        finalize_error: BaseException | None = None
        if publishable and identity.rank == 0:
            try:
                converted = finalize_perf_files([perf_path], merge_existing=False)
                if len(converted) != 1:
                    raise VllmMoeA2ABenchmarkError("finalization did not produce exactly one parquet")
                parquet_path = Path(converted[0])
            except BaseException as error:
                finalize_error = error
        raise_for_stage(
            agree_stage(
                "parquet_finalize",
                finalize_error,
                agreement=agree,
                peer_error_type=VllmMoeA2APeerError,
            )
        )

        sidecar: Path | None = None
        sidecar_error: BaseException | None = None
        if publishable and identity.rank == 0:
            try:
                assert parquet_path is not None
                sidecar = _write_sidecar(
                    output_dir,
                    runtime_meta=runtime_meta,
                    case_ids=ids,
                    parquet_path=parquet_path,
                    failure_count=failure_count,
                )
            except BaseException as error:
                sidecar_error = error
        raise_for_stage(
            agree_stage(
                "sidecar_write",
                sidecar_error,
                agreement=agree,
                peer_error_type=VllmMoeA2APeerError,
            )
        )

        raise_for_stage(
            agree_stage(
                "final_ready",
                None,
                agreement=agree,
                peer_error_type=VllmMoeA2APeerError,
            )
        )
        dist.barrier(group=agreement_group)

        if identity.rank == 0:
            if not publishable:
                print(
                    f"[vllm moe_a2a] diagnostic transport staged rows at {perf_path}; "
                    "parquet and collection_meta.yaml will not be finalized",
                    flush=True,
                )
            else:
                print(
                    f"[vllm moe_a2a] wrote {parquet_path} and {sidecar}; {failure_count} classified failures",
                    flush=True,
                )
    except BaseException as error:
        run_failed = True
        try:
            recorded = _write_rank_error(output_dir, error, identity, stage="fatal_runtime")
            print(f"[vllm moe_a2a] recorded fatal failure in {recorded}: {error}", file=sys.stderr, flush=True)
        except Exception as record_error:
            print(
                f"[vllm moe_a2a] failed to record fatal failure {error!r}: {record_error}",
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as error:
                if not run_failed:
                    raise
                print(f"[vllm moe_a2a] process-group destroy after failure also failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
