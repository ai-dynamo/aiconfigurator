# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone, single-node vLLM DeepEP serving-parity collector.

Launch with ``torchrun --standalone --nproc-per-node {2,4,8} ...``.  The
collector calls vLLM's prepare/finalize implementations directly and allocates
no model weights.  vLLM imports are intentionally confined to
``VllmBenchmarkAdapter`` so population and persistence can be tested on CPU.

The adapter mirrors vLLM commit ``d8c70f2``:

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
import json
import os
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
from collector.helper import finalize_perf_files, log_perf
from collector.registry_types import PerfFile
from collector.wideep.backend_contracts import contract_for
from collector.wideep.sglang.collect_moe_a2a import (
    DistIdentity,
    MoeA2ADeclarationError,
    MoeA2AShape,
    PhaseTiming,
    _build_moe_a2a_row,
    _git_collector_ref,
    derive_dist_identity,
    get_moe_a2a_workload_grid,
)

MODULE_NAME = "collector.wideep.vllm.collect_moe_a2a"
OP_NAME = "moe_a2a"
FRAMEWORK = "vLLM"
KERNEL_SOURCE = "deepep"
COMM_DTYPE = "default"
PHASES = ("combine", "dispatch")
BACKENDS = ("deepep_ht", "deepep_ll", "deepep_v2")
BACKEND_CONTRACTS = {backend: contract_for("vllm", backend) for backend in BACKENDS}
SUPPORTED_WORLD_SIZES = (2, 4, 8)
HT_SMS = 20
LL_SMS = 0
HT_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024
TARGET_VLLM_SOURCE_COMMIT = "d8c70f22434afcbd6644aa43d3f23aecb6e5a09f"
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


def get_vllm_moe_a2a_shapes() -> list[MoeA2AShape]:
    """Resolve correlated WideEP shapes through the vLLM case population."""
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model

    recipes = get_common_moe_test_cases(backend="vllm")
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
    backends: tuple[str, ...] = BACKENDS,
) -> list[VllmMoeA2ACase]:
    """Build deterministic single-node cases with serving-owned policies.

    v2 uses the union of the two declared token axes.  Tokens on the LL axis
    use generation/cudagraph mode; remaining HT-only tokens use context/eager
    mode.  This gives every persisted v2 key exactly one invocation identity.
    """
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise VllmMoeA2ADeclarationError(
            f"single-node vLLM moe_a2a supports world sizes {SUPPORTED_WORLD_SIZES}, got {world_size}"
        )
    unknown = sorted(set(backends) - set(BACKENDS))
    if unknown:
        raise VllmMoeA2ADeclarationError(f"unsupported DeepEP backend(s): {unknown}")
    if not backends:
        raise VllmMoeA2ADeclarationError("no DeepEP backends selected")

    ht_tokens = _validated_axis(grid, "ht_token_counts")
    ll_tokens = _validated_axis(grid, "ll_token_counts")
    cases: list[VllmMoeA2ACase] = []
    dropped = 0
    for shape in shapes:
        if shape.num_experts % world_size:
            dropped += 1
            continue
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
            f"moe_a2a expanded to zero cases: {len(shapes)} shapes, world_size={world_size}, "
            f"{dropped} not divisible by world size"
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
    static_keys = [case.persisted_key(ep_size=world_size, node_num=1) for case in cases if case.sms is not None]
    if len(static_keys) != len(set(static_keys)):
        raise VllmMoeA2ADeclarationError("duplicate vLLM DeepEP persisted key")
    print(
        f"moe_a2a vllm: {len(cases)} cases from {len(shapes)} shapes "
        f"(dropped {dropped} shapes not divisible by world_size={world_size})",
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


def case_plan_ids(cases: list[VllmMoeA2ACase], *, world_size: int) -> list[str]:
    ids = []
    for case in cases:
        payload = {
            "capacity": case.capacity,
            "comm_backend": case.comm_backend,
            "ep_size": world_size,
            "hidden_size": case.shape.hidden_size,
            "inference_phase": case.inference_phase,
            "node_num": 1,
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
    failure_agreement: Callable[[bool], bool] = bool,
) -> CollectionResult:
    """Pure collection loop used by GPU-free tests and the torchrun entrypoint."""
    rows: list[dict[str, Any]] = []
    failures: list[CaseFailure] = []
    resolved_cases: list[VllmMoeA2ACase] = []
    resolved_keys: set[tuple[Any, ...]] = set()
    try:
        for case in cases:
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
                key = resolved.persisted_key(ep_size=world_size, node_num=1)
                if key in resolved_keys:
                    raise VllmMoeA2ADeclarationError(f"duplicate resolved persisted key: {key}")
                for phase in PHASES:
                    timing = result.timings[phase]
                    case_rows.append(
                        _build_moe_a2a_row(
                            comm_backend=case.comm_backend,
                            phase=phase,
                            ep_size=world_size,
                            node_num=1,
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
            if failure_agreement(local_error is not None):
                error = local_error or VllmMoeA2APeerError("another rank failed this queued case")
                failures.append(CaseFailure(case, type(error).__name__, str(error)))
                continue

            assert resolved is not None
            resolved_keys.add(resolved.persisted_key(ep_size=world_size, node_num=1))
            resolved_cases.append(resolved)
            rows.extend(case_rows)
    finally:
        adapter.close()
    return CollectionResult(rows, failures, resolved_cases)


def _init_nccl_group(identity: DistIdentity):
    """Initialize one direct NCCL process group; no vLLM engine/model state."""
    import torch
    import torch.distributed as dist

    if identity.node_num != 1 or identity.world_size not in SUPPORTED_WORLD_SIZES:
        raise VllmMoeA2ADeclarationError(
            f"vLLM standalone collector requires one node and world size {SUPPORTED_WORLD_SIZES}; "
            f"got nodes={identity.node_num}, world={identity.world_size}"
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
    return dist.new_group(list(range(identity.world_size)), backend="nccl")


class VllmBenchmarkAdapter:
    """vLLM-specific invocation adapter for commit ``d8c70f2``."""

    def __init__(self, group, identity: DistIdentity, *, warmups: int = 3, runs: int = 10):
        self.group = group
        self.identity = identity
        self.warmups = warmups
        self.runs = runs
        self._buffer = None

    def close(self) -> None:
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None

    def benchmark(self, case: VllmMoeA2ACase) -> BenchmarkResult:
        """Run exact public prepare/finalize calls with synthetic activations."""
        self.close()
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
        from vllm.config import VllmConfig
        from vllm.forward_context import set_forward_context

        # vLLM d8c70f2 wraps model execution in set_forward_context
        # (vllm/v1/worker/gpu/model_runner.py:1457-1467). DeepEP v2 decode
        # reads that context to bound its receive allocation
        # (prepare_finalize/deepep_v2.py:121-141).
        return set_forward_context(None, VllmConfig(), num_tokens=case.num_tokens)

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
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
            DeepEPV2PrepareAndFinalize,
        )
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceDelegate,
            TopKWeightAndReduceNoOP,
        )

        local_experts = case.shape.num_experts // self.identity.world_size
        rank_offset = self.identity.rank * local_experts
        quant_config = FusedMoEQuantConfig.make(None)
        if case.comm_backend == "deepep_ht":
            self._buffer = deep_ep.Buffer(
                group=self.group,
                num_nvl_bytes=case.capacity,
                num_rdma_bytes=0,
                low_latency_mode=False,
                num_qps_per_rank=1,
                explicitly_destroy=True,
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
            rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                num_max_dispatch_tokens_per_rank=case.capacity,
                hidden=case.shape.hidden_size,
                num_ranks=self.identity.world_size,
                num_experts=case.shape.num_experts,
            )
            self._buffer = deep_ep.Buffer(
                group=self.group,
                num_nvl_bytes=HT_BUFFER_SIZE_BYTES,
                num_rdma_bytes=rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=local_experts,
                allow_nvlink_for_low_latency_mode=True,
                explicitly_destroy=True,
            )
            prepare_finalize = DeepEPLLPrepareAndFinalize(
                self._buffer,
                max_tokens_per_rank=case.capacity,
                num_dispatchers=self.identity.world_size,
                use_fp8_dispatch=False,
            )
            reduce_impl = TopKWeightAndReduceDelegate()
        elif case.comm_backend == "deepep_v2":
            self._probe_v2_gin()
            self._buffer = deep_ep.ElasticBuffer(
                group=self.group,
                num_max_tokens_per_rank=case.capacity,
                hidden=case.shape.hidden_size,
                num_topk=case.shape.topk,
                use_fp8_dispatch=False,
                allow_hybrid_mode=False,
                prefer_overlap_with_compute=False,
                allow_multiple_reduction=False,
                explicitly_destroy=True,
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

    def _probe_v2_gin(self) -> None:
        """Use vLLM's live NCCL-GIN query; unsupported v2 is a classified raise."""
        import torch
        import torch.distributed as dist
        from vllm.utils.nccl import query_nccl_gin_type

        dist.all_reduce(torch.zeros(1, device="cuda"), group=self.group)
        gin_type = query_nccl_gin_type(self.group)
        if not gin_type:
            raise VllmMoeA2ABenchmarkError(
                "deepep_v2 requires NCCL GIN; the live process group reported no GIN support"
            )


def attest_vllm_runtime(
    *,
    source_root: str | Path,
    observed_abi: dict[str, str],
    observed_image_digest: str,
    installed_version_getter=distribution_version,
    source_commit_getter=None,
) -> dict[str, Any]:
    """Attest the live installed version and exact source checkout."""
    source_root = Path(source_root)
    if source_commit_getter is None:
        source_commit_getter = _git_head
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
    if observed_abi != runtime.abi:
        raise VllmMoeA2ADeclarationError(f"wideep_vllm ABI mismatch: expected {runtime.abi}, found {observed_abi}")
    matching_images = [
        image.partition("@")
        for image in runtime.images.values()
        if image.partition("@")[1] and image.partition("@")[2] == observed_image_digest
    ]
    if not matching_images:
        expected_digests = sorted(image.partition("@")[2] for image in runtime.images.values() if "@" in image)
        raise VllmMoeA2ADeclarationError(
            f"wideep_vllm image digest mismatch: expected one of {expected_digests!r}, found {observed_image_digest!r}"
        )
    image, _, digest = matching_images[0]
    return {
        "framework": runtime.framework,
        "version": installed_version,
        "image": image,
        "image_digest": digest,
        "source_commit": source_commit,
        "abi": observed_abi,
    }


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
                    "node_num": 1,
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    env = dict(os.environ)
    if args.plan_only and args.world_size is not None:
        env["WORLD_SIZE"] = str(args.world_size)
    identity = derive_dist_identity(env, gpus_per_node=args.gpus_per_node)
    if identity.node_num != 1:
        raise VllmMoeA2ADeclarationError("vLLM moe_a2a collector is single-node only")
    cases = build_case_plan(
        shapes=get_vllm_moe_a2a_shapes(),
        grid=get_moe_a2a_workload_grid(),
        world_size=identity.world_size,
        backends=_parse_backends(args.backends),
    )
    if args.canary:
        cases = select_canary_cases(cases)
    ids = case_plan_ids(cases, world_size=identity.world_size)
    if args.plan_only:
        print(json.dumps({"cases": len(cases), "case_plan_hash": provenance.case_plan_hash(ids)}, indent=2))
        return

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
    runtime_meta = attest_vllm_runtime(
        source_root=args.vllm_source_root,
        observed_abi=observed_abi,
        observed_image_digest=args.image_digest,
    )
    group = _init_nccl_group(identity)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[vllm moe_a2a] host={socket.gethostname()} rank={identity.rank}/{identity.world_size} "
        f"source={runtime_meta['source_commit']}",
        flush=True,
    )
    run_failed = False
    try:

        def agree_case_failure(failed: bool) -> bool:
            failure = torch.tensor([int(failed)], device="cuda", dtype=torch.int64)
            dist.all_reduce(failure, op=dist.ReduceOp.MAX, group=group)
            return bool(failure.item())

        result = collect_with_adapter(
            cases,
            adapter=VllmBenchmarkAdapter(group, identity),
            world_size=identity.world_size,
            failure_agreement=agree_case_failure,
        )
        _write_failures(output_dir, result.failures, identity)
        failure_tensor = torch.tensor([len(result.failures)], device="cuda", dtype=torch.int64)
        dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX, group=group)
        failure_count = int(failure_tensor.item())
        if failure_count == len(cases):
            raise VllmMoeA2ABenchmarkError(
                f"all {len(cases)} cases failed; no parquet or complete sidecar will be written"
            )
        if identity.rank == 0:
            perf_path = output_dir / PerfFile.MOE_A2A.value
            _write_rows(
                result.rows,
                perf_path=perf_path,
                runtime_meta=runtime_meta,
                device_name=torch.cuda.get_device_name(identity.local_rank),
            )
            converted = finalize_perf_files([perf_path], merge_existing=False)
            if len(converted) != 1:
                raise VllmMoeA2ABenchmarkError("finalization did not produce exactly one parquet")
            sidecar = _write_sidecar(
                output_dir,
                runtime_meta=runtime_meta,
                case_ids=ids,
                parquet_path=converted[0],
                failure_count=failure_count,
            )
            print(
                f"[vllm moe_a2a] wrote {converted[0]} and {sidecar}; {failure_count} classified failures",
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
            if not run_failed:
                dist.barrier(group=group)
            try:
                dist.destroy_process_group()
            except Exception as error:
                if not run_failed:
                    raise
                print(f"[vllm moe_a2a] process-group destroy after failure also failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
