# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone TensorRT-LLM serving-parity DeepEP ``moe_a2a`` collector.

The implementation is pinned to TensorRT-LLM source commit
``14efb6ac673c0cbe828e1206cc5c7d5748d05ffa`` and mirrors
``tests/microbenchmarks/bench_moe_comm.py`` at that commit.  In particular it
constructs MPI ``Mapping``/``ModelConfig`` objects and calls
``CommunicationFactory._create_forced_method`` directly.  Calling
``create_strategy`` here would be incorrect: its serving selector tries the
MNNVL NVLink methods before DeepEP, so an NVLink-capable machine could produce
successfully measured but mislabeled rows.

The two persisted identities come from ``wideep.backend_contracts``:
``trtllm_deepep_ht`` forces ``DEEPEP`` and ``trtllm_deepep_ll`` forces
``DEEPEPLOWLATENCY``.  Both execute only dispatch and combine (prepare is setup,
never a row), record ``kernel_source=deepep`` and ``sms=0``, and share the
unified row builder/finalizer with the other ``moe_a2a`` producers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from collector.framework_manifest import get_collector_runtime
from collector.helper import finalize_perf_files, log_perf
from collector.registry_types import PerfFile
from collector.wideep.backend_contracts import contract_for
from collector.wideep.sglang.collect_moe_a2a import (
    MoeA2AShape,
    _build_moe_a2a_row,
    write_moe_a2a_sidecar,
)

__compat__ = "trtllm==1.3.0rc11"

MODULE_NAME = "collector.wideep.trtllm.collect_moe_a2a"
OP_NAME = "moe_a2a"
FRAMEWORK = "TRTLLM"
MANIFEST_FRAMEWORK = "trtllm"
MANIFEST_WORKLOAD = "wideep"
TARGET_SOURCE_COMMIT = "14efb6ac673c0cbe828e1206cc5c7d5748d05ffa"
KERNEL_SOURCE = "deepep"
SMS = 0

COMM_BACKEND_HT = "trtllm_deepep_ht"
COMM_BACKEND_LL = "trtllm_deepep_ll"
FORCED_METHODS = {
    COMM_BACKEND_HT: "DEEPEP",
    COMM_BACKEND_LL: "DEEPEPLOWLATENCY",
}
INFERENCE_PHASES = {
    COMM_BACKEND_HT: "context",
    COMM_BACKEND_LL: "generation",
}
# TensorRT-LLM 14efb6a:
# _torch/modules/fused_moe/communication/deep_ep_low_latency.py
# DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES. This constrains only which declared
# case represents LL in a canary; the full plan remains unchanged.
LL_CANARY_HIDDEN_SIZES = frozenset({2048, 2560, 3584, 4096, 5120, 6144, 7168})
# DeepEP 5be51b2 csrc/kernels/internode_ll.cu:341-349.
LL_CANARY_MAX_TOPK = 9
PHASES = ("combine", "dispatch")
ERRORS_FILENAME_TEMPLATE = "errors_moe_a2a_trtllm.rank{rank}.json"


class MoeA2ADeclarationError(RuntimeError):
    """A declared case, runtime identity, or persisted identity is invalid."""


class MoeA2ABenchmarkError(RuntimeError):
    """A queued benchmark case failed; it must become classified failure data."""


class MoeA2AWriteError(RuntimeError):
    """A row/finalization write failed; collection must not look successful."""


class MoeA2APeerError(RuntimeError):
    """Another distributed rank failed the same queued case."""


@dataclass(frozen=True)
class DistIdentity:
    rank: int
    world_size: int
    local_rank: int
    gpus_per_node: int
    node_num: int

    @property
    def ep_size(self) -> int:
        return self.world_size


@dataclass(frozen=True)
class QuantSpec:
    """TensorRT-LLM quant recipe and truthful persisted communication identity."""

    comm_dtype: str
    quant_algo: str | None


# Source proof:
# * DeepEP.supports_post_quant_dispatch: NVFP4 only.
# * DeepEPLowLatency.supports_post_quant_dispatch: FP8 QDQ, NVFP4, W4AFP8.
# BF16 is the pre-quant baseline.  ``fp4`` is not listed as an invocation:
# TensorRT-LLM uses that identity only for a low-precision combine result.  It
# must not be relabeled as a dispatch mode.
QUANT_SPECS = {
    COMM_BACKEND_HT: (
        QuantSpec("bfloat16", None),
        QuantSpec("nvfp4", "NVFP4"),
    ),
    COMM_BACKEND_LL: (
        QuantSpec("bfloat16", None),
        QuantSpec("fp8", "FP8"),
        QuantSpec("nvfp4", "NVFP4"),
        QuantSpec("w4afp8", "W4A8_AWQ"),
    ),
}


@dataclass(frozen=True)
class MoeA2ACase:
    comm_backend: str
    inference_phase: str
    quant: QuantSpec
    shape: MoeA2AShape
    num_tokens: int
    ep_size: int
    node_num: int
    sms: int = SMS

    def physical_key(self, phase: str, comm_dtype: str | None = None) -> tuple[Any, ...]:
        """Exact current consumer key for one emitted phase row."""
        return (
            self.comm_backend,
            phase,
            comm_dtype or self.quant.comm_dtype,
            self.ep_size,
            self.node_num,
            self.shape.hidden_size,
            self.shape.topk,
            self.shape.num_experts,
            self.sms,
            self.num_tokens,
        )

    def invocation_key(self) -> tuple[Any, ...]:
        """Everything that changes the selected communication invocation."""
        return (
            self.comm_backend,
            FORCED_METHODS[self.comm_backend],
            self.inference_phase,
            self.quant.comm_dtype,
            self.quant.quant_algo,
            self.ep_size,
            self.node_num,
            self.shape.hidden_size,
            self.shape.topk,
            self.shape.num_experts,
            self.num_tokens,
        )

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.comm_backend,
            self.quant.comm_dtype,
            self.shape.hidden_size,
            self.shape.topk,
            self.shape.num_experts,
            self.num_tokens,
        )


@dataclass(frozen=True)
class PhaseMeasurement:
    phase: str
    latency_us: float
    comm_dtype: str


@dataclass(frozen=True)
class BenchmarkResult:
    measurements: tuple[PhaseMeasurement, ...]


class BenchmarkAdapter(Protocol):
    """Injectable boundary around CUDA/MPI/TensorRT-LLM benchmark execution."""

    def run(self, case: MoeA2ACase) -> BenchmarkResult: ...


def derive_dist_identity(env: dict[str, str], *, gpus_per_node: int) -> DistIdentity:
    if gpus_per_node <= 0:
        raise MoeA2ADeclarationError(f"--gpus-per-node must be positive, got {gpus_per_node}")
    rank = int(env.get("OMPI_COMM_WORLD_RANK", env.get("RANK", env.get("SLURM_PROCID", "0"))))
    world_size = int(env.get("OMPI_COMM_WORLD_SIZE", env.get("WORLD_SIZE", env.get("SLURM_NTASKS", "1"))))
    local_rank = int(
        env.get(
            "OMPI_COMM_WORLD_LOCAL_RANK",
            env.get("LOCAL_RANK", env.get("SLURM_LOCALID", str(rank % gpus_per_node))),
        )
    )
    if world_size < 2:
        raise MoeA2ADeclarationError("DeepEP moe_a2a requires an MPI world of at least two ranks")
    if world_size % gpus_per_node:
        raise MoeA2ADeclarationError(
            f"WORLD_SIZE={world_size} is not an integral number of nodes at "
            f"gpus_per_node={gpus_per_node}; node_num is a persisted key"
        )
    return DistIdentity(rank, world_size, local_rank, gpus_per_node, world_size // gpus_per_node)


def get_moe_a2a_shapes() -> list[MoeA2AShape]:
    """Return declared TensorRT-LLM WideEP geometry, physically deduplicated."""
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model

    recipes = get_common_moe_test_cases(backend="trtllm")
    shapes = {
        MoeA2AShape(int(recipe.hidden_size), int(recipe.topk), int(recipe.num_experts))
        for recipe in recipes
        if is_wideep_moe_model(recipe.model_name)
    }
    ordered = sorted(shapes)
    print(
        f"trtllm moe_a2a: {len(ordered)} physical shapes from {len(recipes)} declared backend='trtllm' recipes",
        flush=True,
    )
    if not ordered:
        raise MoeA2ADeclarationError("declared backend='trtllm' WideEP recipes expanded to zero moe_a2a shapes")
    return ordered


def get_moe_a2a_token_grid() -> dict[str, list[int]]:
    """Read the shared declared context/generation token axes."""
    from collector.case_generator import get_base_common_case_values

    values = get_base_common_case_values("moe_a2a") or {}
    grid: dict[str, list[int]] = {}
    for key in ("ht_token_counts", "ll_token_counts"):
        raw = values.get(key)
        if not isinstance(raw, list) or not raw:
            raise MoeA2ADeclarationError(f"common_case_values.moe_a2a.{key} must be non-empty")
        grid[key] = sorted({int(value) for value in raw})
    return grid


def resolve_modes(raw_modes: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in raw_modes.split(",") if part.strip())
    unknown = set(modes) - FORCED_METHODS.keys()
    if not modes or unknown:
        raise MoeA2ADeclarationError(
            f"unsupported --modes value(s) {sorted(unknown) if unknown else modes}; expected {sorted(FORCED_METHODS)}"
        )
    return modes


def build_case_plan(
    *,
    shapes: list[MoeA2AShape],
    token_grid: dict[str, list[int]],
    ep_size: int,
    node_num: int,
    modes: tuple[str, ...] = (COMM_BACKEND_HT, COMM_BACKEND_LL),
) -> list[MoeA2ACase]:
    """Expand declared shapes/tokens and deduplicate on invocation + physical keys."""
    if not shapes:
        raise MoeA2ADeclarationError("moe_a2a cannot build a plan from zero shapes")
    candidates: list[MoeA2ACase] = []
    dropped_alignment = 0
    for shape in shapes:
        if shape.num_experts % ep_size:
            dropped_alignment += 1
            continue
        for backend in modes:
            contract_for("trtllm", backend)
            token_key = "ht_token_counts" if backend == COMM_BACKEND_HT else "ll_token_counts"
            for quant in QUANT_SPECS[backend]:
                for num_tokens in token_grid[token_key]:
                    candidates.append(
                        MoeA2ACase(
                            comm_backend=backend,
                            inference_phase=INFERENCE_PHASES[backend],
                            quant=quant,
                            shape=shape,
                            num_tokens=int(num_tokens),
                            ep_size=ep_size,
                            node_num=node_num,
                        )
                    )

    by_invocation: dict[tuple[Any, ...], MoeA2ACase] = {}
    physical_owners: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    duplicates = 0
    for case in candidates:
        invocation = case.invocation_key()
        physical_keys = tuple(case.physical_key(phase) for phase in PHASES)
        previous = by_invocation.get(invocation)
        if previous is not None:
            duplicates += 1
            continue
        for physical_key in physical_keys:
            owner = physical_owners.get(physical_key)
            if owner is not None and owner != invocation:
                raise MoeA2ADeclarationError(
                    "distinct TensorRT-LLM invocations collide on a moe_a2a physical key: "
                    f"key={physical_key}, first={owner}, second={invocation}"
                )
            physical_owners[physical_key] = invocation
        by_invocation[invocation] = case

    cases = sorted(by_invocation.values(), key=MoeA2ACase.sort_key)
    print(
        f"trtllm moe_a2a: {len(cases)} cases from {len(shapes)} shapes "
        f"(dropped: {dropped_alignment} with num_experts % ep_size != 0; "
        f"deduplicated: {duplicates} identical invocation/physical keys)",
        flush=True,
    )
    if not cases:
        raise MoeA2ADeclarationError(
            f"trtllm moe_a2a expanded to zero cases for ep_size={ep_size}; "
            f"{dropped_alignment} shapes failed expert alignment"
        )
    return cases


def case_plan_ids(cases: list[MoeA2ACase]) -> list[str]:
    if not cases:
        raise MoeA2ADeclarationError("cannot attest an empty moe_a2a case plan")
    return [
        f"{MODULE_NAME}:run_case:"
        + json.dumps(
            {
                "comm_backend": case.comm_backend,
                "comm_dtype": case.quant.comm_dtype,
                "ep_size": case.ep_size,
                "hidden_size": case.shape.hidden_size,
                "inference_phase": case.inference_phase,
                "node_num": case.node_num,
                "num_experts": case.shape.num_experts,
                "num_tokens": case.num_tokens,
                "quant_algo": case.quant.quant_algo,
                "sms": case.sms,
                "topk": case.shape.topk,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    ]


def select_canary_cases(cases: list[MoeA2ACase]) -> list[MoeA2ACase]:
    """Keep one runnable invocation for every backend/communication dtype."""
    required = {(case.comm_backend, case.quant.comm_dtype) for case in cases}
    selected: dict[tuple[str, str], MoeA2ACase] = {}
    for case in cases:
        if case.comm_backend == COMM_BACKEND_LL and (
            case.shape.hidden_size not in LL_CANARY_HIDDEN_SIZES or case.shape.topk > LL_CANARY_MAX_TOPK
        ):
            continue
        selected.setdefault((case.comm_backend, case.quant.comm_dtype), case)
    missing = required - selected.keys()
    if missing:
        raise MoeA2ADeclarationError(
            f"canary plan has no source-supported representative for backend/dtype pairs: {sorted(missing)}"
        )
    return sorted(selected.values(), key=MoeA2ACase.sort_key)


def create_forced_communication(
    factory: Any,
    *,
    case: MoeA2ACase,
    model_config: Any,
    experts_per_rank: int,
) -> Any:
    """Call the pinned factory contract without entering its NVLink priority chain."""
    method = FORCED_METHODS[case.comm_backend]
    backend = factory._create_forced_method(
        method,
        model_config,
        case.shape.num_experts,
        case.shape.num_experts,
        case.shape.topk,
        experts_per_rank,
        payload_in_workspace=False,
        alltoall_result_do_sum=True,
        use_flashinfer=False,
    )
    if backend is None:
        raise MoeA2ABenchmarkError(
            f"CommunicationFactory._create_forced_method({method!r}) returned None; "
            "refusing to fall back or relabel another communication method"
        )
    return backend


@dataclass(frozen=True)
class _DummyPretrainedConfig:
    hidden_size: int
    torch_dtype: Any


class TensorRTLLMBenchmarkAdapter:
    """CUDA/MPI adapter mirroring the pinned ``bench_moe_comm`` call contract."""

    def __init__(
        self,
        *,
        warmup: int = 20,
        iterations: int = 200,
        max_num_tokens_per_rank: int | None = None,
    ):
        self.warmup = warmup
        self.iterations = iterations
        self.max_num_tokens_per_rank = max_num_tokens_per_rank
        self._active_resource_key: tuple[Any, ...] | None = None
        self._active_backend: Any | None = None
        self._active_moe: Any | None = None

    def _resource_key(self, case: MoeA2ACase) -> tuple[Any, ...]:
        """Identity of one reusable rc11 communication/quantization setup.

        The pinned framework benchmark constructs one backend before its local
        token-count loop and reuses it for every token count
        (tests/microbenchmarks/bench_moe_comm.py:1116-1167
        @14efb6ac673c0cbe828e1206cc5c7d5748d05ffa). Recreating DeepEP for every
        token leaks native NVSHMEM heap resources and eventually fails
        ``cuMemCreate``. Keep every constructor input in this key; only the
        runtime token axis is intentionally excluded.
        """
        return (
            case.comm_backend,
            case.quant.comm_dtype,
            case.quant.quant_algo,
            case.ep_size,
            case.node_num,
            case.shape.hidden_size,
            case.shape.topk,
            case.shape.num_experts,
            self.max_num_tokens_per_rank,
        )

    def _destroy_active_resources(self) -> None:
        backend = self._active_backend
        self._active_resource_key = None
        self._active_backend = None
        self._active_moe = None
        if backend is not None:
            destroy = getattr(backend, "destroy", None)
            if destroy is not None:
                destroy()

    def run(self, case: MoeA2ACase) -> BenchmarkResult:
        import torch
        from tensorrt_llm._torch.model_config import ModelConfig
        from tensorrt_llm._torch.modules.fused_moe import CutlassFusedMoE
        from tensorrt_llm._torch.modules.fused_moe.communication import CommunicationFactory
        from tensorrt_llm._torch.modules.fused_moe.routing import DefaultMoeRoutingMethod
        from tensorrt_llm._utils import mpi_allgather, mpi_barrier, mpi_rank
        from tensorrt_llm.mapping import Mapping
        from tensorrt_llm.models.modeling_utils import QuantConfig
        from tensorrt_llm.quantization.mode import QuantAlgo

        if case.shape.num_experts % case.ep_size:
            raise MoeA2ABenchmarkError("num_experts must be divisible by ep_size")
        experts_per_rank = case.shape.num_experts // case.ep_size
        mapping = Mapping(
            rank=mpi_rank(),
            tp_size=case.ep_size,
            moe_ep_size=case.ep_size,
            enable_attention_dp=True,
            world_size=case.ep_size,
        )
        quant_algo = getattr(QuantAlgo, case.quant.quant_algo) if case.quant.quant_algo else QuantAlgo.NO_QUANT
        quant_config = QuantConfig(quant_algo=None if quant_algo == QuantAlgo.NO_QUANT else quant_algo)
        max_num_tokens = self.max_num_tokens_per_rank or case.num_tokens
        if max_num_tokens < case.num_tokens:
            raise MoeA2ADeclarationError(
                f"max_num_tokens_per_rank={max_num_tokens} is below case tokens={case.num_tokens}"
            )
        resource_key = self._resource_key(case)
        if resource_key != self._active_resource_key:
            self._destroy_active_resources()
            model_config = ModelConfig(
                pretrained_config=_DummyPretrainedConfig(
                    hidden_size=case.shape.hidden_size, torch_dtype=torch.bfloat16
                ),
                mapping=mapping,
                quant_config=quant_config,
                max_num_tokens=max_num_tokens,
                moe_max_num_tokens=max_num_tokens,
                use_cuda_graph=False,
                use_low_precision_moe_combine=False,
            )
            backend = create_forced_communication(
                CommunicationFactory,
                case=case,
                model_config=model_config,
                experts_per_rank=experts_per_rank,
            )
            try:
                moe = None
                if quant_algo != QuantAlgo.NO_QUANT and backend.supports_post_quant_dispatch():
                    moe = CutlassFusedMoE(
                        routing_method=DefaultMoeRoutingMethod(top_k=case.shape.topk),
                        num_experts=case.shape.num_experts,
                        hidden_size=case.shape.hidden_size,
                        intermediate_size=case.shape.hidden_size * 4,
                        dtype=torch.bfloat16,
                        reduce_results=False,
                        model_config=model_config,
                        init_load_balancer=False,
                        without_comm=True,
                    ).to(torch.device("cuda"))
                elif quant_algo != QuantAlgo.NO_QUANT:
                    raise MoeA2ABenchmarkError(
                        f"forced {FORCED_METHODS[case.comm_backend]} does not support truthful "
                        f"post-quant identity {case.quant.comm_dtype!r}"
                    )
            except Exception:
                destroy = getattr(backend, "destroy", None)
                if destroy is not None:
                    destroy()
                raise
            self._active_resource_key = resource_key
            self._active_backend = backend
            self._active_moe = moe
        backend = self._active_backend
        moe = self._active_moe
        assert backend is not None
        all_rank_num_tokens = list(mpi_allgather(case.num_tokens))
        if not backend.is_workload_feasible(all_rank_num_tokens, num_chunks=1):
            raise MoeA2ABenchmarkError(
                f"forced {FORCED_METHODS[case.comm_backend]} reports workload infeasible "
                f"for tokens={all_rank_num_tokens}; refusing silent fallback"
            )

        device = torch.device("cuda")
        hidden_states = torch.randn(case.num_tokens, case.shape.hidden_size, dtype=torch.bfloat16, device=device)
        hidden_states_sf = None
        token_selected_slots = torch.randint(
            0,
            case.shape.num_experts,
            (case.num_tokens, case.shape.topk),
            dtype=torch.int32,
            device=device,
        )
        token_final_scales = torch.rand(case.num_tokens, case.shape.topk, dtype=torch.float32, device=device)

        # Serving order from ConfigurableMoE, mirrored by bench_moe_comm.py:
        # Quantize -> Dispatch. Quantization is outside the timed comm region.
        if moe is not None:
            hidden_states, hidden_states_sf = moe.quantize_input(hidden_states, post_quant_comm=True)

        # W4AFP8 dispatch consumes the per-channel activation scale that the
        # quantization method attaches to the module
        # (deep_ep_low_latency.py:271 @14efb6ac673c0cbe828e1206cc5c7d5748d05ffa).
        # Forward the framework's own tensor exactly as ConfigurableMoE does
        # (configurable_moe.py:774-776 @14efb6ac673c0cbe828e1206cc5c7d5748d05ffa)
        # rather than reconstructing a scale here; backends that do not read it
        # absorb it through **kwargs, matching serving.
        dispatch_kwargs: dict[str, Any] = {}
        quant_scales = getattr(moe, "quant_scales", None) if moe is not None else None
        if quant_scales is not None and hasattr(quant_scales, "pre_quant_scale_1"):
            dispatch_kwargs["pre_quant_scale"] = quant_scales.pre_quant_scale_1

        def iteration() -> tuple[float, float]:
            backend.prepare_dispatch(token_selected_slots, all_rank_num_tokens)
            dispatch_start = torch.cuda.Event(enable_timing=True)
            dispatch_end = torch.cuda.Event(enable_timing=True)
            combine_start = torch.cuda.Event(enable_timing=True)
            combine_end = torch.cuda.Event(enable_timing=True)
            dispatch_start.record()
            recv_hidden, _, _, _ = backend.dispatch(
                hidden_states,
                hidden_states_sf,
                token_selected_slots,
                token_final_scales,
                all_rank_num_tokens,
                **dispatch_kwargs,
            )
            dispatch_end.record()
            output_shape = list(recv_hidden.shape)
            output_shape[-1] = case.shape.hidden_size
            moe_output = torch.empty(tuple(output_shape), dtype=torch.bfloat16, device=device)
            combine_start.record()
            backend.combine(moe_output, all_rank_max_num_tokens=max(all_rank_num_tokens))
            combine_end.record()
            torch.cuda.synchronize()
            return (
                dispatch_start.elapsed_time(dispatch_end) * 1000.0,
                combine_start.elapsed_time(combine_end) * 1000.0,
            )

        mpi_barrier()
        for _ in range(self.warmup):
            iteration()
        samples = [iteration() for _ in range(self.iterations)]
        mpi_barrier()
        if not samples:
            raise MoeA2ABenchmarkError("benchmark produced zero timing samples")
        dispatch_us = sum(sample[0] for sample in samples) / len(samples)
        combine_us = sum(sample[1] for sample in samples) / len(samples)
        return BenchmarkResult(
            (
                PhaseMeasurement("combine", combine_us, case.quant.comm_dtype),
                PhaseMeasurement("dispatch", dispatch_us, case.quant.comm_dtype),
            )
        )


def build_unified_rows(case: MoeA2ACase, result: BenchmarkResult) -> list[dict[str, Any]]:
    """Validate and build the case's dispatch/combine rows; never emit prepare."""
    phases = [measurement.phase for measurement in result.measurements]
    if sorted(phases) != list(PHASES) or len(set(phases)) != len(PHASES):
        raise MoeA2ABenchmarkError(f"benchmark must return exactly combine+dispatch (no prepare), got {phases}")
    contract = contract_for("trtllm", case.comm_backend)
    rows = []
    for measurement in sorted(result.measurements, key=lambda item: item.phase):
        if measurement.comm_dtype not in contract.comm_dtypes:
            raise MoeA2ABenchmarkError(
                f"{measurement.comm_dtype!r} is not a truthful dtype for {case.comm_backend}; "
                "refusing to relabel an unsupported mode"
            )
        if measurement.latency_us <= 0:
            raise MoeA2ABenchmarkError(f"{measurement.phase} returned non-positive latency {measurement.latency_us}")
        rows.append(
            _build_moe_a2a_row(
                comm_backend=case.comm_backend,
                phase=measurement.phase,
                comm_dtype=measurement.comm_dtype,
                ep_size=case.ep_size,
                node_num=case.node_num,
                shape=case.shape,
                num_tokens=case.num_tokens,
                sms=SMS,
                transmit_us=measurement.latency_us,
                notify_us=0.0,
            )
        )
    return rows


def resolve_runtime_meta(
    installed_version: str,
    *,
    source_commit: str,
    observed_abi: dict[str, str],
    observed_image_digest: str,
) -> dict[str, Any]:
    """Fail closed unless version, source and DeepEP/NVSHMEM ABI match the pin."""
    from packaging.version import InvalidVersion, Version

    runtime = get_collector_runtime(MANIFEST_FRAMEWORK, workload=MANIFEST_WORKLOAD)
    try:
        version_matches = Version(installed_version).public == Version(runtime.version).public
    except InvalidVersion as error:
        raise MoeA2ADeclarationError(f"invalid installed tensorrt_llm version {installed_version!r}") from error
    if not version_matches:
        raise MoeA2ADeclarationError(
            f"wideep_trtllm requires package version {runtime.version}, found {installed_version}"
        )
    if runtime.source_commit != TARGET_SOURCE_COMMIT or source_commit != runtime.source_commit:
        raise MoeA2ADeclarationError(
            f"wideep_trtllm requires source commit {runtime.source_commit}, found {source_commit}"
        )
    expected_abi = runtime.abi or {}
    if observed_abi != expected_abi:
        raise MoeA2ADeclarationError(f"wideep_trtllm ABI mismatch: expected {expected_abi}, found {observed_abi}")
    image, separator, digest = runtime.image().partition("@")
    if not separator or observed_image_digest != digest:
        raise MoeA2ADeclarationError(
            f"wideep_trtllm image digest mismatch: expected {digest!r}, found {observed_image_digest!r}"
        )
    meta: dict[str, Any] = {
        "framework": runtime.framework,
        "version": installed_version,
        "image": image,
        "source_commit": source_commit,
        "abi": observed_abi,
    }
    if separator:
        meta["image_digest"] = digest
    return meta


def record_failure(
    output_dir: Path,
    case: MoeA2ACase,
    error: BaseException,
    *,
    rank: int,
) -> None:
    path = output_dir / ERRORS_FILENAME_TEMPLATE.format(rank=rank)
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(
        {
            "module": MODULE_NAME,
            "op": OP_NAME,
            "classification": "unexpected",
            "error_type": type(error).__name__,
            "error": str(error),
            "rank": rank,
            "case": {
                "comm_backend": case.comm_backend,
                "comm_dtype": case.quant.comm_dtype,
                "inference_phase": case.inference_phase,
                "ep_size": case.ep_size,
                "node_num": case.node_num,
                "hidden_size": case.shape.hidden_size,
                "topk": case.shape.topk,
                "num_experts": case.shape.num_experts,
                "num_tokens": case.num_tokens,
                "sms": case.sms,
            },
        }
    )
    path.write_text(json.dumps(records, indent=2))


def run_collection(
    *,
    cases: list[MoeA2ACase],
    adapter: BenchmarkAdapter,
    output_dir: Path,
    rank: int,
    version: str,
    device_name: str,
    runtime_meta: dict[str, Any],
    finalize: Callable[[list[str]], list[Path]] = finalize_perf_files,
    failure_agreement: Callable[[bool], bool] = bool,
) -> Path | None:
    """Execute, write, finalize and attest; zero/partial/write states fail closed."""
    if not cases:
        raise MoeA2ADeclarationError("refusing to run or attest a zero-case plan")
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_path = str(output_dir / PerfFile.MOE_A2A.value)
    if rank == 0 and Path(perf_path).exists():
        raise MoeA2AWriteError(f"stale staging file exists at {perf_path}; resume/merge must be explicit")
    error_path = output_dir / f"errors_moe_a2a_trtllm.rank{rank}.json"
    if error_path.exists():
        raise MoeA2AWriteError(f"stale failure staging exists at {error_path}; resume/merge must be explicit")
    failures = 0
    rows_written = 0
    for case in cases:
        result = None
        local_error: BaseException | None = None
        try:
            result = adapter.run(case)
        except Exception as error:
            local_error = error

        # Every rank reaches this agreement point before any row is persisted.
        # A failure observed only by a non-writer rank therefore cannot leave
        # rank 0 with a falsely complete table.
        if failure_agreement(local_error is not None):
            failures += 1
            record_failure(
                output_dir,
                case,
                local_error or MoeA2APeerError("another rank failed this queued case"),
                rank=rank,
            )
            continue

        assert result is not None
        try:
            rows = build_unified_rows(case, result)
            if rank == 0:
                for row in rows:
                    wrote = log_perf(
                        item_list=[row],
                        framework=FRAMEWORK,
                        version=version,
                        device_name=device_name,
                        op_name=OP_NAME,
                        kernel_source=KERNEL_SOURCE,
                        perf_filename=perf_path,
                    )
                    if not wrote:
                        raise MoeA2AWriteError(
                            f"log_perf rejected {case.physical_key(row['phase'], row['comm_dtype'])}"
                        )
                    rows_written += 1
        except Exception as error:
            failures += 1
            record_failure(output_dir, case, error, rank=rank)

    if rank != 0:
        if failures:
            raise MoeA2ABenchmarkError(f"rank {rank} observed {failures}/{len(cases)} failed cases")
        return None
    if rows_written == 0:
        raise MoeA2ABenchmarkError(f"TensorRT-LLM DeepEP produced zero rows; {failures}/{len(cases)} cases failed")
    converted = finalize([perf_path])
    if len(converted) != 1:
        raise MoeA2AWriteError(f"expected one finalized moe_a2a parquet, got {len(converted)}")
    parquet_path = Path(converted[0])
    write_moe_a2a_sidecar(
        output_dir,
        runtime_meta=runtime_meta,
        case_ids=case_plan_ids(cases),
        parquet_path=parquet_path,
        failure_count=failures,
        module_name=MODULE_NAME,
    )
    if failures:
        raise MoeA2ABenchmarkError(
            f"refusing clean completion for partial TensorRT-LLM DeepEP collection: "
            f"{failures}/{len(cases)} cases failed"
        )
    return parquet_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TensorRT-LLM serving-parity DeepEP moe_a2a rows under external MPI"
    )
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--modes", default=f"{COMM_BACKEND_HT},{COMM_BACKEND_LL}")
    parser.add_argument("--output-path", default=os.getcwd())
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--deep-ep-commit", required=True)
    parser.add_argument("--nvshmem-version", required=True)
    parser.add_argument("--nvshmem-archive-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--canary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import tensorrt_llm
    import torch
    from tensorrt_llm._utils import mpi_allgather

    identity = derive_dist_identity(dict(os.environ), gpus_per_node=args.gpus_per_node)
    torch.cuda.set_device(identity.local_rank)
    observed_abi = {
        "build_mode": "source-wheel-over-pinned-base",
        "deep_ep": args.deep_ep_commit,
        "nvshmem": args.nvshmem_version,
        "nvshmem_archive_sha256": args.nvshmem_archive_sha256,
    }
    runtime_meta = resolve_runtime_meta(
        tensorrt_llm.__version__,
        source_commit=args.source_commit,
        observed_abi=observed_abi,
        observed_image_digest=args.image_digest,
    )
    cases = build_case_plan(
        shapes=get_moe_a2a_shapes(),
        token_grid=get_moe_a2a_token_grid(),
        ep_size=identity.ep_size,
        node_num=identity.node_num,
        modes=resolve_modes(args.modes),
    )
    if args.canary:
        cases = select_canary_cases(cases)
    run_collection(
        cases=cases,
        adapter=TensorRTLLMBenchmarkAdapter(
            warmup=args.warmup,
            iterations=args.iterations,
            max_num_tokens_per_rank=max(case.num_tokens for case in cases),
        ),
        output_dir=Path(args.output_path),
        rank=identity.rank,
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(identity.local_rank),
        runtime_meta=runtime_meta,
        failure_agreement=lambda failed: any(mpi_allgather(bool(failed))),
    )


if __name__ == "__main__":
    main()
