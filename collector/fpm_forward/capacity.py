# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIC-backed model residency and per-cell FPM execution profiles."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aiconfigurator.sdk.memory import KVCacheEstimator, estimate_kv_cache
from aiconfigurator.sdk.perf_database import load_system_spec

from .capabilities import ModelCapabilityProfile
from .sampling import SamplingDesign
from .types import FPMPoint, ParallelTopology

FPM_MAX_GPU_MEMORY_UTILIZATION = 0.9
FPM_KV_TOLERANCE_FRACTION = 0.05
FPM_MAX_CUDAGRAPH_CAPTURE_TOKENS = 8192
FPM_RUNTIME_RESERVE_FRACTION = 0.1
FPM_RUNTIME_RESERVE_MIN_BYTES = 10 * 1024**3
_FRACTION_DIGITS = 4


@dataclass(frozen=True, slots=True)
class CapacityDecision:
    topology: ParallelTopology
    admitted: bool
    source: str
    reason: str
    rank_local_kv_capacity_tokens: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology.to_dict(),
            "admitted": self.admitted,
            "source": self.source,
            "reason": self.reason,
            "rank_local_kv_capacity_tokens": self.rank_local_kv_capacity_tokens,
        }


@dataclass(frozen=True, slots=True)
class PointCapacityDecision:
    point: FPMPoint
    admitted: bool
    disposition: str
    reason: str
    required_memory_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "point": self.point.to_dict(),
            "admitted": self.admitted,
            "disposition": self.disposition,
            "reason": self.reason,
            "required_memory_fraction": self.required_memory_fraction,
        }


@dataclass(frozen=True, slots=True)
class FPMExecutionProfile:
    """Frozen engine limits and the compatible ordered point population."""

    ordered_points: tuple[FPMPoint, ...]
    selected_point_count: int
    max_batch_size: int
    max_num_tokens: int
    max_seq_len: int
    gpu_memory_utilization: float
    cudagraph_capture_sizes: tuple[int, ...]
    memory_source: str
    memory_fraction_ceiling: float
    kv_tolerance_fraction: float
    total_gpu_capacity_bytes: int
    non_kv_bytes: int | None
    required_kv_bytes: int | None
    decisions: tuple[PointCapacityDecision, ...]
    aic_non_kv_bytes: int | None = None
    runtime_reserve_bytes: int = 0
    runtime_reserve_fraction: float = 0.0
    runtime_reserve_source: str = "max(10_percent_total,10_gib)"

    @property
    def selected(self) -> tuple[FPMPoint, ...]:
        return self.ordered_points[: self.selected_point_count]

    @property
    def compilation_config(self) -> dict[str, object]:
        return {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": list(self.cudagraph_capture_sizes),
            "max_cudagraph_capture_size": max(self.cudagraph_capture_sizes),
            "custom_ops": ["all"],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "ordered_points": [point.to_dict() for point in self.ordered_points],
            "selected_point_count": self.selected_point_count,
            "max_batch_size": self.max_batch_size,
            "max_num_tokens": self.max_num_tokens,
            "max_seq_len": self.max_seq_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "compilation_config": self.compilation_config,
            "memory": {
                "source": self.memory_source,
                "memory_fraction_ceiling": self.memory_fraction_ceiling,
                "kv_tolerance_fraction": self.kv_tolerance_fraction,
                "total_gpu_capacity_bytes": self.total_gpu_capacity_bytes,
                "non_kv_bytes": self.non_kv_bytes,
                "aic_non_kv_bytes": self.aic_non_kv_bytes,
                "runtime_reserve_bytes": self.runtime_reserve_bytes,
                "runtime_reserve_fraction": self.runtime_reserve_fraction,
                "runtime_reserve_source": self.runtime_reserve_source,
                "required_kv_bytes": self.required_kv_bytes,
            },
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class _EnvelopeEstimate:
    max_batch_size: int
    max_num_tokens: int
    max_seq_len: int
    total_gpu_capacity_bytes: int
    non_kv_bytes: int
    tokens_from_kv_bytes: Callable[[float], int]
    kv_bytes_for_sequence: Callable[[int], float] | None


@dataclass(frozen=True, slots=True)
class FPMCapacityModel:
    """Reusable rank-local memory curve for one topology and KV dtype."""

    total_gpu_capacity_bytes: int
    fixed_non_activation_bytes: int | None
    activation_floor_bytes: int | None
    activation_bytes_per_token: float | None
    tokens_from_kv_bytes: Callable[[float], int] | None
    kv_bytes_for_sequence: Callable[[int], float] | None
    source: str
    error: str | None = None

    def envelope(self, *, max_batch_size: int, max_num_tokens: int, max_seq_len: int) -> _EnvelopeEstimate:
        if (
            self.fixed_non_activation_bytes is None
            or self.activation_floor_bytes is None
            or self.activation_bytes_per_token is None
            or self.tokens_from_kv_bytes is None
        ):
            raise ValueError(self.error or "AIC capacity model is unavailable")
        activations = max(
            self.activation_floor_bytes,
            math.ceil(self.activation_bytes_per_token * max_num_tokens),
        )
        return _EnvelopeEstimate(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            total_gpu_capacity_bytes=self.total_gpu_capacity_bytes,
            non_kv_bytes=self.fixed_non_activation_bytes + activations,
            tokens_from_kv_bytes=self.tokens_from_kv_bytes,
            kv_bytes_for_sequence=self.kv_bytes_for_sequence,
        )


def _round_fraction_up(value: float) -> float:
    scale = 10**_FRACTION_DIGITS
    return math.ceil(value * scale) / scale


def _engine_envelope(points: tuple[FPMPoint, ...] | list[FPMPoint]) -> tuple[int, int, int]:
    return (
        max(point.batch_size for point in points),
        max(point.batch_size * point.suffix_length for point in points),
        max(point.prefix_length + point.suffix_length + 1 for point in points),
    )


def _cudagraph_capture_sizes(points: tuple[FPMPoint, ...] | list[FPMPoint]) -> tuple[int, ...]:
    """Cover every <=8K scheduled-query-token shape used by CUDA Graph.

    vLLM dispatches the graph by the number of tokens scheduled in the current
    forward: ``B * S`` for prefill (including prefill with prefix-cache hits)
    and ``B`` for decode because decode fixes ``S=1``.  ``P`` is runtime
    attention/KV state carried by computed-token metadata and block tables; it
    affects the KV-capacity and max-sequence-length envelopes below, but is not
    an additional CUDA Graph capture-size dimension.
    """

    capture_sizes = set()
    for point in points:
        scheduled_tokens = point.batch_size * point.suffix_length
        if scheduled_tokens <= FPM_MAX_CUDAGRAPH_CAPTURE_TOKENS:
            capture_sizes.add(scheduled_tokens)
            continue
        capture_sizes.add(FPM_MAX_CUDAGRAPH_CAPTURE_TOKENS)
        remainder = scheduled_tokens % FPM_MAX_CUDAGRAPH_CAPTURE_TOKENS
        if remainder:
            capture_sizes.add(remainder)
    return tuple(sorted(capture_sizes))


def _minimum_kv_bytes(tokens_from_kv_bytes: Callable[[float], int], tokens: int, capacity_bytes: int) -> int:
    """Invert AIC's KV curve without assuming constant bytes per token."""

    if tokens_from_kv_bytes(float(capacity_bytes)) < tokens:
        return capacity_bytes + 1
    low, high = 0, capacity_bytes
    while low < high:
        middle = (low + high) // 2
        if tokens_from_kv_bytes(float(middle)) >= tokens:
            high = middle
        else:
            low = middle + 1
    return low


def _required_point_kv_bytes(
    point: FPMPoint,
    *,
    block_size: int,
    estimate: _EnvelopeEstimate,
) -> int:
    # Runtime allocates whole blocks independently for every request.  Invert
    # the AIC model for one aligned sequence, then multiply by local DP batch.
    sequence_tokens = math.ceil((point.prefix_length + point.suffix_length) / block_size) * block_size
    if estimate.kv_bytes_for_sequence is not None:
        per_request = math.ceil(estimate.kv_bytes_for_sequence(sequence_tokens))
    else:
        per_request = _minimum_kv_bytes(
            estimate.tokens_from_kv_bytes,
            sequence_tokens,
            estimate.total_gpu_capacity_bytes,
        )
    return point.batch_size * per_request


def _required_fraction(
    estimate: _EnvelopeEstimate,
    required_kv_bytes: int,
    *,
    runtime_reserve_bytes: int = 0,
) -> float:
    # AIC's tolerance reserves part of the modeled KV budget.  Solving the vLLM
    # of-total formula for the requested fraction makes the emitted engine value
    # explicit instead of inheriting a YAML/default value.
    protected_kv_bytes = required_kv_bytes / (1.0 - FPM_KV_TOLERANCE_FRACTION)
    return (estimate.non_kv_bytes + protected_kv_bytes + runtime_reserve_bytes) / estimate.total_gpu_capacity_bytes


def _runtime_reserve_bytes(total_gpu_capacity_bytes: int) -> int:
    """Reserve unmodeled runtime memory inside the 0.9 engine ceiling."""

    if total_gpu_capacity_bytes <= 0:
        raise ValueError("total_gpu_capacity_bytes must be positive")
    return max(
        math.ceil(total_gpu_capacity_bytes * FPM_RUNTIME_RESERVE_FRACTION),
        FPM_RUNTIME_RESERVE_MIN_BYTES,
    )


def build_capacity_model(
    *,
    model_path: str,
    system: str,
    backend: str,
    capability: ModelCapabilityProfile,
    topology: ParallelTopology,
    kv_cache_dtype: str,
    max_batch_size_hint: int,
    max_num_tokens_hint: int,
) -> FPMCapacityModel:
    system_spec = load_system_spec(system)
    capacity_bytes = int((system_spec.get("gpu") or {}).get("mem_capacity") or 0)
    if capacity_bytes <= 0:
        raise ValueError(f"AIC system spec has no positive GPU memory capacity for {system!r}")

    def build(max_num_tokens: int) -> KVCacheEstimator:
        return KVCacheEstimator.from_request(
            model_path,
            system,
            backend,
            capability.aic_database_version,
            max_num_tokens=max_num_tokens,
            max_batch_size=max_batch_size_hint,
            tp_size=topology.tp,
            pp_size=topology.pp,
            attention_dp_size=topology.dp,
            moe_tp_size=topology.moe_tp,
            moe_ep_size=topology.moe_ep,
            gemm_quant_mode=capability.dtype.gemm_quant_mode,
            moe_quant_mode=capability.dtype.moe_quant_mode,
            kvcache_quant_mode=kv_cache_dtype,
            fmha_quant_mode=capability.dtype.fmha_quant_mode,
            comm_quant_mode=capability.dtype.comm_quant_mode,
        )

    try:
        baseline: dict[str, Any] = build(1).breakdown
        high: dict[str, Any] = build(max(1, max_num_tokens_hint)).breakdown
    except Exception as error:
        if capability.support_level != "bootstrap_template":
            raise ValueError(
                f"AIC cannot build a reusable memory model for topology={topology.to_dict()}: {error}"
            ) from error
        return FPMCapacityModel(
            total_gpu_capacity_bytes=capacity_bytes,
            fixed_non_activation_bytes=None,
            activation_floor_bytes=None,
            activation_bytes_per_token=None,
            tokens_from_kv_bytes=None,
            kv_bytes_for_sequence=None,
            source="bootstrap_runtime_verified",
            error=str(error),
        )

    fixed = sum(int(baseline[name]) for name in ("weights_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"))
    activation_floor = int(baseline["activations_bytes"])
    high_activation = int(high["activations_bytes"])
    activation_slope = high_activation / max(1, max_num_tokens_hint) if high_activation > activation_floor else 0.0
    tokens_from_kv_bytes = baseline["tokens_from_kv_bytes"]
    model = getattr(tokens_from_kv_bytes, "__self__", None)
    kv_bytes_for_sequence = getattr(model, "get_kvcache_bytes_per_sequence", None)
    if not callable(kv_bytes_for_sequence):
        kv_bytes_for_sequence = None
    return FPMCapacityModel(
        total_gpu_capacity_bytes=capacity_bytes,
        fixed_non_activation_bytes=fixed,
        activation_floor_bytes=activation_floor,
        activation_bytes_per_token=activation_slope,
        tokens_from_kv_bytes=tokens_from_kv_bytes,
        kv_bytes_for_sequence=kv_bytes_for_sequence,
        source="aic_native_execution_envelope",
    )


def filter_memory_feasible_points(
    points: tuple[FPMPoint, ...] | list[FPMPoint],
    *,
    capacity_model: FPMCapacityModel,
    block_size: int,
) -> tuple[tuple[FPMPoint, ...], tuple[PointCapacityDecision, ...]]:
    """Build a pointwise memory-feasible population before sparse sampling."""

    unique = tuple(sorted(set(points)))
    if not unique:
        raise ValueError("memory-feasibility population must not be empty")
    if capacity_model.error is not None:
        # Bootstrap models have no AIC memory curve. Their sampling population
        # remains intact and the explicit runtime capacity gate stays fail-closed.
        return unique, ()

    runtime_reserve_bytes = _runtime_reserve_bytes(capacity_model.total_gpu_capacity_bytes)
    feasible = []
    decisions = []
    for point in unique:
        max_batch, max_tokens, max_length = _engine_envelope((point,))
        estimate = capacity_model.envelope(
            max_batch_size=max_batch,
            max_num_tokens=max_tokens,
            max_seq_len=max_length,
        )
        required_kv = _required_point_kv_bytes(point, block_size=block_size, estimate=estimate)
        required_fraction = _round_fraction_up(
            _required_fraction(
                estimate,
                required_kv,
                runtime_reserve_bytes=runtime_reserve_bytes,
            )
        )
        admitted = required_fraction <= FPM_MAX_GPU_MEMORY_UTILIZATION
        if admitted:
            feasible.append(point)
        decisions.append(
            PointCapacityDecision(
                point=point,
                admitted=admitted,
                disposition="feasible_population" if admitted else "rejected_before_sampling",
                reason=(
                    "pointwise AIC envelope plus static runtime reserve fits the memory ceiling"
                    if admitted
                    else "pointwise AIC envelope plus static runtime reserve exceeds the memory ceiling"
                ),
                required_memory_fraction=required_fraction,
            )
        )
    if not feasible:
        raise ValueError("no points remain after pointwise AIC memory-feasibility filtering")
    return tuple(feasible), tuple(decisions)


def build_memory_selection_admission(
    *,
    capacity_model: FPMCapacityModel,
    block_size: int,
) -> Callable[[tuple[FPMPoint, ...], FPMPoint], bool]:
    """Return a monotonic joint-envelope predicate for sparse sampling."""

    if capacity_model.error is not None:
        return lambda _selected, _candidate: True

    runtime_reserve_bytes = _runtime_reserve_bytes(capacity_model.total_gpu_capacity_bytes)
    kv_bytes: dict[FPMPoint, int] = {}
    cached_selected: tuple[FPMPoint, ...] | None = None
    cached_envelope = (0, 0, 0, 0)

    def point_kv(point: FPMPoint) -> int:
        value = kv_bytes.get(point)
        if value is None:
            max_batch, max_tokens, max_length = _engine_envelope((point,))
            estimate = capacity_model.envelope(
                max_batch_size=max_batch,
                max_num_tokens=max_tokens,
                max_seq_len=max_length,
            )
            value = _required_point_kv_bytes(point, block_size=block_size, estimate=estimate)
            kv_bytes[point] = value
        return value

    def admit(selected: tuple[FPMPoint, ...], candidate: FPMPoint) -> bool:
        nonlocal cached_selected, cached_envelope
        if selected != cached_selected:
            if selected:
                max_batch, max_tokens, max_length = _engine_envelope(selected)
                max_kv = max(point_kv(point) for point in selected)
                cached_envelope = (max_batch, max_tokens, max_length, max_kv)
            else:
                cached_envelope = (0, 0, 0, 0)
            cached_selected = selected
        max_batch, max_tokens, max_length, max_kv = cached_envelope
        candidate_length = candidate.prefix_length + candidate.suffix_length + 1
        estimate = capacity_model.envelope(
            max_batch_size=max(max_batch, candidate.batch_size),
            max_num_tokens=max(max_tokens, candidate.batch_size * candidate.suffix_length),
            max_seq_len=max(max_length, candidate_length),
        )
        required_fraction = _round_fraction_up(
            _required_fraction(
                estimate,
                max(max_kv, point_kv(candidate)),
                runtime_reserve_bytes=runtime_reserve_bytes,
            )
        )
        return required_fraction <= FPM_MAX_GPU_MEMORY_UTILIZATION

    return admit


def _bootstrap_execution_profile(
    design: SamplingDesign,
    *,
    capacity_bytes: int,
    error: Exception,
    runtime_reserve_bytes: int,
) -> FPMExecutionProfile:
    """Freeze a fail-closed runtime-verified profile when AIC cannot model it."""

    selected = design.selected
    max_batch, max_tokens, max_length = _engine_envelope(selected)
    compatible = tuple(
        point
        for point in design.ordered_points
        if point.batch_size <= max_batch
        and point.batch_size * point.suffix_length <= max_tokens
        and point.prefix_length + point.suffix_length + 1 <= max_length
    )
    selected_set = set(selected)
    ordered = tuple(selected) + tuple(point for point in compatible if point not in selected_set)
    ordered_set = set(ordered)
    decisions = tuple(
        PointCapacityDecision(
            point,
            point in ordered_set,
            "selected" if point in selected_set else "reserve" if point in ordered_set else "rejected",
            f"runtime verification required because AIC memory modeling is unavailable: {error}",
            None,
        )
        for point in design.ordered_points
    )
    capture_sizes = _cudagraph_capture_sizes(ordered)
    return FPMExecutionProfile(
        ordered_points=ordered,
        selected_point_count=len(selected),
        max_batch_size=max_batch,
        max_num_tokens=max_tokens,
        max_seq_len=max_length,
        gpu_memory_utilization=FPM_MAX_GPU_MEMORY_UTILIZATION,
        cudagraph_capture_sizes=capture_sizes,
        memory_source="bootstrap_runtime_verified",
        memory_fraction_ceiling=FPM_MAX_GPU_MEMORY_UTILIZATION,
        kv_tolerance_fraction=FPM_KV_TOLERANCE_FRACTION,
        total_gpu_capacity_bytes=capacity_bytes,
        non_kv_bytes=None,
        required_kv_bytes=None,
        decisions=decisions,
        aic_non_kv_bytes=None,
        runtime_reserve_bytes=runtime_reserve_bytes,
        runtime_reserve_fraction=runtime_reserve_bytes / capacity_bytes,
    )


def build_execution_profile(
    *,
    model_path: str,
    system: str,
    backend: str,
    capability: ModelCapabilityProfile,
    topology: ParallelTopology,
    kv_cache_dtype: str,
    design: SamplingDesign,
    block_size: int,
    capacity_model: FPMCapacityModel | None = None,
    runtime_reserve_bytes_override: int | None = None,
    pre_admission_decisions: tuple[PointCapacityDecision, ...] = (),
) -> FPMExecutionProfile:
    """Select the first capacity-legal sparse points and derive engine limits."""

    if capacity_model is None:
        capacity_model = build_capacity_model(
            model_path=model_path,
            system=system,
            backend=backend,
            capability=capability,
            topology=topology,
            kv_cache_dtype=kv_cache_dtype,
            max_batch_size_hint=max(point.batch_size for point in design.ordered_points),
            max_num_tokens_hint=max(point.batch_size * point.suffix_length for point in design.ordered_points),
        )
    capacity_bytes = capacity_model.total_gpu_capacity_bytes
    runtime_reserve_bytes = (
        _runtime_reserve_bytes(capacity_bytes)
        if runtime_reserve_bytes_override is None
        else runtime_reserve_bytes_override
    )
    if runtime_reserve_bytes < 0:
        raise ValueError("runtime_reserve_bytes_override must be non-negative")
    if capacity_model.error is not None:
        return _bootstrap_execution_profile(
            design,
            capacity_bytes=capacity_bytes,
            error=ValueError(capacity_model.error),
            runtime_reserve_bytes=runtime_reserve_bytes,
        )

    target_count = len(design.selected)
    selected: list[FPMPoint] = list(design.selected)
    rejected: dict[FPMPoint, PointCapacityDecision] = {}

    def evaluate(points: list[FPMPoint] | tuple[FPMPoint, ...]) -> tuple[_EnvelopeEstimate, int, float]:
        max_batch, max_tokens, max_length = _engine_envelope(points)
        estimate = capacity_model.envelope(
            max_batch_size=max_batch,
            max_num_tokens=max_tokens,
            max_seq_len=max_length,
        )
        required_kv = max(_required_point_kv_bytes(point, block_size=block_size, estimate=estimate) for point in points)
        return (
            estimate,
            required_kv,
            _required_fraction(
                estimate,
                required_kv,
                runtime_reserve_bytes=runtime_reserve_bytes,
            ),
        )

    remaining_index = target_count
    while True:
        try:
            estimate, required_kv, required_fraction = evaluate(selected)
        except Exception as error:
            raise ValueError(
                f"AIC cannot evaluate an execution-envelope memory estimate for topology={topology.to_dict()}, "
                f"phase={design.phase}: {error}"
            ) from error
        if required_fraction <= FPM_MAX_GPU_MEMORY_UTILIZATION:
            break
        if remaining_index >= len(design.ordered_points):
            raise ValueError(
                f"fewer than {target_count} {design.phase} points fit topology={topology.to_dict()} "
                f"at gpu_memory_utilization<={FPM_MAX_GPU_MEMORY_UTILIZATION}"
            )

        point_kv = {
            point: _required_point_kv_bytes(point, block_size=block_size, estimate=estimate) for point in selected
        }
        if estimate.non_kv_bytes + runtime_reserve_bytes >= capacity_bytes * FPM_MAX_GPU_MEMORY_UTILIZATION:
            culprit = max(
                selected,
                key=lambda point: (
                    point.batch_size * point.suffix_length,
                    point.batch_size,
                    point.prefix_length + point.suffix_length,
                    point.key,
                ),
            )
            reason = "point forces the activation envelope plus static runtime reserve above the ceiling"
        else:
            culprit = max(
                selected,
                key=lambda point: (
                    point_kv[point],
                    point.batch_size * (point.prefix_length + point.suffix_length),
                    point.key,
                ),
            )
            reason = "point forces block-aligned KV capacity above the FPM memory-fraction ceiling"
        rejected[culprit] = PointCapacityDecision(
            culprit,
            False,
            "rejected",
            reason,
            _round_fraction_up(required_fraction),
        )
        selected.remove(culprit)
        selected.append(design.ordered_points[remaining_index])
        remaining_index += 1

    estimate, required_kv, required_fraction = evaluate(selected)
    memory_fraction = _round_fraction_up(required_fraction)
    reserves: list[FPMPoint] = []
    for point in design.ordered_points[remaining_index:]:
        point_required_kv = _required_point_kv_bytes(point, block_size=block_size, estimate=estimate)
        compatible = (
            point.batch_size <= estimate.max_batch_size
            and point.batch_size * point.suffix_length <= estimate.max_num_tokens
            and point.prefix_length + point.suffix_length + 1 <= estimate.max_seq_len
            and _required_fraction(
                estimate,
                point_required_kv,
                runtime_reserve_bytes=runtime_reserve_bytes,
            )
            <= memory_fraction
        )
        if compatible:
            reserves.append(point)
        else:
            rejected[point] = PointCapacityDecision(
                point,
                False,
                "rejected",
                "reserve point is outside the frozen selected-point execution profile",
                _round_fraction_up(
                    _required_fraction(
                        estimate,
                        point_required_kv,
                        runtime_reserve_bytes=runtime_reserve_bytes,
                    )
                ),
            )

    selected_set = set(selected)
    ordered = tuple(selected + reserves)
    sampled_decisions = tuple(
        rejected.get(point)
        or PointCapacityDecision(
            point,
            True,
            "selected" if point in selected_set else "reserve",
            "admitted by AIC execution-envelope, static runtime reserve, and block-aligned KV capacity",
            memory_fraction,
        )
        for point in design.ordered_points
    )
    pre_by_point = {decision.point: decision for decision in pre_admission_decisions}
    sampled_by_point = {decision.point: decision for decision in sampled_decisions}
    decisions = tuple(
        sampled_by_point.get(point) or pre_by_point[point]
        for point in sorted(set(pre_by_point) | set(sampled_by_point))
    )
    capture_sizes = _cudagraph_capture_sizes(ordered)
    return FPMExecutionProfile(
        ordered_points=ordered,
        selected_point_count=target_count,
        max_batch_size=estimate.max_batch_size,
        max_num_tokens=estimate.max_num_tokens,
        max_seq_len=estimate.max_seq_len,
        gpu_memory_utilization=memory_fraction,
        cudagraph_capture_sizes=capture_sizes,
        memory_source=f"{capacity_model.source}+static_runtime_reserve",
        memory_fraction_ceiling=FPM_MAX_GPU_MEMORY_UTILIZATION,
        kv_tolerance_fraction=FPM_KV_TOLERANCE_FRACTION,
        total_gpu_capacity_bytes=estimate.total_gpu_capacity_bytes,
        non_kv_bytes=estimate.non_kv_bytes + runtime_reserve_bytes,
        required_kv_bytes=required_kv,
        decisions=decisions,
        aic_non_kv_bytes=estimate.non_kv_bytes,
        runtime_reserve_bytes=runtime_reserve_bytes,
        runtime_reserve_fraction=runtime_reserve_bytes / estimate.total_gpu_capacity_bytes,
    )


def admit_model_residency(
    *,
    model_path: str,
    system: str,
    backend: str,
    capability: ModelCapabilityProfile,
    topologies: tuple[ParallelTopology, ...],
) -> tuple[tuple[ParallelTopology, ...], tuple[CapacityDecision, ...], float]:
    """Keep tuples whose rank-local non-KV state leaves a positive KV budget.

    This is deliberately a model-residency check (one token, one request), not
    a claim that every workload point fits. Point-level capacity remains a
    separate collection concern.
    """

    memory_fraction = FPM_MAX_GPU_MEMORY_UTILIZATION
    system_spec = load_system_spec(system)
    capacity_bytes = int((system_spec.get("gpu") or {}).get("mem_capacity") or 0)
    if capacity_bytes <= 0:
        raise ValueError(f"AIC system spec has no positive GPU memory capacity for {system!r}")
    decisions: list[CapacityDecision] = []
    admitted: list[ParallelTopology] = []
    for topology in topologies:
        try:
            estimate = estimate_kv_cache(
                model_path,
                system,
                backend,
                capability.aic_database_version,
                max_num_tokens=1,
                max_batch_size=1,
                memory_fraction_kind="of_total",
                memory_fraction_value=memory_fraction,
                tp_size=topology.tp,
                pp_size=topology.pp,
                attention_dp_size=topology.dp,
                moe_tp_size=topology.moe_tp,
                moe_ep_size=topology.moe_ep,
                gemm_quant_mode=capability.dtype.gemm_quant_mode,
                moe_quant_mode=capability.dtype.moe_quant_mode,
                kvcache_quant_mode=capability.dtype.native_kv_cache_dtype,
                fmha_quant_mode=capability.dtype.fmha_quant_mode,
                comm_quant_mode=capability.dtype.comm_quant_mode,
                gpu_memory_capacity_bytes_override=capacity_bytes,
                allow_naive_fallback=capability.support_level == "bootstrap_template",
            )
        except ValueError as error:
            if capability.support_level == "bootstrap_template" and str(error).startswith(
                "unsupported model/backend/GPU for KV-cache estimation"
            ):
                # The generic template is specifically the path for models AIC
                # cannot build. Keep the candidate, but make the unverified
                # residency explicit so execution cannot mistake it for proof.
                decisions.append(
                    CapacityDecision(
                        topology,
                        True,
                        "bootstrap_unverified",
                        f"AIC and naive residency estimators unavailable: {error}",
                        None,
                    )
                )
                admitted.append(topology)
                continue
            decisions.append(CapacityDecision(topology, False, "aic_memory", str(error), None))
            continue

        kv_tokens = int(estimate["total_kv_size_tokens"])
        source = str(estimate["source"])
        decisions.append(
            CapacityDecision(
                topology,
                True,
                source,
                "rank-local model state leaves a positive baseline KV budget",
                kv_tokens,
            )
        )
        admitted.append(topology)

    if not admitted:
        reasons = "; ".join(decision.reason for decision in decisions)
        raise ValueError(f"no typical topology passes AIC model-residency admission: {reasons}")
    return tuple(admitted), tuple(decisions), memory_fraction
