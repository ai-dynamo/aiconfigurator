# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large-EP MoE communication family, unified across SGLang, vLLM, and TRT-LLM.

Models the all-to-all communication of large-scale expert-parallel MoE
(dispatch/combine, plus TRT-LLM's prepare phase) with one comm-backend
registry shared by all three inference backends. On TRT-LLM this covers the
*wideEP* path only — non-wideEP TRT-LLM paths are untouched.

``MOE_A2A_BACKENDS`` maps backend name to its :class:`MoECommBackendSpec`
(framework/phase applicability plus feasibility rules).
``MoEAllToAll`` is the op class over the unified ``moe_a2a_perf.parquet``
comm table, served by the engine table view (``_moe_a2a_data``, with legacy
per-backend adapters folded engine-side) as one nested dict keyed by
``[comm_backend][phase][comm_dtype][ep_size][node_num][hidden_size][topk]
[num_experts][sms][num_tokens]``. The class owns the view binding
(class-level cache + ``load_data``).

The module also owns the large-EP compute side of the same family:
``MoEExpertCompute`` over the unified ``moe_expert_compute_perf.parquet``
table (view attribute ``_moe_ep_data``, legacy sglang/trtllm wideep adapters
folded engine-side), keyed by ``[kernel_source][quant][distribution]
[inference_phase][topk][num_experts][num_slots][hidden_size][inter_size]
[moe_tp_size][moe_ep_size][num_tokens]``. The Python parsers for both tables
retired with the deprecation-cleanup PR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family."""
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


@dataclass(frozen=True)
class MoECommBackendSpec:
    """Static description of one MoE all-to-all comm backend."""

    name: str
    frameworks: tuple[str, ...]  # ("sglang", "vllm") or ("trtllm",)
    inference_phases: tuple[str, ...]  # ("context",) | ("generation",) | ("context", "generation")
    comm_phases: tuple[str, ...]  # ("dispatch", "combine") | ("prepare", "dispatch", "combine")
    min_sm: int = 0
    max_topk: int = 8

    def feasible(
        self,
        *,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        sm_version: int | None = None,
    ) -> bool:
        """Whether this backend can serve the given MoE parallelism config."""
        return (
            topk <= self.max_topk
            and moe_tp_size == 1
            and 1 < moe_ep_size <= num_experts
            and num_experts % moe_ep_size == 0
            and (sm_version is None or sm_version >= self.min_sm)
        )


MOE_A2A_BACKENDS: dict[str, MoECommBackendSpec] = {
    "deepep_ht": MoECommBackendSpec(
        name="deepep_ht",
        frameworks=("sglang", "vllm"),
        inference_phases=("context",),
        comm_phases=("dispatch", "combine"),
    ),
    "deepep_ll": MoECommBackendSpec(
        name="deepep_ll",
        frameworks=("sglang", "vllm"),
        inference_phases=("generation",),
        comm_phases=("dispatch", "combine"),
    ),
    "nvlink_two_sided": MoECommBackendSpec(
        name="nvlink_two_sided",
        frameworks=("trtllm",),
        inference_phases=("context", "generation"),
        comm_phases=("prepare", "dispatch", "combine"),
        min_sm=100,
    ),
    "nvlink_one_sided": MoECommBackendSpec(
        name="nvlink_one_sided",
        frameworks=("trtllm",),
        inference_phases=("context", "generation"),
        comm_phases=("dispatch", "combine"),
        min_sm=100,
    ),
}


def nodes_for(ep_size: int, num_gpus_per_node: int) -> int:
    """Node count needed to host ``ep_size`` EP ranks (ceil division)."""
    return -(-ep_size // num_gpus_per_node)


_A2A_PHASES = ("prepare", "dispatch", "combine")


def _validate_a2a_request(comm_backend: str, phase: str) -> None:
    """Shared ctor/query validation: unknown backend or phase is a ValueError.

    The per-backend check is a guard, not a live code path: the block builder
    iterates ``spec.comm_phases`` itself and the coverage probe walks table
    keys without validating — so a combination like ``("deepep_ht",
    "prepare")`` can only come from future misuse, and should fail here where
    the intent is expressed rather than later as a data miss.
    """
    if comm_backend not in MOE_A2A_BACKENDS:
        raise ValueError(f"Invalid comm_backend '{comm_backend}'. Must be one of {sorted(MOE_A2A_BACKENDS)}")
    if phase not in _A2A_PHASES:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {list(_A2A_PHASES)}")
    supported = MOE_A2A_BACKENDS[comm_backend].comm_phases
    if phase not in supported:
        raise ValueError(
            f"comm_backend '{comm_backend}' does not implement phase '{phase}'; supported: {list(supported)}"
        )


class MoEAllToAll(Operation):
    """Unified large-EP MoE all-to-all comm op (one phase per instance).

    Owns ``_moe_a2a_data`` — the unified comm table loaded by
    the engine view (new-schema ``moe_a2a_perf.parquet`` plus the
    three legacy per-backend adapters). Loaded on every inference backend
    ({"sglang", "vllm", "trtllm"} all have legacy comm sources); ``None``
    otherwise. Comm ops see per-rank token counts: ``query(x=...)`` scales by
    ``scale_factor`` only — never by ``attention_dp_size``.
    ``attention_tp_size`` divides the token key before the lookup — legacy
    fidelity with ``MoEDispatch``'s ``num_tokens // self._scale_num_tokens``
    (plain floor division, no ``max(1, ...)`` guard). ``comm_dtype``
    resolves exact-first, then the legacy ``fp8_block`` -> ``fp8`` behavioral
    aliasing, then the sole-collected-dtype fallback (typed miss otherwise).
    """

    _data_cache: ClassVar[dict] = {}

    _SUPPORTED_BACKENDS: ClassVar[tuple[str, ...]] = ("sglang", "vllm", "trtllm")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        phase: str,
        comm_backend: str,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_ep_size: int,
        node_num: int,
        comm_dtype: str = "default",
        sms: int = 0,
        attention_tp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        _validate_a2a_request(comm_backend, phase)
        self._phase = phase
        self._comm_backend = comm_backend
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_ep_size = moe_ep_size
        self._node_num = node_num
        self._comm_dtype = comm_dtype
        self._sms = sms
        self._attention_tp_size = attention_tp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's unified moe_a2a table view (new
        schema + legacy adapters, merged engine-side) on the three inference
        backends; binds ``None`` otherwise.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend in cls._SUPPORTED_BACKENDS:
                cls._data_cache[key] = load_view(database, "_moe_a2a_data", PerfDataFilename.moe_a2a)
            else:
                cls._data_cache[key] = None

            cls._record_load()

        if "_moe_a2a_data" not in database.__dict__:
            database._moe_a2a_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"


# ---------------------------------------------------------------------------
# EP MoE compute (moe_expert_compute_perf.parquet) — same family, compute side
# ---------------------------------------------------------------------------


_EP_PHASES = ("context", "generation")


def _validate_ep_phase(inference_phase: str) -> None:
    """Shared ctor/query validation: an unknown inference phase is a ValueError."""
    if inference_phase not in _EP_PHASES:
        raise ValueError(f"Invalid inference_phase '{inference_phase}'. Must be one of {list(_EP_PHASES)}")


class MoEExpertCompute(Operation):
    """Unified large-EP MoE expert-compute op (one inference phase per instance).

    Owns ``_moe_ep_data`` — the unified compute table loaded by
    the engine view (new-schema ``moe_expert_compute_perf.parquet`` plus the
    legacy sglang wideep context/generation and trtllm wideep adapters).
    Loaded on every inference backend ({"sglang", "vllm", "trtllm"} all have
    legacy compute sources); ``None`` otherwise. ``query(x=...)`` scales
    tokens by ``attention_dp_size`` (attention DP globalizes tokens through
    the A2A dispatch — the same scaling as the legacy ``MoE`` /
    ``TrtLLMWideEPMoE`` query paths) and always queries ``moe_tp_size=1``:
    the large-EP family is EP-only. ``num_slots`` defaults to ``num_experts``
    (no EPLB redundancy); ``kernel_source=None`` auto-resolves per backend at
    query time (see :meth:`_resolve_kernel_source`). ``enable_eplb=True`` is
    legacy fidelity with the sglang MoE query: tokens become
    ``int(tokens * 0.8)`` before the table lookup when the phase is context
    AND the resolved kernel leg is sglang-adapted
    (``_SGLANG_ADAPTED_KERNEL_SOURCES``) — never on the trtllm legs, whose
    EPLB effect rides the ``_eplb`` distribution suffix instead.
    """

    _data_cache: ClassVar[dict] = {}

    _SUPPORTED_BACKENDS: ClassVar[tuple[str, ...]] = ("sglang", "vllm", "trtllm")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        attention_dp_size: int,
        inference_phase: str,
        num_slots: int | None = None,
        kernel_source: str | None = None,
        is_gated: bool = True,
        enable_eplb: bool = False,
    ) -> None:
        super().__init__(name, scale_factor)
        _validate_ep_phase(inference_phase)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._topk = topk
        self._num_experts = num_experts
        self._num_slots = num_slots if num_slots is not None else num_experts
        self._moe_ep_size = moe_ep_size
        self._quant_mode = quant_mode
        self._workload_distribution = workload_distribution
        self._attention_dp_size = attention_dp_size
        self._inference_phase = inference_phase
        self._kernel_source = kernel_source
        self._is_gated = is_gated
        self._enable_eplb = enable_eplb
        # Weight bytes retired to the engine (Op::weight_bytes): EP-only
        # sizing by num_experts NOT num_slots stays parity-pinned there
        # (AIC-1674 tracks the intentional num_slots delta).

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's unified moe_ep table view (new
        schema + legacy adapters, merged engine-side) on the three inference
        backends; binds ``None`` otherwise.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend in cls._SUPPORTED_BACKENDS:
                cls._data_cache[key] = load_view(database, "_moe_ep_data", PerfDataFilename.moe_expert_compute)
            else:
                cls._data_cache[key] = None

            cls._record_load()

        if "_moe_ep_data" not in database.__dict__:
            database._moe_ep_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # kernel_source auto-resolution (kernel_source=None)
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_kernel_source(cls, database: PerfDatabase, quant_mode: common.MoEQuantMode) -> str:
        """Resolve the collected kernel key when the caller pins none.

        sglang/vllm large-EP MoE has a single collected kernel
        (``"deepep_moe"``, spec §4.2). trtllm replicates
        ``TrtLLMWideEPMoE._select_kernel`` (TensorRT-LLM's
        ``MoEOpSelector.select_op``) against the unified table's kernel keys:
        Blackwell (SM >= 100) + fp8_block -> ``"deepgemm"``, otherwise
        ``"moe_torch_flow"`` (Cutlass); an absent preferred kernel falls back
        to the first collected kernel key. Copied, not imported — the legacy
        classmethod consults its trtllm-only ``_wideep_moe_compute_data``
        table, which this family retires.
        """
        if database.backend in ("sglang", "vllm"):
            return "deepep_moe"

        cls.load_data(database)
        sm_version = database.system_spec["gpu"]["sm_version"]
        is_blackwell = sm_version >= 100
        quant_mode_str = quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode)
        preferred = "deepgemm" if is_blackwell and "fp8_block" in quant_mode_str else "moe_torch_flow"

        ep_data = database._moe_ep_data
        if ep_data:
            available_kernels = list(ep_data.keys())
            if preferred in available_kernels:
                return preferred
            if available_kernels:
                fallback = available_kernels[0]
                logger.debug(f"Preferred MoE kernel '{preferred}' not available, falling back to '{fallback}'")
                return fallback

        return preferred

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"

    def _engine_query_plan(self, kwargs: dict):
        """Legacy per-call ``quant_mode`` override: rebuild the twin with the
        requested quant before engine evaluation."""
        op, eval_kwargs = super()._engine_query_plan(kwargs)
        quant_mode = kwargs.get("quant_mode")
        if quant_mode is not None and quant_mode != self._quant_mode:
            import copy

            op = copy.copy(self)
            op._quant_mode = quant_mode
        return op, eval_kwargs
