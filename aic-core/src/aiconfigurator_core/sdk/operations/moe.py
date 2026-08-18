# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE family (ISSUE-12 / ISSUE-13).

Op classes migrated from ``_legacy.py``:

- ``MoE`` (ISSUE-12) — Mixture-of-Experts compute op. Owns:
    * ``_moe_data`` — regular MoE table
    * ``_moe_low_latency_data`` — TRT-LLM low-latency NVFP4 kernel table
      (folded from the same perf table as the regular MoE data by the engine
      view — `table_view.rs::view_moe` returns the twin pair)
    * ``_wideep_context_moe_data`` — SGLang WideEP context MoE table
    * ``_wideep_generation_moe_data`` — SGLang WideEP generation MoE table
  Table selection (backend + ``moe_backend`` + ``num_tokens`` +
  ``quant_mode`` + ``is_gated``) happens in the engine; Python keeps all
  four tables loaded on the data plane.

- ``MoEDispatch`` (ISSUE-12) — MoE comm-cost op. Owns:
    * ``_wideep_deepep_normal_data`` — SGLang DeepEP normal-mode dispatch
    * ``_wideep_deepep_ll_data`` — SGLang DeepEP low-latency dispatch
  Dispatches at query time across NCCL, CustomAllReduce, TRT-LLM AllToAll,
  and SGLang DeepEP based on backend + ``_sm_version`` + ``_moe_backend``.

- ``TrtLLMWideEPMoE`` (ISSUE-13) — TRT-LLM WideEP MoE compute op. Owns:
    * ``_wideep_moe_compute_data`` — TRT-LLM WideEP compute table
  Pulls kernel selection logic (``_select_moe_kernel``) onto the class
  alongside the data it consults.

- ``TrtLLMWideEPMoEDispatch`` (ISSUE-13) — TRT-LLM WideEP All2All op. Owns:
    * ``_trtllm_alltoall_data`` — TRT-LLM All2All table (prepare/dispatch/combine)
  Pulls ``_select_alltoall_kernel`` and the FP8/FP8-block quant-mode
  normalization helper onto the class alongside the data.

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``. The
WideEP tables are loaded only when ``database.backend == "sglang"`` (MoE /
MoEDispatch SGLang-only WideEP slots) or ``database.backend == "trtllm"``
(``TrtLLMWideEPMoE`` / ``TrtLLMWideEPMoEDispatch``); on other backends the
corresponding cache slot is ``None`` and consumers must guard.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows

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


# Per-quant achieved-util LEVEL e(q) for MoE, keyed by the (memory, compute) profile
# (which encodes (weight, activation) precision: memory∈{2,1,0.5}↔w{16,8,4},
# compute∈{1,2,4}↔a{16,8,4}). Used ONLY by the cross-PROFILE transfer tier: when the
# query quant has no data of any profile, borrow the nearest collected quant's util
# curve and rescale it by e(query)/e(ref). util is the achieved kernel efficiency
# (SOL already absorbs the coefficients); its LEVEL differs systematically by quant —
# 4-bit-weight kernels run far below their (higher) roofline, and efficiency rises mildly
# as activation precision drops. The RATIO e(query)/e(ref) is what matters and is
# ~stack-stable (≈10%; e.g. w4a16/fp8 = 0.17 on b200 vs 0.18 on h100). Levels are
# data-derived on b200/trtllm where collected, inferred from the structure otherwise.
# A SINGLE scalar per quant by design: the analytic SOL's compute/mem split is not
# trustworthy enough to calibrate per-component (validated — splitting blows up because
# the SOL attribution doesn't match the kernel's real bottleneck). Levels are relative
# and tunable; only ratios are consumed.
# PROJECTION of the engine's table (PR-6): the Rust
# `operators/moe.rs::MOE_QUANT_UTIL_LEVEL` is the single source (with the
# same per-row [data]/[inferred] provenance notes); rebuilding the dict from
# the FFI ends the two-sided sync discipline every new quant used to need.
_MOE_QUANT_UTIL_LEVEL: dict[tuple[float, float], float] = {
    (memory, compute): level for memory, compute, level in aiconfigurator_core.moe_quant_util_levels()
}
_MOE_QUANT_UTIL_DEFAULT = 0.30  # unlisted profile: mid-range relative level


def xprofile_util_level_known(quant_mode) -> bool:
    """Whether the MoE util-LEVEL table lists this quant's profile.

    The runtime ladder falls back to ``_MOE_QUANT_UTIL_DEFAULT`` for unlisted
    profiles; the validate gate deliberately does NOT (admitting a quant
    nobody calibrated would hide the missing level line the add-a-quant
    recipe requires), so it asks this instead of reaching into the table."""
    return util_empirical.quant_profile(quant_mode) in _MOE_QUANT_UTIL_LEVEL


# ───────────────────────────────────────────────────────────────────────
# MoE
# ───────────────────────────────────────────────────────────────────────


class MoE(Operation):
    """MoE operation with power tracking."""

    # CP-invariant: the A2A dispatch globalizes tokens across all (cp*ep) ranks,
    # so expert compute sees the full token set regardless of CP and deliberately
    # ignores ``seq_split`` (per-rank cp sharding does not reduce expert work).
    # Marked audited so the post-construction CP wiring (gemma4/hybrid) does not
    # trip the _CP_AWARE gate.
    _CP_AWARE: ClassVar[bool] = True
    _data_cache: ClassVar[dict] = {}
    _low_latency_data_cache: ClassVar[dict] = {}
    _wideep_context_data_cache: ClassVar[dict] = {}
    _wideep_generation_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        attention_dp_size: int,
        is_context: bool = True,
        is_gated: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._quant_mode = quant_mode
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._workload_distribution = workload_distribution
        self._is_context = is_context
        self._is_gated = is_gated
        self._moe_backend = kwargs.get("moe_backend")
        self._enable_eplb = kwargs.get("enable_eplb", False)

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the regular MoE table (tuple of regular +
        low-latency) on all backends, and the SGLang WideEP context /
        generation MoE tables only when ``database.backend == "sglang"``.

        Binds these instance attributes for downstream consumers:
        - ``_moe_data``
        - ``_moe_low_latency_data``
        - ``_wideep_context_moe_data`` (None on non-SGLang)
        - ``_wideep_generation_moe_data`` (None on non-SGLang)
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if (
            key not in cls._data_cache
            or key not in cls._low_latency_data_cache
            or key not in cls._wideep_context_data_cache
            or key not in cls._wideep_generation_data_cache
        ):
            # Fetch every view into locals first so a failure on a later
            # fetch can't leave the caches half-populated — a retry would
            # then KeyError at the bind lines below, masking the real error
            # until clear_cache (same hardening as GEMM.load_data).
            # Regular MoE table — the engine folds one moe_perf read into the
            # default and low-latency views (rows tagged
            # ``kernel_source="moe_torch_flow_min_latency"`` route to the twin).
            moe_loaded = load_view(database, "_moe_data", PerfDataFilename.moe)
            low_latency_loaded = load_view(database, "_moe_low_latency_data", PerfDataFilename.moe)

            # WideEP MoE tables — SGLang-only.
            if database.backend == "sglang":
                wideep_context_loaded = load_view(
                    database, "_wideep_context_moe_data", PerfDataFilename.wideep_context_moe
                )
                wideep_generation_loaded = load_view(
                    database, "_wideep_generation_moe_data", PerfDataFilename.wideep_generation_moe
                )
            else:
                wideep_context_loaded = None
                wideep_generation_loaded = None

            cls._data_cache[key] = moe_loaded
            cls._low_latency_data_cache[key] = low_latency_loaded
            cls._wideep_context_data_cache[key] = wideep_context_loaded
            cls._wideep_generation_data_cache[key] = wideep_generation_loaded
            cls._record_load()

        if "_moe_data" not in database.__dict__:
            database._moe_data = cls._data_cache[key]
        if "_moe_low_latency_data" not in database.__dict__:
            database._moe_low_latency_data = cls._low_latency_data_cache[key]
        if "_wideep_context_moe_data" not in database.__dict__:
            database._wideep_context_moe_data = cls._wideep_context_data_cache[key]
        if "_wideep_generation_moe_data" not in database.__dict__:
            database._wideep_generation_moe_data = cls._wideep_generation_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._low_latency_data_cache.clear()
        cls._wideep_context_data_cache.clear()
        cls._wideep_generation_data_cache.clear()

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


# ───────────────────────────────────────────────────────────────────────
# MoEDispatch
# ───────────────────────────────────────────────────────────────────────


# a comm op to deduce the communication cost of MoE
class MoEDispatch(Operation):
    """MoE dispatch operation. For fine-grained MoE dispatch.

    Owns the SGLang DeepEP tables. On non-SGLang backends, both caches are
    bound to ``None`` and consumers must guard before dereference. Most of
    ``MoEDispatch.query()``'s body delegates to other ops' query methods
    (NCCL, CustomAllReduce, TRT-LLM AllToAll) — only the SGLang DeepEP
    branch consults this class's own tables.
    """

    _normal_data_cache: ClassVar[dict] = {}
    _ll_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        attention_dp_size: int,
        pre_dispatch: bool,
        enable_fp4_all2all: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._enable_fp4_all2all = enable_fp4_all2all
        self._pre_dispatch = pre_dispatch
        self.num_gpus = self._moe_ep_size * self._moe_tp_size
        self._attention_tp_size = moe_tp_size * moe_ep_size // self._attention_dp_size
        self._sms = kwargs.get("sms", 12)
        self._moe_backend = kwargs.get("moe_backend")
        self._is_context = kwargs.get("is_context", True)
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)
        self._quant_mode = kwargs.get("quant_mode")
        self._reduce_results = kwargs.get("reduce_results", True)
        self._attn_cp_size = kwargs.get("attn_cp_size", 1)
        self._attn_ar_modeled = kwargs.get("attn_ar_modeled", False)

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    def get_weights(self, **kwargs):
        """Comm op — no resident weights. Kept as a LOCAL constant (not the
        engine route): the ``moe_backend='deepep_moe'`` variant (still built
        by qwen35) has no opspec variant — its serializer raises the AIC-1601
        tombstone — so the base FFI route would crash memory estimation for
        an op whose weight was always 0.0."""
        return 0.0

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's SGLang DeepEP normal +
        low-latency table views on ``backend == "sglang"`` only; binds
        ``None`` on other backends.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._normal_data_cache or key not in cls._ll_data_cache:
            # Locals first, commit last — a failed second fetch must not
            # leave the caches half-populated (see GEMM.load_data).
            if database.backend == "sglang":
                normal_loaded = load_view(database, "_wideep_deepep_normal_data", PerfDataFilename.wideep_deepep_normal)
                ll_loaded = load_view(database, "_wideep_deepep_ll_data", PerfDataFilename.wideep_deepep_ll)
            else:
                normal_loaded = None
                ll_loaded = None

            cls._normal_data_cache[key] = normal_loaded
            cls._ll_data_cache[key] = ll_loaded
            cls._record_load()

        if "_wideep_deepep_normal_data" not in database.__dict__:
            database._wideep_deepep_normal_data = cls._normal_data_cache[key]
        if "_wideep_deepep_ll_data" not in database.__dict__:
            database._wideep_deepep_ll_data = cls._ll_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._normal_data_cache.clear()
        cls._ll_data_cache.clear()

    # ------------------------------------------------------------------
    # The legacy deepep raw-table queries retired with the per-call stack
    # (#1357 PR-5); the engine reads the same rows via its moe_a2a adapters.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract — legacy body lifted verbatim. Heavy branching across
    # backends; calls ``database.query_*`` helpers that are already
    # migrated (NCCL, CustomAllReduce, TRT-LLM AllToAll) or live in this
    # same class (DeepEP normal / ll via the database delegations).
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"


# ───────────────────────────────────────────────────────────────────────
# TrtLLMWideEPMoE
# ───────────────────────────────────────────────────────────────────────


class TrtLLMWideEPMoE(Operation):
    """TensorRT-LLM WideEP MoE compute op (excludes All2All — see
    ``TrtLLMWideEPMoEDispatch``).

    Owns ``_wideep_moe_compute_data``, loaded only on
    ``database.backend == "trtllm"``. On other backends the cache slot
    binds to ``None`` and the engine's table lookup raises
    ``PerfDataNotAvailableError`` via the standard silicon/hybrid flow.

    Supports three EPLB modes:
    - EPLB off: ``workload_distribution`` without ``_eplb`` suffix,
      ``num_slots = num_experts``
    - EPLB on: ``workload_distribution`` with ``_eplb`` suffix,
      ``num_slots = num_experts``
    - EPLB redundant: ``workload_distribution`` with ``_eplb`` suffix,
      ``num_slots > num_experts``
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        attention_dp_size: int,
        num_slots: int | None = None,  # EPLB slots, defaults to num_experts
        is_gated: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._quant_mode = quant_mode
        self._topk = topk
        self._num_experts = num_experts
        self._num_slots = num_slots if num_slots is not None else num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._workload_distribution = workload_distribution
        self._is_gated = is_gated

    def get_weights(self, **kwargs):
        """Local math: this deprecated class has no OpSpec variant (its
        table is engine-absorbed; PR-7 removes the class), so the base
        engine-routed get_weights cannot serve it. Mirrors Op::weight_bytes
        for the MoE family."""
        num_gemms = 3 if self._is_gated else 2
        weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * self._quant_mode.value.memory
            * num_gemms
            // self._moe_ep_size
            // self._moe_tp_size
        )
        return weights * self._scale_factor

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's ``_wideep_moe_compute_data``
        table view only when ``database.backend == "trtllm"``; binds ``None``
        otherwise.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend == "trtllm":
                cls._data_cache[key] = load_view(
                    database, "_wideep_moe_compute_data", PerfDataFilename.wideep_moe_compute
                )
            else:
                cls._data_cache[key] = None

            cls._record_load()

        if "_wideep_moe_compute_data" not in database.__dict__:
            database._wideep_moe_compute_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Kernel selection (formerly PerfDatabase._select_moe_kernel).
    # Lives here because it consults ``_wideep_moe_compute_data`` and
    # has no other callers.
    # ------------------------------------------------------------------

    @classmethod
    def _select_kernel(cls, database: PerfDatabase, quant_mode: common.MoEQuantMode) -> str:
        """Automatically select MoE computation kernel based on GPU architecture
        and quantization mode.

        Selection logic (based on TensorRT-LLM's MoEOpSelector.select_op):
        1. SM >= 100 (Blackwell) with fp8_block -> deepgemm (DeepGemm kernel)
        2. Otherwise -> moe_torch_flow (Cutlass kernel)
        """
        sm_version = database.system_spec["gpu"]["sm_version"]
        is_blackwell = sm_version >= 100

        # Convert quant_mode to string for comparison if needed
        quant_mode_str = quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode)
        is_fp8_block = "fp8_block" in quant_mode_str

        # Preferred kernel based on hardware and quant mode
        if is_blackwell and is_fp8_block:
            # Blackwell + FP8 block scales -> DeepGemm kernel
            preferred = "deepgemm"
        else:
            # Default: Cutlass kernel
            preferred = "moe_torch_flow"

        # Check if preferred kernel is available in data, otherwise fallback
        if database._wideep_moe_compute_data:
            available_kernels = list(database._wideep_moe_compute_data.keys())
            if preferred in available_kernels:
                return preferred
            elif available_kernels:
                # Fallback to any available kernel
                fallback = available_kernels[0]
                logger.debug(f"Preferred MoE kernel '{preferred}' not available, falling back to '{fallback}'")
                return fallback

        return preferred

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_wideep_moe_compute)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"

    def _engine_query_plan(self, kwargs: dict):
        """No native twin of its own: route through the unified expert-compute
        op. The engine's moe_expert_compute table absorbed the legacy
        ``wideep_moe_perf.parquet`` rows verbatim (native kernel_source,
        ``num_slots`` pass-through, rows fanned to both phases), so the
        unified twin reproduces this op's values exactly. Supports the legacy
        per-call ``quant_mode`` override. The twin carries no tp axis and the
        retired math divided its SOL by ``moe_tp_size``, so non-default tp is
        rejected loudly rather than silently answered as tp=1."""
        if self._moe_tp_size != 1:
            raise NotImplementedError(
                f"{type(self).__name__}.query shim supports moe_tp_size=1 only (the unified "
                "expert-compute twin carries no tp axis); evaluate a real op list via "
                "EngineHandle.evaluate_ops_json."
            )
        _, eval_kwargs = super()._engine_query_plan(kwargs)
        from aiconfigurator_core.sdk.operations.moe_comm import MoEExpertCompute

        twin = MoEExpertCompute(
            self._name,
            self._scale_factor,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            moe_ep_size=self._moe_ep_size,
            quant_mode=kwargs.get("quant_mode") or self._quant_mode,
            workload_distribution=self._workload_distribution,
            attention_dp_size=self._attention_dp_size,
            inference_phase="context",
            num_slots=self._num_slots,
            is_gated=self._is_gated,
        )
        return twin, eval_kwargs


# ───────────────────────────────────────────────────────────────────────
# TrtLLMWideEPMoEDispatch
# ───────────────────────────────────────────────────────────────────────


class TrtLLMWideEPMoEDispatch(Operation):
    """TensorRT-LLM WideEP MoE dispatch op using NVLink Two-Sided All2All.

    Owns ``_trtllm_alltoall_data`` (loaded only on
    ``database.backend == "trtllm"``). Handles WideEP-specific All2All
    communication for expert parallelism in TRT-LLM (prepare, dispatch,
    combine phases).

    Communication phases:
    - Pre-dispatch: prepare + dispatch operations
    - Post-dispatch: combine or combine_low_precision operation
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        attention_dp_size: int,
        pre_dispatch: bool,
        quant_mode: common.MoEQuantMode,
        use_low_precision_combine: bool = False,
        node_num: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._pre_dispatch = pre_dispatch
        self._quant_mode = quant_mode
        self._use_low_precision_combine = use_low_precision_combine
        self._node_num = node_num
        self.num_gpus = self._moe_ep_size * self._moe_tp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    def get_weights(self, **kwargs):
        """Zero, locally: this deprecated tombstone class has no OpSpec
        variant (PR-7 removes it), so the base engine-routed get_weights
        cannot serve it. Dispatch has no weight memory."""
        return 0.0

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's ``_trtllm_alltoall_data`` table
        view only when ``database.backend == "trtllm"``; binds ``None``
        otherwise.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend == "trtllm":
                cls._data_cache[key] = load_view(database, "_trtllm_alltoall_data", PerfDataFilename.trtllm_alltoall)
            else:
                cls._data_cache[key] = None

            cls._record_load()

        if "_trtllm_alltoall_data" not in database.__dict__:
            database._trtllm_alltoall_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Helpers (formerly PerfDatabase._normalize_alltoall_moe_quant_mode_for_table
    # and ._select_alltoall_kernel)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_quant_mode_for_table(
        quant_mode: common.MoEQuantMode,
    ) -> common.MoEQuantMode:
        """Normalize MoE quant modes for TRT-LLM alltoall perf table lookup.

        ``fp8_block`` is a behavioral mode that reuses the ``fp8`` alltoall tables.
        """
        if quant_mode == common.MoEQuantMode.fp8_block:
            return common.MoEQuantMode.fp8
        return quant_mode

    @classmethod
    def _select_alltoall_kernel(
        cls,
        database: PerfDatabase,
        quant_mode: common.MoEQuantMode,
        moe_ep_size: int,
        topk: int,
        moe_backend: str | None = None,
    ) -> str:
        """Automatically select All2All communication method based on GPU
        architecture, MoE backend type, and configuration.

        Aligned with TensorRT-LLM's per-backend select_alltoall_method_type:

        CutlassFusedMoE / TRTLLMGenFusedMoE:
          - Requires supports_mnnvl() (approximated as SM >= 100)
          - Returns NVLinkOneSided
          - Does NOT support DeepEP / DeepEPLowLatency

        WideEPMoE:
          - If supports_mnnvl() -> NVLinkTwoSided
          - Else if DeepEP feasible -> DeepEP (inter-node) or DeepEPLowLatency (intra-node)
          - Does NOT support NVLinkOneSided

        DeepGemmFusedMoE / CuteDslFusedMoE:
          - Always NotEnabled
        """
        if moe_backend is not None and moe_backend.upper() in {"DEEPGEMM", "CUTE_DSL"}:
            return "NotEnabled"

        sm_version = database.system_spec["gpu"]["sm_version"]
        num_gpus_per_node = database.system_spec["node"]["num_gpus_per_node"]
        is_inter_node = moe_ep_size > num_gpus_per_node
        is_wideep = moe_backend is not None and moe_backend.upper() == "WIDEEP"

        supports_mnnvl = sm_version >= 100

        if is_wideep:
            if supports_mnnvl:
                preferred = "NVLinkTwoSided"
            else:
                deepep_feasible = moe_ep_size > 1 and topk <= 8
                if deepep_feasible and is_inter_node:
                    preferred = "DeepEP"
                elif deepep_feasible:
                    preferred = "DeepEPLowLatency"
                else:
                    preferred = "NotEnabled"
        else:
            if supports_mnnvl:
                preferred = "NVLinkOneSided"
            else:
                preferred = "NotEnabled"

        if preferred == "NotEnabled":
            return preferred

        if database._trtllm_alltoall_data:
            available_kernels = list(database._trtllm_alltoall_data.keys())
            if preferred in available_kernels:
                return preferred
            else:
                logger.warning(
                    f"Preferred All2All kernel '{preferred}' not in available kernels {available_kernels}. "
                    f"Returning preferred anyway; downstream will fall back to HYBRID estimation."
                )

        return preferred

    # ------------------------------------------------------------------
    # The per-phase alltoall raw-table query retired with the per-call stack
    # (#1357 PR-5); the loaded table remains the raw data plane.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract — legacy body lifted verbatim
    # ------------------------------------------------------------------

    def _engine_query_plan(self, kwargs: dict):
        """Tombstone: this legacy op walked the raw per-phase trtllm_alltoall
        table and folded the phases — no single per-op engine expression
        exists (the compiled ``MoEDispatch`` op folds them under its own
        topology gates)."""
        raise NotImplementedError(
            "TrtLLMWideEPMoEDispatch.query was retired with the Python per-call query stack "
            "(#1357 PR-5). The per-phase rows remain loadable "
            "(database._trtllm_alltoall_data); live models express dispatch through the "
            "compiled MoEDispatch op (EngineHandle.evaluate_ops_json)."
        )


# ─────────────────────────────────────────────────────────
# Perf-table loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_wideep_context_moe_data(wideep_context_moe_file):
    """
    Load the SGLang WideEP context MoE data from wideep_context_moe_perf.parquet
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(wideep_context_moe_file)
    if rows is None:
        logger.debug(f"Context MoE data file {wideep_context_moe_file} not found.")
        return None

    wideep_context_moe_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )

    logger.debug(f"Loading SGLang wideep context MoE data from: {wideep_context_moe_file}")
    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_context_moe) - power will default to 0.0")

    for row in rows:
        # Parse the perf-table row schema with num_tokens instead of batch_size and input_len
        quant_mode = row["moe_dtype"]
        num_tokens = int(row["num_tokens"])
        hidden_size = int(row["hidden_size"])
        inter_size = int(row["inter_size"])
        topk = int(row["topk"])
        num_experts = int(row["num_experts"])
        moe_tp_size = int(row["moe_tp_size"])
        moe_ep_size = int(row["moe_ep_size"])
        distribution = row["distribution"]
        latency = float(row["latency"])
        quant_mode = common.MoEQuantMode[quant_mode]

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        try:
            # Check for conflict: first source wins (shared-layer contract).
            wideep_context_moe_data[quant_mode][distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][
                moe_ep_size
            ][num_tokens]
            logger.debug(
                f"value conflict in wideep context moe data: {quant_mode} {distribution} {topk} "
                f"{num_experts} {hidden_size} {inter_size} {moe_tp_size} {moe_ep_size} {num_tokens}"
            )
            continue
        except KeyError:
            pass
        # Store all three values
        wideep_context_moe_data[quant_mode][distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][
            moe_ep_size
        ][num_tokens] = {
            "latency": latency,
            "power": power,
            "energy": energy,  # NEW: precomputed energy
        }
        logger.debug(
            f"Loaded SGLang wideep context MoE data: {quant_mode}, {distribution}, {topk}, "
            f"{num_experts}, {hidden_size}, {inter_size}, {moe_tp_size}, "
            f"{moe_ep_size}, {num_tokens} -> {latency}"
        )

    return wideep_context_moe_data


def load_wideep_generation_moe_data(wideep_generation_moe_file):
    """
    Load the SGLang WideEP generation MoE data from wideep_generation_moe_perf.parquet
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(wideep_generation_moe_file)
    if rows is None:
        logger.debug(f"Generation MoE data file {wideep_generation_moe_file} not found.")
        return None

    wideep_generation_moe_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )

    logger.debug(f"Loading SGLang wideep generation MoE data from: {wideep_generation_moe_file}")
    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_generation_moe) - power will default to 0.0")

    for row in rows:
        # Parse the perf-table row schema with num_tokens instead of batch_size and input_len
        quant_mode = row["moe_dtype"]
        num_tokens = int(row["num_tokens"])
        hidden_size = int(row["hidden_size"])
        inter_size = int(row["inter_size"])
        topk = int(row["topk"])
        num_experts = int(row["num_experts"])
        moe_tp_size = int(row["moe_tp_size"])
        moe_ep_size = int(row["moe_ep_size"])
        distribution = row["distribution"]
        latency = float(row["latency"])
        quant_mode = common.MoEQuantMode[quant_mode]

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        try:
            # Check for conflict: first source wins (shared-layer contract).
            wideep_generation_moe_data[quant_mode][distribution][topk][num_experts][hidden_size][inter_size][
                moe_tp_size
            ][moe_ep_size][num_tokens]
            logger.debug(
                f"value conflict in wideep generation moe data: {quant_mode} {distribution} {topk} "
                f"{num_experts} {hidden_size} {inter_size} {moe_tp_size} {moe_ep_size} {num_tokens}"
            )
            continue
        except KeyError:
            pass
        # Store all three values
        wideep_generation_moe_data[quant_mode][distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][
            moe_ep_size
        ][num_tokens] = {
            "latency": latency,
            "power": power,
            "energy": energy,  # NEW: precomputed energy
        }
        logger.debug(
            f"Loaded SGLang wideep generation MoE data: {quant_mode}, {distribution}, {topk}, "
            f"{num_experts}, {hidden_size}, {inter_size}, {moe_tp_size}, "
            f"{moe_ep_size}, {num_tokens} -> {latency}"
        )

    return wideep_generation_moe_data


def load_wideep_deepep_ll_data(wideep_deepep_ll_file):
    """
    Load the SGLang WideEP DeepEP LL data from wideep_deepep_ll_perf.parquet
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(wideep_deepep_ll_file)
    if rows is None:
        logger.debug(f"SGLang wideep deepep LL operation data file {wideep_deepep_ll_file} not found.")
        return None

    wideep_deepep_ll_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_deepep_ll) - power will default to 0.0")

    for row in rows:
        hidden_size = int(row["hidden_size"])
        node_num = int(row["node_num"])
        num_token = int(row["num_token"])
        num_topk = int(row["num_topk"])
        num_experts = int(row["num_experts"])
        combine_avg_t_us = float(row["combine_avg_t_us"])
        dispatch_avg_t_us = float(row["dispatch_avg_t_us"])
        lat = combine_avg_t_us + dispatch_avg_t_us

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * lat  # watt-milliseconds

        # Store the data with key structure: [hidden_size][num_topk][num_experts][num_token]
        # -> timing data
        if num_token in wideep_deepep_ll_data[node_num][hidden_size][num_topk][num_experts]:
            logger.debug(
                f"value conflict in SGLang wideep deepep LL operation data: "
                f"{hidden_size} {num_topk} {num_experts} {num_token}"
            )
        else:
            # Store all three values
            wideep_deepep_ll_data[node_num][hidden_size][num_topk][num_experts][num_token] = {
                "latency": lat,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_deepep_ll_data


def load_wideep_deepep_normal_data(wideep_deepep_normal_file):
    """
    Load the SGLang WideEP DeepEP normal data from wideep_deepep_normal_perf.parquet
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(wideep_deepep_normal_file)
    if rows is None:
        logger.debug(f"SGLang wideep deepep normal operation data file {wideep_deepep_normal_file} not found.")
        return None

    wideep_deepep_normal_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    )

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_deepep_normal) - power will default to 0.0")

    for row in rows:
        num_token = int(row["num_token"])
        topk = int(row["num_topk"])
        node_num = int(row["node_num"])
        num_experts = int(row["num_experts"])
        hidden_size = int(row["hidden_size"])
        dispatch_sms = int(row["dispatch_sms"])
        dispatch_transmit_us = float(row["dispatch_transmit_us"])
        dispatch_notify_us = float(row["dispatch_notify_us"])
        combine_transmit_us = float(row["combine_transmit_us"])
        combine_notify_us = float(row["combine_notify_us"])
        lat = dispatch_transmit_us + dispatch_notify_us + combine_transmit_us + combine_notify_us

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * lat  # watt-milliseconds

        # Store the data with key structure:
        # [hidden_size][topk][num_experts][dispatch_sms][num_token] -> timing data
        if num_token in wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts][dispatch_sms]:
            logger.debug(
                f"value conflict in deepep normal data: {hidden_size} {topk} {num_experts} {dispatch_sms} {num_token}"
            )
        else:
            # Store all three values
            wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts][dispatch_sms][num_token] = {
                "latency": lat,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_deepep_normal_data


def load_wideep_moe_compute_data(wideep_moe_compute_file):
    """
    Load the TensorRT-LLM WideEP MoE compute data from wideep_moe_perf.parquet.
    This data represents pure computation time (excluding All2All communication).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
        Structure: [kernel_source][quant_mode][distribution][topk][num_experts][hidden_size][inter_size]
                   [num_slots][moe_tp_size][moe_ep_size][num_tokens] -> {latency, power, energy}

    Note:
        kernel_source identifies the MoE computation kernel:
        - "moe_torch_flow": Cutlass-based kernel (default for SM < 100)
        - "deepgemm": DeepGemm kernel (SM >= 100 with fp8_block)
        If data file does not have 'kernel_source' column, it defaults to "moe_torch_flow".
    """
    rows = _read_filtered_rows(wideep_moe_compute_file)
    if rows is None:
        logger.debug(f"TensorRT-LLM wideep MoE compute data file {wideep_moe_compute_file} not found.")
        return None

    wideep_moe_compute_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                            lambda: defaultdict(
                                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                            )
                        )
                    )
                )
            )
        )
    )

    logger.debug(f"Loading TensorRT-LLM wideep MoE compute data from: {wideep_moe_compute_file}")
    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_moe_compute) - power will default to 0.0")

    # Check if kernel_source column exists
    has_kernel_source = len(rows) > 0 and "kernel_source" in rows[0]
    if not has_kernel_source:
        logger.debug("kernel_source column not found (wideep_moe_compute) - will default to 'moe_torch_flow'")

    for row in rows:
        quant_mode = row["moe_dtype"]
        num_tokens = int(row["num_tokens"])
        hidden_size = int(row["hidden_size"])
        inter_size = int(row["inter_size"])
        topk = int(row["topk"])
        num_experts = int(row["num_experts"])
        num_slots = int(row["num_slots"])
        moe_tp_size = int(row["moe_tp_size"])
        moe_ep_size = int(row["moe_ep_size"])
        distribution = row["distribution"]
        latency = float(row["latency"])
        quant_mode = common.MoEQuantMode[quant_mode]

        # Get kernel_source from data or use default
        kernel_source = row.get("kernel_source", "moe_torch_flow")

        # Read power with backward compatibility
        power = float(row.get("power", 0.0))
        energy = power * latency  # watt-milliseconds

        try:
            # Check for conflict: first source wins (shared-layer contract).
            wideep_moe_compute_data[kernel_source][quant_mode][distribution][topk][num_experts][hidden_size][
                inter_size
            ][num_slots][moe_tp_size][moe_ep_size][num_tokens]
            logger.debug(
                f"value conflict in wideep moe compute data: {kernel_source} {quant_mode} {distribution} "
                f"{topk} {num_experts} {hidden_size} {inter_size} {num_slots} {moe_tp_size} {moe_ep_size} {num_tokens}"
            )
            continue
        except KeyError:
            pass
        # Store all three values with kernel_source dimension
        wideep_moe_compute_data[kernel_source][quant_mode][distribution][topk][num_experts][hidden_size][inter_size][
            num_slots
        ][moe_tp_size][moe_ep_size][num_tokens] = {
            "latency": latency,
            "power": power,
            "energy": energy,
        }
        # logger.debug(
        #     f"Loaded TensorRT-LLM wideep MoE compute data: kernel={kernel_source}, {quant_mode}, "
        #     f"{distribution}, {topk}, {num_experts}, {hidden_size}, {inter_size}, {num_slots}, "
        #     f"{moe_tp_size}, {moe_ep_size}, {num_tokens} -> {latency}"
        # )

    return wideep_moe_compute_data


def load_trtllm_alltoall_data(trtllm_alltoall_file):
    """
    Load TensorRT-LLM AlltoAll data from trtllm_alltoall_perf.parquet.
    Covers both WideEP (NVLinkTwoSided) and CutlassFusedMoE (NVLinkOneSided) paths.

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
        Structure: [kernel_source][op_name][quant_mode][num_nodes][hidden_size][topk][num_experts]
                   [moe_ep_size][num_tokens] -> {latency, power, energy}
        op_name can be: alltoall_prepare, alltoall_dispatch, alltoall_combine, alltoall_combine_low_precision

    Note:
        kernel_source identifies the All2All communication method:
        - "NVLinkTwoSided": NVLink Two-Sided via MNNVL (GB200, SM >= 100)
        - "NVLinkOneSided": NVLink One-Sided (CutlassFusedMoE on GB200)
        - "DeepEP": DeepEP normal mode (H100/H200, cross-node)
        - "DeepEPLowLatency": DeepEP low-latency mode (H100/H200, intra-node)
        - "NCCL": Standard NCCL communication (fallback)
        If data file does not have 'kernel_source' column, it defaults to "NVLinkTwoSided".

        If data file does not have 'num_nodes' column, it will be computed as moe_ep_size // 4.
        This assumes 4 GPUs per node (e.g., GB200 NVL4).
    """
    rows = _read_filtered_rows(trtllm_alltoall_file)
    if rows is None:
        logger.debug(f"TensorRT-LLM AlltoAll data file {trtllm_alltoall_file} not found.")
        return None

    trtllm_alltoall_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )

    logger.debug(f"Loading TensorRT-LLM AlltoAll data from: {trtllm_alltoall_file}")
    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (trtllm_alltoall) - power will default to 0.0")

    # Check if num_nodes column exists
    has_num_nodes = len(rows) > 0 and "num_nodes" in rows[0]
    if not has_num_nodes:
        logger.debug("num_nodes column not found (trtllm_alltoall) - will be computed as moe_ep_size // 4")

    # Check if kernel_source column exists
    has_kernel_source = len(rows) > 0 and "kernel_source" in rows[0]
    if not has_kernel_source:
        logger.debug("kernel_source column not found (trtllm_alltoall) - will default to 'NVLinkTwoSided'")

    for row in rows:
        op_name = row["op_name"]  # alltoall_prepare, alltoall_dispatch, alltoall_combine, etc.
        quant_mode = row["moe_dtype"]
        num_tokens = int(row["num_tokens"])
        hidden_size = int(row["hidden_size"])
        topk = int(row["topk"])
        num_experts = int(row["num_experts"])
        moe_ep_size = int(row["moe_ep_size"])
        latency = float(row["latency"])
        quant_mode = common.MoEQuantMode[quant_mode]

        # Get kernel_source from data or use default
        kernel_source = row.get("kernel_source", "NVLinkTwoSided")

        # Get num_nodes from data or compute from moe_ep_size
        if has_num_nodes:
            num_nodes = int(row["num_nodes"])
        else:
            # Default: assume 4 GPUs per node
            if moe_ep_size % 4 != 0:  # FIXME this is only for GB200 needs to be generalized for other systems
                logger.warning(
                    f"moe_ep_size={moe_ep_size} is not divisible by 4, using moe_ep_size // 4 = {moe_ep_size // 4}"
                )
            num_nodes = max(1, moe_ep_size // 4)

        # Read power with backward compatibility
        power = float(row.get("power", 0.0))
        energy = power * latency  # watt-milliseconds

        try:
            # Check for conflict: first source wins (shared-layer contract).
            trtllm_alltoall_data[kernel_source][op_name][quant_mode][num_nodes][hidden_size][topk][num_experts][
                moe_ep_size
            ][num_tokens]
            logger.debug(
                f"value conflict in trtllm alltoall data: {kernel_source} {op_name} {quant_mode} "
                f"{num_nodes} {hidden_size} {topk} {num_experts} {moe_ep_size} {num_tokens}"
            )
            continue
        except KeyError:
            pass
        # Store all three values with kernel_source and num_nodes dimensions
        trtllm_alltoall_data[kernel_source][op_name][quant_mode][num_nodes][hidden_size][topk][num_experts][
            moe_ep_size
        ][num_tokens] = {
            "latency": latency,
            "power": power,
            "energy": energy,
        }
        # logger.debug(
        #     f"Loaded TensorRT-LLM wideep All2All data: kernel={kernel_source}, {op_name}, {quant_mode}, "
        #     f"num_nodes={num_nodes}, {hidden_size}, {topk}, {num_experts}, {moe_ep_size}, "
        #     f"{num_tokens} -> {latency}"
        # )

    return trtllm_alltoall_data
