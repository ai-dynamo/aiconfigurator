# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE family (ISSUE-12 / ISSUE-13).

Op classes migrated from ``_legacy.py``:

- ``MoE`` (ISSUE-12) — Mixture-of-Experts compute op. Owns:
    * ``_moe_data`` — regular MoE table
    * ``_moe_low_latency_data`` — TRT-LLM low-latency NVFP4 kernel table
      (loaded from the same CSV as the regular MoE table; ``load_moe_data``
      is the only loader that returns a tuple of two tables)
    * ``_wideep_context_moe_data`` — SGLang WideEP context MoE table
    * ``_wideep_generation_moe_data`` — SGLang WideEP generation MoE table
  Dispatches to the right table inside ``query_moe`` based on backend +
  ``moe_backend`` + ``num_tokens`` + ``quant_mode`` + ``is_gated``.

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
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common, perf_interp
from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path
from aiconfigurator_core.sdk.performance_result import PerformanceResult

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
_MOE_QUANT_UTIL_LEVEL: dict[tuple[float, float], float] = {
    (2, 1): 0.53,  # w16a16 / bfloat16              [data]
    (1, 1): 0.45,  # w8a16                          [inferred]
    (0.5, 1): 0.07,  # w4a16 (int4_wo, mxfp4)       [data]
    (1, 2): 0.40,  # w8a8 / fp8(_block)             [data]
    (0.5, 2): 0.15,  # w4a8 (w4afp8, mxfp4_mxfp8)   [data]
    (1, 4): 0.30,  # w8a4                           [inferred]
    (0.5, 4): 0.23,  # w4a4                         [data ≈ nvfp4]
    (0.5625, 4): 0.23,  # w4a4 / nvfp4              [data]
}
_MOE_QUANT_UTIL_DEFAULT = 0.30  # unlisted profile: mid-range relative level


def _moe_quant_util_level(quant_mode) -> float:
    """Achieved-util level e(q) for a MoE quant, by (memory, compute) profile."""
    return _MOE_QUANT_UTIL_LEVEL.get((quant_mode.value.memory, quant_mode.value.compute), _MOE_QUANT_UTIL_DEFAULT)


def _xprofile_moe_quants(query_quant, table) -> list:
    """Collected quants with a DIFFERENT (memory, compute) profile than the query,
    nearest-profile first. Same-profile quants are handled by the same-profile tier;
    these are the cross-profile transfer references, rescaled by the util-level ratio."""
    qp = (query_quant.value.memory, query_quant.value.compute)

    def dist(q):
        return abs(q.value.memory - qp[0]) + abs(q.value.compute - qp[1])

    return sorted(
        (q for q in table if q is not query_quant and (q.value.memory, q.value.compute) != qp),
        key=dist,
    )


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
        # 3 GEMMs for gated (gate, up, down), 2 GEMMs for non-gated (up, down)
        num_gemms = 3 if is_gated else 2
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            * num_gemms
            // self._moe_ep_size
            // self._moe_tp_size
        )

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
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            # Regular MoE table — ``load_moe_data`` returns ``(default, low_latency)``
            # because rows tagged ``kernel_source="moe_torch_flow_min_latency"``
            # are routed into a separate accumulator.
            moe_primary = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.moe.value
            )
            moe_sources = database._build_op_sources(PerfDataFilename.moe, moe_primary, system_data_root)
            moe_result = load_moe_data(moe_sources)
            if isinstance(moe_result, tuple):
                moe_default, moe_low_latency = moe_result
            else:
                moe_default, moe_low_latency = moe_result, None
            cls._data_cache[key] = LoadedOpData(moe_default, PerfDataFilename.moe, moe_primary)
            cls._low_latency_data_cache[key] = LoadedOpData(moe_low_latency, PerfDataFilename.moe, moe_primary)

            # WideEP MoE tables — SGLang-only.
            if database.backend == "sglang":
                ctx_primary = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.wideep_context_moe.value
                )
                ctx_sources = database._build_op_sources(
                    PerfDataFilename.wideep_context_moe, ctx_primary, system_data_root
                )
                cls._wideep_context_data_cache[key] = LoadedOpData(
                    load_wideep_context_moe_data(ctx_sources),
                    PerfDataFilename.wideep_context_moe,
                    ctx_primary,
                )

                gen_primary = resolve_op_data_path(
                    system_data_root,
                    database.backend,
                    database.version,
                    PerfDataFilename.wideep_generation_moe.value,
                )
                gen_sources = database._build_op_sources(
                    PerfDataFilename.wideep_generation_moe, gen_primary, system_data_root
                )
                cls._wideep_generation_data_cache[key] = LoadedOpData(
                    load_wideep_generation_moe_data(gen_sources),
                    PerfDataFilename.wideep_generation_moe,
                    gen_primary,
                )
            else:
                cls._wideep_context_data_cache[key] = None
                cls._wideep_generation_data_cache[key] = None

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
    # Query table (formerly PerfDatabase.query_moe)
    # ------------------------------------------------------------------

    @classmethod
    def _query_moe_table(
        cls,
        database: PerfDatabase,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        is_context: bool = True,
        moe_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
        is_gated: bool = True,
        enable_eplb: bool = False,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_moe`` body."""
        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

        cls.load_data(database)

        num_gemms = 3 if is_gated else 2  # gated (SwiGLU): 3 GEMMs; non-gated (Relu2): 2 GEMMs

        def get_sol(
            num_tokens: int,
            hidden_size: int,
            inter_size: int,
            topk: int,
            num_experts: int,
            moe_tp_size: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            workload_distribution: str,
        ) -> tuple[float, float, float]:
            # we ignore router part. only consider mlp
            # tp already impacted inter_size.
            # only consider even workload.
            total_tokens = num_tokens * topk
            ops = total_tokens * hidden_size * inter_size * num_gemms * 2 // moe_ep_size // moe_tp_size
            mem_bytes = quant_mode.value.memory * (
                total_tokens // moe_ep_size * hidden_size * 2  # input+output
                + total_tokens // moe_ep_size * inter_size * num_gemms // moe_tp_size  # intermediate
                + hidden_size
                * inter_size
                * num_gemms
                // moe_tp_size
                * min(num_experts // moe_ep_size, total_tokens // moe_ep_size)
            )
            sol_math = ops / (database.system_spec["gpu"]["bfloat16_tc_flops"] * quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            num_tokens: int,
            hidden_size: int,
            inter_size: int,
            topk: int,
            num_experts: int,
            moe_tp_size: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            workload_distribution: str,
        ) -> float:
            # SOL / util, util read best-effort from own collected data (the
            # num_tokens curve for this slice); raises EmpiricalNotImplementedError if no data.
            sol_time = get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )[0]

            # Mirror get_silicon's kernel selection: nvfp4 + small tokens + gated
            # uses the low-latency kernel table (~3x faster than the regular one).
            # Building util from the wrong table over-estimates by that factor.
            def _use_low_latency():
                return (
                    num_tokens <= 128
                    and quant_mode == common.MoEQuantMode.nvfp4
                    and is_gated
                    and database._moe_low_latency_data is not None
                    and not (database.backend == common.BackendName.sglang.value and moe_backend == "deepep_moe")
                )

            def _moe_table():
                cls.load_data(database)
                if database.backend == common.BackendName.sglang.value and moe_backend == "deepep_moe":
                    return database._wideep_context_moe_data if is_context else database._wideep_generation_moe_data
                if _use_low_latency():
                    ll = database._moe_low_latency_data
                    try:
                        quant_data = util_empirical.require_data_slice(ll, quant_mode)
                        ll_wl = workload_distribution if workload_distribution in quant_data else "uniform"
                        util_empirical.require_data_slice(
                            quant_data,
                            ll_wl,
                            topk,
                            num_experts,
                            hidden_size,
                            inter_size,
                            moe_tp_size,
                            moe_ep_size,
                        )
                        return ll
                    except PerfDataNotAvailableError:
                        pass
                return database._moe_data

            moe_table = _moe_table()
            # kernel_tag distinguishes the kernel/table the util grid is built from so the
            # process-global grid cache can't serve a wideep grid to a regular query (or
            # vice versa) for an otherwise-identical shape.
            if moe_table is database._moe_low_latency_data:
                kernel_tag = "ll"
            elif moe_table is database._wideep_context_moe_data or moe_table is database._wideep_generation_moe_data:
                kernel_tag = "wideep"
            else:
                kernel_tag = "std"

            def _slice():
                moe_table.raise_if_not_loaded()
                quant_data = util_empirical.require_data_slice(moe_table, quant_mode)
                wl = workload_distribution if workload_distribution in quant_data else "uniform"
                return util_empirical.require_data_slice(
                    quant_data,
                    wl,
                    topk,
                    num_experts,
                    hidden_size,
                    inter_size,
                    moe_tp_size,
                    moe_ep_size,
                )

            grid = util_empirical.grid_for(
                (
                    "moe",
                    database.system,
                    database.backend,
                    database.version,
                    quant_mode.name,
                    kernel_tag,
                    topk,
                    num_experts,
                    hidden_size,
                    inter_size,
                    moe_tp_size,
                    moe_ep_size,
                    workload_distribution,
                    num_gemms,
                ),
                _slice,
                lambda c: get_sol(
                    c[0],
                    hidden_size,
                    inter_size,
                    topk,
                    num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    quant_mode,
                    workload_distribution,
                )[0],
                depth=1,
            )

            util_scale = 1.0
            prov = "empirical"  # own-shape util; tiers below override with their transfer kind
            if grid is None or not grid.samples:
                # Transfer tiers (all borrow a collected MoE config's util curve and
                # reconstruct with SOL). `sol_quant` is the quant whose SOL builds the
                # borrowed util: the QUERY quant for same-profile tiers (coefficients
                # match), the REFERENCE quant for cross-profile (so util = its own
                # SOL/measured = that kernel's true efficiency, then rescaled below).
                def _collect(q, sol_quant, provenance):
                    moe_table.raise_if_not_loaded()
                    wl = workload_distribution if workload_distribution in moe_table[q] else "uniform"
                    wl_data = moe_table[q].get(wl, {})
                    out = []
                    for tk in wl_data:
                        for ne in wl_data[tk]:
                            for hs in wl_data[tk][ne]:
                                for isz in wl_data[tk][ne][hs]:
                                    node = wl_data[tk][ne][hs][isz].get(moe_tp_size, {}).get(moe_ep_size)
                                    if not node:
                                        continue
                                    out.append(
                                        util_empirical.ReferenceCandidate(
                                            features=(tk, ne, hs, isz),
                                            node=node,
                                            sol_fn=(
                                                lambda c, _hs=hs, _isz=isz, _tk=tk, _ne=ne, _sq=sol_quant: get_sol(
                                                    c[0],
                                                    _hs,
                                                    _isz,
                                                    _tk,
                                                    _ne,
                                                    moe_tp_size,
                                                    moe_ep_size,
                                                    _sq,
                                                    workload_distribution,
                                                )[0]
                                            ),
                                            provenance=provenance,
                                        )
                                    )
                    return out

                policy = database.transfer_policy

                def _moe_candidates():
                    # Tier 1 (XSHAPE): cross-shape within the query quant (closest measurement).
                    cands = (
                        _collect(quant_mode, quant_mode, "xshape")
                        if (common.TransferKind.XSHAPE in policy and quant_mode in moe_table)
                        else []
                    )
                    if cands:
                        return cands
                    # Tier 2 (XQUANT): cross-quant within the same (memory, compute) profile.
                    # Same profile => same SOL coefficients and binding regime, so util
                    # transfers (measured ~13% MAPE for fp8_block <- fp8; the query quant's
                    # SOL is used unchanged). Only when the query quant has no data of any shape.
                    if common.TransferKind.XQUANT in policy:
                        qp = (quant_mode.value.memory, quant_mode.value.compute)
                        for q in moe_table:
                            if q is quant_mode or (q.value.memory, q.value.compute) != qp:
                                continue
                            cands.extend(_collect(q, quant_mode, "xquant"))
                    return cands

                grid = util_empirical.grid_from_reference(
                    (
                        "moe_xshape",
                        database.system,
                        database.backend,
                        database.version,
                        quant_mode.name,
                        kernel_tag,
                        topk,
                        num_experts,
                        hidden_size,
                        inter_size,
                        moe_tp_size,
                        moe_ep_size,
                        workload_distribution,
                        num_gemms,
                    ),
                    (topk, num_experts, hidden_size, inter_size),
                    _moe_candidates,
                    depth=1,
                    selection_key=(id(moe_table), policy, workload_distribution, num_gemms),
                )
                if grid is not None and grid.samples and grid.reference_provenance:
                    prov = grid.reference_provenance

                # Tier 3: cross-PROFILE. No own- or same-profile data at all -> borrow the
                # nearest collected quant's util curve, built with the REFERENCE quant's own
                # SOL, and rescale by the per-quant util-LEVEL ratio e(query)/e(ref). The
                # cross-profile error is ~pure systematic kernel-efficiency bias, which this
                # ratio removes (raw ~58% -> ~24% MAPE LOO). Last resort, lowest confidence.
                if (grid is None or not grid.samples) and common.TransferKind.XPROFILE in policy:
                    for ref_q in _xprofile_moe_quants(quant_mode, moe_table):
                        g = util_empirical.grid_from_reference(
                            (
                                "moe_xprofile",
                                database.system,
                                database.backend,
                                database.version,
                                quant_mode.name,
                                ref_q.name,
                                kernel_tag,
                                topk,
                                num_experts,
                                hidden_size,
                                inter_size,
                                moe_tp_size,
                                moe_ep_size,
                                workload_distribution,
                                num_gemms,
                            ),
                            (topk, num_experts, hidden_size, inter_size),
                            (lambda _rq=ref_q: _collect(_rq, _rq, "xprofile")),
                            depth=1,
                            selection_key=(id(moe_table), policy, workload_distribution, num_gemms),
                        )
                        if g is not None and g.samples:
                            grid = g
                            util_scale = _moe_quant_util_level(quant_mode) / _moe_quant_util_level(ref_q)
                            prov = "xprofile"
                            break
            latency, _ = util_empirical.estimate(sol_time, (num_tokens,), grid, util_scale=util_scale, provenance=prov)
            return latency

        def _require_moe_token_points(
            moe_dict: dict,
            query_tokens: int,
            used_workload_distribution: str,
        ) -> list[int]:
            token_points = sorted(moe_dict.keys())
            if token_points:
                # A singleton above the query cannot define the low-token
                # launch-overhead regime. Freezing its measured latency would
                # silently present an unmeasured underflow as silicon (e.g. a
                # decode query at 7 tokens backed only by a 1024-token row).
                # Surface the coverage gap so HYBRID can use the explicitly
                # empirical boundary-util fallback instead. Keep multi-point
                # underflow and singleton overflow behavior unchanged.
                if len(token_points) == 1 and query_tokens < token_points[0]:
                    raise PerfDataNotAvailableError(
                        "MoE silicon token underflow has only one measured point; "
                        "cannot infer low-token latency from a singleton. "
                        f"num_tokens={query_tokens}, measured_token={token_points[0]}, "
                        f"hidden_size={hidden_size}, inter_size={inter_size}, topk={topk}, "
                        f"num_experts={num_experts}, moe_tp_size={moe_tp_size}, "
                        f"moe_ep_size={moe_ep_size}, quant_mode={quant_mode}, "
                        f"workload_distribution='{used_workload_distribution}'."
                    )
                return token_points

            raise PerfDataNotAvailableError(
                "No MoE silicon data points for requested shape. "
                f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                f"num_tokens={query_tokens}, hidden_size={hidden_size}, inter_size={inter_size}, "
                f"topk={topk}, num_experts={num_experts}, moe_tp_size={moe_tp_size}, "
                f"moe_ep_size={moe_ep_size}, quant_mode={quant_mode}, "
                f"workload_distribution='{used_workload_distribution}'."
            )

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")
        else:
            # SILICON or HYBRID mode - use database
            def get_silicon():
                def _resolve_tokens(moe_dict, query_tokens, used_workload_distribution):
                    # Guard first: singleton-underflow must stay a structured miss
                    # (a 1024-token row cannot define the low-token launch floor).
                    _require_moe_token_points(moe_dict, query_tokens, used_workload_distribution)
                    # 1-D tokens curve on the raw table: RAW lerp in range; beyond
                    # the last collected token the engine holds the boundary util
                    # (k_tail=1) and lets the MoE SOL carry the growth — exactly
                    # what the retired hand-rolled overflow estimator did
                    # (util above 1 preserved; energy scaled with latency).
                    config = perf_interp.OpInterpConfig(
                        axes=("num_tokens",),
                        resolver=perf_interp.Grid(),
                        sol_fn=lambda t: get_sol(
                            t,
                            hidden_size,
                            inter_size,
                            topk,
                            num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            quant_mode,
                            workload_distribution,
                        )[0],
                    )
                    result = perf_interp.query(config, moe_dict, query_tokens)
                    lat = perf_interp.get_value(result, "latency")
                    energy = perf_interp.get_value(result, "energy")
                    return database._interp_pr(lat, energy=energy)

                if database.backend == common.BackendName.sglang.value:
                    # deepep_moe is for sglang wideep only
                    # Apply num_tokens correction when eplb is enabled (only during prefill)
                    num_tokens_corrected = int(num_tokens * 0.8) if enable_eplb and is_context else num_tokens
                    if moe_backend == "deepep_moe":
                        if is_context:
                            moe_data = database._wideep_context_moe_data
                        else:
                            moe_data = database._wideep_generation_moe_data
                    else:
                        moe_data = database._moe_data

                    moe_data.raise_if_not_loaded()

                    quant_data = util_empirical.require_data_slice(moe_data, quant_mode)
                    used_workload_distribution = (
                        workload_distribution if workload_distribution in quant_data else "uniform"
                    )
                    moe_dict = util_empirical.require_data_slice(
                        quant_data,
                        used_workload_distribution,
                        topk,
                        num_experts,
                        hidden_size,
                        inter_size,
                        moe_tp_size,
                        moe_ep_size,
                    )
                    return _resolve_tokens(moe_dict, num_tokens_corrected, used_workload_distribution)
                elif database.backend == common.BackendName.trtllm.value:
                    if database._moe_data is None and database._moe_low_latency_data is None:
                        raise PerfDataNotAvailableError(
                            f"MoE perf table is missing for system='{database.system}', "
                            f"backend='{database.backend}', version='{database.version}'. "
                            "Please use HYBRID or EMPIRICAL database mode, or provide the data file."
                        )
                    # aligned with trtllm, kernel source selection.
                    # Low-latency kernel only available for gated MoE (SwiGLU), not for Relu2
                    if (
                        num_tokens <= 128
                        and database._moe_low_latency_data
                        and quant_mode == common.MoEQuantMode.nvfp4
                        and is_gated
                    ):
                        try:
                            quant_data = util_empirical.require_data_slice(database._moe_low_latency_data, quant_mode)
                            used_workload_distribution = (
                                workload_distribution if workload_distribution in quant_data else "uniform"
                            )
                            moe_dict = util_empirical.require_data_slice(
                                quant_data,
                                used_workload_distribution,
                                topk,
                                num_experts,
                                hidden_size,
                                inter_size,
                                moe_tp_size,
                                moe_ep_size,
                            )
                            if not isinstance(moe_dict, Mapping):
                                raise TypeError(
                                    "Malformed low-latency MoE performance data: expected a token mapping, "
                                    f"got {type(moe_dict).__name__}."
                                )
                            logger.debug(
                                f"Using low-latency kernel for nvfp4 moe "
                                f"{workload_distribution} {topk} {num_experts} {hidden_size} "
                                f"{inter_size} {moe_tp_size} {moe_ep_size}."
                            )
                        except PerfDataNotAvailableError:
                            quant_data = util_empirical.require_data_slice(database._moe_data, quant_mode)
                            used_workload_distribution = (
                                workload_distribution if workload_distribution in quant_data else "uniform"
                            )
                            moe_dict = util_empirical.require_data_slice(
                                quant_data,
                                used_workload_distribution,
                                topk,
                                num_experts,
                                hidden_size,
                                inter_size,
                                moe_tp_size,
                                moe_ep_size,
                            )
                    else:
                        quant_data = util_empirical.require_data_slice(database._moe_data, quant_mode)
                        used_workload_distribution = (
                            workload_distribution if workload_distribution in quant_data else "uniform"
                        )
                        moe_dict = util_empirical.require_data_slice(
                            quant_data,
                            used_workload_distribution,
                            topk,
                            num_experts,
                            hidden_size,
                            inter_size,
                            moe_tp_size,
                            moe_ep_size,
                        )
                    return _resolve_tokens(moe_dict, num_tokens, used_workload_distribution)
                elif database.backend == common.BackendName.vllm.value:
                    database._moe_data.raise_if_not_loaded()
                    quant_data = util_empirical.require_data_slice(database._moe_data, quant_mode)
                    used_workload_distribution = (
                        workload_distribution if workload_distribution in quant_data else "uniform"
                    )
                    moe_dict = util_empirical.require_data_slice(
                        quant_data,
                        used_workload_distribution,
                        topk,
                        num_experts,
                        hidden_size,
                        inter_size,
                        moe_tp_size,
                        moe_ep_size,
                    )
                    return _resolve_tokens(moe_dict, num_tokens, used_workload_distribution)
                else:
                    raise NotImplementedError(f"backend {database.backend} not supported for moe")

            return database._query_silicon_or_hybrid(
                get_silicon=get_silicon,
                get_empirical=lambda: get_empirical(
                    num_tokens,
                    hidden_size,
                    inter_size,
                    topk,
                    num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    quant_mode,
                    workload_distribution,
                ),
                database_mode=database_mode,
                error_msg=(
                    f"Failed to query moe data for {num_tokens=}, {hidden_size=}, {inter_size=}, {topk=}, "
                    f"{num_experts=}, {moe_tp_size=}, {moe_ep_size=}, {quant_mode=}, {workload_distribution=}"
                ),
            )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MoE latency with energy data."""
        # attention dp size will scale up the total input tokens.
        x = kwargs.get("x") * self._attention_dp_size
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode

        result = database.query_moe(
            num_tokens=x,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            moe_tp_size=self._moe_tp_size,
            moe_ep_size=self._moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=self._workload_distribution,
            is_context=self._is_context,
            moe_backend=self._moe_backend,
            is_gated=self._is_gated,
            enable_eplb=self._enable_eplb,
        )

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


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
        self._weights = 0.0
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

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads SGLang DeepEP normal + low-latency tables on
        ``backend == "sglang"`` only; binds ``None`` on other backends.
        """
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._normal_data_cache:
            if database.backend == "sglang":
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

                normal_primary = resolve_op_data_path(
                    system_data_root,
                    database.backend,
                    database.version,
                    PerfDataFilename.wideep_deepep_normal.value,
                )
                normal_sources = database._build_op_sources(
                    PerfDataFilename.wideep_deepep_normal, normal_primary, system_data_root
                )
                cls._normal_data_cache[key] = LoadedOpData(
                    load_wideep_deepep_normal_data(normal_sources),
                    PerfDataFilename.wideep_deepep_normal,
                    normal_primary,
                )

                ll_primary = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.wideep_deepep_ll.value
                )
                ll_sources = database._build_op_sources(PerfDataFilename.wideep_deepep_ll, ll_primary, system_data_root)
                cls._ll_data_cache[key] = LoadedOpData(
                    load_wideep_deepep_ll_data(ll_sources),
                    PerfDataFilename.wideep_deepep_ll,
                    ll_primary,
                )
            else:
                cls._normal_data_cache[key] = None
                cls._ll_data_cache[key] = None

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
    # Query tables (formerly PerfDatabase.query_wideep_deepep_normal /
    # query_wideep_deepep_ll)
    # ------------------------------------------------------------------

    @classmethod
    def _query_wideep_deepep_ll_table(
        cls,
        database: PerfDatabase,
        node_num: int,
        num_tokens: int,
        num_experts: int,
        topk: int,
        hidden_size: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_wideep_deepep_ll``."""
        cls.load_data(database)

        def get_sol(num_tokens: int, topk: int, num_experts: int) -> tuple[float, float, float]:
            raise NotImplementedError("WideEP deepep ll operation's sol is not implemented yet")
            return

        def get_empirical(num_tokens: int, topk: int, num_experts: int) -> float:
            raise NotImplementedError("WideEP deepep ll operation's empirical is not implemented yet")
            return

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(num_tokens, topk, num_experts)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(num_tokens, topk, num_experts)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(num_tokens, topk, num_experts), energy=0.0, source="empirical")
        else:
            data = database._wideep_deepep_ll_data[node_num][hidden_size][topk][num_experts]
            # 1-D tokens curve. Dispatch has no implemented roofline, but util-hold
            # only needs the SOL *ratio*: dispatch bytes scale ~linearly with
            # tokens (hidden/topk fixed per slice), so a linear proxy is
            # ratio-equivalent to any bandwidth roofline.
            config = perf_interp.OpInterpConfig(
                axes=("num_tokens",), resolver=perf_interp.Grid(), sol_fn=lambda t: float(t)
            )
            result = perf_interp.query(config, data, num_tokens)
            lat = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(lat / 1000.0, energy=energy / 1000.0)

    @classmethod
    def _query_wideep_deepep_normal_table(
        cls,
        database: PerfDatabase,
        node_num: int,
        num_tokens: int,
        num_experts: int,
        topk: int,
        hidden_size: int,
        sms: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_wideep_deepep_normal``."""
        cls.load_data(database)

        def get_sol(num_tokens: int, num_experts: int, topk: int, hidden_size: int) -> tuple[float, float, float]:
            raise NotImplementedError("WideEP deepep normal operation's sol is not implemented yet")
            return

        def get_empirical(num_tokens: int, num_experts: int, topk: int, hidden_size: int) -> float:
            raise NotImplementedError("WideEP deepep normal operation's empirical is not implemented yet")
            return

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(num_tokens, num_experts, topk, hidden_size)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(num_tokens, num_experts, topk, hidden_size)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(
                get_empirical(num_tokens, num_experts, topk, hidden_size), energy=0.0, source="empirical"
            )
        else:
            if node_num == 1 and sms == 20:  # only collect sm=20 for now
                data = database._wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts][sms]
                # 1-D tokens curve; linear token proxy SOL (see deepep_ll note).
                config = perf_interp.OpInterpConfig(
                    axes=("num_tokens",), resolver=perf_interp.Grid(), sol_fn=lambda t: float(t)
                )
                result = perf_interp.query(config, data, num_tokens)
                lat = perf_interp.get_value(result, "latency")
                energy = perf_interp.get_value(result, "energy")
            else:
                data = database._wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts]
                # 2-axis grid (sms, tokens). Only sm=20 is collected today, so an
                # off-grid sms snaps to the nearest collected value (the legacy
                # 2-D scattered interp simply failed on a single-sms cloud);
                # tokens use the linear proxy SOL (see the DeepEP ll note; SOL is
                # constant in sms — no data supports an sms scaling story yet).
                config = perf_interp.OpInterpConfig(
                    axes=("sms", "num_tokens"),
                    resolver=perf_interp.Grid(),
                    sol_fn=lambda _sm, t: float(t),
                )
                result = perf_interp.query(config, data, sms, num_tokens)
                lat = perf_interp.get_value(result, "latency")
                energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(lat / 1000.0, energy=energy / 1000.0)

    # ------------------------------------------------------------------
    # Op contract — legacy body lifted verbatim. Heavy branching across
    # backends; calls ``database.query_*`` helpers that are already
    # migrated (NCCL, CustomAllReduce, TRT-LLM AllToAll) or live in this
    # same class (DeepEP normal / ll via the database delegations).
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        num_tokens = kwargs.get("x")
        volume = num_tokens * self._hidden_size
        _sm_version = database.system_spec["gpu"].get("sm_version", -1)
        _num_gpus_per_node = database.system_spec["node"]["num_gpus_per_node"]
        _node_num = self.num_gpus / _num_gpus_per_node

        if self._quant_mode is not None:
            _quant_compress = self._quant_mode.value.memory / 2.0
        else:
            _quant_compress = 0.25

        if database.backend == common.BackendName.trtllm.value:
            assert self._attention_tp_size == 1 or self._attention_dp_size == 1, (
                "trtllm does not support TP>1 and DP>1 for attn simultaneously"
            )
            if _sm_version == 100:
                logger.debug("MoEDispatch: In trtllm SM100 execution path")

                _alltoall_backends = {"CUTLASS", "TRTLLM"}
                backend_supports_alltoall = self._moe_backend is None or self._moe_backend.upper() in _alltoall_backends
                is_nvl72 = _num_gpus_per_node >= 72
                enable_alltoall = (
                    backend_supports_alltoall and self._attention_dp_size > 1 and self._moe_tp_size == 1 and is_nvl72
                )

                # Quantize-aware communication volume.
                # When quant_mode is known, compute compressed volume:
                #   nvfp4: volume/4 + scale_factor volume
                #   fp8:   volume/2
                #   others / unknown: full volume (BF16)
                quant_mode = self._quant_mode
                if quant_mode is not None and quant_mode == common.MoEQuantMode.nvfp4:
                    dispatch_x_volume = volume / 4
                    dispatch_sf_volume = volume / 4 / 8
                elif quant_mode is not None and quant_mode in (common.MoEQuantMode.fp8, common.MoEQuantMode.fp8_block):
                    dispatch_x_volume = volume / 2
                    dispatch_sf_volume = 0
                else:
                    dispatch_x_volume = volume
                    dispatch_sf_volume = 0

                if enable_alltoall and quant_mode is None:
                    raise ValueError("MoEDispatch requires quant_mode when TRTLLM alltoall path is enabled.")

                if self._pre_dispatch:
                    if enable_alltoall:
                        dispatch_result = database.query_trtllm_alltoall(
                            op_name="alltoall_dispatch",
                            num_tokens=num_tokens,
                            hidden_size=self._hidden_size,
                            topk=self._topk,
                            num_experts=self._num_experts,
                            moe_ep_size=self._moe_ep_size,
                            quant_mode=quant_mode,
                            moe_backend=self._moe_backend,
                        )
                        comm_latency = float(dispatch_result)
                    elif self._attention_dp_size > 1:
                        all_gather_volume = (dispatch_x_volume + dispatch_sf_volume) * self._attention_dp_size
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half, self.num_gpus, "all_gather", all_gather_volume
                        )
                    elif self._attention_tp_size > 1:
                        if self._reduce_results:
                            if _num_gpus_per_node == 72 and self.num_gpus > 4:
                                comm_latency = database.query_nccl(
                                    common.CommQuantMode.half, self.num_gpus, "all_reduce", volume
                                )
                            else:
                                comm_latency = database.query_custom_allreduce(
                                    common.CommQuantMode.half, self.num_gpus, volume
                                )
                        else:
                            comm_latency = 0
                    else:
                        comm_latency = 0
                else:
                    if enable_alltoall:
                        combine_result = database.query_trtllm_alltoall(
                            op_name="alltoall_combine",
                            num_tokens=num_tokens,
                            hidden_size=self._hidden_size,
                            topk=self._topk,
                            num_experts=self._num_experts,
                            moe_ep_size=self._moe_ep_size,
                            quant_mode=quant_mode,
                            moe_backend=self._moe_backend,
                        )
                        comm_latency = float(combine_result)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                    elif self._attention_tp_size > 1:
                        if self._reduce_results:
                            if _num_gpus_per_node == 72 and self.num_gpus > 4:
                                comm_latency = database.query_nccl(
                                    common.CommQuantMode.half, self.num_gpus, "all_reduce", volume
                                )
                            else:
                                comm_latency = database.query_custom_allreduce(
                                    common.CommQuantMode.half, self.num_gpus, volume
                                )
                        else:
                            comm_latency = 0
                    else:
                        comm_latency = 0
            else:  # sm < 100 or > 100 (for now)
                logger.debug("MoEDispatch: In trtllm SM<100 or >100 execution path")
                if self._pre_dispatch:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "all_gather",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
        elif database.backend == common.BackendName.vllm.value:
            assert self._moe_tp_size == 1 or self._moe_ep_size == 1, (
                "vllm does not support MoE TP and MoE EP at the same time"
            )

            comm_latency = 0

            # Add allreduce latency when TP > 1
            if self._attention_tp_size > 1:
                comm_latency += database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)

            if self._attention_dp_size > 1:
                comm_latency += database.query_nccl(
                    common.CommQuantMode.half,
                    self.num_gpus,
                    "all_gather" if self._pre_dispatch else "reduce_scatter",
                    volume * self._attention_dp_size,
                )
        elif database.backend == common.BackendName.sglang.value:
            if self._moe_backend == "deepep_moe":
                logger.debug("MoEDispatch: In SGLang DeepEP execution path")
                num_tokens = num_tokens // self._scale_num_tokens
                if self._is_context:
                    comm_latency = database.query_wideep_deepep_normal(
                        node_num=_node_num,
                        num_tokens=num_tokens,
                        num_experts=self._num_experts,
                        topk=self._topk,
                        hidden_size=self._hidden_size,
                        sms=self._sms,
                    )
                else:
                    comm_latency = database.query_wideep_deepep_ll(
                        node_num=_node_num,
                        num_tokens=num_tokens,
                        num_experts=self._num_experts,
                        topk=self._topk,
                        hidden_size=self._hidden_size,
                    )
            else:
                logger.debug("MoEDispatch: In SGLang non-DeepEP execution path")
                combined_attention_tpdp = self._attention_tp_size > 1 and self._attention_dp_size > 1
                if self._pre_dispatch:
                    if combined_attention_tpdp:
                        # Matches SGLang DP attention: shard across attention TP, then gather across the full TP world.
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self._attention_tp_size,
                            "reduce_scatter",
                            volume,
                        )
                        comm_latency += database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "all_gather",
                            volume * self._attention_dp_size,
                        )
                    elif self._attn_cp_size > 1:
                        if self._is_context:
                            # prefill: tokens are CP-sharded across ranks -> all_gather
                            # to assemble the full token set for expert routing.
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "all_gather", volume
                            )
                        else:
                            # decode: CP does not run; attention is replicated across the
                            # CP ranks so every rank already holds all tokens -> the
                            # pre-dispatch selection is local, no comm.
                            comm_latency = 0
                    elif self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "all_gather",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
                else:
                    if combined_attention_tpdp:
                        # Reverse path: reduce-scatter across the full TP world, then rebuild each attention TP group.
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                        comm_latency += database.query_nccl(
                            common.CommQuantMode.half,
                            self._attention_tp_size,
                            "all_gather",
                            volume,
                        )
                    elif self._attn_cp_size > 1:
                        if self._is_context:
                            # prefill: scatter results back to the CP-sharded layout.
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "reduce_scatter", volume
                            )
                        else:
                            # decode: each rank computed its owned experts' partial outputs
                            # for all (replicated) tokens; combine into the full per-token
                            # sum every rank needs (next layer re-replicates) -> all_reduce.
                            comm_latency = database.query_custom_allreduce(
                                common.CommQuantMode.half, self.num_gpus, volume
                            )
                    elif self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
        else:  # other backends
            raise NotImplementedError(f"MoEDispatch: Not implemented for backend {database.backend}")

        scaled = comm_latency * self._scale_factor
        return PerformanceResult(
            float(scaled),
            energy=getattr(scaled, "energy", 0.0),
            source=getattr(scaled, "source", "empirical"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

    def query_ideal(self, database: PerfDatabase, **kwargs):
        """
        Ideal communication cost for MoE dispatch. For reference only.
        """
        num_tokens = kwargs.get("x")
        volume = num_tokens * self._hidden_size

        if self._pre_dispatch:
            reduce_scatter1_v = volume / self.num_gpus
            reduce_scatter1_num_gpus = self._attention_tp_size

            all2all1_v = volume * self._topk / self.num_gpus
            all2all1_num_gpus = self.num_gpus

            allgather1_v = volume / self._moe_tp_size
            allgather1_num_gpus = self._moe_tp_size

            comm_latency = (
                database.query_nccl(
                    common.CommQuantMode.half,
                    reduce_scatter1_num_gpus,
                    "reduce_scatter",
                    reduce_scatter1_v,
                )
                + database.query_nccl(common.CommQuantMode.half, all2all1_num_gpus, "alltoall", all2all1_v)
                + database.query_nccl(common.CommQuantMode.half, allgather1_num_gpus, "all_gather", allgather1_v)
            )
        else:
            reduce_scatter2_v = volume
            reduce_scatter2_num_gpus = self._moe_tp_size

            all2all2_v = volume * self._topk / self.num_gpus
            all2all2_num_gpus = self.num_gpus

            allgather2_v = volume / self.num_gpus
            allgather2_num_gpus = self._attention_tp_size

            comm_latency = (
                database.query_nccl(
                    common.CommQuantMode.half,
                    reduce_scatter2_num_gpus,
                    "reduce_scatter",
                    reduce_scatter2_v,
                )
                + database.query_nccl(common.CommQuantMode.half, all2all2_num_gpus, "alltoall", all2all2_v)
                + database.query_nccl(common.CommQuantMode.half, allgather2_num_gpus, "all_gather", allgather2_v)
            )

        return comm_latency * self._scale_factor


# ───────────────────────────────────────────────────────────────────────
# TrtLLMWideEPMoE
# ───────────────────────────────────────────────────────────────────────


class TrtLLMWideEPMoE(Operation):
    """TensorRT-LLM WideEP MoE compute op (excludes All2All — see
    ``TrtLLMWideEPMoEDispatch``).

    Owns ``_wideep_moe_compute_data``, loaded only on
    ``database.backend == "trtllm"``. On other backends the cache slot
    binds to ``None`` and ``_query_compute_table`` raises
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

        # Calculate weights: 3 GEMMs for gated (gate, up, down), 2 GEMMs for non-gated (up, down)
        num_gemms = 3 if is_gated else 2
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            * num_gemms
            // self._moe_ep_size
            // self._moe_tp_size
        )

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads ``_wideep_moe_compute_data`` only when
        ``database.backend == "trtllm"``; binds ``None`` otherwise.
        """
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend == "trtllm":
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                primary = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.wideep_moe_compute.value
                )
                sources = database._build_op_sources(PerfDataFilename.wideep_moe_compute, primary, system_data_root)
                cls._data_cache[key] = LoadedOpData(
                    load_wideep_moe_compute_data(sources),
                    PerfDataFilename.wideep_moe_compute,
                    primary,
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

    @classmethod
    def _query_compute_table(
        cls,
        database: PerfDatabase,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        num_slots: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        database_mode: common.DatabaseMode | None = None,
        is_gated: bool = True,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_wideep_moe_compute``."""
        cls.load_data(database)

        num_gemms = 3 if is_gated else 2

        def get_sol(
            num_tokens: int,
            hidden_size: int,
            inter_size: int,
            topk: int,
            num_experts: int,
            num_slots: int,
            moe_tp_size: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            workload_distribution: str,
        ) -> tuple[float, float, float]:
            """Get the SOL (Speed of Light) time using Roofline model.

            Uses num_slots instead of num_experts for weight memory calculation,
            since WideEP EPLB redundant mode may replicate experts across slots.
            """
            total_tokens = num_tokens * topk
            ops = total_tokens * hidden_size * inter_size * num_gemms * 2 // moe_ep_size // moe_tp_size
            mem_bytes = quant_mode.value.memory * (
                total_tokens // moe_ep_size * hidden_size * 2  # input+output
                + total_tokens // moe_ep_size * inter_size * num_gemms // moe_tp_size  # intermediate activations
                + hidden_size
                * inter_size
                * num_gemms
                // moe_tp_size
                * min(num_slots // moe_ep_size, total_tokens // moe_ep_size)  # weights (use num_slots)
            )
            sol_math = ops / (database.system_spec["gpu"]["bfloat16_tc_flops"] * quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical_from_sol(
            num_tokens: int,
            hidden_size: int,
            inter_size: int,
            topk: int,
            num_experts: int,
            num_slots: int,
            moe_tp_size: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            workload_distribution: str,
        ) -> float:
            """Empirical via SOL / util (util best-effort from own data; raises if no data)."""
            sol_time = get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                num_slots,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )[0]
            kernel_source = cls._select_kernel(database, quant_mode)

            def _slice():
                cls.load_data(database)
                wrapper = database._wideep_moe_compute_data
                wrapper.raise_if_not_loaded()
                kd = util_empirical.require_data_slice(wrapper, kernel_source)
                quant_data = util_empirical.require_data_slice(kd, quant_mode)
                dists = list(quant_data.keys())
                dist = workload_distribution if workload_distribution in dists else (dists[0] if dists else None)
                if dist is None:
                    raise PerfDataNotAvailableError("No WideEP MoE workload distribution is available.")
                return util_empirical.require_data_slice(
                    quant_data,
                    dist,
                    topk,
                    num_experts,
                    hidden_size,
                    inter_size,
                    num_slots,
                    moe_tp_size,
                    moe_ep_size,
                )

            grid = util_empirical.grid_for(
                (
                    "wideep_moe",
                    database.system,
                    database.backend,
                    database.version,
                    kernel_source,
                    quant_mode.name,
                    topk,
                    num_experts,
                    hidden_size,
                    inter_size,
                    num_slots,
                    moe_tp_size,
                    moe_ep_size,
                ),
                _slice,
                lambda c: get_sol(
                    c[0],
                    hidden_size,
                    inter_size,
                    topk,
                    num_experts,
                    num_slots,
                    moe_tp_size,
                    moe_ep_size,
                    quant_mode,
                    workload_distribution,
                )[0],
                depth=1,
            )
            latency, _ = util_empirical.estimate(sol_time, (num_tokens,), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode

        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                num_slots,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                num_slots,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical_from_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                num_slots,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        # Automatically select MoE kernel based on GPU architecture and quant mode
        kernel_source = cls._select_kernel(database, quant_mode)
        logger.debug(f"query_wideep_moe_compute: auto-selected kernel_source='{kernel_source}'")

        # SILICON or HYBRID mode - use database
        def get_silicon():
            database._wideep_moe_compute_data.raise_if_not_loaded()
            # Find the best matching distribution
            kernel_data = util_empirical.require_data_slice(database._wideep_moe_compute_data, kernel_source)
            quant_data = util_empirical.require_data_slice(kernel_data, quant_mode)
            available_distributions = list(quant_data.keys())
            if workload_distribution in available_distributions:
                used_distribution = workload_distribution
            else:
                # Fallback: try to find a similar distribution or use the first available
                used_distribution = available_distributions[0] if available_distributions else None
                if used_distribution is None:
                    raise PerfDataNotAvailableError(
                        f"No distribution available for kernel={kernel_source}, quant_mode={quant_mode}"
                    )
                logger.debug(f"Distribution '{workload_distribution}' not found, using '{used_distribution}' instead")

            moe_dict = util_empirical.require_data_slice(
                quant_data,
                used_distribution,
                topk,
                num_experts,
                hidden_size,
                inter_size,
                num_slots,
                moe_tp_size,
                moe_ep_size,
            )

            # 1-D tokens curve: RAW lerp in range; boundary util-hold via the
            # wideep-MoE roofline beyond it (the SOL's weight-read term keeps
            # small-token holds above the launch/weight floor).
            config = perf_interp.OpInterpConfig(
                axes=("num_tokens",),
                resolver=perf_interp.Grid(),
                sol_fn=lambda t: get_sol(
                    t,
                    hidden_size,
                    inter_size,
                    topk,
                    num_experts,
                    num_slots,
                    moe_tp_size,
                    moe_ep_size,
                    quant_mode,
                    workload_distribution,
                )[0],
            )
            result = perf_interp.query(config, moe_dict, num_tokens)
            lat = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")

            return database._interp_pr(lat, energy=energy)

        def get_empirical() -> float:
            # Data-calibrated util; delegates to get_empirical_from_sol.
            return get_empirical_from_sol(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                num_slots,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg=(
                f"Failed to query wideep moe compute data (kernel={kernel_source}) for "
                f"{num_tokens=}, {hidden_size=}, {inter_size=}, {topk=}, {num_experts=}, "
                f"{num_slots=}, {moe_tp_size=}, {moe_ep_size=}, {quant_mode=}, {workload_distribution=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query TrtLLM WideEP MoE compute latency with energy data."""
        # Scale input tokens by attention_dp_size
        x = kwargs.get("x") * self._attention_dp_size
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode

        logger.debug(f"TrtLLMWideEPMoE: Querying compute with num_slots={self._num_slots}")

        result = database.query_wideep_moe_compute(
            num_tokens=x,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            num_slots=self._num_slots,
            moe_tp_size=self._moe_tp_size,
            moe_ep_size=self._moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=self._workload_distribution,
        )

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        """Get the weight memory size for this MoE layer."""
        return self._weights * self._scale_factor


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
        self._weights = 0.0  # MoEDispatch has no weight memory
        self.num_gpus = self._moe_ep_size * self._moe_tp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads ``_trtllm_alltoall_data`` only when
        ``database.backend == "trtllm"``; binds ``None`` otherwise.
        """
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend == "trtllm":
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                primary = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.trtllm_alltoall.value
                )
                sources = database._build_op_sources(PerfDataFilename.trtllm_alltoall, primary, system_data_root)
                cls._data_cache[key] = LoadedOpData(
                    load_trtllm_alltoall_data(sources),
                    PerfDataFilename.trtllm_alltoall,
                    primary,
                )
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
    # Query table (formerly PerfDatabase.query_trtllm_alltoall)
    # ------------------------------------------------------------------

    @classmethod
    def _query_alltoall_table(
        cls,
        database: PerfDatabase,
        op_name: str,
        num_tokens: int,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        node_num: int | None = None,
        database_mode: common.DatabaseMode | None = None,
        moe_backend: str | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_trtllm_alltoall``."""
        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

        cls.load_data(database)

        def get_sol(
            num_tokens: int,
            hidden_size: int,
            topk: int,
            num_experts: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            node_num: int,
        ) -> tuple[float, float, float]:
            """Get the SOL time for All2All communication.

            All2All transfers token data between GPUs:
            - prepare: lightweight metadata exchange (topk * 4 bytes per token)
            - dispatch: each token sent once per unique remote rank (deduplication).
              remote_ranks = min(topk, num_experts, ep_size - 1), bytes use quant_mode precision.
            - combine: standard returns results in bfloat16 (2 B/elem);
              low-precision variant returns results in fp4 (0.5 B/elem).
              remote_ranks = min(topk, num_experts, ep_size - 1).
            """
            is_inter_node = node_num > 1

            if is_inter_node:
                bw = database.system_spec["node"]["inter_node_bw"]
            else:
                bw = database.system_spec["node"]["intra_node_bw"]

            remote_ranks = min(topk, num_experts, moe_ep_size - 1)

            if op_name == "alltoall_prepare":
                data_bytes = num_tokens * topk * 4  # token routing indices, ~4 bytes per entry
            elif "combine" in op_name:
                bytes_per_element = 0.5 if "low_precision" in op_name else 2
                data_bytes = num_tokens * remote_ranks * hidden_size * bytes_per_element
            else:
                # dispatch: per-rank deduplication, use quant_mode precision
                data_bytes = num_tokens * remote_ranks * hidden_size * quant_mode.value.memory

            sol_comm = data_bytes / bw * 1000  # ms
            sol_time = sol_comm
            return sol_time, sol_comm, 0.0

        def get_empirical_from_sol(
            num_tokens: int,
            hidden_size: int,
            topk: int,
            num_experts: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            node_num: int,
            kernel_source: str,
        ) -> float:
            """Empirical via SOL / util (util best-effort from own data; raises if no data)."""
            sol_time = get_sol(
                num_tokens,
                hidden_size,
                topk,
                num_experts,
                moe_ep_size,
                quant_mode,
                node_num,
            )[0]
            tqm = cls._normalize_quant_mode_for_table(quant_mode)

            def _slice():
                cls.load_data(database)
                wrapper = database._trtllm_alltoall_data
                if not wrapper:
                    raise PerfDataNotAvailableError("No TRT-LLM alltoall data is loaded.")
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(
                    wrapper,
                    kernel_source,
                    op_name,
                    tqm,
                    node_num,
                    hidden_size,
                    topk,
                    num_experts,
                    moe_ep_size,
                )

            grid = util_empirical.grid_for(
                (
                    "alltoall",
                    database.system,
                    database.backend,
                    database.version,
                    kernel_source,
                    op_name,
                    tqm.name,
                    node_num,
                    hidden_size,
                    topk,
                    num_experts,
                    moe_ep_size,
                ),
                _slice,
                lambda c: get_sol(c[0], hidden_size, topk, num_experts, moe_ep_size, quant_mode, node_num)[0],
                depth=1,
            )
            latency, _ = util_empirical.estimate(sol_time, (num_tokens,), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode

        table_quant_mode = cls._normalize_quant_mode_for_table(quant_mode)

        # Compute node_num if not provided
        if node_num is None:
            if moe_ep_size < 4:
                node_num = 1
            else:
                node_num = moe_ep_size // 4
            logger.debug(f"query_trtllm_alltoall: node_num not specified, using {node_num} (moe_ep_size={moe_ep_size})")

        valid_op_names = ["alltoall_prepare", "alltoall_dispatch", "alltoall_combine", "alltoall_combine_low_precision"]
        if op_name not in valid_op_names:
            raise ValueError(f"Invalid op_name '{op_name}'. Must be one of {valid_op_names}")

        kernel_source = cls._select_alltoall_kernel(database, quant_mode, moe_ep_size, topk, moe_backend=moe_backend)
        logger.debug(
            f"query_trtllm_alltoall: auto-selected kernel_source='{kernel_source}' (moe_backend={moe_backend})"
        )

        if kernel_source == "NotEnabled":
            if database_mode == common.DatabaseMode.SOL_FULL:
                return (0.0, 0.0, 0.0)
            source = "sol" if database_mode == common.DatabaseMode.SOL else "empirical"
            return PerformanceResult(0.0, energy=0.0, source=source)

        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(
                num_tokens,
                hidden_size,
                topk,
                num_experts,
                moe_ep_size,
                quant_mode,
                node_num,
            )[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(
                num_tokens,
                hidden_size,
                topk,
                num_experts,
                moe_ep_size,
                quant_mode,
                node_num,
            )
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical_from_sol(
                num_tokens,
                hidden_size,
                topk,
                num_experts,
                moe_ep_size,
                quant_mode,
                node_num,
                kernel_source,
            )
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        # SILICON or HYBRID mode - use database
        def get_silicon():
            if not getattr(database, "_trtllm_alltoall_data", None):
                raise PerfDataNotAvailableError(
                    f"TRT-LLM alltoall perf data not available for version '{database.version}'. "
                    "Use HYBRID or EMPIRICAL database mode."
                )
            database._trtllm_alltoall_data.raise_if_not_loaded()
            alltoall_dict = util_empirical.require_data_slice(
                database._trtllm_alltoall_data,
                kernel_source,
                op_name,
                table_quant_mode,
                node_num,
                hidden_size,
                topk,
                num_experts,
                moe_ep_size,
            )

            # 1-D tokens curve: RAW lerp in range; boundary util-hold via the
            # alltoall communication SOL beyond it.
            config = perf_interp.OpInterpConfig(
                axes=("num_tokens",),
                resolver=perf_interp.Grid(),
                sol_fn=lambda t: get_sol(t, hidden_size, topk, num_experts, moe_ep_size, quant_mode, node_num)[0],
            )
            result = perf_interp.query(config, alltoall_dict, num_tokens)
            lat = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")

            return database._interp_pr(lat, energy=energy)

        def get_empirical() -> float:
            return get_empirical_from_sol(
                num_tokens,
                hidden_size,
                topk,
                num_experts,
                moe_ep_size,
                quant_mode,
                node_num,
                kernel_source,
            )

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg=(
                f"Failed to query trtllm alltoall data for {op_name} (kernel={kernel_source}), "
                f"moe_backend={moe_backend}, node_num={node_num}, {num_tokens=}, {hidden_size=}, "
                f"{topk=}, {num_experts=}, {moe_ep_size=}, {quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract — legacy body lifted verbatim
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query TrtLLM WideEP All2All communication latency."""
        num_tokens = kwargs.get("x")

        phase = "Pre-dispatch" if self._pre_dispatch else "Post-dispatch"
        precision = (
            "low-precision combine"
            if self._use_low_precision_combine and not self._pre_dispatch
            else "standard precision"
        )
        logger.debug(f"TrtLLMWideEPMoEDispatch: {phase} with {precision}")

        def _as_performance_result(result) -> PerformanceResult:
            if isinstance(result, PerformanceResult):
                return result

            energy = getattr(result, "energy", 0.0)
            if not isinstance(energy, int | float):
                energy = 0.0

            source = getattr(result, "source", "silicon")
            if not isinstance(source, str):
                source = "silicon"

            return PerformanceResult(float(result), energy=energy, source=source)

        if self._pre_dispatch:
            prepare_result = database.query_trtllm_alltoall(
                op_name="alltoall_prepare",
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                moe_backend="wideep",
                node_num=self._node_num,
            )
            dispatch_result = database.query_trtllm_alltoall(
                op_name="alltoall_dispatch",
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                moe_backend="wideep",
                node_num=self._node_num,
            )
            comm_latency = _as_performance_result(prepare_result) + _as_performance_result(dispatch_result)
        else:
            combine_op = "alltoall_combine_low_precision" if self._use_low_precision_combine else "alltoall_combine"
            combine_result = database.query_trtllm_alltoall(
                op_name=combine_op,
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                moe_backend="wideep",
                node_num=self._node_num,
            )
            comm_latency = _as_performance_result(combine_result)

        scaled = comm_latency * self._scale_factor
        return PerformanceResult(
            float(scaled),
            energy=getattr(scaled, "energy", 0.0),
            source=getattr(scaled, "source", "empirical"),
        )

    def get_weights(self, **kwargs):
        """MoE dispatch has no weight memory."""
        return 0.0


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_moe_data(moe_file):
    """
    Load the moe data with power support (backward compatible).

    Returns:
        tuple: (moe_default_data, moe_low_latency_data) where leaf values are dicts
               with 'latency', 'power', and 'energy' keys. For old formats,
               power/energy default to 0.0. Both elements are `None` when the file
               is missing.
    """
    rows = _read_filtered_rows(moe_file)
    if rows is None:
        logger.debug(f"MOE data file {moe_file} not found.")
        return None, None

    moe_default_data = defaultdict(
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
    moe_low_latency_data = defaultdict(
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

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (moe) - power will default to 0.0")

    for row in rows:
        (
            quant_mode,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            workload_distribution,
            latency,
        ) = (
            row["moe_dtype"],
            row["num_tokens"],
            row["hidden_size"],
            row["inter_size"],
            row["topk"],
            row["num_experts"],
            row["moe_tp_size"],
            row["moe_ep_size"],
            row["distribution"],
            row["latency"],
        )
        kernel_source = row["kernel_source"]  # moe_torch_flow, moe_torch_flow_min_latency, moe_torch_flow
        num_tokens = int(num_tokens)
        hidden_size = int(hidden_size)
        inter_size = int(inter_size)
        topk = int(topk)
        num_experts = int(num_experts)
        moe_tp_size = int(moe_tp_size)
        moe_ep_size = int(moe_ep_size)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.MoEQuantMode[quant_mode]

        # DeepSeek-V4-Pro's Blackwell MoE runs the trtllm-gen MXFP4xMXFP8 kernel
        # (moe_runner_backend=flashinfer_mxfp4 -> Mxfp4FlashinferTrtllmMoEMethod ->
        # trtllm_fp4_block_scale_routed_moe -> bmm_MxE4m3_MxE2m1MxE4m3 ... sm100f),
        # which is a distinct precision from the flashinfer cutedsl kernel that the
        # collector also logs under moe_dtype=w4a8_mxfp4_mxfp8. Route those rows to
        # the dedicated quant mode so DeepSeek-V4 modeling can select it on Blackwell.
        if quant_mode is common.MoEQuantMode.w4a8_mxfp4_mxfp8 and kernel_source == "sglang_mxfp4_flashinfer_trtllm_moe":
            quant_mode = common.MoEQuantMode.w4a8_mxfp4_mxfp8_trtllm

        # Same idea on Hopper: DeepSeek-V4-Pro runs the flashinfer cutlass SM90
        # mixed-GEMM (cutlass_fused_moe(use_w4_group_scaling=True), MXFP4 weight x
        # BF16 act), a distinct backend from GPT-OSS's triton_kernels mxfp4 that the
        # collector also logs under moe_dtype=w4a16_mxfp4. Route those rows to the
        # dedicated quant mode so DeepSeek-V4 modeling can select it on Hopper.
        if quant_mode is common.MoEQuantMode.w4a16_mxfp4 and kernel_source == "sglang_flashinfer_cutlass_moe":
            quant_mode = common.MoEQuantMode.w4a16_mxfp4_cutlass

        moe_data = moe_low_latency_data if kernel_source == "moe_torch_flow_min_latency" else moe_default_data

        try:
            # Check for conflict
            moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][
                moe_ep_size
            ][num_tokens]
            logger.debug(
                f"value conflict in moe data: {workload_distribution} {quant_mode} {topk} "
                f"{num_experts} {hidden_size} {inter_size} {moe_tp_size} {moe_ep_size} "
                f"{num_tokens}"
            )
        except KeyError:
            # Store all three values
            moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][
                moe_ep_size
            ][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return moe_default_data, moe_low_latency_data


def load_wideep_context_moe_data(wideep_context_moe_file):
    """
    Load the SGLang wideep context MoE data from wideep_context_moe_perf.txt
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
        # Parse the CSV format with num_tokens instead of batch_size and input_len
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
    Load the SGLang wideep generation MoE data from wideep_generation_moe_perf.txt
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
        # Parse the CSV format with num_tokens instead of batch_size and input_len
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
    Load the SGLang wideep deepep LL operation data from wideep_deepep_ll_perf.txt
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
    Load the SGLang wideep deepep normal operation data from wideep_deepep_normal_perf.txt
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
    Load the TensorRT-LLM wideep MoE compute data from wideep_moe_compute_perf.txt.
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
    Load TensorRT-LLM AlltoAll communication perf data from trtllm_alltoall_perf.txt.
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
