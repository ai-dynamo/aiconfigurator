# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE family (ISSUE-12 / ISSUE-13).

Op classes migrated from ``_legacy.py``:

- ``MoE`` — Mixture-of-Experts compute op. Owns:
    * ``_moe_data`` — regular MoE table
    * ``_moe_low_latency_data`` — TRT-LLM low-latency NVFP4 kernel table
      (loaded from the same CSV as the regular MoE table; ``load_moe_data``
      is the only loader that returns a tuple of two tables)
    * ``_wideep_context_moe_data`` — SGLang WideEP context MoE table
    * ``_wideep_generation_moe_data`` — SGLang WideEP generation MoE table
  Dispatches to the right table inside ``query_moe`` based on backend +
  ``moe_backend`` + ``num_tokens`` + ``quant_mode`` + ``is_gated``.

- ``MoEDispatch`` — MoE comm-cost op. Owns:
    * ``_wideep_deepep_normal_data`` — SGLang DeepEP normal-mode dispatch
    * ``_wideep_deepep_ll_data`` — SGLang DeepEP low-latency dispatch
  Dispatches at query time across NCCL, CustomAllReduce, TRT-LLM AllToAll,
  and SGLang DeepEP based on backend + ``_sm_version`` + ``_moe_backend``.

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``. The
WideEP tables are loaded only when ``database.backend == "sglang"``; on
other backends the corresponding cache slot is ``None`` and consumers must
guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations._legacy import logger
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family."""
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


# ───────────────────────────────────────────────────────────────────────
# MoE
# ───────────────────────────────────────────────────────────────────────


class MoE(Operation):
    """MoE operation with power tracking."""

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

        from aiconfigurator.sdk.perf_database import (
            LoadedOpData,
            PerfDataFilename,
            load_moe_data,
            load_wideep_context_moe_data,
            load_wideep_generation_moe_data,
        )

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)

            # Regular MoE table — ``load_moe_data`` returns ``(default, low_latency)``
            # because rows tagged ``kernel_source="moe_torch_flow_min_latency"``
            # are routed into a separate accumulator.
            moe_primary = os.path.join(data_dir, PerfDataFilename.moe.value)
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
                ctx_primary = os.path.join(data_dir, PerfDataFilename.wideep_context_moe.value)
                ctx_sources = database._build_op_sources(
                    PerfDataFilename.wideep_context_moe, ctx_primary, system_data_root
                )
                cls._wideep_context_data_cache[key] = LoadedOpData(
                    load_wideep_context_moe_data(ctx_sources),
                    PerfDataFilename.wideep_context_moe,
                    ctx_primary,
                )

                gen_primary = os.path.join(data_dir, PerfDataFilename.wideep_generation_moe.value)
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
        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

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
            latency = get_sol(
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
            scale_factor = 0.4
            return latency / scale_factor

        def _estimate_overflow_with_last_token_util(
            query_tokens: int,
            moe_dict: dict,
            hidden_size: int,
            inter_size: int,
            topk: int,
            num_experts: int,
            moe_tp_size: int,
            moe_ep_size: int,
            quant_mode: common.MoEQuantMode,
            workload_distribution: str,
        ) -> PerformanceResult:
            """Estimate overflow latency using utilization at the largest collected token.
            Call only when query_tokens > max(moe_dict.keys()).
            """
            token_points = sorted(moe_dict.keys())
            last_token = token_points[-1]
            last_point = moe_dict[last_token]
            if isinstance(last_point, dict):
                last_latency = float(last_point["latency"])
                last_power = float(last_point.get("power", 0.0))
                last_energy = float(last_point.get("energy", 0.0))
            else:
                last_latency = float(last_point)
                last_power = 0.0
                last_energy = 0.0

            sol_last = get_sol(
                last_token,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )[0]
            sol_query = get_sol(
                query_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
            )[0]

            util = min(1.0, sol_last / last_latency)  # clamp MFU ≤ 1.0
            util = max(util, 1e-8)  # guard against near-zero sol_last
            est_latency = sol_query / util

            est_energy = 0.0
            if last_power > 0:
                est_energy = last_power * est_latency
            elif last_energy > 0:
                est_energy = last_energy * (est_latency / last_latency)

            # Overflow estimate anchored on the last silicon point's utilization
            # and scaled by SOL ratio. It is still silicon-derived, not a pure
            # formula fallback, so keep the source tag aligned with _interp_pr.
            return database._interp_pr(est_latency, energy=est_energy)

        def _require_moe_token_points(
            moe_dict: dict,
            query_tokens: int,
            used_workload_distribution: str,
        ) -> list[int]:
            token_points = sorted(moe_dict.keys())
            if token_points:
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

                    used_workload_distribution = (
                        workload_distribution if workload_distribution in moe_data[quant_mode] else "uniform"
                    )
                    moe_dict = moe_data[quant_mode][used_workload_distribution][topk][num_experts][hidden_size][
                        inter_size
                    ][moe_tp_size][moe_ep_size]
                    token_points = _require_moe_token_points(
                        moe_dict,
                        num_tokens_corrected,
                        used_workload_distribution,
                    )
                    if num_tokens_corrected > token_points[-1]:
                        return _estimate_overflow_with_last_token_util(
                            num_tokens_corrected,
                            moe_dict,
                            hidden_size,
                            inter_size,
                            topk,
                            num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            quant_mode,
                            workload_distribution,
                        )
                    num_left, num_right = database._nearest_1d_point_helper(
                        num_tokens_corrected,
                        list(moe_dict.keys()),
                        inner_only=False,
                    )
                    result = database._interp_1d(
                        [num_left, num_right],
                        [moe_dict[num_left], moe_dict[num_right]],
                        num_tokens_corrected,
                    )
                    if isinstance(result, dict):
                        lat = result["latency"]
                        energy = result.get("energy", 0.0)
                    else:
                        lat = result
                        energy = 0.0
                    return database._interp_pr(lat, energy=energy)
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
                            used_workload_distribution = (
                                workload_distribution
                                if workload_distribution in database._moe_low_latency_data[quant_mode]
                                else "uniform"
                            )
                            moe_dict = database._moe_low_latency_data[quant_mode][used_workload_distribution][topk][
                                num_experts
                            ][hidden_size][inter_size][moe_tp_size][moe_ep_size]
                            if not moe_dict:
                                # Shape not present in low-latency table (nested defaultdict returned
                                # an empty dict instead of raising KeyError). Fall back to regular data.
                                raise KeyError(
                                    f"No low-latency data for nvfp4 shape "
                                    f"[{hidden_size}, {inter_size}, {topk}, {num_experts}]"
                                )
                            logger.debug(
                                f"Using low-latency kernel for nvfp4 moe "
                                f"{workload_distribution} {topk} {num_experts} {hidden_size} "
                                f"{inter_size} {moe_tp_size} {moe_ep_size}."
                            )
                        except:
                            used_workload_distribution = (
                                workload_distribution
                                if workload_distribution in database._moe_data[quant_mode]
                                else "uniform"
                            )
                            moe_dict = database._moe_data[quant_mode][used_workload_distribution][topk][num_experts][
                                hidden_size
                            ][inter_size][moe_tp_size][moe_ep_size]
                    else:
                        used_workload_distribution = (
                            workload_distribution
                            if workload_distribution in database._moe_data[quant_mode]
                            else "uniform"
                        )
                        moe_dict = database._moe_data[quant_mode][used_workload_distribution][topk][num_experts][
                            hidden_size
                        ][inter_size][moe_tp_size][moe_ep_size]
                    token_points = _require_moe_token_points(moe_dict, num_tokens, used_workload_distribution)
                    if num_tokens > token_points[-1]:
                        return _estimate_overflow_with_last_token_util(
                            num_tokens,
                            moe_dict,
                            hidden_size,
                            inter_size,
                            topk,
                            num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            quant_mode,
                            workload_distribution,
                        )
                    num_left, num_right = database._nearest_1d_point_helper(
                        num_tokens,
                        list(moe_dict.keys()),
                        inner_only=False,
                    )
                    result = database._interp_1d(
                        [num_left, num_right],
                        [moe_dict[num_left], moe_dict[num_right]],
                        num_tokens,
                    )
                    if isinstance(result, dict):
                        lat = result["latency"]
                        energy = result.get("energy", 0.0)
                    else:
                        lat = result
                        energy = 0.0
                    return database._interp_pr(lat, energy=energy)
                elif database.backend == common.BackendName.vllm.value:
                    database._moe_data.raise_if_not_loaded()
                    used_workload_distribution = (
                        workload_distribution if workload_distribution in database._moe_data[quant_mode] else "uniform"
                    )
                    moe_dict = database._moe_data[quant_mode][used_workload_distribution][topk][num_experts][
                        hidden_size
                    ][inter_size][moe_tp_size][moe_ep_size]
                    token_points = _require_moe_token_points(moe_dict, num_tokens, used_workload_distribution)
                    if num_tokens > token_points[-1]:
                        return _estimate_overflow_with_last_token_util(
                            num_tokens,
                            moe_dict,
                            hidden_size,
                            inter_size,
                            topk,
                            num_experts,
                            moe_tp_size,
                            moe_ep_size,
                            quant_mode,
                            workload_distribution,
                        )
                    num_left, num_right = database._nearest_1d_point_helper(
                        num_tokens, list(moe_dict.keys()), inner_only=False
                    )
                    result = database._interp_1d(
                        [num_left, num_right], [moe_dict[num_left], moe_dict[num_right]], num_tokens
                    )
                    if isinstance(result, dict):
                        latency = result["latency"]
                        energy = result.get("energy", 0.0)
                    else:
                        latency = result
                        energy = 0.0
                    return database._interp_pr(latency, energy=energy)
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

        from aiconfigurator.sdk.perf_database import (
            LoadedOpData,
            PerfDataFilename,
            load_wideep_deepep_ll_data,
            load_wideep_deepep_normal_data,
        )

        key = cls._cache_key(database)
        if key not in cls._normal_data_cache:
            if database.backend == "sglang":
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                data_dir = os.path.join(system_data_root, database.backend, database.version)

                normal_primary = os.path.join(data_dir, PerfDataFilename.wideep_deepep_normal.value)
                normal_sources = database._build_op_sources(
                    PerfDataFilename.wideep_deepep_normal, normal_primary, system_data_root
                )
                cls._normal_data_cache[key] = LoadedOpData(
                    load_wideep_deepep_normal_data(normal_sources),
                    PerfDataFilename.wideep_deepep_normal,
                    normal_primary,
                )

                ll_primary = os.path.join(data_dir, PerfDataFilename.wideep_deepep_ll.value)
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
            num_left, num_right = database._nearest_1d_point_helper(num_tokens, list(data.keys()), inner_only=False)
            result = database._interp_1d([num_left, num_right], [data[num_left], data[num_right]], num_tokens)
            lat = result["latency"] if isinstance(result, dict) else result
            energy = result.get("energy", 0.0) if isinstance(result, dict) else 0.0
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
                num_left, num_right = database._nearest_1d_point_helper(num_tokens, list(data.keys()), inner_only=False)
                result = database._interp_1d([num_left, num_right], [data[num_left], data[num_right]], num_tokens)
                lat = result["latency"] if isinstance(result, dict) else result
                energy = result.get("energy", 0.0) if isinstance(result, dict) else 0.0
            else:
                data = database._wideep_deepep_normal_data[node_num][hidden_size][topk][num_experts]
                result = database._interp_2d_linear(sms, num_tokens, data)
                lat = result["latency"] if isinstance(result, dict) else result
                energy = result.get("energy", 0.0) if isinstance(result, dict) else 0.0
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
