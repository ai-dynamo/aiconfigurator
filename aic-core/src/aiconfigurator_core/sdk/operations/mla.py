# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLA (Multi-head Latent Attention) family (ISSUE-08 / AIC-540).

Six op classes migrate from ``_legacy.py`` into ``operations/mla.py``:

- ``ContextMLA`` / ``GenerationMLA`` — regular MLA ops; own
  ``_context_mla_data`` / ``_generation_mla_data`` respectively. Both
  delegate to ``PerfDatabase.query_context_mla`` / ``query_generation_mla``
  which become one-line forwards.
- ``MLABmm`` — pre/post BMM op for MLA decoding. Owns ``_mla_bmm_data``.
- ``MLAModule`` — module-level MLA (both context and generation in one
  class, dispatched by ``is_context`` flag). Owns BOTH
  ``_context_mla_module_data`` AND ``_generation_mla_module_data`` since
  ``MLAModule.query`` chooses between them at runtime.
- ``WideEPContextMLA`` / ``WideEPGenerationMLA`` — SGLang-only variants.
  Their CSV tables are loaded only when ``backend == "sglang"`` (matching
  the legacy conditional ``if backend == "sglang"`` block in
  ``PerfDatabase.__init__``).

No SOL clamping for any MLA variant in the legacy ``_correct_data``.
Extrapolation present for all 4 regular + 2 module variants + 2 WideEP
variants (the WideEP variants extrapolate only when their data was
loaded — SGLang-only).

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``. For
WideEP variants, ``backend`` in the key naturally encodes the SGLang
constraint (cache misses on non-SGLang backends).
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
    """Shared cache key — same shape as every other migrated op family.

    TODO: hoist to ``operations/base.py`` once Phase 3 settles (7 op
    families duplicating this helper now).
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


# fmt: on


class ContextMLA(Operation):
    """
    Context MLA operation. Owns ``_context_mla_data``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        # Context parallelism (sglang AllGather zigzag in-seq split). When cp>1,
        # query() models CP rank 0's two zigzag chunks (prev: prefix..+c; next:
        # prefix+isl-c..isl), same as ContextAttention. c = ceil(isl/(2*cp)).
        self._cp_size = cp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads context_mla CSV, applies extrapolation, binds
        ``database._context_mla_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.context_mla.value
            )
            sources = database._build_op_sources(PerfDataFilename.context_mla, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_context_mla_data(sources), PerfDataFilename.context_mla, primary_path
            )
            # No load-time grid pre-expansion: queries resolve on the RAW grid via perf_interp.
            cls._record_load()

        if "_context_mla_data" not in database.__dict__:
            database._context_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_mla)
    # ------------------------------------------------------------------

    @classmethod
    def _query_context_mla_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        prefix: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query context MLA table. Verbatim port of the legacy body."""

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            full_s = s + prefix
            ops = (
                b * num_heads * 2 / 2 * (192 + 128) * (full_s * full_s - prefix * prefix)
            )  # 2 for fma, 2 for causality. num_heads, for local heads
            mem_bytes = (
                b * num_heads * (kvcache_quant_mode.value.memory * full_s * (192 + 128) + 2 * s * (192 + 128))
            )  # 2 for qk, TODO
            sol_math = ops / database.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            # SOL / util from own (num_heads, full_s, b) grid; raises if no data.
            sol_time = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]

            def _slice():
                cls.load_data(database)
                wrapper = database._context_mla_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(wrapper, fmha_quant_mode, kvcache_quant_mode)

            grid = util_empirical.grid_for(
                (
                    "ctx_mla",
                    database.system,
                    database.backend,
                    database.version,
                    fmha_quant_mode.name,
                    kvcache_quant_mode.name,
                ),
                _slice,
                lambda c: get_sol(c[2], c[1], 0, c[0], kvcache_quant_mode, fmha_quant_mode)[
                    0
                ],  # c=(num_heads, full_s, b)
                depth=3,
            )
            latency, _ = util_empirical.estimate(sol_time, (num_heads, s + prefix, b), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._context_mla_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            full_s = s + prefix
            prefix_correction = (full_s * full_s - prefix * prefix) / (full_s * full_s)
            mla_dict = util_empirical.require_data_slice(data_wrapper, fmha_quant_mode, kvcache_quant_mode)
            # Context MLA ~ seq^2 -> context grid (sqrt on seq only); samples are prefix=0.
            config = perf_interp.context_grid_config(
                sol_fn=lambda n_v, s_v, b_v: get_sol(b_v, s_v, 0, n_v, kvcache_quant_mode, fmha_quant_mode)[0]
            )
            result = perf_interp.query(config, mla_dict, num_heads, full_s, b)
            latency = perf_interp.get_value(result, "latency") * prefix_correction
            energy = perf_interp.get_value(result, "energy") * prefix_correction
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query context mla data for {b=}, {s=}, {prefix=}, {num_heads=}, "
                f"{kvcache_quant_mode=}, {fmha_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context MLA latency with energy data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        def _q(s, pfx):
            return database.query_context_mla(
                b=batch_size,
                s=s,
                prefix=pfx,
                num_heads=self._num_heads,
                kvcache_quant_mode=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
            )

        if self._cp_size and self._cp_size > 1:
            cp = self._cp_size
            c = max(1, -(-isl // (2 * cp)))  # ceil(isl/(2*cp)) — rank-0 zigzag halves
            result = _q(c, prefix) + _q(c, prefix + isl - c)
        else:
            result = _q(isl, prefix)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationMLA(Operation):
    """
    Generation MLA operation (MQA part). Owns ``_generation_mla_data``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0.0
        self._kv_cache_dtype = kv_cache_dtype

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads generation_mla CSV, applies extrapolation, binds
        ``database._generation_mla_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.generation_mla.value
            )
            sources = database._build_op_sources(PerfDataFilename.generation_mla, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_generation_mla_data(sources), PerfDataFilename.generation_mla, primary_path
            )
            # No load-time grid pre-expansion: queries resolve on the RAW grid via perf_interp.
            cls._record_load()

        if "_generation_mla_data" not in database.__dict__:
            database._generation_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_mla)
    # ------------------------------------------------------------------

    @classmethod
    def _query_generation_mla_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query generation MLA table. Verbatim port of the legacy body."""

        def get_sol(
            b: int, s: int, num_heads: int, kvcache_quant_mode: common.KVCacheQuantMode
        ) -> tuple[float, float, float]:
            if kvcache_quant_mode == common.KVCacheQuantMode.fp8:
                quant_mode_gen = common.FMHAQuantMode.fp8
            else:
                quant_mode_gen = common.FMHAQuantMode.bfloat16
            ops = 2 * b * num_heads * 1088 * s
            mem_bytes = b * (num_heads * 1088 * 2 + (s - 1) * 576 * kvcache_quant_mode.value.memory)
            sol_math = ops / database.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / quant_mode_gen.value.compute
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> float:
            # SOL / util from own (num_heads, b, s) grid; raises if no data.
            sol_time = get_sol(b, s, num_heads, kvcache_quant_mode)[0]

            def _slice():
                cls.load_data(database)
                wrapper = database._generation_mla_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(wrapper, kvcache_quant_mode)

            grid = util_empirical.grid_for(
                ("gen_mla", database.system, database.backend, database.version, kvcache_quant_mode.name),
                _slice,
                lambda c: get_sol(c[1], c[2], c[0], kvcache_quant_mode)[0],  # c=(num_heads, b, s)
                depth=3,
            )
            latency, _ = util_empirical.estimate(sol_time, (num_heads, b, s), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, num_heads, kvcache_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, num_heads, kvcache_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, num_heads, kvcache_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._generation_mla_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            mla_dict = util_empirical.require_data_slice(data_wrapper, kvcache_quant_mode)
            # Generation MLA ~ linear in seq -> raw generation grid.
            config = perf_interp.generation_grid_config(
                sol_fn=lambda n_v, b_v, s_v: get_sol(b_v, s_v, n_v, kvcache_quant_mode)[0]
            )
            result = perf_interp.query(config, mla_dict, num_heads, b, s)
            latency = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, num_heads, kvcache_quant_mode),
            database_mode=database_mode,
            error_msg=f"Failed to query generation mla data for {b=}, {s=}, {num_heads=}, {kvcache_quant_mode=}",
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation MLA latency with energy data."""
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_mla(batch_size, s, self._num_heads, self._kv_cache_dtype)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class MLABmm(Operation):
    """
    MLABmm operation — pre/post BMM for MLA decoding. Owns ``_mla_bmm_data``.
    No extrapolation in the legacy ``__init__`` path; data is 1D-keyed by
    num_tokens within each (quant_mode, op_name, num_heads) bucket.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0.0
        self._quant_mode = quant_mode
        self._if_pre = if_pre

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads mla_bmm CSV, binds ``database._mla_bmm_data``.
        No extrapolation (1D table)."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.mla_bmm.value
            )
            sources = database._build_op_sources(PerfDataFilename.mla_bmm, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(load_mla_bmm_data(sources), PerfDataFilename.mla_bmm, primary_path)
            cls._record_load()

        if "_mla_bmm_data" not in database.__dict__:
            database._mla_bmm_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_mla_bmm)
    # ------------------------------------------------------------------

    @classmethod
    def _query_mla_bmm_table(
        cls,
        database: PerfDatabase,
        num_tokens: int,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query MLA BMM table. Verbatim port of the legacy body."""

        def get_sol(
            num_tokens: int, num_heads: int, quant_mode: common.GEMMQuantMode, if_pre: bool
        ) -> tuple[float, float, float]:
            ops = 2 * num_tokens * num_heads * 128 * 512
            mem_bytes = num_heads * (num_tokens * 640 + 128 * 512) * quant_mode.value.memory
            sol_math = ops / (database.system_spec["gpu"]["bfloat16_tc_flops"] * quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            num_tokens: int,
            num_heads: int,
            quant_mode: common.GEMMQuantMode,
            if_pre: bool,
        ) -> float:
            # SOL / util from own num_tokens curve; raises if no data.
            sol_time = get_sol(num_tokens, num_heads, quant_mode, if_pre)[0]
            op_name = "mla_gen_pre" if if_pre else "mla_gen_post"

            def _slice():
                cls.load_data(database)
                wrapper = database._mla_bmm_data
                wrapper.raise_if_not_loaded()
                qm = quant_mode if quant_mode in wrapper else common.GEMMQuantMode.bfloat16
                return util_empirical.require_data_slice(wrapper, qm, op_name, num_heads)

            grid = util_empirical.grid_for(
                ("mla_bmm", database.system, database.backend, database.version, quant_mode.name, op_name, num_heads),
                _slice,
                lambda c: get_sol(c[0], num_heads, quant_mode, if_pre)[0],  # c=(num_tokens,)
                depth=1,
            )
            latency, _ = util_empirical.estimate(sol_time, (num_tokens,), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(num_tokens, num_heads, quant_mode, if_pre)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(num_tokens, num_heads, quant_mode, if_pre)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(num_tokens, num_heads, quant_mode, if_pre)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._mla_bmm_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            quant_mode_lookup = quant_mode if quant_mode in data_wrapper else common.GEMMQuantMode.bfloat16
            mla_bmm_dict = util_empirical.require_data_slice(
                data_wrapper,
                quant_mode_lookup,
                "mla_gen_pre" if if_pre else "mla_gen_post",
                num_heads,
            )
            # 1-D tokens curve on the raw table: RAW lerp in range (BMM is
            # ~linear in tokens); boundary util-hold beyond it via the BMM SOL
            # (replaces the legacy raw two-point extrapolation).
            config = perf_interp.OpInterpConfig(
                axes=("num_tokens",),
                resolver=perf_interp.Grid(),
                sol_fn=lambda t: get_sol(t, num_heads, quant_mode, if_pre)[0],
            )
            result = perf_interp.query(config, mla_bmm_dict, num_tokens)
            lat = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(lat, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(num_tokens, num_heads, quant_mode, if_pre),
            database_mode=database_mode,
            error_msg=f"Failed to query mla bmm data for {num_tokens=}, {num_heads=}, {quant_mode=}, {if_pre=}",
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MLA BMM latency with power data."""
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")

        result = database.query_mla_bmm(batch_size, self._num_heads, self._quant_mode, self._if_pre)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class MLAModule(Operation):
    """
    Module-level MLA op for both context and generation phases.

    Owns BOTH ``_context_mla_module_data`` (via ``_context_data_cache``)
    AND ``_generation_mla_module_data`` (via ``_generation_data_cache``)
    because ``query()`` chooses between them at runtime based on the
    ``is_context`` flag.

    Models the complete MLA attention block as a single profiled operation.
    For context: replaces q_b_proj + kv_b_proj + ContextMLA + proj.
    For generation: replaces MLABmm(pre) + GenerationMLA + MLABmm(post).
    """

    _context_data_cache: ClassVar[dict] = {}
    _generation_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        is_context: bool,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._is_context = is_context
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership — two tables, one per phase
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads BOTH context and generation module CSVs,
        applies extrapolation to each, binds
        ``database._context_mla_module_data`` and
        ``database._generation_mla_module_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._context_data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            context_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.mla_context_module.value
            )
            context_sources = database._build_op_sources(
                PerfDataFilename.mla_context_module, context_path, system_data_root
            )
            cls._context_data_cache[key] = LoadedOpData(
                load_context_mla_module_data(context_sources), PerfDataFilename.mla_context_module, context_path
            )

            gen_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.mla_generation_module.value
            )
            gen_sources = database._build_op_sources(PerfDataFilename.mla_generation_module, gen_path, system_data_root)
            cls._generation_data_cache[key] = LoadedOpData(
                load_generation_mla_module_data(gen_sources), PerfDataFilename.mla_generation_module, gen_path
            )

            # No load-time grid pre-expansion: queries resolve on the RAW grid via perf_interp.
            cls._record_load()

        if "_context_mla_module_data" not in database.__dict__:
            database._context_mla_module_data = cls._context_data_cache[key]
        if "_generation_mla_module_data" not in database.__dict__:
            database._generation_mla_module_data = cls._generation_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._context_data_cache.clear()
        cls._generation_data_cache.clear()

    # ------------------------------------------------------------------
    # Query tables (formerly PerfDatabase.query_context_mla_module /
    # query_generation_mla_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_context_mla_module_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        prefix: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query context MLA module table. Verbatim port of the legacy body."""

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            # Reuse the same SOL model as query_context_mla
            full_s = s + prefix
            ops = b * num_heads * 2 / 2 * (192 + 128) * (full_s * full_s - prefix * prefix)
            mem_bytes = b * num_heads * (kvcache_quant_mode.value.memory * full_s * (192 + 128) + 2 * s * (192 + 128))
            sol_math = ops / database.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            # SOL / util from own (num_heads, full_s, b) grid; raises if no data.
            sol_time = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]

            def _slice():
                cls.load_data(database)
                wrapper = database._context_mla_module_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(
                    wrapper,
                    fmha_quant_mode,
                    kvcache_quant_mode,
                    gemm_quant_mode,
                )

            grid = util_empirical.grid_for(
                (
                    "ctx_mla_mod",
                    database.system,
                    database.backend,
                    database.version,
                    fmha_quant_mode.name,
                    kvcache_quant_mode.name,
                    gemm_quant_mode.name,
                ),
                _slice,
                lambda c: get_sol(c[2], c[1], 0, c[0], kvcache_quant_mode, fmha_quant_mode)[
                    0
                ],  # c=(num_heads, full_s, b)
                depth=3,
            )
            latency, _ = util_empirical.estimate(sol_time, (num_heads, s + prefix, b), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._context_mla_module_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            full_s = s + prefix
            prefix_correction = (full_s * full_s - prefix * prefix) / (full_s * full_s)
            mla_dict = util_empirical.require_data_slice(
                data_wrapper,
                fmha_quant_mode,
                kvcache_quant_mode,
                gemm_quant_mode,
            )
            # Context MLA module ~ seq^2 -> context grid (sqrt on seq only); samples are prefix=0.
            config = perf_interp.context_grid_config(
                sol_fn=lambda n_v, s_v, b_v: get_sol(b_v, s_v, 0, n_v, kvcache_quant_mode, fmha_quant_mode)[0]
            )
            result = perf_interp.query(config, mla_dict, num_heads, full_s, b)
            latency = perf_interp.get_value(result, "latency") * prefix_correction
            energy = perf_interp.get_value(result, "energy") * prefix_correction
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query context MLA module data for {b=}, {s=}, {prefix=}, "
                f"{num_heads=}, {kvcache_quant_mode=}, {fmha_quant_mode=}, {gemm_quant_mode=}"
            ),
        )

    @classmethod
    def _query_generation_mla_module_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query generation MLA module table. Verbatim port of the legacy body."""

        # Reuse the same SOL model as query_generation_mla — the module captures
        # the same operations, just profiled together. For a proper SOL we'd
        # also include BMM pre/post, but that's a refinement for later; the
        # primary purpose here is SILICON mode with real data.
        def get_sol(
            b: int, s: int, num_heads: int, kv_cache_dtype: common.KVCacheQuantMode
        ) -> tuple[float, float, float]:
            if kv_cache_dtype == common.KVCacheQuantMode.fp8:
                quant_mode_gen = common.FMHAQuantMode.fp8
            else:
                quant_mode_gen = common.FMHAQuantMode.bfloat16
            # MLA attention ops
            attn_ops = 2 * b * num_heads * 1088 * s
            mem_bytes = b * (num_heads * 1088 * 2 + (s - 1) * 576 * kv_cache_dtype.value.memory)
            sol_math = attn_ops / database.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / quant_mode_gen.value.compute
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            # Add BMM pre + post SOL (same as query_mla_bmm)
            bmm_ops = 2 * 2 * b * num_heads * 128 * 512  # pre + post
            bmm_mem = 2 * num_heads * (b * 640 + 128 * 512) * gemm_quant_mode.value.memory
            bmm_math = (
                bmm_ops / (database.system_spec["gpu"]["bfloat16_tc_flops"] * gemm_quant_mode.value.compute) * 1000
            )
            bmm_mem_time = bmm_mem / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_math += bmm_math
            sol_mem += bmm_mem_time
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(b: int, s: int, num_heads: int, kv_cache_dtype: common.KVCacheQuantMode) -> float:
            # SOL / util from own (num_heads, b, s) grid; raises if no data.
            sol_time = get_sol(b, s, num_heads, kv_cache_dtype)[0]

            def _slice():
                cls.load_data(database)
                wrapper = database._generation_mla_module_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(
                    wrapper,
                    kv_cache_dtype,
                    gemm_quant_mode,
                )

            grid = util_empirical.grid_for(
                (
                    "gen_mla_mod",
                    database.system,
                    database.backend,
                    database.version,
                    kv_cache_dtype.name,
                    gemm_quant_mode.name,
                ),
                _slice,
                lambda c: get_sol(c[1], c[2], c[0], kv_cache_dtype)[0],  # c=(num_heads, b, s)
                depth=3,
            )
            latency, _ = util_empirical.estimate(sol_time, (num_heads, b, s), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, num_heads, kv_cache_dtype)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, num_heads, kv_cache_dtype)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, num_heads, kv_cache_dtype)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._generation_mla_module_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            mla_dict = util_empirical.require_data_slice(
                data_wrapper,
                kv_cache_dtype,
                gemm_quant_mode,
            )
            # Generation MLA module ~ linear in seq -> raw generation grid.
            config = perf_interp.generation_grid_config(
                sol_fn=lambda n_v, b_v, s_v: get_sol(b_v, s_v, n_v, kv_cache_dtype)[0]
            )
            result = perf_interp.query(config, mla_dict, num_heads, b, s)
            latency = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, num_heads, kv_cache_dtype),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query generation MLA module data for {b=}, {s=}, "
                f"{num_heads=}, {kv_cache_dtype=}, {gemm_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MLA module latency with energy data."""
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        if self._is_context:
            prefix = kwargs.get("prefix", 0)
            result = database.query_context_mla_module(
                b=batch_size,
                s=s,
                prefix=prefix,
                num_heads=self._num_heads,
                kvcache_quant_mode=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
            )
        else:
            beam_width = kwargs.get("beam_width")
            if beam_width != 1:
                raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
            result = database.query_generation_mla_module(
                b=batch_size,
                s=s,
                num_heads=self._num_heads,
                kv_cache_dtype=self._kvcache_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
            )

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class WideEPGenerationMLA(Operation):
    """
    WideEP Generation MLA operation (SGLang-only). Owns
    ``_wideep_generation_mla_data``. Loaded only when ``backend == "sglang"``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads wideep_generation_mla CSV (SGLang only),
        applies extrapolation, binds ``database._wideep_generation_mla_data``.

        Non-SGLang backends get ``None`` (matching the legacy
        ``if backend == "sglang"`` guard in ``__init__``)."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend != "sglang":
                cls._data_cache[key] = None
            else:
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                primary_path = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.wideep_generation_mla.value
                )
                sources = database._build_op_sources(
                    PerfDataFilename.wideep_generation_mla, primary_path, system_data_root
                )
                cls._data_cache[key] = LoadedOpData(
                    load_wideep_generation_mla_data(sources),
                    PerfDataFilename.wideep_generation_mla,
                    primary_path,
                )
                # No load-time grid pre-expansion: queries resolve on the RAW grid via perf_interp.
            cls._record_load()

        if "_wideep_generation_mla_data" not in database.__dict__:
            database._wideep_generation_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_wideep_generation_mla)
    # ------------------------------------------------------------------

    @classmethod
    def _query_wideep_generation_mla_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attention_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query WideEP generation MLA table. Verbatim port of the legacy body."""
        # Decode attention compute dtype follows the kv-cache dtype; the fmha
        # label is inert for generation (kernel tables key on kv dtype).
        # Derive the SOL dtype from kv so label changes cannot move decode SOL
        # -- mirrors query_generation_mla's get_sol.
        fmha_quant_mode = (
            common.FMHAQuantMode.fp8
            if kvcache_quant_mode == common.KVCacheQuantMode.fp8
            else common.FMHAQuantMode.bfloat16
        )

        def get_sol(
            b: int,
            s: int,
            tp_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            hidden_size = 7168
            q_lora_rank = 1536
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            qk_nope_head_dim = 128
            v_head_dim = 128
            num_head = 128 // tp_size

            # NOTE: qkv_a projection is now modeled as a standalone GEMM op
            # (generation_qkv_a_proj_gemm) outside of the MLA attention forward path,
            # matching sglang >=0.5.6 where qkv_a_proj was moved out of attention.

            # q_b projection
            q_b_flop = 2 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b
            q_b_mem = (
                b * q_lora_rank
                + q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim)
                + 2 * b * num_head * (qk_rope_head_dim + qk_nope_head_dim)
            )

            # q_w_kc (attention computation)
            q_w_kc_flop = 2 * num_head * qk_nope_head_dim * kv_lora_rank * b
            q_w_kc_mem = (
                b * num_head * qk_nope_head_dim
                + num_head * kv_lora_rank * qk_nope_head_dim
                + 2 * b * num_head * kv_lora_rank
            )

            attn_flop = 2 * b * s * num_head * (qk_rope_head_dim + kv_lora_rank * 2)
            attn_mem = (
                b * num_head * (kv_lora_rank + qk_rope_head_dim)
                + b * s * (qk_rope_head_dim + kv_lora_rank)
                + b * num_head * kv_lora_rank
            )

            # s_w_vc (attention output projection)
            s_w_vc_flop = 2 * b * num_head * kv_lora_rank * v_head_dim
            s_w_vc_mem = (
                b * num_head * kv_lora_rank + num_head * v_head_dim * kv_lora_rank + 2 * b * num_head * v_head_dim
            )

            # attention output projection
            attn_out_flop = 2 * num_head * v_head_dim * hidden_size * b
            attn_out_mem = b * num_head * v_head_dim + num_head * v_head_dim * hidden_size + 2 * b * hidden_size

            ops = q_b_flop + q_w_kc_flop + s_w_vc_flop + attn_out_flop
            mem_bytes = (q_b_mem + q_w_kc_mem + attn_mem * 2 + s_w_vc_mem + attn_out_mem) * fmha_quant_mode.value.memory
            sol_math = ops / (database.system_spec["gpu"]["bfloat16_tc_flops"] * fmha_quant_mode.value.compute) * 1000
            sol_math += attn_flop / (database.system_spec["gpu"]["bfloat16_tc_flops"]) * 1000
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)

            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            tp_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            # SOL / util from own (num_heads, b, s) grid; num_heads = 128 // tp_size
            # (mirrors get_silicon).
            sol_time = get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
            attn_backend = attention_backend or "flashinfer"

            def _slice():
                cls.load_data(database)
                wrapper = database._wideep_generation_mla_data
                if wrapper is None:
                    raise PerfDataNotAvailableError("WideEP generation MLA data is SGLang-only.")
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(wrapper, attn_backend, kvcache_quant_mode)

            grid = util_empirical.grid_for(
                (
                    "wideep_gen_mla",
                    database.system,
                    database.backend,
                    database.version,
                    attn_backend,
                    kvcache_quant_mode.name,
                ),
                _slice,
                lambda c: get_sol(c[1], c[2], round(128 / c[0]), kvcache_quant_mode, fmha_quant_mode)[0],
                depth=3,
            )
            latency, _ = util_empirical.estimate(sol_time, (128 // tp_size, b, s), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_time = get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_time, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(
                get_empirical(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode), energy=0.0, source="empirical"
            )

        cls.load_data(database)
        data_wrapper = database._wideep_generation_mla_data
        if data_wrapper is None:
            # Non-SGLang backends never load this table; ``load_data`` binds
            # ``None`` rather than a ``LoadedOpData(None)`` so that calling
            # ``raise_if_not_loaded()`` is not an option. Surface a structured
            # error here instead of an opaque ``NoneType`` attribute crash.
            raise PerfDataNotAvailableError(
                f"WideEP generation MLA perf data is SGLang-only; backend='{database.backend}' has no table."
            )

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            attn_backend = attention_backend or "flashinfer"
            if attn_backend not in {"flashinfer", "fa3"}:
                raise ValueError(f"Unsupported attention backend: {attn_backend}")
            attn_data = util_empirical.require_data_slice(data_wrapper, attn_backend)
            # Convert tp_size to num_heads (assuming 128 total heads for DeepSeek)
            num_heads = 128 // tp_size
            mla_dict = util_empirical.require_data_slice(attn_data, kvcache_quant_mode)
            # Raw generation grid; the table is keyed by num_heads while the SOL
            # takes tp_size, so the sol_fn maps n_v -> 128 // n_v.
            config = perf_interp.generation_grid_config(
                sol_fn=lambda n_v, b_v, s_v: get_sol(b_v, s_v, 128 // n_v, kvcache_quant_mode, fmha_quant_mode)[0]
            )
            result = perf_interp.query(config, mla_dict, num_heads, b, s)
            latency = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, tp_size, kvcache_quant_mode, fmha_quant_mode),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query wideep generation mla data for {b=}, {s=}, {tp_size=}, "
                f"{kvcache_quant_mode=}, {fmha_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query WideEP generation MLA latency with power data."""
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_wideep_generation_mla(
            batch_size,
            s,
            self._tp_size,
            self._kvcache_quant_mode,
            self._fmha_quant_mode,
            self._attn_backend,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class WideEPContextMLA(Operation):
    """
    WideEP Context MLA operation (SGLang-only). Owns
    ``_wideep_context_mla_data``. Loaded only when ``backend == "sglang"``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend
        # CP (sglang AllGather zigzag); see ContextMLA. cp>1 -> rank-0 two-chunk.
        self._cp_size = cp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads wideep_context_mla CSV (SGLang only),
        applies extrapolation, binds ``database._wideep_context_mla_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend != "sglang":
                cls._data_cache[key] = None
            else:
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                primary_path = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.wideep_context_mla.value
                )
                sources = database._build_op_sources(
                    PerfDataFilename.wideep_context_mla, primary_path, system_data_root
                )
                cls._data_cache[key] = LoadedOpData(
                    load_wideep_context_mla_data(sources),
                    PerfDataFilename.wideep_context_mla,
                    primary_path,
                )
                # No load-time grid pre-expansion: queries resolve on the RAW grid via perf_interp.
            cls._record_load()

        if "_wideep_context_mla_data" not in database.__dict__:
            database._wideep_context_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_wideep_context_mla)
    # ------------------------------------------------------------------

    @classmethod
    def _query_wideep_context_mla_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        prefix: int,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attention_backend: str | None = None,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query WideEP context MLA table. Verbatim port of the legacy body."""

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            tp_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            hidden_size = 7168
            q_lora_rank = 1536
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            qk_nope_head_dim = 128
            v_head_dim = 128
            num_head = 128 // tp_size

            # NOTE: qkv_a projection is now modeled as a standalone GEMM op in the pipeline
            # (context_qkv_a_proj_gemm), so it is excluded from this SOL calculation.

            # q_b projection
            q_b_flop = 2 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b * s
            q_b_mem = (
                b * q_lora_rank * s
                + q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim)
                + 2 * b * num_head * (qk_rope_head_dim + qk_nope_head_dim) * s
            )

            # kv_b projection
            kv_b_flop = 2 * kv_lora_rank * num_head * (qk_nope_head_dim + v_head_dim) * b * s
            kv_b_mem = (
                b * s * kv_lora_rank
                + num_head * (qk_nope_head_dim + v_head_dim) * kv_lora_rank
                + 2 * b * num_head * (qk_nope_head_dim + v_head_dim) * s
            )

            # attention computation (prefill mode)
            full_s = s + prefix
            attn_flop = (
                2 * num_head * (qk_nope_head_dim * 2 + qk_rope_head_dim) * b * (full_s * full_s - prefix * prefix) // 2
            )
            attn_mem = (
                b * s * num_head * (qk_nope_head_dim + qk_rope_head_dim)  # q read
                + b * full_s * num_head * (qk_nope_head_dim + qk_rope_head_dim)  # k read
                + b * full_s * num_head * qk_nope_head_dim  # v read
                + b * s * num_head * qk_nope_head_dim  # write
            )

            # attention output projection
            attn_out_flop = 2 * num_head * v_head_dim * hidden_size * b * s
            attn_out_mem = b * num_head * v_head_dim * s + num_head * v_head_dim * hidden_size + 2 * b * hidden_size * s

            ops = q_b_flop + kv_b_flop + attn_out_flop
            mem_bytes = (q_b_mem + kv_b_mem + attn_mem * 2 + attn_out_mem) * fmha_quant_mode.value.memory
            sol_math = ops / (database.system_spec["gpu"]["bfloat16_tc_flops"] * fmha_quant_mode.value.compute) * 1000
            sol_math += attn_flop / (database.system_spec["gpu"]["bfloat16_tc_flops"]) * 1000
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            tp_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            # SOL / util from own (num_heads, full_s, b) grid; num_heads = 128 // tp_size.
            # Samples are prefix=0; SOL(query) carries prefix natively.
            sol_time = get_sol(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
            attn_backend = attention_backend or "flashinfer"

            def _slice():
                cls.load_data(database)
                wrapper = database._wideep_context_mla_data
                if wrapper is None:
                    raise PerfDataNotAvailableError("WideEP context MLA data is SGLang-only.")
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(
                    wrapper,
                    attn_backend,
                    fmha_quant_mode,
                    kvcache_quant_mode,
                )

            grid = util_empirical.grid_for(
                (
                    "wideep_ctx_mla",
                    database.system,
                    database.backend,
                    database.version,
                    attn_backend,
                    fmha_quant_mode.name,
                    kvcache_quant_mode.name,
                ),
                _slice,
                lambda c: get_sol(c[2], c[1], 0, round(128 / c[0]), kvcache_quant_mode, fmha_quant_mode)[0],
                depth=3,
            )
            latency, _ = util_empirical.estimate(sol_time, (128 // tp_size, s + prefix, b), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_time = get_sol(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_time, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(
                get_empirical(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode),
                energy=0.0,
                source="empirical",
            )

        cls.load_data(database)
        data_wrapper = database._wideep_context_mla_data
        if data_wrapper is None:
            # See WideEPGenerationMLA above for rationale.
            raise PerfDataNotAvailableError(
                f"WideEP context MLA perf data is SGLang-only; backend='{database.backend}' has no table."
            )

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            attn_backend = attention_backend or "flashinfer"
            if attn_backend not in {"flashinfer", "fa3"}:
                raise ValueError(f"Unsupported attention backend: {attn_backend}")
            attn_data = util_empirical.require_data_slice(data_wrapper, attn_backend)

            # Convert tp_size to num_heads (assuming 128 total heads for DeepSeek)
            num_heads = 128 // tp_size
            mla_dict = util_empirical.require_data_slice(attn_data, fmha_quant_mode, kvcache_quant_mode)
            full_s = s + prefix
            prefix_correction = (full_s * full_s - prefix * prefix) / (full_s * full_s)
            # Context grid (sqrt on seq only); table keyed by num_heads, SOL takes
            # tp_size -> map n_v -> 128 // n_v; samples are prefix=0.
            config = perf_interp.context_grid_config(
                sol_fn=lambda n_v, s_v, b_v: get_sol(b_v, s_v, 0, 128 // n_v, kvcache_quant_mode, fmha_quant_mode)[0]
            )
            result = perf_interp.query(config, mla_dict, num_heads, full_s, b)
            latency = perf_interp.get_value(result, "latency") * prefix_correction
            energy = perf_interp.get_value(result, "energy") * prefix_correction
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, prefix, tp_size, kvcache_quant_mode, fmha_quant_mode),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query wideep context mla data for {b=}, {s=}, {prefix=}, {tp_size=}, "
                f"{kvcache_quant_mode=}, {fmha_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query WideEP context MLA latency with power data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        def _q(s, pfx):
            return database.query_wideep_context_mla(
                b=batch_size,
                s=s,
                prefix=pfx,
                tp_size=self._tp_size,
                kvcache_quant_mode=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                attention_backend=self._attn_backend,
            )

        if self._cp_size and self._cp_size > 1:
            cp = self._cp_size
            c = max(1, -(-isl // (2 * cp)))  # ceil(isl/(2*cp)) — rank-0 zigzag halves
            result = _q(c, prefix) + _q(c, prefix + isl - c)
        else:
            result = _q(isl, prefix)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_context_mla_data(context_mla_file):
    """
    Load the context mla data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(context_mla_file)
    if rows is None:
        logger.debug(f"Context mla data file {context_mla_file} not found.")
        return None
    context_mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (context_mla) - power will default to 0.0")

    for row in rows:
        (
            quant_mode,
            kv_cache_dtype,
            b,
            s,
            latency,
        ) = row["mla_dtype"], row["kv_cache_dtype"], row["batch_size"], row["isl"], row["latency"]

        if "num_heads" not in row:
            tp_size = int(row["tp_size"])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b]
            logger.debug(f"value conflict in context mla data: {quant_mode} {kv_cache_dtype} {num_heads} {s} {b}")
        except KeyError:
            # Store all three values
            context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return context_mla_data


def load_generation_mla_data(generation_mla_file):
    """
    Load the generation mla data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(generation_mla_file)
    if rows is None:
        logger.debug(f"Generation mla data file {generation_mla_file} not found.")
        return None
    generation_mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (generation_mla) - power will default to 0.0")

    for row in rows:
        quant_mode, kv_cache_dtype, b, s, step, latency = (  # noqa: F841
            row["mla_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["step"],
            row["latency"],
        )

        if "num_heads" not in row:
            tp_size = int(row["tp_size"])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            generation_mla_data[kv_cache_dtype][num_heads][b][s]
            logger.debug(f"value conflict in generation mla data: {kv_cache_dtype} {num_heads} {b} {s} ")
        except KeyError:
            # Store all three values
            generation_mla_data[kv_cache_dtype][num_heads][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return generation_mla_data


def load_mla_bmm_data(mla_bmm_file):
    """
    Load the mla bmm data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(mla_bmm_file)
    if rows is None:
        logger.debug(f"MLA BMM data file {mla_bmm_file} not found.")
        return None
    mla_bmm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (mla_bmm) - power will default to 0.0")

    for row in rows:
        quant_mode, num_tokens, num_heads, latency, op_name = (
            row["bmm_dtype"],
            row["num_tokens"],
            row["num_heads"],
            row["latency"],
            row["op_name"],
        )
        num_tokens = int(num_tokens)
        num_heads = int(num_heads)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            mla_bmm_data[quant_mode][op_name][num_heads][num_tokens]
            logger.debug(f"value conflict in mla bmm data: {op_name} {quant_mode} {num_heads} {num_tokens} ")
        except KeyError:
            # Store all three values
            mla_bmm_data[quant_mode][op_name][num_heads][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return mla_bmm_data


def load_wideep_context_mla_data(wideep_context_mla_file):
    """
    Load the SGLang wideep context mla data from wideep_context_mla_perf.txt
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(wideep_context_mla_file)
    if rows is None:
        logger.debug(f"SGLang wideep context mla data file {wideep_context_mla_file} not found.")
        return None
    wideep_context_mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
    )

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_context_mla) - power will default to 0.0")

    for row in rows:
        (
            quant_mode,
            kv_cache_dtype,
            b,
            s,
            latency,
        ) = row["mla_dtype"], row["kv_cache_dtype"], row["batch_size"], row["isl"], row["latency"]

        kernel_source = row.get("kernel_source", "flashinfer")

        if "num_heads" not in row:
            tp_size = int(row["tp_size"])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b]
            logger.debug(
                f"value conflict in context mla data: {kernel_source} {quant_mode} {kv_cache_dtype} {num_heads} {s} {b}"
            )
        except KeyError:
            # Store all three values
            wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_context_mla_data


def load_wideep_generation_mla_data(wideep_generation_mla_file):
    """
    Load the SGLang wideep generation mla data from wideep_generation_mla_perf.txt
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(wideep_generation_mla_file)
    if rows is None:
        logger.debug(f"SGLang wideep generation mla data file {wideep_generation_mla_file} not found.")
        return None
    wideep_generation_mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    )

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_generation_mla) - power will default to 0.0")

    for row in rows:
        kv_cache_dtype, b, s, step, latency = (
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["step"],
            row["latency"],
        )

        kernel_source = row.get("kernel_source", "flashinfer")

        if "num_heads" not in row:
            tp_size = int(row["tp_size"])
            num_heads = 128 // tp_size
        else:
            num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            wideep_generation_mla_data[kernel_source][kv_cache_dtype][num_heads][b][s]
            logger.debug(
                f"value conflict in generation mla data: {kernel_source} {kv_cache_dtype} {num_heads} {b} {s} "
            )
        except KeyError:
            # Store all three values
            wideep_generation_mla_data[kernel_source][kv_cache_dtype][num_heads][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_generation_mla_data


def load_context_mla_module_data(mla_module_file: str):
    """
    Load context MLA module-level performance data.

    CSV columns: framework, version, device, op_name, kernel_source, model,
    architecture, mla_dtype, kv_cache_dtype, gemm_type, num_heads,
    batch_size, isl, tp_size, step, latency [, power]

    Dict structure (matches context_mla_data nesting):
        data[fmha_quant_mode][kv_cache_quant_mode][gemm_quant_mode][num_heads][s][b]
    """
    rows = _read_filtered_rows(mla_module_file)
    if rows is None:
        logger.debug(f"MLA context module data file {mla_module_file} not found.")
        return None

    mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
    )

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
        num_heads = int(row["num_heads"])
        b = int(row["batch_size"])
        s = int(row["isl"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency

        fmha_mode = common.FMHAQuantMode[row["mla_dtype"]]
        kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]

        mla_data[fmha_mode][kv_dtype][gemm_mode][num_heads][s][b] = {
            "latency": latency,
            "power": power,
            "energy": energy,
        }

    return mla_data


def load_generation_mla_module_data(mla_module_file: str):
    """
    Load generation MLA module-level performance data.

    CSV columns: framework, version, device, op_name, kernel_source, model,
    architecture, mla_dtype, kv_cache_dtype, gemm_type, num_heads,
    batch_size, isl, tp_size, step, latency [, power]

    Dict structure:
        data[kv_cache_quant_mode][gemm_quant_mode][num_heads][b][s]

    The ``mla_dtype`` column is ignored: decode MLA compute dtype follows the
    KV cache dtype (collectors hardcode ``bfloat16`` in that column), so it is
    not a real axis — mirroring ``load_generation_mla_data``, which likewise
    drops it.
    """
    rows = _read_filtered_rows(mla_module_file)
    if rows is None:
        logger.debug(f"MLA generation module data file {mla_module_file} not found.")
        return None

    mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
        num_heads = int(row["num_heads"])
        b = int(row["batch_size"])
        s = int(row["isl"]) + int(row["step"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency

        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]

        mla_data[kv_dtype][gemm_mode][num_heads][b][s] = {
            "latency": latency,
            "power": power,
            "energy": energy,
        }

    return mla_data
