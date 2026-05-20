# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context + Generation attention ops (ISSUE-06 / AIC-543).

Both classes own their CSV-backed perf tables, SOL correction (generation
only — context attention has no SOL clamp in the legacy
``_correct_data``), and grid extrapolation.
``PerfDatabase.query_context_attention`` / ``query_generation_attention``
delegate here.

``ContextAttention.query`` switches its three ``query_mem_op`` callers
(QK-norm, apply-RoPE, KV-write) to ``interpolation.estimate_mem_op``
directly — removing one of the cross-cutting ``database.*`` couplings
the refactor aims to eliminate.

Cache key is ``(systems_root, system, backend, version,
enable_shared_layer)``, same as GEMM (and every other migrated op).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import common, interpolation
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


# Extrapolation target grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__`` so behavior stays bit-identical.

# fmt: off
_CONTEXT_ATTENTION_TARGET_X: list[int] = [
    1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48,
    56, 72, 96, 128,
]  # n
_CONTEXT_ATTENTION_TARGET_Y: list[int] = (
    [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
    + [4096 + i * 2048 for i in range(14)]
    + [32768 + 16384 * i for i in range(6)]
    + [131072 + 32768 * i for i in range(12)]
    + [524288 + 65536 * i for i in range(9)]
)  # s
_CONTEXT_ATTENTION_TARGET_Z: list[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 384, 1024, 2048,
]  # b

_GENERATION_ATTENTION_TARGET_X: list[int] = [
    1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48,
    56, 72, 96, 128,
]  # n
_GENERATION_ATTENTION_TARGET_Y: list[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 8192,
]  # b
_GENERATION_ATTENTION_TARGET_Z: list[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    32768, 65536, 131072, 262144, 2097152 * 8,
]  # s
# fmt: on


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as GEMM's, used by both Attention ops.

    TODO: hoist to ``operations/base.py`` once a third op family (Phase 3
    NCCL / MLA / Mamba) lands and needs the same key shape — preferring
    duplication over premature abstraction with only two callers.
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


class ContextAttention(Operation):
    """
    Context (prefill) attention operation.

    Owns ``_data_cache: {key: LoadedOpData}`` for the context attention CSV.
    No SOL clamp on the loaded table (legacy ``_correct_data`` did not
    correct context attention) — only grid extrapolation runs in ``load_data``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        window_size: int = 0,
        head_size: int = 128,
        use_qk_norm: bool = False,
    ) -> None:
        """Initialize context attention query parameters."""
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._window_size = window_size
        self._head_size = head_size
        self._use_qk_norm = use_qk_norm

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads context_attention CSV into the class cache,
        applies grid extrapolation, binds ``database._context_attention_data``.

        Mirrors ``GEMM.load_data``: correction/extrapolation operate on the
        canonical class-cache value (passed explicitly), then the instance
        attr is bound, respecting any pre-set test override."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename, load_context_attention_data

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.context_attention.value)
            sources = database._build_op_sources(PerfDataFilename.context_attention, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_context_attention_data(sources), PerfDataFilename.context_attention, primary_path
            )

            cls._extrapolate(cls._data_cache[key])
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_context_attention_data" not in database.__dict__:
            database._context_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    @classmethod
    def _extrapolate(cls, data_wrapper) -> None:
        """Apply the legacy 4-level (quant_mode → kv_cache_dtype → num_kv_heads
        → head_size → window_size → grid) extrapolation."""
        if data_wrapper is None or not getattr(data_wrapper, "loaded", False):
            return

        for quant_mode in data_wrapper:
            for kv_cache_dtype in data_wrapper[quant_mode]:
                for num_kv_heads in data_wrapper[quant_mode][kv_cache_dtype]:
                    for head_size in data_wrapper[quant_mode][kv_cache_dtype][num_kv_heads]:
                        for window_size in data_wrapper[quant_mode][kv_cache_dtype][num_kv_heads][head_size]:
                            data_dict = data_wrapper[quant_mode][kv_cache_dtype][num_kv_heads][head_size][window_size]
                            min_x = min(data_dict.keys())
                            filtered_x = [i for i in _CONTEXT_ATTENTION_TARGET_X if i >= min_x]
                            interpolation.extrapolate_data_grid(
                                data_dict=data_dict,
                                target_x_list=filtered_x,
                                target_y_list=_CONTEXT_ATTENTION_TARGET_Y,
                                target_z_list=_CONTEXT_ATTENTION_TARGET_Z,
                                sqrt_y_value=True,
                            )

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_attention)
    # ------------------------------------------------------------------

    @classmethod
    def _query_context_attention_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        prefix: int,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: common.DatabaseMode | None = None,
        window_size: int = 0,
        head_size: int = 128,
    ):
        """Query context attention table. Verbatim port of the legacy body."""

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            full_s = s + prefix
            if w > 0 and full_s > w:
                ops = 2 * b * (full_s - prefix) * w * n * h * 2
            else:
                ops = 2 * b * (full_s * full_s - prefix * prefix) * n * h * 2 / 2
            mem_bytes = 2 * b * (
                n * (full_s - prefix) * h + n * (full_s - prefix) * h
            ) + kvcache_quant_mode.value.memory * b * (2 * n_kv * full_s * h)
            sol_math = ops / database.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            n: int,
            n_kv: int,
            head_size: int,
            window_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            latency = get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
            scale_factor = 0.6
            return latency / scale_factor

        assert n_kv <= n, "n_kv must be less than or equal to n"

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(
                b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode
            )
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._context_attention_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            full_s = s + prefix
            prefix_correction = (full_s * full_s - prefix * prefix) / (full_s * full_s)
            n_kv_lookup = 0 if n == n_kv else n_kv
            attention_dict = data_wrapper[fmha_quant_mode][kvcache_quant_mode][n_kv_lookup][head_size][window_size]
            result = database._interp_3d(n, full_s, b, attention_dict, "cubic")
            latency = result["latency"] * prefix_correction
            energy = result.get("energy", 0.0) * prefix_correction
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(
                b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode
            ),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query context attention data for {b=}, {s=}, {prefix=}, {n=}, {n_kv=}, "
                f"{head_size=}, {window_size=}, {kvcache_quant_mode=}, {fmha_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context attention latency with energy data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        result = database.query_context_attention(
            batch_size,
            isl,
            prefix,
            self._n,
            self._n_kv,
            self._kvcache_quant_mode,
            self._fmha_quant_mode,
            window_size=self._window_size,
            head_size=self._head_size,
        )
        q_num = self._n * self._head_size
        k_num = self._n_kv * self._head_size
        v_num = self._n_kv * self._head_size
        # QK-norm / apply-RoPE / KV-write mem-op latencies. Switched from
        # ``database.query_mem_op`` to ``interpolation.estimate_mem_op`` as
        # part of ISSUE-06 — see Phase-2 design doc Section 5.4.
        gpu_spec = database.system_spec["gpu"]
        mode = database._default_database_mode
        extra_latency = 0
        if self._use_qk_norm:
            qk_norm_latency = 2 * interpolation.estimate_mem_op(
                gpu_spec, q_num * 2, mode
            ) + 2 * interpolation.estimate_mem_op(gpu_spec, k_num * 2, mode)
            extra_latency += qk_norm_latency * 2  # elementwise before norm
        apply_rope_latency = 2 * interpolation.estimate_mem_op(gpu_spec, q_num * 2 + k_num * 2, mode)

        kv_write_latency = interpolation.estimate_mem_op(
            gpu_spec, k_num * self._fmha_quant_mode.value.memory, mode
        ) + interpolation.estimate_mem_op(gpu_spec, v_num * self._fmha_quant_mode.value.memory, mode)
        extra_latency += apply_rope_latency + kv_write_latency
        result += extra_latency * 1.1  # correction factor for extra latency

        seq_imbalance_correction_scale = float(kwargs.get("seq_imbalance_correction_scale", 1.0))
        if seq_imbalance_correction_scale != 1.0:
            result = result * seq_imbalance_correction_scale

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationAttention(Operation):
    """
    Generation (decode) attention operation.

    Owns ``_data_cache: {key: LoadedOpData}`` for the generation attention
    CSV. ``load_data`` applies both SOL clamping AND grid extrapolation
    (legacy ``_correct_data`` clamped, then ``__init__`` extrapolated).
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        window_size: int = 0,
        head_size: int = 128,
        use_qk_norm: bool = False,
    ) -> None:
        """Initialize generation attention query parameters."""
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype
        self._window_size = window_size
        self._head_size = head_size
        self._use_qk_norm = use_qk_norm

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads generation_attention CSV, clamps to SOL, applies
        grid extrapolation, binds ``database._generation_attention_data``.

        Mirrors ``GEMM.load_data``: correction/extrapolation operate on the
        canonical class-cache value (passed explicitly), then the instance
        attr is bound, respecting any pre-set test override."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename, load_generation_attention_data

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.generation_attention.value)
            sources = database._build_op_sources(PerfDataFilename.generation_attention, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_generation_attention_data(sources), PerfDataFilename.generation_attention, primary_path
            )

            cls._correct_sol(database, cls._data_cache[key])
            cls._extrapolate(cls._data_cache[key])
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_generation_attention_data" not in database.__dict__:
            database._generation_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    @classmethod
    def _correct_sol(cls, database: PerfDatabase, data_wrapper=None) -> None:
        """Clamp generation-attention table latencies to ≥ SOL.

        ``data_wrapper`` defaults to ``database._generation_attention_data``
        so the backward-compat call from ``PerfDatabase._correct_data``
        works after tests mutate the instance attr."""
        if data_wrapper is None:
            data_wrapper = getattr(database, "_generation_attention_data", None)
        if data_wrapper is None or not getattr(data_wrapper, "loaded", False):
            return

        for quant_mode in data_wrapper:
            for n_kv in data_wrapper[quant_mode]:
                for head_size in data_wrapper[quant_mode][n_kv]:
                    for window_size in data_wrapper[quant_mode][n_kv][head_size]:
                        for n in data_wrapper[quant_mode][n_kv][head_size][window_size]:
                            for b in data_wrapper[quant_mode][n_kv][head_size][window_size][n]:
                                for s in data_wrapper[quant_mode][n_kv][head_size][window_size][n][b]:
                                    n_kv_local = n if n_kv == 0 else n_kv
                                    sol = cls._query_generation_attention_table(
                                        database,
                                        b,
                                        s,
                                        n,
                                        n_kv_local,
                                        quant_mode,
                                        database_mode=common.DatabaseMode.SOL,
                                        window_size=window_size,
                                        head_size=head_size,
                                    )
                                    data = data_wrapper[quant_mode][n_kv][head_size][window_size][n][b][s]
                                    current_latency = data["latency"] if isinstance(data, dict) else data
                                    if sol > current_latency:
                                        logger.debug(
                                            f"generation attention quant {quant_mode} n{n} "
                                            f"n_kv{n_kv_local} b{b} s{s}: sol {sol} > "
                                            f"perf_db {current_latency}"
                                        )
                                        if isinstance(data, dict):
                                            data_wrapper[quant_mode][n_kv][head_size][window_size][n][b][s][
                                                "latency"
                                            ] = float(sol)
                                        else:
                                            data_wrapper[quant_mode][n_kv][head_size][window_size][n][b][s] = float(sol)

    @classmethod
    def _extrapolate(cls, data_wrapper) -> None:
        """Apply the legacy 4-level extrapolation grid."""
        if data_wrapper is None or not getattr(data_wrapper, "loaded", False):
            return

        for kv_cache_dtype in data_wrapper:
            for num_kv_heads in data_wrapper[kv_cache_dtype]:
                for head_size in data_wrapper[kv_cache_dtype][num_kv_heads]:
                    for window_size in data_wrapper[kv_cache_dtype][num_kv_heads][head_size]:
                        data_dict = data_wrapper[kv_cache_dtype][num_kv_heads][head_size][window_size]
                        min_x = min(data_dict.keys())
                        filtered_x = [i for i in _GENERATION_ATTENTION_TARGET_X if i >= min_x]
                        interpolation.extrapolate_data_grid(
                            data_dict=data_dict,
                            target_x_list=filtered_x,
                            target_y_list=_GENERATION_ATTENTION_TARGET_Y,
                            target_z_list=_GENERATION_ATTENTION_TARGET_Z,
                        )

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_attention)
    # ------------------------------------------------------------------

    @classmethod
    def _query_generation_attention_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        database_mode: common.DatabaseMode | None = None,
        window_size: int = 0,
        head_size: int = 128,
    ):
        """Query generation attention table. Verbatim port of legacy body."""

        def get_sol(
            b: int,
            s: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> tuple[float, float, float]:
            if kvcache_quant_mode == common.KVCacheQuantMode.fp8:
                quant_mode_gen = common.FMHAQuantMode.fp8
            else:
                quant_mode_gen = common.FMHAQuantMode.bfloat16
            if w > 0:
                kv_len = min(s - 1, w)
            else:
                kv_len = s - 1
            ops = 2 * b * n * h * 2 * (kv_len)
            mem_bytes = b * (n * h * 2 + 2 * n_kv * (kv_len) * h * kvcache_quant_mode.value.memory + n * h * 2)

            sol_math = ops / database.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / quant_mode_gen.value.compute
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> float:
            latency = get_sol(b, s, n, n_kv, h, w, kvcache_quant_mode)[0]
            scale_factor = 0.8
            return latency / scale_factor

        assert n_kv <= n, "n_kv must be less than or equal to n"

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._generation_attention_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            n_kv_lookup = n_kv if n_kv != n else 0

            attention_dict = data_wrapper[kvcache_quant_mode][n_kv_lookup][head_size][window_size]
            s_min = max(1, int(s * 0.9))
            s_max = max(s_min, int(s * 1.1))
            sample_cnt = 5
            s_samples = [s_min + (s_max - s_min) * i // (sample_cnt - 1) for i in range(sample_cnt)]

            latency_sum = 0.0
            energy_sum = 0.0
            for s_i in s_samples:
                r = database._interp_3d(n, b, s_i, attention_dict, "bilinear")
                latency_sum += float(r["latency"])
                energy_sum += float(r.get("energy", 0.0))

            latency = latency_sum / sample_cnt
            energy = energy_sum / sample_cnt
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query generation attention data for {b=}, {s=}, {n=}, {n_kv=}, "
                f"{head_size=}, {window_size=}, {kvcache_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation attention latency with energy data."""
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_attention(
            batch_size,
            s,
            self._n,
            self._n_kv,
            self._kv_cache_dtype,
            window_size=self._window_size,
            head_size=self._head_size,
        )
        gen_seq_imbalance_correction_scale = float(
            kwargs.get(
                "gen_seq_imbalance_correction_scale",
                kwargs.get("seq_imbalance_correction_scale", 1.0),
            )
        )
        if gen_seq_imbalance_correction_scale != 1.0:
            result = result * gen_seq_imbalance_correction_scale
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
