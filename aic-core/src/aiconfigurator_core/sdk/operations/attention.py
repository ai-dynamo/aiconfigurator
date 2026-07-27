# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context + Generation attention ops (ISSUE-06 / AIC-543).

Both classes own their CSV-backed perf tables, SOL correction (generation
only — context attention has no SOL clamp in the legacy
``_correct_data``), and grid extrapolation.
``PerfDatabase.query_context_attention`` / ``query_generation_attention``
delegate here.

``ContextAttention.query`` keeps its three ``query_mem_op`` callers
(QK-norm, apply-RoPE, KV-write) pointed at ``database.query_mem_op`` —
deciding a long-term home for the analytical mem-op formula is deferred
to the post-refactor cleanup.

Cache key is ``(systems_root, system, backend, version,
enable_shared_layer)``, same as GEMM (and every other migrated op).
"""

from __future__ import annotations

import logging
import math
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

# Attention utilisation vs head_size, relative to head_size=128. Used to transfer
# util across head_size when the exact head_size has no collected data: borrow the
# nearest collected head_size's util curve and rescale by ratio[target]/ratio[ref].
#
# Measured on Blackwell (b200) GQA where 64/128/256 are collected; the cross-GPU
# spread within a backend is small, so a per-backend table is enough. PREFILL util
# rises with head_size (compute-bound: larger heads pack the tensor cores better);
# 192 is log2-interpolated and 512 EXTRAPOLATED under the observed diminishing-
# returns trend (util saturates) -- both are estimates, not measured. 512 is held
# conservatively close to 256 (only a small bump) since there is no data and util
# is near-saturated there; under-crediting its efficiency errs toward over- rather
# than under-estimating latency. DECODE util is ~head_size-independent (memory-bound
# KV read), so its ratio is 1.0 (no table).
_ATTN_PREFILL_HS_RATIO: dict[str, dict[int, float]] = {
    "trtllm": {64: 0.58, 128: 1.00, 192: 1.10, 256: 1.17, 512: 1.20},
    "sglang": {64: 0.60, 128: 1.00, 192: 1.18, 256: 1.32, 512: 1.38},
    "vllm": {64: 0.60, 128: 1.00, 192: 1.27, 256: 1.51, 512: 1.60},
}


def _attn_prefill_hs_ratio(backend: str, head_size: int) -> float:
    """Prefill-attention util ratio vs head_size=128, log2-interpolated between
    table points and clamped at the ends. Unknown backend -> 1.0 (no correction)."""
    table = _ATTN_PREFILL_HS_RATIO.get(backend)
    if not table:
        return 1.0
    if head_size in table:
        return table[head_size]
    keys = sorted(table)
    if head_size <= keys[0]:
        return table[keys[0]]
    if head_size >= keys[-1]:
        return table[keys[-1]]
    lo = max(k for k in keys if k < head_size)
    hi = min(k for k in keys if k > head_size)
    t = (math.log2(head_size) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return table[lo] + t * (table[hi] - table[lo])


def _ref_head_size(available, target: int):
    """Pick the reference head_size to transfer util from. Prefer 128 -- the canonical,
    most densely collected head_size and the head_size-util table's reference point --
    so the borrowed util grid is reliable; otherwise the nearest collected head_size in
    log2 space. Returns None if nothing is collected."""
    avail = [h for h in available if h]
    if not avail:
        return None
    if 128 in avail:
        return 128
    return min(avail, key=lambda h: abs(math.log2(h) - math.log2(target)))


def _ctx_headsize_ref_grid(database, fmha_quant_mode, kvcache_quant_mode, n_kv_lookup, target_hs, window_size, get_sol):
    """Reference util grid for context attention borrowed from the nearest collected
    head_size (same fmha/kv/n_kv/window). Returns ``(grid, ref_hs)`` or ``(None, None)``."""
    try:
        from aiconfigurator_core.sdk.operations.attention import ContextAttention

        ContextAttention.load_data(database)
        wrapper = database._context_attention_data
        wrapper.raise_if_not_loaded()
        by_hs = util_empirical.require_data_slice(wrapper, fmha_quant_mode, kvcache_quant_mode, n_kv_lookup)
        ref_hs = _ref_head_size(list(by_hs.keys()), target_hs)
    except PerfDataNotAvailableError:
        return None, None
    if ref_hs is None:
        return None, None

    def _ref_slice():
        return util_empirical.require_data_slice(
            database._context_attention_data,
            fmha_quant_mode,
            kvcache_quant_mode,
            n_kv_lookup,
            ref_hs,
            window_size,
        )

    def _ref_sol(c):  # c = (n, full_s, b)
        nkv = c[0] if n_kv_lookup == 0 else n_kv_lookup
        return get_sol(c[2], c[1], 0, c[0], nkv, ref_hs, window_size, kvcache_quant_mode, fmha_quant_mode)[0]

    grid = util_empirical.grid_for(
        (
            "ctx_attn_xhs",
            database.system,
            database.backend,
            database.version,
            fmha_quant_mode.name,
            kvcache_quant_mode.name,
            n_kv_lookup,
            ref_hs,
            window_size,
        ),
        _ref_slice,
        _ref_sol,
        depth=3,
    )
    if grid is None or not grid.samples:
        return None, None
    return grid, ref_hs


def _gen_headsize_ref_grid(database, kvcache_quant_mode, n_kv_lookup, target_hs, window_size, get_sol):
    """Reference util grid for generation attention borrowed from the nearest collected
    head_size (same kv/n_kv/window). Returns ``(grid, ref_hs)`` or ``(None, None)``."""
    try:
        from aiconfigurator_core.sdk.operations.attention import GenerationAttention

        GenerationAttention.load_data(database)
        wrapper = database._generation_attention_data
        wrapper.raise_if_not_loaded()
        by_hs = util_empirical.require_data_slice(wrapper, kvcache_quant_mode, n_kv_lookup)
        ref_hs = _ref_head_size(list(by_hs.keys()), target_hs)
    except PerfDataNotAvailableError:
        return None, None
    if ref_hs is None:
        return None, None

    def _ref_slice():
        return util_empirical.require_data_slice(
            database._generation_attention_data,
            kvcache_quant_mode,
            n_kv_lookup,
            ref_hs,
            window_size,
        )

    def _ref_sol(c):  # c = (n, b, s)
        nkv = c[0] if n_kv_lookup == 0 else n_kv_lookup
        return get_sol(c[1], c[2], c[0], nkv, ref_hs, window_size, kvcache_quant_mode)[0]

    grid = util_empirical.grid_for(
        (
            "gen_attn_xhs",
            database.system,
            database.backend,
            database.version,
            kvcache_quant_mode.name,
            n_kv_lookup,
            ref_hs,
            window_size,
        ),
        _ref_slice,
        _ref_sol,
        depth=3,
    )
    if grid is None or not grid.samples:
        return None, None
    return grid, ref_hs


# Extrapolation target grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__`` so behavior stays bit-identical.

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
        cp_size: int = 1,
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
        # Context parallelism (sglang AllGather, zigzag in-seq split). When
        # cp_size > 1, query() models ONE representative CP rank (rank 0): the
        # sequence is split into 2*cp contiguous chunks, rank 0 owns chunk 0
        # (prefix=0) and chunk 2*cp-1 (prefix=isl-c). Its two halves' work sums
        # to the balanced per-rank total isl^2/(2*cp). See
        # docs/CONTEXT_PARALLEL_DSA_MODELING.md (dense analog).
        self._cp_size = cp_size

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

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.context_attention.value
            )
            sources = database._build_op_sources(PerfDataFilename.context_attention, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_context_attention_data(sources), PerfDataFilename.context_attention, primary_path
            )

            # No load-time grid pre-expansion: queries resolve on the RAW grid via
            # perf_interp (sqrt-space blend; the truncated large-seq x large-batch
            # corner is ordinary out-of-range util-hold).
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_context_attention_data" not in database.__dict__:
            database._context_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

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
            # latency = SOL / util, util read best-effort from collected data. The query
            # SOL always uses the real window_size (get_sol's windowed branch already
            # captures the reduced O(seq*window) work). The UTIL is borrowed by slice:
            # exact window first, then window=0 (full attention) -- the window axis is
            # collected per-model (scattered: hs64/win128 gpt-oss, hs256/win1024 gemma3,
            # ...), so an uncollected window has no own basis; full-attention util is the
            # right carrier since the kernel efficiency is ~window-independent and SOL
            # absorbs the work difference.
            sol_time = get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
            n_kv_lookup = 0 if n == n_kv else n_kv

            def _own_grid(slice_window):
                def _slice():
                    cls.load_data(database)
                    wrapper = database._context_attention_data
                    wrapper.raise_if_not_loaded()
                    return util_empirical.require_data_slice(
                        wrapper,
                        fmha_quant_mode,
                        kvcache_quant_mode,
                        n_kv_lookup,
                        head_size,
                        slice_window,
                    )

                def _sol(c):  # c = (n, full_s, b); samples are full attention (prefix=0)
                    nkv = c[0] if n_kv_lookup == 0 else n_kv_lookup
                    return get_sol(
                        c[2], c[1], 0, c[0], nkv, head_size, slice_window, kvcache_quant_mode, fmha_quant_mode
                    )[0]

                return util_empirical.grid_for(
                    (
                        "ctx_attn",
                        database.system,
                        database.backend,
                        database.version,
                        fmha_quant_mode.name,
                        kvcache_quant_mode.name,
                        n_kv_lookup,
                        head_size,
                        slice_window,
                    ),
                    _slice,
                    _sol,
                    depth=3,
                )

            # Try exact window, then full-attention (window=0) as the util carrier.
            for slice_window in [window_size, 0] if window_size > 0 else [window_size]:
                grid = _own_grid(slice_window)
                if grid is not None and grid.samples:
                    latency, _ = util_empirical.estimate(sol_time, (n, s + prefix, b), grid)
                    return latency
                # Cross-head_size transfer (XSHAPE): this head_size has no data, but
                # num_heads is already an in-grid axis, so only head_size differs. Borrow
                # the nearest collected head_size's util and rescale by the prefill
                # head_size-util ratio (SOL still uses the query's own head_size).
                ref_grid, ref_hs = (
                    _ctx_headsize_ref_grid(
                        database, fmha_quant_mode, kvcache_quant_mode, n_kv_lookup, head_size, slice_window, get_sol
                    )
                    if common.TransferKind.XSHAPE in database.transfer_policy
                    else (None, None)
                )
                if ref_grid is not None:
                    scale = _attn_prefill_hs_ratio(database.backend, head_size) / _attn_prefill_hs_ratio(
                        database.backend, ref_hs
                    )
                    latency, _ = util_empirical.estimate(
                        sol_time, (n, s + prefix, b), ref_grid, util_scale=scale, provenance="xshape"
                    )
                    return latency

            # No own-window, full-attention, or cross-head basis -> raise honestly.
            latency, _ = util_empirical.estimate(sol_time, (n, s + prefix, b), None)
            return latency

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
            # Use the real windowed slice when present -- validation shows it beats a
            # window=0 + SOL-ratio reconstruction (the latter is ~25-77% off vs measured
            # windowed data). When the windowed slice is absent or too sparse to
            # interpolate, perf_interp fails accurately (raises) and HYBRID/EMPIRICAL
            # fall back to get_empirical's window=0 + SOL derivation.
            attention_dict = util_empirical.require_data_slice(
                data_wrapper,
                fmha_quant_mode,
                kvcache_quant_mode,
                n_kv_lookup,
                head_size,
                window_size,
            )
            # Resolve on the raw (n, full_s, batch) grid: sqrt-space blend for the
            # ~seq^2 curvature; past the staircase frontier (large seq x large
            # batch, uncollected) the engine holds the boundary util and lets the
            # SOL carry the growth. Samples are full attention, so the sol_fn is
            # evaluated at prefix=0 with the slice's own kv-head/window setup.
            config = perf_interp.context_attention_config(
                sol_fn=lambda n_v, s_v, b_v: get_sol(
                    b_v,
                    s_v,
                    0,
                    n_v,
                    n_v if n_kv_lookup == 0 else n_kv_lookup,
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                )[0]
            )
            result = perf_interp.query(config, attention_dict, n, full_s, b)
            latency = perf_interp.get_value(result, "latency") * prefix_correction
            energy = perf_interp.get_value(result, "energy") * prefix_correction
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

        def _ctx(s, pfx):
            return database.query_context_attention(
                batch_size,
                s,
                pfx,
                self._n,
                self._n_kv,
                self._kvcache_quant_mode,
                self._fmha_quant_mode,
                window_size=self._window_size,
                head_size=self._head_size,
            )

        if self._cp_size and self._cp_size > 1:
            # Context-parallel (zigzag) prefill: model rank 0's two halves.
            # c = ceil(isl / (2*cp)); prev chunk attends [0, prefix+c), next
            # chunk attends [0, prefix+isl). Sum = balanced per-rank work.
            cp = self._cp_size
            c = max(1, -(-isl // (2 * cp)))
            result = _ctx(c, prefix) + _ctx(c, prefix + isl - c)
        else:
            result = _ctx(isl, prefix)
        q_num = self._n * self._head_size
        k_num = self._n_kv * self._head_size
        v_num = self._n_kv * self._head_size
        extra_latency = 0
        if self._use_qk_norm:
            qk_norm_latency = 2 * database.query_mem_op(q_num * 2) + 2 * database.query_mem_op(k_num * 2)
            extra_latency += qk_norm_latency * 2  # elementwise before norm
        apply_rope_latency = 2 * database.query_mem_op(q_num * 2 + k_num * 2)  # apply rope

        kv_write_latency = database.query_mem_op(k_num * self._fmha_quant_mode.value.memory) + database.query_mem_op(
            v_num * self._fmha_quant_mode.value.memory
        )
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

    Owns an extrapolated SILICON working cache plus a raw measured cache for
    empirical utilization calibration. ``load_data`` applies SOL clamping,
    snapshots the measured rows, then expands the working grid.
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}

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
        """Idempotent. Loads generation_attention CSV, clamps to SOL,
        preserves measured rows, extrapolates the working grid, and binds both
        database views.

        Mirrors ``GEMM.load_data``: correction/extrapolation operate on the
        canonical class-cache value (passed explicitly), then the instance
        attr is bound, respecting any pre-set test override."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.generation_attention.value
            )
            sources = database._build_op_sources(PerfDataFilename.generation_attention, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_generation_attention_data(sources), PerfDataFilename.generation_attention, primary_path
            )

            cls._correct_sol(database, cls._data_cache[key])
            # No load-time grid pre-expansion: queries resolve on the RAW table
            # via perf_interp, so the table IS the raw data (the former deepcopy
            # for _raw_generation_attention_data is now just an alias).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_generation_attention_data" not in database.__dict__:
            database._generation_attention_data = cls._data_cache[key]
            database._raw_generation_attention_data = cls._raw_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()

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
            # latency = SOL / util. The query SOL uses the real window w (get_sol caps
            # kv_len at the window); the UTIL is borrowed by slice -- exact window first,
            # then window=0 (full attention). The window axis is collected per-model
            # (scattered), so an uncollected window borrows full-attention util: decode
            # is memory-bound KV read, ~window-independent in efficiency, and SOL absorbs
            # the smaller windowed KV span.
            sol_time = get_sol(b, s, n, n_kv, h, w, kvcache_quant_mode)[0]
            n_kv_lookup = n_kv if n_kv != n else 0

            def _own_grid(slice_window):
                def _slice():
                    cls.load_data(database)
                    wrapper = getattr(database, "_raw_generation_attention_data", None)
                    if wrapper is None:
                        raise PerfDataNotAvailableError("Raw generation attention data is not loaded.")
                    wrapper.raise_if_not_loaded()
                    return util_empirical.require_data_slice(
                        wrapper,
                        kvcache_quant_mode,
                        n_kv_lookup,
                        h,
                        slice_window,
                    )

                def _sol(c):  # c = (n, b, s)
                    nkv = c[0] if n_kv_lookup == 0 else n_kv_lookup
                    return get_sol(c[1], c[2], c[0], nkv, h, slice_window, kvcache_quant_mode)[0]

                return util_empirical.grid_for(
                    (
                        "gen_attn",
                        database.system,
                        database.backend,
                        database.version,
                        kvcache_quant_mode.name,
                        n_kv_lookup,
                        h,
                        slice_window,
                    ),
                    _slice,
                    _sol,
                    depth=3,
                )

            for slice_window in [w, 0] if w > 0 else [w]:
                grid = _own_grid(slice_window)
                if grid is not None and grid.samples:
                    latency, _ = util_empirical.estimate(sol_time, (n, b, s), grid)
                    return latency
                # Cross-head_size transfer (XSHAPE, decode): borrow the nearest collected
                # head_size's util. Decode util is ~head_size-independent (memory-bound
                # KV read), so util_scale stays 1.0 -- no prefill-style correction.
                ref_grid, _ref_hs = (
                    _gen_headsize_ref_grid(database, kvcache_quant_mode, n_kv_lookup, h, slice_window, get_sol)
                    if common.TransferKind.XSHAPE in database.transfer_policy
                    else (None, None)
                )
                if ref_grid is not None:
                    latency, _ = util_empirical.estimate(sol_time, (n, b, s), ref_grid, provenance="xshape")
                    return latency

            latency, _ = util_empirical.estimate(sol_time, (n, b, s), None)
            return latency

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

            # Use the real windowed slice when present (more accurate than a window=0 +
            # SOL reconstruction); when absent/too sparse, perf_interp raises and
            # HYBRID/EMPIRICAL fall back to get_empirical's window=0 + SOL derivation.
            attention_dict = util_empirical.require_data_slice(
                data_wrapper,
                kvcache_quant_mode,
                n_kv_lookup,
                head_size,
                window_size,
            )
            # Generation is ~linear in seq -> RAW grid resolve on the raw table.
            # The +-10% seq-sample averaging is op-level smoothing (decode s
            # drifts across a request) and is kept as-is.
            config = perf_interp.generation_attention_config(
                sol_fn=lambda n_v, b_v, s_v: get_sol(
                    b_v,
                    s_v,
                    n_v,
                    (n_v if n_kv_lookup == 0 else n_kv_lookup),
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                )[0]
            )
            s_min = max(1, int(s * 0.9))
            s_max = max(s_min, int(s * 1.1))
            sample_cnt = 5
            s_samples = [s_min + (s_max - s_min) * i // (sample_cnt - 1) for i in range(sample_cnt)]

            latency_sum = 0.0
            energy_sum = 0.0
            for s_i in s_samples:
                r = perf_interp.query(config, attention_dict, n, b, s_i)
                latency_sum += perf_interp.get_value(r, "latency")
                energy_sum += perf_interp.get_value(r, "energy")

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


class EncoderAttention(Operation):
    """
    Non-causal encoder attention: full N^2, MHA, no KV cache, optional partial RoPE.

    Used to model bidirectional encoders — ViT (vision), audio encoders, and any
    other omni-modal encoder where the kernel runs full N^2 attention without a
    causal mask and without writing a KV cache. The optional
    ``partial_rotary_factor`` accounts for partial-rotation RoPE variants such as
    Qwen3-VL (factor=0.5, rotating half of head_dim). Defaults to 0.0 (no RoPE),
    matching CLIP / SigLIP / Whisper; set to 0.5 / 1.0 only for RoPE encoders.

    Owns ``_data_cache: {key: LoadedOpData}`` for the encoder attention CSV.
    Schema is simpler than context attention: MHA only (no n_kv), no KV cache
    (no kvcache_quant_mode), no sliding window. No SOL clamp. Grid extrapolation
    resolves on the raw grid via perf_interp.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        head_size: int,
        fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.bfloat16,
        partial_rotary_factor: float = 0.0,
    ) -> None:
        super().__init__(name, scale_factor)
        # Encoder kernels currently only have bfloat16 perf data;
        if fmha_quant_mode != common.FMHAQuantMode.bfloat16:
            raise ValueError(f"EncoderAttention only supports FMHAQuantMode.bfloat16, got {fmha_quant_mode}")
        if not 0.0 <= partial_rotary_factor <= 1.0:
            raise ValueError(f"partial_rotary_factor must be in [0.0, 1.0], got {partial_rotary_factor}")
        self._n = num_heads
        self._head_size = head_size
        self._fmha_quant_mode = fmha_quant_mode
        self._partial_rotary_factor = partial_rotary_factor
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads encoder_attention CSV into the class cache,
        applies grid extrapolation, binds ``database._encoder_attention_data``.
        """
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.encoder_attention.value
            )
            sources = database._build_op_sources(PerfDataFilename.encoder_attention, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_encoder_attention_data(sources), PerfDataFilename.encoder_attention, primary_path
            )

            # No load-time grid pre-expansion: queries resolve on the RAW grid via perf_interp.
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_encoder_attention_data" not in database.__dict__:
            database._encoder_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_encoder_attention)
    # ------------------------------------------------------------------

    @classmethod
    def _query_encoder_attention_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        n: int,
        head_size: int,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query encoder attention table. Verbatim port of the legacy body."""

        def get_sol(
            b: int, s: int, n: int, h: int, fmha_quant_mode: common.FMHAQuantMode
        ) -> tuple[float, float, float]:
            # Non-causal full N^2: no /2 for causality
            ops = 2 * b * s * s * n * h * 2  # 2 for fma, 2 for q*k^t + *v
            # Encoder has no KV cache read; Q/K/V are all read once
            mem_bytes = 2 * b * (3 * n * s * h + n * s * h)  # Q/K/V read + output write, bf16
            sol_math = ops / database.system_spec["gpu"]["bfloat16_tc_flops"] * 1000 / fmha_quant_mode.value.compute
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(b: int, s: int, n: int, h: int, fmha_quant_mode: common.FMHAQuantMode) -> float:
            # SOL / util, util read best-effort from collected encoder-attention
            # data (the (n, s, b) grid for this slice); raises EmpiricalNotImplementedError if none.
            sol_time = get_sol(b, s, n, h, fmha_quant_mode)[0]

            def _slice():
                cls.load_data(database)
                wrapper = database._encoder_attention_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(wrapper, fmha_quant_mode, h)

            def _sol(c):  # c = (n, s, b)
                return get_sol(c[2], c[1], c[0], h, fmha_quant_mode)[0]

            grid = util_empirical.grid_for(
                ("encoder_attn", database.system, database.backend, database.version, fmha_quant_mode.name, h),
                _slice,
                _sol,
                depth=3,
            )
            latency, _ = util_empirical.estimate(sol_time, (n, s, b), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, n, head_size, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, n, head_size, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, n, head_size, fmha_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)
        data_wrapper = database._encoder_attention_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            attention_dict = util_empirical.require_data_slice(data_wrapper, fmha_quant_mode, head_size)
            # Encoder is full N^2 (~seq^2 along seq, ~linear along heads/batch):
            # context_attention_config = sqrt on the seq axis only, raw elsewhere.
            config = perf_interp.context_attention_config(
                sol_fn=lambda n_v, s_v, b_v: get_sol(b_v, s_v, n_v, head_size, fmha_quant_mode)[0]
            )
            result = perf_interp.query(config, attention_dict, n, s, b)
            latency = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, n, head_size, fmha_quant_mode),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query encoder attention data for {b=}, {s=}, {n=}, {head_size=}, {fmha_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query encoder attention latency with energy data."""
        batch_size = kwargs.get("batch_size")
        seq_len = kwargs.get("s")

        result = database.query_encoder_attention(
            batch_size,
            seq_len,
            self._n,
            self._head_size,
            self._fmha_quant_mode,
        )

        # Partial RoPE: factor=1.0 -> full rotation (full Q+K read/write),
        # factor=0.5 -> half head_dim rotated (Qwen3-VL), factor=0.0 -> no RoPE.
        if self._partial_rotary_factor > 0:
            qk_num = self._n * self._head_size  # MHA: q_num == k_num
            # Q + K bytes (bf16) scaled by total tokens, so RoPE overhead grows with workload size.
            qk_bytes = 2 * (qk_num * 2) * (batch_size * seq_len)
            apply_rope_latency = self._partial_rotary_factor * 2 * database.query_mem_op(qk_bytes)
            result += apply_rope_latency * 1.1  # correction factor

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


def load_context_attention_data(context_attention_file):
    """
    Load the context attention data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(context_attention_file)
    if rows is None:
        logger.debug(f"Context attention data file {context_attention_file} not found.")
        return None
    context_attention_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                )
            )
        )
    )

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (context_attention) - power will default to 0.0")

    for row in rows:
        try:
            window_size = row["window_size"]
        except KeyError:  # catch potential error for backward comptability
            window_size = 0
        quant_mode, kv_cache_dtype, b, s, n, kv_n, head_size, latency = (
            row["attn_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["num_key_value_heads"],
            row["head_dim"],
            row["latency"],
        )
        b = int(b)
        s = int(s)
        n = int(n)
        kv_n = int(kv_n)
        head_size = int(head_size)
        window_size = int(window_size)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads.
        # Use kv_n = 0 to mean n_kv == n.
        kv_n = 0 if n == kv_n else kv_n

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][window_size][n][s][b]
            logger.debug(
                f"value conflict in context attention data: {quant_mode} {kv_cache_dtype} "
                f"{head_size} {window_size} {kv_n} {n} {s}"
            )
        except KeyError:
            # Store all three values
            context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][window_size][n][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return context_attention_data


def load_generation_attention_data(generation_attention_file):
    """
    Load the generation attention data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(generation_attention_file)
    if rows is None:
        logger.debug(f"Generation attention data file {generation_attention_file} not found.")
        return None
    generation_attention_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
        )
    )

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (generation_attention) - power will default to 0.0")

    for row in rows:
        try:
            window_size = row["window_size"]
        except KeyError:
            window_size = 0
        quant_mode, kv_cache_dtype, b, s, n, kv_n, head_size, step, latency = (  # noqa: F841
            row["attn_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["num_key_value_heads"],
            row["head_dim"],
            row["step"],
            row["latency"],
        )
        b = int(b)
        s = int(s)
        n = int(n)
        kv_n = int(kv_n)
        head_size = int(head_size)
        window_size = int(window_size)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads.
        # Use kv_n = 0 to mean n_kv == n.
        kv_n = 0 if n == kv_n else kv_n
        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s]
            logger.debug(
                f"value conflict in generation attention data: {kv_cache_dtype} {kv_n} "
                f"{head_size} {window_size} {n} {b}"
            )
        except KeyError:
            # Store all three values
            generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return generation_attention_data


def load_encoder_attention_data(encoder_attention_file):
    """
    Load the non-causal encoder attention data (ViT, audio encoder, etc.).

    Schema is intentionally simplified vs. context attention:
    - MHA only (n_kv == n), so no n_kv dimension
    - No KV cache (encoder is single-pass), so no kv_cache_dtype dimension
    - No sliding window, so no window_size dimension

    Returns:
        dict: Nested dict [fmha_quant_mode][head_size][n][s][b] -> {latency, power, energy}.
    """
    rows = _read_filtered_rows(encoder_attention_file)
    if rows is None:
        logger.debug(f"Encoder attention data file {encoder_attention_file} not found.")
        return None
    encoder_attention_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    )

    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (encoder_attention) - power will default to 0.0")

    for row in rows:
        quant_mode, b, s, n, head_size, latency = (
            row["attn_dtype"],
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["head_dim"],
            row["latency"],
        )
        b = int(b)
        s = int(s)
        n = int(n)
        head_size = int(head_size)
        latency = float(latency)

        power = float(row.get("power", 0.0))
        energy = power * latency

        quant_mode = common.FMHAQuantMode[quant_mode]

        try:
            encoder_attention_data[quant_mode][head_size][n][s][b]
            logger.debug(f"value conflict in encoder attention data: {quant_mode} {head_size} {n} {s} {b}")
        except KeyError:
            encoder_attention_data[quant_mode][head_size][n][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return encoder_attention_data
