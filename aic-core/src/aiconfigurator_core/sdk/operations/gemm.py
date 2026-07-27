# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM operation and its associated CSV-backed data (compute_scale, scale_matrix).

Stage 2 of ISSUE-05/AIC-542: GEMM now owns its three CSV tables, SOL
correction, and grid extrapolation. ``PerfDatabase.query_gemm /
query_compute_scale / query_scale_matrix`` delegate here.

Lazy-load Pattern A: ``query()`` (and the delegating ``_query_*_table``
classmethods) trigger ``load_data`` on cache miss. ``_data_cache`` /
``_compute_scale_cache`` / ``_scale_matrix_cache`` are keyed by
``(systems_root, system, backend, version, enable_shared_layer)`` so the
same op class serves multiple databases in one process. ``systems_root``
is part of the key because test fixtures swap to a fresh ``tmp_path``
between tests and must get distinct cache entries.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from aiconfigurator_core.sdk import common, perf_interp
from aiconfigurator_core.sdk.errors import (
    EmpiricalNotImplementedError,
    InterpolationDataNotAvailableError,
    PerfDataNotAvailableError,
)
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


class _ZeroAwareDeltaLookup:
    """Preprocessed nearest-neighbour lookup for one immutable delta table.

    The table is retained by strong reference.  Besides keeping its normalized
    arrays alive, this makes an ``id(table)`` cache key safe from object-id
    reuse; callers still verify identity before accepting a cached lookup.
    """

    def __init__(self, table) -> None:
        self.table = table
        observations: list[tuple[tuple[float, float], float]] = []
        for coords, leaf in util_empirical.iter_grid(table, depth=2):
            latency = util_empirical.leaf_latency(leaf)
            if latency is not None and latency >= 0:
                observations.append(((float(coords[0]), float(coords[1])), float(latency)))

        self._coords = np.asarray([coords for coords, _latency in observations], dtype=float)
        self._latencies = np.asarray([latency for _coords, latency in observations], dtype=float)
        if not observations:
            return

        log_coords = np.log(np.maximum(self._coords, 1e-9))
        self._mins = log_coords.min(axis=0)
        spans = log_coords.max(axis=0) - self._mins
        self._spans = np.where(spans > 0, spans, 1.0)
        self._norm_coords = (log_coords - self._mins) / self._spans

    def estimate(self, query: tuple[float, float], sol_fn) -> float:
        if not self._latencies.size:
            raise EmpiricalNotImplementedError(f"No empirical compute_scale delta data is available at query={query}.")

        norm_query = (np.log(np.maximum(np.asarray(query, dtype=float), 1e-9)) - self._mins) / self._spans
        nearest = int(np.square(self._norm_coords - norm_query).sum(axis=1).argmin())
        reference_coords = self._coords[nearest]
        reference_latency = float(self._latencies[nearest])

        util_empirical.note_provenance("empirical")
        if reference_latency == 0:
            return 0.0

        reference_sol = sol_fn(*reference_coords)
        query_sol = sol_fn(*query)
        if reference_sol <= 0 or query_sol <= 0:
            raise EmpiricalNotImplementedError(
                f"No positive SOL reference is available for compute_scale at query={query}."
            )
        return query_sol / (reference_sol / reference_latency)


def _estimate_zero_aware_delta(
    table,
    query: tuple[float, float],
    sol_fn,
    cache: dict[int, _ZeroAwareDeltaLookup],
) -> float:
    """Estimate a non-negative latency *delta* without discarding zeroes.

    ``compute_scale`` stores ``max(dynamic_quant - static_quant, 0)``.  Zero is
    therefore a measured, meaningful delta, not a missing/invalid latency.  A
    normal util grid cannot represent it (``SOL / 0``), and dropping zeroes can
    make one positive noise sample become the reference for the entire table.

    Select the nearest point on the complete 2-D grid first.  A selected zero
    stays zero; a positive point still uses frozen utilization so extrapolation
    scales with the query's amount of work.
    """
    key = id(table)
    lookup = cache.get(key)
    if lookup is None or lookup.table is not table:
        lookup = _ZeroAwareDeltaLookup(table)
        cache[key] = lookup
    return lookup.estimate(query, sol_fn)


class GEMM(Operation):
    """
    GEMM operation with power tracking.

    Owns three CSV-backed tables:
    - ``_data_cache``: gemm latency/energy keyed by ``quant_mode -> m -> n -> k``
    - ``_compute_scale_cache``: compute_scale latency/energy keyed by ``quant_mode -> m -> k``
    - ``_scale_matrix_cache``: scale_matrix latency/energy keyed by ``quant_mode -> m -> k``

    All three are class-level dicts keyed by
    ``(systems_root, system, backend, version, enable_shared_layer)``.
    """

    # Per-op subclass overrides of Operation._data_cache. Keyed by
    # (systems_root, system, backend, version, enable_shared_layer).
    _data_cache: ClassVar[dict] = {}
    _compute_scale_cache: ClassVar[dict] = {}
    _scale_matrix_cache: ClassVar[dict] = {}
    _compute_scale_delta_lookup_cache: ClassVar[dict[int, _ZeroAwareDeltaLookup]] = {}
    _CP_AWARE: ClassVar[bool] = True  # query divides x (token count) by self._seq_split

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=kwargs.get("seq_split", 1))
        self._n = n
        self._k = k
        self._quant_mode = quant_mode
        self._weights = self._n * self._k * quant_mode.value.memory
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)
        self._low_precision_input = kwargs.get("low_precision_input", False)

    # ------------------------------------------------------------------
    # Data ownership: load + cache + clear
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        """Cache key uniquely identifying the loaded data set.

        ``systems_root`` is included so test fixtures that swap to a fresh
        ``tmp_path`` between tests get distinct entries (otherwise the
        shared-layer test suite collides). ``enable_shared_layer`` is also
        part of the key because HYBRID unions sibling-row inheritance, so
        a SILICON-only load and a HYBRID load produce different dicts.
        """
        return (
            database.systems_root,
            database.system,
            database.backend,
            database.version,
            database.enable_shared_layer,
        )

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. On cache miss: parses the three CSVs into the class-
        level caches, applies SOL correction + grid extrapolation directly
        on those canonical wrappers, and records the load. Always: binds
        ``database._gemm_data``/``_compute_scale_data``/``_scale_matrix_data``
        to the cached wrappers.

        Tests that have already set those instance attributes (e.g.
        ``db._gemm_data = LoadedOpData(...)``) are respected — the binds
        below are gated on ``"_gemm_data" not in database.__dict__`` so
        intentional overrides survive."""
        import os

        # Lazy import to avoid the circular dependency between gemm.py and
        # perf_database.py (perf_database delegates to GEMM at query time).
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache or key not in cls._compute_scale_cache or key not in cls._scale_matrix_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            def _load(filename_enum, loader):
                primary_path = resolve_op_data_path(
                    system_data_root, database.backend, database.version, filename_enum.value
                )
                sources = database._build_op_sources(filename_enum, primary_path, system_data_root)
                return LoadedOpData(loader(sources), filename_enum, primary_path)

            # Load all three into locals first so a loader failure on the second
            # or third file doesn't leave the cache half-populated (which would
            # let a subsequent ``key in cls._data_cache`` early-out skip past
            # the missing siblings and crash downstream).
            gemm_loaded = _load(PerfDataFilename.gemm, load_gemm_data)
            compute_scale_loaded = _load(PerfDataFilename.compute_scale, load_compute_scale_data)
            scale_matrix_loaded = _load(PerfDataFilename.scale_matrix, load_scale_matrix_data)

            # Clamp the CANONICAL class-cache values to the SOL floor directly
            # (not via ``database._gemm_data``) so a pre-set test override
            # can't leave the cached wrapper uncorrected — a later DB sharing
            # the same cache key would otherwise bind unclamped data.
            #
            # No load-time grid pre-expansion: queries resolve on the RAW table
            # via perf_interp (site curves + util-hold + nearest-site transfer),
            # so rectangularizing the scattered (n, k) shapes is both redundant
            # and harmful (it mangles asymmetric dense-m sweeps).
            cls._correct_sol(database, gemm_loaded)

            # All three loads + correction + extrapolation succeeded — commit
            # atomically so partially-populated cache state can never be observed.
            cls._data_cache[key] = gemm_loaded
            cls._compute_scale_cache[key] = compute_scale_loaded
            cls._scale_matrix_cache[key] = scale_matrix_loaded

            cls._record_load()

        # Bind instance attrs (respect intentional test pre-overrides).
        if "_gemm_data" not in database.__dict__:
            database._gemm_data = cls._data_cache[key]
        if "_compute_scale_data" not in database.__dict__:
            database._compute_scale_data = cls._compute_scale_cache[key]
        if "_scale_matrix_data" not in database.__dict__:
            database._scale_matrix_data = cls._scale_matrix_cache[key]
        return

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all three GEMM caches plus base-class state."""
        cls._data_cache.clear()
        cls._compute_scale_cache.clear()
        cls._scale_matrix_cache.clear()
        cls._compute_scale_delta_lookup_cache.clear()
        perf_interp.clear_caches()  # engine site indexes are keyed off these tables
        query = cls.__dict__.get("query")
        if query is not None and hasattr(query, "cache_clear"):
            query.cache_clear()

    @classmethod
    def supported_quant_modes(cls, database: PerfDatabase) -> set:
        """Return the quant modes for which loaded GEMM data is available.

        Triggers ``load_data`` on first call so the answer reflects what
        actually loaded for this database."""
        cls.load_data(database)
        gemm_data = cls._data_cache.get(cls._cache_key(database))
        if gemm_data is None or not gemm_data.loaded:
            return set()
        return set(gemm_data.keys())

    # ------------------------------------------------------------------
    # Static helpers (shared with perf_database.py callers)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_quant_tc_flops(system_spec, quant_mode) -> float:
        """Resolve actual tensor-core FLOPS for a given quant mode.

        Maps the quant mode's compute factor (1/2/4) to the corresponding
        ``*_tc_flops`` entry in the system GPU spec. Falls back to
        ``bfloat16_tc_flops * compute_factor`` when the spec entry is
        missing.

        Static so non-GEMM SOL callers in perf_database.py (DSV4, MLA,
        attention) can delegate without an instance. Until ISSUE-16
        retires those wrappers, ``PerfDatabase._get_quant_tc_flops``
        forwards here."""
        compute_to_flops_key = {1: "bfloat16_tc_flops", 2: "fp8_tc_flops", 4: "fp4_tc_flops"}
        gpu = system_spec["gpu"]
        key = compute_to_flops_key.get(quant_mode.value.compute)
        if key is not None and key in gpu:
            return gpu[key]
        return gpu["bfloat16_tc_flops"] * quant_mode.value.compute

    @staticmethod
    def _normalize_gemm_quant_mode_for_table(
        quant_mode: common.GEMMQuantMode,
    ) -> common.GEMMQuantMode:
        """Normalize modeled GEMM quant modes for perf table lookup.

        ``fp8_static`` is modeled from the dynamic ``fp8`` GEMM row plus
        separately collected activation-quantization overhead tables.
        """
        if quant_mode == common.GEMMQuantMode.fp8_static:
            return common.GEMMQuantMode.fp8
        return quant_mode

    # ------------------------------------------------------------------
    # SOL correction (formerly in PerfDatabase._correct_data)
    # ------------------------------------------------------------------

    @classmethod
    def _correct_sol(cls, database: PerfDatabase, gemm_data=None) -> None:
        """Clamp loaded GEMM latencies to be no less than the SOL bound.

        Mirrors the legacy ``_correct_data`` block — for each table entry,
        if SOL latency exceeds the measured latency, the measured value is
        raised to SOL (preserving power/energy fields unchanged).

        ``gemm_data`` defaults to ``database._gemm_data`` (the instance
        attribute) so the backward-compat call from ``PerfDatabase._correct_data``
        operates on whatever the test has injected. ``load_data`` passes the
        canonical class-cache value explicitly so corrections always land on
        the wrapper future databases inherit."""
        if gemm_data is None:
            gemm_data = getattr(database, "_gemm_data", None)
        # Defensive: a test may have set ``db._gemm_data`` to a non-LoadedOpData
        # sentinel (e.g. ``object()``) for an override-respect check; the fallback
        # path must not crash on the missing ``.loaded`` attribute.
        if gemm_data is None or not getattr(gemm_data, "loaded", False):
            return

        for quant_mode in gemm_data:
            for m in gemm_data[quant_mode]:
                for n in gemm_data[quant_mode][m]:
                    for k in gemm_data[quant_mode][m][n]:
                        sol = cls._query_gemm_table(
                            database, m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL
                        )
                        data = gemm_data[quant_mode][m][n][k]
                        current_latency = data["latency"] if isinstance(data, dict) else data
                        if sol > current_latency:
                            logger.debug(
                                f"gemm quant {quant_mode} m{m} n{n} k{k}: sol {sol} > perf_db {current_latency}"
                            )
                            if isinstance(data, dict):
                                gemm_data[quant_mode][m][n][k]["latency"] = float(max(sol, current_latency))
                            else:
                                gemm_data[quant_mode][m][n][k] = float(max(sol, current_latency))

    # ------------------------------------------------------------------
    # Grid extrapolation (formerly in PerfDatabase.__init__)
    # ------------------------------------------------------------------

    # GEMM extrapolation target grid (lifted verbatim from perf_database.py
    # so behavior stays bit-identical).
    _EXTRAPOLATION_TARGET_X: ClassVar[list] = [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        128,
        160,
        192,
        224,
        256,
        320,
        384,
        448,
        512,
        640,
        768,
        896,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        131072,
        524288,
        1048576,
        2097152 * 8,
    ]  # num_tokens

    _EXTRAPOLATION_TARGET_Y: ClassVar[list] = [
        32,
        64,
        128,
        256,
        512,
        768,
        1024,
        1536,
        2048,
        2560,
        3072,
        3584,
        4096,
        5120,
        6144,
        7168,
        8192,
        10240,
        12288,
        14336,
        16384,
        20480,
        24576,
        28672,
        32768,
        40960,
        49152,
        57344,
        65536,
        131072,
        262144,
    ]  # to fit vocab gemm

    # ------------------------------------------------------------------
    # Table query classmethods (formerly PerfDatabase.query_*)
    # ------------------------------------------------------------------

    @classmethod
    def _query_gemm_table(
        cls,
        database: PerfDatabase,
        m: int,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query GEMM table — preserves PR #721 exact-hit → 1D → 3D fast path."""

        def get_sol(m_v: int, n_v: int, k_v: int, qm: common.GEMMQuantMode) -> tuple[float, float, float]:
            tc_flops = cls._get_quant_tc_flops(database.system_spec, qm)
            sol_math = 2 * m_v * n_v * k_v / tc_flops * 1000
            sol_mem = (
                qm.value.memory * (m_v * n_v + m_v * k_v + n_v * k_v) / database.system_spec["gpu"]["mem_bw"] * 1000
            )
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(m_v: int, n_v: int, k_v: int, qm: common.GEMMQuantMode) -> float:
            # SOL / util, where util is read best-effort from this op's own
            # collected data; raises EmpiricalNotImplementedError when no data.
            sol_time = get_sol(m_v, n_v, k_v, qm)[0]
            tqm = cls._normalize_gemm_quant_mode_for_table(qm)

            def _slice():
                cls.load_data(database)
                wrapper = database._gemm_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(wrapper, tqm)  # m -> n -> k -> leaf

            grid = util_empirical.grid_for(
                ("gemm", database.system, database.backend, database.version, tqm.name),
                _slice,
                lambda c: get_sol(c[0], c[1], c[2], qm)[0],
                depth=3,
            )
            latency, _ = util_empirical.estimate(sol_time, (m_v, n_v, k_v), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode

        table_quant_mode = cls._normalize_gemm_quant_mode_for_table(quant_mode)

        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, n, k, quant_mode)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, n, k, quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, n, k, quant_mode), energy=0.0, source="empirical")

        # SILICON or HYBRID mode — use database. ``load_data`` is idempotent;
        # it populates the class cache and binds the instance attrs only when
        # the test hasn't already pre-set them.
        cls.load_data(database)
        gemm_data_wrapper = database._gemm_data

        def get_silicon():
            def _to_performance_result(result, *, source: str = "silicon"):
                """Normalize GEMM table entries into a PerformanceResult.

                Interpolated/extrapolated GEMM values are still derived from
                silicon table data; only explicit formula fallbacks are
                tagged as empirical.

                If ``result`` is already a ``PerformanceResult``, return it
                unchanged so upstream attribution (e.g. ``"empirical"`` /
                ``"mixed"`` set by an inner ``_query_silicon_or_hybrid`` /
                ``_interp_pr`` call) is preserved instead of being silently
                overwritten with the ``source`` default."""
                if isinstance(result, PerformanceResult):
                    return result
                if isinstance(result, dict):
                    return PerformanceResult(result["latency"], energy=result.get("energy", 0.0), source=source)
                return PerformanceResult(result, energy=0.0, source=source)

            gemm_data_wrapper.raise_if_not_loaded()
            if table_quant_mode not in gemm_data_wrapper:
                supported = sorted([q.name for q in gemm_data_wrapper])
                raise PerfDataNotAvailableError(
                    "GEMM perf data not available for requested quant mode. "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                    f"quant_mode='{quant_mode.name}'. "
                    f"Supported gemm modes: {supported}"
                )

            gemm_data = gemm_data_wrapper[table_quant_mode]

            # Resolve on the raw table: exact hit -> the collected (n, k) site's
            # own m-curve -> nearest-site util transfer -> util-hold beyond the
            # sweep. See sdk/perf_interp/config.py for the design.
            config = perf_interp.gemm_config(sol_fn=lambda m_v, n_v, k_v: get_sol(m_v, n_v, k_v, quant_mode)[0])
            try:
                result = perf_interp.query(config, gemm_data, m, n, k)
            except InterpolationDataNotAvailableError as exc:
                raise PerfDataNotAvailableError(
                    "GEMM perf data not available for requested shape. "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                    f"quant_mode='{quant_mode.name}', m={m}, n={n}, k={k}."
                ) from exc
            return _to_performance_result(result)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(m, n, k, quant_mode),
            database_mode=database_mode,
            error_msg=f"Failed to query gemm data for {m=}, {n=}, {k=}, {quant_mode=}",
        )

    @classmethod
    def _query_compute_scale_table(
        cls,
        database: PerfDatabase,
        m: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query compute_scale (dynamic minus static quantization) table."""

        def get_sol(m_v: int, k_v: int) -> tuple[float, float, float]:
            sol_mem = 2 * m_v * k_v / database.system_spec["gpu"]["mem_bw"] * 1000.0
            sol_time = sol_mem
            return sol_time, 0, sol_mem

        table_quant_mode = cls._normalize_gemm_quant_mode_for_table(quant_mode)

        def get_empirical(m_v: int, k_v: int) -> float:
            # compute_scale is a non-negative latency delta.  Unlike an actual
            # kernel latency, zero is meaningful and must participate in the
            # nearest-point decision (see _estimate_zero_aware_delta).
            try:
                cls.load_data(database)
                wrapper = database._compute_scale_data
                wrapper.raise_if_not_loaded()
                table = util_empirical.require_data_slice(wrapper, table_quant_mode)
            except PerfDataNotAvailableError as exc:
                raise EmpiricalNotImplementedError(
                    f"No empirical compute_scale data is available for m={m_v}, k={k_v}."
                ) from exc

            return _estimate_zero_aware_delta(
                table,
                (float(m_v), float(k_v)),
                lambda m_ref, k_ref: get_sol(m_ref, k_ref)[0],
                cls._compute_scale_delta_lookup_cache,
            )

        if database_mode is None:
            database_mode = database._default_database_mode

        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, k)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, k)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, k), energy=0.0, source="empirical")

        cls.load_data(database)
        compute_scale_wrapper = database._compute_scale_data

        def get_silicon():
            compute_scale_wrapper.raise_if_not_loaded()
            if table_quant_mode not in compute_scale_wrapper:
                supported = sorted([q.name for q in compute_scale_wrapper])
                from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

                raise PerfDataNotAvailableError(
                    "Compute scale perf data not available for requested quant mode. "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                    f"quant_mode='{quant_mode.name}'. "
                    f"Supported modes: {supported}"
                )
            table = compute_scale_wrapper[table_quant_mode]
            # Clamp into the collected range FIRST (preserving the legacy
            # contract), then resolve the interior on the engine (RAW 2-axis).
            m_keys = sorted(table.keys())
            m_c = max(m_keys[0], min(int(m), m_keys[-1]))
            k_min = min(min(row) for row in table.values() if row)
            k_max = max(max(row) for row in table.values() if row)
            k_c = max(k_min, min(int(k), k_max))
            config = perf_interp.OpInterpConfig(
                axes=("m", "k"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda m_v, k_v: get_sol(m_v, k_v)[0],
            )
            result = perf_interp.query(config, table, m_c, k_c)
            interpolated = database._interp_pr(
                perf_interp.get_value(result, "latency"),
                energy=perf_interp.get_value(result, "energy"),
            )
            # compute_scale is a quantization-overhead DELTA: beyond the grid it
            # is deliberately held FLAT at the clamped boundary (legacy contract).
            return interpolated

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(m, k),
            database_mode=database_mode,
            error_msg=f"Failed to query compute_scale data for {m=}, {k=}, {quant_mode=}",
        )

    @classmethod
    def _query_scale_matrix_table(
        cls,
        database: PerfDatabase,
        m: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query scale_matrix (static quantization) table."""

        def get_sol(m_v: int, k_v: int) -> tuple[float, float, float]:
            sol_mem = 3 * m_v * k_v / database.system_spec["gpu"]["mem_bw"] * 1000.0
            sol_time = sol_mem
            return sol_time, 0, sol_mem

        table_quant_mode = cls._normalize_gemm_quant_mode_for_table(quant_mode)

        def get_empirical(m_v: int, k_v: int) -> float:
            # SOL / util, util read best-effort from collected scale_matrix data
            # (the (m, k) grid for this quant); raises EmpiricalNotImplementedError if none.
            sol_time = get_sol(m_v, k_v)[0]

            def _slice():
                cls.load_data(database)
                wrapper = database._scale_matrix_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(wrapper, table_quant_mode)

            grid = util_empirical.grid_for(
                ("scale_matrix", database.system, database.backend, database.version, table_quant_mode.name),
                _slice,
                lambda c: get_sol(c[0], c[1])[0],
                depth=2,
            )
            latency, _ = util_empirical.estimate(sol_time, (float(m_v), float(k_v)), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode

        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, k)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, k)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, k), energy=0.0, source="empirical")

        cls.load_data(database)
        scale_matrix_wrapper = database._scale_matrix_data

        def get_silicon():
            scale_matrix_wrapper.raise_if_not_loaded()
            if table_quant_mode not in scale_matrix_wrapper:
                supported = sorted([q.name for q in scale_matrix_wrapper])
                from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

                raise PerfDataNotAvailableError(
                    "Scale matrix perf data not available for requested quant mode. "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                    f"quant_mode='{quant_mode.name}'. "
                    f"Supported modes: {supported}"
                )
            table = scale_matrix_wrapper[table_quant_mode]
            # Clamp into the collected range FIRST (preserving the legacy
            # contract), then resolve the interior on the engine (RAW 2-axis).
            m_keys = sorted(table.keys())
            m_c = max(m_keys[0], min(int(m), m_keys[-1]))
            k_min = min(min(row) for row in table.values() if row)
            k_max = max(max(row) for row in table.values() if row)
            k_c = max(k_min, min(int(k), k_max))
            config = perf_interp.OpInterpConfig(
                axes=("m", "k"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda m_v, k_v: get_sol(m_v, k_v)[0],
            )
            result = perf_interp.query(config, table, m_c, k_c)
            interpolated = database._interp_pr(
                perf_interp.get_value(result, "latency"),
                energy=perf_interp.get_value(result, "energy"),
            )
            if m_c == int(m) and k_c == int(k):
                return interpolated
            # Outside the grid, freeze utilization at the clamped boundary:
            # L(q) = L(boundary) * SOL(q)/SOL(boundary) (a real memory kernel,
            # unlike the compute_scale delta above).
            boundary_sol = get_sol(m_c, k_c)[0]
            query_sol = get_sol(int(m), int(k))[0]
            ratio = query_sol / boundary_sol
            return PerformanceResult(
                latency=float(interpolated) * ratio,
                energy=interpolated.energy * ratio,
                source=getattr(interpolated, "source", "silicon"),
            )

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(m, k),
            database_mode=database_mode,
            error_msg=f"Failed to query scale_matrix data for {m=}, {k=}, {quant_mode=}",
        )

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query GEMM latency with energy data.

        For `fp8_static` quant mode, subtracts compute_scale overhead.
        For GEMMs marked as low-precision input under `fp8_static`, also subtract scale_matrix.

        Returns:
            PerformanceResult: Behaves like float (scaled latency in ms).
                              Energy data accessible via .energy attribute.
                              Power can be derived as energy/latency.
        """
        x = kwargs.get("x")
        x //= self._scale_num_tokens
        x = -(-x // self._seq_split)  # CP: per-rank token count (ceil = busiest rank)
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode
        is_fp8_static = quant_mode == common.GEMMQuantMode.fp8_static
        latency_floor = 0.0

        # Query with energy
        result = database.query_gemm(x, self._n, self._k, quant_mode)
        latency = float(result)
        energy = result.energy
        source = getattr(result, "source", "silicon")

        # Static-FP8 GEMM is modeled from the dynamic FP8 base measurement
        # across backends; subtract the separately collected activation-
        # quantization pieces for BF16-input and low-precision-input cases.
        if is_fp8_static:
            compute_scale_result = database.query_compute_scale(x, self._k, quant_mode)
            latency -= float(compute_scale_result)
            energy -= compute_scale_result.energy
            sub_src = getattr(compute_scale_result, "source", "silicon")
            if sub_src != source:
                source = "mixed"
            if self._low_precision_input:
                scale_matrix_result = database.query_scale_matrix(x, self._k, quant_mode)
                latency -= float(scale_matrix_result)
                energy -= scale_matrix_result.energy
                sub_src = getattr(scale_matrix_result, "source", "silicon")
                if sub_src != source:
                    source = "mixed"
            # fp8_static is modeled from dynamic FP8 plus overhead tables, so
            # expose source="estimated" instead of measured silicon.
            source = "estimated"

            # The subtraction leaves a path that still contains the GEMM
            # (GEMM-only for low-precision input, static-quant + GEMM otherwise).
            # Independently interpolated component tables can cross, but that
            # path cannot be faster than the GEMM's own roofline bound. Keep the
            # physical SOL floor instead of turning a negative residual into 0.
            latency_floor = float(
                database.query_gemm(
                    x,
                    self._n,
                    self._k,
                    quant_mode,
                    database_mode=common.DatabaseMode.SOL,
                )
            )

        # Latency has a physical roofline floor. Energy has no analogous SOL
        # model here, so retain the existing conservative non-negative clamp
        # rather than inventing energy when the latency floor fires.
        latency_clamped = max(latency_floor, latency)
        energy_clamped = max(0.0, energy)
        if latency_clamped != latency or energy_clamped != energy:
            logger.warning(
                "GEMM.query applied latency SOL floor / non-negative energy clamp. "
                "op=%s m=%s n=%s k=%s quant_mode=%s post_sub(lat=%.6f, eng=%.6f) floor=%.6f",
                self._name,
                x,
                self._n,
                self._k,
                quant_mode.name,
                latency,
                energy,
                latency_floor,
            )

        latency = latency_clamped
        energy = energy_clamped

        return PerformanceResult(
            latency=latency * self._scale_factor,
            energy=energy * self._scale_factor,
            source=source,
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_gemm_data(gemm_file):
    """
    Load the gemm data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with
              'latency', 'power', and 'energy' keys.
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    rows = _read_filtered_rows(gemm_file)
    if rows is None:
        logger.debug(f"GEMM data file {gemm_file} not found.")
        return None
    gemm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (gemm) - power will default to 0.0")

    for row in rows:
        quant_mode, m, n, k, latency = (
            row["gemm_dtype"],
            row["m"],
            row["n"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        n = int(n)
        k = int(k)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))
        # Note: power_limit is available in row.get("power_limit") if needed for validation

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            gemm_data[quant_mode][m][n][k]
            logger.debug(f"value conflict in gemm data: {quant_mode} {m} {n} {k}")
        except KeyError:
            # Store all three values
            gemm_data[quant_mode][m][n][k] = {
                "latency": latency,
                "power": power,  # Keep for reference
                "energy": energy,  # NEW: precomputed energy
            }

    return gemm_data


def load_compute_scale_data(compute_scale_file):
    """
    Load the compute scale data with power support (backward compatible).

    Returns:
        dict: Nested dict structure {quant_mode: {m: {k: {latency, power, energy}}}}
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    rows = _read_filtered_rows(compute_scale_file)
    if rows is None:
        logger.debug(f"Compute scale data file {compute_scale_file} not found.")
        return None
    compute_scale_data = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (compute_scale) - power will default to 0.0")

    for row in rows:
        quant_mode, m, k, latency = (
            row["quant_dtype"],
            row["m"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        k = int(k)
        latency = float(latency)

        # Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            compute_scale_data[quant_mode][m][k]
            logger.debug(f"value conflict in compute_scale data: {quant_mode} {m} {k}")
        except KeyError:
            # Store all three values
            compute_scale_data[quant_mode][m][k] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return compute_scale_data


def load_scale_matrix_data(scale_matrix_file):
    """
    Load the scale matrix data with power support (backward compatible).

    Returns:
        dict: Nested dict structure {quant_mode: {m: {k: {latency, power, energy}}}}
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    rows = _read_filtered_rows(scale_matrix_file)
    if rows is None:
        logger.debug(f"Scale matrix data file {scale_matrix_file} not found.")
        return None
    scale_matrix_data = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (scale_matrix) - power will default to 0.0")

    for row in rows:
        quant_mode, m, k, latency = (
            row["quant_dtype"],
            row["m"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        k = int(k)
        latency = float(latency)

        # Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            scale_matrix_data[quant_mode][m][k]
            logger.debug(f"value conflict in scale_matrix data: {quant_mode} {m} {k}")
        except KeyError:
            # Store all three values
            scale_matrix_data[quant_mode][m][k] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return scale_matrix_data
