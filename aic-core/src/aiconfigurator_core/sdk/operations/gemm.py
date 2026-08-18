# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM operation over the engine table views (gemm, compute_scale, scale_matrix).

GEMM owns its three CSV-backed raw tables (data plane only — per-op
values come from the compiled engine, #1357 PR-5).
``PerfDatabase.query_compute_scale / query_scale_matrix`` are tombstoned;
``query_gemm`` is an engine-routed deprecation shim.

Lazy-load Pattern A: consumers trigger ``load_data`` on cache miss. ``_data_cache`` /
``_compute_scale_cache`` / ``_scale_matrix_cache`` are keyed by
``(systems_root, system, backend, version, enable_shared_layer)`` so the
same op class serves multiple databases in one process. ``systems_root``
is part of the key because test fixtures swap to a fresh ``tmp_path``
between tests and must get distinct cache entries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import Operation

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


# Per-quant achieved-util LEVEL e(q) for GEMM, keyed by the (memory, compute)
# profile — the GEMM counterpart of moe.py's _MOE_QUANT_UTIL_LEVEL, consumed
# ONLY by the cross-PROFILE relation of the quant-transfer primitive as the
# ratio e(query)/e(ref) (see util_empirical.quant_transfer_grid). A SINGLE
# scalar per profile by design (the per-component split was validated as
# untrustworthy on MoE; the same SOL-attribution argument holds here).
#
# [data] rows: median of util = SOL/measured over the clearly compute-bound
# region (m >= 1024, n >= 2048, k >= 2048) of every collected gemm table on
# b200/h200/h100 x trtllm/vllm/sglang (2026-08 snapshot); the range across
# those six stacks is quoted per row. The bf16/fp8 level RATIO spans
# 1.38-2.00 (~±20% around 1.6) — looser than MoE's ~10% but acceptable for
# the last-resort relation. LOO on the mechanism (predict a collected quant
# from its nearest-profile sibling at shared shapes, m >= 64): nvfp4 <- fp8
# = 22-33% MAPE, comparable to MoE's ~24% xprofile LOO. [inferred] rows
# follow the structure (efficiency drops with weight precision, mildly
# recovers as activation precision drops); levels are relative and tunable —
# only ratios are consumed.
# PROJECTION of the engine's table (PR-6): the Rust
# `operators/gemm.rs::GEMM_QUANT_UTIL_LEVEL` is the single source (with the
# same per-row [data]/[inferred] provenance notes); rebuilding the dict from
# the FFI ends the two-sided sync discipline every new quant used to need.
_GEMM_QUANT_UTIL_LEVEL: dict[tuple[float, float], float] = {
    (memory, compute): level for memory, compute, level in aiconfigurator_core.gemm_quant_util_levels()
}
_GEMM_QUANT_UTIL_DEFAULT = 0.45  # unlisted profile: mid-range relative level


def xprofile_util_level_known(quant_mode) -> bool:
    """Whether the GEMM util-LEVEL table lists this quant's profile.

    The runtime ladder falls back to ``_GEMM_QUANT_UTIL_DEFAULT`` for
    unlisted profiles; the validate gate deliberately does NOT (admitting a
    quant nobody calibrated would hide the missing level line the
    add-a-quant recipe requires), so it asks this instead of reaching into
    the table."""
    return util_empirical.quant_profile(quant_mode) in _GEMM_QUANT_UTIL_LEVEL


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
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)
        self._low_precision_input = kwargs.get("low_precision_input", False)
        self._below_grid_sol = kwargs.get("below_grid_sol", False)

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
        """Idempotent. On cache miss: fetches the three engine table views
        (raw as-collected rows in the retired parsers' exact shape — the
        engine owns parsing, clamping and interpolation) and records the
        load. Always: binds
        ``database._gemm_data``/``_compute_scale_data``/``_scale_matrix_data``
        to the cached wrappers.

        Tests that have already set those instance attributes (e.g.
        ``db._gemm_data = LoadedOpData(...)``) are respected — the binds
        below are gated on ``"_gemm_data" not in database.__dict__`` so
        intentional overrides survive."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache or key not in cls._compute_scale_cache or key not in cls._scale_matrix_cache:
            # Fetch all three into locals first so a failure on the second or
            # third view doesn't leave the cache half-populated (which would
            # let a subsequent ``key in cls._data_cache`` early-out skip past
            # the missing siblings and crash downstream).
            gemm_loaded = load_view(database, "_gemm_data", PerfDataFilename.gemm)
            compute_scale_loaded = load_view(database, "_compute_scale_data", PerfDataFilename.compute_scale)
            scale_matrix_loaded = load_view(database, "_scale_matrix_data", PerfDataFilename.scale_matrix)

            # All three loads succeeded — commit atomically so partially-
            # populated cache state can never be observed.
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

    # ------------------------------------------------------------------
    # SOL correction (formerly in PerfDatabase._correct_data)
    # ------------------------------------------------------------------

    # NOTE(#1357 PR-5): the load-time SOL clamp (`_correct_sol`) retired with
    # the Python query math. The loaded table is now the RAW collected data
    # plane (enumeration/charts); the compiled engine applies the same clamp
    # to its own load (see perf_database/gemm.rs), so QUERY values stay
    # SOL-floored via the single oracle.

    # ------------------------------------------------------------------
    # Table query classmethods (formerly PerfDatabase.query_*)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"

    def _engine_query_plan(self, kwargs: dict):
        """Legacy per-call ``quant_mode`` override: rebuild the twin with the
        requested quant before engine evaluation (an uncovered override quant
        must MISS loudly, exactly like the retired lookup did)."""
        op, eval_kwargs = super()._engine_query_plan(kwargs)
        quant_mode = kwargs.get("quant_mode")
        if quant_mode is not None and quant_mode != self._quant_mode:
            import copy

            op = copy.copy(self)
            op._quant_mode = quant_mode
        return op, eval_kwargs
