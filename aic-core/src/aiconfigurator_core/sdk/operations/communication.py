# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Communication ops: NCCL + CustomAllReduce + P2P (ISSUE-07 / AIC-541).

Tables bind the engine table views (PR-6); parsing lives in the compiled
engine (`perf_database/table_view.rs`).

- ``CustomAllReduce`` owns ``custom_allreduce_perf.parquet`` — keyed by
  ``(quant_mode, tp_size, strategy)``. ``PerfDatabase.query_custom_allreduce``
  delegates here. No SOL clamp, no extrapolation in the legacy
  ``_correct_data`` / ``__init__`` path.

- ``NCCL`` owns ``nccl_perf.parquet`` AND the optional oneCCL fallback table.
  ``PerfDatabase.query_nccl`` delegates here. The oneCCL fallback is loaded
  alongside NCCL data because ``query_nccl`` picks between them at query
  time (XPU systems load oneCCL when NCCL is empty).

- ``P2P`` has no silicon table — latency is computed analytically from
  ``inter_node_bw`` + ``p2p_latency``. The base ``Operation.load_data``
  no-op default applies (the retired per-call lookup was factored out for
  parity with the other ops.

Cache key matches every other migrated op: ``(systems_root, system,
backend, version, enable_shared_layer)``. ``_build_op_sources`` early-
exits for ``nccl`` / ``oneccl`` (framework-agnostic dirs, no shared-layer
inheritance), so HYBRID mode doesn't union sibling rows for those.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation, resolve_op_data_path

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as GEMM and Attention.

    TODO: hoist to ``operations/base.py`` once Phase 3 lands and there
    are 4-5 op families duplicating this helper. Two callers (GEMM,
    Attention) was below the abstraction threshold; with Communication
    + DSA + MLA + Mamba + DSV4 coming, the threshold is now crossed.
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


class CustomAllReduce(Operation):
    """
    Custom AllReduce operation with power tracking.

    Owns ``_data_cache`` for the packaged custom_allreduce Parquet perf table.
    """

    _data_cache: ClassVar[dict] = {}
    _CP_AWARE: ClassVar[bool] = True  # query divides x by self._seq_split (smaller per-rank AR payload)

    def __init__(self, name: str, scale_factor: float, h: int, tp_size: int, *, seq_split: int = 1) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._h = h
        self._tp_size = tp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's custom_allreduce table view and binds
        ``database._custom_allreduce_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_custom_allreduce_data", PerfDataFilename.custom_allreduce)
            cls._record_load()

        if "_custom_allreduce_data" not in database.__dict__:
            database._custom_allreduce_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_custom_allreduce)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"


class NCCL(Operation):
    """
    NCCL collective communication operation with power tracking.

    Owns ``_data_cache`` for the packaged NCCL Parquet perf table plus ``_oneccl_data_cache``
    for the optional oneCCL fallback (loaded together because
    ``query_nccl`` picks between them at query time when NCCL data is
    empty on XPU systems).
    """

    _data_cache: ClassVar[dict] = {}
    _oneccl_data_cache: ClassVar[dict] = {}
    _CP_AWARE: ClassVar[bool] = True  # query divides x by self._seq_split (smaller per-rank payload)

    def __init__(
        self,
        name: str,
        scale_factor: float,
        nccl_op: str,
        num_elements_per_token: int,
        num_gpus: int,
        comm_quant_mode: common.CommQuantMode,
        *,
        seq_split: int = 1,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._nccl_op = nccl_op
        self._num_elements_per_token = num_elements_per_token
        self._num_gpus = num_gpus
        self._comm_quant_mode = comm_quant_mode

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's NCCL table view plus the optional oneCCL fallback,
        binds ``database._nccl_data`` and ``database._oneccl_data``."""
        import os

        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache or key not in cls._oneccl_data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            # NCCL data lives under ``systems_data_root/nccl/<nccl_version>/``
            # (legacy) or ``systems_data_root/<family>/nccl/<nccl_version>/``
            # (family-first), NOT under ``backend/version/`` — so the wrapper's
            # ``filepath`` keeps the nccl_version-resolved primary. NCCL ops
            # never inherit shared-layer sibling rows (the engine view loads
            # the single system-wide file).
            # Optional like oneccl below (the Rust comm_root resolution is
            # Option-tolerant): a spec without misc.nccl_version binds an
            # unloaded wrapper instead of KeyError-ing the whole load.
            # Locals first, commit last — a failed oneccl fetch must not
            # leave only the nccl side cached (see GEMM.load_data).
            nccl_version = (database.system_spec.get("misc") or {}).get("nccl_version")
            if nccl_version:
                nccl_primary = resolve_op_data_path(system_data_root, "nccl", nccl_version, PerfDataFilename.nccl.value)
                nccl_loaded = LoadedOpData(
                    fetch_table_view(database, "_nccl_data"), PerfDataFilename.nccl, nccl_primary
                )
            else:
                nccl_loaded = LoadedOpData(None, PerfDataFilename.nccl, PerfDataFilename.nccl.value)

            # oneCCL fallback (XPU systems). Only loaded when system_spec
            # declares an ``oneccl_version`` under ``misc``.
            oneccl_version = database.system_spec.get("misc", {}).get("oneccl_version")
            if oneccl_version:
                oneccl_primary = resolve_op_data_path(
                    system_data_root, "oneccl", oneccl_version, PerfDataFilename.oneccl.value
                )
                oneccl_loaded = LoadedOpData(
                    fetch_table_view(database, "_oneccl_data"), PerfDataFilename.oneccl, oneccl_primary
                )
            else:
                oneccl_loaded = None

            cls._data_cache[key] = nccl_loaded
            cls._oneccl_data_cache[key] = oneccl_loaded
            cls._record_load()

        if "_nccl_data" not in database.__dict__:
            database._nccl_data = cls._data_cache[key]
        if "_oneccl_data" not in database.__dict__:
            database._oneccl_data = cls._oneccl_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._oneccl_data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_nccl)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"


class P2P(Operation):
    """
    P2P (point-to-point) communication operation with power tracking.

    Purely analytical — no silicon table. The base ``Operation.load_data``
    no-op default handles the missing perf table (the retired per-call lookup was factored
    out only for parity with the other migrated ops.
    """

    _CP_AWARE: ClassVar[bool] = True  # query divides x by self._seq_split (smaller per-rank payload)

    def __init__(self, name: str, scale_factor: float, h: int, pp_size: int, *, seq_split: int = 1) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._h = h
        self._pp_size = pp_size
        self._bytes_per_element = 2

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_p2p)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"
