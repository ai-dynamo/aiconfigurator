# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Communication ops: NCCL + CustomAllReduce + P2P (ISSUE-07 / AIC-541).

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
  no-op default applies; ``_query_p2p_table`` is factored out for
  parity with the other ops.

Cache key matches every other migrated op: ``(systems_root, system,
backend, version, enable_shared_layer)``. ``_build_op_sources`` early-
exits for ``nccl`` / ``oneccl`` (framework-agnostic dirs, no shared-layer
inheritance), so HYBRID mode doesn't union sibling rows for those.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path

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
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the packaged custom_allreduce Parquet perf table and binds
        ``database._custom_allreduce_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.custom_allreduce.value
            )
            sources = database._build_op_sources(PerfDataFilename.custom_allreduce, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_custom_allreduce_data(sources), PerfDataFilename.custom_allreduce, primary_path
            )
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

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


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
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the packaged NCCL Parquet perf table plus the optional oneCCL fallback,
        binds ``database._nccl_data`` and ``database._oneccl_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            # NCCL data lives under ``systems_data_root/nccl/<nccl_version>/``
            # (legacy) or ``systems_data_root/<family>/nccl/<nccl_version>/``
            # (family-first), NOT under ``backend/version/``. Per
            # ``_build_op_sources`` early-exit, NCCL ops never inherit
            # shared-layer sibling rows.
            nccl_version = database.system_spec["misc"]["nccl_version"]
            nccl_primary = resolve_op_data_path(system_data_root, "nccl", nccl_version, PerfDataFilename.nccl.value)
            nccl_sources = database._build_op_sources(PerfDataFilename.nccl, nccl_primary, system_data_root)
            cls._data_cache[key] = LoadedOpData(load_nccl_data(nccl_sources), PerfDataFilename.nccl, nccl_primary)

            # oneCCL fallback (XPU systems). Only loaded when system_spec
            # declares an ``oneccl_version`` under ``misc``.
            oneccl_version = database.system_spec.get("misc", {}).get("oneccl_version")
            if oneccl_version:
                oneccl_primary = resolve_op_data_path(
                    system_data_root, "oneccl", oneccl_version, PerfDataFilename.oneccl.value
                )
                oneccl_sources = database._build_op_sources(PerfDataFilename.oneccl, oneccl_primary, system_data_root)
                cls._oneccl_data_cache[key] = LoadedOpData(
                    load_nccl_data(oneccl_sources), PerfDataFilename.oneccl, oneccl_primary
                )
            else:
                cls._oneccl_data_cache[key] = None

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

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class P2P(Operation):
    """
    P2P (point-to-point) communication operation with power tracking.

    Purely analytical — no silicon table. The base ``Operation.load_data``
    no-op default handles the missing perf table; ``_query_p2p_table`` is factored
    out only for parity with the other migrated ops.
    """

    _CP_AWARE: ClassVar[bool] = True  # query divides x by self._seq_split (smaller per-rank payload)

    def __init__(self, name: str, scale_factor: float, h: int, pp_size: int, *, seq_split: int = 1) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._h = h
        self._pp_size = pp_size
        self._bytes_per_element = 2
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_p2p)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ─────────────────────────────────────────────────────────
# Perf-table loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_custom_allreduce_data(custom_allreduce_file):
    """
    Load the custom allreduce data with power support (backward compatible).

    Supports multiple data formats:
    - TRTLLM: kernel_source="TRTLLM", last column="implementation"
    - vLLM/SGLang: kernel_source="*_graph" or "*_eager", last column="backend"

    For vLLM/SGLang with both graph and eager modes, only graph mode data is kept
    (better performance for decode phase).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(custom_allreduce_file)
    if rows is None:
        logger.debug(f"Custom allreduce data file {custom_allreduce_file} not found.")
        return None
    custom_allreduce_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (custom_allreduce) - power will default to 0.0")

    if isinstance(custom_allreduce_file, str):
        is_b60 = "b60" in custom_allreduce_file
    else:
        is_b60 = any("b60" in path for path, _ in custom_allreduce_file)

    for row in rows:
        # Check kernel_source to filter graph vs eager mode (for vLLM/SGLang)
        kernel_source = row.get("kernel_source", "")
        backend = row.get("backend", "")

        # For vLLM/SGLang format: only keep graph mode data (skip eager mode)
        # kernel_source patterns: "vLLM_custom_graph", "SGLang_CustomAllReduce_graph", etc.
        # backend patterns: "vllm_graph", "sglang_graph", etc.
        if (kernel_source.endswith("_eager") or backend.endswith("_eager")) and not is_b60:
            continue  # Skip eager mode, use graph mode only

        dtype, tp_size, message_size, latency = (
            row["allreduce_dtype"],
            row["num_gpus"],
            row["message_size"],
            row["latency"],
        )
        allreduce_strategy = "AUTO"
        message_size = int(message_size)
        latency = float(latency)
        tp_size = int(tp_size)
        dtype = common.CommQuantMode.half  # TODO

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        try:
            # Check for conflict
            custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size]
            logger.debug(
                f"value conflict in custom allreduce data: {dtype} {tp_size} {allreduce_strategy} {message_size}"
            )
        except KeyError:
            # Store all three values
            custom_allreduce_data[dtype][tp_size][allreduce_strategy][message_size] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return custom_allreduce_data


def load_nccl_data(nccl_file):
    """
    Load the nccl data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(nccl_file)
    if rows is None:
        logger.debug(f"NCCL data file {nccl_file} not found.")
        return None
    nccl_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (nccl) - power will default to 0.0")

    for row in rows:
        dtype, num_gpus, message_size, op_name, latency = (
            row["nccl_dtype"],
            row["num_gpus"],
            row["message_size"],
            row["op_name"],
            row["latency"],
        )
        message_size = int(message_size)
        latency = float(latency)
        num_gpus = int(num_gpus)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        dtype = common.CommQuantMode[dtype]
        try:
            # Check for conflict
            nccl_data[dtype][op_name][num_gpus][message_size]
            logger.debug(f"value conflict in nccl data: {dtype} {op_name} {num_gpus} {message_size}")
        except KeyError:
            # Store all three values
            nccl_data[dtype][op_name][num_gpus][message_size] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return nccl_data
