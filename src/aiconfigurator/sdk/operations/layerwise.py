# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import interpolation
from aiconfigurator.sdk.operations.base import Operation, _read_filtered_rows
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


def _cache_key(database: PerfDatabase) -> tuple:
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


def load_layerwise_data(layerwise_file):
    rows = _read_filtered_rows(layerwise_file)
    if rows is None:
        logger.debug("Layerwise data file %s not found.", layerwise_file)
        return None

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    for row in rows:
        model = str(row["model"]).lower()
        phase = str(row["phase"]).upper()
        tp_size = int(row["tp_size"])
        batch_size = int(row["batch_size"])
        seq_len_q = int(row.get("seq_len_q") or row.get("new_tokens") or 1)
        seq_len_kv_cache = int(row.get("seq_len_kv_cache") or row.get("past_kv") or 0)
        if row.get("latency_ms") not in (None, ""):
            latency_ms = float(row["latency_ms"])
        else:
            latency_ms = float(row["total_time_us"]) / 1000.0

        entry = {"latency": latency_ms, "energy": 0.0}
        if phase == "CTX":
            data[model][phase][tp_size][seq_len_q][seq_len_kv_cache] = entry
        else:
            data[model][phase][tp_size][batch_size][seq_len_kv_cache] = entry

    return data


class Layerwise(Operation):
    _data_cache: ClassVar[dict] = {}

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.layerwise.value)
            sources = database._build_op_sources(PerfDataFilename.layerwise, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(load_layerwise_data(sources), PerfDataFilename.layerwise, primary_path)
            cls._record_load()

        if "_layerwise_data" not in database.__dict__:
            database._layerwise_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    @classmethod
    def _query_layerwise_table(
        cls,
        database: PerfDatabase,
        model: str,
        phase: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        seq_len_kv_cache: int = 0,
    ) -> PerformanceResult:
        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

        model = model.lower()
        phase = phase.upper()
        cls.load_data(database)
        data_wrapper = database._layerwise_data
        if data_wrapper is None or not data_wrapper.loaded:
            raise PerfDataNotAvailableError("Layerwise data not available for this system/backend/version")

        data = data_wrapper.data
        if model not in data:
            raise PerfDataNotAvailableError(f"Model {model!r} not found in layerwise data")
        if phase not in data[model]:
            raise PerfDataNotAvailableError(f"Phase {phase!r} not found in layerwise data for {model}")
        if tp_size not in data[model][phase]:
            raise PerfDataNotAvailableError(f"tp_size={tp_size} not found in layerwise data for {model}/{phase}")

        tp_data = data[model][phase][tp_size]
        if phase == "CTX":
            if seq_len in tp_data and seq_len_kv_cache in tp_data[seq_len]:
                result = tp_data[seq_len][seq_len_kv_cache]
            elif len(tp_data) < 2:
                raise PerfDataNotAvailableError(f"Not enough CTX layerwise data points for tp_size={tp_size}")
            else:
                result = interpolation.interp_2d_linear(
                    seq_len,
                    seq_len_kv_cache,
                    tp_data,
                    database._extracted_metrics_cache,
                )
        elif batch_size in tp_data and seq_len in tp_data[batch_size]:
            result = tp_data[batch_size][seq_len]
        else:
            result = interpolation.interp_2d_linear(batch_size, seq_len, tp_data, database._extracted_metrics_cache)

        latency = result["latency"] if isinstance(result, dict) else float(result)
        energy = result.get("energy", 0.0) if isinstance(result, dict) else 0.0
        return PerformanceResult(latency, energy=energy, source="silicon")

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        return self._query_layerwise_table(
            database,
            kwargs["model"],
            kwargs["phase"],
            kwargs["tp_size"],
            kwargs["batch_size"],
            kwargs["seq_len"],
            kwargs.get("seq_len_kv_cache", 0),
        )

    def get_weights(self, **kwargs):
        return 0.0
