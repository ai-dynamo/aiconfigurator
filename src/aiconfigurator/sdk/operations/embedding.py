# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding operation (ISSUE-04 / AIC-477).

No CSV-backed data — latency derived analytically from ``mem_bw``. The
base ``Operation.load_data`` no-op default handles the missing table.
``query()`` calls ``interpolation.estimate_mem_op`` directly instead of
the ``PerfDatabase.query_mem_op`` wrapper (which ISSUE-16 retires once
every caller has migrated).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiconfigurator.sdk import interpolation
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase


class Embedding(Operation):
    """
    Embedding operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        row_size: int,
        column_size: int,
        empirical_bw_scaling_factor: float = 0.3,
    ) -> None:
        super().__init__(name, scale_factor)
        self._row_size = row_size
        self._column_size = column_size
        self._weights = row_size * column_size * 2
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6  # 5us

    # sol only
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query embedding latency with power data."""
        x = kwargs.get("x")
        d2d_bytes = x * self._column_size * 2

        result = interpolation.estimate_mem_op(database.system_spec["gpu"], d2d_bytes, database._default_database_mode)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
