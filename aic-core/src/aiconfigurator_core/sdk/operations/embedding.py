# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding operation (ISSUE-04 / AIC-477).

No CSV-backed data — latency derived analytically from ``mem_bw``. The
base ``Operation.load_data`` no-op default handles the missing table.
``query()`` calls ``database.query_mem_op`` (the legacy entry point on
``PerfDatabase``); deciding a long-term home for the analytical mem-op
formula is deferred to the post-refactor cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiconfigurator_core.sdk.operations.base import Operation
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase


class Embedding(Operation):
    """
    Embedding operation.
    """

    _CP_AWARE = True  # query divides x (token count) by self._seq_split

    def __init__(
        self,
        name: str,
        scale_factor: float,
        row_size: int,
        column_size: int,
        empirical_bw_scaling_factor: float = 0.3,
        *,
        seq_split: int = 1,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._row_size = row_size
        self._column_size = column_size
        self._weights = row_size * column_size * 2
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6  # 5us

    # sol only
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query embedding latency with power data."""
        x = kwargs.get("x")
        if x is None:
            raise ValueError("Embedding.query requires 'x' (num tokens).")
        x = -(-x // self._seq_split)  # CP: per-rank token count (ceil = busiest rank)
        d2d_bytes = x * self._column_size * 2

        result = database.query_mem_op(d2d_bytes)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
