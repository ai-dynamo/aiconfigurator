# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ElementWise operation (ISSUE-04 / AIC-477).

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


class ElementWise(Operation):
    """
    Element-wise operation.
    """

    _CP_AWARE = True  # query divides x (token count) by self._seq_split

    def __init__(
        self,
        name: str,
        scale_factor: float,
        dim_in: int,
        dim_out: int,
        empirical_bw_scaling_factor: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=kwargs.get("seq_split", 1))
        self._weights = 0.0
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6  # 5us
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)

    # sol only
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query element-wise operation latency with power data."""
        x = kwargs.get("x")  # num tokens
        if x is None:
            raise ValueError("ElementWise.query requires 'x' (num tokens).")
        if self._scale_num_tokens <= 0:
            raise ValueError(f"ElementWise.query: scale_num_tokens must be > 0, got {self._scale_num_tokens}.")
        x //= self._scale_num_tokens
        x = -(-x // self._seq_split)  # CP: per-rank token count (ceil = busiest rank)
        read_bytes = x * self._dim_in * 2  # bfloat16 for act
        write_bytes = x * self._dim_out * 2

        result = database.query_mem_op(read_bytes + write_bytes)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
