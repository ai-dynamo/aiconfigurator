# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ElementWise operation (ISSUE-04 / AIC-477).

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


class ElementWise(Operation):
    """
    Element-wise operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        dim_in: int,
        dim_out: int,
        empirical_bw_scaling_factor: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
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
        x //= self._scale_num_tokens
        read_bytes = x * self._dim_in * 2  # bfloat16 for act
        write_bytes = x * self._dim_out * 2

        result = interpolation.estimate_mem_op(
            database.system_spec["gpu"], read_bytes + write_bytes, database._default_database_mode
        )
        # ``estimate_mem_op`` always returns a tagged PerformanceResult
        # (``"sol"`` / ``"empirical"``) — read the tag directly. Mem-op is
        # never silicon-tagged because there is no silicon table for raw
        # memory ops.
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=result.source,
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
