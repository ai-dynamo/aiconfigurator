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

if TYPE_CHECKING:
    pass


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
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6  # 5us

    # sol only
    _ENGINE_QUERY_SHAPE = "tokens"
