# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite operations (ISSUE-14).

Two op classes migrated from ``_legacy.py``:

- ``FallbackOp`` — try a primary op, fall back to a sequence of ops on
  ``PerfDataNotAvailableError``. Mutates
  ``database._default_database_mode`` during the primary attempt (forces
  SILICON inside HYBRID so HYBRID doesn't silently swallow a miss with an
  empirical estimate); restores via ``finally``.
- ``OverlapOp`` — model two op groups that execute in parallel (TRT-LLM
  ``maybe_execute_in_parallel`` behavior on different CUDA streams during
  generation with CUDA Graph enabled). ``latency = max(sum_a, sum_b)``,
  ``energy = sum_a + sum_b``.

Neither op owns any CSV data — they delegate to inner ``Operation``
instances and their ``query()`` methods. No ``_data_cache``, no
``load_data``, no ``clear_cache``; the ``Operation`` base class provides
empty defaults that suffice.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase


logger = logging.getLogger(__name__)


class FallbackOp(Operation):
    """
    Try a primary operation first; if it raises PerfDataNotAvailableError,
    fall back to a sequence of fallback operations (summed).

    This supports transitional periods where some systems have module-level
    profiling data (single op) while others still have granular per-kernel data
    (multiple ops). The fallback is symmetric: either group can be primary.

    In HYBRID mode, the primary is queried in SILICON mode so that HYBRID does
    not silently swallow a miss with an empirical estimate — the fallback ops
    (which have real data) should be preferred over an empirical guess. In
    explicit EMPIRICAL/SOL modes, the primary respects the requested mode.

    Once the primary fails on the first call, it is skipped on all subsequent
    calls to avoid redundant work.

    Latency = primary.query()  OR  sum(fallback[i].query())
    Energy  = same source as whichever succeeds
    Weights = sum of whichever group is used (primary or fallback)
    """

    _CP_AWARE: ClassVar[bool] = True  # wrapper: inner ops carry their own seq_split

    def __init__(self, name: str, primary: Operation, fallback: list[Operation], *, seq_split: int = 1) -> None:
        """
        Args:
            name: Operation name for latency breakdown reporting.
            primary: Single operation to try first.
            fallback: List of operations to sum if primary fails.
            seq_split: Carried for API uniformity. The wrapper delegates to
                inner ops which carry their own ``seq_split``; this one is
                stored on the base class for completeness but not used here.
        """
        super().__init__(name, 1.0, seq_split=seq_split)  # scale_factor handled by inner ops
        self._primary = primary
        self._fallback = fallback
        self._primary_unavailable = False

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        import logging as _logging

        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

        if not self._primary_unavailable:
            prev_mode = database._default_database_mode
            force_primary_silicon = prev_mode == common.DatabaseMode.HYBRID
            if force_primary_silicon:
                # Force SILICON mode on the primary so HYBRID does not silently
                # return an empirical estimate when module data is missing.
                database._default_database_mode = common.DatabaseMode.SILICON

            # Suppress ERROR-level logs from perf_database during the primary
            # attempt, since a failure here is expected and handled by fallback.
            perf_db_logger = _logging.getLogger("aiconfigurator.sdk.perf_database")
            prev_log_level = perf_db_logger.level
            perf_db_logger.setLevel(_logging.CRITICAL)
            try:
                return self._primary.query(database, **kwargs)
            except (PerfDataNotAvailableError, KeyError, AssertionError) as e:
                if isinstance(e, PerfDataNotAvailableError):
                    self._primary_unavailable = True
                logger.debug(
                    "FallbackOp '%s': primary op '%s' failed (%s: %s), using fallback ops",
                    self._name,
                    self._primary._name,
                    type(e).__name__,
                    e,
                )
            finally:
                if force_primary_silicon:
                    database._default_database_mode = prev_mode
                perf_db_logger.setLevel(prev_log_level)

        total = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._fallback:
            total += op.query(database, **kwargs)
        return total

    def get_weights(self, **kwargs):
        # Use primary weights if available, otherwise sum fallback weights.
        # In practice both should be equivalent since they model the same block.
        if not self._primary_unavailable:
            primary_w = self._primary.get_weights(**kwargs)
            if primary_w > 0:
                return primary_w
        return sum(op.get_weights(**kwargs) for op in self._fallback)


class OverlapOp(Operation):
    """
    Two groups of operations that execute in parallel (overlap).

    This models the TRT-LLM `maybe_execute_in_parallel` behavior where two
    operation groups run concurrently on different CUDA streams during
    generation phase (CUDA Graph enabled).

    Latency = max(sum(group_a latencies), sum(group_b latencies))
    Energy  = sum(all ops in both groups)  # both groups consume power
    Weights = sum(all ops in both groups)
    """

    _CP_AWARE: ClassVar[bool] = True  # wrapper: inner ops carry their own seq_split

    def __init__(self, name: str, group_a: list, group_b: list, *, seq_split: int = 1) -> None:
        """
        Args:
            name: Operation name for latency breakdown reporting.
            group_a: List of Operation objects for the first parallel group
                     (e.g., routed expert path on main stream).
            group_b: List of Operation objects for the second parallel group
                     (e.g., shared expert path on aux stream).
            seq_split: Carried for API uniformity. Inner ops carry their own.
        """
        super().__init__(name, 1.0, seq_split=seq_split)  # scale_factor handled by inner ops
        self._group_a = group_a
        self._group_b = group_b

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query overlap operation latency.

        Returns:
            PerformanceResult with latency = max(group_a, group_b)
            and energy = sum of all ops.
        """
        total_a = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._group_a:
            total_a += op.query(database, **kwargs)

        total_b = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._group_b:
            total_b += op.query(database, **kwargs)

        merged = total_a + total_b
        return PerformanceResult(
            latency=max(float(total_a), float(total_b)),
            energy=total_a.energy + total_b.energy,
            source=merged.source,
        )

    def get_weights(self, **kwargs):
        weights = 0.0
        for op in self._group_a + self._group_b:
            weights += op.get_weights(**kwargs)
        return weights
