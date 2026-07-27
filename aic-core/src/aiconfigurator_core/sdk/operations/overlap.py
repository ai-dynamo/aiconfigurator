# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite operations (ISSUE-14).

Two op classes migrated from ``_legacy.py``:

- ``FallbackOp`` — try a primary op, fall back to a sequence of ops on
  ``PerfDataNotAvailableError``. In HYBRID mode the primary runs against a
  SILICON-configured copy, so HYBRID does not silently swallow a miss with an
  empirical estimate and the caller's database is never mutated.
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

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase


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

    A data miss applies only to the current query. Later shapes try the primary
    again because a missing interpolation bracket does not imply that the whole
    module table is unavailable. Raw schema/programming errors propagate.

    Latency = primary.query()  OR  sum(fallback[i].query())
    Energy  = same source as whichever succeeds
    Weights = primary weights when defined, otherwise the fallback sum
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

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError, _get_configured_database_view

        primary_database = (
            _get_configured_database_view(
                database,
                common.DatabaseMode.SILICON,
                getattr(database, "transfer_policy", None),
            )
            if database._default_database_mode == common.DatabaseMode.HYBRID
            else database
        )

        try:
            return self._primary.query(primary_database, **kwargs)
        except PerfDataNotAvailableError as e:
            logger.debug(
                "FallbackOp '%s': primary op '%s' failed (%s: %s), using fallback ops",
                self._name,
                self._primary._name,
                type(e).__name__,
                e,
            )

        total = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._fallback:
            total += op.query(database, **kwargs)
        return total

    def get_weights(self, **kwargs):
        # Use primary weights if available, otherwise sum fallback weights.
        # In practice both should be equivalent since they model the same block.
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
