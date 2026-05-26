# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base class and shared infrastructure for the operations package.

This module defines the ``Operation`` ABC plus two pieces of shared
infrastructure that future op classes will rely on:

- **Class-level ``_data_cache``** — each Operation subclass that owns CSV data
  overrides this in its own class. Keyed by ``(system_path, db_mode)`` so the
  same op type can serve multiple databases in one process.
- **``_load_data_call_count`` instrumentation** — used by tests to assert
  which op classes actually loaded data during a model run. The expected set
  for Minimax M2.5 NVFP4 is the canonical lazy-load success assertion
  (see ``~/forks/sdk-refactor-regression/tests/test_load_data_counts.py``).
- **``supported_quant_modes`` classmethod** — placeholder API used by
  ``inference_session`` post-Phase-4 to build the support-matrix warning.
  Default returns the empty set; ops with quant-mode-keyed CSVs override.

``clear_all_op_caches()`` is a module-level utility that walks every
``Operation`` subclass and clears both its data cache and any LRU on
``query``. Exported from the ``aiconfigurator.sdk.operations`` package — same
function powers a pytest ``autouse`` fixture and serves as a manual eviction
lever for long-running webapps.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


class Operation:
    """
    Base operation class.

    Note: query() returns PerformanceResult (float-like) instead of plain float.
    The class behaves as a float for backward compatibility while carrying
    energy data and a ``source`` tag ("silicon" / "empirical" / "mixed").
    """

    # Subclasses that own CSV data override this. Keyed by (system_path, db_mode).
    _data_cache: ClassVar[dict] = {}

    # Test/observability counter. Each subclass's load_data() calls
    # Operation._record_load(cls) after a successful parse (NOT on cache hit).
    _load_data_call_count: ClassVar[dict[type, int]] = defaultdict(int)

    def __init__(self, name: str, scale_factor: float) -> None:
        self._name = name
        self._scale_factor = scale_factor

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Return latency (scaled by ``scale_factor``) plus energy/source data."""
        raise NotImplementedError

    def get_weights(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Subclasses with CSV data override; default no-op for
        ops like ``ElementWise`` that compute analytically from system spec.

        The full ``database`` is passed (not just ``system_path``/``system_spec``)
        so subclasses can derive their own cache key plus reuse PerfDatabase
        helpers like ``_build_op_sources`` for HYBRID-mode source discovery."""
        return None

    @classmethod
    def clear_cache(cls):
        """Clear this op's data cache and any LRU on ``query``. Subclasses
        with their own ``_data_cache`` override the class attribute; if a
        subclass never declared one, fall back to evicting the shared
        ``Operation._data_cache`` so ``clear_all_op_caches()`` doesn't
        silently skip it."""
        cache = cls.__dict__.get("_data_cache")
        if cache is None:
            cache = Operation._data_cache
        cache.clear()
        # query may be wrapped in functools.lru_cache — clear if present.
        query = cls.__dict__.get("query")
        if query is not None and hasattr(query, "cache_clear"):
            query.cache_clear()

    @classmethod
    def supported_quant_modes(cls, database: PerfDatabase) -> set:
        """Return the quant modes for which this op has CSV data on the
        given database. Default empty — ops with quant-mode-keyed data
        override. Used by ``_update_support_matrix`` (moves to
        ``inference_session`` in ISSUE-16).

        Takes the full ``database`` for symmetry with ``load_data``."""
        return set()

    @classmethod
    def _record_load(cls):
        """Subclasses call this from load_data() after a successful parse,
        NOT on a cache hit. The instrumentation lets tests assert which op
        classes loaded for a given model run."""
        Operation._load_data_call_count[cls] += 1


def _all_operation_subclasses(root: type = Operation) -> set[type]:
    """Recursively collect every Operation subclass currently imported."""
    seen: set[type] = set()
    stack: list[type] = [root]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


def clear_all_op_caches() -> None:
    """Walk every imported Operation subclass and call its ``clear_cache()``.

    Used by:
    - production callers (long-running webapps) that need a manual eviction
      lever; the per-op ``_data_cache`` is process-wide and never auto-evicts
    - test helpers that need a fully clean slate (the conftest autouse
      fixture clears only the counter, not data caches — clearing the
      caches would force a fresh-disk reload mid-suite)

    Also clears the shared instrumentation counter.

    Note: this does NOT clear the ``@functools.lru_cache`` on the
    ``PerfDatabase.query_*`` wrappers — those caches live on each database
    instance and must be cleared separately via
    ``database.clear_runtime_caches()`` if callers also want to invalidate
    interpolated/extrapolated query results."""
    for cls in _all_operation_subclasses():
        cls.clear_cache()
    Operation._load_data_call_count.clear()
