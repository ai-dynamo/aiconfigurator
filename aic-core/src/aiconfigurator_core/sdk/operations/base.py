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
``query``. Exported from the ``aiconfigurator_core.sdk.operations`` package — same
function powers a pytest ``autouse`` fixture and serves as a manual eviction
lever for long-running webapps.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

import yaml

from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


def _read_perf_rows(perf_file: str) -> list[dict[str, object]]:
    if perf_file.lower().endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Loading parquet perf data requires the 'pyarrow' package. "
                "Install aiconfigurator with its declared runtime dependencies."
            ) from exc
        return [
            {key: "" if value is None else value for key, value in row.items()}
            for row in pq.read_table(perf_file).to_pylist()
        ]

    with open(perf_file, encoding="utf-8", newline="") as f:
        return [{key: "" if value is None else value for key, value in row.items()} for row in csv.DictReader(f)]


def _resolve_perf_data_path(perf_file: str) -> str:
    if os.path.exists(perf_file):
        return perf_file
    stem, suffix = os.path.splitext(perf_file)
    if suffix.lower() == ".parquet":
        legacy_file = f"{stem}.txt"
        if os.path.exists(legacy_file):
            return legacy_file
    return perf_file


# CANONICAL definition of the first-level dirs under <system>/ that are
# backend dirs (legacy layout) rather than family dirs. The SDK loader
# imports this (perf_database.KNOWN_BACKEND_DIRS is an alias); the standalone
# copies that cannot import aic-core must stay textually identical and each
# cross-reference this site:
#   tools/perf_database/migrate_family_layout.py  (KNOWN_BACKEND_DIRS)
#   tools/sanity_check/create_charts.py           (_KNOWN_BACKEND_DIRS)
#   aic-core/rust/aiconfigurator-core/src/perf_database/mod.rs (KNOWN_BACKEND_DIRS)
# (tools/perf_database/audit_kernel_source.py's _LEGACY_BACKEND_DIRS is a
# deliberate 3-entry variant — consumer backends only, no comm pseudo-backends.)
_KNOWN_BACKEND_DIRS = frozenset({"trtllm", "sglang", "vllm", "nccl", "oneccl"})


def _version_dir_is_partial(version_dir: str) -> bool:
    """Yaml-first partial-dir check: collection_meta.yaml status:partial, with
    INCOMPLETE.txt as the legacy fallback.

    Duplicated (not imported) from aiconfigurator_core.sdk.perf_database
    ._version_dir_state, the source of truth for this semantic — perf_database
    imports this module at load time, so importing it back here would be
    circular. Keep in sync with that function's partial-detection rule.

    CONTRACT NOTE — the lenient/strict split is intentional design, not drift:
    this RESOLVER-side copy deliberately swallows read/parse errors and
    returns False, because its only job is cheap candidate skipping on the
    path-resolution hot path. Strictness is owned by the ADMISSION layer:
    perf_database's _version_dir_state (via _load_collection_meta_yaml) raises
    ValueError naming the file on a malformed sidecar, so bad metadata still
    surfaces loudly when the database is loaded. The copies of this predicate
    and their strictness (mirroring the _KNOWN_BACKEND_DIRS copy list above):
      aic-core/src/aiconfigurator_core/sdk/perf_database.py
                                       (_version_dir_state — strict, canonical)
      tools/prediction_regression_gate/grid.py  (_dir_is_incomplete — strict)
      tools/sanity_check/create_charts.py       (_dir_is_incomplete — strict)
    """
    meta_path = os.path.join(version_dir, "collection_meta.yaml")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = yaml.safe_load(f)
        except Exception:
            return False
        tables = meta.get("tables") if isinstance(meta, dict) else None
        if not isinstance(tables, dict):
            return False
        return any(isinstance(t, dict) and t.get("status") == "partial" for t in tables.values())
    return os.path.isfile(os.path.join(version_dir, "INCOMPLETE.txt"))


def resolve_op_data_path(system_data_root: str, backend: str, version: str, op_filename: str) -> str:
    """Resolve one op table under the family-first layout, legacy fallback.

    Family dirs are discovered structurally (any first-level dir that is not
    a known backend dir); dirs marked partial (yaml-first, txt fallback — see
    ``_version_dir_is_partial``) are skipped. Candidates run through the
    .parquet->.txt fallback. When nothing exists, returns the legacy-shaped
    path so callers keep their missing-file semantics.
    """
    op_filename = str(op_filename)
    try:
        entries = os.listdir(system_data_root)
    except Exception:
        entries = []
    for entry in entries:
        if entry.startswith(".") or entry in _KNOWN_BACKEND_DIRS:
            continue
        version_dir = os.path.join(system_data_root, entry, backend, version)
        if not os.path.isdir(version_dir) or _version_dir_is_partial(version_dir):
            continue
        candidate = _resolve_perf_data_path(os.path.join(version_dir, op_filename))
        if os.path.exists(candidate):
            return candidate
    legacy = _resolve_perf_data_path(os.path.join(system_data_root, backend, version, op_filename))
    return legacy


def _read_filtered_rows(file_or_sources):
    """Read perf rows from one or more sources. Used by every ``load_*_data``
    in this package.

    Accepts:
      - A single path string: yields all rows. Returns ``None`` if the file is
        missing, an empty list if it exists but has no rows. Preserves the
        legacy distinction the per-op ``load_*`` functions rely on.
      - An iterable of ``(path, kernel_source_filter)`` tuples: yields rows
        from each source in order; missing files are skipped; rows are
        filtered by ``kernel_source`` when a filter is provided. Returns
        ``None`` only if **every** path is missing.

    The order of the returned list mirrors the order of the input sources, so
    when the per-row loaders skip on key conflict, the earliest source wins on
    every coordinate — same first-wins semantic the shared-layer loader needs
    without a separate merge step.

    Lives here (not in ``perf_database``) so the per-op-module loaders can
    import it without a circular dependency on ``perf_database`` at module
    load time.
    """
    if isinstance(file_or_sources, str):
        path = _resolve_perf_data_path(file_or_sources)
        if not os.path.exists(path):
            return None
        return _read_perf_rows(path)

    rows: list[dict] = []
    any_exists = False
    for path, ks_filter in file_or_sources:
        path = _resolve_perf_data_path(path)
        if not os.path.exists(path):
            continue
        any_exists = True
        for row in _read_perf_rows(path):
            if ks_filter is None or row.get("kernel_source") in ks_filter:
                rows.append(row)
    return rows if any_exists else None


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

    # Context-parallel opt-in. Subclasses set True after auditing how they
    # respond to ``seq_split``. Constructing an op with ``seq_split > 1`` on a
    # class that has NOT opted in raises -- protects against a new op silently
    # mis-modeling CP. Token-major ops (GEMM/Embedding/ElementWise/NCCL/AR/P2P)
    # divide their per-rank token count ``x`` by ``self._seq_split`` in query().
    _CP_AWARE: ClassVar[bool] = False

    def __init__(self, name: str, scale_factor: float, *, seq_split: int = 1) -> None:
        if seq_split > 1 and not self._CP_AWARE:
            raise NotImplementedError(
                f"{type(self).__name__} has not been audited for context parallelism "
                f"(seq_split={seq_split}). Set ``_CP_AWARE = True`` on the class after "
                f"verifying query() divides its token-count input by self._seq_split "
                f"(or is handled CP-style-specifically at the model construction site)."
            )
        self._name = name
        self._scale_factor = scale_factor
        # Sequence-axis shard factor (= cp_size under context parallelism). Token-
        # major ops divide ``x`` by this in query(); default 1 means no shard.
        self._seq_split: int = seq_split

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

    Also clears empirical utilization grids and the shared instrumentation
    counter. Util grids are derived from per-op data, so retaining them after
    their source caches are evicted can mix an old custom ``systems_root`` or
    shared-layer view into newly loaded data.

    Note: this does NOT clear the ``@functools.lru_cache`` on the
    ``PerfDatabase.query_*`` wrappers — those caches live on each database
    instance and must be cleared separately via
    ``database.clear_runtime_caches()`` if callers also want to invalidate
    interpolated/extrapolated query results."""
    for cls in _all_operation_subclasses():
        cls.clear_cache()
    # Import lazily to avoid a base <-> util_empirical module cycle at import
    # time. This is part of the same eviction contract as the per-op caches.
    from aiconfigurator_core.sdk.operations import util_empirical

    util_empirical.clear_grid_cache()
    Operation._load_data_call_count.clear()


def warm_all_op_data(database: PerfDatabase) -> None:
    """Eagerly call ``load_data`` on every ``Operation`` subclass against
    ``database``.

    The lazy-load contract (lazy per-op data ownership) defers per-op CSV reads until the
    first query (or the first read of ``database.supported_quant_mode``
    for the op's key). Diagnostic tooling that walks every op's instance
    attribute directly — notebooks, sanity-check scripts, support-matrix
    dumpers — wants the legacy "everything loaded" semantics; this
    helper restores them in one call.

    Idempotent: every ``load_data`` is cache-key gated, so calling this
    repeatedly is cheap. Op classes that don't own CSV data inherit the
    base ``Operation.load_data`` no-op and are walked without effect.

    Production callers that read ``database.supported_quant_mode[<key>]``
    or call ``database.query_<op>(...)`` should NOT use this — those
    paths trigger the lazy load on the ops they actually need, which is
    the whole point of lazy per-op data ownership."""
    for cls in _all_operation_subclasses():
        cls.load_data(database)
