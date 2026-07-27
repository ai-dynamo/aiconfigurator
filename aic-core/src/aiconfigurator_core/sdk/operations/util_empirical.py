# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-calibrated empirical estimation via SOL-utilization.

Replaces each op's fixed ``scale_factor`` with a utilization read from real
collected data::

    empirical_latency = SOL(query) / util

where ``util = SOL / measured`` is a positive effective calibration factor
taken best-effort from collected samples in per-axis normalised log space. It
is not constrained to one: hardware/kernel effects omitted by the analytic SOL
may legitimately produce ``util > 1``. Very large values should be treated as
data/model sanity signals rather than silently clamped. Every grid uses the
same two-neighbour inverse-distance weighting, without requiring a Cartesian
product. Queries outside the measured range are clamped per axis before the
neighbours are selected, so extrapolation freezes boundary utilization.

Like util-space silicon interpolation, this clamps extrapolated query
coordinates to the measured bounds. But when *no* samples exist for the
requested slice (no own-shape, no cross-shape/sibling transfer reference), it raises
:class:`~aiconfigurator_core.sdk.errors.EmpiricalNotImplementedError` rather than
returning a fabricated ``SOL / constant``. Missing coverage thus surfaces
honestly. (The legacy ``fallback_scale`` constant was a placeholder and has been
removed; genuinely table-less ops -- mem / p2p / element-wise -- keep their own
analytic formulas and never call :func:`estimate`.)

Extension seams (designed in, not yet wired):

* **cross-op transfer** (similar-op reuse): :class:`UtilGrid` / :func:`estimate`
  are agnostic to where samples come from. A future reference-op layer can
  build a grid from a *similar* op's data (matched on shape features) and pass
  it here unchanged -- only the sample source moves.
* **cross-precision** (quant reuse): the *slice selector* that decides which
  collected slice feeds :func:`build_samples` lives in the caller; it may fall
  back to a different quant's slice when the exact one is absent. The util
  reconstruction below does not care which quant produced the samples.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from aiconfigurator_core.sdk.errors import (
    EmpiricalNotImplementedError,
    InterpolationDataNotAvailableError,
    PerfDataNotAvailableError,
)

Coords = tuple[float, ...]


# ---------------------------------------------------------------------------
# Provenance capture: record which empirical path produced a value, so a run's
# overall data source can be summarised (e.g. the support matrix labelling a
# config SILICON vs HYBRID and at which transfer tier). Any call to estimate()
# means the silicon path was missed and an empirical/transfer value was used; the
# caller tags the specific tier. Recording is a no-op unless a capture is active.
# Tags ordered by DECREASING confidence (later = relies on more aggressive transfer).
# ---------------------------------------------------------------------------
PROVENANCE_ORDER: tuple[str, ...] = (
    "silicon",  # pure silicon table data (never recorded here; the default when nothing fired)
    "empirical",  # own-shape util (no transfer)
    "xshape",  # cross-shape, same quant
    "xquant",  # cross-quant, same profile
    "xprofile",  # cross-quant, cross profile
    "xop",  # cross-op (borrowed a different op's util)
)
_PROVENANCE_RANK = {tag: i for i, tag in enumerate(PROVENANCE_ORDER)}
_PROVENANCE: contextvars.ContextVar = contextvars.ContextVar("aic_provenance", default=None)


def note_provenance(tag: str) -> None:
    """Record that an empirical path of kind ``tag`` fired (no-op outside a capture)."""
    sink = _PROVENANCE.get()
    if sink is not None:
        sink.add(tag)


@contextlib.contextmanager
def capture_provenance():
    """Collect the set of empirical-path tags fired within the block."""
    sink: set[str] = set()
    token = _PROVENANCE.set(sink)
    try:
        yield sink
    finally:
        _PROVENANCE.reset(token)


def worst_provenance(tags) -> str:
    """The least-confident tag in ``tags`` (the run's effective data source);
    ``"silicon"`` when empty (no empirical path fired)."""
    return max(tags, key=lambda t: _PROVENANCE_RANK.get(t, 0), default="silicon")


@dataclass(frozen=True)
class UtilSample:
    coords: Coords  # continuous-axis coordinates of a collected point
    util: float  # positive effective calibration factor: SOL / measured


def leaf_latency(leaf) -> float | None:
    """Extract a latency scalar from a grid leaf (scalar / dict / PerformanceResult)."""
    if isinstance(leaf, dict):
        return leaf.get("latency")
    return getattr(leaf, "latency", leaf)


def iter_grid(node, depth: int, prefix: Coords = ()):
    """Yield ``(coords, leaf)`` for a ``depth``-deep nested dict keyed by axis values."""
    if depth == 0:
        yield prefix, node
        return
    for key, child in node.items():
        yield from iter_grid(child, depth - 1, prefix + (key,))


def build_samples(node, depth: int, sol_fn: Callable[[Coords], float]) -> list[UtilSample]:
    """Flatten a nested grid into util samples. ``sol_fn(coords) -> sol_ms``."""
    samples = []
    for coords, leaf in iter_grid(node, depth):
        lat = leaf_latency(leaf)
        if lat and lat > 0:
            sol = sol_fn(coords)
            if sol and sol > 0:
                samples.append(UtilSample(tuple(float(c) for c in coords), sol / lat))
    return samples


class UtilGrid:
    """Two-neighbour util lookup in per-axis normalised log space.

    The query is clamped independently on every axis, then the two nearest
    samples are combined with inverse-distance weights (``k=2``, ``p=1``).
    Exact hits return the collected utilization unchanged. This works for
    ragged grids without requiring operation-specific Cartesian bracketing;
    callers remain responsible for slicing categorical/kernel-regime axes.
    """

    def __init__(self, samples: list[UtilSample]):
        self.samples = samples
        self.reference_provenance: str | None = None
        if not samples:
            return
        coords = np.asarray([s.coords for s in samples], dtype=float)
        logc = np.log(np.maximum(coords, 1e-9))
        self._mins = logc.min(axis=0)
        self._maxs = logc.max(axis=0)
        spans = self._maxs - self._mins
        self._spans = np.where(spans > 0, spans, 1.0)
        self._norm = (logc - self._mins) / self._spans
        self._utils = np.asarray([s.util for s in samples], dtype=float)

    def util(self, query: Coords) -> float | None:
        if not self.samples:
            return None
        query_log = np.log(np.maximum(np.asarray(query, dtype=float), 1e-9))
        q = np.clip((query_log - self._mins) / self._spans, 0.0, 1.0)
        distances = np.sqrt(((self._norm - q) ** 2).sum(axis=1))
        order = np.argsort(distances, kind="stable")

        # Preserve exact collected points (and deterministic first-sample wins
        # for duplicate/log-floor-collapsed coordinates).
        if distances[order[0]] == 0.0:
            return float(self._utils[order[0]])

        nearest = order[: min(2, len(order))]
        weights = 1.0 / distances[nearest]
        return float(np.dot(self._utils[nearest], weights) / weights.sum())


# Process-lifetime cache of built grids. Collected data is itself cached for the
# process lifetime, so a plain dict keyed by an op/slice identifier is enough.
_GRID_CACHE: dict = {}
_REFERENCE_SELECTION_CACHE: dict = {}


def get_grid(cache_key, builder: Callable[[], UtilGrid]) -> UtilGrid:
    grid = _GRID_CACHE.get(cache_key)
    if grid is None:
        grid = builder()
        _GRID_CACHE[cache_key] = grid
    return grid


def clear_grid_cache() -> None:
    """Drop all built util grids.

    Grid keys include the concrete selected data identity, which isolates live
    database views. PerfDatabase still clears eagerly when a caller mutates one
    instance's mode or transfer policy in place, both for backward compatibility
    and to discard grids that are no longer reachable under that policy.
    """
    _GRID_CACHE.clear()
    _REFERENCE_SELECTION_CACHE.clear()


def require_data_slice(root: object, *keys: object) -> object:
    """Return a requested nested perf-data slice or raise the typed coverage signal.

    Missing keys and an absent final node mean the requested measured slice is
    not covered. A non-mapping intermediate node is malformed data and raises
    ``TypeError`` so schema problems are not converted into fallback.
    """
    if root is None:
        raise PerfDataNotAvailableError("The requested silicon performance table is not loaded.")

    node = root
    traversed: list[object] = []
    for key in keys:
        if not isinstance(node, Mapping):
            raise TypeError(
                f"Malformed performance data at path {tuple(traversed)!r}: "
                f"expected a mapping, got {type(node).__name__}."
            )
        if key not in node:
            raise PerfDataNotAvailableError(
                "Missing silicon data for the requested lookup; "
                f"requested slice {(*traversed, key)!r} is not available."
            )
        node = node[key]
        traversed.append(key)

    if node is None or (isinstance(node, Mapping) and not node):
        raise PerfDataNotAvailableError(
            f"Missing silicon data for the requested lookup; requested slice {tuple(traversed)!r} is not available."
        )
    return node


def grid_for(cache_key, slice_fn: Callable[[], object], sol_fn: Callable[[Coords], float], depth: int):
    """Best-effort build/fetch of a :class:`UtilGrid`.

    ``slice_fn()`` returns the nested data sub-grid for the requested slice
    (and may load data lazily). Typed data-coverage failures return ``None``;
    :func:`estimate` then raises :class:`EmpiricalNotImplementedError` (no
    fabricated constant). Programming and schema errors propagate to the
    caller instead of being misreported as missing empirical coverage.
    """
    try:
        node = slice_fn()
        # The logical key names an op/shape; the node identity names the
        # concrete loaded table behind this database view. Two live views with
        # different roots/shared-layer data therefore cannot alias one grid.
        identity_key = (cache_key, id(node))
        return get_grid(identity_key, lambda: UtilGrid(build_samples(node, depth, sol_fn)))
    except (PerfDataNotAvailableError, InterpolationDataNotAvailableError):
        return None


def estimate(sol_query: float, query: Coords, grid: UtilGrid | None, util_scale: float = 1.0, provenance="empirical"):
    """Return ``(latency_ms, util)`` from the util grid, or raise.

    Raises :class:`EmpiricalNotImplementedError` when no util sample is available
    for the slice (``grid`` is ``None`` / empty) -- there is no own-shape,
    cross-shape, or sibling data to calibrate from, so we surface the gap instead
    of inventing a ``SOL / constant`` placeholder.

    ``util_scale`` is the cross-op level-alignment hook (default 1.0 = no change,
    used for own-data / same-op transfer). When a CROSS-OP transfer borrows a
    *different* op's util grid, the caller passes a scale ``k`` (supplied by the
    modeller) so ``latency = SOL / (util * k)`` -- this pulls the borrowed util to
    the target op's level (e.g. MLA runs ~1.4x the SOL-utilisation of MHA). Not
    auto-calibrated and not table-backed by design; it is a manual injection point.
    """
    util = grid.util(query) if grid is not None else None
    if util and util > 0:
        note_provenance(provenance)
        return sol_query / (util * util_scale), util
    raise EmpiricalNotImplementedError(
        f"No empirical utilisation data to estimate this op at query={query}: "
        "no own-shape, cross-shape, or sibling transfer reference available."
    )


# ---------------------------------------------------------------------------
# Cross-shape transfer (observation 5): when an op's own slice has no data,
# borrow the util curve of the *nearest* collected sibling slice (matched on
# categorical shape features), reconstructed with the query's own SOL:
#
#     latency_query(c) = SOL_query(c) / util_ref(c),  util_ref = SOL_ref / measured_ref
#
# SOL absorbs the structural difference (experts/topk/hidden/...); util carries
# only the shared kernel-efficiency. ``ReferenceCandidate.sol_fn`` MUST compute
# SOL with the *reference* slice's shape (not the query's), or the ratio is wrong.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCandidate:
    features: Coords  # categorical shape features for the nearest-neighbour match
    node: object  # the reference slice's nested data sub-grid
    sol_fn: Callable[[Coords], float]  # SOL bound to THIS reference's shape
    provenance: str | None = None


def _nearest_candidate(query_features: Coords, candidates: list[ReferenceCandidate]) -> ReferenceCandidate:
    feats = np.log(np.maximum(np.asarray([c.features for c in candidates], dtype=float), 1e-9))
    mins = feats.min(axis=0)
    spans = np.where(feats.max(axis=0) - mins > 0, feats.max(axis=0) - mins, 1.0)
    q = (np.log(np.maximum(np.asarray(query_features, dtype=float), 1e-9)) - mins) / spans
    dist2 = (((feats - mins) / spans) - q) ** 2
    return candidates[int(dist2.sum(axis=1).argmin())]


def grid_from_reference(
    cache_key,
    query_features: Coords,
    candidates_fn: Callable[[], list],
    depth: int,
    *,
    selection_key=None,
):
    """Util grid borrowed from the nearest sibling slice.

    ``candidates_fn()`` returns a list of :class:`ReferenceCandidate` (the op
    enumerates its sibling slices). Picks the nearest by ``features`` in per-dim
    normalised log space and builds the grid from that sibling's data using the
    sibling's own ``sol_fn``. Typed coverage misses and an empty candidate list
    return no usable samples; programming/schema failures propagate.
    """

    def select_reference():
        selected_cache_key = (cache_key, tuple(query_features), selection_key) if selection_key is not None else None
        if selected_cache_key is not None and selected_cache_key in _REFERENCE_SELECTION_CACHE:
            return _REFERENCE_SELECTION_CACHE[selected_cache_key]
        candidates = candidates_fn()
        ref = _nearest_candidate(query_features, candidates) if candidates else None
        if selected_cache_key is not None:
            _REFERENCE_SELECTION_CACHE[selected_cache_key] = ref
        return ref

    try:
        ref = select_reference()
        if ref is None:
            return UtilGrid([])
        # Reference selection is policy-dependent. Select first, then key by
        # the chosen table identity so interleaved policies cannot reuse one
        # another's calibration grid.
        identity_key = (cache_key, id(ref.node), selection_key, ref.provenance)

        def build_reference_grid():
            grid = UtilGrid(build_samples(ref.node, depth, ref.sol_fn))
            grid.reference_provenance = ref.provenance
            return grid

        return get_grid(identity_key, build_reference_grid)
    except (PerfDataNotAvailableError, InterpolationDataNotAvailableError):
        return None
