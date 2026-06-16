"""Data-calibrated empirical estimation via SOL-utilization.

Replaces each op's fixed ``scale_factor`` with a utilization read from real
collected data::

    empirical_latency = SOL(query) / util

where ``util = SOL / measured`` (in ``(0, 1]`` after SOL clamping) is taken
best-effort from collected samples by nearest-neighbour lookup in per-axis
normalised log space.

This is the *non-failing sibling* of util-space silicon interpolation: silicon
requires a full interpolation bracket and raises when it is missing; this
clamps to the nearest known util and always returns a value. When no samples
exist for the requested slice it falls back to the op's constant
``fallback_scale`` (today's behaviour), so an op with zero data is unchanged.

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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Coords = tuple[float, ...]


@dataclass(frozen=True)
class UtilSample:
    coords: Coords  # continuous-axis coordinates of a collected point
    util: float  # SOL / measured at that point, in (0, 1]


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
    """Nearest-neighbour util lookup in per-axis normalised log space.

    Nearest-neighbour clamps naturally on extrapolation (a query past the grid
    edge resolves to the boundary sample's util), which is the conservative
    behaviour we want -- collected boundary utilization is already flat.
    """

    def __init__(self, samples: list[UtilSample]):
        self.samples = samples
        if not samples:
            return
        coords = np.asarray([s.coords for s in samples], dtype=float)
        logc = np.log(np.maximum(coords, 1e-9))
        self._mins = logc.min(axis=0)
        spans = logc.max(axis=0) - self._mins
        self._spans = np.where(spans > 0, spans, 1.0)
        self._norm = (logc - self._mins) / self._spans
        self._utils = np.asarray([s.util for s in samples], dtype=float)

    def util(self, query: Coords) -> float | None:
        if not self.samples:
            return None
        q = (np.log(np.maximum(np.asarray(query, dtype=float), 1e-9)) - self._mins) / self._spans
        dist2 = ((self._norm - q) ** 2).sum(axis=1)
        return float(self._utils[int(dist2.argmin())])


# Process-lifetime cache of built grids. Collected data is itself cached for the
# process lifetime, so a plain dict keyed by an op/slice identifier is enough.
_GRID_CACHE: dict = {}


def get_grid(cache_key, builder: Callable[[], UtilGrid]) -> UtilGrid:
    grid = _GRID_CACHE.get(cache_key)
    if grid is None:
        grid = builder()
        _GRID_CACHE[cache_key] = grid
    return grid


def grid_for(cache_key, slice_fn: Callable[[], object], sol_fn: Callable[[Coords], float], depth: int):
    """Best-effort build/fetch of a :class:`UtilGrid`.

    ``slice_fn()`` returns the nested data sub-grid for the requested slice
    (and may load data lazily). Any failure -- missing data files, an absent
    slice (``KeyError``) -- returns ``None`` so the caller falls back to its
    constant ``fallback_scale``. This is what keeps zero-data ops unchanged.
    """
    try:
        return get_grid(cache_key, lambda: UtilGrid(build_samples(slice_fn(), depth, sol_fn)))
    except Exception:
        return None


def estimate(sol_query: float, query: Coords, grid: UtilGrid | None, fallback_scale: float, util_scale: float = 1.0):
    """Return ``(latency_ms, util)``. ``util`` is ``None`` when the constant
    fallback was used (no samples for the slice).

    ``util_scale`` is the cross-op level-alignment hook (default 1.0 = no change,
    used for own-data / same-op transfer). When a CROSS-OP transfer borrows a
    *different* op's util grid, the caller passes a scale ``k`` (supplied by the
    modeller) so ``latency = SOL / (util * k)`` -- this pulls the borrowed util to
    the target op's level (e.g. MLA runs ~1.4x the SOL-utilisation of MHA). Not
    auto-calibrated and not table-backed by design; it is a manual injection point.
    """
    util = grid.util(query) if grid is not None else None
    if util and util > 0:
        return sol_query / (util * util_scale), util
    return sol_query / fallback_scale, None


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


def _nearest_candidate(query_features: Coords, candidates: list[ReferenceCandidate]) -> ReferenceCandidate:
    feats = np.log(np.maximum(np.asarray([c.features for c in candidates], dtype=float), 1e-9))
    mins = feats.min(axis=0)
    spans = np.where(feats.max(axis=0) - mins > 0, feats.max(axis=0) - mins, 1.0)
    q = (np.log(np.maximum(np.asarray(query_features, dtype=float), 1e-9)) - mins) / spans
    dist2 = (((feats - mins) / spans) - q) ** 2
    return candidates[int(dist2.sum(axis=1).argmin())]


def grid_from_reference(cache_key, query_features: Coords, candidates_fn: Callable[[], list], depth: int):
    """Best-effort util grid borrowed from the nearest sibling slice.

    ``candidates_fn()`` returns a list of :class:`ReferenceCandidate` (the op
    enumerates its sibling slices). Picks the nearest by ``features`` in per-dim
    normalised log space and builds the grid from that sibling's data using the
    sibling's own ``sol_fn``. Returns ``None`` on any failure / no candidates so
    the caller still falls back to its constant.
    """

    def build() -> UtilGrid:
        candidates = candidates_fn()
        if not candidates:
            return UtilGrid([])
        ref = _nearest_candidate(query_features, candidates)
        return UtilGrid(build_samples(ref.node, depth, ref.sol_fn))

    try:
        return get_grid(cache_key, build)
    except Exception:
        return None
