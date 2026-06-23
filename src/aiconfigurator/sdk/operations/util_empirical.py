"""Data-calibrated empirical estimation via SOL-utilization.

Replaces each op's fixed ``scale_factor`` with a utilization read from real
collected data::

    empirical_latency = SOL(query) / util

where ``util = SOL / measured`` (in ``(0, 1]`` after SOL clamping) is taken
best-effort from collected samples by nearest-neighbour lookup in per-axis
normalised log space.

Like util-space silicon interpolation, this clamps to the nearest known util on
extrapolation. But when *no* samples exist for the requested slice (no own-shape,
no cross-shape/sibling transfer reference), it raises
:class:`~aiconfigurator.sdk.errors.EmpiricalNotImplementedError` rather than
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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

Coords = tuple[float, ...]


def _as_components(sol) -> tuple[float, ...]:
    """Normalise a SOL value to a component tuple.

    A scalar -> one component (the legacy single-SOL behaviour). A ``(sol_compute,
    sol_mem)`` tuple -> two components, so utilisation can be tracked per roofline
    bound (MFU and achieved-BW) instead of only for whichever bound dominates.
    Splitting matters for transfer ACROSS the binding regime (a source point that
    is compute-bound predicting a target that is memory-bound, or vice versa).
    """
    if isinstance(sol, tuple | list):
        return tuple(float(x) for x in sol)
    return (float(sol),)


@dataclass(frozen=True)
class UtilSample:
    coords: Coords  # continuous-axis coordinates of a collected point
    utils: tuple[float, ...]  # per-component SOL_i / measured (compute, mem), in (0, 1]


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


def build_samples(node, depth: int, sol_fn: Callable[[Coords], object]) -> list[UtilSample]:
    """Flatten a nested grid into util samples. ``sol_fn(coords)`` returns either a
    scalar SOL (one component) or a ``(sol_compute, sol_mem)`` tuple."""
    samples = []
    for coords, leaf in iter_grid(node, depth):
        lat = leaf_latency(leaf)
        if lat and lat > 0:
            sols = _as_components(sol_fn(coords))
            if any(s > 0 for s in sols):
                samples.append(UtilSample(tuple(float(c) for c in coords), tuple(s / lat for s in sols)))
    return samples


class UtilGrid:
    """Nearest-neighbour util lookup in per-axis normalised log space.

    Nearest-neighbour clamps naturally on extrapolation (a query past the grid
    edge resolves to the boundary sample's util), which is the conservative
    behaviour we want -- collected boundary utilization is already flat. Each
    sample carries a per-component utilisation vector (compute / mem).
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
        self._utils = np.asarray([s.utils for s in samples], dtype=float)  # (n_samples, n_components)

    def util(self, query: Coords) -> tuple[float, ...] | None:
        if not self.samples:
            return None
        q = (np.log(np.maximum(np.asarray(query, dtype=float), 1e-9)) - self._mins) / self._spans
        dist2 = ((self._norm - q) ** 2).sum(axis=1)
        return tuple(float(u) for u in self._utils[int(dist2.argmin())])


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
    slice (``KeyError``) -- returns ``None``; :func:`estimate` then raises
    :class:`EmpiricalNotImplementedError` (no fabricated constant).
    """
    try:
        return get_grid(cache_key, lambda: UtilGrid(build_samples(slice_fn(), depth, sol_fn)))
    except Exception:
        return None


def estimate(sol_query, query: Coords, grid: UtilGrid | None, util_scale: float = 1.0):
    """Return ``(latency_ms, util)`` from the util grid, or raise.

    ``sol_query`` is the query's SOL: a scalar (single roofline bound -- legacy
    behaviour) or a ``(sol_compute, sol_mem)`` tuple. With components, the latency
    is reconstructed per bound and the binding one wins::

        latency = max_i( sol_query_i / (util_i * util_scale) )

    For a scalar this reduces exactly to ``sol_query / (util * util_scale)``. The
    per-bound form is what makes transfer correct ACROSS the binding regime: a
    compute-bound source point predicting a memory-bound target (or vice versa)
    uses each bound's own measured utilisation rather than the single dominant one.

    Raises :class:`EmpiricalNotImplementedError` when no util sample is available
    for the slice (``grid`` is ``None`` / empty) -- there is no own-shape,
    cross-shape, or sibling data to calibrate from, so we surface the gap instead
    of inventing a ``SOL / constant`` placeholder.

    ``util_scale`` is the cross-op level-alignment hook (default 1.0 = no change).
    A CROSS-OP transfer borrowing a *different* op's util grid passes a scale ``k``
    (supplied by the modeller) so the borrowed util is pulled to the target op's
    level (e.g. MLA runs ~1.4x the SOL-utilisation of MHA). Manual injection point.
    """
    util = grid.util(query) if grid is not None else None
    if util is not None:
        sols = _as_components(sol_query)
        cands = [s / (u * util_scale) for s, u in zip(sols, util, strict=False) if u > 0 and s > 0]
        if cands:
            return max(cands), util
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
