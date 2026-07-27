# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared resolver engine for perf-table interpolation (v2).

Executes the four-step resolution declared in :mod:`config`:

    1. exact hit            -> return the measured leaf verbatim
    2. resolve in the data  -> Grid: nested bracket+blend (value_transform space)
                               ScatteredSites: site curve eval; unknown site ->
                               nearest-site transfer in util space
    3. beyond the range     -> hold the boundary util (k_tail median anchor),
                               latency = SOL(query) / util
    4. nothing to anchor on -> raise InterpolationDataNotAvailableError

The engine is N-axis: tables are 3 levels (GEMM m/n/k; attention heads/seq/batch)
or more (context DSA/CSA is [heads][prefix][seq][batch] — the past-KV axis).
Every walk recurses on ``len(cfg.axes)``; nothing assumes 3.

Conventions:
- Bracket weights use the plain (linear) coordinate; curvature is handled by
  ``value_transform`` (raw for ~linear physics, sqrt for ~seq^2). Site
  DISTANCES use log2 coordinates (scale-free shape similarity).
- A leaf is a float (legacy: latency only) or ``{"latency", "power", "energy"}``.
  Energy is carried as average power (= energy/latency, smooth and bounded),
  blended with the same weights as latency, then re-multiplied.
- Misses raise ``errors.InterpolationDataNotAvailableError`` (a
  ValueError), the same structured error the legacy path used, so op-level
  ``PerfDataNotAvailableError`` wrapping keeps working unchanged.
"""

from __future__ import annotations

import bisect
import logging
import math
import statistics
from collections import OrderedDict

from aiconfigurator_core.sdk.errors import InterpolationDataNotAvailableError
from aiconfigurator_core.sdk.perf_interp.config import OpInterpConfig, ScatteredSites, ValueTransform


class _OutOfRangeError(Exception):
    """Internal: a coordinate fell outside the collected range (-> util-hold)."""


# ---------------------------------------------------------------------------
# Leaf and value-space helpers
# ---------------------------------------------------------------------------


def _leaf_lat(leaf) -> float:
    return float(leaf["latency"]) if isinstance(leaf, dict) else float(leaf)


def _leaf_power(leaf) -> float:
    """Average power = energy / latency; falls back to the leaf's explicit
    "power" field when energy is absent (some tables carry power-only rows);
    0.0 for legacy float leaves."""
    if isinstance(leaf, dict):
        lat = float(leaf.get("latency", 0.0))
        energy = float(leaf.get("energy", 0.0) or 0.0)
        if energy > 0 and lat > 0:
            return energy / lat
        return float(leaf.get("power", 0.0) or 0.0)
    return 0.0


def _to_space(vt: ValueTransform, lat: float) -> float:
    if vt is ValueTransform.SQRT:
        return math.sqrt(lat) if lat > 0 else 0.0
    return lat


def _from_space(vt: ValueTransform, v: float) -> float:
    if vt is ValueTransform.SQRT:
        return v * v
    return v


def _miss(cfg: OpInterpConfig, coords, reason: str) -> InterpolationDataNotAvailableError:
    return InterpolationDataNotAvailableError(
        f"perf_interp: no data to anchor query {dict(zip(cfg.axes, coords, strict=True))} ({reason})"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)

_MISSING = object()


def get_value(data_value, metric: str = "latency"):
    """Extract a metric from a query result / table leaf (dict or legacy float)."""
    if isinstance(data_value, dict):
        return data_value.get(metric, 0.0)
    # Legacy format: raw float is latency, power is 0
    return data_value if metric == "latency" else 0.0


def query(cfg: OpInterpConfig, data: dict, *coords):
    """Resolve one query against a raw nested table. Returns the measured leaf
    verbatim on an exact hit, else ``{"latency", "power", "energy"}``."""
    if len(coords) != len(cfg.axes):
        raise ValueError(f"query has {len(coords)} coords; table axes are {cfg.axes}")
    if not data:
        raise _miss(cfg, coords, "empty table")

    node = data  # exact hit: walk the nesting verbatim
    for c in coords:
        if not isinstance(node, dict) or c not in node:
            node = _MISSING
            break
        node = node[c]
    if node is not _MISSING:
        return node

    if isinstance(cfg.resolver, ScatteredSites):
        lat, power = _resolve_scattered(cfg, data, coords)
    else:
        lat, power = _resolve_grid(cfg, data, coords)
    return {"latency": lat, "power": power, "energy": power * lat}


# ---------------------------------------------------------------------------
# ScatteredSites: site curves + nearest-site util transfer (GEMM)
# ---------------------------------------------------------------------------

# Bounded LRU of site indexes, keyed by id(data) with an identity check (the
# strong reference keeps the id stable). Bounded so long-lived processes that
# touch many tables (multiple databases/versions, LOO folds) can't grow it
# without limit; evicting a live table only costs one O(N) index rebuild.
#
# CONTRACT: tables are immutable after load. In-place mutation of a cached
# table keeps its id and would serve a stale index — replace the dict, or call
# ``clear_caches()`` (op-level ``clear_cache()`` does this for you).
_SITE_INDEX_CACHE: OrderedDict[int, tuple] = OrderedDict()
_SITE_INDEX_CACHE_MAX = 32


def clear_caches() -> None:
    """Drop engine-internal caches (site indexes). Op ``clear_cache()`` calls this."""
    _SITE_INDEX_CACHE.clear()


def _walk_leaves(node, depth: int, n_axes: int, prefix: list, out: list) -> None:
    if depth == n_axes:
        out.append((tuple(prefix), node))
        return
    for key, sub in node.items():
        prefix.append(key)
        _walk_leaves(sub, depth + 1, n_axes, prefix, out)
        prefix.pop()


def _site_index(cfg: OpInterpConfig, data: dict):
    res = cfg.resolver
    key = (id(data), cfg.axes, res.site_axes, res.curve_axis)
    cached = _SITE_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is data:
        _SITE_INDEX_CACHE.move_to_end(key)
        return cached[1]

    n_axes = len(cfg.axes)
    curve_pos = cfg.axes.index(res.curve_axis)
    site_pos = tuple(cfg.axes.index(a) for a in res.site_axes)

    leaves: list = []
    _walk_leaves(data, 0, n_axes, [], leaves)
    sites: dict[tuple, list] = {}
    for c, leaf in leaves:
        sites.setdefault(tuple(c[p] for p in site_pos), []).append((c[curve_pos], leaf))
    for curve in sites.values():
        curve.sort(key=lambda t: t[0])
    site_keys = list(sites)
    site_logs = [tuple(math.log2(max(v, 1e-12)) for v in key) for key in site_keys]

    index = (sites, site_keys, site_logs, curve_pos, site_pos, n_axes)
    _SITE_INDEX_CACHE[key] = (data, index)
    if len(_SITE_INDEX_CACHE) > _SITE_INDEX_CACHE_MAX:
        _SITE_INDEX_CACHE.popitem(last=False)
    return index


def _full_coords(n_axes: int, curve_pos: int, site_pos: tuple, curve_val, site_vals) -> tuple:
    coords = [None] * n_axes
    coords[curve_pos] = curve_val
    for p, v in zip(site_pos, site_vals, strict=True):
        coords[p] = v
    return tuple(coords)


def _hold_util(cfg: OpInterpConfig, tail, q, n_axes, curve_pos, site_pos, site_vals, coords):
    """util-hold: anchor on the median util/power of the boundary tail points."""
    utils, powers = [], []
    for cv, leaf in tail:
        lat = _leaf_lat(leaf)
        sol = cfg.sol_fn(*_full_coords(n_axes, curve_pos, site_pos, cv, site_vals))
        if lat > 0 and sol > 0:
            utils.append(sol / lat)
            powers.append(_leaf_power(leaf))
    if not utils:
        raise _miss(cfg, coords, "no positive-util boundary anchor")
    sol_q = cfg.sol_fn(*_full_coords(n_axes, curve_pos, site_pos, q, site_vals))
    if sol_q <= 0:
        raise _miss(cfg, coords, "non-positive SOL at query")
    anchor_util = statistics.median(utils)
    if logger.isEnabledFor(logging.DEBUG):
        boundary = tail[-1][0] if q > tail[-1][0] else tail[0][0]
        logger.debug(
            "perf_interp util-hold (curve): coords=%s anchor_util=%.4g distance=%.2fx",
            dict(zip(cfg.axes, coords, strict=True)),
            anchor_util,
            q / boundary if boundary else float("inf"),
        )
    return sol_q / anchor_util, statistics.median(powers)


def _eval_curve(cfg: OpInterpConfig, curve: list, q, n_axes, curve_pos, site_pos, site_vals, coords):
    """Evaluate one site's curve at coordinate ``q`` -> (latency, power)."""
    ms = [c for c, _ in curve]
    i = bisect.bisect_left(ms, q)
    if i < len(ms) and ms[i] == q:  # exact point on the curve
        leaf = curve[i][1]
        return _leaf_lat(leaf), _leaf_power(leaf)

    k_tail = cfg.resolver.k_tail
    if q < ms[0] or q > ms[-1] or len(curve) < 2:  # beyond the sweep -> util-hold
        tail = curve[:k_tail] if q < ms[0] else curve[-k_tail:]
        return _hold_util(cfg, tail, q, n_axes, curve_pos, site_pos, site_vals, coords)

    (c_lo, leaf_lo), (c_hi, leaf_hi) = curve[i - 1], curve[i]
    w = (q - c_lo) / (c_hi - c_lo)
    lat_lo, lat_hi = _leaf_lat(leaf_lo), _leaf_lat(leaf_hi)
    if cfg.value_transform is ValueTransform.UTIL:
        if lat_lo <= 0 or lat_hi <= 0:
            raise _miss(cfg, coords, "non-positive latency anchor for util interpolation")
        u_lo = cfg.sol_fn(*_full_coords(n_axes, curve_pos, site_pos, c_lo, site_vals)) / lat_lo
        u_hi = cfg.sol_fn(*_full_coords(n_axes, curve_pos, site_pos, c_hi, site_vals)) / lat_hi
        u = u_lo + (u_hi - u_lo) * w
        if not math.isfinite(u) or u <= 0:
            raise _miss(cfg, coords, "non-positive interpolated util")
        lat = cfg.sol_fn(*_full_coords(n_axes, curve_pos, site_pos, q, site_vals)) / u
    else:
        vt = cfg.value_transform
        lat = _from_space(vt, _to_space(vt, lat_lo) + (_to_space(vt, lat_hi) - _to_space(vt, lat_lo)) * w)
    power = _leaf_power(leaf_lo) + (_leaf_power(leaf_hi) - _leaf_power(leaf_lo)) * w
    return lat, power


def _resolve_scattered(cfg: OpInterpConfig, data: dict, coords):
    sites, site_keys, site_logs, curve_pos, site_pos, n_axes = _site_index(cfg, data)
    res = cfg.resolver
    site_key = tuple(coords[p] for p in site_pos)
    q = coords[curve_pos]

    if site_key in sites:  # collected shape: its own curve answers alone
        return _eval_curve(cfg, sites[site_key], q, n_axes, curve_pos, site_pos, site_key, coords)

    # Unknown shape: transfer util from the nearest collected sites.
    if not site_keys:
        raise _miss(cfg, coords, "no sites collected")
    q_log = tuple(math.log2(max(v, 1e-12)) for v in site_key)

    def dist(i: int) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(site_logs[i], q_log, strict=True)))

    candidates = list(range(len(site_keys)))
    if res.require_curve_coverage:
        covering = [i for i in candidates if sites[site_keys[i]][0][0] <= q <= sites[site_keys[i]][-1][0]]
        if covering:  # else: fall back to all sites, each held at its own curve end
            candidates = covering

    ranked = sorted(candidates, key=dist)
    if res.max_site_distance is not None:
        ranked = [i for i in ranked if dist(i) <= res.max_site_distance]
        if not ranked:
            raise _miss(cfg, coords, "no site within max_site_distance")

    wsum = u_acc = p_acc = 0.0
    for i in ranked[: res.nn_sites]:
        neigh = site_keys[i]
        try:
            lat_i, p_i = _eval_curve(cfg, sites[neigh], q, n_axes, curve_pos, site_pos, neigh, coords)
        except InterpolationDataNotAvailableError:  # one bad neighbour must not poison the query
            continue
        sol_i = cfg.sol_fn(*_full_coords(n_axes, curve_pos, site_pos, q, neigh))
        if not (math.isfinite(lat_i) and lat_i > 0 and math.isfinite(sol_i) and sol_i > 0):
            continue
        w = 1.0 / (dist(i) ** 2 + 1e-12)
        u_acc += w * (sol_i / lat_i)
        p_acc += w * p_i
        wsum += w
    if wsum <= 0:
        raise _miss(cfg, coords, "no usable neighbour site")

    sol_q = cfg.sol_fn(*coords)
    if sol_q <= 0:
        raise _miss(cfg, coords, "non-positive SOL at query")
    return sol_q / (u_acc / wsum), p_acc / wsum


# ---------------------------------------------------------------------------
# Grid: nested bracket+blend; out-of-range (incl. truncated corner) -> util-hold
# ---------------------------------------------------------------------------


def _resolve_grid(cfg: OpInterpConfig, data: dict, coords):
    if cfg.value_transform is ValueTransform.UTIL:
        raise NotImplementedError("in-slice UTIL transform is not wired for Grid (no op has won LOO with it)")
    try:
        return _grid_interior(cfg, data, coords, 0)
    except _OutOfRangeError:
        return _grid_hold(cfg, data, coords)


def _grid_interior(cfg: OpInterpConfig, node, coords, depth: int):
    if depth == len(cfg.axes):  # leaf
        return _leaf_lat(node), _leaf_power(node)
    if not node:
        raise _miss(cfg, coords, f"empty branch at axis {cfg.axes[depth]!r}")

    c = coords[depth]
    if c in node:  # exact key collapses this level
        return _grid_interior(cfg, node[c], coords, depth + 1)

    keys = sorted(node)
    if c < keys[0] or c > keys[-1]:
        raise _OutOfRangeError()
    i = bisect.bisect_left(keys, c)
    k_lo, k_hi = keys[i - 1], keys[i]

    results, errors = [], []
    for k in (k_lo, k_hi):
        try:
            results.append((k, _grid_interior(cfg, node[k], coords, depth + 1)))
        except (_OutOfRangeError, InterpolationDataNotAvailableError) as exc:  # ragged branch: drop + renormalize
            errors.append(exc)
    if not results:
        # Both branches failed. Out-of-range anywhere below means the query sits
        # past the staircase frontier -> let util-hold anchor it.
        if any(isinstance(e, _OutOfRangeError) for e in errors):
            raise _OutOfRangeError()
        raise _miss(cfg, coords, f"no usable branch at axis {cfg.axes[depth]!r}")
    if len(results) == 1:
        # One bracket branch dropped (ragged table). Returning the survivor
        # verbatim would CLAMP this axis with no correction -- measured -41%
        # median on one-sided seq-row folds (seq^2 physics, surviving anchor
        # a whole bracket step away). Keep the survivor's resolved value (it
        # carries the measured inner-axis structure) and re-scale along THIS
        # axis by the SOL ratio, i.e. hold the survivor's util across the
        # dropped axis. LOO: 41% -> ~10% median; unlike a full _grid_hold
        # escalation it keeps the max tail bounded (52.9% vs 104.4%).
        k_surv, (lat, p) = results[0]
        snapped = tuple(k_surv if j == depth else coords[j] for j in range(len(coords)))
        sol_q, sol_s = cfg.sol_fn(*coords), cfg.sol_fn(*snapped)
        if math.isfinite(sol_q) and math.isfinite(sol_s) and sol_q > 0 and sol_s > 0:
            return lat * (sol_q / sol_s), p
        return lat, p

    (_, (lat_lo, p_lo)), (_, (lat_hi, p_hi)) = results
    w = (c - k_lo) / (k_hi - k_lo)
    # Curvature is per-axis: apply the transform only when blending along the
    # configured axis (e.g. sqrt on seq); other axes are ~linear -> raw.
    vt = cfg.value_transform
    if cfg.transform_axis is not None and cfg.axes[depth] != cfg.transform_axis:
        vt = ValueTransform.RAW
    lat = _from_space(vt, _to_space(vt, lat_lo) + (_to_space(vt, lat_hi) - _to_space(vt, lat_lo)) * w)
    return lat, p_lo + (p_hi - p_lo) * w


def _grid_hold(cfg: OpInterpConfig, data: dict, coords):
    """Anchor past-the-frontier queries: snap to the nearest collected path,
    hold the boundary util (k_tail median along the innermost axis), and let
    SOL(query) carry the growth."""
    node = data
    snapped = []
    for depth in range(len(cfg.axes) - 1):
        if not node:
            raise _miss(cfg, coords, f"empty branch at axis {cfg.axes[depth]!r}")
        c = coords[depth]
        key = c if c in node else min(node.keys(), key=lambda k: abs(k - c))
        snapped.append(key)
        node = node[key]
    if not node:
        raise _miss(cfg, coords, f"empty branch at axis {cfg.axes[-1]!r}")

    keys = sorted(node)
    c = coords[-1]
    k_tail = cfg.resolver.k_tail
    if c > keys[-1]:
        tail = keys[-k_tail:]
    elif c < keys[0]:
        tail = keys[:k_tail]
    else:  # innermost is in range; an OUTER axis was snapped
        tail = [min(keys, key=lambda k: abs(k - c))]

    utils, powers = [], []
    for t in tail:
        leaf = node[t]
        lat = _leaf_lat(leaf)
        sol = cfg.sol_fn(*snapped, t)
        if lat > 0 and sol > 0:
            utils.append(sol / lat)
            powers.append(_leaf_power(leaf))
    if not utils:
        raise _miss(cfg, coords, "no positive-util boundary anchor")
    sol_q = cfg.sol_fn(*coords)
    if sol_q <= 0:
        raise _miss(cfg, coords, "non-positive SOL at query")
    anchor_util = statistics.median(utils)
    if logger.isEnabledFor(logging.DEBUG):
        c_last = coords[-1]
        edge = keys[-1] if c_last > keys[-1] else keys[0]
        logger.debug(
            "perf_interp util-hold (grid): coords=%s snapped=%s anchor_util=%.4g inner_distance=%.2fx",
            dict(zip(cfg.axes, coords, strict=True)),
            snapped,
            anchor_util,
            (c_last / edge) if edge else float("inf"),
        )
    return sol_q / anchor_util, statistics.median(powers)
