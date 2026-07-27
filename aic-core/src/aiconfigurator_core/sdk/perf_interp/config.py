# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-op config schema for the perf-table interpolation engine (v2).

Declarative interface only: each op describes its table's shape and physics in
one ``OpInterpConfig`` record; a single shared resolver engine (next commit)
executes it. Adding an op = adding a record, not a code path.

The problem
-----------
A perf table is a ragged nested dict ``data[x][y][z] -> leaf`` (leaf = latency
float or ``{"latency", "energy"}``). Exactly two table shapes exist today, and
the engine models exactly those two — nothing more generic:

* **Scattered sites + curve** (GEMM): ``(n, k)`` are discrete real matmul
  shapes — scattered, NOT a Cartesian grid (the reverse shape usually doesn't
  exist, and dense m-sweeps are collected at specific shapes only). Modeled as
  independent sites, each owning its own m-curve.
* **Grid, possibly corner-truncated** (attention / MLA): near-regular
  (heads, seq, batch) grids where the large-seq x large-batch corner is omitted
  (OOM / collection cost), leaving a staircase frontier.

Resolution order (the whole engine in four steps)
-------------------------------------------------
1. **exact hit** -> return the measured leaf verbatim.
2. **resolve within the data**
   - ``Grid``: descend the nesting; per level, exact key collapses the level,
     otherwise bracket between existing keys and blend in ``value_transform``
     space; a missing branch is dropped and weight renormalized (raggedness).
   - ``ScatteredSites``: if the site (e.g. exact ``(n, k)``) is collected,
     evaluate its curve alone; otherwise pick the nearest sites (log-space
     distance, filtered to sites whose curve range covers the query) and
     combine their curve evaluations in UTIL space by inverse-distance weight.
3. **beyond the collected range** -> hold the boundary util
   (``util = SOL/latency``, anchored on the median of the last ``k_tail``
   points so a sawtooth edge doesn't bias it) and return
   ``latency = SOL(query) / util``. Extrapolation is UNBOUNDED by design:
   outside the data, the analytic SOL is the only signal we have — holding
   measured efficiency and letting physics carry the growth is the honest
   answer at any distance, so there is no distance cap.
4. **nothing to anchor on** -> genuine miss: raise. (Empty table / empty
   branch, or no site within ``max_site_distance``.) The engine never
   fabricates a value from nothing.

Invariants (deliberately NOT knobs)
-----------------------------------
* Site **distances** are measured in log space (scale-free shape similarity).
  Bracket **blend weights** use the plain coordinate — curvature is what
  ``value_transform`` is for (raw is exact for ~linear physics, sqrt for
  ~seq^2), and a log-space lerp would break that exactness.
* **Cross-site transfer and extrapolation are always in util space.** A
  neighbouring shape's latency is meaningless at the query shape, but its
  efficiency transfers, and ``SOL(query)`` re-applies the correct scaling.
* **A measured point is returned exactly** — the engine never smooths
  collected data (protects the GEMM-m wave-quantization sawtooth; fine
  structure is the collector's job to sample, ours to preserve).
* Energy rides along: interpolate measured power (= energy/latency, smooth and
  bounded) with the same weights, then ``energy = power * latency``.

The one knob that stays a knob
------------------------------
``value_transform`` — the space used for interpolation BETWEEN measured points
of the same slice/grid. Between two bracketing anchors SOL roughly cancels, so
util buys nothing there and its ``max(compute, mem)`` regime kink can hurt;
``sqrt`` linearises the ~seq^2 curvature of context attention. Defaults are set
per op from leave-one-out evidence and re-decided by the LOO harness — not by
taste.

Deliberately not built
----------------------
No extrapolation distance cap (see step 3). No per-axis mode enum and no EXACT
mode (an exact key collapses a level naturally). No ``on_miss`` policy knob
(a genuine miss raises; that is the SDK contract). No plugin/Protocol layer for
hypothetical future engines. No Delaunay/griddata/RBF/learned surrogate. No
per-axis coordinate-transform knob (log is fixed).

Validation
----------
Leave-one-out against MEASURED latency (hold out a collected point / an entire
site / the boundary shell; predict; compare). Deviation from a previous
system's prediction is a sanity signal only — single-digit % is expected and
fine; only 20-30%+ warrants investigation. Every knob above must win its LOO
A/B or be deleted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class ValueTransform(str, Enum):
    """Space used for interpolation between measured points of one slice/grid.

    (Cross-site transfer and extrapolation are always util — not configurable.)
    """

    #: Interpolate latency directly. Default; right for near-linear axes (GEMM m,
    #: generation seq).
    RAW = "raw"
    #: Interpolate sqrt(latency); linearises ~seq^2 curvature (context attention).
    SQRT = "sqrt"
    #: Interpolate util = SOL/latency. Opt-in: SOL cancels between two anchors so
    #: this rarely beats RAW in-slice, and the max() kink can hurt. Keep only for
    #: ops where it measurably wins leave-one-out.
    UTIL = "util"


@dataclass(frozen=True)
class ScatteredSites:
    """Resolver for scattered-sites-plus-curve tables (GEMM).

    The ``site_axes`` values (e.g. ``(n, k)``) identify a collected site; each
    site owns a dense-ish sweep along ``curve_axis`` (e.g. ``m``). Sites never
    form a Cartesian product and are never rectangularized.
    """

    #: Axis names that jointly identify a site, e.g. ("n", "k").
    site_axes: tuple[str, ...]
    #: The swept axis interpolated within one site's curve, e.g. "m".
    curve_axis: str
    #: How many nearest sites to blend (inverse-distance, util space) when the
    #: query site is not collected. 1 = single nearest.
    nn_sites: int = 4
    #: Log-space distance gate: no site within this radius -> genuine miss.
    #: None = always accept the nearest site.
    max_site_distance: float | None = None
    #: Only consider sites whose curve range covers the query coordinate (a
    #: decode-only site with m<=64 must not answer an m=8192 query). If no site
    #: covers it, fall back to nearest sites with per-site curve-end util-hold.
    require_curve_coverage: bool = True
    #: Boundary-util anchor = median of the last k_tail curve points (sawtooth-
    #: robust). 1 = plain boundary point. A curve with fewer than k_tail points
    #: anchors on the median of what exists (degrading to the plain boundary
    #: point) — k_tail is a robustness knob, not a data-sufficiency gate; a
    #: genuine miss is raised only when NO positive-util anchor exists.
    k_tail: int = 3


@dataclass(frozen=True)
class Grid:
    """Resolver for grid-like, possibly corner-truncated tables (attention/MLA).

    Descends the table's own nesting order; per level: exact key collapses the
    level, otherwise bracket + blend; beyond the collected range (including the
    truncated corner) clamp and hold the boundary util.
    """

    #: Boundary-util anchor = median of the last k_tail points along the
    #: innermost axis. 1 = plain boundary point (grids have no sawtooth).
    #: Fewer than k_tail collected points -> median of what exists, same as
    #: ScatteredSites (never a miss on its own).
    k_tail: int = 1


@dataclass(frozen=True)
class OpInterpConfig:
    """Everything op-specific the shared engine needs. One record per op family."""

    #: Names of the nested-dict levels, outer -> inner. Usually 3 (GEMM m/n/k;
    #: attention heads/seq/batch) but N is supported — context DSA/CSA carries a
    #: past-KV axis: [num_heads][prefix][seq][batch].
    axes: tuple[str, ...]
    #: Which of the two table shapes this op is.
    resolver: ScatteredSites | Grid
    #: Analytic speed-of-light SOL(x, y, z) -> float in axes order (the op's
    #: existing roofline, max(compute, mem)). Required: util-hold extrapolation
    #: and cross-site transfer are built on it.
    sol_fn: Callable[..., float]
    #: In-slice interpolation space (see module docstring).
    value_transform: ValueTransform = ValueTransform.RAW
    #: Curvature is PER-AXIS, not per-table: attention is ~seq^2 along seq but
    #: ~linear along batch/heads, so sqrt must apply only when blending along
    #: seq (LOO: global sqrt 9.4% vs seq-only sqrt beats raw's 2.0%; the legacy
    #: pre-expansion's sqrt_y_value=True encoded the same fact). None = apply
    #: value_transform on every axis.
    transform_axis: str | None = None

    def __post_init__(self) -> None:
        if len(self.axes) < 1:
            raise ValueError(f"need at least 1 axis, got {self.axes}")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(f"duplicate axis names in {self.axes}")
        if self.transform_axis is not None and self.transform_axis not in self.axes:
            raise ValueError(f"transform_axis {self.transform_axis!r} is not one of the table axes {self.axes}")
        if self.resolver.k_tail < 1:
            raise ValueError(f"k_tail must be >= 1, got {self.resolver.k_tail}")
        if isinstance(self.resolver, Grid) and self.value_transform is ValueTransform.UTIL:
            raise ValueError("in-slice UTIL transform is not wired for Grid (no op has won LOO with it)")
        if isinstance(self.resolver, ScatteredSites):
            names = set(self.axes)
            unknown = [a for a in (*self.resolver.site_axes, self.resolver.curve_axis) if a not in names]
            if unknown:
                raise ValueError(f"resolver references unknown axes {unknown}; table axes are {self.axes}")
            if self.resolver.curve_axis in self.resolver.site_axes:
                raise ValueError(f"curve_axis {self.resolver.curve_axis!r} cannot also be a site axis")
            if len(set(self.resolver.site_axes)) != len(self.resolver.site_axes):
                raise ValueError(f"duplicate site axes in {self.resolver.site_axes}")
            if len(self.resolver.site_axes) + 1 != len(self.axes):
                raise ValueError("every axis must be either a site axis or the curve axis")


# ---------------------------------------------------------------------------
# Example records (illustrative — the real sol_fn is injected per (op, database)
# from the op's existing get_sol). Adding an op = adding a record here.
# ---------------------------------------------------------------------------


def gemm_config(sol_fn: Callable[[float, float, float], float]) -> OpInterpConfig:
    """GEMM data[m][n][k]: (n,k) = scattered real shapes, m = swept curve.
    Near-linear in m -> RAW in-slice; SOL ~ m carries extrapolation.

    max_site_distance = 2.0 octaves joint (n,k) distance (~4x per dimension):
    util transfers well between comparable shapes (and SOL re-applies the exact
    m*n*k scaling), but a shape with no collected neighbour within ~4x has no
    efficiency basis -> structured miss. Tune with the site-holdout LOO."""
    return OpInterpConfig(
        axes=("m", "n", "k"),
        resolver=ScatteredSites(site_axes=("n", "k"), curve_axis="m", max_site_distance=2.0),
        sol_fn=sol_fn,
    )


def context_grid_config(sol_fn: Callable[[float, float, float], float]) -> OpInterpConfig:
    """CONTEXT-type grid ops: data[num_heads][seq_len][batch], latency ~ seq^2
    along seq but ~linear along heads/batch -> SQRT on the seq axis only. The
    corner-truncated staircase is ordinary out-of-range util-hold.

    Consumers: context attention (GQA), encoder attention (full N^2), context
    MLA, MLAModule-context, WideEP context MLA."""
    return OpInterpConfig(
        axes=("num_heads", "seq_len", "batch"),
        resolver=Grid(),
        sol_fn=sol_fn,
        value_transform=ValueTransform.SQRT,
        transform_axis="seq_len",
    )


def generation_grid_config(sol_fn: Callable[[float, float, float], float]) -> OpInterpConfig:
    """GENERATION-type grid ops: data[num_heads][batch][seq_len], ~linear in
    seq -> RAW everywhere.

    Consumers: generation attention, generation MLA, MLAModule-generation,
    WideEP generation MLA."""
    return OpInterpConfig(
        axes=("num_heads", "batch", "seq_len"),
        resolver=Grid(),
        sol_fn=sol_fn,
    )


# Named aliases (the physics recipe is the config; ops share it).
context_attention_config = context_grid_config
generation_attention_config = generation_grid_config


#: Registry sketch: op name -> factory(sol_fn) -> OpInterpConfig.
#: dsa (4-axis, raw) gets its own record when migrated.
OP_CONFIG_FACTORIES: dict[str, Callable[[Callable[[float, float, float], float]], OpInterpConfig]] = {
    "gemm": gemm_config,
    "context_attention": context_grid_config,
    "generation_attention": generation_grid_config,
}


__all__ = [
    "OP_CONFIG_FACTORIES",
    "Grid",
    "OpInterpConfig",
    "ScatteredSites",
    "ValueTransform",
    "context_attention_config",
    "context_grid_config",
    "gemm_config",
    "generation_attention_config",
    "generation_grid_config",
]
