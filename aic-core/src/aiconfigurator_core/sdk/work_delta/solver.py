# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Solving the work-delta unit prices from measured calibration batches.

Each calibration batch was built around one average point holding both totals
fixed, so subtracting the average point's latency removes everything that
depends on the totals alone and leaves the cost of the spread:

    y = T_batch - T_uniform

Two prices, because a prefill batch is charged on two different shapes: the
candidate-scoring trapezoid a row presents by crossing ``topk``, and the
attention pairs every row actually reads. See :mod:`planner` for what each
column does and does not include -- in particular, a sub-``topk`` row's indexer
scan is priced through ``c_mla``, not ``c_idx``.

Nothing here fits both at once. The order is forced by what each segment
can move, and each step subtracts what the previous one already fixed:

    1. c_mla  from the unsaturated segments. Those batches short-circuit the
       indexer on every row, so ``x_idx`` is identically
       zero and the segment pins the dense price on its own. It is fitted per
       batch size rather than per cell -- it is a property of the dense kernel,
       and at a cell whose average request is long it cannot be measured at all,
       because a dense row is capped at ``topk`` tokens while the cell's own
       work grows with ``s_bar^2``.

    2. c_idx  from each cell's saturated segment. Every row is above ``topk``
       so the indexer is pinned there for all of them, which makes the attention
       term linear in ``s``; with ``sum(s)`` conserved its deviation cancels
       exactly and only the gated column survives.

    3. the remaining price from the mixed segment, with whatever step 1 and step 2
       already fixed subtracted out. At a saturated average point that leaves
       one unknown; at an unsaturated one the cell has no saturated segment, so
       ``c_idx`` is unknown too and the mixed data carries both.

The rungs within a segment are not redundancy. A coefficient fitted at one
imbalance magnitude says nothing about whether the relation is linear; rungs
spanning the segment's own range make the residual meaningful, and a residual
above the tolerance is the signal that the linear form does not hold for that
cell rather than that the measurement was noisy.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "CellFit",
    "Measurement",
    "Prices",
    "predict_delta",
    "solve_cell",
]

UNSAT = "unsat"
MIXED = "mixed"
SAT = "sat"

# How far a label must clear the engine's own latency spread at that shape
# before it is allowed into a fit. Measured on GLM-5: at cells whose average
# request is short the prefill step sits at a flat ~240 ms whatever the spread,
# so a batch carrying twice the modelled attention work moves the clock by
# 0.07 ms. Those labels are not noisy measurements of a small effect; they are
# measurements of an effect the step does not have.
MIN_LABEL_SNR = 3.0

# Smallest column deviation worth pricing, in millions of attention-pair reads.
# Absolute, not a fraction of the cell's own work: a relative gate divides a
# real 200 ms delta by a large cell's large denominator and discards it, while
# keeping a 10 ms delta in a small one -- backwards from the physics. Each
# threshold is the machine's own 3-sigma step-time jitter divided by that
# column's measured price, so below it the correction is smaller than the
# spread it would have to be verified against.
#
# Sized from the MEDIAN price, not a typical one -- the prices are not constant
# across cells and are not meant to be, since cells run at different MFU:
#
#   c_idx    1.09 ms per M reads (median of 18 cells, spread 0.14-3.05)
#   c_mla    6.13 ms per M       (median of 14 cells, spread 2.13-44.18)
#   jitter   9.0 ms median 3-sigma  ->  9.0/1.09 = 8.3 -> 10 M,  9.0/6.13 -> 2 M
#
# A cell far from the median therefore gets a gate that is loose or tight for
# it. That is the intended trade: a per-cell gate would be derived from the very
# fit it is supposed to guard. Re-derive all three when the model, parallelism
# or hardware changes -- the formula transfers, these numbers do not.
MIN_ABS_DELTA_IDX_M = 10.0
MIN_ABS_DELTA_MLA_M = 2.0


MIN_USABLE_COLUMN = 1e-9


# Smallest pivot the normalised Gram matrix may present before its columns are
# treated as dependent. Set well above float noise: at 1e-3 the prices are
# already being decided by the third digit of the measurement.
MIN_GRAM_PIVOT = 1e-3


@dataclass(frozen=True)
class Measurement:
    """One measured calibration batch, already reduced against its average point."""

    b: int
    s_bar: int
    p_bar: int
    regime: str
    avg_is_sat: bool
    x_idx: float
    x_mla: float
    y: float
    # Spread of the engine's own latency at this shape, in the same units as
    # ``y``. A label smaller than its own noise carries no information about
    # any coefficient, and the planner cannot know it in advance: the floor it
    # applies is on PREDICTED work, and at a cell whose step is dominated by
    # fixed overhead a large work change moves the clock not at all.
    noise: float = 0.0

    @property
    def usable(self) -> bool:
        return self.noise <= 0.0 or abs(self.y) >= MIN_LABEL_SNR * self.noise

    @property
    def cell(self) -> tuple[int, int, int]:
        return self.b, self.s_bar, self.p_bar


class Prices(Protocol):
    """Anything that carries a calibrated price pair.

    Both `CellFit` and `field.Interpolated` satisfy it, which is what lets the
    exact-hit and interpolated paths share one gated application.
    """

    c_idx: float | None
    c_mla: float | None


@dataclass
class CellFit:
    b: int
    s_bar: int
    p_bar: int
    avg_is_sat: bool
    c_idx: float | None = None
    c_mla: float | None = None
    residuals: dict[str, float] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
    # Labels that never entered the fit because they sat inside the engine's
    # own latency spread. Reported, not silent: a cell that drops most of its
    # batches here is telling you its step is overhead-bound.
    below_noise: int = 0

    @property
    def accepted(self) -> bool:
        # Residuals are diagnostics. A cell whose segments each isolated their
        # own price is determined, however large the scatter of a segment that
        # happened to carry a spare rung.
        return not self.rejected


# ------------------------------------------------------------------- solving


def _project(points, feature, target) -> float | None:
    """Least squares through the origin: the price that best explains ``target``.

    Through the origin because a balanced batch has zero deviation and must cost
    zero extra; an intercept would let the fit charge for imbalance that is not
    there.
    """
    denom = sum(feature(p) ** 2 for p in points)
    if denom <= MIN_USABLE_COLUMN:
        return None
    return sum(target(p) * feature(p) for p in points) / denom


def _relative_residual(points, predict, target) -> float:
    """Scatter around the fit, scaled by the labels themselves.

    Scaled rather than absolute so the tolerance means the same thing at a cell
    whose deltas are milliseconds and one whose deltas are seconds.
    """
    scale = sum(abs(target(p)) for p in points)
    if scale <= 0.0:
        return 0.0
    return sum(abs(target(p) - predict(p)) for p in points) / scale


def _solve_n(points, cols, target) -> tuple[float, ...] | None:
    """Least squares for any number of prices, rejected when ill-conditioned.

    Used when no pure segment survived to pin a coefficient first. Gauss-Jordan
    on the normal equations rather than a library call: the module has no
    dependencies and the systems here are 2x2 or 3x3.

    The guard is on the normalised Gram determinant, which is 1 when the columns
    are orthogonal and 0 when any of them is a combination of the others. Near
    zero the individual prices are decided by noise even though their weighted
    sum is well determined, and reporting them would be inventing numbers.
    """
    n = len(cols)
    # n, not n+1. With exactly n rungs the system is square, so it reproduces
    # its labels whatever the prices are and leaves no residual to read. That is
    # a reason to report no residual, not a reason to refuse: residuals are
    # diagnostics here and never reject a fit, and collection cannot afford a
    # spare rung -- the planner emits two mixed rungs because two is what a
    # mixed segment needs. Demanding a third rejected 2 cells outright whose
    # prices were fully determined.
    if len(points) < n:
        return None
    norms = [math.sqrt(sum(c(p) ** 2 for p in points)) for c in cols]
    if any(v <= MIN_USABLE_COLUMN for v in norms):
        return None
    # Gram matrix of the unit-normalised columns.
    g = [[sum(cols[i](p) * cols[j](p) for p in points) / (norms[i] * norms[j]) for j in range(n)] for i in range(n)]
    rhs = [sum(target(p) * cols[i](p) for p in points) / norms[i] for i in range(n)]
    aug = [row[:] + [rhs[i]] for i, row in enumerate(g)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < MIN_GRAM_PIVOT:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col], strict=True)]
    # Undo the normalisation so the prices are in the original units.
    return tuple(aug[i][n] / norms[i] for i in range(n))


def solve_cell(b: int, s_bar: int, p_bar: int, avg_is_sat: bool, measurements: list[Measurement]) -> CellFit:
    """Fit this average point's two prices, staged by what each segment isolates.

    The regime decides which column is even live, and that is what makes the
    stages exact rather than a least-squares compromise:

        pure saturated    every row is above topk, so the attention term is
                          linear in s and its deviation cancels: only x_idx
                          survives. One rung fixes c_idx outright.
        pure unsaturated  every row is at or below topk, so the top-k selection
                          discarded nothing and x_idx is identically zero. One
                          rung fixes c_mla.
        mixed             both columns are live, but the cell's pure segment has
                          already fixed one, so the remaining price follows from
                          the rungs with a degree of freedom left over.

    A cell expresses exactly two of the three regimes, so one pure rung and one
    mixed rung determine both prices, and the second mixed rung the planner
    emits leaves a residual worth reading. That matches what collection can
    afford: one rung per pure segment and two for mixed.

    Repeated rungs in a pure segment are averaged by median rather than
    least-squares -- the segment is one-dimensional, so each rung is an
    independent estimate of the same ratio, and the median ignores the odd rung
    whose label is dominated by something other than the work.
    """
    fit = CellFit(b, s_bar, p_bar, avg_is_sat)
    usable = []
    for p in measurements:
        if p.usable:
            usable.append(p)
        else:
            fit.below_noise += 1
    if not usable:
        fit.rejected.append("no label clears the engine's own latency spread")
        return fit

    # ---- stage 1: the pure segment, one column live, one rung is enough
    pure_idx = [p.y / p.x_idx for p in usable if p.regime == SAT and abs(p.x_idx) > MIN_USABLE_COLUMN]
    pure_mla = [p.y / p.x_mla for p in usable if p.regime == UNSAT and abs(p.x_mla) > MIN_USABLE_COLUMN]
    if pure_idx:
        fit.c_idx = statistics.median(pure_idx)
    if pure_mla:
        fit.c_mla = statistics.median(pure_mla)

    # ---- stage 2: mixed closes whichever price the pure segment left open
    mixed = [p for p in usable if p.regime == MIXED]
    if mixed:
        if fit.c_idx is None and fit.c_mla is not None:
            num = sum(p.x_idx * (p.y - fit.c_mla * p.x_mla) for p in mixed)
            den = sum(p.x_idx * p.x_idx for p in mixed)
            if den > MIN_USABLE_COLUMN:
                fit.c_idx = num / den
        elif fit.c_mla is None and fit.c_idx is not None:
            num = sum(p.x_mla * (p.y - fit.c_idx * p.x_idx) for p in mixed)
            den = sum(p.x_mla * p.x_mla for p in mixed)
            if den > MIN_USABLE_COLUMN:
                fit.c_mla = num / den
        elif fit.c_idx is None and fit.c_mla is None:
            solved = _solve_n(mixed, (lambda p: p.x_idx, lambda p: p.x_mla), lambda p: p.y)
            if solved is None:
                fit.rejected.append(
                    f"{len(mixed)} mixed rungs cannot carry both prices with no pure "
                    "segment to anchor one: too few, or their columns are parallel"
                )
            else:
                fit.c_idx, fit.c_mla = solved

    if fit.c_idx is None and fit.c_mla is None:
        fit.rejected.append("no segment isolated a price")
        return fit

    # Residuals are reported, never used to reject. A segment solved exactly has
    # none to report; one with spare rungs does, and it is worth seeing.
    predict = lambda p: (fit.c_idx or 0.0) * p.x_idx + (fit.c_mla or 0.0) * p.x_mla
    by_regime: dict[str, list[Measurement]] = {}
    for p in usable:
        by_regime.setdefault(p.regime, []).append(p)
    for name, pts in by_regime.items():
        if len(pts) > 1:
            fit.residuals[name] = _relative_residual(pts, predict, lambda p: p.y)
    return fit


def column_gate(x_idx: float, x_mla: float) -> tuple:
    """Whether each column moves enough work for its price to be worth applying."""
    return (abs(x_idx) >= MIN_ABS_DELTA_IDX_M * 1e6, abs(x_mla) >= MIN_ABS_DELTA_MLA_M * 1e6)


def predict_delta(x_idx: float, x_mla: float, prices: Prices, noise: float = 0.0) -> float:
    """Latency to add for a batch with these column deviations.

    ``prices`` is anything carrying ``c_idx`` and ``c_mla`` -- a `CellFit` from
    the batch's own cell, or the pair `CoefficientField` carried in from
    neighbouring batch sizes. Both go through the same gates here rather than
    each re-deriving them, so an interpolated correction is held to what a
    calibrated one is.

    Three gates, in order, and all-or-nothing: a partial correction leaves the
    surviving column to explain the whole delta and overshoots (measured max
    error 87.6% -> 207.8%, worse than not correcting at all).

    1. An unpriced column must be carrying no work. A price comes back ``None``
       when the cell's segments could not identify it, and the usual reason is
       that the column is identically zero throughout the cell -- a purely
       unsaturated cell prices nothing for crossing ``topk`` because none of
       its batches can cross it. There the missing price multiplies zero and
       the correction is complete as it stands. What is not allowed is the
       other case: an unpriced column that does move work, where dropping it
       silently leaves the priced column to explain that work too. The
       threshold is the one from gate 3, so a column too small to be worth
       pricing is also too small to block on.
    2. A price is the marginal cost of one more unit of that column's work and
       cannot be below zero -- doing more does not take less time. A negative
       one means the fit is not describing the hardware, usually a mismeasured
       uniform batch, which is the subtrahend of every label in the cell.
    3. Some column must move enough work to be worth pricing, and the resulting
       milliseconds must clear the step's own jitter when the caller knows it.
    """
    c_idx, c_mla = prices.c_idx, prices.c_mla
    moves = column_gate(x_idx, x_mla)
    for price, column_moves in zip((c_idx, c_mla), moves, strict=True):
        if price is None:
            if column_moves:
                return 0.0
        elif price < 0.0:
            return 0.0
    if not any(moves):
        return 0.0
    delta = (c_idx or 0.0) * x_idx + (c_mla or 0.0) * x_mla
    if noise > 0.0 and abs(delta) < MIN_LABEL_SNR * noise:
        return 0.0
    return delta
