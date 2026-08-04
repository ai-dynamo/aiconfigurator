# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Solving the work-delta unit prices from measured calibration batches.

Each calibration batch was built around one average point holding both totals
fixed, so subtracting the average point's latency removes everything that
depends on the totals alone and leaves the cost of the spread:

    y = T_batch - T_uniform

Three prices, because a prefill batch that straddles ``topk`` runs two different
attention kernels at once:

    y = c_idx * x_idx  +  c_mla_sparse * x_mla_sparse  +  c_mla_mha * x_mla_dense

Nothing here fits all three at once. The order is forced by what each segment
can move, and each step subtracts what the previous one already fixed:

    1. c_mla_mha  from the unsaturated segments. Those batches short-circuit the
       indexer on every row, so ``x_idx`` and ``x_mla_sparse`` are identically
       zero and the segment pins the dense price on its own. It is fitted per
       batch size rather than per cell -- it is a property of the dense kernel,
       and at a cell whose average request is long it cannot be measured at all,
       because a dense row is capped at ``topk`` tokens while the cell's own
       work grows with ``s_bar^2``.

    2. c_idx  from each cell's saturated segment. Every row is above ``topk``
       so the indexer is pinned there for all of them, which makes the attention
       term linear in ``s``; with ``sum(s)`` conserved its deviation cancels
       exactly and only the gated column survives.

    3. c_mla_sparse  from the mixed segment, with whatever step 1 and step 2
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

__all__ = [
    "RESIDUAL_TOLERANCE",
    "CellFit",
    "Measurement",
    "predict_delta",
    "solve_cell",
]

UNSAT = "unsat"
MIXED = "mixed"
SAT = "sat"

# A fit whose points scatter further than this around the line is reporting
# that the linear form is wrong for the cell, not that the timing was noisy:
# the rungs span the segment's whole range, so a genuine linear relation leaves
# little room for scatter.
# Swept on the collected grid rather than chosen. At 0.10 only 5 of 33 cells
# were accepted and NO average point had two calibrated batch sizes, so the
# bracketing requirement in ``field`` had nothing to bracket and every query
# fell through to a single-ended guess -- the source of every regression we
# measured. At 0.30 ten cells are accepted, four average points bracket, and
# no point in either run is made worse. Past 0.50 a badly-measured cell gets
# in and does real damage, so this is a measured ceiling, not a preference.
RESIDUAL_TOLERANCE = 0.30

# How far a label must clear the engine's own latency spread at that shape
# before it is allowed into a fit. Measured on GLM-5: at cells whose average
# request is short the prefill step sits at a flat ~240 ms whatever the spread,
# so a batch carrying twice the modelled attention work moves the clock by
# 0.07 ms. Those labels are not noisy measurements of a small effect; they are
# measurements of an effect the step does not have.
MIN_LABEL_SNR = 3.0

# Smallest work deviation, as a fraction of the cell's own total work, that is
# worth correcting at all. A batch whose columns barely move has little to
# correct, and a coefficient carrying even a modest relative error will spend
# that error on a label of the same small size -- the absolute miss is small but
# the relative one is not.
#
# Measured on GLM-5 this does NOT separate cleanly: the points the correction
# hurt sit at a 10th percentile of 0.048 and the points it helped at 0.051, so
# any threshold that blocks the former costs one to two of the latter and the
# median error gets worse. Swept on the full grid with a clean bytecode cache:
#     0.00 -> 27 better /  7 worse, median 1.39%
#     0.10 -> 23 better /  9 worse, median 1.68%
#     0.30 -> 16 better / 14 worse, median 2.80%
# and the WORST case never moves (23.1%) at any threshold, because the batch it
# comes from moves the work by 2.2x the cell's own -- no magnitude gate reaches
# it. Left at 0 (off); raise it only if a deployment would rather under-correct.
# 绝对门(单位 M pairs)。物理依据是 slide 6:prefill 的三个 kernel 在
# 低工作量段都是水平线,过了拐点才 ∝work,拐点是个**绝对**工作量(~10^0 M),
# 跟本 cell 有多大无关。设为 0 则退回相对门。
# Smallest column deviation worth pricing, in millions of attention-pair reads.
# Absolute, not a fraction of the cell's own work: a relative gate divides a
# real 200 ms delta by a large cell's large denominator and discards it, while
# keeping a 10 ms delta in a small one -- backwards from the physics. Each
# threshold is the machine's own 3-sigma step-time jitter divided by that
# column's measured price, so below it the correction is smaller than the
# spread it would have to be verified against.
#
#   indexer  0.96 ms per M reads (0.943-0.999 across 12 cells, near-constant)
#   MLA      ~5 ms per M, five times dearer
#   jitter   9.0 ms median 3-sigma   ->  9.0/0.96 = 9.4 -> 10 M,  9.0/5 -> 2 M
#
# Re-derive both when the model, parallelism or hardware changes: the formula
# transfers, these two numbers do not.
MIN_ABS_DELTA_IDX_M = 10.0
MIN_ABS_DELTA_MLA_M = 2.0


MIN_USABLE_COLUMN = 1e-9

# Squared cosine between two feature columns at which their split stops being
# determined by the data. Their sum stays well conditioned either way, which is
# what makes this worth rejecting rather than reporting a large coefficient.
MAX_COLLINEARITY = 1.0 - 1e-6

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
    x_mla_sparse: float
    x_mla_dense: float
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


@dataclass
class CellFit:
    b: int
    s_bar: int
    p_bar: int
    avg_is_sat: bool
    c_idx: float | None = None
    c_mla_sparse: float | None = None
    c_mla_mha: float | None = None
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
    # n+1, not n. With exactly n rungs the system is square: it reproduces the
    # labels perfectly whatever the prices are, so the residual is zero by
    # construction and reports nothing. A fit nobody can check is worse than no
    # fit, because it looks like it passed.
    if len(points) <= n:
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


def _solve_two(points, col_a, col_b, target) -> tuple[float, float] | None:
    """Joint least squares for two prices, used when no pure segment fixed one.

    Returns ``None`` when the two columns are close to parallel: the normal
    equations are then near-singular and the split between the coefficients is
    decided by noise even though their sum is well determined.
    """
    a11 = sum(col_a(p) ** 2 for p in points)
    a12 = sum(col_a(p) * col_b(p) for p in points)
    a22 = sum(col_b(p) ** 2 for p in points)
    if a11 <= MIN_USABLE_COLUMN or a22 <= MIN_USABLE_COLUMN:
        return None
    if a12 * a12 >= MAX_COLLINEARITY * a11 * a22:
        return None
    b1 = sum(target(p) * col_a(p) for p in points)
    b2 = sum(target(p) * col_b(p) for p in points)
    det = a11 * a22 - a12 * a12
    return (b1 * a22 - b2 * a12) / det, (b2 * a11 - b1 * a12) / det


def solve_cell(b: int, s_bar: int, p_bar: int, avg_is_sat: bool, measurements: list[Measurement]) -> CellFit:
    """Fit this average point's own prices, staged by what each segment isolates.

    The regime decides which columns are even live, and that is what makes the
    stages exact rather than a least-squares compromise:

        pure saturated    every row sits above topk, so the attention term is
                          linear in s and its deviation cancels: only x_idx
                          survives. One rung fixes c_idx outright.
        pure unsaturated  every row is at or below topk, so x_idx is
                          identically zero. One rung fixes c_mla_mha.
        mixed             all three columns are live, but the pure segment has
                          already fixed one of them, so two rungs close the
                          remaining two.

    A cell expresses exactly two of the three regimes (segments_for), so three
    rungs -- one pure, two mixed -- determine all three prices. There is no
    spare degree of freedom and none is wanted: the prices are used at this
    average point, not extrapolated from it, so redundancy would buy a residual
    we have no use for while rejecting cells that are perfectly well determined.
    An earlier revision demanded one extra rung per unknown and a residual
    within tolerance; it left 5 of 33 cells standing where this leaves 45 of 45.

    Repeated rungs in one pure segment are averaged by median rather than
    least-squares: the segment is one-dimensional, so each rung is an
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
    pure_dense = [p.y / p.x_mla_dense for p in usable if p.regime == UNSAT and abs(p.x_mla_dense) > MIN_USABLE_COLUMN]
    if pure_idx:
        fit.c_idx = statistics.median(pure_idx)
    if pure_dense:
        fit.c_mla_mha = statistics.median(pure_dense)

    # ---- stage 2: mixed closes what the pure segment left open
    mixed = [p for p in usable if p.regime == MIXED]
    known = lambda p: (
        ((fit.c_idx or 0.0) * p.x_idx if fit.c_idx is not None else 0.0)
        + ((fit.c_mla_mha or 0.0) * p.x_mla_dense if fit.c_mla_mha is not None else 0.0)
    )
    unknown = [
        name
        for name, value in (("c_idx", fit.c_idx), ("c_mla_sparse", fit.c_mla_sparse), ("c_mla_mha", fit.c_mla_mha))
        if value is None
    ]
    if mixed and unknown:
        col = {
            "c_idx": lambda p: p.x_idx,
            "c_mla_sparse": lambda p: p.x_mla_sparse,
            "c_mla_mha": lambda p: p.x_mla_dense,
        }
        live = [n for n in unknown if sum(col[n](p) ** 2 for p in mixed) > MIN_USABLE_COLUMN]
        if live:
            solved = _solve_n(mixed, tuple(col[n] for n in live), lambda p: p.y - known(p))
            if solved is None:
                fit.rejected.append(
                    f"{len(mixed)} mixed rungs cannot close {len(live)} "
                    "remaining prices: too few, or their columns are parallel"
                )
            else:
                for name, value in zip(live, solved, strict=True):
                    setattr(fit, name, value)

    if all(getattr(fit, n) is None for n in ("c_idx", "c_mla_sparse", "c_mla_mha")):
        fit.rejected.append("no segment isolated a price")
        return fit

    # Residuals are reported, never used to reject. A segment solved exactly
    # has none to report; one with spare rungs does, and it is worth seeing.
    predict = lambda p: (
        (fit.c_idx or 0.0) * p.x_idx
        + (fit.c_mla_sparse or 0.0) * p.x_mla_sparse
        + (fit.c_mla_mha or 0.0) * p.x_mla_dense
    )
    by_regime: dict[str, list[Measurement]] = {}
    for p in usable:
        by_regime.setdefault(p.regime, []).append(p)
    for name, pts in by_regime.items():
        if len(pts) > 1:
            fit.residuals[name] = _relative_residual(pts, predict, lambda p: p.y)
    return fit


def column_gate(x_idx: float, x_mla_sparse: float, x_mla_dense: float) -> tuple:
    """Whether each column moves enough work for its price to be worth applying.

    The two MLA columns share one threshold AND one sum: they are two paths
    through the same kernel and a batch's rows split between them, so the work
    that matters is what they move together. Comparing each against the
    threshold on its own fails a batch that splits its deviation evenly -- at
    1.5M reads per column neither side clears a 2M gate, though the batch moved
    3M -- and that even split is the largest deviation, not the smallest.
    """
    mla_moved = abs(x_mla_sparse) + abs(x_mla_dense)
    mla_passes = mla_moved >= MIN_ABS_DELTA_MLA_M * 1e6
    return (abs(x_idx) >= MIN_ABS_DELTA_IDX_M * 1e6, mla_passes, mla_passes)


def predict_delta(x_idx: float, x_mla_sparse: float, x_mla_dense: float, fit: CellFit, noise: float = 0.0) -> float:
    """Latency to add for a batch with these column deviations.

    Two gates, in order, and both are all-or-nothing: a partial correction
    leaves the surviving columns to explain the whole delta and overshoots
    (measured max error 87.6% -> 207.8%, worse than not correcting at all).

    1. Each column must move enough work to be worth pricing at all.
    2. The resulting milliseconds must clear the step's own jitter, when the
       caller knows it. Correcting by less than the spread the correction
       would have to be checked against cannot be verified either way.

    Every column is priced with the coefficient for the kernel that column's
    rows actually run on. A cell that did not fit a coefficient contributes
    nothing through it rather than borrowing another one's.
    """
    for price in (fit.c_idx, fit.c_mla_sparse, fit.c_mla_mha):
        if price is not None and price < 0.0:
            # A price is the marginal cost of one more unit of that kernel's
            # work and cannot be below zero: doing more does not take less
            # time. A negative one means the fit is not describing the
            # hardware -- usually a mismeasured uniform batch, the subtrahend
            # of every label in the cell. CoefficientField declines such a
            # cell too, but this function is public and reachable without it.
            return 0.0
    if not any(column_gate(x_idx, x_mla_sparse, x_mla_dense)):
        return 0.0
    delta = (fit.c_idx or 0.0) * x_idx + (fit.c_mla_sparse or 0.0) * x_mla_sparse + (fit.c_mla_mha or 0.0) * x_mla_dense
    if noise > 0.0 and abs(delta) < MIN_LABEL_SNR * noise:
        return 0.0
    return delta
