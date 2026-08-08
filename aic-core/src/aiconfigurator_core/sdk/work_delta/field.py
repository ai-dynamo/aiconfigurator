# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coefficient lookup for a query that lands between calibrated cells.

Calibration fits one cell at a time; deployment asks about batch sizes nobody
measured. This module is the bridge, and it is deliberately narrow: a query is
answered from the cell's own fit, or from two fits that bracket it along the
batch-size axis, or not at all.

Why only that axis. The three grid axes are not interchangeable. Moving along
``b`` holds ``avg_s + avg_p``, so every request keeps its kernel path and only
the parallelism changes -- ``c_idx`` stays inside 0.943-0.999 across a fourfold
span. Moving along ``avg_s`` lets a row cross ``topk``, and ``c_mla`` was measured at
+6.0e-06 at one average point and -4.1e-06 at another. Carrying a value across
that axis does not produce a worse estimate, it produces one with the wrong
sign.

Why nothing weaker than bracketing. Earlier revisions fell back to single-ended
extrapolation and then to inverse-distance weighting over the whole field, so
that every query got an answer. Every point those two levels made worse -- 5 of
43 corrections in one run, 9 of 66 in the other, the worst turning a 0.30%
estimate into 11.50% -- came from them, and the points they damaged had errors
of 0.1-2.5% to begin with: batches that were already fine. Declining to answer
returns the caller to the uniform estimate, which is a known baseline. Answering
badly is undetectable in production.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiconfigurator_core.sdk.work_delta.solver import CellFit

__all__ = ["COLUMNS", "CoefficientField"]


def _has_negative(fit: CellFit) -> bool:
    """Whether any solved price is negative.

    A price is the marginal cost of one more unit of that kernel's work, so it
    cannot be below zero: doing more of something does not take less time. A
    negative value is therefore never a property of the hardware, only a sign
    that the fit it came out of is not describing the hardware -- most often a
    mismeasured uniform batch, which is the subtrahend of every label in the
    cell and drags the whole column negative when it reads high.

    Applied to a cell's own prices and to both ends of an interpolation alike:
    a negative anywhere disqualifies the query, and the caller falls back to
    the uniform estimate.
    """
    return any(v is not None and v < 0.0 for v in (fit.c_idx, fit.c_mla))


COLUMNS = ("c_idx", "c_mla")


@dataclass(frozen=True)
class Interpolated:
    """A coefficient triple carried to an uncalibrated batch size."""

    c_idx: float | None
    c_mla: float | None
    source: str


class CoefficientField:
    """The accepted per-cell fits, queryable at batch sizes between them."""

    def __init__(self, fits: dict[tuple[int, int, int], CellFit], topk: int):
        # Only accepted fits enter. A fit whose residual exceeded tolerance is
        # worth reporting and worth keeping in the calibration record, but
        # using it is not a lesser form of correcting -- one such cell, whose
        # uniform batch had been mismeasured, produced coefficients that drove
        # a 1709.8 ms estimate negative.
        self._fits = {k: v for k, v in fits.items() if v.accepted}
        self._topk = topk

    def __len__(self) -> int:
        return len(self._fits)

    def bracketing_points(self) -> list[tuple[int, int]]:
        """Average points with at least two calibrated batch sizes.

        The only coordinates this field can answer away from a measured cell,
        so it is the number to watch when designing a calibration grid: a grid
        that spreads thin over many average points and never repeats a batch
        size within one produces an empty list and a field that can only ever
        answer exact hits.
        """
        by_point: dict[tuple[int, int], list[int]] = {}
        for b, s_bar, p_bar in self._fits:
            by_point.setdefault((s_bar, p_bar), []).append(b)
        return sorted(k for k, v in by_point.items() if len(v) >= 2)

    def at(self, b: int, s_bar: int, p_bar: int) -> Interpolated | None:
        """Coefficients for this cell, or ``None`` when it cannot be answered."""
        exact = self._fits.get((b, s_bar, p_bar))
        if exact is not None:
            if _has_negative(exact):
                return None
            return Interpolated(exact.c_idx, exact.c_mla, "calibrated")

        same = sorted(k[0] for k in self._fits if k[1] == s_bar and k[2] == p_bar)
        lower = [x for x in same if x < b]
        upper = [x for x in same if x > b]
        if not (lower and upper):
            return None
        b_lo, b_hi = max(lower), min(upper)

        lo, hi = self._fits[(b_lo, s_bar, p_bar)], self._fits[(b_hi, s_bar, p_bar)]
        if _has_negative(lo) or _has_negative(hi):
            return None

        weight = (b - b_lo) / (b_hi - b_lo)

        def blend(name: str) -> float | None:
            a, c = getattr(lo, name), getattr(hi, name)
            if a is None or c is None:
                return None
            return a + (c - a) * weight

        values = {name: blend(name) for name in COLUMNS}
        if all(v is None for v in values.values()):
            return None
        return Interpolated(values["c_idx"], values["c_mla"], f"b={b_lo}->{b_hi}")
