# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _ceil_pow2(value: float) -> int:
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


def _floor_pow2(value: float) -> int:
    if value <= 1:
        return 1
    return 1 << math.floor(math.log2(value))


def _pow2_edges(values: list[float]) -> list[int]:
    if not values:
        return [1, 2]
    lo = _floor_pow2(max(1.0, min(values)))
    hi = _ceil_pow2(max(2.0, max(values)))
    edges = []
    cur = lo
    while cur <= hi:
        edges.append(cur)
        cur *= 2
    if len(edges) < 2:
        edges.append(edges[-1] * 2)
    return edges


def _bucket_idx(edges: list[int], value: float) -> int:
    idx = bisect.bisect_right(edges, value) - 1
    return max(0, min(idx, len(edges) - 2))


def _sum_floor_upto(n: int, divisor: int) -> int:
    if n <= 0:
        return 0
    q, r = divmod(n, divisor)
    return divisor * q * (q - 1) // 2 + q * (r + 1)


def _sum_min_prefix(prefix: int, isl: int, cap: int) -> int:
    n_uncapped = max(0, min(isl, cap - prefix))
    uncapped = prefix * n_uncapped + n_uncapped * (n_uncapped + 1) // 2
    capped = (isl - n_uncapped) * cap
    return uncapped + capped


def dsv4_sparse_effective_work(kernel: str, bs: int, isl: int, past_kv: int) -> int:
    if kernel == "paged_mqa_logits":
        return bs * (_sum_floor_upto(past_kv + isl, 4) - _sum_floor_upto(past_kv, 4))
    if kernel == "hca_attn":
        swa = _sum_min_prefix(past_kv, isl, 128)
        c128 = _sum_floor_upto(past_kv + isl, 128) - _sum_floor_upto(past_kv, 128)
        return bs * (swa + c128)
    raise ValueError(f"unsupported DSV4 sparse kernel: {kernel}")


@dataclass(frozen=True)
class _Sample:
    m: int
    work: int
    kbar: float
    latency: float


@dataclass(frozen=True)
class _LinearFit:
    coeff: tuple[float, float, float]
    means: tuple[float, float]
    scales: tuple[float, float]

    @classmethod
    def fit(cls, samples: list[_Sample]) -> Optional[_LinearFit]:
        if len(samples) < 3:
            return None

        work_values = [float(sample.work) for sample in samples]
        m_values = [float(sample.m) for sample in samples]
        work_mean = sum(work_values) / len(work_values)
        m_mean = sum(m_values) / len(m_values)
        work_scale = max(max(abs(v - work_mean) for v in work_values), 1.0)
        m_scale = max(max(abs(v - m_mean) for v in m_values), 1.0)

        x = np.array(
            [
                [
                    1.0,
                    (float(sample.work) - work_mean) / work_scale,
                    (float(sample.m) - m_mean) / m_scale,
                ]
                for sample in samples
            ],
            dtype=np.float64,
        )
        y = np.array([sample.latency for sample in samples], dtype=np.float64)
        try:
            coeff = np.linalg.lstsq(x, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(coeff)):
            return None
        return cls(
            coeff=(float(coeff[0]), float(coeff[1]), float(coeff[2])),
            means=(work_mean, m_mean),
            scales=(work_scale, m_scale),
        )

    def predict(self, *, m: int, work: int) -> float:
        work_mean, m_mean = self.means
        work_scale, m_scale = self.scales
        x = (
            1.0,
            (float(work) - work_mean) / work_scale,
            (float(m) - m_mean) / m_scale,
        )
        return max(0.0, sum(c * v for c, v in zip(self.coeff, x, strict=True)))


@dataclass(frozen=True)
class _ShapeModel:
    m_edges: list[int]
    kbar_edges: list[int]
    global_fit: _LinearFit
    cell_fits: dict[tuple[int, int], _LinearFit]

    def predict(self, *, m: int, work: int) -> float:
        kbar = float(work) / max(m, 1)
        cell_key = (_bucket_idx(self.m_edges, m), _bucket_idx(self.kbar_edges, kbar))
        fit = self.cell_fits.get(cell_key, self.global_fit)
        return fit.predict(m=m, work=work)


class Dsv4SparseKernelPredictor:
    """Small calibrated work predictor for DSV4 sparse attention submodules.

    The predictor is intentionally simple: it consumes the raw sparse-kernel
    table and fits ``latency ~= f(work, M)`` once at database load time.  Query
    time only computes effective work, buckets ``(M, work/M)``, and evaluates a
    tiny linear fit.
    """

    def __init__(
        self,
        kernel: str,
        raw_data: Optional[dict],
        *,
        min_m: int = 256,
        min_cell_samples: int = 4,
    ) -> None:
        self.kernel = kernel
        self.min_m = min_m
        self._models: dict[tuple[str, int], _ShapeModel] = {}
        if raw_data:
            self._build(raw_data, min_cell_samples=min_cell_samples)

    def predict(
        self,
        *,
        bs: int,
        isl: int,
        past_kv: int,
        tp_size: int,
        architecture: str,
    ) -> Optional[float]:
        m = bs * isl
        if m < self.min_m:
            return None
        model = self._models.get((architecture, tp_size)) or self._models.get((architecture, 1))
        if model is None:
            return None
        try:
            work = dsv4_sparse_effective_work(self.kernel, bs, isl, past_kv)
        except ValueError:
            return None
        if work <= 0:
            return None
        return model.predict(m=m, work=work)

    def _build(self, raw_data: dict, *, min_cell_samples: int) -> None:
        for architecture, per_tp in raw_data.items():
            for tp_size, per_past in per_tp.items():
                samples = self._samples_from_tp_data(per_past)
                global_fit = _LinearFit.fit(samples)
                if global_fit is None:
                    continue
                m_edges = _pow2_edges([sample.m for sample in samples])
                kbar_edges = _pow2_edges([sample.kbar for sample in samples])
                by_cell: dict[tuple[int, int], list[_Sample]] = {}
                for sample in samples:
                    cell = (
                        _bucket_idx(m_edges, sample.m),
                        _bucket_idx(kbar_edges, sample.kbar),
                    )
                    by_cell.setdefault(cell, []).append(sample)

                cell_fits = {}
                for cell, cell_samples in by_cell.items():
                    if len(cell_samples) < min_cell_samples:
                        continue
                    fit = _LinearFit.fit(cell_samples)
                    if fit is not None:
                        cell_fits[cell] = fit
                self._models[(architecture, int(tp_size))] = _ShapeModel(
                    m_edges=m_edges,
                    kbar_edges=kbar_edges,
                    global_fit=global_fit,
                    cell_fits=cell_fits,
                )

    def _samples_from_tp_data(self, per_past: dict) -> list[_Sample]:
        samples = []
        for past_kv, per_isl in per_past.items():
            for isl, per_bs in per_isl.items():
                for bs, metrics in per_bs.items():
                    latency = metrics.get("latency") if isinstance(metrics, dict) else metrics
                    if latency is None:
                        continue
                    m = int(bs) * int(isl)
                    if m <= 0:
                        continue
                    try:
                        work = dsv4_sparse_effective_work(self.kernel, int(bs), int(isl), int(past_kv))
                    except ValueError:
                        continue
                    if work <= 0:
                        continue
                    samples.append(_Sample(m=m, work=work, kbar=float(work) / m, latency=float(latency)))
        return samples
