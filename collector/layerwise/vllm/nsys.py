# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nsight Systems parsing and repeat-aggregation helpers for layerwise traces."""

from __future__ import annotations

import argparse
import statistics
from typing import Any

DEFAULT_ROLLUP = r"^(CUDAGraphWrapper)$"


def _stable_median_values(values: list[float]) -> list[float]:
    """Return the stable low-latency repeat cluster after trimming high plateaus."""

    cluster = sorted(float(value) for value in values)
    while len(cluster) >= 6:
        gaps = [cluster[idx + 1] - cluster[idx] for idx in range(len(cluster) - 1)]
        gap_idx = max(range(len(gaps)), key=gaps.__getitem__)
        lower = cluster[: gap_idx + 1]
        upper = cluster[gap_idx + 1 :]
        if len(lower) < 3 or len(upper) < 2:
            break
        lower_median = float(statistics.median(lower))
        upper_median = float(statistics.median(upper))
        min_gap = max(1_000.0, lower_median * 0.25)
        if gaps[gap_idx] < min_gap or upper_median < lower_median * 1.5:
            break
        cluster = lower
    return cluster


def _lower_quartile(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[int((len(ordered) - 1) * 0.25)]


def _aggregate_step_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    out: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["step"],
            row["batch_size"],
            row["past_kv"],
            int(row.get("measure_run", 0)),
        )
        agg = out.setdefault(
            key,
            {
                "gpu_us": 0.0,
                "rms_us": 0.0,
                "span_us": 0.0,
                "start_ns": None,
                "end_ns": None,
                "kernel_count": 0,
                "rms_kernel_count": 0,
            },
        )
        agg["gpu_us"] += row["gpu_us"]
        agg["rms_us"] += row.get("rms_us", 0.0)
        if "start_ns" in row and "end_ns" in row:
            start_ns = int(row["start_ns"])
            end_ns = int(row["end_ns"])
            if agg["start_ns"] is None or start_ns < agg["start_ns"]:
                agg["start_ns"] = start_ns
            if agg["end_ns"] is None or end_ns > agg["end_ns"]:
                agg["end_ns"] = end_ns
            agg["span_us"] = (agg["end_ns"] - agg["start_ns"]) / 1000.0
        else:
            # Backward-compatible path for older parser rows.
            agg["span_us"] += row.get("span_us", row["gpu_us"])
        agg["kernel_count"] += row["kernel_count"]
        agg["rms_kernel_count"] += row.get("rms_kernel_count", 0)
    return out


def _latency_us_from_agg(agg: dict[str, Any], latency_source: str) -> float:
    if latency_source == "span":
        return float(agg["span_us"])
    if latency_source == "gpu":
        return float(agg["gpu_us"])
    if latency_source == "gpu_capped":
        return min(float(agg["gpu_us"]), float(agg["span_us"]))
    raise ValueError(f"unsupported latency source: {latency_source}")


def _lookup_aggs(
    parsed: dict[tuple[int, int, int, int], dict[str, Any]],
    expected_key: tuple[int, int, int],
) -> list[dict[str, Any]]:
    exact = [value for key, value in sorted(parsed.items()) if key[:3] == expected_key]
    if exact:
        return exact

    step, _batch_size, past_kv = expected_key
    candidate_items = [(key, value) for key, value in sorted(parsed.items()) if key[0] == step and key[2] == past_kv]
    candidate_batches = {key[1] for key, _ in candidate_items}
    if len(candidate_batches) == 1:
        return [value for _, value in candidate_items]
    return []


def _reduce_agg_latency(
    aggs: list[dict[str, Any]],
    *,
    latency_source: str,
    aggregation: str,
) -> tuple[float, float, int, int, int]:
    if not aggs:
        raise ValueError("cannot reduce empty aggregate list")
    values = [_latency_us_from_agg(agg, latency_source) for agg in aggs]
    rms_values = [float(agg.get("rms_us", 0.0)) for agg in aggs]
    if aggregation == "median":
        latency_us = float(statistics.median(values))
        rms_us = float(statistics.median(rms_values))
    elif aggregation == "mean":
        latency_us = float(statistics.fmean(values))
        rms_us = float(statistics.fmean(rms_values))
    elif aggregation == "trimmed_mean":
        if len(values) < 3:
            latency_us = float(statistics.fmean(values))
            rms_us = float(statistics.fmean(rms_values))
        else:
            latency_us = float(statistics.fmean(sorted(values)[1:-1]))
            rms_us = float(statistics.fmean(sorted(rms_values)[1:-1]))
    elif aggregation == "min":
        latency_us = float(min(values))
        rms_us = float(min(rms_values))
    elif aggregation == "stable_median":
        latency_us = float(statistics.median(_stable_median_values(values)))
        rms_us = float(statistics.median(_stable_median_values(rms_values)))
    elif aggregation == "stable_p25":
        latency_us = float(_lower_quartile(_stable_median_values(values)))
        rms_us = float(_lower_quartile(_stable_median_values(rms_values)))
    else:
        raise ValueError(f"unsupported repeat aggregation: {aggregation}")
    kernel_count = int(statistics.median([int(agg["kernel_count"]) for agg in aggs]))
    rms_kernel_count = int(statistics.median([int(agg.get("rms_kernel_count", 0)) for agg in aggs]))
    return latency_us, rms_us, kernel_count, rms_kernel_count, len(aggs)


def _effective_rollup(args: argparse.Namespace) -> str:
    if args.rollup:
        return str(args.rollup)
    return DEFAULT_ROLLUP
