# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerf-style summary metrics for the DES oracle."""

from __future__ import annotations

import json
from statistics import mean, quantiles

from vllm_sim import Request


def _pcts(values: list[float]) -> dict:
    if not values:
        return {}
    if len(values) == 1:
        v = values[0]
        return {"mean": v, "p50": v, "p90": v, "p99": v, "max": v}
    qs = quantiles(values, n=100, method="inclusive")
    return {
        "mean": mean(values),
        "p50": qs[49],
        "p90": qs[89],
        "p99": qs[98],
        "max": max(values),
    }


def summarize(requests: list[Request]) -> dict:
    done = [r for r in requests if r.completed_ms >= 0]
    ttft = [r.token_times[0] - r.dispatch_ms for r in done if r.token_times]
    itl: list[float] = []
    for r in done:
        itl.extend(b - a for a, b in zip(r.token_times, r.token_times[1:], strict=False))
    e2e = [r.completed_ms - r.dispatch_ms for r in done]
    queue = [r.admitted_ms - r.dispatch_ms for r in done if r.admitted_ms >= 0]
    preemptions = sum(r.num_preemptions for r in requests)
    cached = [r.cached_tokens_at_admission for r in done]
    span_ms = max(r.completed_ms for r in done) - min(r.dispatch_ms for r in done) if done else 0.0
    out_tokens = sum(len(r.token_times) for r in done)

    return {
        "requests_completed": len(done),
        "duration_s": span_ms / 1000.0,
        "request_throughput_rps": len(done) / (span_ms / 1000.0) if span_ms else 0.0,
        "output_token_throughput_tps": out_tokens / (span_ms / 1000.0) if span_ms else 0.0,
        "ttft_ms": _pcts(ttft),
        "itl_ms": _pcts(itl),
        "e2e_ms": _pcts(e2e),
        "queue_ms": _pcts(queue),
        "total_preemptions": preemptions,
        "mean_cached_prefix_tokens": mean(cached) if cached else 0.0,
    }


def print_table(summary: dict) -> None:
    print(f"{'metric':<28}{'mean':>10}{'p50':>10}{'p90':>10}{'p99':>10}{'max':>10}")
    for key in ("ttft_ms", "itl_ms", "e2e_ms", "queue_ms"):
        row = summary.get(key) or {}
        if not row:
            continue
        print(f"{key:<28}" + "".join(f"{row[k]:>10.2f}" for k in ("mean", "p50", "p90", "p99", "max")))
    print(f"\nrequests completed        : {summary['requests_completed']}")
    print(f"duration                  : {summary['duration_s']:.2f} s")
    print(f"request throughput        : {summary['request_throughput_rps']:.3f} req/s")
    print(f"output token throughput   : {summary['output_token_throughput_tps']:.1f} tok/s")
    print(f"total preemptions         : {summary['total_preemptions']}")
    print(f"mean cached prefix tokens : {summary['mean_cached_prefix_tokens']:.1f}")


def write_json(summary: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
