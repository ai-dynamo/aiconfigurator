# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-form fast path: mean estimates in O(1) arithmetic.

Used on the sweep hot path (30k-100k evaluations) where the limit-cycle
evaluator's O(passes) cost is unnecessary. Every term is annotated with its
provenance. The limit-cycle evaluator is the in-package reference; the DES
oracle (tools/queueing_oracle) is the external one.

Terms:
  B_eff       running decodes spend the unified budget first
              (mocker core.rs: running set scheduled before waiting)
  staircase   initial burst of C simultaneous arrivals admitted
              ceil(C*isl_eff/B_eff) chunk-passes deep (chunked-prefill loop)
  W_res       renewal residual life E[T^2]/(2E[T]) — inspection paradox;
              long passes are likelier to be hit by an arrival
  W_q (open)  M/D/1 Pollaczek-Khinchine wait with prefill-pass service
"""

from __future__ import annotations

import math

from .spec import EngineSpec, TimingModel, WorkloadSpec


def effective_budget(wl: WorkloadSpec, eng: EngineSpec, concurrency: int) -> int:
    return max(1, eng.max_num_batched_tokens - concurrency)


def steady_pass_times(wl: WorkloadSpec, eng: EngineSpec, timing: TimingModel, concurrency: int) -> dict:
    """Characteristic pass durations at the steady operating point."""
    c = concurrency
    ctx = wl.isl + wl.osl // 2
    b_eff = effective_budget(wl, eng, c)
    t_gen = timing.decode_ms(c, ctx)
    # a mix pass carrying one full-budget prefill chunk
    chunk = min(wl.effective_isl, b_eff)
    t_mix = timing.prefill_ms(1, wl.prefix + chunk, wl.prefix) + t_gen
    return {"t_gen": t_gen, "t_mix": t_mix, "b_eff": b_eff, "ctx": ctx}


def ttft_steady_mean(wl: WorkloadSpec, eng: EngineSpec, timing: TimingModel) -> float:
    """Closed-loop steady TTFT: residual of the pass in flight at arrival
    plus the request's own prefill chunk passes."""
    c = wl.concurrency or 1
    p = steady_pass_times(wl, eng, timing, c)
    chunks = math.ceil(wl.effective_isl / p["b_eff"])
    # pass-type frequencies per completion epoch (token conservation):
    # each completion implies osl decode emissions and isl_eff prefill tokens
    n_mix = max(1.0, wl.effective_isl / p["b_eff"])
    n_gen = max(0.0, wl.osl - n_mix)
    t1, t2 = p["t_mix"], p["t_gen"]
    mean_t = (n_mix * t1 + n_gen * t2) / (n_mix + n_gen)
    mean_t2 = (n_mix * t1 * t1 + n_gen * t2 * t2) / (n_mix + n_gen)
    w_res = mean_t2 / (2.0 * mean_t) if mean_t > 0 else 0.0
    return w_res + chunks * p["t_mix"]


def ttft_transient_mean(wl: WorkloadSpec, eng: EngineSpec, timing: TimingModel) -> float:
    """Initial burst staircase: the k-th of C simultaneous arrivals waits
    ceil(k*isl_eff/B_eff) full-chunk passes (first-order; cohort echo after
    the first generation is captured only by the limit-cycle evaluator)."""
    c = wl.concurrency or 1
    p = steady_pass_times(wl, eng, timing, c)
    total = 0.0
    for k in range(1, c + 1):
        passes = math.ceil(k * wl.effective_isl / p["b_eff"])
        total += passes * p["t_mix"]
    return total / c


def operating_point_columns(
    isl: int,
    osl: int,
    batch_size: int,
    ctx_tokens: int,
    mix_step_ms: float,
    genonly_step_ms: float,
    prefill_step_ms: float,
    num_mix_steps: float,
    num_genonly_steps: float,
) -> dict:
    """Queueing columns from quantities run_agg already computed — pure
    arithmetic, zero extra perf-database queries, safe on the sweep hot path.

    Maps run_agg's operating point onto the pass calendar:
      ctx_tokens        <-> per-pass prefill chunk budget (B_eff)
      mix_step_ms       <-> mix-pass duration t_mix
      genonly_step_ms   <-> gen-only pass duration t_gen
      num_mix/gen_steps <-> pass-type frequencies over one request lifetime

    TTFT_steady = W_res + ceil(isl/ctx_tokens) * t_mix, with W_res the
    renewal residual over the pass-length mixture (time-weighted pass-type
    hit probabilities, residual uniform within a pass). The transient block
    is the admission staircase of `batch_size` simultaneous arrivals.
    """
    t_mix = float(mix_step_ms)
    t_gen = float(genonly_step_ms)
    n_mix = max(float(num_mix_steps), 1e-9)
    n_gen = max(float(num_genonly_steps), 0.0)
    chunks = max(1, math.ceil(isl / max(1, ctx_tokens)))
    own = chunks * t_mix

    # residual life over the pass mixture: a pass of type i is hit with
    # probability proportional to n_i * t_i; within it the residual is
    # uniform on [0, t_i] (discretized at deciles for the percentiles)
    total_time = n_mix * t_mix + n_gen * t_gen
    residual_vals: list = []
    residual_wts: list = []
    for t_i, n_i in ((t_mix, n_mix), (t_gen, n_gen)):
        if n_i <= 0 or t_i <= 0:
            continue
        p_hit = n_i * t_i / total_time
        for d in range(10):
            residual_vals.append((d + 0.5) / 10.0 * t_i)
            residual_wts.append(p_hit / 10.0)

    ttft_vals = [r + own for r in residual_vals] or [own]
    ttft_wts = residual_wts or [1.0]

    def _q(q: float) -> float:
        pairs = sorted(zip(ttft_vals, ttft_wts, strict=True))
        acc, target = 0.0, q * sum(ttft_wts)
        for v, w in pairs:
            acc += w
            if acc >= target:
                return v
        return pairs[-1][0]

    ttft_steady_mean = sum(v * w for v, w in zip(ttft_vals, ttft_wts, strict=True)) / sum(ttft_wts)

    # transient staircase for the initial burst of `batch_size` arrivals
    stair = [math.ceil(k * isl / max(1, ctx_tokens)) * t_mix for k in range(1, max(1, batch_size) + 1)]
    # ITL mixture: gaps are pass durations, weighted by pass-type frequency
    itl_mean = (n_mix * t_mix + n_gen * t_gen) / (n_mix + n_gen)
    itl_p50 = t_gen if n_gen >= n_mix else t_mix
    itl_p99 = t_mix if n_mix / (n_mix + n_gen) >= 0.01 else t_gen

    return {
        "ttft_steady_mean": ttft_steady_mean,
        "ttft_steady_p50": _q(0.50),
        "ttft_steady_p90": _q(0.90),
        "ttft_steady_p99": _q(0.99),
        "ttft_transient_mean": sum(stair) / len(stair),
        "ttft_transient_max": stair[-1],
        "itl_mean": itl_mean,
        "itl_p50": itl_p50,
        "itl_p99": itl_p99,
    }


QUEUEING_COLUMNS = [
    "ttft_steady_mean",
    "ttft_steady_p50",
    "ttft_steady_p90",
    "ttft_steady_p99",
    "ttft_transient_mean",
    "ttft_transient_max",
    "itl_mean",
    "itl_p50",
    "itl_p99",
]


def static_degenerate_columns(ttft_ms: float, tpot_ms: float) -> dict:
    """Static batching / disagg-static mapping: no queueing, no interference
    — all distributions collapse onto the legacy scalars by construction."""
    return {
        "ttft_steady_mean": ttft_ms,
        "ttft_steady_p50": ttft_ms,
        "ttft_steady_p90": ttft_ms,
        "ttft_steady_p99": ttft_ms,
        "ttft_transient_mean": ttft_ms,
        "ttft_transient_max": ttft_ms,
        "itl_mean": tpot_ms,
        "itl_p50": tpot_ms,
        "itl_p99": tpot_ms,
    }


def open_loop_queue_wait(wl: WorkloadSpec, eng: EngineSpec, timing: TimingModel) -> float:
    """Open-loop (Poisson rate) prefill queue wait, M/D/1 P-K form.

    Service = the request's own prefill chunk passes; utilization from
    prefill-token conservation. Returns inf when the rate exceeds prefill
    capacity. Approximation: deterministic service, single queue."""
    if wl.request_rate is None:
        raise ValueError("open_loop_queue_wait requires request_rate")
    lam = wl.request_rate / 1000.0  # req/ms
    # decode batch at the operating point via Little's law estimate
    b = max(
        1, min(eng.max_num_seqs, round(wl.request_rate * wl.osl * timing.decode_ms(eng.max_num_seqs, wl.isl) / 1000.0))
    )
    p = steady_pass_times(wl, eng, timing, b)
    chunks = math.ceil(wl.effective_isl / p["b_eff"])
    service = chunks * p["t_mix"]
    rho = lam * service
    if rho >= 1.0:
        return float("inf")
    return rho * service / (2.0 * (1.0 - rho))
