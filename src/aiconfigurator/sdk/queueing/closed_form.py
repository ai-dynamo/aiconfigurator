# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-form fast path: mean estimates in O(1) arithmetic.

Used on the sweep hot path (30k-100k evaluations) where the limit-cycle
evaluator's O(passes) cost is unnecessary. Every term is annotated with its
provenance; the limit-cycle evaluator is the in-package reference, and the
validation methodology is recorded in docs/design/queueing_model.md §5.

Terms:
  B_eff       running decodes spend the unified budget first
              (vLLM v1 scheduler: running set scheduled before waiting)
  staircase   initial burst of C simultaneous arrivals admitted
              ceil(C*isl_eff/B_eff) chunk-passes deep (chunked-prefill loop)
  W_res       renewal residual life E[T^2]/(2E[T]) — inspection paradox;
              long passes are likelier to be hit by an arrival

Scope: closed-loop (concurrency) operating points — the shape run_agg
sweeps. Open-loop (request-rate) queueing terms are future work and are
deliberately not exposed until validated.
"""

from __future__ import annotations

import math


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
    # ITL mixture: a gap equals the duration of the pass it spans. A mix
    # pass contributes gaps only for the (c-1)/c of the batch NOT being
    # prefilled in it (with c=1 there is nobody else to stall, so mix
    # passes never appear as gaps). Weights are gap counts, not pass counts.
    c = max(1, batch_size)
    w_mix = n_mix * (c - 1) / c
    w_gen = n_gen + n_mix / c
    total_w = w_mix + w_gen
    itl_mean = (w_mix * t_mix + w_gen * t_gen) / total_w
    mix_frac = w_mix / total_w
    itl_p50 = t_mix if mix_frac >= 0.5 else t_gen
    itl_p99 = t_mix if mix_frac >= 0.01 else t_gen

    # cohort bracket: the steady limit cycle can lock anywhere between solo
    # prompts (one per pass) and full-budget packing — both extremes are
    # computable from this operating point, and the true steady p99 lies
    # inside the bracket by construction (cohort size is in [1, B_eff/isl]).
    # Used by the sweep funnel: reject only when even `lo` violates the SLA;
    # send straddlers (lo <= SLA < hi) to the quantitative tier.
    p_step = max(0.0, float(prefill_step_ms))
    if isl <= ctx_tokens:
        # linear token scaling of the full-chunk prefill cost; prefill is
        # superlinear in tokens, so this bounds the solo pass from ABOVE —
        # conservative in the keep direction (lo never overshoots down)
        own_lo = p_step * (isl / max(1, ctx_tokens)) + t_gen
    else:
        own_lo = own  # prompt spans full budget: the bracket collapses
    ttft_p99_lo = min(own_lo, own)
    ttft_p99_hi = own + 2.0 * t_mix  # max residual + one cohort-misalignment pass

    return {
        "ttft_steady_mean": ttft_steady_mean,
        "ttft_steady_p50": _q(0.50),
        "ttft_steady_p75": _q(0.75),
        "ttft_steady_p90": _q(0.90),
        "ttft_steady_p95": _q(0.95),
        "ttft_steady_p99": _q(0.99),
        "ttft_steady_p999": _q(0.999),
        "ttft_steady_p99_lo": ttft_p99_lo,
        "ttft_steady_p99_hi": ttft_p99_hi,
        "ttft_transient_mean": sum(stair) / len(stair),
        "ttft_transient_max": stair[-1],
        "itl_mean": itl_mean,
        "itl_p50": itl_p50,
        "itl_p99": itl_p99,
        "queueing_tier": "screening",
    }


TTFT_QUANTILE_COLUMNS = {
    0.5: "ttft_steady_p50",
    0.75: "ttft_steady_p75",
    0.9: "ttft_steady_p90",
    0.95: "ttft_steady_p95",
    0.99: "ttft_steady_p99",
    0.999: "ttft_steady_p999",
}


def ttft_quantile_column(q: float) -> str:
    """Exact stored column for a supported TTFT percentile."""
    return TTFT_QUANTILE_COLUMNS.get(q, "ttft_steady_p50")


def screening_quantile(row: dict, metric: str, q: float):
    """Screening-tier quantile lookup from stored columns.

    ttft quantiles are stored exactly for every supported percentile. itl
    is a two-mass distribution: p50 reads the low mass, anything above
    maps to the tail mass (conservative — every intermediate quantile of a
    two-mass distribution equals one of the two masses). tpot / e2e have no screening
    distribution — quantile enforcement for those requires the evaluator
    (returns None so callers fall back to the legacy scalar screen).
    """
    if metric == "ttft":
        return row.get(ttft_quantile_column(q))
    if metric == "itl":
        return row.get("itl_p50" if q <= 0.5 else "itl_p99")
    return None


QUEUEING_COLUMNS = [
    "ttft_steady_mean",
    "ttft_steady_p50",
    "ttft_steady_p75",
    "ttft_steady_p90",
    "ttft_steady_p95",
    "ttft_steady_p99",
    "ttft_steady_p999",
    "ttft_steady_p99_lo",
    "ttft_steady_p99_hi",
    "ttft_transient_mean",
    "ttft_transient_max",
    "itl_mean",
    "itl_p50",
    "itl_p99",
    "queueing_tier",
]


def static_degenerate_columns(ttft_ms: float, tpot_ms: float, tier: str = "static") -> dict:
    """Static batching / disagg-static mapping: no queueing, no interference
    — all distributions collapse onto the legacy scalars by construction
    (the p99 bracket collapses to a point for the same reason)."""
    return {
        "ttft_steady_mean": ttft_ms,
        "ttft_steady_p50": ttft_ms,
        "ttft_steady_p75": ttft_ms,
        "ttft_steady_p90": ttft_ms,
        "ttft_steady_p95": ttft_ms,
        "ttft_steady_p99": ttft_ms,
        "ttft_steady_p999": ttft_ms,
        "ttft_steady_p99_lo": ttft_ms,
        "ttft_steady_p99_hi": ttft_ms,
        "ttft_transient_mean": ttft_ms,
        "ttft_transient_max": ttft_ms,
        "itl_mean": tpot_ms,
        "itl_p50": tpot_ms,
        "itl_p99": tpot_ms,
        "queueing_tier": tier,
    }
