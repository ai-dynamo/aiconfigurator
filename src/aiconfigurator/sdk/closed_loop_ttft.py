# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal closed-loop TTFT estimator: the pass-calendar recursion as one
pure function, for POST-PROCESSING existing sweep results.

This is the smallest faithful core of the quantitative queueing evaluator:
a deterministic replay of the continuous-batching scheduler's own budget
arithmetic at pass granularity (no RNG, no fitted constants, no event
heap). It exists so result DataFrames can be refined non-invasively — the
prediction pipeline itself is untouched; timing enters through two injected
callables (the same quantities the pipeline already computes).

Semantics modeled (vLLM-v1-style fused pass, which TRT-LLM's in-flight
batching also follows): running decodes spend one budget token each and
emit; queued prefills consume the remaining budget as chunks in admission
order; a prefill completer emits its first token in the same pass off the
final chunk's logits; closed loop replaces a completed request immediately,
visible to the scheduler after ``turnaround_ms``. Not modeled here (use the
full queueing evaluator): mixed-pass fused timing hooks, KV-pressure
admission, variable shapes, open-loop arrivals, SGLang's alternating mode.
"""

from __future__ import annotations

from collections.abc import Callable


def estimate_closed_loop_latency(
    concurrency: int,
    isl: int,
    osl: int,
    prefill_ms: Callable[[int, int, int], float],
    decode_ms: Callable[[int, int], float],
    max_num_batched_tokens: int = 8192,
    prefix: int = 0,
    turnaround_ms: float = 0.0,
    enable_chunked_prefill: bool = True,
    warmup_generations: int = 4,
    window_generations: int = 4,
) -> dict:
    """Steady-state closed-loop latency at one operating point.

    ``prefill_ms(batch, mean_isl, mean_prefix)`` and
    ``decode_ms(batch, context_len)`` price one prefill batch / one decode
    iteration — the caller wires them to whatever timing source it already
    has. Returns a dict with ``ttft_steady_mean/p50/p99``,
    ``ttft_transient_max``, ``tpot_mean``, ``itl_p50/p99`` and
    ``throughput_rps`` (all ms / req-per-s).
    """
    if concurrency < 1 or isl < 1 or osl < 1:
        raise ValueError("concurrency, isl and osl must be >= 1")
    if not 0 <= prefix < isl:
        raise ValueError("prefix must be in [0, isl)")
    eff_isl = max(1, isl - prefix)

    # slot = [remaining_prefill, generated, arrival_ms, first_token_ms,
    #         last_token_ms, gap_sum, gap_count, gaps_list_or_None]
    def new_slot(arrival: float) -> list:
        return [eff_isl, 0, arrival, -1.0, -1.0, []]

    slots = [new_slot(0.0) for _ in range(concurrency)]
    pending: list = []  # (visible_ms, slot) replacements not yet admitted
    now = 0.0
    completions = 0
    warmup = warmup_generations * concurrency
    target = (warmup_generations + window_generations) * concurrency
    steady_start = None
    ttfts: list = []
    transient_max = 0.0
    gaps: list = []
    tpots: list = []
    steady_completions = 0

    max_passes = 200 * (warmup_generations + window_generations) * osl
    for _ in range(max_passes):
        if completions >= target:
            break
        if pending:  # sync scheduler: arrivals visible by pass start join it
            admitted = [s for t, s in pending if t <= now]
            if admitted:
                slots.extend(admitted)
                pending = [(t, s) for t, s in pending if t > now]

        # --- one fused pass: budget arithmetic in admission order ---
        budget = max_num_batched_tokens
        n_prefill = 0
        prefill_tokens = 0
        prefill_prefix = 0
        completers = []
        decoders = []
        for s in slots:
            if s[0] > 0:
                if budget <= 0:
                    continue
                if not enable_chunked_prefill and s[0] > budget:
                    break  # whole prompts only: admission stops here
                chunk = s[0] if s[0] <= budget else budget
                done_before = prefix + (eff_isl - s[0])
                s[0] -= chunk
                budget -= chunk
                n_prefill += 1
                prefill_tokens += done_before + chunk
                prefill_prefix += done_before
                if s[0] == 0:
                    completers.append(s)
            elif s[1] < osl and budget > 0:
                budget -= 1
                decoders.append(s)

        duration = 0.0
        if n_prefill:
            duration += prefill_ms(n_prefill, prefill_tokens // n_prefill, prefill_prefix // n_prefill)
        if decoders:
            ctx = sum(isl + s[1] for s in decoders) // len(decoders)
            duration += decode_ms(len(decoders), ctx)
        if duration <= 0.0 and not completers and not decoders:
            if not pending:
                raise RuntimeError("stalled: no schedulable work (budget too small?)")
            now = max(now, min(t for t, _ in pending))
            continue
        now += duration

        # --- emissions & accounting ---
        for s in completers + decoders:
            s[1] += 1
            if s[3] < 0:
                s[3] = now
                t = now - s[2]
                if completions >= warmup:
                    ttfts.append(t)
                transient_max = max(transient_max, t)
            else:
                s[5].append(now - s[4])
            s[4] = now
        finished = [s for s in completers + decoders if s[1] >= osl]
        for s in finished:
            completions += 1
            if completions == warmup:
                steady_start = now
            if completions > warmup:
                steady_completions += 1
                gaps.extend(s[5])
                if s[5]:
                    tpots.append(sum(s[5]) / len(s[5]))
            slots.remove(s)
            pending.append((now + turnaround_ms, new_slot(now)))
    else:
        raise RuntimeError("did not converge within max_passes")

    window = now - (steady_start or 0.0)
    q = lambda v, x: sorted(v)[min(len(v) - 1, max(0, round(x * len(v)) - 1))] if v else float("nan")
    return {
        "ttft_steady_mean": sum(ttfts) / len(ttfts) if ttfts else float("nan"),
        "ttft_steady_p50": q(ttfts, 0.5),
        "ttft_steady_p99": q(ttfts, 0.99),
        "ttft_transient_max": transient_max,
        "tpot_mean": sum(tpots) / len(tpots) if tpots else float("nan"),
        "itl_p50": q(gaps, 0.5),
        "itl_p99": q(gaps, 0.99),
        "throughput_rps": steady_completions / (window / 1000.0) if window > 0 else 0.0,
    }


def refine_closed_loop_ttft(
    df,
    prefill_ms: Callable[[int, int, int], float],
    decode_ms: Callable[[int, int], float],
    turnaround_ms: float = 0.0,
    enable_chunked_prefill: bool = True,
):
    """Post-process an agg result DataFrame (``ColumnsAgg`` schema): run the
    pass-calendar estimate per row and RETURN A COPY with additive columns
    ``ttft_refined`` / ``tpot_refined`` / ``throughput_refined`` — existing
    columns (including the legacy ``ttft``) are untouched, so downstream
    consumers opt in per column. Rows it cannot price (missing/invalid
    inputs) keep NaN in the new columns.
    """
    out = df.copy()
    ttfts, tpots, xs = [], [], []
    for _, row in out.iterrows():
        try:
            r = estimate_closed_loop_latency(
                concurrency=int(row["bs"]),
                isl=int(row["isl"]),
                osl=int(row["osl"]),
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                max_num_batched_tokens=int(row["ctx_tokens"]),
                prefix=int(row.get("prefix", 0) or 0),
                turnaround_ms=turnaround_ms,
                enable_chunked_prefill=enable_chunked_prefill,
            )
            ttfts.append(r["ttft_steady_mean"])
            tpots.append(r["tpot_mean"])
            xs.append(r["throughput_rps"])
        except (KeyError, ValueError, RuntimeError):
            ttfts.append(float("nan"))
            tpots.append(float("nan"))
            xs.append(float("nan"))
    out["ttft_refined"] = ttfts
    out["tpot_refined"] = tpots
    out["throughput_refined"] = xs
    return out
