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


def estimate_disagg_closed_loop_latency(
    concurrency: int,
    isl: int,
    osl: int,
    prefill_workers: int,
    decode_workers: int,
    prefill_step_ms: float,
    decode_step_ms: float,
    prefill_batch: int = 1,
    decode_max_seqs: int = 256,
    turnaround_ms: float = 0.0,
    handoff_ms: float = 0.0,
    warmup_generations: int = 4,
    window_generations: int = 4,
) -> dict:
    """Closed-loop latency for a P/D-disaggregated deployment (mini tandem).

    Serving flow: each prefill worker runs one batch of up to
    ``prefill_batch`` whole prompts per pass at ``prefill_step_ms`` (the raw
    solo context latency the disagg row records as ``(p)prefill_step_ms`` —
    TRT-LLM native disagg deploys ctx workers without chunked prefill); a
    static round-robin router assigns arrivals; the first token becomes
    user-visible after the KV handoff (decode-attach flow, ``handoff_ms``
    per request — ~0 on NVLink at ms scale); decode workers emit one token
    per ``decode_step_ms`` iteration (the row's ``tpot`` — disagg decode has
    no prefill interference) for up to ``decode_max_seqs`` running
    sequences. Closed loop: a completed request's replacement dispatches
    immediately and becomes visible after ``turnaround_ms``.
    """
    if min(concurrency, isl, osl, prefill_workers, decode_workers, prefill_batch) < 1:
        raise ValueError("concurrency/isl/osl/workers/prefill_batch must be >= 1")

    # request = [visible_ms, arrival_ms, generated, first_ms, last_ms, gaps]
    p_free = [0.0] * prefill_workers
    p_queues: list[list] = [[] for _ in range(prefill_workers)]
    d_free = [0.0] * decode_workers
    d_running: list[list] = [[] for _ in range(decode_workers)]
    state = {"rr_p": 0, "rr_d": 0, "completions": 0, "steady_start": None, "steady_completions": 0}
    warmup = warmup_generations * concurrency
    target = (warmup_generations + window_generations) * concurrency
    ttfts: list = []
    gaps: list = []
    tpots: list = []

    def dispatch(arrival_ms: float, visible_ms: float) -> None:
        r = [visible_ms, arrival_ms, 0, -1.0, -1.0, []]
        p_queues[state["rr_p"] % prefill_workers].append(r)
        state["rr_p"] += 1

    def complete(r: list, end_ms: float) -> None:
        state["completions"] += 1
        if state["completions"] == warmup:
            state["steady_start"] = end_ms
        if state["completions"] > warmup:
            state["steady_completions"] += 1
            gaps.extend(r[5])
            if r[5]:
                tpots.append(sum(r[5]) / len(r[5]))
        dispatch(end_ms, end_ms + turnaround_ms)

    for _ in range(concurrency):
        dispatch(0.0, 0.0)

    # For osl >= 2 every dispatch comes from a decode completion and events
    # are processed in non-decreasing time, so each queue is FIFO in visible
    # time: the earliest entry is [0] and the eligible set is a leading run.
    # That turns the per-iteration O(C) min-scans (the profile's 78%) into
    # O(1) lookups with bit-identical results. osl == 1 interleaves prefill
    # and decode completions inside one event and keeps the full scans.
    fifo = osl >= 2

    now = 0.0
    max_iters = 400 * (target + 1) * osl
    for _ in range(max_iters):
        if state["completions"] >= target:
            break
        if fifo:
            t_p = min((max(p_free[i], q[0][0]) for i, q in enumerate(p_queues) if q), default=float("inf"))
            t_d = min((max(d_free[i], run[0][0]) for i, run in enumerate(d_running) if run), default=float("inf"))
        else:
            t_p = min(
                (max(p_free[i], min(r[0] for r in q)) for i, q in enumerate(p_queues) if q),
                default=float("inf"),
            )
            t_d = min(
                (max(d_free[i], min(r[0] for r in run)) for i, run in enumerate(d_running) if run),
                default=float("inf"),
            )
        now = min(t_p, t_d)
        if now == float("inf"):
            raise RuntimeError("tandem stalled — invalid configuration")

        for i, q in enumerate(p_queues):  # whole-prompt prefill batches
            if not q or p_free[i] > now:
                continue
            if fifo:
                k = 0
                for r in q:
                    if r[0] > now or k >= prefill_batch:
                        break
                    k += 1
                batch = q[:k]
                del q[:k]
            else:
                batch = [r for r in q if r[0] <= now][:prefill_batch]
            if not batch:
                continue
            end = now + prefill_step_ms
            p_free[i] = end
            for r in batch:
                if not fifo:
                    q.remove(r)
                first = end + handoff_ms  # decode-attach: TTFT includes handoff
                r[2] = 1
                r[3] = first
                r[4] = first
                if state["completions"] >= warmup:
                    ttfts.append(first - r[1])
                if r[2] >= osl:
                    complete(r, first)
                else:
                    r[0] = first  # joins its decode worker when the KV lands
                    d_running[state["rr_d"] % decode_workers].append(r)
                    state["rr_d"] += 1

        for i, run in enumerate(d_running):  # one token per iteration
            if not run or d_free[i] > now:
                continue
            if fifo:
                k = 0
                for r in run:
                    if r[0] > now or k >= decode_max_seqs:
                        break
                    k += 1
                batch = run[:k]
            else:
                batch = [r for r in run if r[0] <= now][:decode_max_seqs]
            if not batch:
                continue
            end = now + decode_step_ms
            d_free[i] = end
            for r in batch:
                r[2] += 1
                r[5].append(end - r[4])
                r[4] = end
                if r[2] >= osl:
                    run.remove(r)
                    complete(r, end)
    else:
        raise RuntimeError("tandem did not converge within max_iters")

    window = now - (state["steady_start"] or 0.0)

    def q(v: list, x: float) -> float:
        return sorted(v)[min(len(v) - 1, max(0, round(x * len(v)) - 1))] if v else float("nan")

    return {
        "ttft_steady_mean": sum(ttfts) / len(ttfts) if ttfts else float("nan"),
        "ttft_steady_p50": q(ttfts, 0.5),
        "ttft_steady_p99": q(ttfts, 0.99),
        "tpot_mean": sum(tpots) / len(tpots) if tpots else float("nan"),
        "itl_p50": q(gaps, 0.5),
        "itl_p99": q(gaps, 0.99),
        "throughput_rps": state["steady_completions"] / (window / 1000.0) if window > 0 else 0.0,
    }


def refine_closed_loop_latency(df):
    """Post-process a result DataFrame using only its own columns.

    Agg rows (``ColumnsAgg`` + the recorded step timings) replay the fused
    pass calendar; disagg rows (``ColumnsDisagg`` with ``(p)prefill_step_ms``)
    replay the mini tandem. Returns a COPY with additive columns
    ``ttft_refined`` / ``tpot_refined`` / ``throughput_refined`` — existing
    columns (including the legacy ``ttft``) are untouched, and the three
    refined columns are jointly consistent (they satisfy the closed-loop
    identity C/X = TTFT + (osl-1)*TPOT), so consume them as a set. Rows that
    cannot be priced keep NaN.
    """
    out = df.copy()
    priced = [price_closed_loop_row(row) for _, row in out.iterrows()]
    out["ttft_refined"] = [p[0] for p in priced]
    out["tpot_refined"] = [p[1] for p in priced]
    out["throughput_refined"] = [p[2] for p in priced]
    return out


def price_closed_loop_row(row):
    """Price one result row -> ``(ttft_ms, tpot_ms, rps)``.

    ``row`` is any mapping (a summary ``dict`` or a DataFrame row); disagg
    rows are recognized by the ``(p)workers`` key. Returns a NaN triple when
    the row cannot be priced. Results are memoized on the pricing inputs —
    sweeps re-encounter the same operating point once per latency-target
    pair, and its refined values do not depend on the target.
    """
    try:
        if "(p)workers" in row:
            key = tuple(
                row[k]
                for k in (
                    "concurrency",
                    "isl",
                    "osl",
                    "(p)workers",
                    "(d)workers",
                    "(p)prefill_step_ms",
                    "tpot",
                    "(p)bs",
                    "(d)bs",
                )
            )
        else:
            key = tuple(row[k] for k in ("bs", "isl", "osl", "ctx_tokens", "prefill_step_ms", "genonly_step_ms")) + (
                row.get("prefix", 0) or 0,
            )
    except (KeyError, TypeError):
        return float("nan"), float("nan"), float("nan")
    cached = _PRICE_CACHE.get(key)
    if cached is None:
        cached = _price_closed_loop_row_uncached(row)
        if len(_PRICE_CACHE) < 65536:
            _PRICE_CACHE[key] = cached
    return cached


_PRICE_CACHE: dict = {}


def _price_closed_loop_row_uncached(row):
    try:
        if "(p)workers" in row:
            r = estimate_disagg_closed_loop_latency(
                concurrency=int(row["concurrency"]),
                isl=int(row["isl"]),
                osl=int(row["osl"]),
                prefill_workers=int(row["(p)workers"]),
                decode_workers=int(row["(d)workers"]),
                prefill_step_ms=float(row["(p)prefill_step_ms"]),
                decode_step_ms=float(row["tpot"]),
                prefill_batch=int(row["(p)bs"]),
                decode_max_seqs=int(row["(d)bs"]),
            )
        else:
            chunk_ref = max(1, min(int(row["ctx_tokens"]), int(row["isl"]) - int(row.get("prefix", 0) or 0)))
            step = float(row["prefill_step_ms"])
            t_gen = float(row["genonly_step_ms"])
            r = estimate_closed_loop_latency(
                concurrency=int(row["bs"]),
                isl=int(row["isl"]),
                osl=int(row["osl"]),
                # linear-in-tokens reconstruction of the recorded
                # per-chunk prefill cost; decode priced at the recorded
                # operating-point iteration time
                prefill_ms=lambda b, i, p, _s=step, _c=chunk_ref: _s * (b * max(1, i - p)) / _c,
                decode_ms=lambda b, c, _g=t_gen: _g,
                max_num_batched_tokens=int(row["ctx_tokens"]),
                prefix=int(row.get("prefix", 0) or 0),
            )
        if not (r["ttft_steady_mean"] == r["ttft_steady_mean"]):  # NaN guard
            raise ValueError("estimate returned NaN")
        return r["ttft_steady_mean"], r["tpot_mean"], r["throughput_rps"]
    except (KeyError, ValueError, TypeError, RuntimeError):
        return float("nan"), float("nan"), float("nan")


def filter_closed_loop_sla(df, ttft_ms=None, tpot_ms=None):
    """Approximate SLA post-filter on the refined closed-loop values.

    Refines ``df`` (see :func:`refine_closed_loop_latency`) and drops rows
    whose ``ttft_refined`` / ``tpot_refined`` exceed the given targets — the
    steady closed-loop values a fixed-concurrency benchmark at the row's
    operating point would measure. Rows that cannot be priced (refined NaN)
    are KEPT: they already passed the pipeline's legacy SLA filter and this
    post-filter only ever tightens that verdict where it has evidence.
    Returns the refined copy, filtered.
    """
    out = refine_closed_loop_latency(df)
    if ttft_ms is not None:
        out = out[~(out["ttft_refined"] > float(ttft_ms))]
    if tpot_ms is not None:
        out = out[~(out["tpot_refined"] > float(tpot_ms))]
    return out


# deployment identity: the columns that stay fixed while the concurrency
# ladder sweeps the operating point
_AGG_CONFIG_KEYS = ["model", "parallel"]
_DISAGG_CONFIG_KEYS = [
    "model",
    "(p)parallel",
    "(p)bs",
    "(p)workers",
    "(d)parallel",
    "(d)bs",
    "(d)workers",
]


def pick_under_closed_loop_sla(df, ttft_ms=None, tpot_ms=None):
    """Re-pick each deployment's operating point under the refined SLA.

    The pipeline picks operating points before this module runs, so
    post-filtering an already-picked table can drop a deployment whose
    lower-concurrency rows were still compliant. Feed this the FULL
    per-operating-point summary frame instead (sweep with a loose TTFT
    target so those rows survive): it applies
    :func:`filter_closed_loop_sla` and keeps each deployment's best
    surviving row — equivalent to enforcing the SLA at the original
    filter. Priced rows rank by ``throughput_refined``; rows the refiner
    cannot price keep their legacy verdict and rank by recorded
    ``tokens/s``.
    """
    out = filter_closed_loop_sla(df, ttft_ms=ttft_ms, tpot_ms=tpot_ms)
    if out.empty:
        return out
    keys = _DISAGG_CONFIG_KEYS if "(p)workers" in out.columns else _AGG_CONFIG_KEYS
    keys = [k for k in keys if k in out.columns]
    sort_cols = [c for c in ("throughput_refined", "tokens/s") if c in out.columns]
    ranked = out.sort_values(sort_cols, ascending=False, na_position="last")
    if not keys:
        return ranked
    return ranked.groupby(keys, dropna=False, sort=False).head(1)
