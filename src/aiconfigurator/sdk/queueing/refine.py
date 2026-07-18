# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantitative-tier refinement: stage 2 of the sweep SLA funnel.

Stage 1 (screening) keeps every candidate whose cohort-bracket lower bound
passes the TTFT SLA — wide-keep, so screening bias can never falsely
reject. This module re-scores candidates with the limit-cycle evaluator
(SDK phase-runner timing) and enforces the SLA constraints at their
requested percentiles on the refined distributions. Refined rows carry
``queueing_tier == "quantitative"``.

Constraint semantics: each SLA target is a (value_ms, percentile) pair on
the steady-state distribution — supported percentiles 0.5/0.75/0.9/0.95/
0.99/0.999. The cohort bracket [lo, hi] bounds the distribution's support,
so one bracket serves every percentile: reject when lo violates, pass
without refinement when hi complies, refine the straddlers.

Note on p999: a deterministic stationary model yields the *calendar* p999
(the tail of the limit cycle's mass distribution). Real-deployment p999 is
additionally driven by stochastic effects outside the model (length
variance, stragglers), so calendar-p999 should be read as a lower-bound
style estimate.

Cost model: one evaluator run is ~ms-scale and proportional to
concurrency x osl; a per-call budget caps total work, and skipped
candidates are kept (conservative) with their screening tier visible —
no silent drops.
"""

from __future__ import annotations

import logging

import pandas as pd

from .calendar import evaluate_closed_loop
from .spec import EngineSpec, QueueingReport, WorkloadSpec
from .timing import DatabaseTimingModel

logger = logging.getLogger(__name__)

SUPPORTED_PERCENTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.999)

# evaluator cost grows ~ concurrency * osl (slot-updates per limit cycle);
# above this the refine would dominate sweep time, so we keep the screening
# estimate (conservative direction: the row stays, tier stays visible)
_MAX_EVAL_COMPLEXITY = 120_000


def _report_quantile(rep: QueueingReport, metric: str, q: float) -> float:
    dist = {"ttft": rep.ttft_steady, "tpot": rep.tpot, "itl": rep.itl, "e2e": rep.e2e}[metric]
    return dist.quantile(q)


def refine_rows(
    df: pd.DataFrame,
    indices,
    *,
    model,
    database,
    backend,
    enable_chunked_prefill: bool = True,
    max_refines: int = 8,
) -> dict:
    """Re-score `indices` rows in place with the evaluator.

    Returns {index: QueueingReport} for the rows actually refined. Rows
    beyond `max_refines`, multimodal rows (encoder latency is outside the
    queueing model), and rows above the evaluator complexity cap are left
    at screening tier and logged.
    """
    # both caches live on the backend instance so they survive across the
    # per-(ttft,tpot)-pair sweep calls for the same parallel config
    timing = getattr(backend, "_queueing_timing_model", None)
    if timing is None or timing._model is not model:
        timing = DatabaseTimingModel(model, database, backend)
        backend._queueing_timing_model = timing
    report_cache = getattr(backend, "_queueing_report_cache", None)
    if report_cache is None:
        report_cache = {}
        backend._queueing_report_cache = report_cache
    calendar_backend = database.backend if database.backend in ("vllm", "sglang", "trtllm") else "vllm"
    refined: dict = {}
    skipped_budget = 0
    skipped_scope = 0

    for idx in indices:
        if len(refined) >= max_refines:
            skipped_budget += 1
            continue
        row = df.loc[idx]
        bs = int(row["bs"])
        osl = int(row["osl"])
        if float(row.get("encoder_latency", 0.0) or 0.0) > 0.0 or bs * osl > _MAX_EVAL_COMPLEXITY:
            skipped_scope += 1
            continue

        wl = WorkloadSpec(
            isl=int(row["isl"]),
            osl=osl,
            prefix=int(row.get("prefix", 0) or 0),
            concurrency=bs,
        )
        # reconstruct the engine budget from the operating point: the sweep's
        # ctx_tokens is the prefill share left after the running decodes
        # (B_eff), so B = ctx_tokens + bs. max_num_seqs mirrors the
        # generator's deployment rule (engine batch cap >= swept bs).
        eng = EngineSpec(
            max_num_batched_tokens=int(row["ctx_tokens"]) + bs,
            max_num_seqs=max(256, bs),
            enable_chunked_prefill=enable_chunked_prefill,
        )
        cache_key = (
            wl.isl,
            wl.osl,
            wl.prefix,
            bs,
            eng.max_num_batched_tokens,
            enable_chunked_prefill,
            calendar_backend,
        )
        rep = report_cache.get(cache_key)
        if rep is None:
            # short window: 2 warmup + 2 sampled generations — quantile
            # resolution is bounded by the limit cycle's mass structure,
            # not by sample count, so the short window loses little
            rep = evaluate_closed_loop(
                wl, eng, timing, backend=calendar_backend, warmup_generations=2, window_generations=2
            )
            report_cache[cache_key] = rep

        p99 = rep.ttft_steady.p99
        df.loc[idx, "ttft_steady_mean"] = rep.ttft_steady.mean
        df.loc[idx, "ttft_steady_p50"] = rep.ttft_steady.p50
        df.loc[idx, "ttft_steady_p90"] = rep.ttft_steady.p90
        df.loc[idx, "ttft_steady_p99"] = p99
        # the bracket collapses once the quantitative tier has spoken
        df.loc[idx, "ttft_steady_p99_lo"] = p99
        df.loc[idx, "ttft_steady_p99_hi"] = p99
        df.loc[idx, "ttft_transient_mean"] = rep.ttft_transient.mean
        df.loc[idx, "ttft_transient_max"] = rep.ttft_transient.maximum
        df.loc[idx, "itl_mean"] = rep.itl.mean
        df.loc[idx, "itl_p50"] = rep.itl.p50
        df.loc[idx, "itl_p99"] = rep.itl.p99
        df.loc[idx, "queueing_tier"] = "quantitative"
        refined[idx] = rep

    if skipped_budget or skipped_scope:
        logger.info(
            "queueing refine: %d refined, %d kept at screening tier (budget), "
            "%d out of evaluator scope (multimodal / complexity)",
            len(refined),
            skipped_budget,
            skipped_scope,
        )
    return refined


def apply_sla_funnel(
    df: pd.DataFrame,
    *,
    model,
    database,
    backend,
    constraints: dict,
    enable_chunked_prefill: bool = True,
    top_k: int = 0,
    max_refines: int = 8,
    refine_top: bool = False,
) -> pd.DataFrame:
    """Resolve percentile SLA feasibility on a wide-kept candidate set.

    With ``refine_top=False`` (sweep hot path) only feasibility is resolved
    — the tier upgrade of the finally-reported rows belongs to the report
    boundary, done once, not inside every per-constraint sweep call.

    `constraints` maps metric name ("ttft" | "tpot" | "itl" | "e2e") to a
    (target_ms, percentile) pair; only present metrics are enforced.

    1. TTFT straddlers (bracket crosses the target) are refined in
       throughput order and dropped if the refined quantile violates;
    2. unrefined straddlers are KEPT — conservative — with screening tier
       visible in ``queueing_tier``;
    3. optionally (``refine_top``) the top rows are also refined so their
       reported numbers are quantitative; every refined row is checked
       against ALL requested constraints.
    """
    if df.empty or not constraints:
        return df
    for metric, (_, q) in constraints.items():
        if q not in SUPPORTED_PERCENTILES:
            raise ValueError(f"unsupported percentile {q} for {metric}; supported: {SUPPORTED_PERCENTILES}")
    df = df.sort_values(by="seq/s", ascending=False)

    def _enforce(reports: dict) -> pd.DataFrame:
        drop = []
        for idx, rep in reports.items():
            for metric, (target, q) in constraints.items():
                if _report_quantile(rep, metric, q) > target:
                    drop.append(idx)
                    break
        return df.drop(index=drop) if drop else df

    budget = max_refines
    if "ttft" in constraints:
        ttft_target = constraints["ttft"][0]
        # lazy resolution in throughput order: selection only ever consumes
        # the best feasible rows, so once `top_k` rows are confirmed
        # feasible, remaining straddlers stay unresolved-and-kept
        # (conservative; their screening tier is visible)
        need = max(1, top_k)
        confirmed = 0
        to_refine: list = []
        for i in df.index:
            hi = df.at[i, "ttft_steady_p99_hi"]
            if pd.isna(hi):
                continue  # external row without queueing columns
            if hi <= ttft_target:
                confirmed += 1
            else:
                to_refine.append(i)
            if confirmed >= need and not to_refine:
                break
            if confirmed >= need:
                break
        reports = refine_rows(
            df,
            to_refine,
            model=model,
            database=database,
            backend=backend,
            enable_chunked_prefill=enable_chunked_prefill,
            max_refines=budget,
        )
        budget -= len(reports)
        df = _enforce(reports)

    if refine_top and top_k > 0 and budget > 0 and not df.empty:
        head = [i for i in df.index[:top_k] if df.at[i, "queueing_tier"] == "screening"]
        if head:
            reports = refine_rows(
                df,
                head,
                model=model,
                database=database,
                backend=backend,
                enable_chunked_prefill=enable_chunked_prefill,
                max_refines=budget,
            )
            df = _enforce(reports)
    return df


def refine_report_rows(df: pd.DataFrame, max_refines: int = 32) -> pd.DataFrame:
    """Report-boundary tier upgrade: refine the rows a human will read.

    Feasibility was already resolved by the sweep funnel; this pass only
    upgrades the remaining screening-tier rows of a final (top-N) frame to
    quantitative numbers. No rows are dropped here — for certain-pass rows
    the bracket guarantees the refined p99 also complies (p99 <= hi <=
    target by construction).

    Self-contained: (model, database, backend) are rebuilt from each row's
    own metadata (model/backend/version/system + parallelism columns), so
    it can run at the report boundary where sweep-time objects are gone.
    Any per-group rebuild failure is logged and skipped — the report path
    must never crash on a tier upgrade.
    """
    if df is None or df.empty or "queueing_tier" not in df.columns:
        return df
    mask = df["queueing_tier"] == "screening"
    if not mask.any():
        return df

    from aiconfigurator.sdk import config as sdk_config
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.models import get_model
    from aiconfigurator.sdk.perf_database import get_database

    group_cols = ["model", "backend", "version", "system", "tp", "pp", "dp", "moe_tp", "moe_ep", "cp"]
    budget = max_refines
    for key, group in df[mask].groupby(group_cols, dropna=False):
        if budget <= 0:
            logger.info("report tier upgrade: refine budget exhausted, %d rows left at screening tier", int(mask.sum()))
            break
        (model_path, backend_name, version, system, tp, pp, dp, moe_tp, moe_ep, cp) = key
        try:
            database = get_database(system=system, backend=backend_name, version=str(version))
            model_config = sdk_config.ModelConfig(
                tp_size=int(tp),
                pp_size=int(pp),
                moe_tp_size=int(moe_tp) if moe_tp else None,
                moe_ep_size=int(moe_ep) if moe_ep else None,
                attention_dp_size=int(dp) if dp else 1,
                cp_size=int(cp) if cp else 1,
            )
            model = get_model(model_path=model_path, model_config=model_config, backend_name=backend_name)
            backend = get_backend(backend_name)
        except Exception:
            logger.warning(
                "report tier upgrade skipped for %s/%s tp%s pp%s (rebuild failed)",
                model_path,
                backend_name,
                tp,
                pp,
                exc_info=True,
            )
            continue
        reports = refine_rows(
            df, list(group.index), model=model, database=database, backend=backend, max_refines=budget
        )
        budget -= len(reports)
    return df.round(3)
