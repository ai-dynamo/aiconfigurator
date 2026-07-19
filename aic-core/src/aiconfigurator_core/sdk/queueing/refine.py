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

import dataclasses
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
    runtime_config=None,
) -> dict:
    """Re-score `indices` rows in place with the evaluator.

    Returns {index: QueueingReport} for the rows actually refined. Rows
    beyond `max_refines` and rows above the evaluator complexity cap are
    left at screening tier and logged.

    Multimodal rows are supported when `runtime_config` (carrying the image
    parameters) is provided: the vision tokens join the prefill length and
    the encoder latency shifts the TTFT/e2e distributions additively —
    matching run_agg's own composition. Without `runtime_config` such rows
    are skipped (screening tier stays visible).
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
        encoder_ms = float(row.get("encoder_latency", 0.0) or 0.0)
        if bs * osl > _MAX_EVAL_COMPLEXITY:
            skipped_scope += 1
            continue
        visual_tokens = 0
        if encoder_ms > 0.0:
            if runtime_config is None:
                skipped_scope += 1
                continue
            visual_tokens = int(backend._visual_context_tokens(model, runtime_config))

        wl = WorkloadSpec(
            isl=int(row["isl"]) + visual_tokens,
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
        # additive per-request TTFT stages ahead of / around the prefill —
        # the encoder stage and the per-request CPU dispatch overhead, the
        # same terms run_agg's legacy ttft and the screening columns carry;
        # shift copies so the cached report stays clean
        dispatch_ms = float(getattr(backend, "_prefill_dispatch_overhead_ms", lambda _m: 0.0)(model))
        shift_ms = encoder_ms + dispatch_ms
        if shift_ms > 0.0:
            rep = dataclasses.replace(
                rep,
                ttft_steady=rep.ttft_steady.shifted(shift_ms),
                ttft_transient=rep.ttft_transient.shifted(shift_ms),
                e2e=rep.e2e.shifted(shift_ms),
            )

        p99 = rep.ttft_steady.p99
        df.loc[idx, "ttft_steady_mean"] = rep.ttft_steady.mean
        df.loc[idx, "ttft_steady_p50"] = rep.ttft_steady.p50
        df.loc[idx, "ttft_steady_p75"] = rep.ttft_steady.quantile(0.75)
        df.loc[idx, "ttft_steady_p90"] = rep.ttft_steady.p90
        df.loc[idx, "ttft_steady_p95"] = rep.ttft_steady.quantile(0.95)
        df.loc[idx, "ttft_steady_p99"] = p99
        df.loc[idx, "ttft_steady_p999"] = rep.ttft_steady.quantile(0.999)
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
        logger.debug(
            "queueing refine: %d refined, %d kept at screening tier (budget), "
            "%d out of evaluator scope (complexity / multimodal without runtime context)",
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
    runtime_config=None,
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

    Enforcement is deliberately asymmetric across tiers: rows that pass on
    the bracket alone (``hi <= target``) keep only their screening-tier
    screens for the other metrics (they have no evaluator distributions),
    while refined rows face every requested constraint at its percentile.
    A straddler can therefore be dropped on a non-TTFT constraint that a
    certain-pass row was never tested against — conservative in the keep
    direction, and visible via ``queueing_tier``.
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
            runtime_config=runtime_config,
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
                runtime_config=runtime_config,
            )
            df = _enforce(reports)
    return df


def refine_report_rows(df: pd.DataFrame, max_refines: int = 32, runtime_config=None) -> pd.DataFrame:
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
    mask_composed = df["queueing_tier"] == "composed"
    if not mask.any() and not mask_composed.any():
        return df

    budget = max_refines
    agg_group_cols = ["model", "backend", "version", "system", "tp", "pp", "dp", "moe_tp", "moe_ep", "cp"]
    if not mask.any() or any(col not in df.columns for col in agg_group_cols):
        agg_groups = []
    else:
        agg_groups = df[mask].groupby(agg_group_cols, dropna=False)
    for key, group in agg_groups:
        if budget <= 0:
            logger.info("report tier upgrade: refine budget exhausted, %d rows left at screening tier", int(mask.sum()))
            break
        (model_path, backend_name, version, system, tp, pp, dp, moe_tp, moe_ep, cp) = key
        rebuilt = _rebuild_stage(model_path, backend_name, version, system, tp, pp, dp, moe_tp, moe_ep, cp)
        if rebuilt is None:
            continue
        model, database, backend = rebuilt
        reports = refine_rows(
            df,
            list(group.index),
            model=model,
            database=database,
            backend=backend,
            max_refines=budget,
            runtime_config=runtime_config,
        )
        budget -= len(reports)

    if mask_composed.any() and budget > 0:
        budget = _refine_disagg_report_rows(df, mask_composed, budget)
    return df.round(3)


def _rebuild_stage(model_path, backend_name, version, system, tp, pp, dp, moe_tp, moe_ep, cp):
    """Rebuild (model, database, backend) from row metadata; None on failure
    (the report path must never crash on a tier upgrade)."""
    from aiconfigurator_core.sdk import config as sdk_config
    from aiconfigurator_core.sdk.backends.factory import get_backend
    from aiconfigurator_core.sdk.models import get_model
    from aiconfigurator_core.sdk.perf_database import get_database

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
        return model, database, get_backend(backend_name)
    except Exception:
        logger.warning(
            "report tier upgrade skipped for %s/%s tp%s pp%s (rebuild failed)",
            model_path,
            backend_name,
            tp,
            pp,
            exc_info=True,
        )
        return None


_P_COLS = ("(p)backend", "(p)version", "(p)system", "(p)tp", "(p)pp", "(p)dp", "(p)moe_tp", "(p)moe_ep", "(p)cp")
_D_COLS = ("(d)backend", "(d)version", "(d)system", "(d)tp", "(d)pp", "(d)dp", "(d)moe_tp", "(d)moe_ep")


def _refine_disagg_report_rows(df: pd.DataFrame, mask_composed: pd.Series, budget: int) -> int:
    """Upgrade disagg rows from the composed (static-scalar) tier to the
    tandem-recursion quantitative tier.

    Both stages are rebuilt from the row's own (p)/(d) metadata and priced
    with their own timing models (heterogeneous P/D supported). The
    KV-transfer fabric is wired rank-locally: per-GPU KV bytes per token
    (``get_kvcache_bytes_per_sequence(1)``) over the per-GPU
    ``node.inter_node_bw`` of each stage's system spec (cross-node
    placement assumption). Phase-mixed output (``evaluate_disagg_mixed``)
    keeps the reported numbers robust to the tandem system's cohort-phase
    multi-stability. Multimodal rows stay composed (visible tier).
    """
    from .disagg import DisaggSpec, evaluate_disagg_mixed
    from .spec import EngineSpec, WorkloadSpec
    from .timing import DatabaseTimingModel

    group_cols = ["model", *_P_COLS, *_D_COLS]
    if any(col not in df.columns for col in group_cols):
        return budget

    for key, group in df[mask_composed].groupby(group_cols, dropna=False):
        if budget <= 0:
            logger.info("report tier upgrade: refine budget exhausted, disagg rows left at composed tier")
            break
        eligible = [
            idx
            for idx in group.index
            if float(df.at[idx, "encoder_latency"] or 0.0) <= 0.0  # multimodal stays composed, visibly
            and int(df.at[idx, "concurrency"]) * int(df.at[idx, "osl"]) <= _MAX_EVAL_COMPLEXITY
        ]
        if not eligible:
            continue
        model_path = key[0]
        p_meta = dict(zip(_P_COLS, key[1 : 1 + len(_P_COLS)], strict=True))
        d_meta = dict(zip(_D_COLS, key[1 + len(_P_COLS) :], strict=True))
        p_rebuilt = _rebuild_stage(
            model_path,
            p_meta["(p)backend"],
            p_meta["(p)version"],
            p_meta["(p)system"],
            p_meta["(p)tp"],
            p_meta["(p)pp"],
            p_meta["(p)dp"],
            p_meta["(p)moe_tp"],
            p_meta["(p)moe_ep"],
            p_meta["(p)cp"],
        )
        d_rebuilt = _rebuild_stage(
            model_path,
            d_meta["(d)backend"],
            d_meta["(d)version"],
            d_meta["(d)system"],
            d_meta["(d)tp"],
            d_meta["(d)pp"],
            d_meta["(d)dp"],
            d_meta["(d)moe_tp"],
            d_meta["(d)moe_ep"],
            1,
        )
        if p_rebuilt is None or d_rebuilt is None:
            continue
        p_model, p_db, p_backend = p_rebuilt
        d_model, d_db, d_backend = d_rebuilt
        p_timing = DatabaseTimingModel(p_model, p_db, p_backend)
        d_timing = DatabaseTimingModel(d_model, d_db, d_backend)
        try:
            kv_bpt = int(p_model.get_kvcache_bytes_per_sequence(1))
            egress = float(p_db.system_spec["node"]["inter_node_bw"])
            ingress = float(d_db.system_spec["node"]["inter_node_bw"])
        except Exception:
            logger.warning("disagg tier upgrade skipped (no transfer spec derivable)", exc_info=True)
            continue

        for idx in group.index:
            if budget <= 0:
                break
            row = df.loc[idx]
            if float(row.get("encoder_latency", 0.0) or 0.0) > 0.0:
                continue  # multimodal disagg out of the tandem model's scope
            c = int(row["concurrency"])
            osl = int(row["osl"])
            if c * osl > _MAX_EVAL_COMPLEXITY:
                continue
            wl = WorkloadSpec(isl=int(row["isl"]), osl=osl, prefix=int(row.get("prefix", 0) or 0), concurrency=c)
            # the disagg prefill worker runs static batches of (p)bs prompts:
            # a per-pass token budget of (p)bs * effective_isl reproduces that
            p_eng = EngineSpec(max_num_batched_tokens=max(1, int(row["(p)bs"])) * wl.effective_isl)
            d_eng = EngineSpec(max_num_seqs=max(1, int(row["(d)bs"])))
            spec = DisaggSpec(
                num_prefill_workers=max(1, int(row["(p)workers"])),
                num_decode_workers=max(1, int(row["(d)workers"])),
                kv_bytes_per_token=kv_bpt,
                egress_bytes_per_s=egress,
                ingress_bytes_per_s=ingress,
            )
            try:
                rep = evaluate_disagg_mixed(
                    wl, p_eng, d_eng, p_timing, d_timing, spec, backend=str(d_meta["(d)backend"])
                )
            except Exception:
                logger.warning("disagg tier upgrade failed for row %s", idx, exc_info=True)
                continue
            p99 = rep.ttft_steady.p99
            df.loc[idx, "ttft_steady_mean"] = rep.ttft_steady.mean
            df.loc[idx, "ttft_steady_p50"] = rep.ttft_steady.p50
            df.loc[idx, "ttft_steady_p75"] = rep.ttft_steady.quantile(0.75)
            df.loc[idx, "ttft_steady_p90"] = rep.ttft_steady.p90
            df.loc[idx, "ttft_steady_p95"] = rep.ttft_steady.quantile(0.95)
            df.loc[idx, "ttft_steady_p99"] = p99
            df.loc[idx, "ttft_steady_p999"] = rep.ttft_steady.quantile(0.999)
            df.loc[idx, "ttft_steady_p99_lo"] = p99
            df.loc[idx, "ttft_steady_p99_hi"] = p99
            df.loc[idx, "ttft_transient_mean"] = rep.ttft_transient.mean
            df.loc[idx, "ttft_transient_max"] = rep.ttft_transient.maximum
            df.loc[idx, "itl_mean"] = rep.itl.mean
            df.loc[idx, "itl_p50"] = rep.itl.p50
            df.loc[idx, "itl_p99"] = rep.itl.p99
            df.loc[idx, "queueing_tier"] = "quantitative"
            budget -= 1
    return budget
