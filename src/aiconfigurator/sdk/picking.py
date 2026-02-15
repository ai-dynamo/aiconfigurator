# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone picking functions for selecting engine configurations.

These functions operate on perf DataFrames directly (ColumnsAgg, ColumnsDisagg,
or ColumnsStatic schemas) without requiring a TaskConfig.  They can be called
from either AIC's internal CLI pipeline or from an external real-GPU sweep.

Three picking modes are supported:

- :func:`pick_default` -- maximize throughput for a given GPU budget under SLA.
- :func:`pick_load_match` -- minimize GPUs for a target request-rate or
  concurrency under SLA.
- :func:`pick_autoscale` -- pick prefill and decode engines independently
  (no rate matching), returning a disagg config with 1 replica each.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from aiconfigurator.sdk import common

logger = logging.getLogger(__name__)

# comes from pipeline bubble, especially when benchmarking with concurrency
_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR = 0.9
# comes from not saturating the batchsize slot of decode worker
_RATE_MATCHING_DECODE_DEGRADATION_FACTOR = 0.92

# TTFT correction for concurrent prefill queueing: with N=10 batches and
# local concurrency lc=15-20, formula lc/20+0.95 gives ~1.8
_AUTOSCALE_TTFT_CORRECTION_FACTOR = 1.8


# ---------------------------------------------------------------------------
# Helper: build a disagg summary dict from prefill + decode dicts
# ---------------------------------------------------------------------------


def build_disagg_summary_dict(
    prefill_summary_dict: dict,
    prefill_num_worker: int,
    decode_summary_dict: dict,
    decode_num_worker: int,
    prefill_degradation_factor: float = _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
    decode_degradation_factor: float = _RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
) -> dict:
    """Build a disagg summary row from independent prefill and decode dicts.

    This is the standalone version of
    ``DisaggInferenceSession._get_disagg_summary_dict``.  It performs the same
    rate-matching arithmetic but does not require a session instance.

    Args:
        prefill_summary_dict: Row dict from a prefill ``ColumnsStatic`` DataFrame.
        prefill_num_worker: Number of prefill workers in the deployment.
        decode_summary_dict: Row dict from a decode ``ColumnsStatic`` DataFrame.
        decode_num_worker: Number of decode workers in the deployment.
        prefill_degradation_factor: Multiplicative degradation for prefill
            throughput during rate matching (default 0.9).
        decode_degradation_factor: Multiplicative degradation for decode
            throughput during rate matching (default 0.92).

    Returns:
        Dict with keys matching ``common.ColumnsDisagg``.
    """
    seq_s = min(
        prefill_summary_dict["seq/s"] * prefill_num_worker * prefill_degradation_factor,
        decode_summary_dict["seq/s"] * decode_num_worker * decode_degradation_factor,
    )
    prefill_gpus = prefill_summary_dict["pp"] * prefill_summary_dict["tp"] * prefill_summary_dict["dp"]
    decode_gpus = decode_summary_dict["pp"] * decode_summary_dict["tp"] * decode_summary_dict["dp"]
    num_total_gpus = prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker
    seq_s_gpu = seq_s / num_total_gpus if num_total_gpus > 0 else 0.0

    osl = prefill_summary_dict["osl"]
    tokens_s = seq_s * osl
    tokens_s_gpu = tokens_s / num_total_gpus if num_total_gpus > 0 else 0.0
    request_latency = prefill_summary_dict["ttft"] + decode_summary_dict["tpot"] * max(osl - 1, 0)

    # Weighted average power
    ttft = prefill_summary_dict["ttft"]
    tpot = decode_summary_dict["tpot"]
    decode_time = tpot * max(osl - 1, 0)
    prefill_power = prefill_summary_dict.get("power_w", 0.0)
    decode_power = decode_summary_dict.get("power_w", 0.0)
    total_time = ttft + decode_time
    disagg_power_avg = (prefill_power * ttft + decode_power * decode_time) / total_time if total_time > 0 else 0.0

    return {
        "model": prefill_summary_dict["model"],
        "isl": prefill_summary_dict["isl"],
        "osl": osl,
        "prefix": prefill_summary_dict["prefix"],
        "concurrency": decode_summary_dict["concurrency"] * decode_num_worker,
        "request_rate": seq_s,
        "(p)bs": prefill_summary_dict["bs"],
        "(p)global_bs": prefill_summary_dict["global_bs"],
        "(p)workers": prefill_num_worker,
        "(d)bs": decode_summary_dict["bs"],
        "(d)global_bs": decode_summary_dict["global_bs"],
        "(d)workers": decode_num_worker,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": request_latency,
        "seq/s": seq_s,
        "seq/s/gpu": seq_s_gpu,
        "tokens/s": tokens_s,
        "tokens/s/gpu": tokens_s_gpu,
        "tokens/s/user": decode_summary_dict["tokens/s/user"],
        "(p)seq/s/worker": prefill_summary_dict["seq/s"],
        "(d)seq/s/worker": decode_summary_dict["seq/s"],
        "num_total_gpus": num_total_gpus,
        "(p)tp": prefill_summary_dict["tp"],
        "(p)pp": prefill_summary_dict["pp"],
        "(p)dp": prefill_summary_dict["dp"],
        "(p)moe_tp": prefill_summary_dict["moe_tp"],
        "(p)moe_ep": prefill_summary_dict["moe_ep"],
        "(p)parallel": prefill_summary_dict["parallel"],
        "(p)gemm": prefill_summary_dict["gemm"],
        "(p)kvcache": prefill_summary_dict["kvcache"],
        "(p)fmha": prefill_summary_dict["fmha"],
        "(p)moe": prefill_summary_dict["moe"],
        "(p)comm": prefill_summary_dict["comm"],
        "(p)memory": prefill_summary_dict["memory"],
        "(p)backend": prefill_summary_dict.get("backend", ""),
        "(p)version": prefill_summary_dict.get("version", ""),
        "(p)system": prefill_summary_dict.get("system", ""),
        "(d)tp": decode_summary_dict["tp"],
        "(d)pp": decode_summary_dict["pp"],
        "(d)dp": decode_summary_dict["dp"],
        "(d)moe_tp": decode_summary_dict["moe_tp"],
        "(d)moe_ep": decode_summary_dict["moe_ep"],
        "(d)parallel": decode_summary_dict["parallel"],
        "(d)gemm": decode_summary_dict["gemm"],
        "(d)kvcache": decode_summary_dict["kvcache"],
        "(d)fmha": decode_summary_dict["fmha"],
        "(d)moe": decode_summary_dict["moe"],
        "(d)comm": decode_summary_dict["comm"],
        "(d)memory": decode_summary_dict["memory"],
        "(d)backend": decode_summary_dict.get("backend", ""),
        "(d)version": decode_summary_dict.get("version", ""),
        "(d)system": decode_summary_dict.get("system", ""),
        "power_w": disagg_power_avg,
    }


# ---------------------------------------------------------------------------
# Internal: extract best_latencies from a result DataFrame
# ---------------------------------------------------------------------------


def _extract_best_latencies(df: pd.DataFrame) -> dict[str, float]:
    """Extract latency info from the rank-1 row of a result DataFrame."""
    if df is None or df.empty:
        return {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0}
    row = df.iloc[0]
    result: dict[str, float] = {
        "ttft": float(row["ttft"]) if "ttft" in row.index else 0.0,
        "tpot": float(row["tpot"]) if "tpot" in row.index else 0.0,
        "request_latency": float(row["request_latency"]) if "request_latency" in row.index else 0.0,
    }
    if "replicas_needed" in row.index:
        result["replicas_needed"] = int(row["replicas_needed"])
    if "total_gpus_needed" in row.index:
        result["total_gpus_needed"] = int(row["total_gpus_needed"])
    if "load_served_pct" in row.index:
        result["load_served_pct"] = float(row["load_served_pct"])
    return result


# ---------------------------------------------------------------------------
# pick_default
# ---------------------------------------------------------------------------


def pick_default(
    pareto_df: pd.DataFrame,
    total_gpus: int,
    serving_mode: str,
    target_tpot: float | None = None,
    target_request_latency: float | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Pick configurations that maximize throughput for a fixed GPU budget.

    This is the standalone equivalent of the "default" picking path in
    ``process_experiment_result``.

    Args:
        pareto_df: Performance DataFrame (``ColumnsAgg`` or ``ColumnsDisagg``).
        total_gpus: Total GPUs available for deployment.
        serving_mode: ``"agg"`` or ``"disagg"``.
        target_tpot: TPOT SLA target in ms.  Used when
            ``target_request_latency`` is not set.
        target_request_latency: End-to-end request latency SLA in ms.  Takes
            precedence over ``target_tpot`` when set.
        top_n: Number of top configurations to return.

    Returns:
        Dict with keys:
        - ``best_config_df``: Top-N configurations DataFrame.
        - ``best_throughput``: ``tokens/s/gpu_cluster`` of the rank-1 config.
        - ``best_latencies``: Dict with ``ttft``, ``tpot``, ``request_latency``.
        - ``pareto_frontier_df``: Pareto frontier DataFrame.
    """
    from aiconfigurator.sdk.pareto_analysis import (
        get_best_configs_under_request_latency_constraint,
        get_best_configs_under_tpot_constraint,
        get_pareto_front,
    )

    use_request_latency = target_request_latency is not None and target_request_latency > 0

    if pareto_df is not None and not pareto_df.empty:
        if total_gpus > 0:
            pareto_df = pareto_df.copy()
            pareto_df["tokens/s/gpu_cluster"] = (
                pareto_df["tokens/s/gpu"]
                * (total_gpus // pareto_df["num_total_gpus"])
                * pareto_df["num_total_gpus"]
                / total_gpus
            )
        else:
            pareto_df = pareto_df.copy()
            pareto_df["tokens/s/gpu_cluster"] = pareto_df["tokens/s/gpu"]

        x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"
        pareto_frontier_df = get_pareto_front(
            pareto_df,
            x_axis_col,
            "tokens/s/gpu_cluster",
            maximize_x=not use_request_latency,
            maximize_y=True,
        )
    else:
        pareto_frontier_df = pd.DataFrame()

    group_by_key = "(d)parallel" if serving_mode == "disagg" else "parallel"

    if use_request_latency:
        best_config_df = get_best_configs_under_request_latency_constraint(
            total_gpus=total_gpus,
            pareto_df=pareto_df,
            target_request_latency=target_request_latency,
            top_n=top_n,
            group_by=group_by_key,
        )
    else:
        best_config_df = get_best_configs_under_tpot_constraint(
            total_gpus=total_gpus,
            pareto_df=pareto_df,
            target_tpot=target_tpot,
            top_n=top_n,
            group_by=group_by_key,
        )

    best_throughput = float(best_config_df["tokens/s/gpu_cluster"].values[0]) if not best_config_df.empty else 0.0

    return {
        "best_config_df": best_config_df,
        "best_throughput": best_throughput,
        "best_latencies": _extract_best_latencies(best_config_df),
        "pareto_frontier_df": pareto_frontier_df,
    }


# ---------------------------------------------------------------------------
# pick_load_match
# ---------------------------------------------------------------------------


def pick_load_match(
    pareto_df: pd.DataFrame,
    serving_mode: str,
    target_tpot: float | None = None,
    target_request_latency: float | None = None,
    target_request_rate: float | None = None,
    target_concurrency: float | None = None,
    max_total_gpus: int | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Pick configurations that minimize GPUs for a target load under SLA.

    This is the standalone equivalent of the "load-match" picking path.

    Args:
        pareto_df: Performance DataFrame (``ColumnsAgg`` or ``ColumnsDisagg``).
        serving_mode: ``"agg"`` or ``"disagg"``.
        target_tpot: TPOT SLA target in ms.
        target_request_latency: Request latency SLA in ms (takes precedence).
        target_request_rate: Target system request rate in req/s.
        target_concurrency: Target concurrent requests.
        max_total_gpus: Upper bound on total GPUs.
        top_n: Number of top configurations to return.

    Returns:
        Dict with keys:
        - ``best_config_df``: Top-N configurations DataFrame with
          ``replicas_needed``, ``total_gpus_needed``, ``load_served_pct``
          columns.
        - ``best_throughput``: Comparison metric (``1/total_gpus_needed``
          normally, ``tokens/s/gpu_cluster`` when GPU-capped).
        - ``best_latencies``: Dict with latency + load-match fields.
        - ``pareto_frontier_df``: Pareto frontier DataFrame.
    """
    from aiconfigurator.sdk.pareto_analysis import (
        get_best_configs_for_target_load,
        get_pareto_front,
    )

    use_request_latency = target_request_latency is not None and target_request_latency > 0

    pareto_frontier_df = pd.DataFrame()
    if pareto_df is not None and not pareto_df.empty:
        pareto_df = pareto_df.copy()
        pareto_df["tokens/s/gpu_cluster"] = pareto_df["tokens/s/gpu"]
        x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"
        pareto_frontier_df = get_pareto_front(
            pareto_df,
            x_axis_col,
            "tokens/s/gpu_cluster",
            maximize_x=not use_request_latency,
            maximize_y=True,
        )

    constraint_col = "request_latency" if use_request_latency else "tpot"
    constraint_value = target_request_latency if use_request_latency else target_tpot
    group_by_key = "(d)parallel" if serving_mode == "disagg" else "parallel"

    best_config_df = get_best_configs_for_target_load(
        pareto_df=pareto_df,
        constraint_col=constraint_col,
        constraint_value=constraint_value,
        target_request_rate=target_request_rate,
        target_concurrency=target_concurrency,
        top_n=top_n,
        max_total_gpus=max_total_gpus,
        group_by=group_by_key,
    )

    best_throughput = 0.0
    if not best_config_df.empty:
        row = best_config_df.iloc[0]
        if max_total_gpus and int(row.get("total_gpus_needed", 0)) >= max_total_gpus:
            best_throughput = float(row.get("tokens/s/gpu_cluster", 0.0))
        else:
            gpus = row.get("total_gpus_needed", 1)
            best_throughput = 1.0 / gpus if gpus > 0 else 0.0

    return {
        "best_config_df": best_config_df,
        "best_throughput": best_throughput,
        "best_latencies": _extract_best_latencies(best_config_df),
        "pareto_frontier_df": pareto_frontier_df,
    }


# ---------------------------------------------------------------------------
# pick_autoscale
# ---------------------------------------------------------------------------


def pick_autoscale(
    prefill_df: pd.DataFrame,
    decode_df: pd.DataFrame,
    target_ttft: float,
    target_tpot: float,
    top_n: int = 5,
) -> dict[str, Any]:
    """Pick prefill and decode engines independently for autoscaling.

    No rate matching is performed.  Returns disagg configs with
    ``(p)workers=1`` and ``(d)workers=1``.

    Args:
        prefill_df: Prefill candidates DataFrame (``ColumnsStatic`` schema).
        decode_df: Decode candidates DataFrame (``ColumnsStatic`` schema).
        target_ttft: TTFT SLA target in ms.
        target_tpot: TPOT SLA target in ms.
        top_n: Number of top combinations to return.

    Returns:
        Dict with keys:
        - ``best_config_df``: Top-N configurations DataFrame
          (``ColumnsDisagg`` schema, all rows have ``workers=1``).
        - ``best_latencies``: Dict with ``ttft``, ``tpot``,
          ``request_latency`` from the rank-1 config.
    """
    empty_result: dict[str, Any] = {
        "best_config_df": pd.DataFrame(),
        "best_latencies": {"ttft": 0.0, "tpot": 0.0, "request_latency": 0.0},
    }

    if prefill_df is None or prefill_df.empty or decode_df is None or decode_df.empty:
        return empty_result

    correction_factor = _AUTOSCALE_TTFT_CORRECTION_FACTOR

    # -- Filter prefill candidates by TTFT --
    prefill_candidates = prefill_df.copy()
    prefill_candidates["ttft_corrected"] = prefill_candidates["ttft"] * correction_factor
    prefill_meets_sla = prefill_candidates[prefill_candidates["ttft_corrected"] < target_ttft]

    if prefill_meets_sla.empty:
        logger.warning(
            "pick_autoscale: no prefill candidates meet TTFT < %sms (after %.1fx correction). "
            "Returning closest matches.",
            target_ttft,
            correction_factor,
        )
        prefill_candidates = (
            prefill_candidates.sort_values(by=["ttft_corrected", "seq/s/gpu"], ascending=[True, False])
            .drop_duplicates(subset=["parallel"], keep="first")
            .head(top_n)
            .reset_index(drop=True)
        )
    else:
        prefill_candidates = (
            prefill_meets_sla.sort_values(by=["seq/s/gpu", "global_bs"], ascending=[False, True])
            .drop_duplicates(subset=["parallel"], keep="first")
            .head(top_n)
            .reset_index(drop=True)
        )

    # -- Filter decode candidates by TPOT --
    decode_meets_sla = decode_df[decode_df["tpot"] < target_tpot].copy()

    if decode_meets_sla.empty:
        logger.warning(
            "pick_autoscale: no decode candidates meet TPOT < %sms. Returning closest matches.",
            target_tpot,
        )
        decode_candidates = (
            decode_df.copy()
            .sort_values(by=["tpot", "seq/s/gpu"], ascending=[True, False])
            .drop_duplicates(subset=["parallel"], keep="first")
            .head(top_n)
            .reset_index(drop=True)
        )
    else:
        decode_candidates = (
            decode_meets_sla.sort_values(by=["seq/s/gpu", "global_bs"], ascending=[False, True])
            .drop_duplicates(subset=["parallel"], keep="first")
            .head(top_n)
            .reset_index(drop=True)
        )

    # -- Combine: each prefill x each decode, workers=1, take top_n --
    all_combos: list[dict] = []
    for _, p_row in prefill_candidates.iterrows():
        for _, d_row in decode_candidates.iterrows():
            combo = build_disagg_summary_dict(
                prefill_summary_dict=p_row.to_dict(),
                prefill_num_worker=1,
                decode_summary_dict=d_row.to_dict(),
                decode_num_worker=1,
            )
            all_combos.append(combo)

    if not all_combos:
        return empty_result

    disagg_df = pd.DataFrame(all_combos, columns=common.ColumnsDisagg).round(3)
    disagg_df = disagg_df.sort_values(by=["tokens/s/gpu"], ascending=[False]).head(top_n).reset_index(drop=True)

    return {
        "best_config_df": disagg_df,
        "best_latencies": _extract_best_latencies(disagg_df),
    }
