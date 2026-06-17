# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an unrolled sample (flat config dict) into replay deployment inputs.

Produces the JSON payloads + worker counts the :class:`spica.evaluator.ReplayEvaluator`
feeds to Dynamo replay: per-role ``MockEngineArgs`` dicts (built from the AIC
parallelism), an optional ``PlannerConfig`` dict, worker counts, and router config.

Two replay paths, keyed on the planner policy (see investigation: no planner
config == static/"disabled"):

- ``enable_*_scaling`` both off  -> ``planner_config = None`` -> the plain
  ``run_trace_replay`` (static worker counts; emits gpu_hours, not goodput).
- otherwise -> a ``PlannerConfig`` -> the planner-bridge replay (scaling; goodput
  + gpu_hours). ``optimization_target`` is ``"sla"`` when throughput scaling is on
  (the only target that drives predictive throughput scaling), else ``"load"``.

Pure dict-building (no dynamo import), so it is unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SLATarget

# Per-engine GPU count and the goodput SLA are threaded so the report carries
# gpu_hours / goodput; the planner's own scaling SLA is set independently.

# Planner fields copied straight from the unrolled sample into the PlannerConfig
# payload when present (unroll_sample already decoded the presets to these).
_PLANNER_PASSTHROUGH = (
    "enable_throughput_scaling",
    "enable_load_scaling",
    "throughput_adjustment_interval_seconds",
    "load_adjustment_interval_seconds",
    "max_num_fpm_samples",
    "fpm_sample_bucket_size",
    "load_scaling_down_sensitivity",
    "load_min_observations",
    "load_predictor",
    "load_predictor_log1p",
    "prophet_window_size",
    "kalman_q_level",
    "kalman_q_trend",
    "kalman_r",
    "kalman_min_points",
)

# Load-scaling trigger thresholds the planner validator requires whenever
# optimization_target="load" (its own defaults are None). Fixed sensible v1
# values; the scale-down conservativeness knob is the separate
# load_scaling_down_sensitivity preset. Decode is keyed on KV-cache utilization
# (% 0-100, up > down); prefill (disagg only) on queued prefill tokens.
_LOAD_DECODE_SCALE_UP_KV_RATE = 80.0
_LOAD_DECODE_SCALE_DOWN_KV_RATE = 40.0
_LOAD_PREFILL_SCALE_UP_QUEUE_TOKENS = 1024
_LOAD_PREFILL_SCALE_DOWN_QUEUE_TOKENS = 0


@dataclass(frozen=True)
class DeploymentPlan:
    """Everything the evaluator needs to run one replay for a candidate."""

    deployment_mode: str  # "agg" | "disagg"
    is_static: bool  # True -> plain replay (no planner); False -> planner bridge
    # MockEngineArgs JSON payloads (omit num_gpu_blocks -> replay estimates it).
    agg_engine_args: dict[str, Any] | None
    prefill_engine_args: dict[str, Any] | None
    decode_engine_args: dict[str, Any] | None
    num_workers: int  # agg replica count
    num_prefill_workers: int
    num_decode_workers: int
    router_mode: str
    router_config: dict[str, Any] | None  # kv-router knobs, or None for round_robin
    planner_config: dict[str, Any] | None  # PlannerConfig payload, or None when static


def _role_prefix(role: str) -> str:
    """Field prefix in the unrolled sample for a role ('' for agg shape fields)."""
    return "" if role == "agg" else f"{role}_"


def _engine_args_payload(sample: dict, role: str, *, backend_version: str) -> dict[str, Any]:
    """MockEngineArgs JSON for one role, from the sample's AIC parallelism."""
    p = _role_prefix(role)
    tp = int(sample[f"{p}tp"])
    attention_dp = int(sample[f"{p}attention_dp"])
    moe_tp = int(sample[f"{p}moe_tp"])
    moe_ep = int(sample[f"{p}moe_ep"])
    payload: dict[str, Any] = {
        "worker_type": "aggregated" if role == "agg" else role,
        "aic_backend": sample["backend"],
        "aic_backend_version": backend_version,
        "aic_system": sample["hardware_sku"],
        "aic_model_path": sample["model_name"],
        "aic_tp_size": tp,
        "aic_attention_dp_size": attention_dp,
        # batching + memory knobs for the role
        "max_num_batched_tokens": int(sample[f"{role}_max_num_batched_tokens"]),
        "max_num_seqs": int(sample[f"{role}_max_num_seqs"]),
        "block_size": int(sample[f"{role}_block_size"]),
        "gpu_memory_utilization": float(sample[f"{role}_gpu_memory_utilization"]),
        "enable_prefix_caching": bool(sample[f"{role}_enable_prefix_caching"]),
    }
    # MoE expert sharding only for MoE shapes (tp*attention_dp == moe_tp*moe_ep);
    # dense (moe_tp==moe_ep==1) leaves them unset.
    if moe_tp * moe_ep > 1:
        payload["aic_moe_tp_size"] = moe_tp
        payload["aic_moe_ep_size"] = moe_ep
    if sample.get("aic_nextn") is not None:
        payload["aic_nextn"] = int(sample["aic_nextn"])
    return payload


def _planner_config_payload(sample: dict, *, planner_sla: SLATarget | None) -> dict[str, Any] | None:
    """PlannerConfig JSON for a scaling candidate, or None when the planner is
    disabled (static -> plain replay)."""
    enable_throughput = bool(sample.get("enable_throughput_scaling", False))
    enable_load = bool(sample.get("enable_load_scaling", False))
    if not (enable_throughput or enable_load):
        return None  # "disabled" -> static plain replay

    payload: dict[str, Any] = {"mode": sample["deployment_mode"]}
    # The only target that drives predictive throughput scaling is "sla"; a non-sla
    # target forces load-only scaling (planner validator). hybrid (both) -> "sla".
    payload["optimization_target"] = "sla" if enable_throughput else "load"
    # Spica consumes the trace_report directly and sweeps many candidates, so turn
    # off the planner's per-run diagnostics (a ~5 MB HTML in ./planner_reports/ per
    # candidate) and the live dashboard, both of which default on.
    payload["report_interval_hours"] = None
    payload["live_dashboard_port"] = 0
    for key in _PLANNER_PASSTHROUGH:
        if key in sample:
            payload[key] = sample[key]
    # Per-engine GPU counts for the planner's cost logic (gpu_hours itself is
    # derived by the mocker from aic_tp x aic_attention_dp, not from these).
    if sample["deployment_mode"] == "disagg":
        payload["prefill_engine_num_gpu"] = int(sample["prefill_tp"]) * int(sample["prefill_attention_dp"])
        payload["decode_engine_num_gpu"] = int(sample["decode_tp"]) * int(sample["decode_attention_dp"])
    else:
        payload["decode_engine_num_gpu"] = int(sample["tp"]) * int(sample["attention_dp"])
    # Load scaling needs explicit trigger thresholds (planner defaults are None).
    if payload["optimization_target"] == "load":
        payload["decode_scale_up_kv_rate"] = _LOAD_DECODE_SCALE_UP_KV_RATE
        payload["decode_scale_down_kv_rate"] = _LOAD_DECODE_SCALE_DOWN_KV_RATE
        if sample["deployment_mode"] == "disagg":
            payload["prefill_scale_up_queue_tokens"] = _LOAD_PREFILL_SCALE_UP_QUEUE_TOKENS
            payload["prefill_scale_down_queue_tokens"] = _LOAD_PREFILL_SCALE_DOWN_QUEUE_TOKENS
    # Planner's own scaling SLA (independent of the goodput SLA): seed from the
    # goal's SLA when it carries ttft+itl, else the planner defaults stand.
    if payload["optimization_target"] == "sla" and planner_sla is not None:
        if planner_sla.ttft_ms is not None:
            payload["ttft_ms"] = planner_sla.ttft_ms
        if planner_sla.itl_ms is not None:
            payload["itl_ms"] = planner_sla.itl_ms
    return payload


def _router_config_payload(sample: dict) -> dict[str, Any] | None:
    """kv-router knobs, or None under round_robin."""
    if sample.get("router_mode") != "kv_router":
        return None
    keys = (
        "overlap_score_credit",
        "prefill_load_scale",
        "host_cache_hit_weight",
        "disk_cache_hit_weight",
        "router_temperature",
    )
    return {k: sample[k] for k in keys if k in sample}


def build_deployment(sample: dict, *, backend_version: str, planner_sla: SLATarget | None = None) -> DeploymentPlan:
    """Translate one unrolled sample into a :class:`DeploymentPlan`."""
    mode = sample["deployment_mode"]
    planner_config = _planner_config_payload(sample, planner_sla=planner_sla)
    router_mode = sample.get("router_mode", "round_robin")
    common = dict(
        deployment_mode=mode,
        is_static=planner_config is None,
        router_mode=router_mode,
        router_config=_router_config_payload(sample),
        planner_config=planner_config,
    )
    if mode == "agg":
        return DeploymentPlan(
            agg_engine_args=_engine_args_payload(sample, "agg", backend_version=backend_version),
            prefill_engine_args=None,
            decode_engine_args=None,
            num_workers=int(sample["replicas"]),
            num_prefill_workers=0,
            num_decode_workers=0,
            **common,
        )
    return DeploymentPlan(
        agg_engine_args=None,
        prefill_engine_args=_engine_args_payload(sample, "prefill", backend_version=backend_version),
        decode_engine_args=_engine_args_payload(sample, "decode", backend_version=backend_version),
        num_workers=0,
        num_prefill_workers=int(sample["prefill_replicas"]),
        num_decode_workers=int(sample["decode_replicas"]),
        **common,
    )
