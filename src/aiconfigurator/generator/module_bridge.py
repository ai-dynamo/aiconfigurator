# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from typing import Any

import pandas as pd

from aiconfigurator.sdk.task import TaskConfig

from .aggregators import collect_generator_params
from .rendering import apply_defaults


def _deep_merge(target: dict, extra: dict | None) -> dict:
    if not extra:
        return target
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _series_val(series: pd.Series, key: str, default=None):
    val = series.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    if pd.isna(val):
        return default
    return val


def _safe_int(val, default: int = 0) -> int:
    try:
        if val is None:
            return default
        if isinstance(val, float) and math.isnan(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        if isinstance(val, float) and math.isnan(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def task_config_to_generator_config(
    task_config: TaskConfig,
    result_df: pd.Series,
    generator_overrides: dict | None = None,
) -> dict:
    """Convert a task config/result row into unified generator parameters."""

    overrides = copy.deepcopy(generator_overrides or {})

    def _build_worker_params(prefix: str, extra_overrides: dict | None) -> tuple[dict, int]:
        workers = _safe_int(_series_val(result_df, f"{prefix}workers", 1), 1)
        tp = _safe_int(_series_val(result_df, f"{prefix}tp", 1), 1)
        pp = _safe_int(_series_val(result_df, f"{prefix}pp", 1), 1)
        dp = _safe_int(_series_val(result_df, f"{prefix}dp", 1), 1)
        moe_tp = _safe_int(_series_val(result_df, f"{prefix}moe_tp", 1), 1)
        moe_ep = _safe_int(_series_val(result_df, f"{prefix}moe_ep", 1), 1)
        bs = _safe_int(_series_val(result_df, f"{prefix}bs", 1), 1)
        memory = _safe_float(_series_val(result_df, f"{prefix}memory", None), None)

        quant = {
            "gemm_quant_mode": _series_val(result_df, f"{prefix}gemm", None),
            "moe_quant_mode": _series_val(result_df, f"{prefix}moe", None),
            "kvcache_quant_mode": _series_val(result_df, f"{prefix}kvcache", None),
            "fmha_quant_mode": _series_val(result_df, f"{prefix}fmha", None),
            "comm_quant_mode": _series_val(result_df, f"{prefix}comm", None),
        }

        worker_payload: dict[str, Any] = {
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "data_parallel_size": dp,
            "moe_tensor_parallel_size": moe_tp,
            "moe_expert_parallel_size": moe_ep,
            "max_batch_size": bs,
            **{k: v for k, v in quant.items() if v is not None},
        }

        if memory is not None:
            worker_payload["memory"] = memory
        if quant.get("kvcache_quant_mode"):
            worker_payload["kv_cache_dtype"] = quant["kvcache_quant_mode"]

        worker_payload = _deep_merge(worker_payload, extra_overrides)
        return worker_payload, max(workers, 1)

    backend_name = getattr(task_config, "backend_name", None)
    runtime_cfg = task_config.config.runtime_config
    prefix_tokens = _safe_int(_series_val(result_df, "prefix", runtime_cfg.prefix), runtime_cfg.prefix)
    config_obj = task_config.config

    service_cfg = {
        "model_name": task_config.model_name,
        "served_model_name": task_config.model_name,
        "model_path": task_config.model_name,
        "include_frontend": True,
        "prefix": prefix_tokens,
    }
    service_cfg = _deep_merge(service_cfg, overrides.get("ServiceConfig"))
    service_cfg = apply_defaults("ServiceConfig", service_cfg, backend=backend_name)

    model_cfg = {
        "prefix": prefix_tokens,
        "is_moe": getattr(config_obj, "is_moe", None),
        "nextn": getattr(config_obj, "nextn", None),
        "nextn_accept_rates": getattr(config_obj, "nextn_accept_rates", None),
    }
    model_cfg = {k: v for k, v in model_cfg.items() if v is not None}
    model_cfg = _deep_merge(model_cfg, overrides.get("ModelConfig"))
    model_cfg = apply_defaults("ModelConfig", model_cfg, backend=backend_name)

    k8s_cfg = {}
    k8s_cfg = _deep_merge(k8s_cfg, overrides.get("K8sConfig"))
    k8s_cfg = apply_defaults("K8sConfig", k8s_cfg, backend=backend_name)

    dyn_cfg = {
        "mode": task_config.serving_mode,
    }
    dyn_cfg = _deep_merge(dyn_cfg, overrides.get("DynConfig"))
    dyn_cfg = apply_defaults("DynConfig", dyn_cfg, backend=backend_name)

    worker_overrides = overrides.get("Workers", {})
    worker_count_overrides = overrides.get("WorkerCounts") or overrides.get("WorkerConfig") or {}

    if task_config.serving_mode == "agg":
        agg_params, agg_workers = _build_worker_params("", worker_overrides.get("agg"))
        prefill_params, prefill_workers = None, 0
        decode_params, decode_workers = None, 0
    else:
        agg_params, agg_workers = None, 0
        prefill_params, prefill_workers = _build_worker_params("(p)", worker_overrides.get("prefill"))
        decode_params, decode_workers = _build_worker_params("(d)", worker_overrides.get("decode"))

    if agg_params:
        agg_workers = _safe_int(worker_count_overrides.get("agg_workers"), agg_workers)
    if prefill_params:
        prefill_workers = _safe_int(worker_count_overrides.get("prefill_workers"), prefill_workers)
    if decode_params:
        decode_workers = _safe_int(worker_count_overrides.get("decode_workers"), decode_workers)

    sla_cfg = {
        "isl": runtime_cfg.isl,
        "osl": runtime_cfg.osl,
        "ttft": _safe_float(_series_val(result_df, "ttft", runtime_cfg.ttft), runtime_cfg.ttft),
        "tpot": _safe_float(_series_val(result_df, "tpot", runtime_cfg.tpot), runtime_cfg.tpot),
    }
    sla_cfg = _deep_merge(sla_cfg, overrides.get("SlaConfig"))

    params = collect_generator_params(
        service=service_cfg,
        k8s=k8s_cfg,
        prefill_params=prefill_params,
        decode_params=decode_params,
        agg_params=agg_params,
        prefill_workers=prefill_workers if prefill_params else 0,
        decode_workers=decode_workers if decode_params else 0,
        agg_workers=agg_workers if agg_params else 0,
        sla=sla_cfg,
        dyn_config=dyn_cfg,
        backend=backend_name,
    )

    params = _deep_merge(params, overrides.get("Params"))
    params["ModelConfig"] = model_cfg
    return params
