# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from typing import Any

import pandas as pd

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk.task_v2 import Task

from .aggregators import collect_generator_params
from .rendering import apply_defaults


def _deep_merge(target: dict, extra: dict | None) -> dict:
    """
    Recursively merge the contents of the 'extra' dictionary into 'target',
    performing a deep merge for nested dictionaries.

    Args:
        target: The base dictionary to update.
        extra: An optional dictionary whose values will be merged into 'target'.

    Returns:
        The modified 'target' dictionary with merged values from 'extra'.

    Example:
        >>> a = {'a': 1, 'b': {'c': 2}}
        >>> b = {'b': {'d': 3}, 'e': 4}
        >>> _deep_merge(a, b)
        {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
    """
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
    task_config: Task,
    result_df: pd.Series,
    generator_overrides: dict | None = None,
    num_gpus_per_node: int | None = None,
) -> dict:
    """Convert a task config/result row into unified generator parameters.

    Args:
        task_config: The Task that produced the result.
        result_df: A single row (pd.Series) from the best_configs DataFrame.
        generator_overrides: Optional overrides dict from CLI flags.
        num_gpus_per_node: If set, overrides the NodeConfig.num_gpus_per_node
            in the generated config.
    """

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
            "gpus_per_worker": tp * pp * dp,
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

    backend_name = task_config.primary_backend_name
    prefix_tokens = _safe_int(_series_val(result_df, "prefix", task_config.prefix), task_config.prefix)

    # Fetch num_gpus_per_node from system config (unless caller provided an override)
    if num_gpus_per_node is None:
        num_gpus_per_node = 8
        try:
            db = get_database(
                system=task_config.primary_system_name,
                backend=task_config.primary_backend_name,
                version=task_config.primary_backend_version,
            )
            if db and "node" in db.system_spec:
                num_gpus_per_node = db.system_spec["node"].get("num_gpus_per_node", 8)
        except Exception:
            pass

    service_cfg = {
        "model_path": task_config.primary_model_path,
        "served_model_path": task_config.primary_model_path,
        "include_frontend": True,
        "prefix": prefix_tokens,
    }
    service_cfg = _deep_merge(service_cfg, overrides.get("ServiceConfig"))
    service_cfg = apply_defaults("ServiceConfig", service_cfg, backend=backend_name)

    model_cfg = {
        "prefix": prefix_tokens,
        "is_moe": task_config.is_moe,
        "nextn": task_config.nextn,
        "nextn_accept_rates": task_config.nextn_accept_rates if task_config.nextn else None,
    }
    model_cfg = {k: v for k, v in model_cfg.items() if v is not None}
    model_cfg = _deep_merge(model_cfg, overrides.get("ModelConfig"))
    model_cfg = apply_defaults("ModelConfig", model_cfg, backend=backend_name)

    dyn_cfg = {
        "mode": task_config.serving_mode,
    }
    dyn_cfg = _deep_merge(dyn_cfg, overrides.get("DynConfig"))
    dyn_cfg = apply_defaults("DynConfig", dyn_cfg, backend=backend_name)

    generator_dynamo_version = overrides.get("generator_dynamo_version")
    k8s_cfg = {}
    k8s_cfg = _deep_merge(k8s_cfg, overrides.get("K8sConfig"))
    k8s_cfg = apply_defaults(
        "K8sConfig",
        k8s_cfg,
        backend=backend_name,
        extra_context={"generator_dynamo_version": generator_dynamo_version},
    )
    sflow_cfg = {}
    sflow_cfg = _deep_merge(sflow_cfg, overrides.get("SflowConfig"))
    sflow_cfg = apply_defaults("SflowConfig", sflow_cfg, backend=backend_name)

    worker_overrides = overrides.get("Workers", {})
    worker_count_overrides = overrides.get("WorkerCounts") or overrides.get("WorkerConfig") or {}

    if task_config.serving_mode == "agg":
        agg_params, agg_workers = _build_worker_params("", worker_overrides.get("agg"))
        if task_config.total_gpus:
            tp = agg_params.get("tensor_parallel_size", 1)
            pp = agg_params.get("pipeline_parallel_size", 1)
            dp = agg_params.get("data_parallel_size", 1)
            gpus_per_replica = tp * pp * dp
            agg_workers = task_config.total_gpus // gpus_per_replica
        prefill_params, prefill_workers = None, 0
        decode_params, decode_workers = None, 0
    else:
        agg_params, agg_workers = None, 0
        prefill_params, prefill_workers = _build_worker_params("(p)", worker_overrides.get("prefill"))
        decode_params, decode_workers = _build_worker_params("(d)", worker_overrides.get("decode"))

        # Scale disagg workers based on total_gpus (similar to agg mode)
        if task_config.total_gpus and prefill_params and decode_params:
            p_tp = prefill_params.get("tensor_parallel_size", 1)
            p_pp = prefill_params.get("pipeline_parallel_size", 1)
            p_dp = prefill_params.get("data_parallel_size", 1)
            prefill_gpus_per_worker = p_tp * p_pp * p_dp

            d_tp = decode_params.get("tensor_parallel_size", 1)
            d_pp = decode_params.get("pipeline_parallel_size", 1)
            d_dp = decode_params.get("data_parallel_size", 1)
            decode_gpus_per_worker = d_tp * d_pp * d_dp

            # Each replica uses prefill_workers_per_replica prefill workers + decode_workers_per_replica decode workers
            # For simplicity, assume 1:1 prefill:decode ratio per replica
            gpus_per_replica = (prefill_workers * prefill_gpus_per_worker) + (decode_workers * decode_gpus_per_worker)
            if gpus_per_replica > 0:
                replicas = task_config.total_gpus // gpus_per_replica
                prefill_workers = replicas * prefill_workers
                decode_workers = replicas * decode_workers

    if agg_params:
        agg_workers = _safe_int(worker_count_overrides.get("agg_workers"), agg_workers)
    if prefill_params:
        prefill_workers = _safe_int(worker_count_overrides.get("prefill_workers"), prefill_workers)
    if decode_params:
        decode_workers = _safe_int(worker_count_overrides.get("decode_workers"), decode_workers)

    # Multimodal EPD: the encode worker is not produced by the SDK sweep (the
    # encoder is modeled colocated with prefill). Inject it from explicit
    # overrides -- --generator-set Workers.encode.* + WorkerConfig.encode_workers.
    # Absent -> no encode worker, output unchanged.
    encode_override = worker_overrides.get("encode")
    encode_params = None
    encode_workers = 0
    if isinstance(encode_override, dict) and encode_override:
        encode_params = dict(encode_override)
        e_tp = _safe_int(encode_params.get("tensor_parallel_size"), 1)
        e_pp = _safe_int(encode_params.get("pipeline_parallel_size"), 1)
        e_dp = _safe_int(encode_params.get("data_parallel_size"), 1)
        encode_params.setdefault("gpus_per_worker", e_tp * e_pp * e_dp)
        encode_workers = _safe_int(worker_count_overrides.get("encode_workers"), 1)

    sla_cfg = {
        "isl": task_config.isl,
        "osl": task_config.osl,
        "ttft": _safe_float(_series_val(result_df, "ttft", task_config.ttft), task_config.ttft),
        "tpot": _safe_float(_series_val(result_df, "tpot", task_config.tpot), task_config.tpot),
    }
    sla_cfg = _deep_merge(sla_cfg, overrides.get("SlaConfig"))
    bench_cfg = overrides.get("BenchConfig")

    params = collect_generator_params(
        service=service_cfg,
        k8s=k8s_cfg,
        prefill_params=prefill_params,
        decode_params=decode_params,
        agg_params=agg_params,
        prefill_workers=prefill_workers if prefill_params else 0,
        decode_workers=decode_workers if decode_params else 0,
        agg_workers=agg_workers if agg_params else 0,
        num_gpus_per_node=num_gpus_per_node,
        sla=sla_cfg,
        bench=bench_cfg,
        sflow=sflow_cfg,
        dyn_config=dyn_cfg,
        backend=backend_name,
        generator_dynamo_version=generator_dynamo_version,
        encode_params=encode_params,
        encode_workers=encode_workers if encode_params else None,
    )

    params = _deep_merge(params, overrides.get("Params"))
    # Expose SDK's system identifier to templates via NodeConfig.system_name
    params.setdefault("NodeConfig", {})["system_name"] = task_config.system_name
    rule_name = overrides.get("rule")
    if rule_name:
        params["rule"] = rule_name
    if "preserve_engine_limits" in overrides:
        params["preserve_engine_limits"] = bool(overrides["preserve_engine_limits"])
    params["ModelConfig"] = model_cfg
    return params


def task_config_to_request(
    task_config: Task,
    result_df: pd.Series,
    generator_overrides: dict | None = None,
    num_gpus_per_node: int | None = None,
):
    """Convert a task config/result row into a typed ``GeneratorRequest``.

    Built on top of :func:`task_config_to_generator_config` so it stays
    byte-equivalent with the legacy dict path (proven by the request round-trip
    gate). The dict-returning function above is kept unchanged for the dynamo
    profiler and existing callers; this is the typed alternative used when a
    caller wants to go through ``api.generate_from_request``.
    """
    from .request import from_legacy_params

    params = task_config_to_generator_config(task_config, result_df, generator_overrides, num_gpus_per_node)
    # Task exposes the backend via primary_backend_name (matching the dict bridge);
    # fall back to backend_name for duck-typed/legacy callers.
    backend = getattr(task_config, "primary_backend_name", None) or getattr(task_config, "backend_name", None)
    return from_legacy_params(params, backend=backend)
