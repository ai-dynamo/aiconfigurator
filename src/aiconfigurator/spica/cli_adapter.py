# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import random
import traceback
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from aiconfigurator.generator.dynamo_features import normalize_router_mode
from aiconfigurator.sdk import common, pareto_analysis, perf_database

logger = logging.getLogger(__name__)

SPICA_THOROUGH_SWEEP_ROUNDS = 3
SPICA_THOROUGH_PARALLEL_EVALS = 16
SPICA_THOROUGH_SYNTHETIC_CONCURRENCY_PER_GPU = 128
SPICA_THOROUGH_SYNTHETIC_REQUEST_COUNT = 1000


@dataclass
class _SpicaTraceTask:
    primary_model_path: str
    primary_system_name: str
    primary_backend_name: str
    primary_backend_version: str
    total_gpus: int
    serving_mode: str
    ttft: float
    tpot: float
    request_latency: float | None
    is_moe: bool
    isl: int = 0
    osl: int = 0

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.__dict__, sort_keys=False)


@dataclass
class _SpicaTraceResultBundle:
    candidates: list[dict[str, Any]]
    candidate_df: pd.DataFrame
    tasks: dict[str, _SpicaTraceTask]
    trace_path: str | None
    config_path: str | None
    workload_label: str
    chosen_exp: str
    best_configs: dict[str, pd.DataFrame]
    pareto_fronts: dict[str, pd.DataFrame | None]
    best_throughputs: dict[str, float]
    best_latencies: dict[str, dict[str, float]]
    pareto_x_axis: dict[str, str]


def _positive_int_env(name: str, default: int, *, aliases: tuple[str, ...] = ()) -> int:
    selected_name = name
    raw_value = os.environ.get(name)
    if raw_value is None:
        for alias in aliases:
            raw_value = os.environ.get(alias)
            if raw_value is not None:
                selected_name = alias
                break
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Ignoring %s=%r; expected a positive integer.", selected_name, raw_value)
        return default
    if value < 1:
        logger.warning("Ignoring %s=%r; expected a positive integer.", selected_name, raw_value)
        return default
    return value


def _spica_thorough_sweep_config() -> dict[str, int]:
    sweep_config = {
        "max_rounds": _positive_int_env(
            "AIC_SPICA_THOROUGH_SWEEP_ROUNDS",
            SPICA_THOROUGH_SWEEP_ROUNDS,
            aliases=("AIC_SPICA_TRACE_SWEEP_ROUNDS",),
        ),
        "parallel_evals": _positive_int_env(
            "AIC_SPICA_THOROUGH_PARALLEL_EVALS",
            SPICA_THOROUGH_PARALLEL_EVALS,
            aliases=("AIC_SPICA_TRACE_PARALLEL_EVALS",),
        ),
    }
    candidates_per_round = os.environ.get("AIC_SPICA_THOROUGH_CANDIDATES_PER_ROUND")
    candidates_env = "AIC_SPICA_THOROUGH_CANDIDATES_PER_ROUND"
    if candidates_per_round is None:
        candidates_per_round = os.environ.get("AIC_SPICA_TRACE_CANDIDATES_PER_ROUND")
        candidates_env = "AIC_SPICA_TRACE_CANDIDATES_PER_ROUND"
    if candidates_per_round is not None:
        sweep_config["candidates_per_round"] = _positive_int_env(
            candidates_env,
            sweep_config["parallel_evals"],
        )
    return sweep_config


def _spica_candidate_to_dict(candidate: Any) -> dict[str, Any]:
    if hasattr(candidate, "model_dump"):
        return candidate.model_dump()
    if isinstance(candidate, dict):
        return dict(candidate)
    return {
        "config": getattr(candidate, "config", {}),
        "used_gpus": getattr(candidate, "used_gpus", None),
        "score": getattr(candidate, "score", None),
        "metrics": getattr(candidate, "metrics", {}),
    }


def _metric_value(metrics: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics and metrics[name] is not None:
            return metrics[name]
    return None


def _spica_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
        if pd.isna(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _spica_int(value: Any) -> int | None:
    number = _spica_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return None


def _spica_deployment_mode(payload: dict[str, Any]) -> str:
    config = payload.get("config") or {}
    mode = config.get("deployment_mode")
    return str(mode) if mode else "unknown"


def _spica_parallel_key(config: dict[str, Any], prefix: str | None = None) -> str:
    key_prefix = f"{prefix}_" if prefix else ""
    tp = _spica_int(config.get(f"{key_prefix}tp")) or 1
    pp = _spica_int(config.get(f"{key_prefix}pp")) or 1
    dp = _spica_int(config.get(f"{key_prefix}attention_dp")) or 1
    moe_tp = _spica_int(config.get(f"{key_prefix}moe_tp")) or 1
    moe_ep = _spica_int(config.get(f"{key_prefix}moe_ep")) or 1
    parts = [f"tp{tp}", f"pp{pp}"]
    if dp != 1:
        parts.append(f"dp{dp}")
    if moe_tp != 1 or moe_ep != 1:
        parts.append(f"etp{moe_tp}ep{moe_ep}")
    return "_".join(parts)


def _spica_batch_label(config: dict[str, Any], prefix: str) -> str:
    tokens = config.get(f"{prefix}_max_num_batched_tokens")
    seqs = config.get(f"{prefix}_max_num_seqs")
    if tokens is None and seqs is None:
        return "n/a"
    return f"{tokens or 'n/a'}/{seqs or 'n/a'}"


def _spica_planner(config: dict[str, Any]) -> str:
    if config.get("enable_throughput_scaling") or config.get("enable_load_scaling"):
        return str(config.get("planner_scaling_policy", "enabled"))
    return "static"


def _spica_router(config: dict[str, Any]) -> str:
    return str(config.get("router_mode") or "n/a")


def _spica_metric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    return {
        "score": payload.get("score"),
        "goodput": _metric_value(metrics, "goodput_output_throughput_tok_s"),
        "throughput": _metric_value(metrics, "output_throughput_tok_s"),
        "ttft": _metric_value(metrics, "mean_ttft_ms", "ttft_ms"),
        "tpot": _metric_value(metrics, "mean_tpot_ms", "tpot_ms", "itl_ms"),
        "request_latency": _metric_value(metrics, "mean_e2e_latency_ms", "e2e_latency_ms", "request_latency_ms"),
    }


def _spica_tokens_per_user(metrics: dict[str, Any], metric_summary: dict[str, Any]) -> float | None:
    tokens_per_user = _spica_float(
        _metric_value(
            metrics,
            "mean_output_token_throughput_per_user",
            "output_token_throughput_per_user",
            "tokens_per_second_per_user",
            "tokens/s/user",
        )
    )
    if tokens_per_user is not None:
        return tokens_per_user

    # Legacy Pareto plots use generation speed on the x-axis; mean TPOT is its inverse.
    tpot_ms = _spica_float(metric_summary.get("tpot"))
    if tpot_ms is None or tpot_ms <= 0:
        return None
    return 1000.0 / tpot_ms


def _spica_load_shape(metrics: dict[str, Any], metric_summary: dict[str, Any]) -> tuple[float, float]:
    request_rate = _spica_float(
        _metric_value(metrics, "request_rate", "request_throughput", "request_throughput_rps", "completed_req_s")
    )
    concurrency = _spica_float(_metric_value(metrics, "concurrency", "mean_concurrency", "active_requests"))

    if concurrency is None:
        throughput = _spica_float(metric_summary.get("throughput")) or _spica_float(metric_summary.get("goodput"))
        tokens_per_user = _spica_tokens_per_user(metrics, metric_summary)
        if throughput is not None and tokens_per_user is not None and tokens_per_user > 0:
            concurrency = throughput / tokens_per_user

    if request_rate is None and concurrency is not None:
        latency_ms = _spica_float(metric_summary.get("request_latency"))
        if latency_ms is not None and latency_ms > 0:
            request_rate = concurrency / (latency_ms / 1000.0)

    return request_rate or 0.0, concurrency or 0.0


def _spica_candidates_to_result_df(candidates: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        payload = _spica_candidate_to_dict(candidate)
        config = payload.get("config") or {}
        raw_metrics = payload.get("metrics") or {}
        metrics = _spica_metric_summary(payload)
        used_gpus = _spica_int(payload.get("used_gpus")) or _spica_int(config.get("used_gpus"))
        goodput = _spica_float(metrics["goodput"])
        score = _spica_float(metrics["score"])
        if score is None and goodput is not None and used_gpus and used_gpus > 0:
            score = goodput / used_gpus
        score = score or 0.0
        tokens_per_user = _spica_tokens_per_user(raw_metrics, metrics) or 0.0
        request_rate, concurrency = _spica_load_shape(raw_metrics, metrics)
        mode = _spica_deployment_mode(payload)

        row = {
            "spica_candidate_id": candidate_index,
            "deployment_mode": mode,
            "backend": config.get("backend", "n/a"),
            "tokens/s/user": tokens_per_user,
            "tokens/s/gpu": score,
            "tokens/s/gpu_cluster": score,
            "goodput/s/gpu": score,
            "goodput": goodput,
            "throughput": _spica_float(metrics["throughput"]),
            "ttft": _spica_float(metrics["ttft"]),
            "tpot": _spica_float(metrics["tpot"]),
            "request_latency": _spica_float(metrics["request_latency"]),
            "request_rate": request_rate,
            "concurrency": round(concurrency, 2),
            "num_total_gpus": used_gpus,
            "total_gpus": used_gpus,
            "router": _spica_router(config),
            "planner": _spica_planner(config),
        }
        if mode == "disagg":
            row.update(
                {
                    "(p)workers": _spica_int(config.get("prefill_replicas")) or 1,
                    "(p)tp": _spica_int(config.get("prefill_tp")) or 1,
                    "(p)pp": _spica_int(config.get("prefill_pp")) or 1,
                    "(p)dp": _spica_int(config.get("prefill_attention_dp")) or 1,
                    "(p)moe_tp": _spica_int(config.get("prefill_moe_tp")) or 1,
                    "(p)moe_ep": _spica_int(config.get("prefill_moe_ep")) or 1,
                    "(p)bs": _spica_batch_label(config, "prefill"),
                    "(p)parallel": _spica_parallel_key(config, "prefill"),
                    "(d)workers": _spica_int(config.get("decode_replicas")) or 1,
                    "(d)tp": _spica_int(config.get("decode_tp")) or 1,
                    "(d)pp": _spica_int(config.get("decode_pp")) or 1,
                    "(d)dp": _spica_int(config.get("decode_attention_dp")) or 1,
                    "(d)moe_tp": _spica_int(config.get("decode_moe_tp")) or 1,
                    "(d)moe_ep": _spica_int(config.get("decode_moe_ep")) or 1,
                    "(d)bs": _spica_batch_label(config, "decode"),
                    "(d)parallel": _spica_parallel_key(config, "decode"),
                }
            )
        else:
            row.update(
                {
                    "tp": _spica_int(config.get("tp")) or 1,
                    "pp": _spica_int(config.get("pp")) or 1,
                    "dp": _spica_int(config.get("attention_dp")) or 1,
                    "moe_tp": _spica_int(config.get("moe_tp")) or 1,
                    "moe_ep": _spica_int(config.get("moe_ep")) or 1,
                    "bs": _spica_batch_label(config, "agg"),
                    "parallel": _spica_parallel_key(config),
                }
            )
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row.setdefault(key, value)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    metric_cols = [
        "tokens/s/user",
        "tokens/s/gpu",
        "tokens/s/gpu_cluster",
        "goodput/s/gpu",
        "goodput",
        "throughput",
        "ttft",
        "tpot",
        "request_latency",
        "request_rate",
        "concurrency",
        "num_total_gpus",
        "total_gpus",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _spica_trace_is_moe(candidate_df: pd.DataFrame) -> bool:
    moe_cols = [col for col in candidate_df.columns if col.endswith("moe_tp") or col.endswith("moe_ep")]
    return any((pd.to_numeric(candidate_df[col], errors="coerce").fillna(1) > 1).any() for col in moe_cols)


_SPICA_INTEGER_RESULT_COLUMNS = (
    "spica_candidate_id",
    "num_total_gpus",
    "total_gpus",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "replicas",
    "(p)workers",
    "(p)tp",
    "(p)pp",
    "(p)dp",
    "(p)moe_tp",
    "(p)moe_ep",
    "(d)workers",
    "(d)tp",
    "(d)pp",
    "(d)dp",
    "(d)moe_tp",
    "(d)moe_ep",
)


def _normalize_spica_result_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in _SPICA_INTEGER_RESULT_COLUMNS:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():
            df[col] = numeric.round().astype(int)
    return df


def _spica_search_space_value(config: Any, name: str, default: Any = None) -> Any:
    search_space = getattr(config, "search_space", None)
    return getattr(search_space, name, default)


def _spica_workload_value(config: Any, name: str, default: Any = None) -> Any:
    workload = getattr(config, "workload", None)
    return getattr(workload, name, default)


def _spica_sla_value(config: Any, name: str, default: Any = None) -> Any:
    goal = getattr(config, "goal", None)
    sla = getattr(goal, "sla", None)
    return getattr(sla, name, default)


def _spica_scalar_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            scalar = _spica_scalar_value(item)
            if scalar is not None:
                return scalar
        return None
    if isinstance(value, set):
        for item in sorted(value, key=str):
            scalar = _spica_scalar_value(item)
            if scalar is not None:
                return scalar
        return None
    if isinstance(value, dict):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _spica_first_scalar(*values: Any, default: Any = None) -> Any:
    for value in values:
        scalar = _spica_scalar_value(value)
        if scalar is not None:
            return scalar
    return default


def _spica_pinned_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, dict)):
        return None
    return _spica_scalar_value(value)


def _spica_trace_task(args, mode: str, candidate_df: pd.DataFrame, config: Any | None = None) -> _SpicaTraceTask:
    first_row = candidate_df.iloc[0] if not candidate_df.empty else pd.Series(dtype=object)
    backend = getattr(args, "backend", None) or "spica"
    backend_series = candidate_df["backend"] if "backend" in candidate_df else pd.Series(dtype=object)
    mode_backends = sorted(str(backend_name) for backend_name in backend_series.dropna().unique())
    if mode_backends and (backend == "auto" or config is not None):
        backend = ",".join(mode_backends)

    model_path = _spica_first_scalar(
        first_row.get("model_name"),
        _spica_search_space_value(config, "model_name"),
        getattr(args, "model_path", None),
        default="unknown",
    )
    system = _spica_first_scalar(
        first_row.get("hardware_sku"),
        _spica_search_space_value(config, "hardware_sku"),
        getattr(args, "system", None),
        default="unknown",
    )
    total_gpus = _spica_first_scalar(
        _spica_pinned_scalar(_spica_search_space_value(config, "gpu_budget")),
        first_row.get("gpu_budget"),
        first_row.get("total_gpus"),
        getattr(args, "total_gpus", None),
        default=0,
    )
    ttft = _spica_sla_value(config, "ttft_ms", getattr(args, "ttft", 0.0) or 0.0) or 0.0
    tpot = _spica_sla_value(config, "itl_ms", getattr(args, "tpot", 0.0) or 0.0) or 0.0
    request_latency = _spica_sla_value(config, "e2e_ms", None)
    workload_isl = _spica_first_scalar(_spica_workload_value(config, "isl"), getattr(args, "isl", None), default=0)
    workload_osl = _spica_first_scalar(_spica_workload_value(config, "osl"), getattr(args, "osl", None), default=0)

    return _SpicaTraceTask(
        primary_model_path=str(model_path),
        primary_system_name=str(system),
        primary_backend_name=backend,
        primary_backend_version="spica",
        total_gpus=int(total_gpus),
        serving_mode=mode,
        ttft=float(ttft),
        tpot=float(tpot),
        request_latency=request_latency,
        is_moe=_spica_trace_is_moe(candidate_df),
        isl=int(workload_isl),
        osl=int(workload_osl),
    )


def _spica_trace_path_from_config(config: Any | None) -> str | None:
    trace_path = _spica_workload_value(config, "trace_path", None)
    return str(trace_path) if trace_path else None


def _spica_workload_label(config: Any | None, args, config_path: str | None) -> str:
    if _spica_trace_path_from_config(config):
        return "trace"
    if config_path:
        return "config"
    return "synthetic"


def _build_spica_trace_result_bundle(
    candidates: list[Any],
    args,
    config: Any | None = None,
    config_path: str | None = None,
) -> _SpicaTraceResultBundle:
    from aiconfigurator.cli.utils import process_experiment_result

    payloads = [_spica_candidate_to_dict(candidate) for candidate in candidates]
    candidate_df = _spica_candidates_to_result_df(payloads)
    tasks: dict[str, _SpicaTraceTask] = {}
    best_configs: dict[str, pd.DataFrame] = {}
    best_throughputs: dict[str, float] = {}
    best_latencies: dict[str, dict[str, float]] = {}
    pareto_fronts: dict[str, pd.DataFrame | None] = {}
    pareto_x_axis: dict[str, str] = {}

    for mode, mode_df in candidate_df.groupby("deployment_mode", sort=False):
        mode = str(mode)
        mode_df = mode_df.dropna(axis=1, how="all").copy()
        mode_df = _normalize_spica_result_dtypes(mode_df)
        task = _spica_trace_task(args, mode, mode_df, config)
        tasks[mode] = task
        best_config_df, best_throughput, pareto_frontier_df, x_axis_col, latencies = process_experiment_result(
            task,
            {"pareto_df": mode_df},
            top_n=args.top_n,
            strict_sla=getattr(args, "strict_sla", False),
        )
        best_configs[mode] = best_config_df
        best_throughputs[mode] = best_throughput
        best_latencies[mode] = latencies
        pareto_fronts[mode] = pareto_frontier_df
        pareto_x_axis[mode] = x_axis_col

    chosen_exp = max(best_throughputs, key=best_throughputs.get) if best_throughputs else "none"
    return _SpicaTraceResultBundle(
        candidates=payloads,
        candidate_df=candidate_df,
        tasks=tasks,
        trace_path=_spica_trace_path_from_config(config),
        config_path=config_path,
        workload_label=_spica_workload_label(config, args, config_path),
        chosen_exp=chosen_exp,
        best_configs=best_configs,
        pareto_fronts=pareto_fronts,
        best_throughputs=best_throughputs,
        best_latencies=best_latencies,
        pareto_x_axis=pareto_x_axis,
    )


def _spica_safe_path_component(value: Any, *, strip_extension: bool = False) -> str:
    text = str(value or "unknown")
    if os.path.exists(text):
        text = os.path.basename(os.path.abspath(text))
    if strip_extension:
        stem, ext = os.path.splitext(text)
        if ext:
            text = stem
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in text)
    safe = safe.strip("._-")
    return safe or "unknown"


def _spica_trace_result_dir(result_bundle: _SpicaTraceResultBundle, save_dir: str) -> str:
    first_task = next(iter(result_bundle.tasks.values()), None)
    model = _spica_safe_path_component(getattr(first_task, "primary_model_path", "model"))
    system = _spica_safe_path_component(getattr(first_task, "primary_system_name", "system"))
    backend = _spica_safe_path_component(getattr(first_task, "primary_backend_name", "backend"))
    ttft = int(getattr(first_task, "ttft", 0) or 0)
    tpot = int(getattr(first_task, "tpot", 0) or 0)
    if result_bundle.trace_path:
        trace_name = _spica_safe_path_component(os.path.basename(result_bundle.trace_path), strip_extension=True)
        workload = f"trace_{trace_name}"
    elif result_bundle.config_path:
        workload = "thorough_" + _spica_safe_path_component(
            os.path.basename(result_bundle.config_path),
            strip_extension=True,
        )
    else:
        isl = int(getattr(first_task, "isl", 0) or 0)
        osl = int(getattr(first_task, "osl", 0) or 0)
        workload = f"thorough_isl{isl}_osl{osl}"
    result_prefix = f"{model}_{system}_{backend}_{workload}_ttft{ttft}_tpot{tpot}"
    return os.path.join(save_dir, f"{result_prefix}_{random.randint(0, 1000000)}")


def _spica_first_int(*values: Any, default: int = 1) -> int:
    for value in values:
        number = _spica_int(value)
        if number is not None:
            return number
    return default


def _spica_yaml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _spica_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_spica_yaml_safe(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _spica_value_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _spica_bool(value: Any) -> bool | None:
    if not _spica_value_present(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _spica_deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {key: _spica_deep_merge(value, {}) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _spica_deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _spica_set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if _spica_value_present(value):
        target[key] = _spica_yaml_safe(value)


def _spica_role_worker_overrides(
    row: pd.Series,
    role: str,
    args: argparse.Namespace | None,
) -> dict[str, Any]:
    prefix = "agg" if role == "agg" else role
    worker: dict[str, Any] = {}
    engine_overrides: dict[str, Any] = {}

    max_tokens = _spica_int(row.get(f"{prefix}_max_num_batched_tokens"))
    if max_tokens is not None:
        worker["max_num_tokens"] = max_tokens
        worker["max_prefill_tokens"] = max_tokens
        engine_overrides["max_num_tokens"] = max_tokens

    _spica_set_if_present(worker, "tokens_per_block", _spica_int(row.get(f"{prefix}_block_size")))
    _spica_set_if_present(
        worker,
        "kv_cache_free_gpu_memory_fraction",
        _spica_float(row.get(f"{prefix}_gpu_memory_utilization")),
    )

    prefix_caching = _spica_bool(row.get(f"{prefix}_enable_prefix_caching"))
    if prefix_caching is not None:
        worker["disable_prefix_cache"] = not prefix_caching

    attention_dp_key = "attention_dp" if role == "agg" else f"{prefix}_attention_dp"
    attention_dp = _spica_int(row.get(attention_dp_key))
    if attention_dp is not None and attention_dp > 1:
        worker["enable_attention_dp"] = True
        engine_overrides["enable_attention_dp"] = True

    max_seq_len = _spica_first_int(row.get("context_length"), getattr(args, "max_seq_len", None), default=0)
    if max_seq_len > 0:
        worker["max_seq_len"] = max_seq_len
        engine_overrides["max_seq_len"] = max_seq_len

    transfer_buffer_tokens = max(max_tokens or 0, max_seq_len)
    if role != "agg" and transfer_buffer_tokens > 0:
        worker["cache_transceiver_max_tokens_in_buffer"] = transfer_buffer_tokens
        engine_overrides["cache_transceiver_config"] = {
            "backend": "DEFAULT",
            "max_tokens_in_buffer": transfer_buffer_tokens,
        }

    nextn = _spica_int(row.get("aic_nextn"))
    if nextn is not None and nextn > 0:
        worker["speculative_decoding_type"] = "MTP"
        worker["num_nextn_predict_layers"] = nextn
        engine_overrides["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": nextn,
        }

    if engine_overrides:
        worker["extra_engine_args"] = engine_overrides

    return worker


def _spica_router_enabled(row: pd.Series) -> bool | None:
    router_mode = row.get("router_mode")
    if not _spica_value_present(router_mode):
        return None
    normalized = str(router_mode).strip().lower()
    return normalized not in {"", "n/a", "none", "disabled", "round_robin", "round-robin"}


def _spica_router_mode(row: pd.Series) -> str | None:
    if not _spica_value_present(row.get("router_mode")):
        return None
    return normalize_router_mode(str(row.get("router_mode")))


def _spica_router_block_size(row: pd.Series) -> int | None:
    mode = str(row.get("deployment_mode") or "")
    if mode == "agg":
        return _spica_int(row.get("agg_block_size"))
    return _spica_first_int(row.get("decode_block_size"), row.get("prefill_block_size"), default=0) or None


def _spica_router_config(row: pd.Series) -> dict[str, Any] | None:
    if _spica_router_enabled(row) is not True:
        return None

    router: dict[str, Any] = {}
    _spica_set_if_present(router, "kv_cache_block_size", _spica_router_block_size(row))
    for key in (
        "overlap_score_credit",
        "prefill_load_scale",
        "router_temperature",
    ):
        _spica_set_if_present(router, key, row.get(key))

    no_admission_control = _spica_bool(row.get("no_admission_control"))
    if no_admission_control is True:
        router["admission_control"] = "none"
    else:
        threshold_keys = (
            "active_decode_blocks_threshold",
            "active_prefill_tokens_threshold",
            "active_prefill_tokens_threshold_frac",
        )
        for key in threshold_keys:
            _spica_set_if_present(router, key, row.get(key))
        if any(_spica_value_present(row.get(key)) for key in threshold_keys):
            router["admission_control"] = "token-capacity"

    return router or None


def _spica_engine_num_gpu(row: pd.Series, role: str) -> int:
    if role == "agg":
        return (
            _spica_first_int(row.get("tp"), default=1)
            * _spica_first_int(row.get("pp"), default=1)
            * _spica_first_int(row.get("attention_dp"), default=1)
        )
    return (
        _spica_first_int(row.get(f"{role}_tp"), default=1)
        * _spica_first_int(row.get(f"{role}_pp"), default=1)
        * _spica_first_int(row.get(f"{role}_attention_dp"), default=1)
    )


def _spica_planner_config(
    args: argparse.Namespace | None,
    row: pd.Series,
    task: _SpicaTraceTask | argparse.Namespace | None = None,
) -> dict[str, Any] | None:
    enable_throughput = bool(_spica_bool(row.get("enable_throughput_scaling")))
    enable_load = bool(_spica_bool(row.get("enable_load_scaling")))
    if not (enable_throughput or enable_load):
        return None

    backend = str(
        row.get("backend") or getattr(args, "backend", None) or getattr(args, "primary_backend_name", None) or "trtllm"
    )
    mode = str(row.get("deployment_mode") or "disagg")
    model_name = row.get("model_name")
    if not _spica_value_present(model_name) and task is not None:
        model_name = getattr(task, "primary_model_path", None)
    if not _spica_value_present(model_name) and args is not None:
        model_name = getattr(args, "model_path", None) or getattr(args, "primary_model_path", None)
    planner: dict[str, Any] = {
        "environment": "kubernetes",
        "backend": backend,
        "mode": mode,
        "optimization_target": "sla",
        "enable_throughput_scaling": enable_throughput,
        "enable_load_scaling": enable_load,
    }
    if model_name or _spica_value_present(row.get("model_name")):
        planner["model_name"] = model_name or row.get("model_name")

    for key in (
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
        "max_gpu_budget",
        "min_gpu_budget",
        "min_endpoint",
    ):
        _spica_set_if_present(planner, key, row.get(key))

    gpu_budget = _spica_int(row.get("gpu_budget"))
    if gpu_budget is not None and "max_gpu_budget" not in planner:
        planner["max_gpu_budget"] = gpu_budget

    if mode == "agg":
        planner["decode_engine_num_gpu"] = _spica_engine_num_gpu(row, "agg")
    else:
        planner["prefill_engine_num_gpu"] = _spica_engine_num_gpu(row, "prefill")
        planner["decode_engine_num_gpu"] = _spica_engine_num_gpu(row, "decode")

    ttft = getattr(task, "ttft", None) if task is not None else None
    tpot = getattr(task, "tpot", None) if task is not None else None
    if not _spica_value_present(ttft) and args is not None:
        ttft = getattr(args, "ttft", None)
    if not _spica_value_present(tpot) and args is not None:
        tpot = getattr(args, "tpot", None)
    _spica_set_if_present(planner, "ttft_ms", ttft)
    _spica_set_if_present(planner, "itl_ms", tpot)

    # Spica replay suppresses large periodic planner reports; keep live deploy
    # artifacts similarly quiet unless the user overrides this later.
    planner.setdefault("report_interval_hours", None)
    planner.setdefault("live_dashboard_port", 0)
    return planner


def _spica_kvbm_config(row: pd.Series) -> dict[str, Any] | None:
    kvbm: dict[str, Any] = {}
    num_g2_blocks = _spica_int(row.get("num_g2_blocks"))
    if num_g2_blocks is not None and num_g2_blocks > 0:
        kvbm["cpu_cache_override_num_blocks"] = num_g2_blocks
    offload_batch_size = _spica_int(row.get("offload_batch_size"))
    if offload_batch_size is not None and offload_batch_size > 0:
        kvbm["max_transfer_batch_size"] = offload_batch_size
    return kvbm or None


def _spica_generator_overrides(
    args: argparse.Namespace | None,
    row: pd.Series,
    task: _SpicaTraceTask | argparse.Namespace | None = None,
) -> dict[str, Any]:
    from aiconfigurator.generator.api import load_generator_overrides_from_args

    mode = str(row.get("deployment_mode") or "")
    roles = ["agg"] if mode == "agg" else ["prefill", "decode"]
    spica_overrides: dict[str, Any] = {"Workers": {}}

    for role in roles:
        worker = _spica_role_worker_overrides(row, role, args)
        if worker:
            spica_overrides["Workers"][role] = worker
    if not spica_overrides["Workers"]:
        spica_overrides.pop("Workers")

    dyn_overrides: dict[str, Any] = {}
    router_enabled = _spica_router_enabled(row)
    if router_enabled is not None:
        dyn_overrides["enable_router"] = router_enabled
        router_mode = _spica_router_mode(row)
        if router_mode is not None:
            dyn_overrides["router_mode"] = router_mode
        router_config = _spica_router_config(row)
        if router_config:
            dyn_overrides["router_config"] = router_config
    planner_config = _spica_planner_config(args, row, task)
    if planner_config:
        dyn_overrides["planner_config"] = planner_config
    kvbm_config = _spica_kvbm_config(row)
    if kvbm_config:
        dyn_overrides["kvbm_config"] = kvbm_config
    if dyn_overrides:
        spica_overrides["DynConfig"] = dyn_overrides

    nextn = _spica_int(row.get("aic_nextn"))
    if nextn is not None and nextn > 0:
        spica_overrides["ModelConfig"] = {"nextn": nextn}

    user_overrides = load_generator_overrides_from_args(args) if args is not None else {}
    return _spica_deep_merge(spica_overrides, user_overrides)


def _spica_generator_result_row(row: pd.Series) -> pd.Series:
    generator_row = row.copy()
    mode = str(generator_row.get("deployment_mode") or "")
    if mode == "agg":
        generator_row["workers"] = _spica_first_int(generator_row.get("replicas"), generator_row.get("workers"))
        generator_row["bs"] = _spica_first_int(
            generator_row.get("agg_max_num_seqs"),
            generator_row.get("max_num_seqs"),
            generator_row.get("bs"),
        )
    elif mode == "disagg":
        generator_row["(p)bs"] = _spica_first_int(
            generator_row.get("prefill_max_num_seqs"),
            generator_row.get("(p)bs"),
        )
        generator_row["(d)bs"] = _spica_first_int(
            generator_row.get("decode_max_num_seqs"),
            generator_row.get("(d)bs"),
        )
    return generator_row


def _spica_generator_task(task: _SpicaTraceTask, row: pd.Series) -> argparse.Namespace:
    nextn = _spica_int(row.get("aic_nextn")) or 0
    return argparse.Namespace(
        primary_backend_name=str(row.get("backend") or task.primary_backend_name),
        primary_system_name=task.primary_system_name,
        primary_backend_version=task.primary_backend_version,
        primary_model_path=task.primary_model_path,
        prefix=0,
        is_moe=task.is_moe,
        nextn=nextn,
        nextn_accept_rates=[0.85, 0.8, 0.6, 0.0, 0.0],
        serving_mode=str(row.get("deployment_mode") or task.serving_mode),
        total_gpus=_spica_first_int(row.get("total_gpus"), row.get("num_total_gpus"), task.total_gpus),
        system_name=task.primary_system_name,
        isl=task.isl,
        osl=task.osl,
        ttft=task.ttft,
        tpot=task.tpot,
    )


def _spica_num_gpus_per_node(system_name: str) -> int:
    try:
        spec = perf_database.load_system_spec(system_name)
        return int(((spec or {}).get("node") or {}).get("num_gpus_per_node") or 8)
    except Exception:
        return 8


def _spica_generated_backend_version(args: argparse.Namespace | None, backend_name: str) -> str | None:
    from aiconfigurator.generator.api import (
        get_default_dynamo_version_mapping,
        load_generator_overrides_from_args,
        resolve_backend_version_for_dynamo,
    )

    if args is None:
        return None
    if generated_config_version := getattr(args, "generated_config_version", None):
        return generated_config_version
    generator_overrides = load_generator_overrides_from_args(args)
    if dynamo_version := generator_overrides.get("generator_dynamo_version"):
        try:
            return resolve_backend_version_for_dynamo(dynamo_version, backend_name)
        except ValueError:
            logger.exception("Failed to resolve backend version for generator_dynamo_version=%s.", dynamo_version)
            return None
    default_dynamo_version, default_backend_versions = get_default_dynamo_version_mapping()
    backend_version = default_backend_versions.get(backend_name)
    if backend_version is None:
        logger.warning(
            "No default backend version mapping for backend '%s' in dynamo '%s'; using generator defaults.",
            backend_name,
            default_dynamo_version,
        )
    return backend_version


def _spica_generated_artifact_paths(top_config_dir: str, known_paths: set[str]) -> list[str]:
    paths: list[str] = []
    for root, _, filenames in os.walk(top_config_dir):
        for filename in filenames:
            path = os.path.join(root, filename)
            if path not in known_paths:
                paths.append(path)
    return sorted(paths)


def _generate_spica_backend_artifacts(
    args: argparse.Namespace | None,
    generator_config: dict[str, Any],
    generator_task: argparse.Namespace,
    top_config_dir: str,
    written_paths: list[str],
) -> None:
    from aiconfigurator.generator.api import generate_from_request
    from aiconfigurator.generator.request import from_legacy_params

    try:
        backend_version = _spica_generated_backend_version(args, generator_task.primary_backend_name)
        deployment_target = getattr(args, "deployment_target", "dynamo-j2") if args is not None else "dynamo-j2"
        req = from_legacy_params(generator_config, backend=generator_task.primary_backend_name)
        req = dataclasses.replace(
            req,
            backend=dataclasses.replace(req.backend, generated_config_version=backend_version),
            emit=dataclasses.replace(req.emit, deployment_target=deployment_target, output_dir=top_config_dir),
        )
        known_paths = set(written_paths)
        generate_from_request(req)
        written_paths.extend(_spica_generated_artifact_paths(top_config_dir, known_paths))
    except Exception as exc:
        logger.warning(
            "Failed to generate backend config from Spica trace result: %s, %s",
            exc,
            traceback.format_exc(),
        )


def _save_spica_top_config_artifacts(
    result_bundle: _SpicaTraceResultBundle,
    mode: str,
    mode_dir: str,
    best_config_df: pd.DataFrame,
    args: argparse.Namespace | None = None,
) -> list[str]:
    from aiconfigurator.generator.module_bridge import task_config_to_generator_config

    task = result_bundle.tasks.get(mode)
    if task is None or best_config_df.empty:
        return []

    written_paths: list[str] = []
    num_gpus_per_node = _spica_num_gpus_per_node(task.primary_system_name)
    for rank, (_, row) in enumerate(best_config_df.iterrows(), start=1):
        top_config_dir = os.path.join(mode_dir, f"top{rank}")
        os.makedirs(top_config_dir, exist_ok=True)

        generator_row = _spica_generator_result_row(row)
        generator_task = _spica_generator_task(task, generator_row)
        override_args = args if args is not None else task
        generator_config = task_config_to_generator_config(
            task_config=generator_task,
            result_df=generator_row,
            generator_overrides=_spica_generator_overrides(override_args, generator_row, generator_task),
            num_gpus_per_node=num_gpus_per_node,
        )
        generator_config_yaml = os.path.join(top_config_dir, "generator_config.yaml")
        with open(generator_config_yaml, "w", encoding="utf-8") as fh:
            yaml.safe_dump(_spica_yaml_safe(generator_config), fh, sort_keys=False)
        written_paths.append(generator_config_yaml)
        _generate_spica_backend_artifacts(args, generator_config, generator_task, top_config_dir, written_paths)

        candidate_id = _spica_int(row.get("spica_candidate_id"))
        if candidate_id is not None and 0 <= candidate_id < len(result_bundle.candidates):
            candidate_payload = result_bundle.candidates[candidate_id]
        else:
            candidate_payload = {"row": row.to_dict()}
        candidate_yaml = os.path.join(top_config_dir, "spica_candidate.yaml")
        with open(candidate_yaml, "w", encoding="utf-8") as fh:
            yaml.safe_dump(_spica_yaml_safe(candidate_payload), fh, sort_keys=False)
        written_paths.append(candidate_yaml)

    return written_paths


def _save_spica_pareto_plot(result_bundle: _SpicaTraceResultBundle, save_dir: str) -> str | None:
    pareto_fronts = {name: df for name, df in result_bundle.pareto_fronts.items() if df is not None and not df.empty}
    if not pareto_fronts:
        return None

    path = os.path.join(save_dir, "pareto_frontier.png")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]

    preferred_modes = [mode for mode in ("agg", "disagg") if mode in pareto_fronts]
    other_modes = sorted(mode for mode in pareto_fronts if mode not in {"agg", "disagg"})
    for idx, mode in enumerate([*preferred_modes, *other_modes]):
        df = pareto_fronts[mode]
        pareto_analysis.draw_pareto(
            df,
            "tokens/s/user",
            "tokens/s/gpu",
            ax,
            colors[idx % len(colors)],
            mode,
        )

    combined = pd.concat(pareto_fronts.values(), ignore_index=True)
    if not combined.empty:
        best = combined.sort_values(by="tokens/s/gpu_cluster", ascending=False).head(1)
        ax.scatter(best["tokens/s/user"], best["tokens/s/gpu"], color="black", marker="x", label="best")

    ax.set_title("Spica Pareto Frontier")
    ax.set_xlabel("tokens/s/user")
    ax.set_ylabel("tokens/s/gpu")
    ax.legend()
    plt.savefig(path)
    plt.close(fig)
    return path


def _save_spica_trace_artifacts(
    result_bundle: _SpicaTraceResultBundle,
    save_dir: str,
    args: argparse.Namespace | None = None,
) -> list[str]:
    result_dir = _spica_trace_result_dir(result_bundle, save_dir)
    os.makedirs(result_dir, exist_ok=True)
    written_paths: list[str] = []

    candidates_yaml = os.path.join(result_dir, "spica_candidates.yaml")
    with open(candidates_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump(result_bundle.candidates, fh, sort_keys=False)
    written_paths.append(candidates_yaml)

    if not result_bundle.candidate_df.empty:
        candidates_csv = os.path.join(result_dir, "spica_candidates.csv")
        result_bundle.candidate_df.to_csv(candidates_csv, index=False)
        written_paths.append(candidates_csv)

    nonempty_fronts = [df for df in result_bundle.pareto_fronts.values() if df is not None and not df.empty]
    if nonempty_fronts:
        combined_pareto = pd.concat(nonempty_fronts, ignore_index=True)
    else:
        combined_pareto = result_bundle.candidate_df.iloc[0:0].copy()
    pareto_csv = os.path.join(result_dir, "pareto.csv")
    combined_pareto.to_csv(pareto_csv, index=False)
    written_paths.append(pareto_csv)

    plot_path = _save_spica_pareto_plot(result_bundle, result_dir)
    if plot_path is not None:
        written_paths.append(plot_path)

    for mode, pareto_df in result_bundle.pareto_fronts.items():
        mode_dir = os.path.join(result_dir, str(mode))
        os.makedirs(mode_dir, exist_ok=True)

        if mode in result_bundle.tasks:
            mode_config_yaml = os.path.join(mode_dir, "exp_config.yaml")
            with open(mode_config_yaml, "w", encoding="utf-8") as fh:
                fh.write(result_bundle.tasks[mode].to_yaml())
            written_paths.append(mode_config_yaml)

        mode_pareto_csv = os.path.join(mode_dir, "pareto.csv")
        if pareto_df is None:
            pareto_df = result_bundle.candidate_df.iloc[0:0].copy()
        pareto_df.to_csv(mode_pareto_csv, index=False)
        written_paths.append(mode_pareto_csv)

        mode_best_csv = os.path.join(mode_dir, "best_config_topn.csv")
        best_config_df = result_bundle.best_configs.get(mode, pd.DataFrame())
        best_config_df.to_csv(mode_best_csv, index=False)
        written_paths.append(mode_best_csv)
        written_paths.extend(_save_spica_top_config_artifacts(result_bundle, mode, mode_dir, best_config_df, args))

    return written_paths


def _build_spica_default_search_space(args, backends: list[str]) -> dict[str, Any]:
    search_space: dict[str, Any] = {
        "model_name": args.model_path,
        "hardware_sku": args.system,
        "gpu_budget": args.total_gpus,
        "backend": backends,
        "router_mode": ["round_robin"],
        "overlap_score_credit": [0.0],
        "prefill_load_scale": [0.0],
        "router_temperature": [0.0],
        "planner_scaling_policy": ["disabled"],
        "planner_fpm_sampling": ["default"],
        "planner_load_sensitivity": ["default"],
        "prefill_gpu_memory_utilization": args.free_gpu_memory_fraction,
        "decode_gpu_memory_utilization": args.free_gpu_memory_fraction,
        "agg_gpu_memory_utilization": args.free_gpu_memory_fraction,
    }
    if args.max_seq_len is not None:
        search_space["context_length"] = args.max_seq_len
    if args.nextn > 0:
        search_space["aic_nextn"] = args.nextn

    if args.total_gpus == 1:
        logger.info("Constraining Spica default search to aggregate mode for a single-GPU budget.")
        search_space.update(
            {
                "deployment_mode": ["agg"],
            }
        )

    return search_space


def _build_spica_trace_search_space(args, backends: list[str]) -> dict[str, Any]:
    return _build_spica_default_search_space(args, backends)


def _spica_synthetic_concurrency(args) -> int:
    default = max(1, int(args.total_gpus or 1) * SPICA_THOROUGH_SYNTHETIC_CONCURRENCY_PER_GPU)
    return _positive_int_env("AIC_SPICA_THOROUGH_SYNTHETIC_CONCURRENCY", default)


def _spica_synthetic_request_count() -> int:
    return _positive_int_env("AIC_SPICA_THOROUGH_SYNTHETIC_REQUEST_COUNT", SPICA_THOROUGH_SYNTHETIC_REQUEST_COUNT)


def _build_spica_default_workload(args) -> dict[str, Any]:
    workload: dict[str, Any] = {
        "isl": args.isl,
        "osl": args.osl,
        "concurrency": _spica_synthetic_concurrency(args),
        "request_count": _spica_synthetic_request_count(),
    }
    if getattr(args, "prefix", 0) > 0 and args.isl > 0:
        workload["shared_prefix_ratio"] = min(1.0, max(0.0, float(args.prefix) / float(args.isl)))
        workload["num_prefix_groups"] = 1
    return workload


def _build_spica_thorough_config_data(args, backends: list[str]) -> dict[str, Any]:
    return {
        "search_space": _build_spica_default_search_space(args, backends),
        "workload": _build_spica_default_workload(args),
        "goal": {"target": "goodput_per_gpu", "sla": {"ttft_ms": args.ttft, "itl_ms": args.tpot}},
        "sweep": _spica_thorough_sweep_config(),
    }


def _spica_config_summary(config: Any) -> dict[str, Any]:
    search_space = getattr(config, "search_space", None)
    workload = getattr(config, "workload", None)
    goal = getattr(config, "goal", None)
    sweep = getattr(config, "sweep", None)
    return {
        "model": getattr(search_space, "model_name", None),
        "system": getattr(search_space, "hardware_sku", None),
        "gpu_budget": getattr(search_space, "gpu_budget", None),
        "backend": getattr(search_space, "backend", None),
        "workload": workload,
        "goal": goal,
        "sweep": {
            "max_rounds": getattr(sweep, "max_rounds", None),
            "parallel_evals": getattr(sweep, "parallel_evals", None),
            "candidates_per_round": getattr(sweep, "candidates_per_round", None),
        },
    }


def _spica_extra_input_lines(config: Any, config_path: str | None) -> list[str]:
    workload = getattr(config, "workload", None)
    goal = getattr(config, "goal", None)
    sla = getattr(goal, "sla", None)
    lines: list[str] = []
    if config_path:
        lines.append(f"Spica Config: {config_path}")
    if getattr(workload, "trace_path", None):
        trace_format = getattr(workload, "trace_format", None)
        lines.append(f"Trace: {workload.trace_path}")
        if trace_format:
            lines.append(f"Trace Format: {trace_format}")
        lines.append("Trace workload: request lengths come from replay.")
    else:
        lines.append(
            "Synthetic Workload: "
            f"ISL={getattr(workload, 'isl', 'n/a')}, "
            f"OSL={getattr(workload, 'osl', 'n/a')}, "
            f"concurrency={getattr(workload, 'concurrency', 'n/a')}, "
            f"request_count={getattr(workload, 'request_count', 'n/a')}"
        )
    if sla is not None:
        if getattr(sla, "ttft_ms", None) is not None:
            lines.append(f"TTFT Target: {sla.ttft_ms:.2f}ms")
        if getattr(sla, "itl_ms", None) is not None:
            lines.append(f"TPOT Target: {sla.itl_ms:.2f}ms")
        if getattr(sla, "e2e_ms", None) is not None:
            lines.append(f"Request Latency Target: {sla.e2e_ms:.2f}ms")
    return lines


def _load_spica_config(args, smart_search_config_cls):
    config_path = getattr(args, "thorough_config", None)
    if config_path:
        return smart_search_config_cls.from_yaml(config_path), config_path

    backends = [backend.value for backend in common.BackendName] if args.backend == "auto" else [args.backend]
    config_data = _build_spica_thorough_config_data(args, backends)
    return smart_search_config_cls.model_validate(config_data), None


def _install_dynamo_planner_bridge_compat() -> None:
    """Keep older Spica synthetic replay working with newer Dynamo module layout."""
    try:
        import dynamo.mocker as dynamo_mocker

        if hasattr(dynamo_mocker, "PlannerReplayBridge"):
            return
        from dynamo.llm import PlannerReplayBridge

        dynamo_mocker.PlannerReplayBridge = PlannerReplayBridge
    except Exception as exc:
        logger.debug("Could not install Dynamo PlannerReplayBridge compatibility shim: %s", exc)


class _SpicaReplayEvaluatorCompat:
    """ReplayEvaluator wrapper that installs AIC/Dynamo compatibility in spawned workers."""

    def __init__(self, workload: Any, goal: Any):
        self.workload = workload
        self.goal = goal
        self._evaluator: Any | None = None

    def evaluate(self, plan: Any) -> dict[str, float]:
        _install_dynamo_planner_bridge_compat()
        if self._evaluator is None:
            from spica.evaluator import ReplayEvaluator

            self._evaluator = ReplayEvaluator(self.workload, self.goal)
        return self._evaluator.evaluate(plan)


def run_spica_thorough_default(args) -> list[Any]:
    """Run the Spica smart sweeper for ``default --thorough-sweep`` / ``--thorough-config``."""
    from aiconfigurator.cli.report_and_save import log_final_summary

    if (
        getattr(args, "thorough_config", None) is None
        and getattr(args, "decode_system", None) is not None
        and getattr(args, "decode_system", None) != getattr(args, "system", None)
    ):
        raise SystemExit(
            "Spica thorough sweep currently requires homogeneous hardware when deriving config from CLI inputs; "
            "omit --decode-system, set it to --system, or provide a native --thorough-config."
        )

    if getattr(args, "backend_version", None) is not None:
        logger.warning(
            "--backend-version is currently ignored in Spica thorough mode; Spica resolves backend versions."
        )
    if getattr(args, "database_mode", common.DatabaseMode.SILICON.name) != common.DatabaseMode.SILICON.name:
        logger.warning("--database-mode is currently ignored in Spica thorough mode.")
    if getattr(args, "request_latency", None) is not None:
        logger.warning("--request-latency is currently ignored in Spica thorough mode; using --ttft/--tpot as the SLA.")
    if getattr(args, "enable_wideep", False):
        logger.warning("--enable-wideep is currently ignored in Spica thorough mode.")
    if getattr(args, "moe_backend", None) is not None:
        logger.warning("--moe-backend is currently ignored in Spica thorough mode.")

    try:
        from spica.config import SmartSearchConfig
        from spica.search import run_smart_search
    except ImportError as exc:
        raise SystemExit(
            "Spica thorough sweep requires the optional 'spica' package and its replay dependencies. "
            "Install Spica, then rerun with --thorough-sweep or --thorough-config."
        ) from exc

    _install_dynamo_planner_bridge_compat()
    config, config_path = _load_spica_config(args, SmartSearchConfig)
    config_summary = _spica_config_summary(config)

    logger.info(
        "Running Spica thorough sweep: model=%s, system=%s, gpu_budget=%s, backend=%s, workload=%s, sweep=%s",
        config_summary["model"],
        config_summary["system"],
        config_summary["gpu_budget"],
        config_summary["backend"],
        config_summary["workload"],
        config_summary["sweep"],
    )
    candidates = run_smart_search(config, evaluator=_SpicaReplayEvaluatorCompat(config.workload, config.goal))
    if not candidates:
        logger.error("Spica thorough sweep returned no feasible candidates.")
        raise SystemExit(1)

    result_bundle = _build_spica_trace_result_bundle(candidates, args, config=config, config_path=config_path)
    log_final_summary(
        chosen_exp=result_bundle.chosen_exp,
        best_throughputs=result_bundle.best_throughputs,
        best_configs=result_bundle.best_configs,
        pareto_fronts=result_bundle.pareto_fronts,
        tasks=result_bundle.tasks,
        mode="default",
        pareto_x_axis=result_bundle.pareto_x_axis,
        top_n=args.top_n,
        inclusive_tpot=args.inclusive_tpot,
        extra_input_lines=_spica_extra_input_lines(config, config_path),
    )

    if args.save_dir:
        written_paths = _save_spica_trace_artifacts(result_bundle, args.save_dir, args)
        logger.info("Saved Spica thorough artifacts to %s: %s", args.save_dir, ", ".join(written_paths))

    return candidates
