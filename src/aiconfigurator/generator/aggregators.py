# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from typing import Any, Optional

import yaml

from .rendering import apply_defaults, get_param_keys
from .utils import DEFAULT_BACKEND, coerce_bool, coerce_int, normalize_backend


def _entry_allows_backend(entry: dict[str, Any], backend: str) -> bool:
    allowed = entry.get("backends")
    if not allowed:
        return True
    if isinstance(allowed, str):
        allowed_set = {allowed.strip().lower()}
    elif isinstance(allowed, Iterable):
        allowed_set = {str(item).strip().lower() for item in allowed if item is not None}
    else:
        allowed_set = set()
    return not allowed_set or backend in allowed_set


def collect_generator_params(
    service: dict[str, Any],
    k8s: dict[str, Any],
    prefill_params: Optional[dict[str, Any]] = None,
    decode_params: Optional[dict[str, Any]] = None,
    agg_params: Optional[dict[str, Any]] = None,
    prefill_workers: int = 1,
    decode_workers: int = 1,
    agg_workers: int = 1,
    sla: Optional[dict[str, Any]] = None,
    dyn_config: Optional[dict[str, Any]] = None,
    backend: Optional[str] = None,
) -> dict[str, Any]:
    prefill_params = prefill_params or {}
    decode_params = decode_params or {}
    agg_params = agg_params or {}
    backend_key = normalize_backend(backend, DEFAULT_BACKEND)
    service = apply_defaults("ServiceConfig", service, backend=backend_key)
    k8s = apply_defaults("K8sConfig", k8s, backend=backend_key)
    dyn_cfg = apply_defaults("DynConfig", dyn_config or {}, backend=backend_key)

    mode_value = dyn_cfg.get("mode") or "disagg"
    enable_router = coerce_bool(dyn_cfg.get("enable_router"))
    is_kv = bool(enable_router)
    router_mode = "kv" if is_kv else ""
    mode_tag = "agg" if mode_value == "agg" else "disagg"
    name_prefix = k8s.get("name_prefix") or "dynamo"
    name = f"{name_prefix}-{mode_tag}{('-router' if is_kv else '')}"
    use_engine_cm = k8s.get("k8s_engine_mode", "inline") == "configmap"
    _mc_raw = k8s.get("k8s_model_cache")
    k8s_model_cache = _mc_raw.strip() if isinstance(_mc_raw, str) else ""
    workers_dict = {
        "prefill_workers": int(prefill_workers),
        "decode_workers": int(decode_workers),
        "agg_workers": int(agg_workers),
        "prefill_gpus_per_worker": coerce_int(prefill_params.get("gpus_per_worker")),
        "decode_gpus_per_worker": coerce_int(decode_params.get("gpus_per_worker")),
        "agg_gpus_per_worker": coerce_int(agg_params.get("gpus_per_worker")),
    }
    service_payload = dict(service)
    service_payload.update(
        {
            "model_path": service.get("model_path"),
            "served_model_name": service.get("served_model_name"),
            "model_name": service.get("model_name") or service.get("served_model_name"),
            "head_node_ip": service.get("head_node_ip"),
            "port": coerce_int(service.get("port")),
            "include_frontend": coerce_bool(service.get("include_frontend")),
        }
    )
    k8s_payload = dict(k8s)
    k8s_payload.update(
        {
            "name_prefix": name_prefix,
            "mode": mode_value,
            "router_mode": router_mode,
            "is_kv": is_kv,
            "enable_router": is_kv,
            "name": name,
            "k8s_namespace": k8s.get("k8s_namespace"),
            "k8s_image": k8s.get("k8s_image"),
            "k8s_image_pull_secret": k8s.get("k8s_image_pull_secret"),
            "k8s_engine_mode": k8s.get("k8s_engine_mode"),
            "use_engine_cm": use_engine_cm,
            "k8s_model_cache": k8s_model_cache,
            "prefill_engine_args": "/workspace/engine_configs/prefill_config.yaml",
            "decode_engine_args": "/workspace/engine_configs/decode_config.yaml",
            "agg_engine_args": "/workspace/engine_configs/agg_config.yaml",
        }
    )
    return {
        "service": service_payload,
        "k8s": k8s_payload,
        "workers": workers_dict,
        "params": {
            "prefill": prefill_params,
            "decode": decode_params,
            "agg": agg_params,
        },
        "sla": sla or {},
    }


def generate_config_from_yaml(
    yaml_path: str,
    backend: Optional[str] = None,
    schema_path: Optional[str] = None,
) -> dict[str, Any]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    return generate_config_from_input_dict(cfg, schema_path=schema_path, backend=backend)


def _get_by_path(src: dict[str, Any], path: str) -> Any:
    cur = src
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _set_by_path(dst: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = dst
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def generate_config_from_input_dict(
    input_params: dict[str, Any],
    schema_path: Optional[str] = None,
    backend: Optional[str] = None,
) -> dict[str, Any]:
    current_dir = os.path.dirname(__file__)
    if not schema_path:
        schema_path = os.path.join(current_dir, "config", "deployment_config.yaml")
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    if schema is None:
        inputs = []
    elif isinstance(schema, list):
        inputs = schema
    else:
        inputs = schema.get("inputs", [])
    target: dict[str, Any] = {}
    backend_key = normalize_backend(backend, DEFAULT_BACKEND)
    for entry in inputs:
        key = entry.get("key")
        required = bool(entry.get("required", False))
        val = None
        if key:
            parts = key.split(".")
            val = _get_by_path(input_params, parts[0])
            if isinstance(val, dict):
                for p in parts[1:]:
                    val = val.get(p) if isinstance(val, dict) else None
        if not _entry_allows_backend(entry, backend_key):
            continue
        if val is None and required:
            raise ValueError(f"Missing required input: {key}")
        if val is not None and key:
            parts = key.split(".")
            group = parts[0]
            rest = parts[1:]
            if group == "ServiceConfig":
                dest = ".".join(["service"] + rest)
            elif group == "K8sConfig":
                dest = ".".join(["k8s"] + rest)
            elif group == "Workers":
                if len(rest) >= 2:
                    role = rest[0]
                    dest = ".".join(["params", role] + rest[1:])
                else:
                    dest = None
            elif group in {"WorkerCounts", "WorkerConfig"}:
                dest = ".".join(["workers"] + rest)
            elif group == "SlaConfig":
                dest = ".".join(["sla"] + rest)
            elif group == "ModelConfig":
                dest = ".".join(["service"] + rest)
            elif group == "DynConfig":
                dest = ".".join(["dyn_config"] + rest)
            else:
                dest = None
            if dest:
                _set_by_path(target, dest, val)
    target.setdefault("service", {})
    target.setdefault("k8s", {})
    target.setdefault("workers", {})
    target.setdefault("params", {})
    target.setdefault("sla", {})
    target.setdefault("dyn_config", {})
    try:
        default_mapping = os.path.join(os.path.dirname(__file__), "config", "backend_config_mapping.yaml")
        allowed_keys = set(get_param_keys(default_mapping))
    except Exception:
        allowed_keys = set()
    workers_in = input_params.get("Workers", {}) or {}
    for role in ("prefill", "decode", "agg"):
        role_in = workers_in.get(role, {}) or {}
        if isinstance(role_in, dict):
            filtered = {k: v for k, v in role_in.items() if (not allowed_keys or k in allowed_keys)}
            if filtered:
                for k, v in filtered.items():
                    _set_by_path(target, f"params.{role}.{k}", v)
    return collect_generator_params(
        service=target.get("service", {}),
        k8s=target.get("k8s", {}),
        prefill_params=target.get("params", {}).get("prefill"),
        decode_params=target.get("params", {}).get("decode"),
        agg_params=target.get("params", {}).get("agg"),
        prefill_workers=int(target.get("workers", {}).get("prefill_workers", 1)),
        decode_workers=int(target.get("workers", {}).get("decode_workers", 1)),
        agg_workers=int(target.get("workers", {}).get("agg_workers", 1)),
        sla=target.get("sla", {}),
        dyn_config=target.get("dyn_config", {}),
        backend=backend_key,
    )
