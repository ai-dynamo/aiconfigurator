# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for Dynamo feature knobs surfaced through ``DynConfig``."""

from __future__ import annotations

import copy
import json
import shlex
from typing import Any


def _present(value: Any) -> bool:
    return value is not None and value != ""


def normalize_router_mode(value: Any) -> str:
    text = str(value or "round-robin").strip().lower().replace("_", "-")
    if text in {"kv", "kv-router"}:
        return "kv"
    if text in {"round-robin", "round robin"}:
        return "round-robin"
    return text


def frontend_cli_args_from_dyn_config(
    dyn: dict[str, Any] | None,
    service: dict[str, Any] | None = None,
    *,
    include_http_port: bool = True,
) -> list[str]:
    """Build ``dynamo.frontend`` CLI args from generator ``DynConfig``."""

    dyn = dyn or {}
    service = service or {}
    router_config = dyn.get("router_config") or {}
    if not isinstance(router_config, dict):
        router_config = {}

    router_mode = dyn.get("router_mode")
    if not _present(router_mode) and (dyn.get("enable_router") or router_config):
        router_mode = "kv"

    args: list[str] = []
    if include_http_port:
        port = service.get("port")
        if _present(port):
            args.extend(["--http-port", str(port)])

    if _present(router_mode):
        args.extend(["--router-mode", normalize_router_mode(router_mode)])

    value_flags = (
        ("kv_cache_block_size", "--kv-cache-block-size"),
        ("overlap_score_credit", "--router-kv-overlap-score-credit"),
        ("prefill_load_scale", "--router-prefill-load-scale"),
        ("router_temperature", "--router-temperature"),
        ("active_decode_blocks_threshold", "--active-decode-blocks-threshold"),
        ("active_prefill_tokens_threshold", "--active-prefill-tokens-threshold"),
        ("active_prefill_tokens_threshold_frac", "--active-prefill-tokens-threshold-frac"),
        ("admission_control", "--admission-control"),
        ("router_queue_threshold", "--router-queue-threshold"),
        ("router_event_threads", "--router-event-threads"),
        ("router_queue_policy", "--router-queue-policy"),
        ("router_ttl_secs", "--router-ttl-secs"),
        ("router_snapshot_threshold", "--router-snapshot-threshold"),
        ("shared_cache_multiplier", "--shared-cache-multiplier"),
        ("shared_cache_type", "--shared-cache-type"),
        ("router_predicted_ttl_secs", "--router-predicted-ttl-secs"),
    )
    for key, flag in value_flags:
        value = router_config.get(key)
        if _present(value):
            args.extend([flag, str(value)])

    bool_flags = (
        ("router_reset_states", "--router-reset-states"),
        ("use_kv_events", "--router-kv-events"),
        ("durable_kv_events", "--router-durable-kv-events"),
        ("router_replica_sync", "--router-replica-sync"),
        ("router_track_active_blocks", "--router-track-active-blocks"),
        ("router_track_output_blocks", "--router-track-output-blocks"),
        ("router_assume_kv_reuse", "--router-assume-kv-reuse"),
        ("router_track_prefill_tokens", "--router-track-prefill-tokens"),
        ("use_remote_indexer", "--use-remote-indexer"),
        ("serve_indexer", "--serve-indexer"),
        ("load_aware", "--load-aware"),
    )
    for key, flag in bool_flags:
        if key not in router_config:
            continue
        if bool(router_config[key]):
            args.append(flag)
        else:
            args.append(f"--no-{flag[2:]}")

    return args


def frontend_cli_args_string(
    dyn: dict[str, Any] | None,
    service: dict[str, Any] | None = None,
    *,
    include_http_port: bool = True,
) -> str:
    args = frontend_cli_args_from_dyn_config(
        dyn,
        service,
        include_http_port=include_http_port,
    )
    return " ".join(shlex.quote(arg) for arg in args)


def planner_config_from_dyn_config(
    dyn: dict[str, Any] | None,
    *,
    backend: str,
    mode: str,
    service: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    dyn = dyn or {}
    raw = dyn.get("planner_config")
    if not isinstance(raw, dict) or not raw:
        return None
    service = service or {}
    planner = copy.deepcopy(raw)
    planner.setdefault("environment", "kubernetes")
    planner.setdefault("backend", backend)
    planner.setdefault("mode", mode)
    model_name = service.get("model_path") or service.get("served_model_path") or service.get("served_model_name")
    if model_name:
        planner.setdefault("model_name", model_name)
    return planner


def planner_image_from_k8s_config(k8s: dict[str, Any] | None) -> str | None:
    k8s = k8s or {}
    if k8s.get("k8s_planner_image"):
        return k8s.get("k8s_planner_image")
    image = k8s.get("k8s_image")
    if not image:
        return None
    image = str(image)
    for runtime in ("tensorrtllm-runtime", "vllm-runtime", "sglang-runtime"):
        if runtime in image:
            return image.replace(runtime, "dynamo-planner")
    return image


def planner_config_json(config: dict[str, Any]) -> str:
    return json.dumps(config, separators=(",", ":"))


def kvbm_env_from_dyn_config(dyn: dict[str, Any] | None) -> list[dict[str, str]]:
    dyn = dyn or {}
    cfg = dyn.get("kvbm_config") or {}
    if not isinstance(cfg, dict) or not cfg:
        return []

    env: list[dict[str, str]] = []
    mappings = (
        ("cpu_cache_gb", "DYN_KVBM_CPU_CACHE_GB"),
        ("cpu_cache_override_num_blocks", "DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"),
        ("disk_cache_gb", "DYN_KVBM_DISK_CACHE_GB"),
        ("disk_cache_override_num_blocks", "DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS"),
        ("max_transfer_batch_size", "DYN_KVBM_MAX_TRANSFER_BATCH_SIZE"),
        ("max_concurrent_transfers", "DYN_KVBM_MAX_CONCURRENT_TRANSFERS"),
    )
    for key, name in mappings:
        value = cfg.get(key)
        if _present(value):
            env.append({"name": name, "value": str(value)})
    return env


def kvbm_shell_exports_from_dyn_config(dyn: dict[str, Any] | None) -> list[str]:
    return [f"export {entry['name']}={shlex.quote(entry['value'])}" for entry in kvbm_env_from_dyn_config(dyn)]
