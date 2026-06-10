# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim: convert legacy V1 task YAML to the flat V2 ``Task`` schema.

The V1 task config (``sdk.task.TaskConfig``) uses a nested YAML format -- a
``config:`` block with ``worker_config`` / ``prefill_worker_config`` /
``decode_worker_config`` / ``replica_config`` / ``advanced_tuning_config``
subsections, a ``mode`` (patch/override) selector, and a ``profiles`` list.
V2 (``sdk.task_v2.Task``) uses a flat 1:1 schema where every key maps directly
to a dataclass field, and disagg mode forbids shared top-level worker fields
(``model_path`` / ``system_name`` / ...), requiring ``prefill_*`` / ``decode_*``.

This module is the one-way bridge: ``convert_v1_to_v2(v1_dict) -> v2_dict``.
It lets existing V1 YAML keep working (callers emit a ``DeprecationWarning``)
until users migrate to the flat format. It depends only on plain dict ops --
no import of ``sdk.task`` -- so it survives V1 removal.

The field mapping is anchored to the V1<->V2 parity test
(``tests/integration/test_task_v1_v2_parity.py``), which is the ground truth
for how a V1 config corresponds to a V2 ``Task``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# V1 worker_config list fields -> V2 "*_candidates" suffix.
_LIST_FIELDS: dict[str, str] = {
    "num_gpu_per_worker": "num_gpu_candidates",
    "tp_list": "tp_candidates",
    "pp_list": "pp_candidates",
    "dp_list": "dp_candidates",
    "moe_tp_list": "moe_tp_candidates",
    "moe_ep_list": "moe_ep_candidates",
}

# V1 worker_config scalar fields that carry over with an (optional) role prefix.
_WORKER_SCALARS: frozenset[str] = frozenset(
    {
        "gemm_quant_mode",
        "moe_quant_mode",
        "kvcache_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
        "enable_wideep",
        "enable_eplb",
    }
)

# V1 advanced_tuning_config -> V2 flat field (note the _scale -> "" rename).
_ADV_TUNING: dict[str, str] = {
    "prefill_max_batch_size": "prefill_max_batch_size",
    "decode_max_batch_size": "decode_max_batch_size",
    "prefill_latency_correction_scale": "prefill_latency_correction",
    "decode_latency_correction_scale": "decode_latency_correction",
}

# V1 replica_config -> V2 flat field (note worker -> workers).
_REPLICA: dict[str, str] = {
    "num_gpu_per_replica": "num_gpu_per_replica",
    "max_gpu_per_replica": "max_gpu_per_replica",
    "max_prefill_worker": "max_prefill_workers",
    "max_decode_worker": "max_decode_workers",
}

# Top-level V1 scalars that pass through unchanged (same name in V2, both modes).
_TOP_PASSTHROUGH: frozenset[str] = frozenset(
    {
        "serving_mode",
        "isl",
        "osl",
        "prefix",
        "ttft",
        "tpot",
        "request_latency",
        "total_gpus",
        "database_mode",
        "free_gpu_memory_fraction",
        "max_seq_len",
        "engine_step_backend",
        "nextn",
        "nextn_accept_rates",
        "moe_backend",
    }
)

# Global scalars that may appear inside the V1 ``config:`` block.
_CONFIG_SCALARS: frozenset[str] = frozenset({"moe_backend", "nextn", "nextn_accept_rates"})

# Keys that are part of the V1 format itself (handled structurally, never copied).
_STRUCTURAL: frozenset[str] = frozenset({"mode", "config", "profiles"})

# Markers that unambiguously identify a V1 config dict.
_V1_MARKERS: frozenset[str] = frozenset(
    {
        "mode",
        "config",
        "profiles",
        "worker_config",
        "prefill_worker_config",
        "decode_worker_config",
        "replica_config",
        "advanced_tuning_config",
        "attention_backend",
        "wideep_num_slots",
    }
)


def is_v1_config(data: dict) -> bool:
    """Heuristically detect a legacy V1 task config dict.

    Detection keys on V1-only structural markers (``mode`` / ``config`` /
    ``profiles`` / ``*_worker_config`` / ``replica_config`` / ...).  A bare
    disagg dict that only sets a top-level ``model_path`` (without any such
    marker) is NOT treated as V1 -- it is left to V2's prefix-discipline
    check, which rejects it and steers the user to ``prefill_*`` / ``decode_*``.
    """
    return any(key in data for key in _V1_MARKERS)


def _quant_preset_from_profiles(profiles: list, unmapped: list[str]) -> str | None:
    """Map a V1 ``profiles`` list to a single V2 ``quant_preset``."""
    profiles = [p for p in (profiles or []) if p]
    if not profiles:
        return None
    if len(profiles) > 1:
        unmapped.append(f"profiles{profiles!r} (V2 takes a single quant_preset; using {profiles[0]!r})")
    return profiles[0]


def _convert_worker(out: dict, worker_cfg: dict, *, list_prefix: str, scalar_prefix: str, unmapped: list[str]) -> None:
    """Flatten one V1 worker_config block into V2 fields under the given prefixes."""
    for key, value in (worker_cfg or {}).items():
        if key in _LIST_FIELDS:
            out[f"{list_prefix}{_LIST_FIELDS[key]}"] = value
        elif key in _WORKER_SCALARS:
            out[f"{scalar_prefix}{key}"] = value
        else:
            unmapped.append(f"worker_config.{key}")


def convert_v1_to_v2(v1: dict) -> dict:
    """Convert one legacy V1 experiment dict to a flat V2 ``Task`` dict.

    Args:
        v1: A single V1 experiment config (top-level scalars + optional nested
            ``config:`` block). NOT a whole file with an ``exps:`` list.

    Returns:
        A flat dict whose keys are V2 ``Task`` field names, suitable for
        ``Task.from_yaml`` / ``Task(**...)``.

    Notes:
        Unmappable V1 keys (e.g. ``attention_backend``, ``wideep_num_slots``,
        the ``mode`` selector) are collected and reported in a single
        warning; they are dropped, not silently swallowed.
    """
    out: dict = {}
    unmapped: list[str] = []
    serving_mode = v1.get("serving_mode", "agg")
    config = v1.get("config") or {}
    profiles = v1.get("profiles") or []

    # 1. Top-level scalars common to both modes.
    for key in _TOP_PASSTHROUGH:
        if key in v1:
            out[key] = v1[key]

    # 2. Global scalars that may live inside config:.
    for key in _CONFIG_SCALARS:
        if key in config:
            out[key] = config[key]
    if "attention_backend" in config:
        unmapped.append("config.attention_backend")
    if "wideep_num_slots" in v1:
        unmapped.append("wideep_num_slots")

    preset = _quant_preset_from_profiles(profiles, unmapped)

    if serving_mode == "disagg":
        # Fan out shared top-level worker spec to both roles (V2 forbids top-level).
        for role in ("prefill", "decode"):
            if "model_path" in v1:
                out[f"{role}_model_path"] = v1["model_path"]
            if "backend_name" in v1:
                out[f"{role}_backend_name"] = v1["backend_name"]
            if "backend_version" in v1:
                out[f"{role}_backend_version"] = v1["backend_version"]
            if "enable_wideep" in v1:
                out[f"{role}_enable_wideep"] = v1["enable_wideep"]
            if "enable_eplb" in v1:
                out[f"{role}_enable_eplb"] = v1["enable_eplb"]
            if preset is not None:
                out[f"{role}_quant_preset"] = preset
        if "system_name" in v1:
            out["prefill_system_name"] = v1["system_name"]
        # decode_system_name falls back to the (prefill) system_name when absent.
        if "decode_system_name" in v1:
            out["decode_system_name"] = v1["decode_system_name"]
        elif "system_name" in v1:
            out["decode_system_name"] = v1["system_name"]
        _convert_worker(
            out,
            config.get("prefill_worker_config", {}),
            list_prefix="prefill_",
            scalar_prefix="prefill_",
            unmapped=unmapped,
        )
        _convert_worker(
            out,
            config.get("decode_worker_config", {}),
            list_prefix="decode_",
            scalar_prefix="decode_",
            unmapped=unmapped,
        )
    else:
        # agg: top-level worker spec passes through; list candidates get the agg_ prefix.
        for key in ("model_path", "system_name", "backend_name", "backend_version", "enable_wideep", "enable_eplb"):
            if key in v1:
                out[key] = v1[key]
        if preset is not None:
            out["quant_preset"] = preset
        _convert_worker(out, config.get("worker_config", {}), list_prefix="agg_", scalar_prefix="", unmapped=unmapped)

    # 3. replica_config + advanced_tuning_config (disagg-oriented, but map verbatim).
    for v1_key, v2_key in _REPLICA.items():
        if v1_key in config.get("replica_config", {}):
            out[v2_key] = config["replica_config"][v1_key]
    for v1_key, v2_key in _ADV_TUNING.items():
        if v1_key in config.get("advanced_tuning_config", {}):
            out[v2_key] = config["advanced_tuning_config"][v1_key]

    if "mode" in v1:
        # patch/override layering has no V2 equivalent (flat schema); harmless to drop.
        logger.debug("convert_v1_to_v2: dropping V1 'mode'=%r (no V2 equivalent)", v1.get("mode"))

    if unmapped:
        logger.warning(
            "convert_v1_to_v2: %d V1 field(s) have no V2 equivalent and were dropped: %s",
            len(unmapped),
            ", ".join(unmapped),
        )
    return out
