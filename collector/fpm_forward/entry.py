# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bridge from the existing collector CLI into the FPM campaign runner."""

from __future__ import annotations

import argparse
import copy
from typing import Any

import yaml

from .config import FPMCollectionOptions
from .planner import FPMCollectionPlan, build_collection_plan


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _inline_overrides(items: list[str] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"--generator-set requires KEY=VALUE, got {item!r}")
        dotted, raw_value = item.split("=", 1)
        keys = [part for part in dotted.split(".") if part]
        if not keys:
            raise ValueError(f"invalid --generator-set key: {dotted!r}")
        cursor = payload
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise TypeError(f"conflicting --generator-set path: {dotted!r}")
            cursor = child
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return payload


def _load_generator_overrides(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if args.generator_config:
        with open(args.generator_config, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise TypeError("--generator-config must contain a YAML mapping")
        payload = loaded
    payload = _deep_merge(payload, _inline_overrides(args.generator_set))
    if args.generator_dynamo_version:
        payload["generator_dynamo_version"] = args.generator_dynamo_version
    if args.generated_config_version:
        payload["generated_config_version"] = args.generated_config_version
    k8s: dict[str, Any] = {}
    if args.namespace:
        k8s["k8s_namespace"] = args.namespace
    if args.transport:
        k8s["transport"] = args.transport
    if args.image_pull_secret:
        k8s["k8s_image_pull_secret"] = args.image_pull_secret
    if args.model_cache:
        parts = args.model_cache.split(":")
        k8s["k8s_pvc_name"] = parts[0]
        if len(parts) > 1 and parts[1]:
            k8s["k8s_pvc_mount_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            k8s["k8s_model_path_in_pvc"] = parts[2]
    if k8s:
        payload = _deep_merge(payload, {"K8sConfig": k8s})
    return payload


def resolve_inputs(args: argparse.Namespace, case_plan) -> tuple[FPMCollectionPlan, dict[str, Any], str | None]:
    if case_plan is None or not case_plan.model_path:
        raise ValueError("fpm_forward requires a resolved --model-path or single-model case plan")
    if not args.gpu:
        raise ValueError("fpm_forward requires --gpu to identify the target AIC system")

    options = FPMCollectionOptions.from_args(args)
    generator_overrides = copy.deepcopy(_load_generator_overrides(args))
    collector_config = generator_overrides.pop("FpmCollector", {})
    if not isinstance(collector_config, dict):
        raise TypeError("FpmCollector generator-config section must be a mapping")
    runtime_overlay_dir = collector_config.get("runtime_overlay_dir")
    plan = build_collection_plan(
        backend=args.backend,
        model_path=case_plan.model_path,
        system=args.gpu,
        selected_ops=set(case_plan.selected_ops),
        options=options,
        collector_config=collector_config,
        generator_overrides=generator_overrides,
    )
    return plan, generator_overrides, str(runtime_overlay_dir) if runtime_overlay_dir is not None else None


def run_from_args(args: argparse.Namespace, case_plan) -> list[dict[str, object]]:
    if args.limit is not None and not args.smoke:
        raise ValueError("fpm_forward --limit is allowed only with --smoke")
    if args.limit is not None and args.limit < 1:
        raise ValueError("fpm_forward --limit must be a positive cell count")
    if args.fpm_smoke_points is not None and not args.smoke:
        raise ValueError("--fpm-smoke-points requires --smoke")
    plan, generator_overrides, runtime_overlay_dir = resolve_inputs(args, case_plan)
    from .runner import run_collection

    return run_collection(
        plan,
        generator_overrides=generator_overrides,
        checkpoint_dir=args.checkpoint_dir,
        artifact_root=args.fpm_artifact_root or "fpm_forward_artifacts",
        resume=args.resume,
        retry_failed=args.resume_retry_failed,
        smoke=args.smoke,
        cell_limit=args.limit,
        runtime_overlay_dir=runtime_overlay_dir,
        database_root=args.fpm_database_root,
    )
