# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a GeneratorRequest from parsed CLI arguments.

Identity/sizing inputs map to typed fields; everything override-shaped
(--generator-config, --generator-set, and the promoted deployment flags that
``load_generator_overrides_from_args`` already folds into K8sConfig) lands in
``overrides.raw`` as flat ``Section.key`` entries, so existing CLI behavior is
preserved unchanged.
"""
from __future__ import annotations

from typing import Any

from .schema import (
    BackendSpec,
    EmitTargets,
    GeneratorRequest,
    ModelSpec,
    Overrides,
    Platform,
    SlaSpec,
    Topology,
)


def _flatten_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Nested {Section: {key: val}} -> flat {"Section.key": val}.

    ``generator_dynamo_version`` is a typed KEEP field (backend.dynamo_version),
    not an override, so it is dropped here.
    """
    raw: dict[str, Any] = {}
    for section, value in overrides.items():
        if section == "generator_dynamo_version":
            continue
        if isinstance(value, dict):
            for key, val in value.items():
                raw[f"{section}.{key}"] = val
        else:
            raw[section] = value
    return raw


def from_cli(args: Any) -> GeneratorRequest:
    # Lazy import to avoid an api <-> request import cycle.
    from ..api import load_generator_overrides_from_args

    overrides = load_generator_overrides_from_args(args)
    raw = _flatten_overrides(overrides)

    return GeneratorRequest(
        model=ModelSpec(model_path=getattr(args, "model_path", None) or ""),
        backend=BackendSpec(
            name=getattr(args, "backend", None) or "",
            dynamo_version=getattr(args, "generator_dynamo_version", None),
            generated_config_version=getattr(args, "generated_config_version", None),
        ),
        topology=Topology(
            mode=getattr(args, "mode", None) or "disagg",
            total_gpus=getattr(args, "total_gpus", None),
        ),
        sla=SlaSpec(isl=getattr(args, "isl", None), osl=getattr(args, "osl", None)),
        platform=Platform(hardware_profile=getattr(args, "system", None)),
        emit=EmitTargets(
            output_dir=getattr(args, "save_dir", None),
            deployment_target=getattr(args, "deployment_target", None) or "dynamo-j2",
        ),
        overrides=Overrides(raw=raw),
    )
