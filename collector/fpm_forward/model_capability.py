# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-config and declared attention-family resolution for FPM cells."""

from __future__ import annotations

import json
from pathlib import Path

_ATTENTION_SOURCE_OPS = {
    "dsa_module": frozenset({"dsa_context_module", "dsa_generation_module"}),
    "dsv4_module": frozenset(
        {
            "dsv4_csa_context_module",
            "dsv4_hca_context_module",
            "dsv4_csa_generation_module",
            "dsv4_hca_generation_module",
        }
    ),
    "mla_module": frozenset({"mla_context_module", "mla_generation_module"}),
    "dense_attention": frozenset({"attention_context", "attention_generation"}),
}


def _cached_model_config_path(model_path: str) -> Path | None:
    direct = Path(model_path).expanduser()
    if direct.is_dir() and (direct / "config.json").exists():
        return direct / "config.json"
    if direct.is_file():
        return direct
    root = Path(__file__).resolve().parents[2]
    cached = root / "src" / "aiconfigurator" / "model_configs" / f"{model_path.replace('/', '--')}_config.json"
    return cached if cached.exists() else None


def load_model_config(model_path: str) -> tuple[dict[str, object], Path | None]:
    path = _cached_model_config_path(model_path)
    if path is None:
        return {}, None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"model config must be a mapping: {path}")
    text_config = payload.get("text_config")
    if isinstance(text_config, dict):
        payload = {**payload, **text_config}
    return payload, path


def resolve_attention_source(selected_ops: set[str], *, required: bool = True) -> str | None:
    """Return the unique attention family declared by the model case plan."""

    matches = [name for name, ops in _ATTENTION_SOURCE_OPS.items() if ops.issubset(selected_ops)]
    if not matches:
        if not required:
            return None
        raise ValueError(
            "fpm_forward requires one supported AIC context/generation attention pair; "
            f"selected ops were {sorted(selected_ops)}"
        )
    if len(matches) > 1:
        raise ValueError(
            "fpm_forward attention source is ambiguous; model cases must select one pair: " + ", ".join(matches)
        )
    return matches[0]
