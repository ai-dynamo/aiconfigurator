# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layerwise model registry for public vLLM collection runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerwiseModel:
    """Configuration for one model in the layerwise collection registry."""

    model: str
    kind: str
    tp_sizes: tuple[int, ...] = (1, 2, 4, 8)
    ep_sizes: tuple[int, ...] = (1,)
    gemm_quant: str = "bf16"
    moe_quant: str = "bf16"
    attn_quant: str = "bf16"
    kv_quant: str = "bf16"
    num_slots: int | None = None
    gen_driver: str = "prefix_cache"

    @property
    def is_moe(self) -> bool:
        """Return whether this registry entry should use MoE EP expansion."""
        return self.kind == "moe"


DEFAULT_MODELS: tuple[LayerwiseModel, ...] = (
    LayerwiseModel(model="Qwen/Qwen3-32B", kind="dense"),
    LayerwiseModel(model="Qwen/Qwen3.6-35B-A3B", kind="moe", ep_sizes=(1, 2, 4, 8)),
)

OPTIONAL_MODELS: tuple[LayerwiseModel, ...] = (
    LayerwiseModel(
        model="deepseek-ai/DeepSeek-V4-Flash",
        kind="moe",
        ep_sizes=(1, 2, 4, 8),
        gemm_quant="fp8_block",
        moe_quant="w4a8_mxfp4_mxfp8",
        kv_quant="fp8",
    ),
)

REGISTERED_MODELS: tuple[LayerwiseModel, ...] = DEFAULT_MODELS + OPTIONAL_MODELS


def all_models() -> tuple[LayerwiseModel, ...]:
    """Return the default layerwise collection set."""
    return DEFAULT_MODELS


def select_models(raw_models: str | None) -> list[LayerwiseModel]:
    """Return registry entries selected by a comma-separated model filter."""
    if not raw_models:
        return list(DEFAULT_MODELS)
    requested = [value.strip() for value in raw_models.split(",") if value.strip()]
    by_name = {entry.model: entry for entry in REGISTERED_MODELS}
    missing = [model for model in requested if model not in by_name]
    if missing:
        choices = ", ".join(sorted(by_name))
        raise ValueError(f"Unknown layerwise model(s): {missing}. Registered models: {choices}")
    return [by_name[model] for model in requested]
