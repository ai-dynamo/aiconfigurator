# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared ModelConfig construction helpers.

These helpers are used by both the CLI layer and lower modeling/engine paths.
Keeping them in ``sdk`` prevents lower-level code from importing CLI code.
"""

from __future__ import annotations

from aiconfigurator_core.sdk.common import (
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)
from aiconfigurator_core.sdk.config import ModelConfig


def build_model_config(
    tp_size: int,
    pp_size: int,
    attention_dp_size: int,
    moe_tp_size: int,
    moe_ep_size: int,
    gemm_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
) -> ModelConfig:
    """Build a ModelConfig with optional quant mode overrides."""
    return ModelConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        gemm_quant_mode=GEMMQuantMode[gemm_quant_mode] if gemm_quant_mode else None,
        kvcache_quant_mode=KVCacheQuantMode[kvcache_quant_mode] if kvcache_quant_mode else None,
        fmha_quant_mode=FMHAQuantMode[fmha_quant_mode] if fmha_quant_mode else None,
        moe_quant_mode=MoEQuantMode[moe_quant_mode] if moe_quant_mode else None,
        comm_quant_mode=CommQuantMode[comm_quant_mode] if comm_quant_mode else None,
    )


def apply_nextn(
    model_config: ModelConfig,
    nextn: int | None,
    nextn_accepted: float | None,
) -> None:
    """Apply MTP speculative-decoding overrides onto a ModelConfig.

    ``nextn`` is the draft length, ``nextn_accepted`` the average accepted draft
    tokens per step. ``nextn_accepted`` is required when ``nextn > 0`` -- there is
    no built-in acceptance assumption.
    """
    model_config.nextn = int(nextn or 0)
    if model_config.nextn > 0:
        if nextn_accepted is None:
            raise ValueError(
                f"nextn={model_config.nextn} requires 'nextn_accepted' (average accepted draft tokens "
                f"per step, 0 <= nextn_accepted <= nextn); there is no built-in acceptance assumption."
            )
        if not 0 <= nextn_accepted <= model_config.nextn:
            raise ValueError(f"nextn_accepted ({nextn_accepted}) must be within [0, nextn={model_config.nextn}].")
        model_config.nextn_accepted = float(nextn_accepted)
