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


def validate_nextn(nextn: int | None) -> int:
    """Validate and normalize the MTP draft length.

    The ``aic-core`` layer owns only the compute-side draft depth. Accepted-token progress is
    modeled by the upper prediction layer and therefore is intentionally not
    part of this helper or :class:`ModelConfig`.
    """
    if nextn is not None and int(nextn) != nextn:
        raise ValueError(f"nextn ({nextn}) must be an integer draft length.")
    normalized = int(nextn or 0)
    if normalized < 0:
        raise ValueError(f"nextn ({nextn}) must be >= 0.")
    return normalized


def normalize_nextn(nextn: int | None) -> int:
    """Return the MTP draft length normalized for ``aic-core``."""
    return validate_nextn(nextn)


def resolve_nextn_auto(model_path: str) -> int:
    """Resolve ``nextn='auto'`` to the checkpoint's MTP draft depth.

    Reads ``num_nextn_predict_layers`` from the model config (the multimodal
    text sub-config when applicable); absent or 0 means the checkpoint ships no
    MTP layers and MTP stays disabled. The checkpoint is the single source of
    truth -- there is no model-family fallback.
    """
    # Local import: utils pulls in the perf-database layer, which config
    # builders must not depend on at import time.
    from aiconfigurator_core.sdk.common import MULTIMODAL_TEXT_CONFIG_KEY
    from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

    if not model_path:
        raise ValueError("nextn='auto' requires a model path to resolve num_nextn_predict_layers.")
    info = get_model_config_from_model_path(model_path)
    raw = info.get("raw_config", {})
    text_key = MULTIMODAL_TEXT_CONFIG_KEY.get(info["architecture"])
    cfg = raw[text_key] if text_key and text_key in raw else raw
    return int(cfg.get("num_nextn_predict_layers") or 0)


def apply_nextn(
    model_config: ModelConfig,
    nextn: int | None,
) -> None:
    """Apply the MTP compute-side draft depth onto a ModelConfig."""
    model_config.nextn = normalize_nextn(nextn)
