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
    forward_model: str | None = None,
    enable_encoder_dp: bool = True,
    moe_backend: str | None = None,
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
        forward_model=forward_model or "op_level",
        enable_encoder_dp=enable_encoder_dp,
        moe_backend=moe_backend,
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


# Conservative nextn_accepted fraction for DSPARK recommend mode.
# nextn_accepted has no backend equivalent — it is AIC's throughput planning
# assumption (average accepted tokens per step as a fraction of the block size).
# 0.8 is conservative for a well-aligned draft model.
_DSPARK_DEFAULT_ACCEPTANCE = 0.8


def resolve_dspark_nextn(model_path: str) -> tuple[int, float] | None:
    """Resolve DSPARK speculative parameters for the recommend/sizing path.

    DSPARK architectures use a standalone trained draft model whose block size
    is a fixed architectural constant — not stored in the main checkpoint, so
    ``nextn='auto'`` always returns 0 for these models.

    Returns ``(nextn, nextn_accepted)`` when the model is a DSPARK architecture,
    where ``nextn`` is the architectural block size and ``nextn_accepted`` is a
    conservative throughput planning assumption (no backend equivalent).
    Returns ``None`` when the model is not DSPARK or the config cannot be fetched.
    Raises ``ValueError`` when ``model_path`` is empty, matching ``resolve_nextn_auto``.
    """
    from aiconfigurator_core.sdk.common import DSPARK_NEXTN
    from aiconfigurator_core.sdk.utils import get_model_config_from_model_path

    if not model_path:
        raise ValueError("resolve_dspark_nextn requires a model path.")
    try:
        info = get_model_config_from_model_path(model_path)
        block_size = DSPARK_NEXTN.get(info.get("architecture", ""), 0)
        if block_size:
            return block_size, block_size * _DSPARK_DEFAULT_ACCEPTANCE
    except Exception:
        pass
    return None


def apply_nextn(
    model_config: ModelConfig,
    nextn: int | None,
) -> None:
    """Apply the MTP compute-side draft depth onto a ModelConfig."""
    model_config.nextn = normalize_nextn(nextn)
