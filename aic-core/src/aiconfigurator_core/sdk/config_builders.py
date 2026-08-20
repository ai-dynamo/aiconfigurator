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
    speculation=None,
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
        speculation=speculation,
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


def resolve_speculation(model_config: ModelConfig):
    """Normalize (nextn, speculation) into a single resolved SpeculationConfig.

    Exactly one speculative source is allowed:

    * ``nextn > 0`` with no explicit scheme desugars to ``mtp`` at that depth
      (legacy sugar, keeps every existing entry point valid).
    * an explicit ``mtp`` scheme writes its depth back onto ``nextn`` so model
      families keep building their draft scaling from ``_nextn``.
    * a non-MTP scheme requires ``nextn == 0`` — mixing sources is an error,
      never a silent precedence.

    The resolved config is written back to ``model_config.speculation`` and
    returned. Idempotent. Must run BEFORE ``get_model`` constructs the model.
    """
    from aiconfigurator_core.sdk.speculation.base import SpeculationConfig

    spec = model_config.speculation
    nextn = normalize_nextn(model_config.nextn)

    if spec is None or spec.kind == "none":
        spec = SpeculationConfig(kind="mtp", params={"depth": nextn}) if nextn > 0 else (spec or SpeculationConfig())
    elif spec.kind == "mtp":
        # Same contract as legacy nextn: integer draft length (1.9 must be
        # rejected here exactly as normalize_nextn rejects it).
        depth = validate_nextn(spec.params.get("depth", 0))
        if depth < 1:
            raise ValueError(f"speculation kind 'mtp' requires params['depth'] >= 1, got {depth}.")
        if nextn and nextn != depth:
            raise ValueError(
                f"Conflicting speculative inputs: nextn={nextn} but speculation mtp depth={depth}. "
                "Set only one (nextn is legacy sugar for the mtp scheme)."
            )
        model_config.nextn = depth
    else:
        if nextn > 0:
            raise ValueError(
                f"Conflicting speculative inputs: nextn={nextn} cannot be combined with "
                f"speculation kind {spec.kind!r}. nextn is MTP-only sugar; set it to 0."
            )

    model_config.speculation = spec
    return spec
