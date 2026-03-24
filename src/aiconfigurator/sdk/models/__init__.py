# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Models package — one file per model class with a decorator-based registry.

All public names are re-exported here for backward compatibility:
    from aiconfigurator.sdk.models import get_model, BaseModel, GPTModel, ...
"""

from __future__ import annotations

import logging

from aiconfigurator.sdk import config
from aiconfigurator.sdk.models.base import (
    BaseModel,
    _apply_model_quant_defaults,
    _architecture_to_model_family,
    _get_model_info,
    _infer_quant_modes_from_raw_config,
    calc_expectation,
    check_is_moe,
    get_model_family,
)
from aiconfigurator.sdk.models.deepseek import (
    DeepSeekModel,
    TrtllmWideEPDeepSeekModel,
    WideEPDeepSeekModel,
)

# Import all model modules to trigger @register_model decoration
from aiconfigurator.sdk.models.gpt import GPTModel
from aiconfigurator.sdk.models.hybrid_moe import HybridMoEModel
from aiconfigurator.sdk.models.llama import LLAMAModel
from aiconfigurator.sdk.models.moe import MOEModel
from aiconfigurator.sdk.models.nemotron_h import NemotronHModel
from aiconfigurator.sdk.models.nemotron_nas import NemotronNas

logger = logging.getLogger(__name__)


def get_model(
    model_path: str,
    model_config: config.ModelConfig,
    backend_name: str,
) -> BaseModel:
    """
    Get model.
    """
    model_info = _get_model_info(model_path)
    raw_config = model_info.get("raw_config", {})
    architecture = model_info["architecture"]
    layers = model_info["layers"]
    n = model_info["n"]
    n_kv = model_info["n_kv"]
    d = model_info["d"]
    hidden = model_info["hidden_size"]
    inter = model_info["inter_size"]
    vocab = model_info["vocab"]
    context = model_info["context"]
    topk = model_info["topk"]
    num_experts = model_info["num_experts"]
    moe_inter_size = model_info["moe_inter_size"]
    extra_params = model_info["extra_params"]
    # Convert architecture (e.g., 'LlamaForCausalLM') to model family (e.g., 'LLAMA')
    model_family = _architecture_to_model_family(architecture)

    _apply_model_quant_defaults(model_config, raw_config, architecture, backend_name)

    if model_config.overwrite_num_layers > 0:
        layers = model_config.overwrite_num_layers

    # --- Special dispatch for DEEPSEEK (backend/wideep variants) ---
    if model_family == "DEEPSEEK":
        if backend_name == "sglang" and model_config.enable_wideep:
            logger.debug(f"WideEP is enabled for model {model_path} with backend {backend_name}")
            model = WideEPDeepSeekModel(
                topk,
                num_experts,
                moe_inter_size,
                model_path,
                model_family,
                architecture,
                layers,
                n,
                n_kv,
                d,
                hidden,
                inter,
                vocab,
                context,
                model_config,
            )
        elif backend_name == "trtllm" and model_config.enable_wideep:
            logger.debug(f"TensorRT-LLM WideEP is enabled for model {model_path}")
            model = TrtllmWideEPDeepSeekModel(
                topk,
                num_experts,
                moe_inter_size,
                model_path,
                model_family,
                architecture,
                layers,
                n,
                n_kv,
                d,
                hidden,
                inter,
                vocab,
                context,
                model_config,
                extra_params,
            )
        else:
            logger.debug(f"WideEP is not enabled for model {model_path} with backend {backend_name}")
            model = DeepSeekModel(
                topk,
                num_experts,
                moe_inter_size,
                model_path,
                model_family,
                architecture,
                layers,
                n,
                n_kv,
                d,
                hidden,
                inter,
                vocab,
                context,
                model_config,
                extra_params,
            )
        return model

    # --- Standard registry lookup ---
    if model_family == "GPT":
        model = GPTModel(
            model_path,
            model_family,
            architecture,
            layers,
            n,
            n_kv,
            d,
            hidden,
            inter,
            vocab,
            context,
            model_config,
            extra_params,
        )
    elif model_family == "LLAMA":
        model = LLAMAModel(
            model_path,
            model_family,
            architecture,
            layers,
            n,
            n_kv,
            d,
            hidden,
            inter,
            vocab,
            context,
            model_config,
            extra_params,
        )
    elif model_family == "HYBRIDMOE":
        model = HybridMoEModel(
            topk,
            num_experts,
            moe_inter_size,
            model_path,
            model_family,
            architecture,
            layers,
            n,
            n_kv,
            d,
            hidden,
            inter,
            vocab,
            context,
            model_config,
        )
        model.set_hybrid_config(extra_params)
    elif model_family == "MOE":
        # currently we don't support wideep for sglang moe models (other than DS V3)
        model = MOEModel(
            topk,
            num_experts,
            moe_inter_size,
            model_path,
            model_family,
            architecture,
            layers,
            n,
            n_kv,
            d,
            hidden,
            inter,
            vocab,
            context,
            model_config,
            extra_params,
        )
    elif model_family == "NEMOTRONNAS":
        model = NemotronNas(
            model_path,
            model_family,
            architecture,
            layers,
            n,
            n_kv,
            d,
            hidden,
            inter,
            vocab,
            context,
            model_config,
        )
        # NemotronNAS uses extra_params as a list of BlockConfig to build its pipelines.
        # Not all model metadata sources carry these NAS block configs, so only apply them when provided.
        if isinstance(extra_params, list):
            model.context_ops = extra_params
            model.generation_ops = extra_params
        else:
            logger.warning(
                "NemotronNAS model '%s' missing block configs in model metadata; leaving pipelines empty.",
                model_path,
            )
            model.context_ops = []
            model.generation_ops = []
    elif model_family == "NEMOTRONH":
        model = NemotronHModel(
            topk,
            num_experts,
            moe_inter_size,
            model_path,
            model_family,
            architecture,
            layers,
            n,
            n_kv,
            d,
            hidden,
            inter,
            vocab,
            context,
            model_config,
        )
        # extra_params is NemotronHConfig with hybrid layer configuration
        model.set_hybrid_config(extra_params)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return model


# Re-export everything for backward compatibility
__all__ = [
    "BaseModel",
    "DeepSeekModel",
    "GPTModel",
    "HybridMoEModel",
    "LLAMAModel",
    "MOEModel",
    "NemotronHModel",
    "NemotronNas",
    "TrtllmWideEPDeepSeekModel",
    "WideEPDeepSeekModel",
    "_apply_model_quant_defaults",
    "_architecture_to_model_family",
    "_get_model_info",
    "_infer_quant_modes_from_raw_config",
    "calc_expectation",
    "check_is_moe",
    "get_model",
    "get_model_family",
]
