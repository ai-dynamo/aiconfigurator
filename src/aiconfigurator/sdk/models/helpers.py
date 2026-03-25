# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import logging
from functools import cache
from typing import Optional

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.utils import get_model_config_from_model_path

logger = logging.getLogger(__name__)


@cache
def _get_model_info(model_path: str) -> dict:
    """
    Get model configuration info from model path.

    Args:
        model_path: HuggingFace model path (e.g., 'meta-llama/Llama-2-7b-hf') or local path

    Returns:
        dict: Model configuration parameters and raw config under "raw_config".
    """
    return get_model_config_from_model_path(model_path)


def _architecture_to_model_family(architecture: str) -> str:
    """
    Convert architecture name to model family.
    Handles both HuggingFace architecture names (e.g., 'LlamaForCausalLM')
    and internal model family names (e.g., 'LLAMA').
    """
    if architecture in common.ARCHITECTURE_TO_MODEL_FAMILY:
        return common.ARCHITECTURE_TO_MODEL_FAMILY[architecture]
    if architecture in common.ModelFamily:
        return architecture
    raise ValueError(
        f"Unknown architecture or model family: {architecture}. "
        f"Supported architectures: {', '.join(common.ARCHITECTURE_TO_MODEL_FAMILY.keys())}. "
        f"Supported model families: {', '.join(common.ModelFamily)}."
    )


def _infer_quant_modes_from_raw_config(raw_config: dict) -> dict[str, object]:
    quant_algo = raw_config.get("quant_algo")
    quant_dynamic = raw_config.get("quant_dynamic")
    kv_cache_algo = raw_config.get("kv_cache_quant_algo")

    overrides: dict[str, object] = {}

    # GEMM quant mode, MoE quant mode
    if quant_algo == "fp8":
        if quant_dynamic is False:
            overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8_static
        else:
            overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8
        overrides["moe_quant_mode"] = common.MoEQuantMode.fp8
    elif quant_algo == "fp8_block":
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.fp8_block
        overrides["moe_quant_mode"] = common.MoEQuantMode.fp8_block
    elif quant_algo == "nvfp4":
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.nvfp4
        overrides["moe_quant_mode"] = common.MoEQuantMode.nvfp4
    elif quant_algo == "mxfp4":
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.float16
        overrides["moe_quant_mode"] = common.MoEQuantMode.w4a16_mxfp4
    elif quant_algo == "float16":
        overrides["gemm_quant_mode"] = common.GEMMQuantMode.float16
        overrides["moe_quant_mode"] = common.MoEQuantMode.float16
    elif quant_algo is not None:
        raise ValueError(f"Unsupported quant algorithm: {quant_algo}")

    # KVCache quant mode
    # TODO: support fp4 kv cache
    if kv_cache_algo == "fp8":
        overrides["kvcache_quant_mode"] = common.KVCacheQuantMode.fp8
    elif kv_cache_algo == "float16":
        overrides["kvcache_quant_mode"] = common.KVCacheQuantMode.float16
    elif kv_cache_algo is not None:
        raise ValueError(f"Unsupported kv cache algorithm: {kv_cache_algo}")

    # FMHA quant mode
    if quant_algo is not None and (quant_algo in ("fp8", "fp8_block", "nvfp4") or kv_cache_algo in ("fp8",)):
        overrides["fmha_quant_mode"] = common.FMHAQuantMode.fp8
        if kv_cache_algo is None or kv_cache_algo != "fp8":
            overrides["kvcache_quant_mode"] = common.KVCacheQuantMode.fp8

    return overrides


def _apply_model_quant_defaults(
    model_config: config.ModelConfig,
    raw_config: dict,
    architecture: str,
    backend_name: str,
    worker_name: Optional[str] = None,
) -> None:
    # Clone original model_config to track if any modifications were made
    original_config = dataclasses.replace(model_config)

    inferred = _infer_quant_modes_from_raw_config(raw_config)
    applied: list[str] = []
    for key, value in inferred.items():
        if getattr(model_config, key, None) is None:
            setattr(model_config, key, value)
            applied.append(f"{key}={value.name}")

    if model_config.gemm_quant_mode is None:
        model_config.gemm_quant_mode = common.GEMMQuantMode.float16
    if model_config.moe_quant_mode is None:
        model_config.moe_quant_mode = common.MoEQuantMode.float16
    if model_config.kvcache_quant_mode is None:
        model_config.kvcache_quant_mode = common.KVCacheQuantMode.float16
    if model_config.fmha_quant_mode is None:
        model_config.fmha_quant_mode = common.FMHAQuantMode.float16
    if model_config.comm_quant_mode is None:
        model_config.comm_quant_mode = common.CommQuantMode.half

    if applied:
        logger.debug("Using model-provided quantization defaults: %s", ", ".join(applied))

    # FIXME: temporary workaround for Deepseek V3 fp8 fmha quant mode, only float16+fp8kvcache is supported
    if (
        architecture in ("DeepseekV3ForCausalLM", "KimiK25ForConditionalGeneration")
        and model_config.fmha_quant_mode == common.FMHAQuantMode.fp8
    ):
        model_config.fmha_quant_mode = common.FMHAQuantMode.float16

    # FIXME: temporary workaround for Qwen3 32B FP8, only float16+fp8kvcache is supported
    # VLLM perf tables only include float16 FMHA; fall back to float16 for estimation.
    if backend_name == "vllm" and model_config.fmha_quant_mode == common.FMHAQuantMode.fp8:
        model_config.fmha_quant_mode = common.FMHAQuantMode.float16

    # Only log if model_config was modified
    if original_config != model_config:
        logger.info(
            "Resolved quant modes for %s: gemm=%s moe=%s kvcache=%s fmha=%s comm=%s",
            worker_name or architecture,
            model_config.gemm_quant_mode,
            model_config.moe_quant_mode,
            model_config.kvcache_quant_mode,
            model_config.fmha_quant_mode,
            model_config.comm_quant_mode,
        )


def get_model_family(model_path: str) -> str:
    """
    Get model family.
    Converts architecture name to model family if needed.
    """
    architecture = _get_model_info(model_path)["architecture"]
    return _architecture_to_model_family(architecture)


def check_is_moe(model_path: str) -> bool:
    """
    Check if the model is a MoE model.

    For NEMOTRONH models, checks if 'E' (MoE layer) is in hybrid_override_pattern..
    E.g., Nemotron_H is not an MoE model, but Nemotron_3 is an MoE model.
    """
    family = get_model_family(model_path)
    if family in ("MOE", "DEEPSEEK", "HYBRIDMOE"):
        return True
    if family == "NEMOTRONH":
        model_info = _get_model_info(model_path)
        extra_params = model_info.get("extra_params")
        if extra_params is None or not hasattr(extra_params, "hybrid_override_pattern"):
            logger.warning(f"NEMOTRONH model {model_path} missing hybrid_override_pattern, defaulting is_moe=False")
            return False
        # 'E' in pattern means MoE layers are present
        return "E" in extra_params.hybrid_override_pattern
    return False


def calc_expectation(nextn: int, nextn_accept_rates: list[float]) -> float:
    """
    Calculate expectation for mtp
    """
    prob = 1.0
    if nextn == 0:
        return 0.0

    for i in range(nextn):
        prob *= nextn_accept_rates[i]
    if nextn > 1:
        return prob + calc_expectation(nextn - 1, nextn_accept_rates)
    else:
        return prob
