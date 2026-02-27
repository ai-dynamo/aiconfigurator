# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from functools import cache

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.utils import _load_model_config_from_model_path, get_model_config_from_model_path

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
    model_config: config.ModelConfig, raw_config: dict, architecture: str, backend_name: str
) -> None:
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
    if architecture == "DeepseekV3ForCausalLM" and model_config.fmha_quant_mode == common.FMHAQuantMode.fp8:
        model_config.fmha_quant_mode = common.FMHAQuantMode.float16

    # FIXME: temporary workaround for Qwen3 32B FP8, only float16+fp8kvcache is supported
    # VLLM perf tables only include float16 FMHA; fall back to float16 for estimation.
    if backend_name == "vllm" and model_config.fmha_quant_mode == common.FMHAQuantMode.fp8:
        model_config.fmha_quant_mode = common.FMHAQuantMode.float16

    logger.info(
        "Model config (final quant modes): gemm=%s moe=%s kvcache=%s fmha=%s comm=%s",
        model_config.gemm_quant_mode,
        model_config.moe_quant_mode,
        model_config.kvcache_quant_mode,
        model_config.fmha_quant_mode,
        model_config.comm_quant_mode,
    )


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
        )
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
        )
    elif model_family == "DEEPSEEK":
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
        model.context_ops = extra_params
        model.generation_ops = extra_params
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

    return model


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
    if family in ("MOE", "DEEPSEEK"):
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


class BaseModel:
    """
    Base model class.
    """

    def __init__(
        self,
        model_path: str,
        model_family: str,
        architecture: str,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        hidden_size: int,
        inter_size: int,
        vocab_size: int,
        context_length: int,
        model_config: config.ModelConfig,
    ) -> None:
        self.model_path = model_path
        self.model_family = model_family
        self.architecture = architecture
        self.config = model_config
        self.context_ops = []
        self.generation_ops = []

        # internal only
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_size = head_size
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._vocab_size = vocab_size
        self._context_length = context_length
        self._num_kv_heads_per_gpu = (self._num_kv_heads + model_config.tp_size - 1) // model_config.tp_size

        if self._num_layers % model_config.pp_size != 0:
            logger.warning(
                f"num_layers {self._num_layers} is not divisible by pp_size "
                f"{model_config.pp_size}. this will introduce additional rounding error. "
                f"Currently we're nothing to correct this."
            )

        assert self._num_heads % model_config.tp_size == 0, (
            f"num_heads {self._num_heads} should be divisible by tp_size {model_config.tp_size} "
        )

        self._nextn = model_config.nextn
        self._nextn_accept_rates = model_config.nextn_accept_rates


class GPTModel(BaseModel):
    """
    GPT series uses this model impl.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention,
    exact match is required. Same for logits_gemm.
    Other than DS V3, all other models don't support mtp
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, "Only DS V3 supports mtp"

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_gpu = self._num_kv_heads_per_gpu
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "context_qkv_gemm",
                    self._num_layers,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                ),
                ops.ContextAttention(
                    "context_attention",
                    self._num_layers,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),
                ops.GEMM(
                    "context_proj_gemm",
                    self._num_layers,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "context_ffn1_gemm",
                    self._num_layers,
                    self._inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "context_act",
                    self._num_layers,
                    self._inter_size // tp_size,
                    self._inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "context_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._inter_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                ),
            ]
        )

        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("generation_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "generation_qkv_gemm",
                    self._num_layers,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                ),
                ops.GenerationAttention(
                    "generation_attention",
                    self._num_layers,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                ),
                ops.GEMM(
                    "generation_proj_gemm",
                    self._num_layers,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.ElementWise("generation_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "generation_ffn1_gemm",
                    self._num_layers,
                    self._inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_act",
                    self._num_layers,
                    self._inter_size // tp_size,
                    self._inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "generation_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._inter_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.GEMM(
                    "generation_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                ),
            ]
        )

        # when tp_size=0, the comm part will be 0
        self.context_ops.append(ops.CustomAllReduce("context_ar_1", self._num_layers, h, tp_size))
        self.context_ops.append(ops.CustomAllReduce("context_ar_2", self._num_layers, h, tp_size))
        self.generation_ops.append(ops.CustomAllReduce("generation_ar_1", self._num_layers, h, tp_size))
        self.generation_ops.append(ops.CustomAllReduce("generation_ar_2", self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor, h, pp_size))


class LLAMAModel(BaseModel):
    """
    LLAMA series uses this model impl. Other variants without large difference can use this as well,
    e.g., only positional embedding or activation is different.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention,
    exact match is required. Same for logits_gemm.
    Supports MTP (Multi-Token Prediction) speculative decoding simulation.
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)

        # MTP scale factor: throughput boost / compute overhead
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
        )

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_gpu = self._num_kv_heads_per_gpu
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "context_qkv_gemm",
                    self._num_layers,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                ),
                ops.ContextAttention(
                    "context_attention",
                    self._num_layers,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),
                ops.GEMM(
                    "context_proj_gemm",
                    self._num_layers,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "context_gate_ffn1_gemm",
                    self._num_layers,
                    2 * self._inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "context_act_gate",
                    self._num_layers,
                    2 * self._inter_size // tp_size,
                    self._inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "context_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._inter_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                ),
            ]
        )

        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.ElementWise("generation_add_norm_1", self._num_layers * self._mtp_scale_factor, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "generation_qkv_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                ),
                ops.GenerationAttention(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                ),
                ops.GEMM(
                    "generation_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.ElementWise("generation_add_norm_2", self._num_layers * self._mtp_scale_factor, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "generation_gate_ffn1_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_act_gate",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._inter_size // tp_size,
                    self._inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "generation_ffn2_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._inter_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                ),
            ]
        )

        # when tp_message_size=0, the comm part will be 0
        self.context_ops.append(ops.CustomAllReduce("context_ar_1", self._num_layers, h, tp_size))
        self.context_ops.append(ops.CustomAllReduce("context_ar_2", self._num_layers, h, tp_size))
        self.generation_ops.append(
            ops.CustomAllReduce("generation_ar_1", self._num_layers * self._mtp_scale_factor, h, tp_size)
        )
        self.generation_ops.append(
            ops.CustomAllReduce("generation_ar_2", self._num_layers * self._mtp_scale_factor, h, tp_size)
        )

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))


# mostly for mixtral models
class MOEModel(BaseModel):
    """
    Traditional MoE models uses this model impl: Mixtral, LLAMA4_MOE, etc.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention,
    exact match is required. Same for logits_gemm.
    Supports MTP (Multi-Token Prediction) speculative decoding simulation.
    TODO: redesign shared moe part.
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # MTP scale factor: throughput boost / compute overhead
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
        )

        # make sure the paralel width is same
        assert (
            self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"
        assert self.config.tp_size * self.config.attention_dp_size <= 256, (
            f"moe ep size {self.config.moe_ep_size} * moe tp size {self.config.moe_tp_size} "
            f"should not be larger than 256"
        )

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

        # Validate quantized MoE block size alignment
        self._validate_fp8_block_quantized_moe_config()

        self._power_law_alpha = 1.2

        moe_quant_mode = self.config.moe_quant_mode

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_gpu = self._num_kv_heads_per_gpu
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        if self.architecture == "GptOssForCausalLM":
            attn_scale_factor = 2
            window_size = 128
            self.context_ops.append(
                ops.ContextAttention(
                    "context_attention",
                    self._num_layers / attn_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    window_size,
                    self._head_size,
                )
            )
            self.generation_ops.append(
                ops.GenerationAttention(
                    "generation_attention",
                    self._num_layers / attn_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    window_size,
                    self._head_size,
                )
            )
        else:
            attn_scale_factor = 1

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "context_qkv_gemm",
                    self._num_layers,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                ),
                ops.ContextAttention(
                    "context_attention",
                    self._num_layers / attn_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    head_size=self._head_size,
                ),
                ops.GEMM(
                    "context_proj_gemm",
                    self._num_layers,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
            ]
        )

        # router, only take it into account when num_experts >= 128
        if self._num_experts >= 128:
            self.context_ops.extend(
                [
                    ops.GEMM(
                        "context_router_gemm",
                        self._num_layers,
                        self._num_experts,
                        h,
                        common.GEMMQuantMode.float16,
                    )
                ]
            )

        # dispatch tokens to experts, moe calc and get tokens back
        self.context_ops.extend(
            [
                ops.MoEDispatch(
                    "context_moe_pre_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                ),
                ops.MoE(
                    "context_moe",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                ),
                ops.MoEDispatch(
                    "context_moe_post_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,
                ),
            ]
        )

        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.ElementWise("generation_add_norm_1", self._num_layers * self._mtp_scale_factor, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "generation_qkv_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                ),
                ops.GenerationAttention(
                    "generation_attention",
                    self._num_layers / attn_scale_factor * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    head_size=self._head_size,
                ),
                ops.GEMM(
                    "generation_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.ElementWise("generation_add_norm_2", self._num_layers * self._mtp_scale_factor, 2 * h, 2 * h, 0.8),
            ]
        )

        # router, only take it into account when num_experts >= 128
        if self._num_experts >= 128:
            self.generation_ops.extend(
                [
                    ops.GEMM(
                        "generation_router_gemm",
                        self._num_layers * self._mtp_scale_factor,
                        self._num_experts,
                        h,
                        common.GEMMQuantMode.float16,
                    )
                ]
            )

        # dispatch tokens to experts, moe calc and get tokens back
        self.generation_ops.extend(
            [
                ops.MoEDispatch(
                    "generation_moe_pre_dispatch",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                ),
                ops.MoE(
                    "generation_moe",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                ),
                ops.MoEDispatch(
                    "generation_moe_post_dispatch",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,
                ),
            ]
        )
        # logits gemm
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # # # when tp_size=0, the comm part will be 0
        # self.context_ops.append(ops.CustomAllReduce('context_ar_1', self._num_layers, h, tp_size))
        # self.context_ops.append(ops.CustomAllReduce('context_ar_2', self._num_layers, h, tp_size))
        # self.generation_ops.append(ops.CustomAllReduce('generation_ar_1', self._num_layers, h, tp_size))
        # self.generation_ops.append(ops.CustomAllReduce('generation_ar_2', self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

    def _validate_fp8_block_quantized_moe_config(self) -> None:
        """
        Validate that quantized MoE configuration satisfies block size constraints.

        For fp8_block quantized MoE models, the constraint is:
        (moe_intermediate_size / moe_tp_size) % weight_block_size_n == 0

        This ensures proper alignment for quantized weight blocks.
        """
        # Only validate for fp8_block quantization
        if self.config.moe_quant_mode != common.MoEQuantMode.fp8_block:
            return

        # Load raw model config to get block size
        raw_config = _load_model_config_from_model_path(self.model_path)

        # Get weight_block_size from quantization_config (default to [128, 128])
        default_size = [128, 128]
        weight_block_size = raw_config.get("quantization_config", {}).get("weight_block_size", default_size)[0]

        # Check alignment
        moe_size_per_gpu = self._moe_inter_size // self.config.moe_tp_size
        if (moe_size_per_gpu % weight_block_size) != 0:
            raise ValueError(
                f"Invalid quantized MoE configuration: "
                f"(moe_intermediate_size={self._moe_inter_size} / moe_tp_size={self.config.moe_tp_size}) "
                f"% weight_block_size={weight_block_size} != 0. "
            )


class DeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 uses this model impl.
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # make sure the paralel width is same
        assert (
            self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"
        assert self.config.tp_size * self.config.attention_dp_size <= 256, (
            f"moe ep size {self.config.moe_ep_size} * moe tp size {self.config.moe_tp_size} "
            f"should not be larger than 256"
        )

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

        # used to scale the tpot to reflect mtp effect:
        # 1. mtp will reduce the overall time by expected_tokens_per_step
        # 2. mtp module introduces nextn new transformer layers+linear layers
        #    (we ignore the linear layers for now)
        # 3. special correction in agg step due to we leveraging ctx phase for gen tokens
        #    non-attn part
        # meanwhile, needs to scale the actual bs of generation by nextn,
        # this is covered in inferencesession
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
        )
        self._power_law_alpha = 1.01

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode

        mla_bmm_quant_mode = (
            common.GEMMQuantMode.fp8
            if gemm_quant_mode != common.GEMMQuantMode.float16
            else common.GEMMQuantMode.float16
        )

        h = self._hidden_size  # 7168
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size

        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM("context_downscale_gemm", self._num_layers, 2112, h, gemm_quant_mode),  # on every gpu, fused_a
                ops.GEMM(
                    "context_q_b_proj_gemm",
                    self._num_layers,
                    24576 // tp_size,
                    1536,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "context_kv_b_proj_gemm",
                    self._num_layers,
                    32768 // tp_size,
                    512,
                    gemm_quant_mode,
                ),  # agg ctx attn part
                ops.ContextMLA(
                    "context_attention",
                    self._num_layers,
                    128 // tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),  # agg ctx attn part
                ops.GEMM(
                    "context_proj_gemm", self._num_layers, h, 128 * 128 // tp_size, gemm_quant_mode
                ),  # agg ctx attn part
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
            ]
        )

        # shared moe
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_shared_gate_gemm",
                    self._num_layers,
                    self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "context_shared_ffn1_gemm",
                    self._num_layers,
                    self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "context_shared_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size // tp_size,
                    self._moe_inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "context_shared_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size // tp_size,
                    gemm_quant_mode,
                ),
            ]
        )

        # router gemm, num_experts is large enough, cannot be ignored anymore.
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_router_gemm",
                    self._num_layers,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # dispatch tokens to experts, pre-dispatch
        self.context_ops.extend(
            [
                ops.MoEDispatch(
                    "context_moe_pre_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                )
            ]
        )

        # moe part
        self.context_ops.extend(
            [
                ops.MoE(
                    "context_moe",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                )
            ]
        )

        # dispatch tokens to experts, post-dispatch
        self.context_ops.extend(
            [
                ops.MoEDispatch(
                    "context_moe_post_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,
                )
            ]
        )

        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )
        #####generation part, only generation part is scaled by mtp_scale_factor
        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.ElementWise(
                    "generation_add_norm_1",
                    self._num_layers * self._mtp_scale_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
                ops.GEMM(
                    "generation_downscale_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    2112,
                    h,
                    gemm_quant_mode,
                ),  # on every gpu
                ops.GEMM(
                    "generation_q_b_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    24576 // tp_size,
                    1536,
                    gemm_quant_mode,
                ),
                ops.MLABmm(
                    "generation_bmm_pre",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    mla_bmm_quant_mode,
                    if_pre=True,
                ),  # agg gen attn part
                ops.GenerationMLA(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor,
                    128 // tp_size,
                    kvcache_quant_mode,
                ),  # agg gen attn part
                ops.MLABmm(
                    "generation_bmm_post",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    mla_bmm_quant_mode,
                    if_pre=False,
                ),  # agg gen attn part
                ops.GEMM(
                    "generation_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    h // tp_size,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_add_norm_2",
                    self._num_layers * self._mtp_scale_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
            ]
        )

        # shared moe
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_shared_gate_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "generation_shared_ffn1_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_shared_act_gate",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size // tp_size,
                    self._moe_inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "generation_shared_ffn2_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size // tp_size,
                    gemm_quant_mode,
                ),
            ]
        )

        # router gemm, num_experts is large enough, cannot be ignored anymore.
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_router_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # dispatch tokens to experts, pre-dispatch
        self.generation_ops.extend(
            [
                ops.MoEDispatch(
                    "generation_moe_pre_dispatch",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                )
            ]
        )

        # moe part
        self.generation_ops.extend(
            [
                ops.MoE(
                    "generation_moe",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                ),
            ]
        )

        # dispatch tokens to experts, post-dispatch
        self.generation_ops.extend(
            [
                ops.MoEDispatch(
                    "generation_moe_post_dispatch",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,
                )
            ]
        )

        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # when tp_size=0, the comm part will be 0
        # self.context_ops.append(ops.CustomAllReduce('context_ar_1', self._num_layers, h, tp_size))
        # self.context_ops.append(ops.CustomAllReduce('context_ar_2', self._num_layers, h, tp_size))
        # self.generation_ops.append(
        #     ops.CustomAllReduce('generation_ar_1', self._num_layers*self._mtp_scale_factor, h, tp_size)
        # )
        # self.generation_ops.append(
        #     ops.CustomAllReduce('generation_ar_2', self._num_layers*self._mtp_scale_factor, h, tp_size)
        # )

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

        # TODO
        # a lot of quantization ops


class TrtllmWideEPDeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 with TensorRT-LLM WideEP support.

    This model enables WideEP (Wide Expert Parallelism) for TensorRT-LLM backend:
    - MoE computation uses WideEP path (query_wideep_moe_compute) with configurable EPLB modes
    - All2All communication uses WideEP path (query_wideep_alltoall with auto kernel selection)

    Token handling (handled in MoE/MoEDispatch query methods):
    - MoE compute: total tokens (x * attention_dp_size)
    - All2All communication: per-DP tokens (x)

    Kernel auto-selection:
    - MoE kernel: deepgemm (SM>=100 + fp8_block) or moe_torch_flow (default)
    - All2All kernel: MnnvlMoe (SM>=100), DeepEP/DeepEPLowLatency (SM>=90), NCCL (fallback)
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # make sure the parallel width is same
        assert (
            self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"
        assert self.config.tp_size * self.config.attention_dp_size <= 256, (
            f"moe ep size {self.config.moe_ep_size} * moe tp size {self.config.moe_tp_size} "
            f"should not be larger than 256"
        )

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

        # MTP scale factor for generation phase
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
        )
        self._power_law_alpha = 1.01

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode

        mla_bmm_quant_mode = (
            common.GEMMQuantMode.fp8
            if gemm_quant_mode != common.GEMMQuantMode.float16
            else common.GEMMQuantMode.float16
        )

        h = self._hidden_size  # 7168
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size

        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode

        # WideEP workload distribution
        # - EPLB off: "power_law_1.01" (no _eplb suffix)
        # - EPLB on/redundant: "power_law_1.01_eplb" (with _eplb suffix)
        eplb_enabled = self.config.enable_eplb
        if self.config.workload_distribution == "power_law":
            if eplb_enabled:
                workload_distribution = f"{self.config.workload_distribution}_{self._power_law_alpha}_eplb"
            else:
                workload_distribution = f"{self.config.workload_distribution}_{self._power_law_alpha}"
        else:
            workload_distribution = self.config.workload_distribution

        # ===================== WideEP Configuration Validation =====================
        # Based on TensorRT-LLM WideEPMoE constraints (fused_moe_wide_ep.py)

        # 1. Attention DP must be enabled for WideEP
        if attention_dp_size <= 1:
            raise ValueError(
                f"WideEP requires attention_dp_size > 1, got {attention_dp_size}. "
                "Attention DP should be used with WideEP."
            )

        # 2. EP size must be > 1 for WideEP (parallel_size > 1)
        if moe_ep_size <= 1:
            raise ValueError(
                f"WideEP requires moe_ep_size > 1, got {moe_ep_size}. "
                "WideEP should only be enabled with parallel_size > 1."
            )

        # 3. EP size must be > top_k for AlltoAll to be effective
        # FIXME: this warning should make the comm mode fallback to NCCL!!
        if moe_ep_size <= topk:
            logger.warning(
                f"moe_ep_size ({moe_ep_size}) <= top_k ({topk}), "
                "AlltoAll communication will be disabled. Consider increasing moe_ep_size."
            )

        # 4. num_slots validation
        wideep_num_slots = self.config.wideep_num_slots if self.config.wideep_num_slots else num_experts

        # num_slots must be >= num_experts
        if wideep_num_slots < num_experts:
            raise ValueError(
                f"wideep_num_slots ({wideep_num_slots}) must be >= num_experts ({num_experts}). "
                "There should be at least num_experts slots in the model engine."
            )

        # When EPLB is off, num_slots must equal num_experts
        if not eplb_enabled and wideep_num_slots != num_experts:
            raise ValueError(
                f"When enable_eplb=False, wideep_num_slots ({wideep_num_slots}) must equal "
                f"num_experts ({num_experts}). Redundant slots require EPLB to be enabled."
            )

        # ===================== Context Phase =====================
        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
                ops.GEMM("context_downscale_gemm", self._num_layers, 2112, h, gemm_quant_mode),
                ops.GEMM(
                    "context_q_b_proj_gemm",
                    self._num_layers,
                    24576 // tp_size,
                    1536,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "context_kv_b_proj_gemm",
                    self._num_layers,
                    32768 // tp_size,
                    512,
                    gemm_quant_mode,
                ),
                ops.ContextMLA(
                    "context_attention",
                    self._num_layers,
                    128 // tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                ),
                ops.GEMM("context_proj_gemm", self._num_layers, h, 128 * 128 // tp_size, gemm_quant_mode),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8),
            ]
        )

        # shared moe
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_shared_gate_gemm",
                    self._num_layers,
                    self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "context_shared_ffn1_gemm",
                    self._num_layers,
                    self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "context_shared_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size // tp_size,
                    self._moe_inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "context_shared_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size // tp_size,
                    gemm_quant_mode,
                ),
            ]
        )

        # router gemm
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_router_gemm",
                    self._num_layers,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # WideEP: dispatch tokens to experts, pre-dispatch (prepare + dispatch)
        self.context_ops.extend(
            [
                ops.TrtLLMWideEPMoEDispatch(
                    "context_moe_pre_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,  # pre_dispatch
                    quant_mode=moe_quant_mode,
                )
            ]
        )

        # WideEP: MoE computation with EPLB support
        self.context_ops.extend(
            [
                ops.TrtLLMWideEPMoE(
                    "context_moe",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                    num_slots=wideep_num_slots,
                )
            ]
        )

        # WideEP: dispatch tokens to experts, post-dispatch (combine)
        self.context_ops.extend(
            [
                ops.TrtLLMWideEPMoEDispatch(
                    "context_moe_post_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,  # post_dispatch (combine)
                    quant_mode=moe_quant_mode,
                )
            ]
        )

        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # ===================== Generation Phase =====================
        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.ElementWise(
                    "generation_add_norm_1",
                    self._num_layers * self._mtp_scale_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
                ops.GEMM(
                    "generation_downscale_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    2112,
                    h,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "generation_q_b_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    24576 // tp_size,
                    1536,
                    gemm_quant_mode,
                ),
                ops.MLABmm(
                    "generation_bmm_pre",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    mla_bmm_quant_mode,
                    if_pre=True,
                ),
                ops.GenerationMLA(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor,
                    128 // tp_size,
                    kvcache_quant_mode,
                ),
                ops.MLABmm(
                    "generation_bmm_post",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    mla_bmm_quant_mode,
                    if_pre=False,
                ),
                ops.GEMM(
                    "generation_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    h // tp_size,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_add_norm_2",
                    self._num_layers * self._mtp_scale_factor,
                    2 * h,
                    2 * h,
                    0.8,
                ),
            ]
        )

        # shared moe
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_shared_gate_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.GEMM(
                    "generation_shared_ffn1_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._moe_inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_shared_act_gate",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size // tp_size,
                    self._moe_inter_size // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "generation_shared_ffn2_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size // tp_size,
                    gemm_quant_mode,
                ),
            ]
        )

        # router gemm
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_router_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # WideEP: dispatch tokens to experts, pre-dispatch
        self.generation_ops.extend(
            [
                ops.TrtLLMWideEPMoEDispatch(
                    "generation_moe_pre_dispatch",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,  # pre_dispatch
                    quant_mode=moe_quant_mode,
                )
            ]
        )

        # WideEP: MoE computation with EPLB support
        self.generation_ops.extend(
            [
                ops.TrtLLMWideEPMoE(
                    "generation_moe",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    workload_distribution,
                    attention_dp_size,
                    num_slots=wideep_num_slots,
                ),
            ]
        )

        # WideEP: dispatch tokens to experts, post-dispatch (combine)
        self.generation_ops.extend(
            [
                ops.TrtLLMWideEPMoEDispatch(
                    "generation_moe_post_dispatch",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    False,  # post_dispatch (combine)
                    quant_mode=moe_quant_mode,
                )
            ]
        )

        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            ]
        )

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))


class WideEPDeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 disaggregated model for SGLang backend.
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
        )

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size

        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        gemm_quant_mode = self.config.gemm_quant_mode
        moe_backend = self.config.moe_backend
        attn_backend = self.config.attention_backend

        self._power_law_alpha_prefill = 0.6 if self.config.enable_eplb else 1.01
        self._power_law_alpha_decode = 1.01

        context_workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha_prefill}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        generation_workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha_decode}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        sms = self.config.sms

        # context mla attention
        self.context_ops.extend(
            [
                *(
                    [
                        ops.NCCL(
                            "context_tp_all_gather",
                            self._num_layers,
                            "all_gather",
                            h,
                            tp_size,
                            common.CommQuantMode.half,
                        )
                    ]
                    if tp_size > 1
                    else []
                ),
                ops.WideEPContextMLA(
                    "context_attention",
                    self._num_layers,
                    tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    attn_backend,
                ),
                *(
                    [
                        ops.NCCL(
                            "context_tp_reduce_scatter",
                            self._num_layers,
                            "reduce_scatter",
                            h,
                            tp_size,
                            common.CommQuantMode.half,
                        )
                    ]
                    if tp_size > 1
                    else []
                ),
            ]
        )

        # shared expert
        # TODO: support shared expert TP
        self.context_ops.extend(
            [
                ops.GEMM(
                    "context_gate_ffn1_gemm",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    h,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
                ),
                ops.ElementWise(
                    "context_act_gate",
                    self._num_layers,
                    2 * self._moe_inter_size,
                    self._moe_inter_size,
                    0.8,
                    scale_num_tokens=tp_size,
                ),
                ops.GEMM(
                    "context_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    gemm_quant_mode,
                    scale_num_tokens=tp_size,
                ),
            ]
        )

        # dispatch tokens to experts
        self.context_ops.extend(
            [
                ops.MoEDispatch(
                    "context_moe_pre_dispatch",
                    self._num_layers,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                    sms=sms,
                    moe_backend=moe_backend,
                    is_context=True,
                    scale_num_tokens=tp_size,
                )
            ]
        )

        # moe computation
        self.context_ops.extend(
            [
                ops.MoE(
                    "context_moe",
                    self._num_layers,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    context_workload_distribution,
                    attention_dp_size,
                    is_context=True,
                    moe_backend=moe_backend,
                    enable_eplb=self.config.enable_eplb,
                )
            ]
        )

        # generation mla attention
        self.generation_ops.extend(
            [
                ops.WideEPGenerationMLA(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor,
                    tp_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    attn_backend,
                )
            ]
        )

        # shared expert
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_gate_ffn1_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "generation_act_gate",
                    self._num_layers * self._mtp_scale_factor,
                    2 * self._moe_inter_size,
                    self._moe_inter_size,
                    0.8,
                ),
                ops.GEMM(
                    "generation_ffn2_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size,
                    gemm_quant_mode,
                ),
            ]
        )

        # dispatch tokens to experts
        self.generation_ops.extend(
            [
                ops.MoEDispatch(
                    "generation_moe_pre_dispatch",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    attention_dp_size,
                    True,
                    sms=sms,
                    moe_backend=moe_backend,
                    is_context=False,
                )
            ]
        )

        # moe computation
        self.generation_ops.extend(
            [
                ops.MoE(
                    "generation_moe",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._moe_inter_size,
                    self._topk,
                    self._num_experts,
                    moe_tp_size,
                    moe_ep_size,
                    moe_quant_mode,
                    generation_workload_distribution,
                    attention_dp_size,
                    is_context=False,
                    moe_backend=moe_backend,
                    enable_eplb=False,
                )
            ]
        )


class NemotronNas(BaseModel):
    """
    NemotronNas model implementation with configurable block architectures.

    This model supports flexible transformer architectures where each block can have
    different configurations for attention and feed-forward network components.
    The model does not support multi-token prediction (mtp).

    refer to "PUZZLE: DISTILLATION-BASED NAS FOR INFERENCE-OPTIMIZED LLMS"(
    https://arxiv.org/pdf/2411.19146) for the details of creaing this type of
    models
    """

    def __init__(self, *args):
        """
        Initialize NemotronNas model with configurable transformer blocks.

        Args:
            *args: Arguments passed to BaseModel constructor including:
                - model_path (str): Name of the model
                - model_family (str): Model family (should be "NEMOTRONNAS")
                - num_layers (int): Number of transformer layers
                - num_heads (int): Number of attention heads
                - num_kv_heads (int): Number of key-value heads (0 for this model, will set using
                  block_configs)
                - head_size (int): Size of each attention head
                - hidden_size (int): Hidden dimension size
                - inter_size (int): Intermediate size (0 for this model, will set using
                  block_configs)
                - vocab_size (int): Vocabulary size
                - context_length (int): Maximum context length
                - model_config (ModelConfig): Model configuration object
        Raises:
            AssertionError: If model configuration specifies mtp (nextn != 0), as only DS V3
                supports mtp
        """
        super().__init__(*args)

        assert self._nextn == 0, "Only DS V3 supports mtp"

    @property
    def context_ops(self):
        """
        Get the context(prefill) processing operations pipeline.

        Returns:
            List[ops.Operation]: List of operations for processing context
            sequences, including:
                - embedding,
                - attention blocks,
                - FFN blocks,
                - P2P communication,
                - all reduce communication
                - logits computation.
        """
        return self._context_ops

    @context_ops.setter
    def context_ops(self, puzzle_block_configs: list[common.BlockConfig]):
        """
        Set the context(prefill) processing operations pipeline based on block configurations.

        Constructs a pipeline of operations for processing input context by creating operations
        for each configured transformer block. The pipeline includes embedding lookup,
        transformer blocks (with optional attention and FFN components), pipeline parallel
        communication, and final logits computation.

        Args:
            puzzle_block_configs (List[BlockConfig]): List of block configurations where each
                BlockConfig specifies:
                - num_inst (int): Number of instances of this block type
                - attn_no_op (bool): Whether to skip attention operations for this block
                - attn_n_heads_in_group (int): Number of attention heads in group
                  (used if attn_no_op is False)
                - ffn_no_op (bool): Whether to skip FFN operations for this block
                - ffn_ffn_mult (float): FFN size multiplier relative to hidden size
                  (used if ffn_no_op is False)

        """
        self._context_ops = []
        if puzzle_block_configs:
            h = self._hidden_size
            tp_size = self.config.tp_size
            pp_size = self.config.pp_size
            gemm_quant_mode = self.config.gemm_quant_mode
            kvcache_quant_mode = self.config.kvcache_quant_mode
            fmha_quant_mode = self.config.fmha_quant_mode
            pp_scale_factor = pp_size - 1
            self._context_ops.append(ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3))
            for b in puzzle_block_configs:
                count = b.num_inst
                if not b.attn_no_op:
                    num_kv_heads = self._num_heads // b.attn_n_heads_in_group
                    num_kv_heads_per_gpu = (num_kv_heads + tp_size - 1) // tp_size
                    self._context_ops.extend(
                        [
                            ops.ElementWise("context_add_norm_1", count, 2 * h, 2 * h, 0.8),
                            ops.GEMM(
                                "context_qkv_gemm",
                                count,
                                self._num_heads * self._head_size // tp_size
                                + self._head_size * num_kv_heads_per_gpu * 2,
                                h,
                                gemm_quant_mode,
                            ),
                            ops.ContextAttention(
                                "context_attention",
                                count,
                                self._num_heads // tp_size,
                                num_kv_heads_per_gpu,
                                kvcache_quant_mode,
                                fmha_quant_mode,
                            ),
                            ops.GEMM(
                                "context_proj_gemm",
                                count,
                                h,
                                self._num_heads * self._head_size // tp_size,
                                gemm_quant_mode,
                            ),
                            ops.CustomAllReduce("context_ar_1", count, h, tp_size),
                        ]
                    )
                if not b.ffn_no_op:
                    inter_size = self._ffn_mult_to_intermediate_size(b.ffn_ffn_mult)
                    self._context_ops.extend(
                        [
                            ops.ElementWise("context_add_norm_2", count, 2 * h, 2 * h, 0.8),
                            ops.GEMM(
                                "context_gate_ffn1_gemm",
                                count,
                                2 * inter_size // tp_size,
                                h,
                                gemm_quant_mode,
                            ),
                            ops.ElementWise(
                                "context_act_gate",
                                count,
                                2 * inter_size // tp_size,
                                inter_size // tp_size,
                                0.8,
                            ),
                            ops.GEMM(
                                "context_ffn2_gemm",
                                count,
                                h,
                                inter_size // tp_size,
                                gemm_quant_mode,
                            ),
                            ops.CustomAllReduce("context_ar_2", count, h, tp_size),
                        ]
                    )
            self._context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
            self._context_ops.append(
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            )

    @property
    def generation_ops(self):
        """
        Get the generation (decoding) operations pipeline.

        Returns:
            List[ops.Operation]: List of operations for the decoding phase
            including:
                - embedding,
                - attention blocks,
                - FFN blocks,
                - P2P communication,
                - all reduce communication
                - logits computation.
        """
        return self._generation_ops

    @generation_ops.setter
    def generation_ops(self, puzzle_block_configs: list[common.BlockConfig]):
        """
        Set the generation (decoding) operations pipeline based on block configurations.

        Constructs a pipeline of operations for autoregressive generation by creating operations
        for each configured transformer block. Similar to context_ops but uses generation-specific
        attention operations that support KV-cache for efficient autoregressive decoding.

        Args:
            puzzle_block_configs (List[BlockConfig]): List of block configurations where each
                BlockConfig specifies:
                - num_inst (int): Number of instances of this block type
                - attn_no_op (bool): Whether to skip attention operations for this block
                - attn_n_heads_in_group (int): Number of attention heads in group
                  (used if attn_no_op is False)
                - ffn_no_op (bool): Whether to skip FFN operations for this block
                - ffn_ffn_mult (float): FFN size multiplier relative to hidden size
                  (used if ffn_no_op is False)
        """
        self._generation_ops = []
        if puzzle_block_configs:
            h = self._hidden_size
            tp_size = self.config.tp_size
            pp_size = self.config.pp_size
            gemm_quant_mode = self.config.gemm_quant_mode
            kvcache_quant_mode = self.config.kvcache_quant_mode
            pp_scale_factor = pp_size - 1
            self._generation_ops.append(ops.Embedding("generation_embedding", 1, self._vocab_size, h, 0.3))
            for b in puzzle_block_configs:
                count = b.num_inst
                if not b.attn_no_op:
                    num_kv_heads = self._num_heads // b.attn_n_heads_in_group
                    num_kv_heads_per_gpu = (num_kv_heads + tp_size - 1) // tp_size
                    self._generation_ops.extend(
                        [
                            ops.ElementWise("generation_add_norm_1", count, 2 * h, 2 * h, 0.8),
                            ops.GEMM(
                                "generation_qkv_gemm",
                                count,
                                self._num_heads * self._head_size // tp_size
                                + self._head_size * num_kv_heads_per_gpu * 2,
                                h,
                                gemm_quant_mode,
                            ),
                            ops.GenerationAttention(
                                "generation_attention",
                                count,
                                self._num_heads // tp_size,
                                num_kv_heads_per_gpu,
                                kvcache_quant_mode,
                            ),
                            ops.GEMM(
                                "generation_proj_gemm",
                                count,
                                h,
                                self._num_heads * self._head_size // tp_size,
                                gemm_quant_mode,
                            ),
                            ops.CustomAllReduce("generation_ar_1", count, h, tp_size),
                        ]
                    )
                if not b.ffn_no_op:
                    inter_size = self._ffn_mult_to_intermediate_size(b.ffn_ffn_mult)
                    self._generation_ops.extend(
                        [
                            ops.ElementWise("generation_add_norm_2", count, 2 * h, 2 * h, 0.8),
                            ops.GEMM(
                                "generation_gate_ffn1_gemm",
                                count,
                                2 * inter_size // tp_size,
                                h,
                                gemm_quant_mode,
                            ),
                            ops.ElementWise(
                                "generation_act_gate",
                                count,
                                2 * inter_size // tp_size,
                                inter_size // tp_size,
                                0.8,
                            ),
                            ops.GEMM(
                                "generation_ffn2_gemm",
                                count,
                                h,
                                inter_size // tp_size,
                                gemm_quant_mode,
                            ),
                            ops.CustomAllReduce("generation_ar_2", count, h, tp_size),
                        ]
                    )
            self._generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor, h, pp_size))
            self._generation_ops.append(
                ops.GEMM(
                    "generation_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.float16,
                )
            )

    def _ffn_mult_to_intermediate_size(self, ffn_mult: float) -> int:
        """
        Rule used to convert ffn_mult into the intermediate size of the ffn GEMM

        Args:
            ffn_mult (float): FFN size multiplier relative to hidden size
        """
        # conversion codes adopted from
        # https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1/blob/main/modeling_decilm.py
        inter_size = int(2 * ffn_mult * self._hidden_size / 3)
        if inter_size % 256 == 0:
            return inter_size
        return inter_size + 256 - (inter_size % 256)


class NemotronHModel(BaseModel):
    """
    NemotronH hybrid model implementation (Mamba + MoE + Transformer).

    This model supports the hybrid architecture where each layer can be one of:
    - 'M': Mamba2 layer (state-space model)
    - 'E': MoE layer (Mixture of Experts with shared expert)
    - '*': Transformer layer (standard attention)
    - '-': MLP layer (dense feed-forward)

    The layer sequence is defined by the `hybrid_override_pattern` string in NemotronHConfig.
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, "NemotronH does not support mtp"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._hybrid_config: common.NemotronHConfig | None = None
        self._power_law_alpha = 1.01  # follow DeepSeek MoE

    def set_hybrid_config(self, hybrid_config: common.NemotronHConfig) -> None:
        """
        Set the hybrid layer configuration and build operation pipelines.

        Args:
            hybrid_config: NemotronHConfig containing hybrid_override_pattern and layer parameters
        """
        self._hybrid_config = hybrid_config
        self._build_context_ops()
        self._build_generation_ops()

    def _count_layer_types(self) -> dict[str, int]:
        """Count occurrences of each layer type in the pattern."""
        pattern = self._hybrid_config.hybrid_override_pattern
        return {
            "M": pattern.count("M"),
            "E": pattern.count("E"),
            "*": pattern.count("*"),
            "-": pattern.count("-"),
        }

    def _build_context_ops(self) -> None:
        """Build the context (prefill) operations pipeline based on hybrid pattern."""
        if not self._hybrid_config:
            return

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        layer_counts = self._count_layer_types()
        cfg = self._hybrid_config

        # Use base model parameters for standard fields
        num_kv_heads_per_gpu = (self._num_kv_heads + tp_size - 1) // tp_size

        self.context_ops = []

        # Embedding
        self.context_ops.append(ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3))

        # Mamba layers (M): norm, in_proj GEMM, conv1d, ssm, out_proj GEMM, ar
        if layer_counts["M"] > 0:
            count = layer_counts["M"]
            nheads_per_gpu = cfg.mamba_num_heads // tp_size
            d_inner_per_gpu = nheads_per_gpu * cfg.mamba_head_dim
            n_groups_per_gpu = cfg.n_groups // tp_size
            in_proj_out_per_gpu = 2 * d_inner_per_gpu + 2 * n_groups_per_gpu * cfg.ssm_state_size + nheads_per_gpu
            self.context_ops.extend(
                [
                    ops.ElementWise("context_mamba_norm", count, 2 * h, 2 * h, 0.8),
                    ops.GEMM(
                        "context_mamba_in_proj_gemm",
                        count,
                        in_proj_out_per_gpu,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.Mamba2Kernel(
                        "context_mamba_conv1d",
                        count,
                        "causal_conv1d_fn",
                        "context",
                        hidden_size=h,
                        nheads=cfg.mamba_num_heads,
                        head_dim=cfg.mamba_head_dim,
                        d_state=cfg.ssm_state_size,
                        d_conv=cfg.conv_kernel,
                        n_groups=cfg.n_groups,
                        chunk_size=cfg.chunk_size,
                    ),
                    ops.Mamba2Kernel(
                        "context_mamba_ssm",
                        count,
                        "mamba_chunk_scan_combined",
                        "context",
                        hidden_size=h,
                        nheads=cfg.mamba_num_heads,
                        head_dim=cfg.mamba_head_dim,
                        d_state=cfg.ssm_state_size,
                        d_conv=cfg.conv_kernel,
                        n_groups=cfg.n_groups,
                        chunk_size=cfg.chunk_size,
                    ),
                    ops.GEMM(
                        "context_mamba_out_proj_gemm",
                        count,
                        h,
                        d_inner_per_gpu,
                        gemm_quant_mode,
                    ),
                    ops.CustomAllReduce("context_mamba_ar", count, h, tp_size),
                ]
            )

        # Transformer layers (*)
        if layer_counts["*"] > 0:
            count = layer_counts["*"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_attn_norm", count, 2 * h, 2 * h, 0.8),
                    ops.GEMM(
                        "context_qkv_gemm",
                        count,
                        self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ContextAttention(
                        "context_attention",
                        count,
                        self._num_heads // tp_size,
                        num_kv_heads_per_gpu,
                        kvcache_quant_mode,
                        fmha_quant_mode,
                        head_size=self._head_size,
                    ),
                    ops.GEMM(
                        "context_proj_gemm",
                        count,
                        h,
                        self._num_heads * self._head_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    ),
                    ops.CustomAllReduce("context_attn_ar", count, h, tp_size),
                ]
            )

        # MoE layers (E)
        if layer_counts["E"] > 0:
            count = layer_counts["E"]
            # Pre-norm for MoE
            self.context_ops.append(ops.ElementWise("context_moe_norm", count, 2 * h, 2 * h, 0.8))

            # Shared expert (always runs in parallel)
            # NemotronH uses simple MLP (not gated): up_proj -> relu2 -> down_proj
            # See TensorRT-LLM modeling_nemotron_h.py NemotronHMLP class
            self.context_ops.extend(
                [
                    ops.GEMM(
                        "context_shared_up_gemm",
                        count,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ElementWise(
                        "context_shared_relu2",
                        count,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        0.8,
                    ),
                    ops.GEMM(
                        "context_shared_down_gemm",
                        count,
                        h,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    ),
                ]
            )

            # Router GEMM
            self.context_ops.append(
                ops.GEMM(
                    "context_router_gemm",
                    count,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.float16,
                )
            )

            # MoE dispatch and compute
            self.context_ops.extend(
                [
                    ops.MoEDispatch(
                        "context_moe_pre_dispatch",
                        count,
                        h,
                        self._topk,
                        self._num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        attention_dp_size,
                        True,
                    ),
                    ops.MoE(
                        "context_moe",
                        count,
                        h,
                        self._moe_inter_size,
                        self._topk,
                        self._num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        moe_quant_mode,
                        workload_distribution,
                        attention_dp_size,
                        is_gated=False,  # NemotronH uses Relu2 (non-gated)
                    ),
                    ops.MoEDispatch(
                        "context_moe_post_dispatch",
                        count,
                        h,
                        self._topk,
                        self._num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        attention_dp_size,
                        False,
                    ),
                    # TRT-LLM does allreduce after combining routed + shared outputs when TP>1
                    ops.CustomAllReduce("context_moe_ar", count, h, tp_size),
                ]
            )

        # MLP layers (-) - not present in Nemotron-3 Nano but in NemotronH model
        if layer_counts["-"] > 0:
            count = layer_counts["-"]
            # NemotronH MLP is non-gated: up_proj -> relu2 -> down_proj
            # See TensorRT-LLM modeling_nemotron_h.py NemotronHMLP class
            self.context_ops.extend(
                [
                    ops.ElementWise("context_mlp_norm", count, 2 * h, 2 * h, 0.8),
                    ops.GEMM(
                        "context_mlp_up_gemm",
                        count,
                        self._inter_size // tp_size,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ElementWise(
                        "context_mlp_relu2",
                        count,
                        self._inter_size // tp_size,
                        self._inter_size // tp_size,
                        0.8,
                    ),
                    ops.GEMM(
                        "context_mlp_down_gemm",
                        count,
                        h,
                        self._inter_size // tp_size,
                        gemm_quant_mode,
                    ),
                    ops.CustomAllReduce("context_mlp_ar", count, h, tp_size),
                ]
            )

        # P2P communication for PP
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))

        # Logits GEMM
        self.context_ops.append(
            ops.GEMM(
                "context_logits_gemm",
                1,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.float16,
            )
        )

    def _build_generation_ops(self) -> None:
        """Build the generation (decoding) operations pipeline based on hybrid pattern."""
        if not self._hybrid_config:
            return

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )

        layer_counts = self._count_layer_types()
        cfg = self._hybrid_config

        # Use base model parameters for standard fields
        num_kv_heads_per_gpu = (self._num_kv_heads + tp_size - 1) // tp_size

        self.generation_ops = []

        # Embedding
        self.generation_ops.append(ops.Embedding("generation_embedding", 1, self._vocab_size, h, 0.3))

        # Mamba layers (M): norm, in_proj GEMM, conv1d, ssm, out_proj GEMM, ar
        if layer_counts["M"] > 0:
            count = layer_counts["M"]
            nheads_per_gpu = cfg.mamba_num_heads // tp_size
            d_inner_per_gpu = nheads_per_gpu * cfg.mamba_head_dim
            n_groups_per_gpu = cfg.n_groups // tp_size
            in_proj_out_per_gpu = 2 * d_inner_per_gpu + 2 * n_groups_per_gpu * cfg.ssm_state_size + nheads_per_gpu
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_mamba_norm", count, 2 * h, 2 * h, 0.8),
                    ops.GEMM(
                        "generation_mamba_in_proj_gemm",
                        count,
                        in_proj_out_per_gpu,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.Mamba2Kernel(
                        "generation_mamba_conv1d",
                        count,
                        "causal_conv1d_update",
                        "generation",
                        hidden_size=h,
                        nheads=cfg.mamba_num_heads,
                        head_dim=cfg.mamba_head_dim,
                        d_state=cfg.ssm_state_size,
                        d_conv=cfg.conv_kernel,
                        n_groups=cfg.n_groups,
                        chunk_size=cfg.chunk_size,
                    ),
                    ops.Mamba2Kernel(
                        "generation_mamba_ssm",
                        count,
                        "selective_state_update",
                        "generation",
                        hidden_size=h,
                        nheads=cfg.mamba_num_heads,
                        head_dim=cfg.mamba_head_dim,
                        d_state=cfg.ssm_state_size,
                        d_conv=cfg.conv_kernel,
                        n_groups=cfg.n_groups,
                        chunk_size=cfg.chunk_size,
                    ),
                    ops.GEMM(
                        "generation_mamba_out_proj_gemm",
                        count,
                        h,
                        d_inner_per_gpu,
                        gemm_quant_mode,
                    ),
                    ops.CustomAllReduce("generation_mamba_ar", count, h, tp_size),
                ]
            )

        # Transformer layers (*)
        if layer_counts["*"] > 0:
            count = layer_counts["*"]
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_attn_norm", count, 2 * h, 2 * h, 0.8),
                    ops.GEMM(
                        "generation_qkv_gemm",
                        count,
                        self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.GenerationAttention(
                        "generation_attention",
                        count,
                        self._num_heads // tp_size,
                        num_kv_heads_per_gpu,
                        kvcache_quant_mode,
                        head_size=self._head_size,
                    ),
                    ops.GEMM(
                        "generation_proj_gemm",
                        count,
                        h,
                        self._num_heads * self._head_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    ),
                    ops.CustomAllReduce("generation_attn_ar", count, h, tp_size),
                ]
            )

        # MoE layers (E)
        if layer_counts["E"] > 0:
            count = layer_counts["E"]
            # Pre-norm for MoE
            self.generation_ops.append(ops.ElementWise("generation_moe_norm", count, 2 * h, 2 * h, 0.8))

            # Shared expert (always runs in parallel)
            # NemotronH uses simple MLP (not gated): up_proj -> relu2 -> down_proj
            # See TensorRT-LLM modeling_nemotron_h.py NemotronHMLP class
            self.generation_ops.extend(
                [
                    ops.GEMM(
                        "generation_shared_up_gemm",
                        count,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ElementWise(
                        "generation_shared_relu2",
                        count,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        0.8,
                    ),
                    ops.GEMM(
                        "generation_shared_down_gemm",
                        count,
                        h,
                        cfg.moe_shared_expert_intermediate_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    ),
                ]
            )

            # Router GEMM
            self.generation_ops.append(
                ops.GEMM(
                    "generation_router_gemm",
                    count,
                    self._num_experts,
                    h,
                    common.GEMMQuantMode.float16,
                )
            )

            # MoE dispatch and compute
            self.generation_ops.extend(
                [
                    ops.MoEDispatch(
                        "generation_moe_pre_dispatch",
                        count,
                        h,
                        self._topk,
                        self._num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        attention_dp_size,
                        True,
                    ),
                    ops.MoE(
                        "generation_moe",
                        count,
                        h,
                        self._moe_inter_size,
                        self._topk,
                        self._num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        moe_quant_mode,
                        workload_distribution,
                        attention_dp_size,
                        is_gated=False,  # NemotronH uses Relu2 (non-gated)
                    ),
                    ops.MoEDispatch(
                        "generation_moe_post_dispatch",
                        count,
                        h,
                        self._topk,
                        self._num_experts,
                        moe_tp_size,
                        moe_ep_size,
                        attention_dp_size,
                        False,
                    ),
                    # TRT-LLM does allreduce after combining routed + shared outputs when TP>1
                    ops.CustomAllReduce("generation_moe_ar", count, h, tp_size),
                ]
            )

        # MLP layers (-)
        if layer_counts["-"] > 0:
            count = layer_counts["-"]
            # NemotronH MLP is non-gated: up_proj -> relu2 -> down_proj
            # See TensorRT-LLM modeling_nemotron_h.py NemotronHMLP class
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_mlp_norm", count, 2 * h, 2 * h, 0.8),
                    ops.GEMM(
                        "generation_mlp_up_gemm",
                        count,
                        self._inter_size // tp_size,
                        h,
                        gemm_quant_mode,
                    ),
                    ops.ElementWise(
                        "generation_mlp_relu2",
                        count,
                        self._inter_size // tp_size,
                        self._inter_size // tp_size,
                        0.8,
                    ),
                    ops.GEMM(
                        "generation_mlp_down_gemm",
                        count,
                        h,
                        self._inter_size // tp_size,
                        gemm_quant_mode,
                    ),
                    ops.CustomAllReduce("generation_mlp_ar", count, h, tp_size),
                ]
            )

        # P2P communication for PP
        pp_scale_factor = pp_size - 1
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor, h, pp_size))

        # Logits GEMM
        self.generation_ops.append(
            ops.GEMM(
                "generation_logits_gemm",
                1,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.float16,
            )
        )


if __name__ == "__main__":
    # TODO, move to unit tests
    model = get_model(
        "DEEPSEEK_V3",
        config.ModelConfig(
            tp_size=1,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.fp8_ootb,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.fp8,
            moe_quant_mode=common.MoEQuantMode.w4afp8,
            nextn=2,
            nextn_accept_rates=[0.5, 0.5],
        ),
        common.BackendName.trtllm.value,
    )
    print(model.context_ops)
    print(model.generation_ops)
    print(model.config)
