# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, register_model
from aiconfigurator.sdk.models.helpers import calc_expectation
from aiconfigurator.sdk.utils import _load_model_config_from_model_path


@register_model("MOE")
class MOEModel(BaseModel):
    """
    Traditional MoE models uses this model impl: Mixtral, LLAMA4_MOE, MiniMax-M2, etc.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention,
    exact match is required. Same for logits_gemm.
    Supports MTP (Multi-Token Prediction) speculative decoding simulation.
    TODO: redesign shared moe part.
    """

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> MOEModel:
        return cls(
            model_info["topk"],
            model_info["num_experts"],
            model_info["moe_inter_size"],
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
            model_info.get("extra_params"),
        )

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # MTP scale factor: throughput boost / compute overhead
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
            if self._nextn > 0
            else 1.0
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
                    window_size=window_size,
                    head_size=self._head_size,
                    use_qk_norm=self._use_qk_norm,
                )
            )
            self.generation_ops.append(
                ops.GenerationAttention(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor / attn_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    window_size=window_size,
                    head_size=self._head_size,
                    use_qk_norm=self._use_qk_norm,
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
                    use_qk_norm=self._use_qk_norm,
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
                    quant_mode=moe_quant_mode,
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
                    quant_mode=moe_quant_mode,
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
                    use_qk_norm=self._use_qk_norm,
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
                    quant_mode=moe_quant_mode,
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
                    quant_mode=moe_quant_mode,
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
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
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
