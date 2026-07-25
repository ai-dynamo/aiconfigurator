# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model


@register_model("GPT")
class GPTModel(BaseModel):
    """
    GPT series uses this model impl.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention,
    exact match is required. Same for logits_gemm.
    Other than DS V3, all other models don't support mtp
    """

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # Dense GQA prefill CP: SGLang AllGather (zigzag), same as LLAMAModel.
        return backend_name == "sglang"

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        return cls(
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
            model_info["extra_params"],
        )

    def __init__(self, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, f"{type(self).__name__} does not support MTP speculative decoding (nextn must be 0)"

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_gpu = self._num_kv_heads_per_gpu
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        # Context parallelism (sglang AllGather, prefill-only); see LLAMAModel.
        cp = self.config.cp_size

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3, seq_split=cp),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
                ops.GEMM(
                    "context_qkv_gemm",
                    self._num_layers,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                    seq_split=cp,
                ),
                ops.ContextAttention(
                    "context_attention",
                    self._num_layers,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    cp_size=cp,
                ),
                *self._cp_attn_comm_ops(),
                ops.GEMM(
                    "context_proj_gemm",
                    self._num_layers,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                    seq_split=cp,
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
                ops.GEMM(
                    "context_ffn1_gemm",
                    self._num_layers,
                    self._inter_size // tp_size,
                    h,
                    gemm_quant_mode,
                    seq_split=cp,
                ),
                ops.ElementWise(
                    "context_act",
                    self._num_layers,
                    self._inter_size // tp_size,
                    self._inter_size // tp_size,
                    0.8,
                    seq_split=cp,
                ),
                ops.GEMM(
                    "context_ffn2_gemm",
                    self._num_layers,
                    h,
                    self._inter_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                    seq_split=cp,
                ),
                ops.GEMM(
                    "context_logits_gemm",
                    1,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                    seq_split=cp,
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
                    common.GEMMQuantMode.bfloat16,
                ),
            ]
        )

        # when tp_size=0, the comm part will be 0
        self.context_ops.append(ops.CustomAllReduce("context_ar_1", self._num_layers, h, tp_size, seq_split=cp))
        self.context_ops.append(ops.CustomAllReduce("context_ar_2", self._num_layers, h, tp_size, seq_split=cp))
        self.generation_ops.append(ops.CustomAllReduce("generation_ar_1", self._num_layers, h, tp_size))
        self.generation_ops.append(ops.CustomAllReduce("generation_ar_2", self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size, seq_split=cp))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor, h, pp_size))
