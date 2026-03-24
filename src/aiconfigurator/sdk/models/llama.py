# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, calc_expectation, register_model


@register_model("LLAMA")
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
            if self._nextn > 0
            else 1.0
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
                ops.Embedding("context_embedding", 1, self._vocab_size // tp_size, h, 0.3),
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
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size // tp_size, h, 0.3),
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
        self.context_ops.append(ops.CustomAllReduce("context_embedding_ar", 1, h, tp_size))
        self.context_ops.append(ops.CustomAllReduce("context_ar_1", self._num_layers, h, tp_size))
        self.context_ops.append(ops.CustomAllReduce("context_ar_2", self._num_layers, h, tp_size))

        self.generation_ops.append(
            ops.CustomAllReduce("generation_embedding_ar", 1 * self._mtp_scale_factor, h, tp_size)
        )
        self.generation_ops.append(
            ops.CustomAllReduce("generation_ar_1", self._num_layers * self._mtp_scale_factor, h, tp_size)
        )
        self.generation_ops.append(
            ops.CustomAllReduce("generation_ar_2", self._num_layers * self._mtp_scale_factor, h, tp_size)
        )

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))
