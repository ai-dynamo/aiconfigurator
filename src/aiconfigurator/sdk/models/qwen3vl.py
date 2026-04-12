# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, register_model
from aiconfigurator.sdk.models.llama import LLAMAModel
from aiconfigurator.sdk.models.moe import MOEModel


@register_model("QWEN3VL")
class Qwen3VLModel(LLAMAModel):
    """
    Qwen3-VL series. Extends LLAMAModel with a ViT vision encoder.

    The LLM backbone (text_config) is identical to Qwen3 and reuses all
    LLAMAModel context/generation ops. The vision encoder (vision_config)
    runs before the LLM prefill phase and is represented as encoder_ops.

    ViT ops run in bfloat16 regardless of LLM quantization. TP is applied
    to the ViT heads and FFN in the same way as the LLM backbone.
    """

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
            encoder_config=model_info["extra_params"],
        )

    def __init__(self, *args, encoder_config: common.VisionEncoderConfig) -> None:
        super().__init__(*args)

        if encoder_config is None:
            return
        self.encoder_config = encoder_config

        tp_size = self.config.tp_size
        depth = encoder_config.depth
        h_vit = encoder_config.hidden_size
        n_vit = encoder_config.num_heads
        inter_vit = encoder_config.intermediate_size
        h_llm = encoder_config.out_hidden_size
        head_size_vit = h_vit // n_vit  # 1152 // 16 = 72
        n_mergers = 1 + len(encoder_config.deepstack_visual_indexes)

        if tp_size > 1:
            if n_vit % tp_size != 0:
                raise ValueError(f"ViT num_heads ({n_vit}) must be divisible by tp_size ({tp_size})")
            if inter_vit % tp_size != 0:
                raise ValueError(f"ViT intermediate_size ({inter_vit}) must be divisible by tp_size ({tp_size})")

        # ViT always runs in bfloat16 regardless of LLM quantization settings
        vit_gemm_mode = common.GEMMQuantMode.bfloat16
        vit_fmha_mode = common.FMHAQuantMode.bfloat16
        vit_kvcache_mode = common.KVCacheQuantMode.bfloat16

        self.encoder_ops.extend(
            [
                ops.ElementWise("encoder_add_norm_1", depth, 2 * h_vit, 2 * h_vit, 0.8),
                ops.GEMM(
                    "encoder_qkv_gemm",
                    depth,
                    3 * n_vit * head_size_vit // tp_size,
                    h_vit,
                    vit_gemm_mode,
                ),
                ops.ContextAttention(
                    "encoder_attention",
                    depth,
                    n_vit // tp_size,
                    n_vit // tp_size,  # ViT has no GQA: n_kv == n
                    vit_kvcache_mode,
                    vit_fmha_mode,
                    head_size=head_size_vit,
                ),
                ops.GEMM(
                    "encoder_proj_gemm",
                    depth,
                    h_vit,
                    n_vit * head_size_vit // tp_size,
                    vit_gemm_mode,
                    low_precision_input=True,
                ),
                ops.CustomAllReduce("encoder_ar_1", depth, h_vit, tp_size),
                ops.ElementWise("encoder_add_norm_2", depth, 2 * h_vit, 2 * h_vit, 0.8),
                ops.GEMM(
                    "encoder_ffn1_gemm",
                    depth,
                    inter_vit // tp_size,
                    h_vit,
                    vit_gemm_mode,
                ),
                ops.ElementWise(
                    "encoder_act",
                    depth,
                    inter_vit // tp_size,
                    inter_vit // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "encoder_ffn2_gemm",
                    depth,
                    h_vit,
                    inter_vit // tp_size,
                    vit_gemm_mode,
                    low_precision_input=True,
                ),
                ops.CustomAllReduce("encoder_ar_2", depth, h_vit, tp_size),
                # PatchMerger MLP: runs n_mergers times (1 final + deepstack instances)
                # Each merger: Linear(4*h_vit->4*h_vit) + GELU + Linear(4*h_vit->h_llm)
                # Operates on post-merge tokens (spatial_merge_size^2 patches fused per token)
                ops.GEMM(
                    "encoder_merger_fc1",
                    n_mergers,
                    (h_vit * encoder_config.spatial_merge_size**2) // tp_size,
                    h_vit * encoder_config.spatial_merge_size**2,
                    vit_gemm_mode,
                ),
                ops.ElementWise(
                    "encoder_merger_act",
                    n_mergers,
                    (h_vit * encoder_config.spatial_merge_size**2) // tp_size,
                    (h_vit * encoder_config.spatial_merge_size**2) // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "encoder_merger_fc2",
                    n_mergers,
                    h_llm,
                    (h_vit * encoder_config.spatial_merge_size**2) // tp_size,
                    vit_gemm_mode,
                ),
                ops.CustomAllReduce("encoder_merger_ar", n_mergers, h_llm, tp_size),
            ]
        )


@register_model("QWEN3VL_MOE")
class Qwen3VLMoEModel(MOEModel):
    """
    Qwen3-VL MoE variants (30B-A3B, 235B-A22B). Extends MOEModel with a ViT
    vision encoder identical to Qwen3VLModel. The LLM backbone uses sparse MoE
    FFN while the ViT encoder is a standard dense transformer.
    """

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        moe_args = (model_info["topk"], model_info["num_experts"], model_info["moe_inter_size"])
        base_args = (
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
        return cls(*moe_args, *base_args, encoder_config=model_info["extra_params"])

    def __init__(
        self, topk: int, num_experts: int, moe_inter_size: int, *args, encoder_config: common.VisionEncoderConfig
    ) -> None:
        super().__init__(topk, num_experts, moe_inter_size, *args)

        if encoder_config is None:
            return
        self.encoder_config = encoder_config

        tp_size = self.config.tp_size
        depth = encoder_config.depth
        h_vit = encoder_config.hidden_size
        n_vit = encoder_config.num_heads
        inter_vit = encoder_config.intermediate_size
        h_llm = encoder_config.out_hidden_size
        head_size_vit = h_vit // n_vit
        n_mergers = 1 + len(encoder_config.deepstack_visual_indexes)

        if tp_size > 1:
            if n_vit % tp_size != 0:
                raise ValueError(f"ViT num_heads ({n_vit}) must be divisible by tp_size ({tp_size})")
            if inter_vit % tp_size != 0:
                raise ValueError(f"ViT intermediate_size ({inter_vit}) must be divisible by tp_size ({tp_size})")

        # ViT always runs in bfloat16 regardless of LLM quantization settings
        vit_gemm_mode = common.GEMMQuantMode.bfloat16
        vit_fmha_mode = common.FMHAQuantMode.bfloat16
        vit_kvcache_mode = common.KVCacheQuantMode.bfloat16

        self.encoder_ops.extend(
            [
                ops.ElementWise("encoder_add_norm_1", depth, 2 * h_vit, 2 * h_vit, 0.8),
                ops.GEMM(
                    "encoder_qkv_gemm",
                    depth,
                    3 * n_vit * head_size_vit // tp_size,
                    h_vit,
                    vit_gemm_mode,
                ),
                ops.ContextAttention(
                    "encoder_attention",
                    depth,
                    n_vit // tp_size,
                    n_vit // tp_size,
                    vit_kvcache_mode,
                    vit_fmha_mode,
                    head_size=head_size_vit,
                ),
                ops.GEMM(
                    "encoder_proj_gemm",
                    depth,
                    h_vit,
                    n_vit * head_size_vit // tp_size,
                    vit_gemm_mode,
                    low_precision_input=True,
                ),
                ops.CustomAllReduce("encoder_ar_1", depth, h_vit, tp_size),
                ops.ElementWise("encoder_add_norm_2", depth, 2 * h_vit, 2 * h_vit, 0.8),
                ops.GEMM(
                    "encoder_ffn1_gemm",
                    depth,
                    inter_vit // tp_size,
                    h_vit,
                    vit_gemm_mode,
                ),
                ops.ElementWise(
                    "encoder_act",
                    depth,
                    inter_vit // tp_size,
                    inter_vit // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "encoder_ffn2_gemm",
                    depth,
                    h_vit,
                    inter_vit // tp_size,
                    vit_gemm_mode,
                    low_precision_input=True,
                ),
                ops.CustomAllReduce("encoder_ar_2", depth, h_vit, tp_size),
                ops.GEMM(
                    "encoder_merger_fc1",
                    n_mergers,
                    (h_vit * encoder_config.spatial_merge_size**2) // tp_size,
                    h_vit * encoder_config.spatial_merge_size**2,
                    vit_gemm_mode,
                ),
                ops.ElementWise(
                    "encoder_merger_act",
                    n_mergers,
                    (h_vit * encoder_config.spatial_merge_size**2) // tp_size,
                    (h_vit * encoder_config.spatial_merge_size**2) // tp_size,
                    0.8,
                ),
                ops.GEMM(
                    "encoder_merger_fc2",
                    n_mergers,
                    h_llm,
                    (h_vit * encoder_config.spatial_merge_size**2) // tp_size,
                    vit_gemm_mode,
                ),
                ops.CustomAllReduce("encoder_merger_ar", n_mergers, h_llm, tp_size),
            ]
        )
