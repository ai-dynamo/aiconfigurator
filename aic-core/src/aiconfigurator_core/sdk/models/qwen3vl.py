# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.llama import LLAMAModel
from aiconfigurator_core.sdk.models.moe import MOEModel
from aiconfigurator_core.sdk.models.vit_ops import build_encoder_ops


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
        self.encoder_ops.extend(build_encoder_ops(encoder_config, self.config.tp_size))


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
        self.encoder_ops.extend(build_encoder_ops(encoder_config, self.config.tp_size))
