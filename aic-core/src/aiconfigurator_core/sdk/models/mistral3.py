# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.llama import LLAMAModel

from .blocks.vit import build_encoder_ops


@register_model("MISTRAL3")
class Mistral3Model(LLAMAModel):
    """
    Mistral-Medium-3.5 series. Extends LLAMAModel with a Pixtral vision encoder.

    The LLM backbone (text_config) is a dense GQA decoder and reuses all
    LLAMAModel context/generation ops. The Pixtral ViT (vision_config) runs
    before the LLM prefill phase and is represented as encoder_ops.

    ViT ops run in bfloat16 regardless of LLM quantization. The ViT FFN is
    SwiGLU (gated_mlp), so build_encoder_ops emits a separate gate projection.
    Encoder parallelism follows ModelConfig.enable_encoder_dp: DP replicas over
    the tp_size ranks by default, legacy TP sharding otherwise.
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
        # EPD language-only workers keep encoder_config (vision tokens still
        # extend the LLM context) but never host the ViT ops.
        if not self.config.language_only:
            self.encoder_ops.extend(
                build_encoder_ops(encoder_config, self.config.tp_size, self.config.enable_encoder_dp)
            )
