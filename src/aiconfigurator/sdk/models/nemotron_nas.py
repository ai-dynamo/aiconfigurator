# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, register_model

logger = logging.getLogger(__name__)


@register_model("NEMOTRONNAS")
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

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> NemotronNas:
        model = cls(
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
        )
        extra_params = model_info.get("extra_params")
        if isinstance(extra_params, list):
            model.context_ops = extra_params
            model.generation_ops = extra_params
        else:
            logger.warning(
                "NemotronNAS model '%s' missing block configs in model metadata; leaving pipelines empty.",
                model_info["model_path"],
            )
            model.context_ops = []
            model.generation_ops = []
        return model

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
