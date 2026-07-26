# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3: MoE + MiniMax Sparse Attention (MSA), text decoder only.

Structurally a sibling of DeepSeekV32Model (MoE + sparse-attention-module +
shared expert), with the attention block modeled by the MSA op instead of DSA:
MSA is GQA (64 q / 4 kv heads, head_dim 128, partial rope) with a per-block
indexer selecting the top-16 blocks (block_size 128 -> 2048 tokens). There is no
MSA silicon data, so the MSA op runs HYBRID/EMPIRICAL and transfers from DSA's
measured utilisation (scaled by ``msa_dsa_scale_k``).

Like the GLM-5 / DeepSeek-V3.2 modeling, the first ``first_k_dense_replace``
dense layers are approximated as MoE+MSA (a ~5% layer fraction); the vision
tower of the multimodal checkpoint is out of scope (text decoder only).
"""

from __future__ import annotations

import logging

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor

logger = logging.getLogger(__name__)


@register_model("MINIMAXM3")
class MiniMaxM3Model(BaseModel):
    """MiniMax-M3 text decoder (MoE + MSA)."""

    # MSA / sparse-attention-config constants (MiniMax-M3).
    INDEX_N_HEADS = 4
    INDEX_HEAD_DIM = 128
    SPARSE_TOPK_BLOCKS = 16
    SPARSE_BLOCK_SIZE = 128
    V_HEAD_DIM = 128
    DSA_TRANSFER_ARCH = "GlmMoeDsaForCausalLM"  # DSA op to borrow util from

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
        )
        return cls(*moe_args, *base_args, dict(model_info["extra_params"] or {}))

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)
        assert (
            self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size ({self.config.attention_dp_size}) must equal "
            f"moe_tp_size ({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )
        assert num_experts >= self.config.moe_ep_size, f"ep size cannot exceed num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)
        self._power_law_alpha = 1.01

        h = self._hidden_size
        tp_size = self.config.tp_size
        moe_tp_size = self.config.moe_tp_size
        moe_ep_size = self.config.moe_ep_size
        attention_dp_size = self.config.attention_dp_size
        pp_size = self.config.pp_size

        gemm_quant_mode = self.config.gemm_quant_mode
        moe_quant_mode = self.config.moe_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        workload_distribution = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        dsa_scale_k = self.extra_params.get("msa_dsa_scale_k", 1.0) if isinstance(self.extra_params, dict) else 1.0

        local_heads = max(1, self._num_heads // tp_size)
        local_kv_heads = max(1, self._num_kv_heads // tp_size)
        index_topk = self.SPARSE_TOPK_BLOCKS * self.SPARSE_BLOCK_SIZE

        def _msa(cls_op, name, scale):
            return cls_op(
                name,
                scale,
                local_heads,
                local_kv_heads,
                h,
                self._head_size,
                self.V_HEAD_DIM,
                self.INDEX_N_HEADS,
                self.INDEX_HEAD_DIM,
                index_topk,
                self.SPARSE_BLOCK_SIZE,
                kvcache_quant_mode,
                fmha_quant_mode,
                gemm_quant_mode,
                dsa_architecture=self.DSA_TRANSFER_ARCH,
                dsa_scale_k=dsa_scale_k,
            )

        def _shared_gate_up(name, scale):
            return ops.GEMM(name, scale, 2 * self._moe_inter_size // tp_size, h, gemm_quant_mode)

        def _shared_ffn2(name, scale):
            return ops.GEMM(name, scale, h, self._moe_inter_size // tp_size, gemm_quant_mode)

        def _router(name, scale):
            return ops.GEMM(name, scale, self._num_experts, h, common.GEMMQuantMode.bfloat16)

        def _dispatch(name, scale, pre):
            return ops.MoEDispatch(
                name,
                scale,
                h,
                self._topk,
                self._num_experts,
                moe_tp_size,
                moe_ep_size,
                attention_dp_size,
                pre,
                quant_mode=moe_quant_mode,
            )

        def _moe(name, scale):
            return ops.MoE(
                name,
                scale,
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

        nl = self._num_layers
        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.ElementWise("context_add_norm_1", nl, 2 * h, 2 * h, 0.8),
                _msa(ops.ContextMSAModule, "context_attention", nl),
                ops.ElementWise("context_add_norm_2", nl, 2 * h, 2 * h, 0.8),
                _shared_gate_up("context_shared_gate_up_gemm", nl),
                ops.ElementWise(
                    "context_shared_act_gate",
                    nl,
                    2 * self._moe_inter_size // tp_size,
                    self._moe_inter_size // tp_size,
                    0.8,
                ),
                _shared_ffn2("context_shared_ffn2_gemm", nl),
                _router("context_router_gemm", nl),
                _dispatch("context_moe_pre_dispatch", nl, True),
                _moe("context_moe", nl),
                _dispatch("context_moe_post_dispatch", nl, False),
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp_size, h, common.GEMMQuantMode.bfloat16),
            ]
        )

        mtp = self._mtp_scale_factor
        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * mtp, self._vocab_size, h, 0.3),
                ops.ElementWise("generation_add_norm_1", nl * mtp, 2 * h, 2 * h, 0.8),
                _msa(ops.GenerationMSAModule, "generation_attention", nl * mtp),
                ops.ElementWise("generation_add_norm_2", nl * mtp, 2 * h, 2 * h, 0.8),
            ]
        )
        gen_shared_ops = [
            _shared_gate_up("generation_shared_gate_up_gemm", nl * mtp),
            ops.ElementWise(
                "generation_shared_act_gate",
                nl * mtp,
                2 * self._moe_inter_size // tp_size,
                self._moe_inter_size // tp_size,
                0.8,
            ),
            _shared_ffn2("generation_shared_ffn2_gemm", nl * mtp),
        ]
        gen_routed_ops = [
            _router("generation_router_gemm", nl * mtp),
            _dispatch("generation_moe_pre_dispatch", nl * mtp, True),
            _moe("generation_moe", nl * mtp),
            _dispatch("generation_moe_post_dispatch", nl * mtp, False),
        ]
        self.generation_ops.append(
            ops.OverlapOp("generation_moe_overlap", group_a=gen_routed_ops, group_b=gen_shared_ops)
        )
        self.generation_ops.append(
            ops.GEMM("generation_logits_gemm", 1 * mtp, self._vocab_size // tp_size, h, common.GEMMQuantMode.bfloat16)
        )

        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * mtp, h, pp_size))
