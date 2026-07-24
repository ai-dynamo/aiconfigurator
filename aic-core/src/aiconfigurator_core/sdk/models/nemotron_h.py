# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor


@register_model("NEMOTRONH")
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

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        model = cls(
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
        )
        # extra_params is NemotronHConfig with hybrid layer configuration.
        model.set_hybrid_config(model_info["extra_params"])
        return model

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)
        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size
        self._hybrid_config: common.NemotronHConfig | None = None
        self._power_law_alpha = 1.01  # follow DeepSeek MoE
        # MTP (num_nextn_predict_layers > 0): scale generation iteration cost by
        # the small extra-layer factor (nextn+L)/L. Accepted-token progress is
        # applied by the upper prediction layer. nextn == 0 -> factor 1.0.
        #
        # APPROXIMATION (intentional, hybrid architecture). NemotronH is a mixed
        # Mamba / attention / MoE stack (e.g. Ultra-550B = 48 mamba + 48 moe + 12
        # attention over 108 layers) and its MTP block is attention+moe -- NOT the
        # dominant Mamba layer type. DeepSeek's uniform (nextn+L)/L is exact there only
        # because every layer is homogeneous (attn+MoE). Here we still spread that term
        # uniformly over all layer-type groups, so the extra MTP-block cost is slightly
        # mis-attributed: we add a little fake Mamba and under-count the real attn+moe
        # MTP block. This is bounded by ~1/L of one layer's cost (<1% of TPOT for these
        # deep models). Not worth modeling an explicit attn+moe MTP block for
        # sub-1% gain.
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)

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
            # Latent-MoE (e.g. Nemotron-3-Super): experts run on a compressed
            # `moe_latent_size` projection of hidden_states. Shared expert and the
            # router still operate on hidden_size.
            moe_h = cfg.moe_latent_size if cfg.moe_latent_size > 0 else h
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
                    common.GEMMQuantMode.bfloat16,
                )
            )

            # MoE dispatch and compute
            # When latent-MoE is active, fc1_latent_proj (h -> latent) runs before
            # dispatch, so dispatch/experts/combine all communicate in latent dim.
            if cfg.moe_latent_size > 0:
                self.context_ops.append(
                    ops.GEMM(
                        "context_fc1_latent_proj_gemm",
                        count,
                        cfg.moe_latent_size // tp_size,
                        h,
                        gemm_quant_mode,
                    )
                )
            self.context_ops.extend(
                [
                    ops.MoEDispatch(
                        "context_moe_pre_dispatch",
                        count,
                        moe_h,
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
                        count,
                        moe_h,
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
                        moe_h,
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
            if cfg.moe_latent_size > 0:
                # fc2_latent_proj (latent -> h) runs after expert combine,
                # before the post-MoE allreduce. Modeled as a row-parallel GEMM
                # whose partial sums are aggregated by the following AllReduce.
                self.context_ops.append(
                    ops.GEMM(
                        "context_fc2_latent_proj_gemm",
                        count,
                        h,
                        cfg.moe_latent_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    )
                )
            # TRT-LLM does allreduce after combining routed + shared outputs when TP>1
            self.context_ops.append(ops.CustomAllReduce("context_moe_ar", count, h, tp_size))

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
                common.GEMMQuantMode.bfloat16,
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
        mtp = self._mtp_scale_factor
        self.generation_ops.append(ops.Embedding("generation_embedding", 1 * mtp, self._vocab_size, h, 0.3))

        # Mamba layers (M): norm, in_proj GEMM, conv1d, ssm, out_proj GEMM, ar
        if layer_counts["M"] > 0:
            count = layer_counts["M"] * mtp
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
            count = layer_counts["*"] * mtp
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
            count = layer_counts["E"] * mtp
            # Latent-MoE (e.g. Nemotron-3-Super): experts run on a compressed
            # `moe_latent_size` projection of hidden_states. Shared expert and the
            # router still operate on hidden_size.
            moe_h = cfg.moe_latent_size if cfg.moe_latent_size > 0 else h
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
                    common.GEMMQuantMode.bfloat16,
                )
            )

            # MoE dispatch and compute
            # When latent-MoE is active, fc1_latent_proj (h -> latent) runs before
            # dispatch, so dispatch/experts/combine all communicate in latent dim.
            if cfg.moe_latent_size > 0:
                self.generation_ops.append(
                    ops.GEMM(
                        "generation_fc1_latent_proj_gemm",
                        count,
                        cfg.moe_latent_size // tp_size,
                        h,
                        gemm_quant_mode,
                    )
                )
            self.generation_ops.extend(
                [
                    ops.MoEDispatch(
                        "generation_moe_pre_dispatch",
                        count,
                        moe_h,
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
                        count,
                        moe_h,
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
                        moe_h,
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
            if cfg.moe_latent_size > 0:
                # fc2_latent_proj (latent -> h) runs after expert combine,
                # before the post-MoE allreduce. Modeled as a row-parallel GEMM
                # whose partial sums are aggregated by the following AllReduce.
                self.generation_ops.append(
                    ops.GEMM(
                        "generation_fc2_latent_proj_gemm",
                        count,
                        h,
                        cfg.moe_latent_size // tp_size,
                        gemm_quant_mode,
                        low_precision_input=True,
                    )
                )
            # TRT-LLM does allreduce after combining routed + shared outputs when TP>1
            self.generation_ops.append(ops.CustomAllReduce("generation_moe_ar", count, h, tp_size))

        # MLP layers (-)
        if layer_counts["-"] > 0:
            count = layer_counts["-"] * mtp
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
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * mtp, h, pp_size))

        # Logits GEMM
        self.generation_ops.append(
            ops.GEMM(
                "generation_logits_gemm",
                1 * mtp,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )
