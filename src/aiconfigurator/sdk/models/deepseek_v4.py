# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from typing import ClassVar

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, register_model
from aiconfigurator.sdk.models.helpers import calc_expectation


@register_model("DEEPSEEKV4")
class DeepSeekV4Model(BaseModel):
    """DeepSeek-V4 model with mHC plus SWA/CSA/HCA compressed attention."""

    _SUPPORTED_COMPRESS_RATIOS: ClassVar[set[int]] = {0, 4, 128}

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
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
            model_info["extra_params"],
        )

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        if not isinstance(self.extra_params, common.DeepSeekV4Config):
            raise TypeError("DeepSeekV4Model requires DeepSeekV4Config extra_params")
        deepseek_v4_cfg = self.extra_params
        self._compress_ratios = deepseek_v4_cfg.compress_ratios
        unknown_ratios = set(self._compress_ratios) - self._SUPPORTED_COMPRESS_RATIOS
        if unknown_ratios:
            raise ValueError(f"Unsupported DeepSeek-V4 compress_ratios: {sorted(unknown_ratios)}")

        assert (
            self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) should be equal to moe_tp_size "
            f"({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )
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
        local_heads = self._num_heads // tp_size
        local_o_groups = max(1, deepseek_v4_cfg.o_groups // tp_size)
        local_moe_inter_size = self._moe_inter_size // tp_size

        def _attention_ops(is_context: bool, scale_factor: float):
            ratio_counts = Counter(self._compress_ratios)
            # DeepSeek-V4 Flash has a small number of pure SWA layers
            # (compress_ratio=0). Approximate their module latency with HCA
            # (compress_ratio=128) so the model reuses DeepSeek-V4 HCA perf data
            # instead of requiring a dedicated SWA collector. KV cache capacity
            # below still uses the real per-layer ratios.
            ratio_counts[128] += ratio_counts.pop(0, 0)
            op_cls = ops.ContextDeepSeekV4AttentionModule if is_context else ops.GenerationDeepSeekV4AttentionModule
            name = "context_attention" if is_context else "generation_attention"
            return [
                op_cls(
                    name,
                    count * scale_factor,
                    local_heads,
                    h,
                    deepseek_v4_cfg.q_lora_rank,
                    deepseek_v4_cfg.o_lora_rank,
                    deepseek_v4_cfg.head_dim,
                    deepseek_v4_cfg.qk_rope_head_dim,
                    deepseek_v4_cfg.index_n_heads,
                    deepseek_v4_cfg.index_head_dim,
                    deepseek_v4_cfg.index_topk,
                    deepseek_v4_cfg.sliding_window,
                    ratio,
                    local_o_groups,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    gemm_quant_mode,
                    architecture=self.architecture,
                )
                for ratio, count in ratio_counts.items()
                if count > 0
            ]

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
                ops.DeepSeekV4MHCModule(
                    "context_mhc_pre",
                    self._num_layers,
                    "pre",
                    h,
                    deepseek_v4_cfg.hc_mult,
                    deepseek_v4_cfg.hc_sinkhorn_iters,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.ElementWise("context_attn_norm", self._num_layers, h, h, 0.8),
                *_attention_ops(is_context=True, scale_factor=1.0),
                ops.DeepSeekV4MHCModule(
                    "context_mhc_post",
                    self._num_layers,
                    "post",
                    h,
                    deepseek_v4_cfg.hc_mult,
                    deepseek_v4_cfg.hc_sinkhorn_iters,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.ElementWise("context_ffn_norm", self._num_layers, h, h, 0.8),
                ops.GEMM(
                    "context_shared_gate_up_gemm",
                    self._num_layers,
                    2 * local_moe_inter_size,
                    h,
                    gemm_quant_mode,
                ),
                ops.ElementWise(
                    "context_shared_act_gate",
                    self._num_layers,
                    2 * local_moe_inter_size,
                    local_moe_inter_size,
                    0.8,
                ),
                ops.GEMM("context_shared_ffn2_gemm", self._num_layers, h, local_moe_inter_size, gemm_quant_mode),
                ops.GEMM("context_router_gemm", self._num_layers, self._num_experts, h, common.GEMMQuantMode.bfloat16),
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
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp_size, h, common.GEMMQuantMode.bfloat16),
            ]
        )

        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, self._vocab_size, h, 0.3),
                ops.DeepSeekV4MHCModule(
                    "generation_mhc_pre",
                    self._num_layers * self._mtp_scale_factor,
                    "pre",
                    h,
                    deepseek_v4_cfg.hc_mult,
                    deepseek_v4_cfg.hc_sinkhorn_iters,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.ElementWise("generation_attn_norm", self._num_layers * self._mtp_scale_factor, h, h, 0.8),
                *_attention_ops(is_context=False, scale_factor=self._mtp_scale_factor),
                ops.DeepSeekV4MHCModule(
                    "generation_mhc_post",
                    self._num_layers * self._mtp_scale_factor,
                    "post",
                    h,
                    deepseek_v4_cfg.hc_mult,
                    deepseek_v4_cfg.hc_sinkhorn_iters,
                    common.GEMMQuantMode.bfloat16,
                ),
                ops.ElementWise("generation_ffn_norm", self._num_layers * self._mtp_scale_factor, h, h, 0.8),
            ]
        )

        gen_shared_ops = [
            ops.GEMM(
                "generation_shared_gate_up_gemm",
                self._num_layers * self._mtp_scale_factor,
                2 * local_moe_inter_size,
                h,
                gemm_quant_mode,
            ),
            ops.ElementWise(
                "generation_shared_act_gate",
                self._num_layers * self._mtp_scale_factor,
                2 * local_moe_inter_size,
                local_moe_inter_size,
                0.8,
            ),
            ops.GEMM(
                "generation_shared_ffn2_gemm",
                self._num_layers * self._mtp_scale_factor,
                h,
                local_moe_inter_size,
                gemm_quant_mode,
            ),
        ]
        gen_routed_ops = [
            ops.GEMM(
                "generation_router_gemm",
                self._num_layers * self._mtp_scale_factor,
                self._num_experts,
                h,
                common.GEMMQuantMode.bfloat16,
            ),
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
        self.generation_ops.append(
            ops.OverlapOp("generation_moe_overlap", group_a=gen_routed_ops, group_b=gen_shared_ops)
        )
        self.generation_ops.append(
            ops.GEMM(
                "generation_logits_gemm",
                1 * self._mtp_scale_factor,
                self._vocab_size // tp_size,
                h,
                common.GEMMQuantMode.bfloat16,
            )
        )

        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

    def get_kvcache_bytes_per_sequence(self, seq_len: int) -> float:
        deepseek_v4_cfg = self.extra_params
        seq_len = max(0, seq_len)
        total = 0.0
        cache_entry_bytes = deepseek_v4_cfg.head_dim * self.config.kvcache_quant_mode.value.memory
        for ratio in self._compress_ratios:
            total += min(seq_len, deepseek_v4_cfg.sliding_window) * cache_entry_bytes
            if ratio:
                compressed_entries = seq_len // ratio
                total += compressed_entries * cache_entry_bytes
                coff = 2 if ratio == 4 else 1
                # Compressor decode state keeps FP32 kv_state and score_state buffers.
                total += 2 * ratio * coff * deepseek_v4_cfg.head_dim * 4
                if ratio == 4:
                    total += compressed_entries * common.deepseek_v4_indexer_cache_entry_bytes(
                        deepseek_v4_cfg.index_head_dim
                    )
                    # CSA has a second FP4 indexer compressor with its own decode state.
                    total += 2 * ratio * 2 * deepseek_v4_cfg.index_head_dim * 4
        return total
