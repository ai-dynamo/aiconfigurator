# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models.base import BaseModel, calc_expectation, register_model
from aiconfigurator.sdk.utils import _load_model_config_from_model_path

logger = logging.getLogger(__name__)


@register_model("HYBRIDMOE")
class HybridMoEModel(BaseModel):
    """
    Hybrid attention + mixed FFN model (MiMo-V2-Flash, Llama 4 Scout/Maverick, and similar).

    Handles four layer types derived from HybridMoEConfig.attn_layer_pattern and moe_layer_freq:
    - global_moe:  global (full) attention + MoE FFN
    - swa_moe:     SWA/local attention + MoE FFN
    - swa_dense:   SWA/local attention + dense SwiGLU FFN
    - global_dense: global attention + dense SwiGLU FFN (rare but supported)

    SWA/local attention dims fall back to model-level defaults when HybridMoEConfig fields are 0.
    This lets same-dim models (Llama 4) and different-dim models (MiMo-V2-Flash) share one class.
    """

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)
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
        self._mtp_scale_factor = (
            1.0
            / (1 + calc_expectation(self._nextn, self._nextn_accept_rates))
            * (self._nextn + self._num_layers)
            / self._num_layers
            if self._nextn > 0
            else 1.0
        )
        self._validate_fp8_block_quantized_moe_config()
        self._hybrid_config: common.HybridMoEConfig | None = None
        self._power_law_alpha = 1.01

    def _validate_fp8_block_quantized_moe_config(self) -> None:
        """Validate fp8_block MoE alignment: (moe_inter_size / moe_tp_size) % block_size == 0."""
        if self.config.moe_quant_mode != common.MoEQuantMode.fp8_block:
            return
        raw_config = _load_model_config_from_model_path(self.model_path)
        default_size = [128, 128]
        weight_block_size = raw_config.get("quantization_config", {}).get("weight_block_size", default_size)[0]
        moe_size_per_gpu = self._moe_inter_size // self.config.moe_tp_size
        if (moe_size_per_gpu % weight_block_size) != 0:
            raise ValueError(
                f"Invalid quantized MoE configuration: "
                f"(moe_intermediate_size={self._moe_inter_size} / moe_tp_size={self.config.moe_tp_size}) "
                f"% weight_block_size={weight_block_size} != 0. "
            )

    def set_hybrid_config(self, cfg: common.HybridMoEConfig) -> None:
        """Apply HybridMoEConfig and rebuild context/generation ops.

        Validates that attn_layer_pattern and moe_layer_freq have the same length,
        match self._num_layers, and contain only 0/1 values before accepting the config.
        """
        n = len(cfg.attn_layer_pattern)
        if n != len(cfg.moe_layer_freq):
            raise ValueError(
                f"HybridMoEConfig pattern length mismatch: "
                f"attn_layer_pattern has {n} entries "
                f"but moe_layer_freq has {len(cfg.moe_layer_freq)}"
            )
        if n != self._num_layers:
            raise ValueError(f"HybridMoEConfig pattern length ({n}) does not match num_layers ({self._num_layers})")
        for i, (a, m) in enumerate(zip(cfg.attn_layer_pattern, cfg.moe_layer_freq)):
            if a not in (0, 1) or m not in (0, 1):
                raise ValueError(f"HybridMoEConfig layer {i} has invalid values: attn={a}, moe={m} (expected 0 or 1)")
        self._hybrid_config = cfg
        self._build_context_ops()
        self._build_generation_ops()

    def _count_layer_types(self) -> dict[str, int]:
        """Count layers per type: global_moe, swa_moe, swa_dense, global_dense."""
        cfg = self._hybrid_config
        counts: dict[str, int] = {"global_moe": 0, "swa_moe": 0, "swa_dense": 0, "global_dense": 0}
        for attn, moe in zip(cfg.attn_layer_pattern, cfg.moe_layer_freq):
            if attn == 1 and moe == 1:
                counts["global_moe"] += 1
            elif attn == 0 and moe == 1:
                counts["swa_moe"] += 1
            elif attn == 0 and moe == 0:
                counts["swa_dense"] += 1
            else:
                counts["global_dense"] += 1
        return counts

    def _resolve_dims(self, tp_size: int) -> dict:
        """Resolve SWA/local attention dims, falling back to model-level defaults when 0.

        Returns a dict with per-TP KV head counts, QKV GEMM output widths, proj GEMM input widths,
        Q/K head dims for attention kernels, and dense FFN intermediate size per TP.
        """
        cfg = self._hybrid_config
        swa_n_kv = cfg.swa_num_kv_heads if cfg.swa_num_kv_heads > 0 else self._num_kv_heads
        swa_hd = cfg.swa_head_dim if cfg.swa_head_dim > 0 else self._head_size
        swa_v_hd = cfg.swa_v_head_dim if cfg.swa_v_head_dim > 0 else self._head_size
        global_v_hd = cfg.global_v_head_dim if cfg.global_v_head_dim > 0 else self._head_size
        swa_n_kv_per_gpu = (swa_n_kv + tp_size - 1) // tp_size
        global_n_kv_per_gpu = (self._num_kv_heads + tp_size - 1) // tp_size
        dense_inter = cfg.dense_inter_size if cfg.dense_inter_size > 0 else self._inter_size
        return {
            "swa_n_kv_per_gpu": swa_n_kv_per_gpu,
            "global_n_kv_per_gpu": global_n_kv_per_gpu,
            "swa_qkv_out": self._num_heads * swa_hd // tp_size + swa_n_kv_per_gpu * (swa_hd + swa_v_hd),
            "global_qkv_out": self._num_heads * self._head_size // tp_size
            + global_n_kv_per_gpu * (self._head_size + global_v_hd),
            "swa_proj_in": self._num_heads * swa_v_hd // tp_size,
            "global_proj_in": self._num_heads * global_v_hd // tp_size,
            "swa_hd": swa_hd,
            "global_hd": self._head_size,
            "dense_inter_per_tp": dense_inter // tp_size,
        }

    def _moe_ops(
        self,
        prefix: str,
        count: float,
        h: int,
        moe_tp: int,
        moe_ep: int,
        attn_dp: int,
        moe_q: common.MoEQuantMode,
        wl_dist: str,
    ) -> list:
        """Return the three MoE FFN ops (pre-dispatch, compute, post-dispatch)."""
        router_ops = (
            [ops.GEMM(f"{prefix}_router_gemm", count, self._num_experts, h, common.GEMMQuantMode.float16)]
            if self._num_experts >= 128
            else []
        )
        return router_ops + [
            ops.MoEDispatch(
                f"{prefix}_moe_pre_dispatch", count, h, self._topk, self._num_experts, moe_tp, moe_ep, attn_dp, True
            ),
            ops.MoE(
                f"{prefix}_moe",
                count,
                h,
                self._moe_inter_size,
                self._topk,
                self._num_experts,
                moe_tp,
                moe_ep,
                moe_q,
                wl_dist,
                attn_dp,
            ),
            ops.MoEDispatch(
                f"{prefix}_moe_post_dispatch", count, h, self._topk, self._num_experts, moe_tp, moe_ep, attn_dp, False
            ),
        ]

    def _dense_ffn_ops(
        self, prefix: str, count: float, h: int, tp: int, dense_inter_per_tp: int, gemm_q: common.GEMMQuantMode
    ) -> list:
        """Return fused gate_up + activation + down ops for dense SwiGLU FFN."""
        return [
            ops.GEMM(f"{prefix}_dense_gate_up_gemm", count, 2 * dense_inter_per_tp, h, gemm_q),
            ops.ElementWise(f"{prefix}_dense_act", count, 2 * dense_inter_per_tp, dense_inter_per_tp, 0.8),
            ops.GEMM(f"{prefix}_dense_down_gemm", count, h, dense_inter_per_tp, gemm_q, low_precision_input=True),
        ]

    def _build_context_ops(self) -> None:
        """Build the context (prefill) operations for all four layer types."""
        if not self._hybrid_config:
            return

        cfg = self._hybrid_config
        counts = self._count_layer_types()
        h = self._hidden_size
        tp = self.config.tp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        pp = self.config.pp_size
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        fmha_q = self.config.fmha_quant_mode
        moe_q = self.config.moe_quant_mode
        wl_dist = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        d = self._resolve_dims(tp)

        self.context_ops = [ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3)]

        # --- global attention + MoE FFN ---
        if counts["global_moe"] > 0:
            c = counts["global_moe"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_global_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_global_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=0,
                        head_size=d["global_hd"],
                    ),
                    ops.GEMM("context_global_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_global_moe_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._moe_ops("context_global", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        # --- SWA/local attention + MoE FFN ---
        if counts["swa_moe"] > 0:
            c = counts["swa_moe"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_swa_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_swa_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                    ),
                    ops.GEMM("context_swa_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_swa_moe_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._moe_ops("context_swa", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        # --- SWA/local attention + dense FFN ---
        if counts["swa_dense"] > 0:
            c = counts["swa_dense"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_swa_dense_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_swa_dense_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                    ),
                    ops.GEMM("context_swa_dense_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("context_swa_dense_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._dense_ffn_ops("context_swa", c, h, tp, d["dense_inter_per_tp"], gemm_q)
            )

        # --- global attention + dense FFN ---
        if counts["global_dense"] > 0:
            c = counts["global_dense"]
            self.context_ops.extend(
                [
                    ops.ElementWise("context_global_dense_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("context_global_dense_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.ContextAttention(
                        "context_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        fmha_q,
                        window_size=0,
                        head_size=d["global_hd"],
                    ),
                    ops.GEMM(
                        "context_global_dense_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True
                    ),
                    ops.ElementWise("context_global_dense_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._dense_ffn_ops("context_global", c, h, tp, d["dense_inter_per_tp"], gemm_q)
            )

        self.context_ops.extend(
            [
                ops.GEMM("context_logits_gemm", 1, self._vocab_size // tp, h, common.GEMMQuantMode.float16),
                ops.P2P("context_p2p", pp - 1, h, pp),
            ]
        )

    def _build_generation_ops(self) -> None:
        """Build the generation (decoding) operations for all four layer types.

        All generation op counts are scaled by _mtp_scale_factor to account for
        multi-token prediction (nextn > 0), mirroring MOEModel's behavior.
        """
        if not self._hybrid_config:
            return

        cfg = self._hybrid_config
        counts = self._count_layer_types()
        sf = self._mtp_scale_factor
        h = self._hidden_size
        tp = self.config.tp_size
        moe_tp = self.config.moe_tp_size
        moe_ep = self.config.moe_ep_size
        attn_dp = self.config.attention_dp_size
        pp = self.config.pp_size
        gemm_q = self.config.gemm_quant_mode
        kvcache_q = self.config.kvcache_quant_mode
        moe_q = self.config.moe_quant_mode
        wl_dist = (
            self.config.workload_distribution + f"_{self._power_law_alpha}"
            if self.config.workload_distribution == "power_law"
            else self.config.workload_distribution
        )
        d = self._resolve_dims(tp)

        self.generation_ops = [ops.Embedding("generation_embedding", 1 * sf, self._vocab_size, h, 0.3)]

        # --- global attention + MoE FFN ---
        if counts["global_moe"] > 0:
            c = counts["global_moe"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_global_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_global_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=0,
                        head_size=d["global_hd"],
                    ),
                    ops.GEMM(
                        "generation_global_proj_gemm", c, h, d["global_proj_in"], gemm_q, low_precision_input=True
                    ),
                    ops.ElementWise("generation_global_moe_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._moe_ops("generation_global", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        # --- SWA/local attention + MoE FFN ---
        if counts["swa_moe"] > 0:
            c = counts["swa_moe"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_swa_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_swa_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                    ),
                    ops.GEMM("generation_swa_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True),
                    ops.ElementWise("generation_swa_moe_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._moe_ops("generation_swa", c, h, moe_tp, moe_ep, attn_dp, moe_q, wl_dist)
            )

        # --- SWA/local attention + dense FFN ---
        if counts["swa_dense"] > 0:
            c = counts["swa_dense"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_swa_dense_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_swa_dense_qkv_gemm", c, d["swa_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["swa_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=cfg.sliding_window_size,
                        head_size=d["swa_hd"],
                    ),
                    ops.GEMM(
                        "generation_swa_dense_proj_gemm", c, h, d["swa_proj_in"], gemm_q, low_precision_input=True
                    ),
                    ops.ElementWise("generation_swa_dense_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._dense_ffn_ops("generation_swa", c, h, tp, d["dense_inter_per_tp"], gemm_q)
            )

        # --- global attention + dense FFN ---
        if counts["global_dense"] > 0:
            c = counts["global_dense"] * sf
            self.generation_ops.extend(
                [
                    ops.ElementWise("generation_global_dense_attn_norm", c, 2 * h, 2 * h, 0.8),
                    ops.GEMM("generation_global_dense_qkv_gemm", c, d["global_qkv_out"], h, gemm_q),
                    ops.GenerationAttention(
                        "generation_attention",
                        c,
                        self._num_heads // tp,
                        d["global_n_kv_per_gpu"],
                        kvcache_q,
                        window_size=0,
                        head_size=d["global_hd"],
                    ),
                    ops.GEMM(
                        "generation_global_dense_proj_gemm",
                        c,
                        h,
                        d["global_proj_in"],
                        gemm_q,
                        low_precision_input=True,
                    ),
                    ops.ElementWise("generation_global_dense_ffn_norm", c, 2 * h, 2 * h, 0.8),
                ]
                + self._dense_ffn_ops("generation_global", c, h, tp, d["dense_inter_per_tp"], gemm_q)
            )

        self.generation_ops.extend(
            [
                ops.GEMM("generation_logits_gemm", 1 * sf, self._vocab_size // tp, h, common.GEMMQuantMode.float16),
                ops.P2P("generation_p2p", (pp - 1) * sf, h, pp),
            ]
        )
