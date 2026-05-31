// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NemotronH hybrid (Mamba + attention + MoE + MLP) op-list builder.
//!
//! Mirrors `aiconfigurator.sdk.models.nemotron_h.NemotronHModel`. The
//! `hybrid_override_pattern` string indicates per-layer type:
//!   * `M` — Mamba2 state-space layer
//!   * `*` — Standard transformer (GQA) layer
//!   * `E` — MoE layer (with shared expert)
//!   * `-` — Dense MLP layer (non-gated up + relu2 + down)
//!
//! Each unique layer type is emitted once with `scale_factor` equal to
//! its count in the pattern, matching Python.

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, DispatchFlavor, ElementwiseOp, EmbeddingOp, GemmOp,
    GenerationAttentionOp, Mamba2Op, MoEDispatchOp, MoeOp, Op, P2POp,
};

fn dispatch_flavor(backend: BackendKind) -> DispatchFlavor {
    match backend {
        BackendKind::Trtllm => DispatchFlavor::TrtllmAlltoall,
        BackendKind::Sglang => DispatchFlavor::CustomAllReduce,
        BackendKind::Vllm => DispatchFlavor::CustomAllReduce,
    }
}

pub fn build_nemotron_h_model(config: ModelConfig) -> Model {
    let mut model = Model::new(config);
    let cfg = &model.config;
    let h = cfg.spec.hidden_size;
    let tp = cfg.parallel.tp_size.max(1);
    let pp = cfg.parallel.pp_size.max(1);
    let moe_tp = cfg.parallel.moe_tp_size.max(1);
    let moe_ep = cfg.parallel.moe_ep_size.max(1);
    let attn_dp = cfg.parallel.attention_dp_size.max(1);
    let backend = cfg.backend;
    let flavor = dispatch_flavor(backend);
    let head_size = cfg.spec.head_dim;
    let num_heads = cfg.spec.num_attention_heads;
    let num_kv_heads = cfg.spec.num_key_value_heads;
    let vocab_per_tp = cfg.spec.vocab_size / tp;
    let inter_per_tp = cfg.spec.intermediate_size / tp;
    let dtypes = cfg.dtypes;
    let topk = cfg.spec.top_k;
    let num_experts = cfg.spec.num_experts;
    let moe_inter = cfg.spec.moe_intermediate_size;

    let nh_cfg = cfg
        .spec
        .nemotron_h
        .as_ref()
        .expect("build_nemotron_h_model: HfModelConfig missing `nemotron_h` extras");
    let pattern = &nh_cfg.hybrid_override_pattern;
    let count_m = pattern.chars().filter(|c| *c == 'M').count() as f64;
    let count_star = pattern.chars().filter(|c| *c == '*').count() as f64;
    let count_e = pattern.chars().filter(|c| *c == 'E').count() as f64;
    let count_dash = pattern.chars().filter(|c| *c == '-').count() as f64;

    let n_q_per_tp = num_heads / tp;
    let n_kv_per_tp = (num_kv_heads + tp - 1) / tp;
    let qkv_n = n_q_per_tp * head_size + 2 * n_kv_per_tp * head_size;

    // Mamba2 per-tp dims
    let nheads_per_tp = nh_cfg.mamba_num_heads / tp;
    let d_inner_per_tp = nheads_per_tp * nh_cfg.mamba_head_dim;
    let n_groups_per_tp = nh_cfg.n_groups / tp;
    let in_proj_out_per_tp =
        2 * d_inner_per_tp + 2 * n_groups_per_tp * nh_cfg.ssm_state_size + nheads_per_tp;

    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    let act_gate_bytes = (2.0 * inter_per_tp as f64 + inter_per_tp as f64) * 2.0;
    let _ = act_gate_bytes; // dense FFN uses non-gated MLP, see below

    let mut ctx = Vec::new();
    let mut gen = Vec::new();

    ctx.push(Op::Embedding({
        let mut e = EmbeddingOp::new("context_embedding", cfg.spec.vocab_size, h, dtypes.gemm_quant);
        e.scale_factor = 0.3;
        e
    }));
    gen.push(Op::Embedding({
        let mut e = EmbeddingOp::new("generation_embedding", cfg.spec.vocab_size, h, dtypes.gemm_quant);
        e.scale_factor = 0.3;
        e
    }));

    // ---- Mamba layers (M) ----
    if count_m > 0.0 {
        ctx.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("context_mamba_norm", norm_bytes);
            e.scale_factor = count_m;
            e
        }));
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new(
                "context_mamba_in_proj_gemm",
                in_proj_out_per_tp,
                h,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_m;
            g
        }));
        ctx.push(Op::Mamba2(Mamba2Op {
            name: "context_mamba_conv1d".to_string(),
            scale_factor: count_m,
            kernel_source: "causal_conv1d_fn".to_string(),
            phase: "context".to_string(),
            d_model: h,
            d_state: nh_cfg.ssm_state_size,
            d_conv: nh_cfg.conv_kernel,
            nheads: nh_cfg.mamba_num_heads,
            head_dim: nh_cfg.mamba_head_dim,
            n_groups: nh_cfg.n_groups,
            chunk_size: nh_cfg.chunk_size,
        }));
        ctx.push(Op::Mamba2(Mamba2Op {
            name: "context_mamba_ssm".to_string(),
            scale_factor: count_m,
            kernel_source: "mamba_chunk_scan_combined".to_string(),
            phase: "context".to_string(),
            d_model: h,
            d_state: nh_cfg.ssm_state_size,
            d_conv: nh_cfg.conv_kernel,
            nheads: nh_cfg.mamba_num_heads,
            head_dim: nh_cfg.mamba_head_dim,
            n_groups: nh_cfg.n_groups,
            chunk_size: nh_cfg.chunk_size,
        }));
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new("context_mamba_out_proj_gemm", h, d_inner_per_tp, dtypes.gemm_quant);
            g.scale_factor = count_m;
            g
        }));
        ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "context_mamba_ar",
            count_m,
            h,
            tp,
        )));

        gen.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("generation_mamba_norm", norm_bytes);
            e.scale_factor = count_m;
            e
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_mamba_in_proj_gemm",
                in_proj_out_per_tp,
                h,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_m;
            g
        }));
        gen.push(Op::Mamba2(Mamba2Op {
            name: "generation_mamba_conv1d".to_string(),
            scale_factor: count_m,
            kernel_source: "causal_conv1d_update".to_string(),
            phase: "generation".to_string(),
            d_model: h,
            d_state: nh_cfg.ssm_state_size,
            d_conv: nh_cfg.conv_kernel,
            nheads: nh_cfg.mamba_num_heads,
            head_dim: nh_cfg.mamba_head_dim,
            n_groups: nh_cfg.n_groups,
            chunk_size: nh_cfg.chunk_size,
        }));
        gen.push(Op::Mamba2(Mamba2Op {
            name: "generation_mamba_ssm".to_string(),
            scale_factor: count_m,
            kernel_source: "selective_state_update".to_string(),
            phase: "generation".to_string(),
            d_model: h,
            d_state: nh_cfg.ssm_state_size,
            d_conv: nh_cfg.conv_kernel,
            nheads: nh_cfg.mamba_num_heads,
            head_dim: nh_cfg.mamba_head_dim,
            n_groups: nh_cfg.n_groups,
            chunk_size: nh_cfg.chunk_size,
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_mamba_out_proj_gemm",
                h,
                d_inner_per_tp,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_m;
            g
        }));
        gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "generation_mamba_ar",
            count_m,
            h,
            tp,
        )));
    }

    // ---- Transformer layers (*) ----
    if count_star > 0.0 {
        ctx.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("context_attn_norm", norm_bytes);
            e.scale_factor = count_star;
            e
        }));
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new("context_qkv_gemm", qkv_n, h, dtypes.gemm_quant);
            g.scale_factor = count_star;
            g
        }));
        ctx.push(Op::ContextAttention({
            let mut a = ContextAttentionOp::new(
                "context_attention",
                n_q_per_tp,
                n_kv_per_tp,
                head_size,
                dtypes.kv_cache_quant,
                dtypes.fmha_quant,
            );
            a.scale_factor = count_star;
            a.use_qk_norm = cfg.spec.use_qk_norm;
            a
        }));
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new(
                "context_proj_gemm",
                h,
                n_q_per_tp * head_size,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_star;
            g.low_precision_input = true;
            g
        }));
        ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "context_attn_ar",
            count_star,
            h,
            tp,
        )));

        gen.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("generation_attn_norm", norm_bytes);
            e.scale_factor = count_star;
            e
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new("generation_qkv_gemm", qkv_n, h, dtypes.gemm_quant);
            g.scale_factor = count_star;
            g
        }));
        gen.push(Op::GenerationAttention({
            let mut a = GenerationAttentionOp::new(
                "generation_attention",
                n_q_per_tp,
                n_kv_per_tp,
                head_size,
                dtypes.kv_cache_quant,
            );
            a.scale_factor = count_star;
            a
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_proj_gemm",
                h,
                n_q_per_tp * head_size,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_star;
            g.low_precision_input = true;
            g
        }));
        gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "generation_attn_ar",
            count_star,
            h,
            tp,
        )));
    }

    // ---- MoE layers (E) ----
    if count_e > 0.0 {
        let moe_h = if nh_cfg.moe_latent_size > 0 {
            nh_cfg.moe_latent_size
        } else {
            h
        };
        // Pre-norm
        ctx.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("context_moe_norm", norm_bytes);
            e.scale_factor = count_e;
            e
        }));
        // Shared expert (non-gated): up GEMM + Relu2 + down GEMM
        let shared_per_tp = nh_cfg.moe_shared_expert_intermediate_size / tp;
        let relu2_bytes = (shared_per_tp as f64 + shared_per_tp as f64) * 2.0;
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new("context_shared_up_gemm", shared_per_tp, h, dtypes.gemm_quant);
            g.scale_factor = count_e;
            g
        }));
        ctx.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("context_shared_relu2", relu2_bytes);
            e.scale_factor = count_e;
            e
        }));
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new("context_shared_down_gemm", h, shared_per_tp, dtypes.gemm_quant);
            g.scale_factor = count_e;
            g.low_precision_input = true;
            g
        }));
        // Router
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new(
                "context_router_gemm",
                num_experts,
                h,
                GemmQuantMode::Bfloat16,
            );
            g.scale_factor = count_e;
            g
        }));
        // Optional latent projection in
        if nh_cfg.moe_latent_size > 0 {
            ctx.push(Op::Gemm({
                let mut g = GemmOp::new(
                    "context_fc1_latent_proj_gemm",
                    nh_cfg.moe_latent_size / tp,
                    h,
                    dtypes.gemm_quant,
                );
                g.scale_factor = count_e;
                g
            }));
        }
        ctx.push(Op::MoeDispatch({
            let mut d = MoEDispatchOp::new(
                "context_moe_pre_dispatch",
                moe_h,
                topk,
                num_experts,
                moe_tp,
                moe_ep,
                attn_dp,
                true,
                backend,
                flavor,
            );
            d.scale_factor = count_e;
            d
        }));
        ctx.push(Op::Moe({
            let mut m = MoeOp::new(
                "context_moe",
                moe_h,
                moe_inter,
                topk,
                num_experts,
                moe_tp,
                moe_ep,
                dtypes.moe_quant,
                "power_law_1.01",
            );
            m.scale_factor = count_e;
            // Relu² is non-gated, so the small-token nvfp4 low-latency kernel doesn't apply.
            m.is_gated = false;
            m
        }));
        ctx.push(Op::MoeDispatch({
            let mut d = MoEDispatchOp::new(
                "context_moe_post_dispatch",
                moe_h,
                topk,
                num_experts,
                moe_tp,
                moe_ep,
                attn_dp,
                false,
                backend,
                flavor,
            );
            d.scale_factor = count_e;
            d
        }));
        if nh_cfg.moe_latent_size > 0 {
            ctx.push(Op::Gemm({
                let mut g = GemmOp::new(
                    "context_fc2_latent_proj_gemm",
                    h,
                    nh_cfg.moe_latent_size / tp,
                    dtypes.gemm_quant,
                );
                g.scale_factor = count_e;
                g.low_precision_input = true;
                g
            }));
        }
        ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "context_moe_ar",
            count_e,
            h,
            tp,
        )));

        // Mirror for generation.
        gen.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("generation_moe_norm", norm_bytes);
            e.scale_factor = count_e;
            e
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_shared_up_gemm",
                shared_per_tp,
                h,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_e;
            g
        }));
        gen.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("generation_shared_relu2", relu2_bytes);
            e.scale_factor = count_e;
            e
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_shared_down_gemm",
                h,
                shared_per_tp,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_e;
            g.low_precision_input = true;
            g
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_router_gemm",
                num_experts,
                h,
                GemmQuantMode::Bfloat16,
            );
            g.scale_factor = count_e;
            g
        }));
        if nh_cfg.moe_latent_size > 0 {
            gen.push(Op::Gemm({
                let mut g = GemmOp::new(
                    "generation_fc1_latent_proj_gemm",
                    nh_cfg.moe_latent_size / tp,
                    h,
                    dtypes.gemm_quant,
                );
                g.scale_factor = count_e;
                g
            }));
        }
        gen.push(Op::MoeDispatch({
            let mut d = MoEDispatchOp::new(
                "generation_moe_pre_dispatch",
                moe_h,
                topk,
                num_experts,
                moe_tp,
                moe_ep,
                attn_dp,
                true,
                backend,
                flavor,
            );
            d.scale_factor = count_e;
            d
        }));
        gen.push(Op::Moe({
            let mut m = MoeOp::new(
                "generation_moe",
                moe_h,
                moe_inter,
                topk,
                num_experts,
                moe_tp,
                moe_ep,
                dtypes.moe_quant,
                "power_law_1.01",
            );
            m.scale_factor = count_e;
            // Relu² is non-gated, so the small-token nvfp4 low-latency kernel doesn't apply.
            m.is_gated = false;
            m
        }));
        gen.push(Op::MoeDispatch({
            let mut d = MoEDispatchOp::new(
                "generation_moe_post_dispatch",
                moe_h,
                topk,
                num_experts,
                moe_tp,
                moe_ep,
                attn_dp,
                false,
                backend,
                flavor,
            );
            d.scale_factor = count_e;
            d
        }));
        if nh_cfg.moe_latent_size > 0 {
            gen.push(Op::Gemm({
                let mut g = GemmOp::new(
                    "generation_fc2_latent_proj_gemm",
                    h,
                    nh_cfg.moe_latent_size / tp,
                    dtypes.gemm_quant,
                );
                g.scale_factor = count_e;
                g.low_precision_input = true;
                g
            }));
        }
        gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "generation_moe_ar",
            count_e,
            h,
            tp,
        )));
    }

    // ---- MLP layers (-) — non-gated up + Relu2 + down ----
    if count_dash > 0.0 {
        let mlp_relu2_bytes = (inter_per_tp as f64 + inter_per_tp as f64) * 2.0;

        ctx.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("context_mlp_norm", norm_bytes);
            e.scale_factor = count_dash;
            e
        }));
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new("context_mlp_up_gemm", inter_per_tp, h, dtypes.gemm_quant);
            g.scale_factor = count_dash;
            g
        }));
        ctx.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("context_mlp_relu2", mlp_relu2_bytes);
            e.scale_factor = count_dash;
            e
        }));
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new("context_mlp_down_gemm", h, inter_per_tp, dtypes.gemm_quant);
            g.scale_factor = count_dash;
            g
        }));
        ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "context_mlp_ar",
            count_dash,
            h,
            tp,
        )));

        gen.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("generation_mlp_norm", norm_bytes);
            e.scale_factor = count_dash;
            e
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_mlp_up_gemm",
                inter_per_tp,
                h,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_dash;
            g
        }));
        gen.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("generation_mlp_relu2", mlp_relu2_bytes);
            e.scale_factor = count_dash;
            e
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_mlp_down_gemm",
                h,
                inter_per_tp,
                dtypes.gemm_quant,
            );
            g.scale_factor = count_dash;
            g
        }));
        gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "generation_mlp_ar",
            count_dash,
            h,
            tp,
        )));
    }

    // ---- P2P + logits ----
    let pp_scale = (pp as f64 - 1.0).max(0.0);
    let mut p2p_ctx = P2POp::new("context_p2p", pp, h);
    p2p_ctx.scale_factor = pp_scale;
    ctx.push(Op::P2P(p2p_ctx));
    ctx.push(Op::Gemm(GemmOp::new(
        "context_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));
    let mut p2p_gen = P2POp::new("generation_p2p", pp, h);
    p2p_gen.scale_factor = pp_scale;
    gen.push(Op::P2P(p2p_gen));
    gen.push(Op::Gemm(GemmOp::new(
        "generation_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));

    model.context_ops = ctx;
    model.generation_ops = gen;
    model
}
