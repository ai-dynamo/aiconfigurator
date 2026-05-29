// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gemma-4 hybrid MoE family op-list builder.
//!
//! Apple-to-apple port of `aiconfigurator.sdk.models.gemma4_moe.Gemma4MoEModel`.
//! Two attention buckets driven by `layer_types`:
//!
//! - `sliding_attention` (SWA): GQA with `swa_num_kv_heads` x
//!   `swa_head_dim`, separate K and V projections, token window =
//!   `sliding_window_size`.
//! - `full_attention` (global): GQA with `global_num_kv_heads` x
//!   `global_head_dim`, no window. When `attention_k_eq_v=true`, V is
//!   reused from K's projection output, so the QKV-out width collapses
//!   from Q+2KV to Q+K (no v_proj weight or v_proj GEMM on these layers).
//!
//! Every layer runs BOTH a shared dense MLP (gated SwiGLU at
//! `inter_size`) AND a routed top-k MoE (at `moe_inter_size`) in
//! parallel — the two FFN branches are summed before the
//! post-feedforward norm. The router GEMM is emitted only when
//! `num_experts >= 128` (same threshold as MoE / HybridMoE).

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::common::error::AicError;
use crate::models::base::{Model, ModelConfig};
use crate::models::config_loader::Gemma4MoEConfig;
use crate::operators::{
    ContextAttentionOp, DispatchFlavor, ElementwiseOp, EmbeddingOp, GemmOp,
    GenerationAttentionOp, MoEDispatchOp, MoeOp, Op, P2POp,
};

fn dispatch_flavor(backend: BackendKind) -> DispatchFlavor {
    match backend {
        BackendKind::Trtllm => DispatchFlavor::TrtllmAlltoall,
        BackendKind::Sglang => DispatchFlavor::CustomAllReduce,
        BackendKind::Vllm => DispatchFlavor::CustomAllReduce,
    }
}

#[derive(Default)]
struct LayerCounts {
    swa: u32,
    global: u32,
}

fn count_layer_kinds(cfg: &Gemma4MoEConfig) -> LayerCounts {
    let mut c = LayerCounts::default();
    for lt in &cfg.layer_types {
        match lt.as_str() {
            "sliding_attention" => c.swa += 1,
            "full_attention" => c.global += 1,
            _ => {}
        }
    }
    c
}

struct ResolvedDims {
    swa_n_kv_per_gpu: u32,
    global_n_kv_per_gpu: u32,
    swa_qkv_out: u32,
    global_qkv_out: u32,
    swa_proj_in: u32,
    global_proj_in: u32,
    swa_hd: u32,
    global_hd: u32,
    dense_inter_per_tp: u32,
}

fn resolve_dims(cfg: &Gemma4MoEConfig, num_heads: u32, inter: u32, tp: u32) -> ResolvedDims {
    let swa_n_kv_per_gpu = (cfg.swa_num_kv_heads + tp - 1) / tp;
    let global_n_kv_per_gpu = (cfg.global_num_kv_heads + tp - 1) / tp;
    let swa_qkv_out =
        num_heads * cfg.swa_head_dim / tp + swa_n_kv_per_gpu * cfg.swa_head_dim * 2;
    let global_qkv_out = if cfg.attention_k_eq_v {
        num_heads * cfg.global_head_dim / tp + global_n_kv_per_gpu * cfg.global_head_dim
    } else {
        num_heads * cfg.global_head_dim / tp + global_n_kv_per_gpu * cfg.global_head_dim * 2
    };
    ResolvedDims {
        swa_n_kv_per_gpu,
        global_n_kv_per_gpu,
        swa_qkv_out,
        global_qkv_out,
        swa_proj_in: num_heads * cfg.swa_head_dim / tp,
        global_proj_in: num_heads * cfg.global_head_dim / tp,
        swa_hd: cfg.swa_head_dim,
        global_hd: cfg.global_head_dim,
        dense_inter_per_tp: inter / tp,
    }
}

pub fn build_gemma4_moe_model(config: ModelConfig) -> Result<Model, AicError> {
    let gemma: Gemma4MoEConfig = config
        .spec
        .gemma4_moe
        .as_ref()
        .ok_or_else(|| {
            AicError::UnsupportedModel(
                "GEMMA4MOE builder requires gemma4_moe extras in ModelSpec".to_string(),
            )
        })?
        .clone();
    let mut model = Model::new(config);
    let cfg = &model.config;
    let h = cfg.spec.hidden_size;
    let inter = cfg.spec.intermediate_size;
    let tp = cfg.parallel.tp_size.max(1);
    let pp = cfg.parallel.pp_size.max(1);
    let vocab_per_tp = cfg.spec.vocab_size / tp;
    let dtypes = cfg.dtypes;
    let num_heads = cfg.spec.num_attention_heads;
    let num_experts = cfg.spec.num_experts.max(1);
    let topk = cfg.spec.top_k.max(1);
    let moe_inter = cfg.spec.moe_intermediate_size;
    let moe_tp = cfg.parallel.moe_tp_size.max(1);
    let moe_ep = cfg.parallel.moe_ep_size.max(1);
    let attn_dp = cfg.parallel.attention_dp_size.max(1);
    let backend = cfg.backend;
    let flavor = dispatch_flavor(backend);

    let counts = count_layer_kinds(&gemma);
    let dims = resolve_dims(&gemma, num_heads, inter, tp);
    let n_per_tp = num_heads / tp;

    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    let dense_act_bytes =
        (2.0 * dims.dense_inter_per_tp as f64 + dims.dense_inter_per_tp as f64) * 2.0;

    let push_shared_mlp = |ops_out: &mut Vec<Op>, prefix: &str, count_f: f64| {
        let mut gu = GemmOp::new(
            format!("{prefix}_shared_mlp_gate_up_gemm"),
            2 * dims.dense_inter_per_tp,
            h,
            dtypes.gemm_quant,
        );
        gu.scale_factor = count_f;
        ops_out.push(Op::Gemm(gu));
        let mut act = ElementwiseOp::new(format!("{prefix}_shared_mlp_act"), dense_act_bytes);
        act.scale_factor = count_f;
        ops_out.push(Op::Elementwise(act));
        let mut down = GemmOp::new(
            format!("{prefix}_shared_mlp_down_gemm"),
            h,
            dims.dense_inter_per_tp,
            dtypes.gemm_quant,
        );
        down.scale_factor = count_f;
        down.low_precision_input = true;
        ops_out.push(Op::Gemm(down));
    };

    let push_moe_block = |ops_out: &mut Vec<Op>, prefix: &str, count_f: f64| {
        if num_experts >= 128 {
            let mut g = GemmOp::new(
                format!("{prefix}_router_gemm"),
                num_experts,
                h,
                GemmQuantMode::Bfloat16,
            );
            g.scale_factor = count_f;
            ops_out.push(Op::Gemm(g));
        }
        let mut d_pre = MoEDispatchOp::new(
            format!("{prefix}_moe_pre_dispatch"),
            h,
            topk,
            num_experts,
            moe_tp,
            moe_ep,
            attn_dp,
            true,
            backend,
            flavor,
        );
        d_pre.scale_factor = count_f;
        ops_out.push(Op::MoeDispatch(d_pre));
        let mut m = MoeOp::new(
            format!("{prefix}_moe"),
            h,
            moe_inter,
            topk,
            num_experts,
            moe_tp,
            moe_ep,
            dtypes.moe_quant,
            "uniform",
        );
        m.scale_factor = count_f;
        ops_out.push(Op::Moe(m));
        let mut d_post = MoEDispatchOp::new(
            format!("{prefix}_moe_post_dispatch"),
            h,
            topk,
            num_experts,
            moe_tp,
            moe_ep,
            attn_dp,
            false,
            backend,
            flavor,
        );
        d_post.scale_factor = count_f;
        ops_out.push(Op::MoeDispatch(d_post));
    };

    let push_attention_ctx = |ops_out: &mut Vec<Op>, prefix: &str, count_f: f64, swa: bool| {
        let mut norm = ElementwiseOp::new(format!("{prefix}_attn_norm"), norm_bytes);
        norm.scale_factor = count_f;
        ops_out.push(Op::Elementwise(norm));
        let mut qkv = GemmOp::new(
            format!("{prefix}_qkv_gemm"),
            if swa { dims.swa_qkv_out } else { dims.global_qkv_out },
            h,
            dtypes.gemm_quant,
        );
        qkv.scale_factor = count_f;
        ops_out.push(Op::Gemm(qkv));
        let mut a = ContextAttentionOp::new(
            "context_attention",
            n_per_tp,
            if swa { dims.swa_n_kv_per_gpu } else { dims.global_n_kv_per_gpu },
            if swa { dims.swa_hd } else { dims.global_hd },
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
        );
        a.scale_factor = count_f;
        a.window_size = if swa { gemma.sliding_window_size } else { 0 };
        ops_out.push(Op::ContextAttention(a));
        let mut proj = GemmOp::new(
            format!("{prefix}_proj_gemm"),
            h,
            if swa { dims.swa_proj_in } else { dims.global_proj_in },
            dtypes.gemm_quant,
        );
        proj.scale_factor = count_f;
        proj.low_precision_input = true;
        ops_out.push(Op::Gemm(proj));
        let mut ffn_norm = ElementwiseOp::new(format!("{prefix}_ffn_norm"), norm_bytes);
        ffn_norm.scale_factor = count_f;
        ops_out.push(Op::Elementwise(ffn_norm));
    };

    let push_attention_gen = |ops_out: &mut Vec<Op>, prefix: &str, count_f: f64, swa: bool| {
        let mut norm = ElementwiseOp::new(format!("{prefix}_attn_norm"), norm_bytes);
        norm.scale_factor = count_f;
        ops_out.push(Op::Elementwise(norm));
        let mut qkv = GemmOp::new(
            format!("{prefix}_qkv_gemm"),
            if swa { dims.swa_qkv_out } else { dims.global_qkv_out },
            h,
            dtypes.gemm_quant,
        );
        qkv.scale_factor = count_f;
        ops_out.push(Op::Gemm(qkv));
        let mut a = GenerationAttentionOp::new(
            "generation_attention",
            n_per_tp,
            if swa { dims.swa_n_kv_per_gpu } else { dims.global_n_kv_per_gpu },
            if swa { dims.swa_hd } else { dims.global_hd },
            dtypes.kv_cache_quant,
        );
        a.scale_factor = count_f;
        a.window_size = if swa { gemma.sliding_window_size } else { 0 };
        ops_out.push(Op::GenerationAttention(a));
        let mut proj = GemmOp::new(
            format!("{prefix}_proj_gemm"),
            h,
            if swa { dims.swa_proj_in } else { dims.global_proj_in },
            dtypes.gemm_quant,
        );
        proj.scale_factor = count_f;
        proj.low_precision_input = true;
        ops_out.push(Op::Gemm(proj));
        let mut ffn_norm = ElementwiseOp::new(format!("{prefix}_ffn_norm"), norm_bytes);
        ffn_norm.scale_factor = count_f;
        ops_out.push(Op::Elementwise(ffn_norm));
    };

    // ---- Context phase ----
    let mut ctx = Vec::new();
    ctx.push(Op::Embedding({
        let mut e = EmbeddingOp::new("context_embedding", vocab_per_tp, h, dtypes.gemm_quant);
        e.scale_factor = 0.3;
        e
    }));
    if counts.swa > 0 {
        let c = counts.swa as f64;
        push_attention_ctx(&mut ctx, "context_swa", c, true);
        push_shared_mlp(&mut ctx, "context_swa", c);
        push_moe_block(&mut ctx, "context_swa", c);
    }
    if counts.global > 0 {
        let c = counts.global as f64;
        push_attention_ctx(&mut ctx, "context_global", c, false);
        push_shared_mlp(&mut ctx, "context_global", c);
        push_moe_block(&mut ctx, "context_global", c);
    }
    ctx.push(Op::Gemm(GemmOp::new(
        "context_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));
    let pp_scale = (pp as f64 - 1.0).max(0.0);
    let mut p2p_ctx = P2POp::new("context_p2p", pp, h);
    p2p_ctx.scale_factor = pp_scale;
    ctx.push(Op::P2P(p2p_ctx));

    // ---- Generation phase ----
    let mut gen = Vec::new();
    gen.push(Op::Embedding({
        let mut e = EmbeddingOp::new(
            "generation_embedding",
            vocab_per_tp,
            h,
            dtypes.gemm_quant,
        );
        e.scale_factor = 0.3;
        e
    }));
    if counts.swa > 0 {
        let c = counts.swa as f64;
        push_attention_gen(&mut gen, "generation_swa", c, true);
        push_shared_mlp(&mut gen, "generation_swa", c);
        push_moe_block(&mut gen, "generation_swa", c);
    }
    if counts.global > 0 {
        let c = counts.global as f64;
        push_attention_gen(&mut gen, "generation_global", c, false);
        push_shared_mlp(&mut gen, "generation_global", c);
        push_moe_block(&mut gen, "generation_global", c);
    }
    gen.push(Op::Gemm(GemmOp::new(
        "generation_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));
    let mut p2p_gen = P2POp::new("generation_p2p", pp, h);
    p2p_gen.scale_factor = pp_scale;
    gen.push(Op::P2P(p2p_gen));

    model.context_ops = ctx;
    model.generation_ops = gen;
    Ok(model)
}
