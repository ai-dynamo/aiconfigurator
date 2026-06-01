// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GPT-family op-list builder (dense FFN, GQA attention, non-gated MLP).
//!
//! Apple-to-apple port of `aiconfigurator.sdk.models.gpt.GPTModel`. The
//! shape differs from `llama.rs` in two places:
//!   - FFN is non-gated: `ffn1` projects `h -> inter`, then a single-act
//!     elementwise, then `ffn2` projects `inter -> h`. (Llama gates with
//!     `2 * inter` then halves through `act_gate`.)
//!   - AllReduce ops are only the two phase boundaries (`ar_1` after the
//!     attention block, `ar_2` after the FFN block) — no separate
//!     embedding allreduce.

use crate::common::enums::GemmQuantMode;
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, ElementwiseOp, EmbeddingOp, GemmOp,
    GenerationAttentionOp, Op, P2POp,
};

pub fn build_gpt_model(config: ModelConfig) -> Model {
    let mut model = Model::new(config);
    let cfg = &model.config;
    let layers = cfg.spec.num_hidden_layers as f64;
    let h = cfg.spec.hidden_size;
    let inter = cfg.spec.intermediate_size;
    let tp = cfg.parallel.tp_size.max(1);
    let pp = cfg.parallel.pp_size.max(1);
    let kv_per_gpu = cfg.kv_heads_per_gpu();
    let head_size = cfg.spec.head_dim;
    let n_per_tp = cfg.spec.num_attention_heads / tp;
    let qkv_n = n_per_tp * head_size + 2 * kv_per_gpu * head_size;
    let inter_per_tp = inter / tp;
    let vocab_per_tp = cfg.spec.vocab_size / tp;
    let dtypes = cfg.dtypes;

    // Python: ElementWise(num_layers, 2*h, 2*h) -> (2h+2h)*2 bytes/token.
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    // Non-gated activation: read inter, write inter, both in bf16 (2 bytes).
    let act_bytes = (inter_per_tp as f64 + inter_per_tp as f64) * 2.0;

    let mut ctx = Vec::with_capacity(13);

    ctx.push(Op::Embedding({
        let mut e = EmbeddingOp::new("context_embedding", vocab_per_tp, h, dtypes.gemm_quant);
        e.scale_factor = 0.3;
        e
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_add_norm_1", norm_bytes);
        e.scale_factor = layers;
        e
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_qkv_gemm", qkv_n, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::ContextAttention({
        let mut a = ContextAttentionOp::new(
            "context_attention",
            n_per_tp,
            kv_per_gpu,
            head_size,
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
        );
        a.scale_factor = layers;
        a
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_proj_gemm", h, n_per_tp * head_size, dtypes.gemm_quant);
        g.scale_factor = layers;
        g.low_precision_input = true;
        g
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_add_norm_2", norm_bytes);
        e.scale_factor = layers;
        e
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_ffn1_gemm", inter_per_tp, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_act", act_bytes);
        e.scale_factor = layers;
        e
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_ffn2_gemm", h, inter_per_tp, dtypes.gemm_quant);
        g.scale_factor = layers;
        g.low_precision_input = true;
        g
    }));
    // Logits projection (bf16, vocab/tp x hidden, runs once per forward).
    ctx.push(Op::Gemm(GemmOp::new(
        "context_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));
    ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "context_ar_1",
        layers,
        h,
        tp,
    )));
    ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "context_ar_2",
        layers,
        h,
        tp,
    )));
    // Pipeline-parallel P2P: pp_scale_factor = pp_size - 1 (zero when pp=1).
    let pp_scale = (pp.saturating_sub(1)) as f64;
    let mut p2p_ctx = P2POp::new("context_p2p", pp, h);
    p2p_ctx.scale_factor = pp_scale;
    ctx.push(Op::P2P(p2p_ctx));

    model.context_ops = ctx;

    let mut gen = Vec::with_capacity(13);

    gen.push(Op::Embedding({
        let mut e = EmbeddingOp::new("generation_embedding", vocab_per_tp, h, dtypes.gemm_quant);
        e.scale_factor = 0.3;
        e
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_add_norm_1", norm_bytes);
        e.scale_factor = layers;
        e
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new("generation_qkv_gemm", qkv_n, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    gen.push(Op::GenerationAttention({
        let mut a = GenerationAttentionOp::new(
            "generation_attention",
            n_per_tp,
            kv_per_gpu,
            head_size,
            dtypes.kv_cache_quant,
        );
        a.scale_factor = layers;
        a
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_proj_gemm",
            h,
            n_per_tp * head_size,
            dtypes.gemm_quant,
        );
        g.scale_factor = layers;
        g.low_precision_input = true;
        g
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_add_norm_2", norm_bytes);
        e.scale_factor = layers;
        e
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new("generation_ffn1_gemm", inter_per_tp, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_act", act_bytes);
        e.scale_factor = layers;
        e
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new("generation_ffn2_gemm", h, inter_per_tp, dtypes.gemm_quant);
        g.scale_factor = layers;
        g.low_precision_input = true;
        g
    }));
    gen.push(Op::Gemm(GemmOp::new(
        "generation_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));
    gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "generation_ar_1",
        layers,
        h,
        tp,
    )));
    gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "generation_ar_2",
        layers,
        h,
        tp,
    )));
    let mut p2p_gen = P2POp::new("generation_p2p", pp, h);
    p2p_gen.scale_factor = pp_scale;
    gen.push(Op::P2P(p2p_gen));

    model.generation_ops = gen;
    model
}
