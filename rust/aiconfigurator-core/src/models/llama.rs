// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLAMA-family op-list builder (dense FFN, GQA attention).

use crate::common::enums::GemmQuantMode;
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, ElementwiseOp, EmbeddingOp, GemmOp,
    GenerationAttentionOp, Op,
};

pub fn build_llama_model(config: ModelConfig) -> Model {
    let mut model = Model::new(config);
    let cfg = &model.config;
    let layers = cfg.spec.num_hidden_layers as f64;
    let h = cfg.spec.hidden_size;
    let inter = cfg.spec.intermediate_size;
    let tp = cfg.parallel.tp_size.max(1);
    let kv_per_gpu = cfg.kv_heads_per_gpu();
    let head_size = cfg.spec.head_dim;
    let n_per_tp = cfg.spec.num_attention_heads / tp;
    let qkv_n = n_per_tp * head_size + 2 * kv_per_gpu * head_size;
    let inter_per_tp = inter / tp;
    let vocab_per_tp = cfg.spec.vocab_size / tp;
    let dtypes = cfg.dtypes;
    // Python: ElementWise(num_layers, dim_in=2h, dim_out=2h) → (2h + 2h) * 2 bytes/token.
    // The 0.8 in Python's constructor is stored but unused at query time.
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;

    let mut ctx = Vec::with_capacity(12);

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
        a.use_qk_norm = cfg.spec.use_qk_norm;
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
    ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "context_attn_ar",
        layers,
        h,
        tp,
    )));
    // Gated FFN: ffn1 produces 2 * inter (silu+up), activation gate reduces
    // to inter, ffn2 reduces back to h. Mirrors Python `llama.py`'s
    // `context_gate_ffn1_gemm` + `context_act_gate` + `context_ffn2_gemm`.
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_ffn1_gemm", 2 * inter_per_tp, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    // Activation gate: 2*inter (read) + inter (write) bytes per token in bf16.
    let act_gate_bytes = (2.0 * inter_per_tp as f64 + inter_per_tp as f64) * 2.0;
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_act_gate", act_gate_bytes);
        e.scale_factor = layers;
        e
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_ffn2_gemm", h, inter_per_tp, dtypes.gemm_quant);
        g.scale_factor = layers;
        g.low_precision_input = true;
        g
    }));
    // Logits projection (bf16, vocab × hidden, runs once per forward).
    ctx.push(Op::Gemm(GemmOp::new(
        "context_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));
    ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "context_ffn_ar",
        layers,
        h,
        tp,
    )));
    ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "context_embedding_ar",
        1.0,
        h,
        tp,
    )));

    model.context_ops = ctx;

    let mut gen = Vec::with_capacity(12);

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
    gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "generation_attn_ar",
        layers,
        h,
        tp,
    )));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new("generation_ffn1_gemm", 2 * inter_per_tp, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    // Activation gate (mirror context phase).
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_act_gate", act_gate_bytes);
        e.scale_factor = layers;
        e
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new("generation_ffn2_gemm", h, inter_per_tp, dtypes.gemm_quant);
        g.scale_factor = layers;
        g.low_precision_input = true;
        g
    }));
    gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "generation_ffn_ar",
        layers,
        h,
        tp,
    )));
    // Logits projection (bf16 in Python).
    gen.push(Op::Gemm(GemmOp::new(
        "generation_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));
    gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "generation_embedding_ar",
        1.0,
        h,
        tp,
    )));

    model.generation_ops = gen;
    model
}
