// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NemotronNas (Puzzle / DeciLM) op-list builder.
//!
//! Mirrors `aiconfigurator.sdk.models.nemotron_nas.NemotronNas`. The
//! architecture is heterogeneous: each layer is described by a
//! `BlockConfig` and may skip attention, FFN, or both. The block list
//! is parsed from HF `config["block_configs"]` upstream in
//! `config_loader::parse_block_configs`; here we iterate it and push
//! the appropriate per-block op chain.
//!
//! Python groups consecutive identical blocks into run-length entries;
//! we iterate per layer (count=1) instead because the total latency is
//! invariant to the grouping (N * sum_layer == sum_layer * N).

use crate::common::enums::GemmQuantMode;
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, ElementwiseOp, EmbeddingOp, GemmOp,
    GenerationAttentionOp, Op, P2POp,
};

/// Python's `_ffn_mult_to_intermediate_size`: `int(2 * mult * h / 3)`,
/// rounded up to the next multiple of 256.
fn ffn_mult_to_intermediate(ffn_mult: f64, hidden_size: u32) -> u32 {
    let raw = (2.0 * ffn_mult * hidden_size as f64 / 3.0) as u32;
    if raw % 256 == 0 {
        raw
    } else {
        raw + 256 - (raw % 256)
    }
}

pub fn build_nemotron_nas_model(config: ModelConfig) -> Model {
    let mut model = Model::new(config);
    let cfg = &model.config;
    let h = cfg.spec.hidden_size;
    let tp = cfg.parallel.tp_size.max(1);
    let pp = cfg.parallel.pp_size.max(1);
    let head_size = cfg.spec.head_dim;
    let num_heads = cfg.spec.num_attention_heads;
    let vocab_per_tp = cfg.spec.vocab_size / tp;
    let dtypes = cfg.dtypes;
    let block_configs = &cfg.spec.block_configs;

    // Python: ElementWise(count, dim_in=2h, dim_out=2h) → (2h+2h)*2 bytes/token.
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;

    let mut ctx = Vec::with_capacity(block_configs.len() * 8 + 4);
    let mut gen = Vec::with_capacity(block_configs.len() * 8 + 4);

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

    for block in block_configs {
        let count = block.num_inst as f64;

        // ---- Attention sub-block ----
        if !block.attn_no_op {
            // Python: KV count = num_attention_heads / attn_n_heads_in_group;
            // per-GPU = ceil(KV / tp).
            let group = block.attn_n_heads_in_group.max(1);
            let num_kv_heads = num_heads / group;
            let kv_per_gpu = (num_kv_heads + tp - 1) / tp;
            let n_per_tp = num_heads / tp;
            let qkv_n = n_per_tp * head_size + 2 * kv_per_gpu * head_size;

            ctx.push(Op::Elementwise({
                let mut e = ElementwiseOp::new("context_add_norm_1", norm_bytes);
                e.scale_factor = count;
                e
            }));
            ctx.push(Op::Gemm({
                let mut g = GemmOp::new("context_qkv_gemm", qkv_n, h, dtypes.gemm_quant);
                g.scale_factor = count;
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
                a.scale_factor = count;
                a
            }));
            ctx.push(Op::Gemm({
                let mut g = GemmOp::new(
                    "context_proj_gemm",
                    h,
                    n_per_tp * head_size,
                    dtypes.gemm_quant,
                );
                g.scale_factor = count;
                g
            }));
            ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
                "context_ar_1",
                count,
                h,
                tp,
            )));

            gen.push(Op::Elementwise({
                let mut e = ElementwiseOp::new("generation_add_norm_1", norm_bytes);
                e.scale_factor = count;
                e
            }));
            gen.push(Op::Gemm({
                let mut g = GemmOp::new("generation_qkv_gemm", qkv_n, h, dtypes.gemm_quant);
                g.scale_factor = count;
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
                a.scale_factor = count;
                a
            }));
            gen.push(Op::Gemm({
                let mut g = GemmOp::new(
                    "generation_proj_gemm",
                    h,
                    n_per_tp * head_size,
                    dtypes.gemm_quant,
                );
                g.scale_factor = count;
                g
            }));
            gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
                "generation_ar_1",
                count,
                h,
                tp,
            )));
        }

        // ---- FFN sub-block ----
        if !block.ffn_no_op {
            let inter = ffn_mult_to_intermediate(block.ffn_ffn_mult, h);
            let inter_per_tp = inter / tp;
            let act_gate_bytes = (2.0 * inter_per_tp as f64 + inter_per_tp as f64) * 2.0;

            ctx.push(Op::Elementwise({
                let mut e = ElementwiseOp::new("context_add_norm_2", norm_bytes);
                e.scale_factor = count;
                e
            }));
            ctx.push(Op::Gemm({
                let mut g = GemmOp::new("context_gate_ffn1_gemm", 2 * inter_per_tp, h, dtypes.gemm_quant);
                g.scale_factor = count;
                g
            }));
            ctx.push(Op::Elementwise({
                let mut e = ElementwiseOp::new("context_act_gate", act_gate_bytes);
                e.scale_factor = count;
                e
            }));
            ctx.push(Op::Gemm({
                let mut g = GemmOp::new("context_ffn2_gemm", h, inter_per_tp, dtypes.gemm_quant);
                g.scale_factor = count;
                g
            }));
            ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
                "context_ar_2",
                count,
                h,
                tp,
            )));

            gen.push(Op::Elementwise({
                let mut e = ElementwiseOp::new("generation_add_norm_2", norm_bytes);
                e.scale_factor = count;
                e
            }));
            gen.push(Op::Gemm({
                let mut g = GemmOp::new("generation_gate_ffn1_gemm", 2 * inter_per_tp, h, dtypes.gemm_quant);
                g.scale_factor = count;
                g
            }));
            gen.push(Op::Elementwise({
                let mut e = ElementwiseOp::new("generation_act_gate", act_gate_bytes);
                e.scale_factor = count;
                e
            }));
            gen.push(Op::Gemm({
                let mut g = GemmOp::new("generation_ffn2_gemm", h, inter_per_tp, dtypes.gemm_quant);
                g.scale_factor = count;
                g
            }));
            gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
                "generation_ar_2",
                count,
                h,
                tp,
            )));
        }
    }

    // P2P (scaled by pp_size - 1) and logits projection.
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
