// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MOE family op-list builder.
//!
//! Mirrors `aiconfigurator.sdk.models.moe.MOEModel.__init__`. Populates
//! `context_ops` and `generation_ops` with typed `Op` variants in the
//! order they execute. Each op's `scale_factor` encodes how many times
//! the op runs per forward pass (e.g. `num_layers` for per-layer ops, `1`
//! for embedding/logits).

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, DispatchFlavor, ElementwiseOp, EmbeddingOp, GemmOp,
    GenerationAttentionOp, MoEDispatchOp, MoeOp, Op,
};

fn dispatch_flavor(backend: BackendKind) -> DispatchFlavor {
    // Mirrors Python `models/moe.py` + `operations/moe.py`'s MoEDispatch
    // branching. For SGLang without `moe_backend="deepep_moe"` (the
    // default), Python falls through to a custom_allreduce / NCCL path
    // — same comm volume as vLLM. The DeepEpNormal table is only used
    // when the caller explicitly sets `moe_backend="deepep_moe"`, which
    // AIC's smoke / cli_estimate path doesn't do.
    match backend {
        BackendKind::Trtllm => DispatchFlavor::TrtllmAlltoall,
        BackendKind::Sglang => DispatchFlavor::CustomAllReduce,
        BackendKind::Vllm => DispatchFlavor::CustomAllReduce,
    }
}

pub fn build_moe_model(config: ModelConfig) -> Model {
    let mut model = Model::new(config);
    let cfg = &model.config;
    let layers = cfg.spec.num_hidden_layers as f64;
    let h = cfg.spec.hidden_size;
    let tp = cfg.parallel.tp_size.max(1);
    let kv_per_gpu = cfg.kv_heads_per_gpu();
    let head_size = cfg.spec.head_dim;
    let num_heads = cfg.spec.num_attention_heads;
    let n_per_tp = num_heads / tp;
    let qkv_n = n_per_tp * head_size + 2 * kv_per_gpu * head_size;
    let vocab_per_tp = cfg.spec.vocab_size / tp;
    let dtypes = cfg.dtypes;
    let num_experts = cfg.spec.num_experts.max(1);
    let topk = cfg.spec.top_k.max(1);
    let moe_inter = cfg.spec.moe_intermediate_size;
    let moe_tp = cfg.parallel.moe_tp_size.max(1);
    let moe_ep = cfg.parallel.moe_ep_size.max(1);
    let attn_dp = cfg.parallel.attention_dp_size.max(1);
    let backend = cfg.backend;
    let flavor = dispatch_flavor(backend);
    let comm_bytes_per_token = h as f64 * dtypes.comm_quant.mapping().memory;

    // ---- Context phase ----
    let mut ctx = Vec::with_capacity(16);

    // Embedding (runs once per forward; scale_factor 0.3 matches Python).
    ctx.push(Op::Embedding({
        let mut e = EmbeddingOp::new("context_embedding", vocab_per_tp, h, dtypes.gemm_quant);
        e.scale_factor = 0.3;
        e
    }));

    // Norms × 2 per layer.
    // Python: ElementWise(num_layers, dim_in=2h, dim_out=2h) reads/writes
    // (2h + 2h) * 2 bytes/token (bfloat16). The 0.8 in Python's constructor
    // (`empirical_bw_scaling_factor`) is stored but unused at query time;
    // bandwidth scaling is applied inside `mem_op_latency_ms`.
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_add_norm_1", norm_bytes);
        e.scale_factor = layers;
        e
    }));

    // QKV GEMM per layer.
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_qkv_gemm", qkv_n, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));

    // Context attention per layer.
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

    // Out projection per layer.
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_proj_gemm", h, n_per_tp * head_size, dtypes.gemm_quant);
        g.scale_factor = layers;
        g.low_precision_input = true;
        g
    }));

    // Norm 2 per layer.
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_add_norm_2", norm_bytes);
        e.scale_factor = layers;
        e
    }));

    // No per-layer all-reduce here: Python's MOE composes per-layer
    // collective inside `MoEDispatch` (the dispatch op handles
    // attention-rank → expert-rank routing). Only the embedding AR
    // runs once per forward pass.

    // Router GEMM per layer (always bf16).
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_router_gemm", num_experts, h, GemmQuantMode::Bfloat16);
        g.scale_factor = layers;
        g
    }));

    // MoE dispatch pre per layer.
    ctx.push(Op::MoeDispatch({
        let mut d = MoEDispatchOp::new(
            "context_moe_pre_dispatch",
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
        d.scale_factor = layers;
        d
    }));

    // MoE compute per layer.
    ctx.push(Op::Moe({
        let mut m = MoeOp::new(
            "context_moe",
            h,
            moe_inter,
            topk,
            num_experts,
            moe_tp,
            moe_ep,
            dtypes.moe_quant,
            "power_law_1.2",
        );
        m.scale_factor = layers;
        m
    }));

    // MoE dispatch post per layer.
    ctx.push(Op::MoeDispatch({
        let mut d = MoEDispatchOp::new(
            "context_moe_post_dispatch",
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
        d.scale_factor = layers;
        d
    }));

    // Embedding all-reduce (once per forward).
    ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "context_embedding_ar",
        1.0,
        h,
        tp,
    )));

    let _ = comm_bytes_per_token;
    model.context_ops = ctx;

    // ---- Generation phase ----
    let mut gen = Vec::with_capacity(16);

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
        let mut g = GemmOp::new("generation_proj_gemm", h, n_per_tp * head_size, dtypes.gemm_quant);
        g.scale_factor = layers;
        g.low_precision_input = true;
        g
    }));

    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_add_norm_2", norm_bytes);
        e.scale_factor = layers;
        e
    }));

    // No per-layer AR — handled inside MoEDispatch (see context phase above).

    gen.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_router_gemm",
            num_experts,
            h,
            GemmQuantMode::Bfloat16,
        );
        g.scale_factor = layers;
        g
    }));

    gen.push(Op::MoeDispatch({
        let mut d = MoEDispatchOp::new(
            "generation_moe_pre_dispatch",
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
        d.scale_factor = layers;
        d
    }));

    gen.push(Op::Moe({
        let mut m = MoeOp::new(
            "generation_moe",
            h,
            moe_inter,
            topk,
            num_experts,
            moe_tp,
            moe_ep,
            dtypes.moe_quant,
            "power_law_1.2",
        );
        m.scale_factor = layers;
        m
    }));

    gen.push(Op::MoeDispatch({
        let mut d = MoEDispatchOp::new(
            "generation_moe_post_dispatch",
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
        d.scale_factor = layers;
        d
    }));

    // Logits GEMM (once per forward, bf16, vocab-tp).
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
