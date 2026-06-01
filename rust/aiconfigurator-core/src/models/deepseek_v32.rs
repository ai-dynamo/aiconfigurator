// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeekV32 family op-list builder (DeepSeek-V3.2 / GLM-5).
//!
//! Mirrors `aiconfigurator.sdk.models.deepseek_v32.DeepSeekV32Model`.
//! Same overall MoE+shared-expert shape as DeepSeek base, but the
//! attention block is the DSA (DeepSeek Sparse Attention) module
//! instead of MLA. Generation MoE overlaps shared expert with the
//! routed-expert path (`Op::Overlap`), and the smoke vLLM smoke path
//! does NOT use the per-layer tp_allreduce that base DeepSeek emits
//! (Python's `deepseek_v32.py` omits it).

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    DispatchFlavor, DsaModuleOp, ElementwiseOp, EmbeddingOp, GemmOp, MoEDispatchOp, MoeOp, Op,
    OverlapOp, P2POp,
};

fn dispatch_flavor(backend: BackendKind) -> DispatchFlavor {
    match backend {
        BackendKind::Trtllm => DispatchFlavor::TrtllmAlltoall,
        BackendKind::Sglang => DispatchFlavor::CustomAllReduce,
        BackendKind::Vllm => DispatchFlavor::CustomAllReduce,
    }
}

pub fn build_deepseek_v32_model(config: ModelConfig) -> Model {
    let mut model = Model::new(config);
    let cfg = &model.config;
    let layers = cfg.spec.num_hidden_layers as f64;
    // Python `_mtp_scale_factor` (models/base.py:105-110): when `nextn>=1`
    // (default for DSv3.2), every generation-side scale_factor is multiplied
    // by `1/(1+E[accept]) * (nextn+layers)/layers`. `gen_layers` carries the
    // pre-multiplied value for per-layer ops; once-per-forward ops use
    // `mtp_scale` directly. When `nextn==0`, both reduce to `layers` and `1.0`.
    let mtp_scale = cfg.mtp_scale_factor();
    let gen_layers = layers * mtp_scale;
    let h = cfg.spec.hidden_size;
    let tp = cfg.parallel.tp_size.max(1);
    let pp = cfg.parallel.pp_size.max(1);
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
    let architecture = cfg.spec.architecture.clone();
    let local_heads = cfg.spec.num_attention_heads / tp;
    let moe_inter_per_tp = (moe_inter + tp - 1) / tp;

    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    let shared_act_gate_bytes = (2.0 * moe_inter_per_tp as f64 + moe_inter_per_tp as f64) * 2.0;

    // ---- Context phase ----
    let mut ctx = Vec::new();
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
    ctx.push(Op::DsaContext({
        let mut a = DsaModuleOp::new(
            "context_attention",
            local_heads,
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
            dtypes.gemm_quant,
            architecture.clone(),
        );
        a.scale_factor = layers;
        a
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_add_norm_2", norm_bytes);
        e.scale_factor = layers;
        e
    }));
    // Shared expert (gated FFN).
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new(
            "context_shared_gate_up_gemm",
            2 * moe_inter_per_tp,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_shared_act_gate", shared_act_gate_bytes);
        e.scale_factor = layers;
        e
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new(
            "context_shared_ffn2_gemm",
            h,
            moe_inter_per_tp,
            dtypes.gemm_quant,
        );
        g.scale_factor = layers;
        g
    }));
    // Router.
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new(
            "context_router_gemm",
            num_experts,
            h,
            GemmQuantMode::Bfloat16,
        );
        g.scale_factor = layers;
        g
    }));
    // MoE pre-dispatch, compute, post-dispatch.
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
            "power_law_1.01",
        );
        m.scale_factor = layers;
        m
    }));
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
    ctx.push(Op::Gemm(GemmOp::new(
        "context_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));

    // ---- Generation phase ----
    let mut gen = Vec::new();
    gen.push(Op::Embedding({
        let mut e = EmbeddingOp::new("generation_embedding", vocab_per_tp, h, dtypes.gemm_quant);
        e.scale_factor = 0.3 * mtp_scale;
        e
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_add_norm_1", norm_bytes);
        e.scale_factor = gen_layers;
        e
    }));
    gen.push(Op::DsaGeneration({
        let mut a = DsaModuleOp::new(
            "generation_attention",
            local_heads,
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
            dtypes.gemm_quant,
            architecture.clone(),
        );
        a.scale_factor = gen_layers;
        a
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_add_norm_2", norm_bytes);
        e.scale_factor = gen_layers;
        e
    }));

    // Overlap: shared FFN ‖ routed MoE.
    let mut gen_shared = Vec::with_capacity(3);
    gen_shared.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_shared_gate_up_gemm",
            2 * moe_inter_per_tp,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layers;
        g
    }));
    gen_shared.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_shared_act_gate", shared_act_gate_bytes);
        e.scale_factor = gen_layers;
        e
    }));
    gen_shared.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_shared_ffn2_gemm",
            h,
            moe_inter_per_tp,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layers;
        g
    }));

    let mut gen_routed = Vec::with_capacity(4);
    gen_routed.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_router_gemm",
            num_experts,
            h,
            GemmQuantMode::Bfloat16,
        );
        g.scale_factor = gen_layers;
        g
    }));
    gen_routed.push(Op::MoeDispatch({
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
        d.scale_factor = gen_layers;
        d
    }));
    gen_routed.push(Op::Moe({
        let mut m = MoeOp::new(
            "generation_moe",
            h,
            moe_inter,
            topk,
            num_experts,
            moe_tp,
            moe_ep,
            dtypes.moe_quant,
            "power_law_1.01",
        );
        m.scale_factor = gen_layers;
        m
    }));
    gen_routed.push(Op::MoeDispatch({
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
        d.scale_factor = gen_layers;
        d
    }));
    gen.push(Op::Overlap(OverlapOp::new(
        "generation_moe_overlap",
        gen_routed,
        gen_shared,
    )));
    gen.push(Op::Gemm({
        // Logits GEMM fires once per forward; Python `deepseek.py` uses
        // `1 * self._mtp_scale_factor` as the count.
        let mut g = GemmOp::new(
            "generation_logits_gemm",
            vocab_per_tp,
            h,
            GemmQuantMode::Bfloat16,
        );
        g.scale_factor = mtp_scale;
        g
    }));

    // P2P: context-side uses the un-scaled `pp_scale`; generation-side is
    // multiplied by `mtp_scale` (Python applies MTP to generation P2P too).
    let pp_scale = (pp as f64 - 1.0).max(0.0);
    let mut p2p_ctx = P2POp::new("context_p2p", pp, h);
    p2p_ctx.scale_factor = pp_scale;
    ctx.push(Op::P2P(p2p_ctx));
    let mut p2p_gen = P2POp::new("generation_p2p", pp, h);
    p2p_gen.scale_factor = pp_scale * mtp_scale;
    gen.push(Op::P2P(p2p_gen));

    model.context_ops = ctx;
    model.generation_ops = gen;
    model
}
