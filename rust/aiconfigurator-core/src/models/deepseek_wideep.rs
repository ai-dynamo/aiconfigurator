// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang WideEP DeepSeek builder (DeepSeek-V3 / R1 with DeepEP).
//!
//! Apple-to-apple port of `aiconfigurator.sdk.models.deepseek.WideEPDeepSeekModel`.
//! Selected by `WideEpMode::SglangDeepEp` — only fires when the
//! `DeepSeek` family is requested on the SGLang backend with
//! `moe_backend="deepep_moe"`.
//!
//! Differs from the base `build_deepseek_model`:
//! - Attention uses `WideEpContextMlaOp` / `WideEpGenerationMlaOp`
//!   instead of `ContextMlaOp` / `GenerationMlaOp`.
//! - The MLA module is **always** the granular path (no `FallbackOp`):
//!   `WideEPDeepSeekModel` doesn't model a MLA-module / BMM-chain
//!   alternation, just the WideEP-tabled fmha kernels surrounded by
//!   `context_qkv_a_proj_gemm` + `context_downscale_gemm` and an
//!   optional `NCCL all_gather` / `reduce_scatter` when `tp_size > 1`.
//! - Standard `MoeOp` + `MoEDispatchOp` (with the DeepEP normal/LL
//!   dispatch flavors via the dispatch op's existing branching) — the
//!   `moe_backend="deepep_moe"` selection is implicit in the SGLang
//!   routing and Rust's `DispatchFlavor::DeepEpNormal` /
//!   `DeepEpLowLatency` paths.
//!
//! This builder is invoked from `factory.rs` when `WideEpMode::SglangDeepEp`
//! is set on `ModelConfig`. The MTP scale factor matches the base
//! DeepSeek builder (the smoke set runs with `nextn = 0`).

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    DispatchFlavor, ElementwiseOp, EmbeddingOp, GemmOp, MoEDispatchOp, MoeOp, NcclOp, Op, P2POp,
    WideEpContextMlaOp, WideEpGenerationMlaOp,
};

/// Workload-distribution string mirroring Python:
///   - SGLang DeepEP context: `power_law_0.6` when `enable_eplb`,
///     else `power_law_1.01`.
///   - SGLang DeepEP generation: always `power_law_1.01`.
fn workload_string(prefix: &str, alpha: &str) -> String {
    format!("{prefix}_{alpha}")
}

pub fn build_sglang_wideep_deepseek_model(config: ModelConfig) -> Model {
    let mut model = Model::new(config);
    let cfg = &model.config;
    let layers = cfg.spec.num_hidden_layers as f64;
    // Python `_mtp_scale_factor` (models/base.py:105-110); see deepseek.rs
    // for the full rationale. `nextn=0` reduces both to identity.
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

    let context_alpha = if cfg.enable_eplb { "0.6" } else { "1.01" };
    let context_workload = workload_string("power_law", context_alpha);
    let generation_workload = workload_string("power_law", "1.01");

    // Hidden DeepSeek MLA dims used by Python's WideEPDeepSeekModel.
    // These are fixed by the architecture and don't depend on the HF
    // config beyond `hidden_size`.
    const QKV_A_DIM: u32 = 1536 + 512 + 64; // q_lora + kv_lora + qk_rope = 2112
    const MLA_LOCAL_HEADS: u32 = 128;

    let local_heads = MLA_LOCAL_HEADS / tp;
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    let shared_act_bytes = (2.0 * moe_inter as f64 + moe_inter as f64) * 2.0;

    // ---- Context phase ----
    let mut ctx = Vec::new();
    // qkv_a projection — replicated on every GPU (not TP-sharded).
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_qkv_a_proj_gemm", QKV_A_DIM, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    // Optional NCCL all_gather when tp > 1 (sglang's communicator).
    if tp > 1 {
        ctx.push(Op::Nccl(NcclOp::new(
            "context_tp_all_gather",
            layers,
            h,
            tp,
            "all_gather",
        )));
    }
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
        let mut g = GemmOp::new("context_downscale_gemm", QKV_A_DIM, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::WideEpContextMla({
        let mut a = WideEpContextMlaOp::new(
            "context_attention",
            local_heads,
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
        );
        a.scale_factor = layers;
        a
    }));
    if tp > 1 {
        ctx.push(Op::Nccl(NcclOp::new(
            "context_tp_reduce_scatter",
            layers,
            h,
            tp,
            "reduce_scatter",
        )));
    }
    // Shared expert.
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new(
            "context_gate_ffn1_gemm",
            2 * moe_inter,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_act_gate", shared_act_bytes);
        e.scale_factor = layers;
        e
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_ffn2_gemm", h, moe_inter, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    // Standard MoE dispatch + compute + dispatch. DeepEP normal flavor;
    // dispatch op chooses table by (backend, is_context, scheduled_tokens).
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
            BackendKind::Sglang,
            DispatchFlavor::DeepEpNormal,
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
            context_workload.clone(),
        );
        m.scale_factor = layers;
        m
    }));
    ctx.push(Op::Gemm(GemmOp::new(
        "context_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));

    // ---- Generation phase ----
    let mut gen = Vec::new();
    gen.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_qkv_a_proj_gemm",
            QKV_A_DIM,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layers;
        g
    }));
    gen.push(Op::Embedding({
        let mut e = EmbeddingOp::new(
            "generation_embedding",
            vocab_per_tp,
            h,
            dtypes.gemm_quant,
        );
        e.scale_factor = 0.3 * mtp_scale;
        e
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_add_norm_1", norm_bytes);
        e.scale_factor = gen_layers;
        e
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_downscale_gemm",
            QKV_A_DIM,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layers;
        g
    }));
    gen.push(Op::WideEpGenerationMla({
        let mut a = WideEpGenerationMlaOp::new(
            "generation_attention",
            local_heads,
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
        );
        a.scale_factor = gen_layers;
        a
    }));
    // Shared expert (gen).
    gen.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_gate_ffn1_gemm",
            2 * moe_inter,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layers;
        g
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_act_gate", shared_act_bytes);
        e.scale_factor = gen_layers;
        e
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new("generation_ffn2_gemm", h, moe_inter, dtypes.gemm_quant);
        g.scale_factor = gen_layers;
        g
    }));
    // DeepEP low-latency dispatch on decode.
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
            BackendKind::Sglang,
            DispatchFlavor::DeepEpLowLatency,
        );
        d.scale_factor = gen_layers;
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
            generation_workload.clone(),
        );
        m.scale_factor = gen_layers;
        m
    }));
    gen.push(Op::Gemm({
        // Logits GEMM fires once per forward; Python uses `1 * mtp_scale`.
        let mut g = GemmOp::new(
            "generation_logits_gemm",
            vocab_per_tp,
            h,
            GemmQuantMode::Bfloat16,
        );
        g.scale_factor = mtp_scale;
        g
    }));

    // P2P: generation-side picks up MTP scale; context-side does not.
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
