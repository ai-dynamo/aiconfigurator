// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 family op-list builder.
//!
//! Apple-to-apple port of `aiconfigurator.sdk.models.deepseek_v4.DeepSeekV4Model`.
//! Differs from base DeepSeek in three structural ways:
//!
//! - The attention block is the DSv4 compressed-attention module
//!   (`Op::Dsv4Context` / `Op::Dsv4Generation`) instead of MLA.
//!   `compress_ratios` is a per-layer list of {0, 4, 128} where 4 selects
//!   the CSA (compressed-sparse) variant and 128 selects HCA
//!   (hybrid-causal). Pure SWA layers (ratio=0) are approximated using
//!   the HCA tables — matching Python's
//!   `ratio_counts[128] += ratio_counts.pop(0, 0)` mapping. One attention
//!   op per unique ratio with `scale_factor = layer count`.
//! - Each decoder block has a manifold-constrained hyper-connection (mHC)
//!   pair (`mhc_pre` + `mhc_post`) replacing the standard add_norm
//!   elementwise. Currently routed through `Op::Mhc(MhcModuleOp)` which
//!   does not differentiate the `pre`/`post` slice — the Rust MHC table
//!   doesn't carry that key, and no smoke-set perf data exists for DSv4
//!   on b200/h200/h100 systems anyway, so the op-graph mirrors Python's
//!   structure even though numeric values for MHC are approximate when
//!   data does land.
//! - Generation overlaps the routed-MoE chain with the shared expert FFN
//!   (`Op::Overlap`), same idiom as DSv32.
//!
//! `n_shared_experts` is fixed at 1 by Python's default; the shared FFN
//! emits one set of gate_up + act_gate + ffn2 ops.

use std::collections::BTreeMap;

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::common::error::AicError;
use crate::models::base::{Model, ModelConfig};
use crate::models::config_loader::DeepSeekV4Config;
use crate::operators::{
    DispatchFlavor, Dsv4ModuleOp, ElementwiseOp, EmbeddingOp, GemmOp, MhcModuleOp, MoEDispatchOp,
    MoeOp, Op, OverlapOp, P2POp,
};
use crate::perf_database::dsv4::AttnKind;

fn dispatch_flavor(backend: BackendKind) -> DispatchFlavor {
    match backend {
        BackendKind::Trtllm => DispatchFlavor::TrtllmAlltoall,
        BackendKind::Sglang => DispatchFlavor::CustomAllReduce,
        BackendKind::Vllm => DispatchFlavor::CustomAllReduce,
    }
}

/// Map `compress_ratio` to the `AttnKind` whose perf table to query.
/// Mirrors Python's `ratio_counts[128] += ratio_counts.pop(0, 0)` rule:
/// ratio 0 (pure SWA) is approximated with HCA.
fn attn_kind_for_ratio(ratio: u32) -> AttnKind {
    if ratio == 4 {
        AttnKind::Csa
    } else {
        AttnKind::Hca
    }
}

pub fn build_deepseek_v4_model(config: ModelConfig) -> Result<Model, AicError> {
    let dsv4: DeepSeekV4Config = config
        .spec
        .deepseek_v4
        .as_ref()
        .ok_or_else(|| {
            AicError::UnsupportedModel(
                "DEEPSEEKV4 builder requires deepseek_v4 extras in ModelSpec".to_string(),
            )
        })?
        .clone();
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
    let backend = cfg.backend;
    let flavor = dispatch_flavor(backend);
    let architecture = cfg.spec.architecture.clone();
    let local_heads = cfg.spec.num_attention_heads / tp;
    let moe_inter_per_tp = (moe_inter + tp - 1) / tp;

    // Python: attn_norm / ffn_norm use ElementWise(num_layers, h, h, 0.8) =>
    // (h + h) * 2 bytes/token for bf16.
    let attn_norm_bytes = (h as f64 + h as f64) * 2.0;
    let shared_act_gate_bytes = (2.0 * moe_inter_per_tp as f64 + moe_inter_per_tp as f64) * 2.0;

    // Count compress_ratios per Python's `Counter`. ratio 0 -> 128 (HCA approx).
    let mut ratio_counts: BTreeMap<u32, u32> = BTreeMap::new();
    for &r in &dsv4.compress_ratios {
        *ratio_counts.entry(r).or_insert(0) += 1;
    }
    let swa_count = ratio_counts.remove(&0).unwrap_or(0);
    if swa_count > 0 {
        *ratio_counts.entry(128).or_insert(0) += swa_count;
    }

    let push_attention = |ops_out: &mut Vec<Op>, is_context: bool| {
        let prefix = if is_context { "context" } else { "generation" };
        let name = format!("{prefix}_attention");
        for (&ratio, &count) in &ratio_counts {
            if count == 0 {
                continue;
            }
            let kind = attn_kind_for_ratio(ratio);
            let mut op = Dsv4ModuleOp::new(
                name.clone(),
                kind,
                local_heads,
                dtypes.kv_cache_quant,
                dtypes.fmha_quant,
                dtypes.gemm_quant,
                architecture.clone(),
            );
            op.scale_factor = count as f64;
            ops_out.push(if is_context {
                Op::Dsv4Context(op)
            } else {
                Op::Dsv4Generation(op)
            });
        }
    };

    // Helper to emit one mHC op (pre or post). The Rust MHC perf table
    // doesn't differentiate by phase, so both calls hit the same lookup;
    // Python's phase split surfaces when phase-keyed data is actually
    // present (no smoke-set perf data covers this yet).
    let push_mhc = |ops_out: &mut Vec<Op>, name: &str, scale: f64| {
        let mut op = MhcModuleOp::new(name, dsv4.hc_mult, h, architecture.clone());
        op.scale_factor = scale;
        ops_out.push(Op::Mhc(op));
    };

    // ---- Context phase ----
    let mut ctx = Vec::new();
    ctx.push(Op::Embedding({
        let mut e = EmbeddingOp::new("context_embedding", vocab_per_tp, h, dtypes.gemm_quant);
        e.scale_factor = 0.3;
        e
    }));
    push_mhc(&mut ctx, "context_mhc_pre", layers);
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_attn_norm", attn_norm_bytes);
        e.scale_factor = layers;
        e
    }));
    push_attention(&mut ctx, true);
    push_mhc(&mut ctx, "context_mhc_post", layers);
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_ffn_norm", attn_norm_bytes);
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
    // MoE dispatch + compute + dispatch.
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
    push_mhc(&mut gen, "generation_mhc_pre", gen_layers);
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_attn_norm", attn_norm_bytes);
        e.scale_factor = gen_layers;
        e
    }));
    push_attention(&mut gen, false);
    push_mhc(&mut gen, "generation_mhc_post", gen_layers);
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_ffn_norm", attn_norm_bytes);
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

    // P2P: generation side picks up MTP scale (Python `_mtp_scale_factor`
    // multiplies all generation-phase op scales). `nextn=0` reduces this
    // to the un-scaled path.
    let pp_scale = (pp as f64 - 1.0).max(0.0);
    let mut p2p_ctx = P2POp::new("context_p2p", pp, h);
    p2p_ctx.scale_factor = pp_scale;
    ctx.push(Op::P2P(p2p_ctx));
    let mut p2p_gen = P2POp::new("generation_p2p", pp, h);
    p2p_gen.scale_factor = pp_scale * mtp_scale;
    gen.push(Op::P2P(p2p_gen));

    model.context_ops = ctx;
    model.generation_ops = gen;
    Ok(model)
}
