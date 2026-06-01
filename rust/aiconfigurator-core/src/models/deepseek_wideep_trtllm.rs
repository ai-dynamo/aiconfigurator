// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TensorRT-LLM WideEP DeepSeek builder (DeepSeek-V3 / R1).
//!
//! Apple-to-apple port of `aiconfigurator.sdk.models.deepseek.TrtllmWideEPDeepSeekModel`.
//! Selected by `WideEpMode::Trtllm` — only fires when the `DeepSeek`
//! family is requested on the TRT-LLM backend with `enable_wideep=true`.
//!
//! Differs from base `build_deepseek_model`:
//! - Attention is the standard `ContextMlaOp` / `GenerationMlaOp` +
//!   `MlaBmmOp` chain (TRT-LLM uses the granular path, not the MLA
//!   module fallback used by the vLLM-flavored base builder).
//! - MoE compute is `WideEpMoeOp` (querying `wideep_moe_perf.txt`)
//!   instead of `MoeOp` (which queries `moe_perf.txt`).
//! - MoE dispatch is the standard `MoEDispatchOp` with
//!   `DispatchFlavor::TrtllmAlltoall` — the alltoall table is the same
//!   one the existing trtllm path uses.
//! - Generation overlaps shared FFN with the routed-MoE chain
//!   (router + dispatch + WideEp MoE + dispatch) via `Op::Overlap`,
//!   matching Python.
//! - Generation also overlaps `MLABmm(pre)` with a small `rope_kvcache`
//!   elementwise on a parallel stream — modeled here as a single
//!   `Op::Overlap`.
//!
//! `pdl_factor = 0.9` (Python's hand-tuned factor for the CUDA-graph
//! generation path) applied to per-layer counts. MTP factor matches
//! the base DeepSeek builder (smoke runs with `nextn = 0`).

use crate::common::enums::{BackendKind, GemmQuantMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    ContextMlaOp, DispatchFlavor, ElementwiseOp, EmbeddingOp, GemmOp, GenerationMlaOp,
    MlaBmmOp, MoEDispatchOp, Op, OverlapOp, P2POp, WideEpMoeOp,
};

/// `pdl_factor` from Python (`TrtllmWideEPDeepSeekModel`): hand-tuned
/// multiplier on the generation-side per-layer counts to model the
/// post-launch dependency-aware-launch (PDL) overhead inside TRT-LLM's
/// CUDA graph.
const PDL_FACTOR: f64 = 0.9;
const POWER_LAW_ALPHA: &str = "1.01";

fn workload_distribution(enable_eplb: bool) -> String {
    if enable_eplb {
        format!("power_law_{POWER_LAW_ALPHA}_eplb")
    } else {
        format!("power_law_{POWER_LAW_ALPHA}")
    }
}

pub fn build_trtllm_wideep_deepseek_model(config: ModelConfig) -> Result<Model, AicError> {
    let workload = workload_distribution(config.enable_eplb);
    let mut model = Model::new(config);
    let cfg = &model.config;
    let layers = cfg.spec.num_hidden_layers as f64;
    // Python `_mtp_scale_factor` (models/base.py:105-110); see deepseek.rs.
    // The TRT-LLM WideEP variant pre-multiplies by `PDL_FACTOR` for the
    // overlap-aware path; MTP scale stacks on top of that for generation ops.
    let mtp_scale = cfg.mtp_scale_factor();
    let gen_layers = layers * mtp_scale;
    let gen_layer_scale = layers * PDL_FACTOR * mtp_scale;
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

    // Validation that mirrors Python's `TrtllmWideEPDeepSeekModel.__init__`:
    //   1. attention_dp_size > 1 required.
    //   2. moe_ep_size > 1 required.
    //   3. tp_size * attention_dp_size == moe_tp_size * moe_ep_size
    //      (the parallel-layout invariant — without this the per-rank
    //      MLA / MoE shapes can disagree across attention and MoE
    //      stages and produce nonsense latencies).
    if attn_dp <= 1 {
        return Err(AicError::UnsupportedModel(format!(
            "TRT-LLM WideEP requires attention_dp_size > 1, got {attn_dp}"
        )));
    }
    if moe_ep <= 1 {
        return Err(AicError::UnsupportedModel(format!(
            "TRT-LLM WideEP requires moe_ep_size > 1, got {moe_ep}"
        )));
    }
    if tp * attn_dp != moe_tp * moe_ep {
        return Err(AicError::UnsupportedModel(format!(
            "TRT-LLM WideEP requires tp_size * attention_dp_size == \
             moe_tp_size * moe_ep_size; got tp={tp}, attn_dp={attn_dp}, \
             moe_tp={moe_tp}, moe_ep={moe_ep}"
        )));
    }

    // DeepSeek MLA dims (fixed by the architecture).
    const QKV_A_DIM: u32 = 1536 + 512 + 64; // 2112
    const Q_LORA_RANK: u32 = 1536;
    const Q_B_OUT: u32 = 24576;
    const KV_B_OUT: u32 = 32768;
    const MLA_LOCAL_HEADS: u32 = 128;
    const RFC_FUSED_DIM: u32 = 576; // kv_lora_rank(512) + qk_rope_head_dim(64)

    let local_heads = MLA_LOCAL_HEADS / tp;
    let n_heads = cfg.spec.num_attention_heads;
    let n_per_tp = n_heads / tp;
    let head_size = h / 128; // DeepSeek hidden=7168, num_heads=128 -> head_size=128 by default
    let proj_in_per_tp = n_heads * head_size / tp;
    let mla_bmm_quant = if dtypes.gemm_quant != GemmQuantMode::Bfloat16 {
        GemmQuantMode::Fp8
    } else {
        GemmQuantMode::Bfloat16
    };
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    let q_a_layernorm_bytes = (Q_LORA_RANK as f64 + Q_LORA_RANK as f64) * 2.0;
    let rope_kvcache_bytes = (RFC_FUSED_DIM as f64 + RFC_FUSED_DIM as f64) * 2.0;
    let shared_act_bytes = (2.0 * moe_inter as f64 + moe_inter as f64) * 2.0;
    let moe_reduce_bytes = (2.0 * h as f64 + h as f64) * 2.0;

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
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_downscale_gemm", QKV_A_DIM, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_q_a_layernorm", q_a_layernorm_bytes);
        e.scale_factor = layers;
        e
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_q_b_proj_gemm", Q_B_OUT / tp, Q_LORA_RANK, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_kv_b_proj_gemm", KV_B_OUT / tp, 512, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::ContextMla({
        let mut a = ContextMlaOp::new(
            "context_attention",
            local_heads,
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
        );
        a.scale_factor = layers;
        a
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new(
            "context_proj_gemm",
            h,
            128 * head_size / tp,
            dtypes.gemm_quant,
        );
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_add_norm_2", norm_bytes);
        e.scale_factor = layers;
        e
    }));
    // Shared expert (sequential in context — no CUDA-graph overlap).
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new(
            "context_shared_gate_up_gemm",
            2 * moe_inter,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = layers;
        g
    }));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_shared_act_gate", shared_act_bytes);
        e.scale_factor = layers;
        e
    }));
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_shared_ffn2_gemm", h, moe_inter, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    // Router + dispatch + WideEpMoe + dispatch.
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
            BackendKind::Trtllm,
            DispatchFlavor::TrtllmAlltoall,
        );
        d.scale_factor = layers;
        d.moe_quant = dtypes.moe_quant;
        d
    }));
    ctx.push(Op::WideEpMoe({
        let mut m = WideEpMoeOp::new(
            "context_moe",
            h,
            moe_inter,
            topk,
            num_experts,
            moe_tp,
            moe_ep,
            attn_dp,
            dtypes.moe_quant,
            workload.clone(),
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
            BackendKind::Trtllm,
            DispatchFlavor::TrtllmAlltoall,
        );
        d.scale_factor = layers;
        d.moe_quant = dtypes.moe_quant;
        d
    }));
    // moe_reduce_add: sum routed top_k + shared output.
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_moe_reduce_add", moe_reduce_bytes);
        e.scale_factor = layers;
        e
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
        e.scale_factor = gen_layer_scale;
        e
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_downscale_gemm",
            QKV_A_DIM,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layer_scale;
        g
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_q_a_layernorm", q_a_layernorm_bytes);
        e.scale_factor = gen_layer_scale;
        e
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_q_b_proj_gemm",
            Q_B_OUT / tp,
            Q_LORA_RANK,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layer_scale;
        g
    }));
    // BMM_pre || RoPE+KV cache prep (two-stream overlap).
    let mut bmm_pre = MlaBmmOp::new("generation_bmm_pre", n_per_tp, mla_bmm_quant, true);
    bmm_pre.scale_factor = gen_layer_scale;
    let mut rope_kvcache = ElementwiseOp::new("generation_rope_kvcache", rope_kvcache_bytes);
    rope_kvcache.scale_factor = gen_layer_scale;
    gen.push(Op::Overlap(OverlapOp::new(
        "generation_bmm_rope_overlap",
        vec![Op::MlaBmm(bmm_pre)],
        vec![Op::Elementwise(rope_kvcache)],
    )));
    gen.push(Op::GenerationMla({
        let mut a = GenerationMlaOp::new("generation_attention", local_heads, dtypes.kv_cache_quant);
        a.scale_factor = gen_layer_scale;
        a
    }));
    gen.push(Op::MlaBmm({
        let mut g = MlaBmmOp::new("generation_bmm_post", n_per_tp, mla_bmm_quant, false);
        g.scale_factor = gen_layer_scale;
        g
    }));
    gen.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_proj_gemm",
            h,
            h / tp,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layer_scale;
        g
    }));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_add_norm_2", norm_bytes);
        e.scale_factor = gen_layer_scale;
        e
    }));

    // Generation MoE: Shared Expert || Routed Expert via OverlapOp.
    let mut shared_ops: Vec<Op> = Vec::with_capacity(3);
    let mut shared_gemm_up = GemmOp::new(
        "generation_shared_gate_up_gemm",
        2 * moe_inter,
        h,
        dtypes.gemm_quant,
    );
    shared_gemm_up.scale_factor = gen_layer_scale;
    shared_ops.push(Op::Gemm(shared_gemm_up));
    let mut shared_act = ElementwiseOp::new("generation_shared_act_gate", shared_act_bytes);
    shared_act.scale_factor = gen_layer_scale;
    shared_ops.push(Op::Elementwise(shared_act));
    let mut shared_down = GemmOp::new(
        "generation_shared_ffn2_gemm",
        h,
        moe_inter,
        dtypes.gemm_quant,
    );
    shared_down.scale_factor = gen_layer_scale;
    shared_ops.push(Op::Gemm(shared_down));

    let mut routed_ops: Vec<Op> = Vec::with_capacity(4);
    let mut router = GemmOp::new(
        "generation_router_gemm",
        num_experts,
        h,
        GemmQuantMode::Bfloat16,
    );
    router.scale_factor = gen_layer_scale;
    routed_ops.push(Op::Gemm(router));
    let mut pre_disp = MoEDispatchOp::new(
        "generation_moe_pre_dispatch",
        h,
        topk,
        num_experts,
        moe_tp,
        moe_ep,
        attn_dp,
        true,
        BackendKind::Trtllm,
        DispatchFlavor::TrtllmAlltoall,
    );
    pre_disp.scale_factor = gen_layer_scale;
    pre_disp.moe_quant = dtypes.moe_quant;
    routed_ops.push(Op::MoeDispatch(pre_disp));
    let mut moe = WideEpMoeOp::new(
        "generation_moe",
        h,
        moe_inter,
        topk,
        num_experts,
        moe_tp,
        moe_ep,
        attn_dp,
        dtypes.moe_quant,
        workload.clone(),
    );
    moe.scale_factor = gen_layer_scale;
    routed_ops.push(Op::WideEpMoe(moe));
    let mut post_disp = MoEDispatchOp::new(
        "generation_moe_post_dispatch",
        h,
        topk,
        num_experts,
        moe_tp,
        moe_ep,
        attn_dp,
        false,
        BackendKind::Trtllm,
        DispatchFlavor::TrtllmAlltoall,
    );
    post_disp.scale_factor = gen_layer_scale;
    post_disp.moe_quant = dtypes.moe_quant;
    routed_ops.push(Op::MoeDispatch(post_disp));

    gen.push(Op::Overlap(OverlapOp::new(
        "generation_moe_overlap",
        routed_ops,
        shared_ops,
    )));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_moe_reduce_add", moe_reduce_bytes);
        e.scale_factor = gen_layer_scale;
        e
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

    // P2P: generation side picks up MTP scale; context-side does not.
    let pp_scale = (pp as f64 - 1.0).max(0.0);
    let mut p2p_ctx = P2POp::new("context_p2p", pp, h);
    p2p_ctx.scale_factor = pp_scale;
    ctx.push(Op::P2P(p2p_ctx));
    let mut p2p_gen = P2POp::new("generation_p2p", pp, h);
    p2p_gen.scale_factor = pp_scale * mtp_scale;
    gen.push(Op::P2P(p2p_gen));

    // proj_in_per_tp computed for parity with Python's
    // `_num_heads * v_head_dim // tp_size`. Currently unused (proj GEMM
    // uses `h / tp` directly above); retained so the dim helper is
    // future-proof if the v_head_dim ever diverges from head_size.
    let _ = proj_in_per_tp;
    // moe_quant kept consistent across dispatch ops; surface here for
    // the silenced linter pass.
    let _ = MoeQuantMode::Bfloat16;

    model.context_ops = ctx;
    model.generation_ops = gen;
    Ok(model)
}
