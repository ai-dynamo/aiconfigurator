// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen3.5 hybrid (GDN + full-attention) op-list builder.
//!
//! Mirrors `aiconfigurator.sdk.models.qwen35.Qwen35Model`. Each layer
//! is either a Gated DeltaNet ("linear_attention") or a standard GQA
//! transformer ("full_attention"); the per-layer mix is taken from
//! `Qwen35Config.layer_types` and applied as `count = num_inst` on the
//! op `scale_factor`. FFN is dense SwiGLU when `num_experts == 0` and
//! MoE (optionally with a shared expert) otherwise.

use crate::common::enums::GemmQuantMode;
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, DispatchFlavor, ElementwiseOp, EmbeddingOp, GdnOp,
    GemmOp, GenerationAttentionOp, MoEDispatchOp, MoeOp, Op, P2POp,
};

fn dispatch_flavor(backend: crate::common::enums::BackendKind) -> DispatchFlavor {
    match backend {
        crate::common::enums::BackendKind::Trtllm => DispatchFlavor::TrtllmAlltoall,
        crate::common::enums::BackendKind::Sglang => DispatchFlavor::CustomAllReduce,
        crate::common::enums::BackendKind::Vllm => DispatchFlavor::CustomAllReduce,
    }
}

pub fn build_qwen35_model(config: ModelConfig) -> Model {
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
    let dtypes = cfg.dtypes;
    let inter_per_tp = cfg.spec.intermediate_size / tp;
    let topk = cfg.spec.top_k;
    let num_experts = cfg.spec.num_experts;
    let moe_inter = cfg.spec.moe_intermediate_size;
    let shared_inter = cfg.spec.shared_expert_intermediate_size;
    // Qwen35-specific config.
    let qwen35 = cfg
        .spec
        .qwen35
        .as_ref()
        .expect("build_qwen35_model: HfModelConfig missing `qwen35` extras");
    let nk = qwen35.linear_num_key_heads;
    let hk = qwen35.linear_key_head_dim;
    let nv = qwen35.linear_num_value_heads;
    let hv = qwen35.linear_value_head_dim;
    let d_conv = qwen35.linear_conv_kernel_dim;

    let num_linear = qwen35
        .layer_types
        .iter()
        .filter(|t| t == &"linear_attention")
        .count() as f64;
    let num_full = qwen35
        .layer_types
        .iter()
        .filter(|t| t == &"full_attention")
        .count() as f64;
    // Python `_mtp_scale_factor` (models/base.py:105-110): Qwen3.5 family
    // sets `nextn=1` by default (`task.py:448`), so every generation-side
    // op count gets multiplied by `sf = 1/(1+E[accept]) * (nextn+layers)/layers`.
    // Python applies this as `c = counts[*] * sf` in `qwen35.py:305-355+`.
    // `gen_num_linear` and `gen_num_full` carry the pre-scaled values; the
    // once-per-forward ops (embedding, logits) use `mtp_scale` directly.
    // When `nextn==0` all reduce to the un-scaled path.
    let mtp_scale = cfg.mtp_scale_factor();
    let gen_num_linear = num_linear * mtp_scale;
    let gen_num_full = num_full * mtp_scale;

    let n_q_per_tp = num_heads / tp;
    let n_kv_per_tp = (num_kv_heads + tp - 1) / tp;
    // GDN combined projection: Q + K + V + gate(Z) + beta along the
    // sharded inner axis. Layout matches Python's
    // `nk*hk + nk*hk + nv*hv + nv*hv + nk*hk`.
    let gdn_in_proj_out = (nk * hk + nk * hk + nv * hv + nv * hv + nk * hk) / tp;
    let gdn_out_proj_in = nv * hv / tp;

    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    let act_gate_bytes = (2.0 * inter_per_tp as f64 + inter_per_tp as f64) * 2.0;

    let mut ctx = Vec::new();
    let mut gen = Vec::new();

    ctx.push(Op::Embedding({
        let mut e = EmbeddingOp::new("context_embedding", vocab_per_tp, h, dtypes.gemm_quant);
        e.scale_factor = 0.3;
        e
    }));
    ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "context_embedding_ar",
        1.0,
        h,
        tp,
    )));
    gen.push(Op::Embedding({
        let mut e = EmbeddingOp::new("generation_embedding", vocab_per_tp, h, dtypes.gemm_quant);
        e.scale_factor = 0.3 * mtp_scale;
        e
    }));
    gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "generation_embedding_ar",
        mtp_scale,
        h,
        tp,
    )));

    // ---- Linear-attention (GDN) layers ----
    if num_linear > 0.0 {
        push_gdn_phase(
            &mut ctx,
            "context",
            num_linear,
            h,
            tp,
            dtypes,
            gdn_in_proj_out,
            gdn_out_proj_in,
            nk,
            hk,
            nv,
            hv,
            d_conv,
        );
        push_ffn_phase(
            &mut ctx,
            "context_gdn",
            num_linear,
            h,
            tp,
            moe_tp,
            moe_ep,
            attn_dp,
            backend,
            flavor,
            dtypes,
            inter_per_tp,
            act_gate_bytes,
            topk,
            num_experts,
            moe_inter,
            shared_inter,
        );

        push_gdn_phase(
            &mut gen,
            "generation",
            gen_num_linear,
            h,
            tp,
            dtypes,
            gdn_in_proj_out,
            gdn_out_proj_in,
            nk,
            hk,
            nv,
            hv,
            d_conv,
        );
        push_ffn_phase(
            &mut gen,
            "generation_gdn",
            gen_num_linear,
            h,
            tp,
            moe_tp,
            moe_ep,
            attn_dp,
            backend,
            flavor,
            dtypes,
            inter_per_tp,
            act_gate_bytes,
            topk,
            num_experts,
            moe_inter,
            shared_inter,
        );
    }

    // ---- Full-attention (GQA) layers ----
    if num_full > 0.0 {
        let qkv_out = n_q_per_tp * head_size + n_kv_per_tp * head_size * 2;
        ctx.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("context_full_attn_norm", norm_bytes);
            e.scale_factor = num_full;
            e
        }));
        ctx.push(Op::Gemm({
            let mut g = GemmOp::new("context_qkv_gemm", qkv_out, h, dtypes.gemm_quant);
            g.scale_factor = num_full;
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
            a.scale_factor = num_full;
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
            g.scale_factor = num_full;
            g.low_precision_input = true;
            g
        }));
        ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "context_full_ar",
            num_full,
            h,
            tp,
        )));
        push_ffn_phase(
            &mut ctx,
            "context_full",
            num_full,
            h,
            tp,
            moe_tp,
            moe_ep,
            attn_dp,
            backend,
            flavor,
            dtypes,
            inter_per_tp,
            act_gate_bytes,
            topk,
            num_experts,
            moe_inter,
            shared_inter,
        );

        gen.push(Op::Elementwise({
            let mut e = ElementwiseOp::new("generation_full_attn_norm", norm_bytes);
            e.scale_factor = gen_num_full;
            e
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new("generation_qkv_gemm", qkv_out, h, dtypes.gemm_quant);
            g.scale_factor = gen_num_full;
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
            a.scale_factor = gen_num_full;
            a
        }));
        gen.push(Op::Gemm({
            let mut g = GemmOp::new(
                "generation_proj_gemm",
                h,
                n_q_per_tp * head_size,
                dtypes.gemm_quant,
            );
            g.scale_factor = gen_num_full;
            g.low_precision_input = true;
            g
        }));
        gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "generation_full_ar",
            gen_num_full,
            h,
            tp,
        )));
        push_ffn_phase(
            &mut gen,
            "generation_full",
            gen_num_full,
            h,
            tp,
            moe_tp,
            moe_ep,
            attn_dp,
            backend,
            flavor,
            dtypes,
            inter_per_tp,
            act_gate_bytes,
            topk,
            num_experts,
            moe_inter,
            shared_inter,
        );
    }

    // Final logits + P2P.
    let pp_scale = (pp as f64 - 1.0).max(0.0);
    ctx.push(Op::Gemm(GemmOp::new(
        "context_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));
    let mut p2p_ctx = P2POp::new("context_p2p", pp, h);
    p2p_ctx.scale_factor = pp_scale;
    ctx.push(Op::P2P(p2p_ctx));
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
    let mut p2p_gen = P2POp::new("generation_p2p", pp, h);
    p2p_gen.scale_factor = pp_scale * mtp_scale;
    gen.push(Op::P2P(p2p_gen));

    model.context_ops = ctx;
    model.generation_ops = gen;
    model
}

#[allow(clippy::too_many_arguments)]
fn push_gdn_phase(
    ops_out: &mut Vec<Op>,
    phase: &str,
    count: f64,
    h: u32,
    tp: u32,
    dtypes: crate::models::base::DtypeConfig,
    gdn_in_proj_out: u32,
    gdn_out_proj_in: u32,
    nk: u32,
    hk: u32,
    nv: u32,
    hv: u32,
    d_conv: u32,
) {
    let (norm_name, in_proj, out_proj, ar) = match phase {
        "context" => (
            "context_gdn_norm",
            "context_gdn_in_proj_gemm",
            "context_gdn_out_proj_gemm",
            "context_gdn_ar",
        ),
        _ => (
            "generation_gdn_norm",
            "generation_gdn_in_proj_gemm",
            "generation_gdn_out_proj_gemm",
            "generation_gdn_ar",
        ),
    };
    let (conv_kernel, scan_kernel, conv_name, scan_name) = match phase {
        "context" => (
            "causal_conv1d_fn",
            "chunk_gated_delta_rule",
            "context_gdn_conv1d",
            "context_gdn_scan",
        ),
        _ => (
            "causal_conv1d_update",
            "fused_sigmoid_gating_delta_rule_update",
            "generation_gdn_conv1d",
            "generation_gdn_recurrence",
        ),
    };
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;

    ops_out.push(Op::Elementwise({
        let mut e = ElementwiseOp::new(norm_name, norm_bytes);
        e.scale_factor = count;
        e
    }));
    ops_out.push(Op::Gemm({
        let mut g = GemmOp::new(in_proj, gdn_in_proj_out, h, dtypes.gemm_quant);
        g.scale_factor = count;
        g
    }));
    ops_out.push(Op::Gdn(GdnOp {
        name: conv_name.to_string(),
        scale_factor: count,
        kernel_source: conv_kernel.to_string(),
        phase: phase.to_string(),
        d_model: h,
        d_conv,
        num_k_heads: nk,
        head_k_dim: hk,
        num_v_heads: nv,
        head_v_dim: hv,
    }));
    ops_out.push(Op::Gdn(GdnOp {
        name: scan_name.to_string(),
        scale_factor: count,
        kernel_source: scan_kernel.to_string(),
        phase: phase.to_string(),
        d_model: h,
        d_conv,
        num_k_heads: nk,
        head_k_dim: hk,
        num_v_heads: nv,
        head_v_dim: hv,
    }));
    ops_out.push(Op::Gemm({
        let mut g = GemmOp::new(out_proj, h, gdn_out_proj_in, dtypes.gemm_quant);
        g.scale_factor = count;
        g.low_precision_input = true;
        g
    }));
    ops_out.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        ar, count, h, tp,
    )));
}

#[allow(clippy::too_many_arguments)]
fn push_ffn_phase(
    ops_out: &mut Vec<Op>,
    prefix: &str,
    count: f64,
    h: u32,
    tp: u32,
    moe_tp: u32,
    moe_ep: u32,
    attn_dp: u32,
    backend: crate::common::enums::BackendKind,
    flavor: DispatchFlavor,
    dtypes: crate::models::base::DtypeConfig,
    inter_per_tp: u32,
    act_gate_bytes: f64,
    topk: u32,
    num_experts: u32,
    moe_inter: u32,
    shared_inter: u32,
) {
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;
    ops_out.push(Op::Elementwise({
        let mut e = ElementwiseOp::new(format!("{prefix}_ffn_norm"), norm_bytes);
        e.scale_factor = count;
        e
    }));
    if num_experts > 0 {
        // Router GEMM (only emitted when num_experts >= 128 per Python).
        if num_experts >= 128 {
            ops_out.push(Op::Gemm({
                let mut g = GemmOp::new(
                    format!("{prefix}_router_gemm"),
                    num_experts,
                    h,
                    GemmQuantMode::Bfloat16,
                );
                g.scale_factor = count;
                g
            }));
        }
        ops_out.push(Op::MoeDispatch({
            let mut d = MoEDispatchOp::new(
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
            d.scale_factor = count;
            d
        }));
        ops_out.push(Op::Moe({
            let mut m = MoeOp::new(
                format!("{prefix}_moe"),
                h,
                moe_inter,
                topk,
                num_experts,
                moe_tp,
                moe_ep,
                dtypes.moe_quant,
                "power_law_1.2",
            );
            m.scale_factor = count;
            m
        }));
        ops_out.push(Op::MoeDispatch({
            let mut d = MoEDispatchOp::new(
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
            d.scale_factor = count;
            d
        }));
        if shared_inter > 0 {
            let shared_per_tp = shared_inter / tp;
            ops_out.push(Op::Gemm({
                let mut g = GemmOp::new(
                    format!("{prefix}_shared_up_gemm"),
                    shared_per_tp,
                    h,
                    dtypes.gemm_quant,
                );
                g.scale_factor = count;
                g
            }));
            // Python's shared expert uses Relu2: read shared_per_tp,
            // write shared_per_tp. bytes/token = 2 * shared_per_tp * 2.
            let relu2_bytes = (shared_per_tp as f64 + shared_per_tp as f64) * 2.0;
            ops_out.push(Op::Elementwise({
                let mut e = ElementwiseOp::new(format!("{prefix}_shared_relu2"), relu2_bytes);
                e.scale_factor = count;
                e
            }));
            ops_out.push(Op::Gemm({
                let mut g = GemmOp::new(
                    format!("{prefix}_shared_down_gemm"),
                    h,
                    shared_per_tp,
                    dtypes.gemm_quant,
                );
                g.scale_factor = count;
                g.low_precision_input = true;
                g
            }));
        }
    } else {
        // Dense gated FFN.
        ops_out.push(Op::Gemm({
            let mut g = GemmOp::new(
                format!("{prefix}_gate_ffn1_gemm"),
                2 * inter_per_tp,
                h,
                dtypes.gemm_quant,
            );
            g.scale_factor = count;
            g
        }));
        ops_out.push(Op::Elementwise({
            let mut e = ElementwiseOp::new(format!("{prefix}_act_gate"), act_gate_bytes);
            e.scale_factor = count;
            e
        }));
        ops_out.push(Op::Gemm({
            let mut g = GemmOp::new(
                format!("{prefix}_ffn2_gemm"),
                h,
                inter_per_tp,
                dtypes.gemm_quant,
            );
            g.scale_factor = count;
            g.low_precision_input = true;
            g
        }));
        ops_out.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            format!("{prefix}_ffn_ar"),
            count,
            h,
            tp,
        )));
    }
}
