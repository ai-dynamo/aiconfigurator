// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V3 / Kimi-K2.5 op-list builder.
//!
//! Mirrors `aiconfigurator.sdk.models.deepseek.DeepSeekModel.__init__`.
//! Uses module-level MLA queries (vLLM b200 data shape) + MoE FFN with
//! all-to-all dispatch.

use crate::common::enums::{BackendKind, GemmQuantMode};
use crate::models::base::{Model, ModelConfig};
use crate::operators::{
    ContextAttentionOp, ContextMlaOp, CustomAllReduceOp, DispatchFlavor, ElementwiseOp,
    EmbeddingOp, FallbackOp, GemmOp, GenerationAttentionOp, GenerationMlaOp, MlaBmmOp,
    MlaModuleOp, MoEDispatchOp, MoeOp, Op, OverlapOp,
};

fn dispatch_flavor(backend: BackendKind) -> DispatchFlavor {
    // See `models/moe.rs::dispatch_flavor` for the SGLang-default rationale.
    match backend {
        BackendKind::Trtllm => DispatchFlavor::TrtllmAlltoall,
        BackendKind::Sglang => DispatchFlavor::CustomAllReduce,
        BackendKind::Vllm => DispatchFlavor::CustomAllReduce,
    }
}

pub fn build_deepseek_model(config: ModelConfig) -> Model {
    let mut model = Model::new(config);
    let cfg = &model.config;
    let layers = cfg.spec.num_hidden_layers as f64;
    // Python's `_mtp_scale_factor` (models/base.py:105-110): for
    // DeepSeek-family models with `nextn >= 1`, generation ops are scaled
    // by `1/(1+E[accept]) * (nextn+layers)/layers`. Context ops are NOT
    // scaled (Python sets MTP scale only on the generation-side ops).
    // When `nextn == 0` this returns 1.0 — no-op.
    let mtp_scale = cfg.mtp_scale_factor();
    let gen_layers = layers * mtp_scale;
    let h = cfg.spec.hidden_size;
    let tp = cfg.parallel.tp_size.max(1);
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
    // Python: ElementWise(num_layers, dim_in=2h, dim_out=2h) → (2h + 2h) * 2 bytes/token.
    // The 0.8 in Python's constructor is stored but unused at query time.
    let norm_bytes = (h as f64 * 2.0 + h as f64 * 2.0) * 2.0;

    // ---- Context phase ----
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
    // MLA's effective per-rank head count is `128 / tp_size` regardless of
    // `num_attention_heads` (mirrors Python `models/deepseek.py`'s hard-coded
    // `128 // tp_size`). DeepSeek-style MLA always profiles against 128
    // attention heads; smaller models like Kimi-K2.5 (`num_attention_heads
    // = 64`) still use 128 here.
    let mla_num_heads = (128_u32 / tp).max(1);
    let n_per_tp = cfg.spec.num_attention_heads / tp;
    // Mirrors Python `deepseek.py`'s `mla_bmm_quant_mode`: fp8 if the
    // gemm quant isn't bf16, else bf16.
    let mla_bmm_quant = if dtypes.gemm_quant != GemmQuantMode::Bfloat16 {
        GemmQuantMode::Fp8
    } else {
        GemmQuantMode::Bfloat16
    };

    // Build the per-layer fallback chain Python's `FallbackOp` would
    // unfold when `MLAModule` perf data is missing. The middle attention
    // op differs by backend: vLLM absorbs the KV projection into a single
    // `ContextAttention` (matches Python `models/deepseek.py` line 168 —
    // "vLLM absorbs the KV projection and runs standard ContextAttention
    // with v_head_dim=128"); TRT-LLM / SGLang use the granular `ContextMLA`.
    // Neither path wraps the attention in MlaBmm on the context side —
    // Python's context fallback has exactly one attention op.
    let vllm_head_size: u32 = 128; // DeepSeek/Kimi MLA architectural v_head_dim
    let mut context_mla_fallback: Vec<Op> = Vec::new();
    context_mla_fallback.push(Op::Gemm({
        let mut g = GemmOp::new("context_downscale_gemm", 2112, h, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    context_mla_fallback.push(Op::Gemm({
        let mut g = GemmOp::new("context_q_b_proj_gemm", 24576 / tp, 1536, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    context_mla_fallback.push(Op::Gemm({
        let mut g = GemmOp::new("context_kv_b_proj_gemm", 32768 / tp, 512, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));
    if backend == BackendKind::Vllm {
        context_mla_fallback.push(Op::ContextAttention({
            let mut a = ContextAttentionOp::new(
                "context_attention",
                cfg.spec.num_attention_heads / tp,
                cfg.spec.num_key_value_heads / tp,
                vllm_head_size,
                dtypes.kv_cache_quant,
                dtypes.fmha_quant,
            );
            a.scale_factor = layers;
            a
        }));
    } else {
        context_mla_fallback.push(Op::ContextMla({
            let mut a = ContextMlaOp::new(
                "context_attention",
                mla_num_heads,
                dtypes.kv_cache_quant,
                dtypes.fmha_quant,
            );
            a.scale_factor = layers;
            a
        }));
    }
    context_mla_fallback.push(Op::Gemm({
        let mut g = GemmOp::new("context_proj_gemm", h, 128 * 128 / tp, dtypes.gemm_quant);
        g.scale_factor = layers;
        g
    }));

    let context_mla_primary = Op::MlaModuleContext({
        let mut m = MlaModuleOp::new(
            "context_mla_module",
            mla_num_heads,
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
            dtypes.gemm_quant,
        );
        m.scale_factor = layers;
        m
    });
    ctx.push(Op::Fallback(FallbackOp::new(
        "context_mla_block",
        context_mla_primary,
        context_mla_fallback,
    )));
    ctx.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("context_add_norm_2", norm_bytes);
        e.scale_factor = layers;
        e
    }));
    // No per-layer AR — handled inside MoEDispatch.

    // Shared expert FFN (Kimi/DeepSeek: 1 shared expert per layer that
    // every token traverses; gated FFN with moe_intermediate_size).
    let moe_inter_per_tp = (moe_inter + tp - 1) / tp;
    ctx.push(Op::Gemm({
        let mut g = GemmOp::new(
            "context_shared_ffn1_gemm",
            2 * moe_inter_per_tp,
            h,
            dtypes.gemm_quant,
        );
        g.scale_factor = layers;
        g
    }));
    // Python: ElementWise(num_layers, dim_in=2*inter, dim_out=inter, 0.8).
    // bytes/token = (2*inter + inter) * 2 (bf16). The 0.8 is the unused
    // empirical_bw_scaling_factor.
    let shared_act_gate_bytes = (2.0 * moe_inter_per_tp as f64 + moe_inter_per_tp as f64) * 2.0;
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

    ctx.push(Op::Gemm({
        let mut g = GemmOp::new("context_router_gemm", num_experts, h, GemmQuantMode::Bfloat16);
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
    // Context-phase logits projection (Python's DeepSeek includes this
    // in context_ops with scale_factor=1, unlike MOE which puts it only
    // in generation_ops).
    ctx.push(Op::Gemm(GemmOp::new(
        "context_logits_gemm",
        vocab_per_tp,
        h,
        GemmQuantMode::Bfloat16,
    )));

    // vLLM-specific: per-layer TP allreduce. vLLM's `AllReduceFusionPass`
    // only fuses the AR into the kernel for pure-decode CUDA-graph steps;
    // chunked-prefill iterations and other backends pay the explicit cost.
    // scale_factor = 2 * num_layers covers attention-side and FFN-side AR.
    // Python's DeepSeek uses this as the ONLY context AR (no separate
    // embedding_ar).
    if backend == BackendKind::Vllm {
        ctx.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "context_tp_allreduce",
            2.0 * layers,
            h,
            tp,
        )));
    }

    model.context_ops = ctx;

    // ---- Generation phase ----
    // Every generation-side op uses `gen_layers = layers * mtp_scale`
    // (where `mtp_scale = 1.0` when MTP is disabled). The once-per-forward
    // ops (embedding, logits_gemm) multiply their constant by `mtp_scale`
    // directly. Mirrors Python `deepseek.py:317-522` where every
    // generation_op is scaled by `self._mtp_scale_factor`.
    let mut gen = Vec::with_capacity(12);

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
    // Generation fallback chain (SGLang / TRT-LLM shape). The `h // tp`
    // proj_gemm size matches Python (generation uses `h // tp` while
    // context uses `128 * 128 // tp`).
    let mut generation_mla_fallback: Vec<Op> = Vec::new();
    generation_mla_fallback.push(Op::Gemm({
        let mut g = GemmOp::new("generation_downscale_gemm", 2112, h, dtypes.gemm_quant);
        g.scale_factor = gen_layers;
        g
    }));
    generation_mla_fallback.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_q_b_proj_gemm",
            24576 / tp,
            1536,
            dtypes.gemm_quant,
        );
        g.scale_factor = gen_layers;
        g
    }));
    // Generation attention: vLLM absorbs the KV projection into one
    // GenerationAttention (matches Python `deepseek.py` line 357 — "vLLM
    // absorbs the KV projection and runs standard GenerationAttention with
    // v_head_dim=128"); TRT-LLM / SGLang use the full MlaBmm(pre) +
    // GenerationMLA + MlaBmm(post) chain.
    if backend == BackendKind::Vllm {
        generation_mla_fallback.push(Op::GenerationAttention({
            let mut a = GenerationAttentionOp::new(
                "generation_attention",
                cfg.spec.num_attention_heads / tp,
                cfg.spec.num_key_value_heads / tp,
                vllm_head_size,
                dtypes.kv_cache_quant,
            );
            a.scale_factor = gen_layers;
            a
        }));
    } else {
        generation_mla_fallback.push(Op::MlaBmm({
            let mut b = MlaBmmOp::new("generation_bmm_pre", n_per_tp, mla_bmm_quant, true);
            b.scale_factor = gen_layers;
            b
        }));
        generation_mla_fallback.push(Op::GenerationMla({
            let mut a =
                GenerationMlaOp::new("generation_attention", mla_num_heads, dtypes.kv_cache_quant);
            a.scale_factor = gen_layers;
            a
        }));
        generation_mla_fallback.push(Op::MlaBmm({
            let mut b = MlaBmmOp::new("generation_bmm_post", n_per_tp, mla_bmm_quant, false);
            b.scale_factor = gen_layers;
            b
        }));
    }
    generation_mla_fallback.push(Op::Gemm({
        let mut g = GemmOp::new("generation_proj_gemm", h, h / tp, dtypes.gemm_quant);
        g.scale_factor = gen_layers;
        g
    }));

    let generation_mla_primary = Op::MlaModuleGeneration({
        let mut m = MlaModuleOp::new(
            "generation_mla_module",
            mla_num_heads,
            dtypes.kv_cache_quant,
            dtypes.fmha_quant,
            dtypes.gemm_quant,
        );
        m.scale_factor = gen_layers;
        m
    });
    gen.push(Op::Fallback(FallbackOp::new(
        "generation_mla_block",
        generation_mla_primary,
        generation_mla_fallback,
    )));
    gen.push(Op::Elementwise({
        let mut e = ElementwiseOp::new("generation_add_norm_2", norm_bytes);
        e.scale_factor = gen_layers;
        e
    }));
    // No per-layer AR — handled inside MoEDispatch.

    // Mirrors Python `deepseek.py` generation overlap: the shared-expert
    // FFN (aux CUDA stream) runs in parallel with the routed-expert path
    // (main CUDA stream) when CUDA Graph is enabled. Latency =
    // max(sum(routed), sum(shared)) via `Op::Overlap`.
    let mut gen_shared = Vec::with_capacity(3);
    gen_shared.push(Op::Gemm({
        let mut g = GemmOp::new(
            "generation_shared_ffn1_gemm",
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
        // Logits GEMM fires once per forward; Python's `1 * mtp_scale_factor`.
        let mut g = GemmOp::new(
            "generation_logits_gemm",
            vocab_per_tp,
            h,
            GemmQuantMode::Bfloat16,
        );
        g.scale_factor = mtp_scale;
        g
    }));
    // No separate embedding_ar for DeepSeek — Python uses only the
    // conditional tp_allreduce below.

    if backend == BackendKind::Vllm {
        gen.push(Op::CustomAllReduce(CustomAllReduceOp::new(
            "generation_tp_allreduce",
            2.0 * gen_layers,
            h,
            tp,
        )));
    }

    model.generation_ops = gen;
    model
}
