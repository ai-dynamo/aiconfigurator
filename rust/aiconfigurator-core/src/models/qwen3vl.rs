// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qwen3-VL multimodal builders (Qwen3VL + Qwen3VL_MOE).
//!
//! Apple-to-apple port of `aiconfigurator.sdk.models.qwen3vl.{Qwen3VLModel,
//! Qwen3VLMoEModel}` and `aiconfigurator.sdk.models.vit_ops.build_encoder_ops`.
//!
//! The LLM backbone reuses the LLAMA / MOE builder for text-only ops, then
//! a ViT vision encoder (`encoder_ops` on `Model`) is appended before the
//! LLM prefill phase. The encoder always runs in bf16 regardless of the
//! LLM's quantization settings; TP is applied to ViT heads and FFN the
//! same way as the LLM backbone.
//!
//! Op shape (per the Python docstring):
//!   _vit_transformer_ops -> 10 ops, each with count=depth:
//!     encoder_add_norm_1     ElementWise
//!     encoder_qkv_gemm       GEMM
//!     encoder_attention      ContextAttention
//!     encoder_proj_gemm      GEMM (low_precision_input=True)
//!     encoder_ar_1           CustomAllReduce
//!     encoder_add_norm_2     ElementWise
//!     encoder_ffn1_gemm      GEMM
//!     encoder_act            ElementWise
//!     encoder_ffn2_gemm      GEMM (low_precision_input=True)
//!     encoder_ar_2           CustomAllReduce
//!   _projector_ops -> 2*P ops + 1 AR (or 0 ops if projector_dims empty):
//!     encoder_projector_fc{i}_gemm  GEMM
//!     encoder_projector_fc{i}_act   ElementWise (omitted for final layer)
//!     encoder_projector_ar          CustomAllReduce

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::models::base::{Model, ModelConfig};
use crate::models::config_loader::VisionEncoderConfig;
use crate::models::llama::build_llama_model;
use crate::models::moe::build_moe_model;
use crate::operators::{
    ContextAttentionOp, CustomAllReduceOp, ElementwiseOp, GemmOp, Op,
};

/// ViT always runs in bf16 regardless of LLM quantization.
const VIT_GEMM_QUANT: GemmQuantMode = GemmQuantMode::Bfloat16;
const VIT_FMHA_QUANT: FmhaQuantMode = FmhaQuantMode::Bfloat16;
const VIT_KV_QUANT: KvCacheQuantMode = KvCacheQuantMode::Bfloat16;

fn vit_transformer_ops(enc: &VisionEncoderConfig, tp: u32) -> Result<Vec<Op>, AicError> {
    let depth = enc.depth as f64;
    let h_vit = enc.hidden_size;
    let n_vit = enc.num_heads;
    let inter_vit = enc.intermediate_size;
    let head_size_vit = h_vit / n_vit.max(1);

    if tp > 1 {
        if n_vit % tp != 0 {
            return Err(AicError::UnsupportedModel(format!(
                "ViT num_heads ({n_vit}) must be divisible by tp_size ({tp})"
            )));
        }
        if inter_vit % tp != 0 {
            return Err(AicError::UnsupportedModel(format!(
                "ViT intermediate_size ({inter_vit}) must be divisible by tp_size ({tp})"
            )));
        }
    }

    let n_per_tp = n_vit / tp.max(1);
    let inter_per_tp = inter_vit / tp.max(1);
    // ElementWise(2h, 2h) -> (2h + 2h) * 2 bytes/token for bf16.
    let norm_bytes = (h_vit as f64 * 2.0 + h_vit as f64 * 2.0) * 2.0;
    let act_bytes = (inter_per_tp as f64 + inter_per_tp as f64) * 2.0;

    let mut ops = Vec::with_capacity(10);

    let mut e1 = ElementwiseOp::new("encoder_add_norm_1", norm_bytes);
    e1.scale_factor = depth;
    ops.push(Op::Elementwise(e1));

    let mut qkv = GemmOp::new(
        "encoder_qkv_gemm",
        3 * n_per_tp * head_size_vit,
        h_vit,
        VIT_GEMM_QUANT,
    );
    qkv.scale_factor = depth;
    ops.push(Op::Gemm(qkv));

    let mut attn = ContextAttentionOp::new(
        "encoder_attention",
        n_per_tp,
        n_per_tp, // ViT has no GQA: n_kv == n
        head_size_vit,
        VIT_KV_QUANT,
        VIT_FMHA_QUANT,
    );
    attn.scale_factor = depth;
    ops.push(Op::ContextAttention(attn));

    let mut proj = GemmOp::new(
        "encoder_proj_gemm",
        h_vit,
        n_per_tp * head_size_vit,
        VIT_GEMM_QUANT,
    );
    proj.scale_factor = depth;
    proj.low_precision_input = true;
    ops.push(Op::Gemm(proj));

    ops.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "encoder_ar_1",
        depth,
        h_vit,
        tp,
    )));

    let mut e2 = ElementwiseOp::new("encoder_add_norm_2", norm_bytes);
    e2.scale_factor = depth;
    ops.push(Op::Elementwise(e2));

    let mut ffn1 = GemmOp::new("encoder_ffn1_gemm", inter_per_tp, h_vit, VIT_GEMM_QUANT);
    ffn1.scale_factor = depth;
    ops.push(Op::Gemm(ffn1));

    let mut act = ElementwiseOp::new("encoder_act", act_bytes);
    act.scale_factor = depth;
    ops.push(Op::Elementwise(act));

    let mut ffn2 = GemmOp::new("encoder_ffn2_gemm", h_vit, inter_per_tp, VIT_GEMM_QUANT);
    ffn2.scale_factor = depth;
    ffn2.low_precision_input = true;
    ops.push(Op::Gemm(ffn2));

    ops.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "encoder_ar_2",
        depth,
        h_vit,
        tp,
    )));

    Ok(ops)
}

fn projector_ops(enc: &VisionEncoderConfig, tp: u32) -> Vec<Op> {
    let dims = &enc.projector_dims;
    if dims.is_empty() {
        return Vec::new();
    }
    let n_inst = enc.projector_n_instances as f64;
    let n_layers = dims.len();
    let mut ops = Vec::with_capacity(n_layers * 2 + 1);

    for (i, &(in_d, out_d)) in dims.iter().enumerate() {
        let is_last = i == n_layers - 1;
        // Final layer in multi-layer projector takes sharded input from the
        // previous row-parallel layer (column-parallel style). Single-layer
        // and non-final layers always receive a full (non-sharded) input
        // (row-parallel).
        let col_parallel = is_last && n_layers > 1;
        let (m, k) = if col_parallel {
            (out_d, in_d / tp.max(1))
        } else {
            (out_d / tp.max(1), in_d)
        };
        let mut g = GemmOp::new(
            format!("encoder_projector_fc{i}_gemm"),
            m,
            k,
            VIT_GEMM_QUANT,
        );
        g.scale_factor = n_inst;
        ops.push(Op::Gemm(g));
        if !is_last {
            let out_per_tp = out_d / tp.max(1);
            let act_bytes = (out_per_tp as f64 + out_per_tp as f64) * 2.0;
            let mut act = ElementwiseOp::new(format!("encoder_projector_fc{i}_act"), act_bytes);
            act.scale_factor = n_inst;
            ops.push(Op::Elementwise(act));
        }
    }

    let final_out = dims.last().map(|d| d.1).unwrap_or(0);
    ops.push(Op::CustomAllReduce(CustomAllReduceOp::new(
        "encoder_projector_ar",
        n_inst,
        final_out,
        tp,
    )));
    ops
}

/// Build the complete list of encoder ops for a ViT-based vision encoder.
/// Equivalent to Python `vit_ops.build_encoder_ops`.
pub fn build_encoder_ops(enc: &VisionEncoderConfig, tp: u32) -> Result<Vec<Op>, AicError> {
    let mut ops = vit_transformer_ops(enc, tp)?;
    ops.extend(projector_ops(enc, tp));
    Ok(ops)
}

/// Qwen3VL (dense LLaMA-style backbone + ViT). Routes
/// `ModelFamily::Qwen3Vl` in factory.rs.
pub fn build_qwen3vl_model(config: ModelConfig) -> Result<Model, AicError> {
    let vision_encoder = config.spec.vision_encoder.clone();
    let tp = config.parallel.tp_size.max(1);
    let mut model = build_llama_model(config);
    if let Some(enc) = vision_encoder.as_ref() {
        model.encoder_ops = build_encoder_ops(enc, tp)?;
    }
    Ok(model)
}

/// Qwen3VL_MOE (sparse MoE backbone + ViT). Routes
/// `ModelFamily::Qwen3VlMoe` in factory.rs.
pub fn build_qwen3vl_moe_model(config: ModelConfig) -> Result<Model, AicError> {
    let vision_encoder = config.spec.vision_encoder.clone();
    let tp = config.parallel.tp_size.max(1);
    let mut model = build_moe_model(config);
    if let Some(enc) = vision_encoder.as_ref() {
        model.encoder_ops = build_encoder_ops(enc, tp)?;
    }
    Ok(model)
}
