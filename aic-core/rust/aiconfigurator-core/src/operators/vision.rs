// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vision encoder operators.
//!
//! Mirrors `aiconfigurator.sdk.models.vit_ops.build_encoder_ops`. The
//! encoder runs once before the LLM context phase and is composed from
//! standard transformer ops (QKV GEMM + encoder attention + out-proj GEMM
//! + FFN GEMMs + ElementWise norms). This module exposes a single
//! `VisionEncoderOp` that bundles those pieces — the model layer calls it
//! once for the encoder phase.

use serde::{Deserialize, Serialize};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode};
use crate::common::error::AicError;
use crate::operators::attention::EncoderAttentionOp;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::elementwise::ElementwiseOp;
use crate::operators::gemm::GemmOp;
use crate::perf_database::PerfDatabase;

/// ViT-style encoder: QKV → attention → out-proj → FFN → norms.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionEncoderOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_layers: u32,
    pub num_heads: u32,
    pub head_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub fmha_quant: FmhaQuantMode,
    pub gemm_quant: GemmQuantMode,
}

impl VisionEncoderOp {
    /// Encoder forward latency for `num_image_tokens` tokens. Returned as
    /// a single composed `PerformanceResult` — the model layer multiplies
    /// by num-images-per-prompt if needed.
    pub fn query(
        &self,
        db: &PerfDatabase,
        num_image_tokens: u32,
    ) -> Result<PerformanceResult, AicError> {
        if num_image_tokens == 0 {
            return Ok(PerformanceResult::zero());
        }
        let q = self.num_heads * self.head_size;
        let qkv = GemmOp {
            name: format!("{}.qkv", self.name),
            scale_factor: 1.0,
            n: q * 3,
            k: self.hidden_size,
            quant_mode: self.gemm_quant,
            scale_num_tokens: 1,
            low_precision_input: false,
                seq_split: 1,
        };
        let attn = EncoderAttentionOp::new(
            format!("{}.attn", self.name),
            self.num_heads,
            self.head_size,
            self.fmha_quant,
        );
        let out_proj = GemmOp {
            name: format!("{}.out_proj", self.name),
            scale_factor: 1.0,
            n: self.hidden_size,
            k: q,
            quant_mode: self.gemm_quant,
            scale_num_tokens: 1,
            low_precision_input: false,
                seq_split: 1,
        };
        let ffn1 = GemmOp {
            name: format!("{}.ffn1", self.name),
            scale_factor: 1.0,
            n: self.intermediate_size,
            k: self.hidden_size,
            quant_mode: self.gemm_quant,
            scale_num_tokens: 1,
            low_precision_input: false,
                seq_split: 1,
        };
        let ffn2 = GemmOp {
            name: format!("{}.ffn2", self.name),
            scale_factor: 1.0,
            n: self.hidden_size,
            k: self.intermediate_size,
            quant_mode: self.gemm_quant,
            scale_num_tokens: 1,
            low_precision_input: false,
                seq_split: 1,
        };
        let norms = ElementwiseOp::new(
            format!("{}.norms", self.name),
            (self.hidden_size as f64) * 2.0, // bf16 norm in + out
        );

        let mut total = 0.0;
        // batch_size for vision attention is 1 (single image per call).
        total += qkv.query(db, num_image_tokens, None)?.latency_ms;
        total += attn.query(db, 1, num_image_tokens)?.latency_ms;
        total += out_proj.query(db, num_image_tokens, None)?.latency_ms;
        total += ffn1.query(db, num_image_tokens, None)?.latency_ms;
        total += ffn2.query(db, num_image_tokens, None)?.latency_ms;
        // Norms run twice per layer.
        total += 2.0 * norms.query(db, num_image_tokens)?.latency_ms;

        let per_layer = total;
        let all_layers = per_layer * self.num_layers as f64;
        Ok(PerformanceResult::new(all_layers, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}
