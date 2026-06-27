// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 attention module operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.dsv4.DSV4Module`. Each layer of
//! DSv4 picks between CSA (compressed-sparse) and HCA (hybrid-causal)
//! variants depending on the layer index — the model layer decides which
//! `AttnKind` to use; the operator just routes to the right
//! `db.dsv4.query_*` slice.
//!
//! Slice selection uses the physical `(model profile, tp_size, local_heads)`
//! key, mirroring Python `_dsv4_resolve_head_key`. Keeping the three together
//! prevents overlapping local-head counts from different V4 variants or TP
//! shardings from overwriting one another.

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::dsv4::AttnKind;
use crate::perf_database::PerfDatabase;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv4ModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub attn_kind: AttnKind,
    /// Per-rank partitioned head count (`native_heads / tp_size`).
    pub num_heads: u32,
    /// Model total attention head count; selects Flash (64) vs Pro (128).
    pub native_heads: u32,
    /// Tensor-parallel size; part of the physical module lookup key.
    pub tp_size: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    pub architecture: String,
}

impl Dsv4ModuleOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        attn_kind: AttnKind,
        num_heads: u32,
        native_heads: u32,
        tp_size: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        architecture: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            attn_kind,
            num_heads,
            native_heads,
            tp_size,
            kv_cache_dtype,
            fmha_quant_mode,
            gemm_quant_mode,
            architecture: architecture.into(),
        }
    }

    pub fn query_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let raw = db.dsv4.query_context(
            self.attn_kind,
            batch_size,
            isl,
            prefix,
            self.num_heads,
            self.native_heads,
            self.tp_size,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
        )?;
        Ok(PerformanceResult::new(raw, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let latency = db.dsv4.query_generation(
            self.attn_kind,
            batch_size,
            s,
            self.num_heads,
            self.native_heads,
            self.tp_size,
            self.kv_cache_dtype,
            self.gemm_quant_mode,
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}
