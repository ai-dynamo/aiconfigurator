// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (Dynamic Sparse Attention) module operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.dsa.DSAModule`. Wraps the
//! `db.dsa.query_context/generation` raw lookups with prefix correction
//! and clamping. The topk-piecewise interpolation algorithm is handled by
//! the perf-DB layer in a follow-up; this operator currently uses the
//! 3-D interpolated path for the matching prefix slice.

use serde::{Deserialize, Serialize};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

fn prefix_correction(full_s: u32, prefix: u32) -> f64 {
    if full_s == 0 {
        return 0.0;
    }
    let f = full_s as f64;
    let p = prefix as f64;
    (f * f - p * p) / (f * f)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DsaModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    pub architecture: String,
}

impl DsaModuleOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        architecture: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
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
        let full_s = isl + prefix;
        let raw = db.dsa.query_context(
            batch_size,
            full_s,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            &self.architecture,
            prefix,
        )?;
        let latency = raw * prefix_correction(full_s, prefix);
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let latency = db.dsa.query_generation(
            batch_size,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            &self.architecture,
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}
