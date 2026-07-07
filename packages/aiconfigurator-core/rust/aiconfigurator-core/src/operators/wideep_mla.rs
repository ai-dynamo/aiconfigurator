// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang WideEP MLA operators (context + generation).
//!
//! Apple-to-apple port of `aiconfigurator_core.sdk.operations.mla.{WideEPContextMLA,
//! WideEPGenerationMLA}`. These are SGLang-only ops used by the WideEP
//! DeepSeek variant — Python loads the tables lazily and errors at query
//! time when the backend isn't `sglang`. The Rust perf-database layer
//! delegates the table miss to the operator's per-call SOL fallback,
//! matching the legacy MLA / DSA contract.
//!
//! The two ops carry the same configuration (num_heads, quant modes,
//! attention backend), but the Python signatures differ slightly: context
//! takes a `prefix` parameter so the operator can apply
//! `prefix_correction = (full_s^2 - prefix^2) / full_s^2`. Generation has
//! no prefix concept.

use serde::{Deserialize, Serialize};
use crate::common::enums::{FmhaQuantMode, KvCacheQuantMode};
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
pub struct WideEpContextMlaOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    /// Mirrors Python's `attn_backend` argument: `"flashinfer"` (default)
    /// or `"fa3"`. The CSV's `kernel_source` column carries this value.
    pub attn_backend: String,
}

impl WideEpContextMlaOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            attn_backend: "flashinfer".to_string(),
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let full_s = isl + prefix;
        let raw = db.wideep_mla.query_context(
            batch_size,
            full_s,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            &self.attn_backend,
        )?;
        let latency = raw * prefix_correction(full_s, prefix);
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WideEpGenerationMlaOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    /// Python `WideEPGenerationMLA` stores `_fmha_quant_mode` even though
    /// the generation perf-DB nesting doesn't key by it; carried here to
    /// keep the struct shape close to the Python class.
    pub fmha_quant_mode: FmhaQuantMode,
    pub attn_backend: String,
}

impl WideEpGenerationMlaOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            attn_backend: "flashinfer".to_string(),
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let latency = db.wideep_mla.query_generation(
            batch_size,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            &self.attn_backend,
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}
