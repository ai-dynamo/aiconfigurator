// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang WideEP MLA operators (context + generation).
//!
//! Apple-to-apple port of `aiconfigurator.sdk.operations.mla.{WideEPContextMLA,
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
    /// Context-parallel factor (Python `WideEPContextMLA._cp_size`). When
    /// `>1`, prefill MLA is modeled as SGLang AllGather rank-0's two zigzag
    /// chunks: `ctx(c, prefix) + ctx(c, prefix + isl - c)` with
    /// `c = ceil(isl / 2cp)`, mirroring `mla.py:1505-1510` and
    /// `operators/mla.rs::ContextMlaOp`. Absent in pre-CP specs -> 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub cp_size: u32,
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
            cp_size: 1,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        // ctx(s, pfx): the un-sharded wideep context-MLA query for a sequence
        // chunk of length `s` at prefix `pfx`, with prefix correction applied.
        let ctx = |s: u32, pfx: u32| -> Result<f64, AicError> {
            let full_s = s + pfx;
            let raw = db.wideep_mla.query_context(
                batch_size,
                full_s,
                self.num_heads,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
                &self.attn_backend,
            )?;
            Ok(raw * prefix_correction(full_s, pfx))
        };
        // Context parallelism (SGLang AllGather / zigzag): model rank 0's two
        // balanced chunks, c = ceil(isl / 2cp). Mirrors Python
        // `WideEPContextMLA.query` (mla.py:1505-1510) and
        // `operators/mla.rs::ContextMlaOp`.
        let latency = if self.cp_size > 1 {
            let c = isl.div_ceil(2 * self.cp_size).max(1);
            ctx(c, prefix)? + ctx(c, prefix + isl - c)?
        } else {
            ctx(isl, prefix)?
        };
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn b200_sglang_db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "b200_sxm", "sglang", "0.5.10").expect("db loads")
    }

    fn op(cp_size: u32) -> WideEpContextMlaOp {
        let mut op = WideEpContextMlaOp::new(
            "wideep_ctx_mla",
            128,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Fp8Block,
        );
        // b200 sglang 0.5.10 wideep table carries kernel_source=trtllm_mla.
        op.attn_backend = "trtllm_mla".to_string();
        op.cp_size = cp_size;
        op
    }

    /// CP zigzag mirrors Python `WideEPContextMLA.query` (mla.py:1505-1510):
    /// cp>1 models SGLang AllGather rank-0's two chunks
    /// `ctx(c, prefix) + ctx(c, prefix + isl - c)` with `c = ceil(isl / 2cp)`;
    /// cp=1 stays the single full-length query.
    #[test]
    fn cp_zigzag_composition() {
        let db = b200_sglang_db();
        let (b, isl, prefix) = (4u32, 4096u32, 0u32);

        // cp=1 unchanged: exactly the raw single-chunk table query (prefix=0
        // means prefix_correction = 1).
        let baseline = op(1).query(&db, b, isl, prefix).expect("cp=1 query");
        let raw = db
            .wideep_mla
            .query_context(
                b,
                isl + prefix,
                128,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Fp8Block,
                "trtllm_mla",
            )
            .expect("raw table query");
        assert!(
            (baseline.latency_ms - raw).abs() < 1e-12,
            "cp=1 must remain the plain query: {} vs {}",
            baseline.latency_ms,
            raw
        );

        // cp=2 equals the two-chunk sum.
        let cp = 2u32;
        let c = isl.div_ceil(2 * cp).max(1);
        let chunked = op(cp).query(&db, b, isl, prefix).expect("cp=2 query");
        let chunk1 = op(1).query(&db, b, c, prefix).expect("chunk1 query");
        let chunk2 = op(1)
            .query(&db, b, c, prefix + isl - c)
            .expect("chunk2 query");
        assert!(
            (chunked.latency_ms - (chunk1.latency_ms + chunk2.latency_ms)).abs() < 1e-9,
            "cp=2 ({}) must equal two-chunk sum ({} + {})",
            chunked.latency_ms,
            chunk1.latency_ms,
            chunk2.latency_ms
        );
        assert!(
            chunked.latency_ms > 0.0 && chunked.latency_ms < baseline.latency_ms,
            "rank-0 CP work ({}) must be positive and below the full prefill ({})",
            chunked.latency_ms,
            baseline.latency_ms
        );
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
