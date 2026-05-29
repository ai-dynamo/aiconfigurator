// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MLA operators: op-level context/generation, module-level
//! context/generation, and MLA BMM (pre/post).
//!
//! Mirrors `aiconfigurator.sdk.operations.mla.{ContextMLA, GenerationMLA,
//! MLAModule, MLABmm}`. Op-level paths apply Python's prefix-correction
//! multiplier inside the operator; module-level paths do the same since the
//! perf-DB layer returns raw table values. MLA BMM has a quant-mode
//! fallback to bfloat16 inside the perf-DB query.

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

#[derive(Clone, Debug)]
pub struct ContextMlaOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
}

impl ContextMlaOp {
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
        let raw = db.mla.query_context(
            batch_size,
            full_s,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
        )?;
        let latency = raw * prefix_correction(full_s, prefix);
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[derive(Clone, Debug)]
pub struct GenerationMlaOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
}

impl GenerationMlaOp {
    pub fn new(name: impl Into<String>, num_heads: u32, kv_cache_dtype: KvCacheQuantMode) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let latency = db
            .mla
            .query_generation(batch_size, s, self.num_heads, self.kv_cache_dtype)?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

/// Module-level MLA operator (context + generation in one struct since
/// they share config-time fields).
#[derive(Clone, Debug)]
pub struct MlaModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
}

impl MlaModuleOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            gemm_quant_mode,
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
        let raw = db.mla.query_context_module(
            batch_size,
            full_s,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
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
        let latency = db.mla.query_generation_module(
            batch_size,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[derive(Clone, Debug)]
pub struct MlaBmmOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub quant_mode: GemmQuantMode,
    pub is_pre: bool,
}

impl MlaBmmOp {
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        quant_mode: GemmQuantMode,
        is_pre: bool,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            quant_mode,
            is_pre,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let latency = db
            .mla
            .query_bmm(num_tokens, self.num_heads, self.quant_mode, self.is_pre)?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems");
        PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0").expect("db must load")
    }

    #[test]
    fn mla_module_context_smoke() {
        let db = b200_vllm_db();
        let op = MlaModuleOp::new(
            "ctx_mod",
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Bfloat16,
        );
        // Exact-hit row latency=0.1351, prefix=0 means prefix_correction=1.0.
        let result = op.query_context(&db, 1, 1, 0).expect("query must succeed");
        assert!(
            (result.latency_ms - 0.1351).abs() < 1e-6,
            "expected recorded module latency, got {}",
            result.latency_ms
        );
    }

    #[test]
    fn mla_op_context_absent_on_vllm_b200() {
        let db = b200_vllm_db();
        let op = ContextMlaOp::new(
            "ctx_op",
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
        );
        // vLLM b200 only ships module-level MLA — op-level should error.
        let err = op.query(&db, 1, 1024, 0).unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
