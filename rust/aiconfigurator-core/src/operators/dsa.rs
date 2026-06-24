// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (Dynamic Sparse Attention) module operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.dsa.ContextDSAModule` /
//! `GenerationDSAModule`. The context lookup keys the per-`prefix` slice and
//! evaluates at `isl` (the new-token count), running the top-k regime-aware
//! piecewise interpolation first and the DSv4 robust 3-D / batch-scaling
//! lookup as the fallback (see `perf_database::dsa::query_context`).
//!
//! `index_topk` is the top-k boundary (per-architecture; 2048 for both
//! DeepSeek-V3.2 and GLM-5). It is plumbed from the Python op-spec emitter.

use serde::{Deserialize, Serialize};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DsaModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    pub architecture: String,
    /// Top-k boundary for the sparse-attention regime split. Sourced from
    /// `DSA_MODEL_DIMS[architecture]["index_topk"]` on the Python side.
    pub index_topk: u32,
}

impl DsaModuleOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        architecture: impl Into<String>,
        index_topk: u32,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            gemm_quant_mode,
            architecture: architecture.into(),
            index_topk,
        }
    }

    pub fn query_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        // Query at `isl` (new-token count) for the exact `prefix` slice — NOT
        // `isl + prefix`. The perf-DB layer runs the piecewise + robust 3-D
        // dispatch; there is no multiplicative prefix correction (it had no
        // Python counterpart and under-counted context latency ~75%).
        let latency = db.dsa.query_context(
            batch_size,
            isl,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            &self.architecture,
            prefix,
            self.index_topk,
        )?;
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
