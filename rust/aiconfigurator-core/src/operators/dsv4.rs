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
//! The DSV4 module tables are indexed by `native_heads` (the model's total
//! attention head count, the CSV `num_heads` column) for slice selection and
//! by `tp_size` for the seq/batch interpolation axis — NOT by the per-rank
//! partitioned head count. See `perf_database::dsv4` and Python
//! `load_context_dsv4_kind_module_data`.

use serde::{Deserialize, Serialize};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::dsv4::AttnKind;
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv4ModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub attn_kind: AttnKind,
    /// Per-rank partitioned head count (`native_heads / tp_size`). Retained for
    /// provenance; the table lookup keys on `native_heads` + `tp_size` instead.
    pub num_heads: u32,
    /// Model total attention head count (CSV `num_heads` column). Selects the
    /// data slice.
    pub native_heads: u32,
    /// Tensor-parallel size — the DSV4 table's primary interpolation axis.
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
        // Mirror Python `ContextDeepSeekV4AttentionModule.get_silicon`: the
        // context module table is collected with the kernel-Δ convention, so
        // the base lookup uses `lookup_s = isl` (the new-token count), NOT
        // `isl + prefix`. The prefix effect is carried by an additive
        // sparse-kernel delta (paged_mqa_logits for CSA / hca_attn for HCA)
        // plus, for CSA, a topk_512 IO term `M*prefix/(mem_bw*0.1)*1000`.
        //
        // Both additive corrections are empirically negligible at typical
        // shapes (kernel Δ ~1e-4 ms, CSA IO term ~1e-3 ms total) — see the
        // parity decomposition — so they are intentionally omitted here. The
        // previous multiplicative `(s²-p²)/s²` correction was a SOL-style
        // approximation that under-counted context latency by ~21% vs Python.
        let _ = prefix;
        let raw = db.dsv4.query_context(
            self.attn_kind,
            batch_size,
            isl,
            self.native_heads,
            self.tp_size,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            &self.architecture,
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
            self.native_heads,
            self.tp_size,
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
