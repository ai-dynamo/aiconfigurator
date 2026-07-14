// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Element-wise (memory-bandwidth-bound) operators.
//!
//! Mirrors `aiconfigurator.sdk.operations.elementwise`. Activation functions,
//! norms, residual adds, and other ops whose latency is bounded by memory
//! bandwidth, not compute. There is no perf-DB table for these — Python
//! uses the same `query_mem_op` empirical formula for all of them, scaled
//! by per-op token counts and dtype factors.
//!
//! The formula matches Python's `PerfDatabase.query_mem_op`:
//! `(mem_bytes / (mem_bw * empirical_scaling) + constant_latency) * 1000`.

use serde::{Deserialize, Serialize};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::attention::mem_op_latency_ms;
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ElementwiseOp {
    pub name: String,
    pub scale_factor: f64,
    /// Bytes touched per input token (`dim_in * 2 + dim_out * 2`, bf16
    /// activations). The query multiplies this by the runtime token count.
    pub bytes_per_token: f64,
    /// Token-count divisor (Python `_scale_num_tokens`), applied as an
    /// INTEGER floor-division BEFORE the CP ceil-split — matching Python's
    /// `x //= scale; x = ceil(x / seq_split)` order exactly. (Old specs
    /// folded the divisor into `bytes_per_token` instead, which is only
    /// exact when the divisor divides the token count; they deserialize
    /// with the default 1 and keep their folded semantics.)
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub scale_num_tokens: u32,
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank token count is `ceil(num_tokens / seq_split)`. Defaults to 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

impl ElementwiseOp {
    pub fn new(name: impl Into<String>, bytes_per_token: f64) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            bytes_per_token,
            scale_num_tokens: 1,
            seq_split: 1,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // Python order: `x //= scale_num_tokens` (integer floor) THEN the CP
        // ceil-split. Folding the divisor into bytes diverges whenever it
        // does not divide the token count (live on DSv3.2 CP add_norm ops).
        let num_tokens = num_tokens / self.scale_num_tokens.max(1);
        let num_tokens = num_tokens.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        let bytes = self.bytes_per_token * (num_tokens as f64);
        let latency = mem_op_latency_ms(&db.system_spec, bytes);
        Ok(PerformanceResult::new(latency, Source::Empirical).scaled(self.scale_factor))
    }
}
