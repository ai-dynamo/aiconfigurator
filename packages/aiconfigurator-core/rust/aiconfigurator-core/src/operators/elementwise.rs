// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Element-wise (memory-bandwidth-bound) operators.
//!
//! Mirrors `aiconfigurator_core.sdk.operations.elementwise`. Activation functions,
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
    /// Bytes touched per input token. The query multiplies this by the
    /// runtime token count.
    pub bytes_per_token: f64,
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank token count is `ceil(num_tokens / seq_split)`. Defaults to 1.
    /// (Python's `scale_num_tokens` is folded into `bytes_per_token` by the
    /// serializer; `seq_split` is applied here to the token count.)
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

impl ElementwiseOp {
    pub fn new(name: impl Into<String>, bytes_per_token: f64) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            bytes_per_token,
            seq_split: 1,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let num_tokens = num_tokens.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        let bytes = self.bytes_per_token * (num_tokens as f64);
        let latency = mem_op_latency_ms(&db.system_spec, bytes);
        Ok(PerformanceResult::new(latency, Source::Empirical).scaled(self.scale_factor))
    }
}
