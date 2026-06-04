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
    /// Bytes touched per input token. The query multiplies this by the
    /// runtime token count.
    pub bytes_per_token: f64,
}

impl ElementwiseOp {
    pub fn new(name: impl Into<String>, bytes_per_token: f64) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            bytes_per_token,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let bytes = self.bytes_per_token * (num_tokens as f64);
        let latency = mem_op_latency_ms(&db.system_spec, bytes);
        Ok(PerformanceResult::new(latency, Source::Empirical).scaled(self.scale_factor))
    }
}
