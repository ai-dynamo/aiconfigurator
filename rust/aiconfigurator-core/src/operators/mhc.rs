// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MHC (Qwen3.5 multi-head channel) module operator.
//!
//! Wraps `db.mhc.query_module`. The MHC module is collected as a single
//! fused kernel; this operator scales the raw latency by `scale_factor`.

use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MhcModuleOp {
    pub name: String,
    pub scale_factor: f64,
    /// Which half of the mHC layer this op models: `pre`, `post`, or `both`.
    /// Part of the table key — pre and post have distinct latencies.
    pub op: String,
    pub hc_mult: u32,
    pub hidden_size: u32,
    /// Retained as model provenance on the op wire; MHC silicon lookup is
    /// intentionally keyed only by compute shape.
    pub architecture: String,
}

impl MhcModuleOp {
    pub fn new(
        name: impl Into<String>,
        op: impl Into<String>,
        hc_mult: u32,
        hidden_size: u32,
        architecture: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            op: op.into(),
            hc_mult,
            hidden_size,
            architecture: architecture.into(),
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let latency = db
            .mhc
            .query_module(&self.op, num_tokens, self.hc_mult, self.hidden_size)?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}
