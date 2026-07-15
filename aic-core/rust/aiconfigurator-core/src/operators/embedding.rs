// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token embedding operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.embedding`. Embedding latency is
//! a memory-bound per-token lookup: latency = `num_tokens * hidden *
//! dtype_memory / mem_bw` (only the rows that get touched are read, not
//! the whole table). The weight footprint is `vocab * hidden *
//! memory_factor`.

use serde::{Deserialize, Serialize};
use crate::common::enums::GemmQuantMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::attention::mem_op_latency_ms;
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingOp {
    pub name: String,
    pub scale_factor: f64,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub quant_mode: GemmQuantMode,
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank token count is `ceil(num_tokens / seq_split)`. Defaults to 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

impl EmbeddingOp {
    pub fn new(
        name: impl Into<String>,
        vocab_size: u32,
        hidden_size: u32,
        quant_mode: GemmQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            vocab_size,
            hidden_size,
            quant_mode,
            seq_split: 1,
        }
    }

    /// Per-token embedding lookup: each token loads one row of size
    /// `hidden_size * dtype_memory` bytes.
    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let num_tokens = num_tokens.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        let bytes = (num_tokens as f64)
            * (self.hidden_size as f64)
            * self.quant_mode.mapping().memory;
        let latency = mem_op_latency_ms(&db.system_spec, bytes);
        Ok(PerformanceResult::new(latency, Source::Empirical).scaled(self.scale_factor))
    }

    pub fn weights_bytes(&self) -> f64 {
        (self.vocab_size as f64)
            * (self.hidden_size as f64)
            * self.quant_mode.mapping().memory
            * self.scale_factor
    }
}
