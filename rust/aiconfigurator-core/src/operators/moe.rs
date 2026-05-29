// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MoE operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.moe.MoE` SILICON path. The perf-DB
//! layer already handles workload-distribution fallback to `"uniform"` and
//! 1-D extrapolation along `num_tokens`. The SOL-anchored overflow
//! estimation that Python applies for `num_tokens > max_recorded_tokens`
//! is a refinement reserved for the model graph layer — at that point the
//! model has the system spec and topology context to apply it consistently
//! across context vs decode.
//!
//! Weights accounting (per-expert FFN weights + router) is in the model
//! layer too; the operator returns latency only.

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug)]
pub struct MoeOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub inter_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
    pub quant_mode: MoeQuantMode,
    pub workload_distribution: String,
}

impl MoeOp {
    pub fn new(
        name: impl Into<String>,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: MoeQuantMode,
        workload_distribution: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant_mode,
            workload_distribution: workload_distribution.into(),
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let latency = db.moe.query(
            num_tokens,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.moe_tp_size,
            self.moe_ep_size,
            self.quant_mode,
            &self.workload_distribution,
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}
