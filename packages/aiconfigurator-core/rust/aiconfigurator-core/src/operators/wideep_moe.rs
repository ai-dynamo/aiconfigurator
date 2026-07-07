// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TensorRT-LLM WideEP MoE compute operator.
//!
//! Apple-to-apple port of `aiconfigurator_core.sdk.operations.moe.TrtLLMWideEPMoE`.
//! Pure-compute kernel timing (no All2All). The dispatch / combine cost
//! belongs to `MoEDispatchOp` (with the TrtllmAlltoall or DeepEP flavor)
//! or the `wideep` table — depending on which path the model variant
//! exercises.
//!
//! EPLB modes:
//! - EPLB off: `workload_distribution` without `_eplb` suffix,
//!   `num_slots == num_experts`.
//! - EPLB on: `workload_distribution` with `_eplb` suffix,
//!   `num_slots == num_experts`.
//! - EPLB redundant: `workload_distribution` with `_eplb` suffix,
//!   `num_slots > num_experts`.
//!
//! Mirrors Python: `query` multiplies `num_tokens` by `attention_dp_size`
//! before the lookup (the perf table is collected per-rank but the
//! op-level input is per-attention-DP-rank).

use serde::{Deserialize, Serialize};
use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WideEpMoeOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub inter_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
    pub attention_dp_size: u32,
    pub quant_mode: MoeQuantMode,
    pub workload_distribution: String,
    /// EPLB slots; defaults to `num_experts` (no EPLB redundancy).
    pub num_slots: u32,
    /// WideEP MoE compute kernel: `"moe_torch_flow"` (Cutlass; SM<100
    /// default) or `"deepgemm"` (SM>=100 with fp8_block).
    pub kernel_source: String,
}

impl WideEpMoeOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        attention_dp_size: u32,
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
            attention_dp_size,
            quant_mode,
            workload_distribution: workload_distribution.into(),
            num_slots: num_experts,
            kernel_source: "moe_torch_flow".to_string(),
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        num_tokens: u32,
    ) -> Result<PerformanceResult, AicError> {
        // Python: `x = num_tokens * self._attention_dp_size`.
        let scaled = num_tokens.saturating_mul(self.attention_dp_size.max(1));
        let latency = db.wideep_moe.query_compute(
            scaled,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.num_slots,
            self.moe_tp_size,
            self.moe_ep_size,
            self.quant_mode,
            &self.workload_distribution,
            &self.kernel_source,
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}
