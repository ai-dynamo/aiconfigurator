// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Large-EP local expert compute modeled from stock `moe_perf`.
//!
//! This is an approximation, not a measured large-EP kernel query. Uniform
//! routing balance maps one rank to
//! `ceil(x * attention_dp_size / moe_ep_size)` local tokens and
//! `num_experts / moe_ep_size` local experts. The delegated stock [`MoeOp`]
//! retains `topk` and global `num_experts`, and uses `moe_tp=1`,
//! `moe_ep=moe_ep_size`, and the balanced distribution. Stock `moe_perf`
//! interprets that geometry as one rank's local expert shard.
//!
//! `num_slots`, `kernel_source`, `enable_eplb`, and the serialized
//! `workload_distribution` remain only to preserve the schema-v7 positional
//! bincode layout. They are intentionally ignored: EPLB/redundant slots are
//! outside this model.

use serde::{Deserialize, Serialize};

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::moe::MoeOp;
use crate::perf_database::PerfDatabase;

fn default_is_gated() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModeledEpMoeOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub inter_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_ep_size: u32,
    pub quant_mode: MoeQuantMode,
    pub workload_distribution: String,
    pub attention_dp_size: u32,
    pub inference_phase: String,
    #[serde(default)]
    pub num_slots: Option<u32>,
    #[serde(default)]
    pub kernel_source: Option<String>,
    #[serde(default = "default_is_gated")]
    pub is_gated: bool,
    #[serde(default)]
    pub enable_eplb: bool,
}

impl ModeledEpMoeOp {
    pub fn modeled_coordinates(&self, num_tokens: u32) -> Result<(u32, u32), AicError> {
        if !matches!(self.inference_phase.as_str(), "context" | "generation") {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid inference_phase {:?}",
                self.inference_phase
            )));
        }
        if self.moe_ep_size <= 1 || self.num_experts % self.moe_ep_size != 0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "modeled large-EP requires moe_ep_size > 1 and num_experts divisible by it; got experts={}, ep={}",
                self.num_experts, self.moe_ep_size
            )));
        }
        let global = u64::from(num_tokens) * u64::from(self.attention_dp_size.max(1));
        let ep = u64::from(self.moe_ep_size);
        let local_tokens = global.div_ceil(ep).min(u64::from(u32::MAX)) as u32;
        Ok((local_tokens, self.num_experts / self.moe_ep_size))
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let (local_tokens, _local_experts) = self.modeled_coordinates(num_tokens)?;
        let stock = MoeOp {
            name: self.name.clone(),
            scale_factor: self.scale_factor,
            hidden_size: self.hidden_size,
            inter_size: self.inter_size,
            topk: self.topk,
            num_experts: self.num_experts,
            moe_tp_size: 1,
            moe_ep_size: self.moe_ep_size,
            attention_dp_size: 1,
            quant_mode: self.quant_mode,
            workload_distribution: "balanced".into(),
            is_gated: self.is_gated,
            moe_backend: None,
            enable_eplb: false,
            is_context: self.inference_phase == "context",
        };
        let result = stock.query(db, local_tokens)?;
        Ok(PerformanceResult::new(result.latency_ms, Source::Estimated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn op() -> ModeledEpMoeOp {
        ModeledEpMoeOp {
            name: "modeled_ep_moe".into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            inter_size: 2048,
            topk: 8,
            num_experts: 256,
            moe_ep_size: 16,
            quant_mode: MoeQuantMode::Fp8Block,
            workload_distribution: "legacy-value-is-ignored".into(),
            attention_dp_size: 2,
            inference_phase: "context".into(),
            num_slots: Some(512),
            kernel_source: Some("legacy-value-is-ignored".into()),
            is_gated: true,
            enable_eplb: true,
        }
    }

    #[test]
    fn balanced_mapping_globalizes_then_distributes_tokens() {
        // ceil(17 * ADP2 / EP16) = 3, experts/EP = 16.
        assert_eq!(op().modeled_coordinates(17).unwrap(), (3, 16));
    }

    #[test]
    fn mapping_rounds_up_fractional_local_token() {
        let mut modeled = op();
        modeled.attention_dp_size = 1;
        // ceil(17 / 16) = 2.
        assert_eq!(modeled.modeled_coordinates(17).unwrap(), (2, 16));
    }

    #[test]
    fn eplb_and_slots_do_not_change_coordinates() {
        let mut plain = op();
        plain.enable_eplb = false;
        plain.num_slots = None;
        assert_eq!(
            op().modeled_coordinates(17).unwrap(),
            plain.modeled_coordinates(17).unwrap()
        );
    }

    #[test]
    fn stock_moe_query_is_tagged_estimated() {
        let systems_root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems");
        let db = PerfDatabase::load(&systems_root, "h200_sxm", "vllm", "0.19.0").unwrap();
        let mut modeled = op();
        modeled.moe_ep_size = 8;
        modeled.attention_dp_size = 8;
        let result = modeled.query(&db, 128).unwrap();
        assert!(result.latency_ms > 0.0);
        assert_eq!(result.source, Source::Estimated);
    }
}
