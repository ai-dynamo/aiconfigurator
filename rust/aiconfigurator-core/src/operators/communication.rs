// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Communication operators: custom allreduce, NCCL collectives, P2P.
//!
//! Mirrors `aiconfigurator.sdk.operations.communication.{CustomAllReduce,
//! NCCL, P2P}` SILICON paths. This is where the topology-aware scaling
//! lives:
//!
//! - `CustomAllReduceOp`: caps `tp_size` to `num_gpus_per_node` before
//!   the table lookup, then scales by `(tp-1)/tp * (per_node)/(per_node-1)
//!   * intra_bw/p2p_bw` when the actual fan-out exceeds the node.
//! - `NcclOp`: caps `num_gpus` to the table's max recorded fan-out, then
//!   scales by `(num_gpus-1)/num_gpus * max/(max-1) * max_bw/req_bw`.
//! - `P2POp`: pure analytic formula — `(bytes / inter_node_bw +
//!   p2p_latency) * 1000`. No CSV.

use serde::{Deserialize, Serialize};
use crate::common::enums::CommQuantMode;
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomAllReduceOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub tp_size: u32,
    pub quant: CommQuantMode,
}

impl CustomAllReduceOp {
    pub fn new(
        name: impl Into<String>,
        scale_factor: f64,
        hidden_size: u32,
        tp_size: u32,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor,
            hidden_size,
            tp_size,
            quant: CommQuantMode::Half,
        }
    }

    /// Query for `num_tokens` of activation. Python's
    /// `CustomAllReduce.query` computes `size = x * self._h` (element
    /// count, not bytes) and passes it directly to
    /// `query_custom_allreduce`. Mirror that here — the underlying table
    /// is keyed by element count, with dtype implicit in the quant mode.
    pub fn query(
        &self,
        db: &PerfDatabase,
        num_tokens: u32,
    ) -> Result<PerformanceResult, AicError> {
        if self.tp_size <= 1 {
            return Ok(PerformanceResult::zero());
        }
        let message_size = (num_tokens as u64) * (self.hidden_size as u64);
        let spec = &db.system_spec;
        let per_node = spec.node.num_gpus_per_node;
        let effective_tp = self.tp_size.min(per_node);
        let mut latency =
            db.communication
                .query_custom_allreduce(self.quant, effective_tp, message_size)?;
        if self.tp_size > per_node {
            let base_bw = p2p_bandwidth(spec, per_node);
            let target_bw = p2p_bandwidth(spec, self.tp_size);
            let f_tp = self.tp_size as f64;
            let f_pn = per_node as f64;
            let scale = (f_tp - 1.0) / f_tp * f_pn / (f_pn - 1.0).max(1.0) * base_bw / target_bw;
            latency *= scale;
        }
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NcclOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub num_gpus: u32,
    pub dtype: CommQuantMode,
    pub operation: String,
}

impl NcclOp {
    pub fn new(
        name: impl Into<String>,
        scale_factor: f64,
        hidden_size: u32,
        num_gpus: u32,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor,
            hidden_size,
            num_gpus,
            dtype: CommQuantMode::Half,
            operation: operation.into(),
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        num_tokens: u32,
    ) -> Result<PerformanceResult, AicError> {
        if self.num_gpus <= 1 {
            return Ok(PerformanceResult::zero());
        }
        let message_size = (num_tokens as u64) * (self.hidden_size as u64);
        let max_recorded = db
            .communication
            .nccl_max_num_gpus(self.dtype, &self.operation)?
            .unwrap_or(self.num_gpus);
        let effective = self.num_gpus.min(max_recorded);
        let mut latency = db
            .communication
            .query_nccl(self.dtype, &self.operation, effective, message_size)?;
        if self.num_gpus > max_recorded {
            let spec = &db.system_spec;
            let max_bw = p2p_bandwidth(spec, max_recorded);
            let req_bw = p2p_bandwidth(spec, self.num_gpus);
            let f_n = self.num_gpus as f64;
            let f_m = max_recorded as f64;
            let scale =
                (f_n - 1.0) / f_n * f_m / (f_m - 1.0).max(1.0) * max_bw / req_bw;
            latency *= scale;
        }
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

/// Pure analytic P2P latency — no CSV.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct P2POp {
    pub name: String,
    pub scale_factor: f64,
    pub pp_size: u32,
    pub hidden_size: u32,
}

impl P2POp {
    pub fn new(name: impl Into<String>, pp_size: u32, hidden_size: u32) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            pp_size,
            hidden_size,
        }
    }

    pub fn query(&self, db: &PerfDatabase, x: u32) -> Result<PerformanceResult, AicError> {
        if self.pp_size <= 1 {
            return Ok(PerformanceResult::zero());
        }
        let spec = &db.system_spec;
        let bytes = (x as f64) * (self.hidden_size as f64) * 2.0;
        let inter_bw = spec.node.inter_node_bw.max(1.0);
        let latency = (bytes / inter_bw + spec.node.p2p_latency) * 1000.0;
        Ok(PerformanceResult::new(latency, Source::Empirical)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

fn p2p_bandwidth(spec: &SystemSpec, num_gpus: u32) -> f64 {
    spec.get_p2p_bandwidth(num_gpus)
}
