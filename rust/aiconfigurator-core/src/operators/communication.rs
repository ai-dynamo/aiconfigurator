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
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank payload is `ceil(num_tokens / seq_split)`. Defaults to 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
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
            seq_split: 1,
        }
    }

    /// Query for `num_tokens` of activation. Python's
    /// `CustomAllReduce.query` computes `size = x * self._h` (element
    /// count, not bytes) and passes it directly to
    /// `query_custom_allreduce`. Mirror that here — the underlying table
    /// is keyed by element count, with dtype implicit in the quant mode.
    /// Node-fan-out capping, beyond-node bandwidth scaling and the GB200
    /// NVL72 -> NCCL reroute all live in the DB-level `_scaled` query
    /// (mirroring Python's `_query_custom_allreduce_table` funnel).
    pub fn query(
        &self,
        db: &PerfDatabase,
        num_tokens: u32,
    ) -> Result<PerformanceResult, AicError> {
        if self.tp_size <= 1 {
            return Ok(PerformanceResult::zero());
        }
        let per_rank_tokens = num_tokens.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        let message_size = (per_rank_tokens as f64) * (self.hidden_size as f64);
        let latency = db.communication.query_custom_allreduce_scaled(
            &db.system_spec,
            self.quant,
            self.tp_size,
            message_size,
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NcclOp {
    pub name: String,
    pub scale_factor: f64,
    /// Elements moved per token (Python's `_num_elements_per_token`). This is a
    /// float, not an integer: the CP KV all-gather sizes it as
    /// `kvcache_bytes_per_token / comm_bytes`, which can be fractional.
    pub hidden_size: f64,
    pub num_gpus: u32,
    pub dtype: CommQuantMode,
    pub operation: String,
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank payload is `ceil(num_tokens / seq_split)`. Defaults to 1.
    /// Note the CP KV all-gather (`context_cp_all_gather`) itself keeps
    /// `seq_split=1` (it moves the full per-token KV), so this is per-op.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

impl NcclOp {
    pub fn new(
        name: impl Into<String>,
        scale_factor: f64,
        hidden_size: f64,
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
            seq_split: 1,
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
        let per_rank_tokens = num_tokens.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        // Python: message_size = ceil(x/seq_split) * num_elements_per_token —
        // kept as a FLOAT (fractional element counts are real: the gemma4 CP
        // KV all-gather sizes per-token elements as kv_bytes / comm_bytes).
        let message_size = (per_rank_tokens as f64) * self.hidden_size;
        // Fan-out capping + beyond-range bandwidth correction live in the
        // DB-level `_scaled` query (mirroring Python's `_query_nccl_table`).
        let latency = db.communication.query_nccl_scaled(
            &db.system_spec,
            self.dtype,
            &self.operation,
            self.num_gpus,
            message_size,
        )?;
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
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size`): the
    /// per-rank payload is `ceil(x / seq_split)`. Defaults to 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

impl P2POp {
    pub fn new(name: impl Into<String>, pp_size: u32, hidden_size: u32) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            pp_size,
            hidden_size,
            seq_split: 1,
        }
    }

    pub fn query(&self, db: &PerfDatabase, x: u32) -> Result<PerformanceResult, AicError> {
        if self.pp_size <= 1 {
            return Ok(PerformanceResult::zero());
        }
        let spec = &db.system_spec;
        let per_rank_tokens = x.div_ceil(self.seq_split.max(1)); // CP: busiest rank
        let bytes = (per_rank_tokens as f64) * (self.hidden_size as f64) * 2.0;
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
