// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MoE dispatch / combine operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.moe.MoEDispatch` plus
//! `TrtLLMWideEPMoEDispatch`. The dispatch operation moves tokens between
//! attention ranks and expert ranks before and after the MoE GEMMs. It has
//! backend-specific paths:
//!
//! - **vLLM**: tokens flow through a custom AllReduce on TP. Approximated
//!   here by `CustomAllReduceOp` on a message size proportional to
//!   `num_tokens × hidden_size × dtype_memory`.
//! - **SGLang DeepEP**: dispatch + combine latencies come from the
//!   `wideep_deepep_normal` / `wideep_deepep_ll` tables (see
//!   `db.wideep.query_deepep_normal/ll`).
//! - **TRT-LLM WideEP**: uses `db.wideep.query_trtllm_alltoall`.
//!
//! All paths route through the corresponding tables; the higher-level
//! model is responsible for choosing the dispatch flavor.

use crate::common::enums::{BackendKind, CommQuantMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::communication::CustomAllReduceOp;
use crate::perf_database::PerfDatabase;

/// MoE dispatch flavor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchFlavor {
    /// vLLM / non-WideEP backends: custom AllReduce on attention TP.
    CustomAllReduce,
    /// SGLang DeepEP normal mode (high-throughput).
    DeepEpNormal,
    /// SGLang DeepEP low-latency mode (decode).
    DeepEpLowLatency,
    /// TRT-LLM WideEP all-to-all.
    TrtllmAlltoall,
}

#[derive(Clone, Debug)]
pub struct MoEDispatchOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
    pub attention_dp_size: u32,
    pub pre_dispatch: bool,
    pub backend: BackendKind,
    pub flavor: DispatchFlavor,
    pub comm_quant: CommQuantMode,
    pub moe_quant: MoeQuantMode,
}

impl MoEDispatchOp {
    pub fn new(
        name: impl Into<String>,
        hidden_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        attention_dp_size: u32,
        pre_dispatch: bool,
        backend: BackendKind,
        flavor: DispatchFlavor,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            hidden_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            attention_dp_size,
            pre_dispatch,
            backend,
            flavor,
            comm_quant: CommQuantMode::Half,
            moe_quant: MoeQuantMode::Bfloat16,
        }
    }

    fn attention_tp_size(&self) -> u32 {
        let total = self.moe_tp_size * self.moe_ep_size;
        (total / self.attention_dp_size.max(1)).max(1)
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let spec: &SystemSpec = &db.system_spec;
        match self.flavor {
            DispatchFlavor::CustomAllReduce => {
                // Approximate via custom_allreduce on attention TP. Message size
                // = num_tokens * hidden_size * comm-element-bytes (half = 2).
                let tp = self.attention_tp_size();
                let ar = CustomAllReduceOp::new(&self.name, self.scale_factor, self.hidden_size, tp);
                ar.query(db, num_tokens)
            }
            DispatchFlavor::DeepEpNormal => {
                let point = db.wideep.query_deepep_normal(
                    spec.node.num_gpus_per_node,
                    self.hidden_size,
                    num_tokens,
                    self.topk,
                    self.num_experts,
                )?;
                let total_us = if self.pre_dispatch {
                    point.dispatch_transmit_us + point.dispatch_notify_us
                } else {
                    point.combine_transmit_us + point.combine_notify_us
                };
                let latency_ms = total_us / 1000.0;
                Ok(PerformanceResult::new(latency_ms, Source::Silicon)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
            DispatchFlavor::DeepEpLowLatency => {
                let point = db.wideep.query_deepep_ll(
                    spec.node.num_gpus_per_node,
                    self.hidden_size,
                    num_tokens,
                    self.topk,
                    self.num_experts,
                )?;
                let latency_ms = if self.pre_dispatch {
                    point.dispatch_avg_t_us
                } else {
                    point.combine_avg_t_us
                } / 1000.0;
                Ok(PerformanceResult::new(latency_ms, Source::Silicon)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
            DispatchFlavor::TrtllmAlltoall => {
                // Port of Python `MoEDispatch.query` trtllm SM100 branch
                // (operations/moe.py). The Python control flow:
                //   if backend_supports_alltoall && attention_dp > 1
                //      && moe_tp == 1 && is_nvl72:    -> trtllm_alltoall table
                //   elif attention_dp > 1:            -> NCCL all_gather/reduce_scatter
                //   elif attention_tp > 1:            -> custom_allreduce (when
                //                                       reduce_results) else 0
                //   else:                             -> 0
                //
                // Selecting the *flavor* up front (as the model builder does)
                // cannot encode this gating — the choice depends on the system's
                // NVLink topology (`num_gpus_per_node`) and on tp/dp shapes that
                // are only known with the system spec in hand. So
                // `DispatchFlavor::TrtllmAlltoall` now means "trtllm SM100
                // dispatch op; the *table* is picked here at query time".
                let num_gpus_per_node = spec.node.num_gpus_per_node;
                let is_nvl72 = num_gpus_per_node >= 72;
                // `moe_backend` is `None` in all current callers (no caller
                // sets a non-default backend); treat as supporting alltoall.
                let backend_supports_alltoall = true;
                let enable_alltoall = backend_supports_alltoall
                    && self.attention_dp_size > 1
                    && self.moe_tp_size == 1
                    && is_nvl72;
                let attention_tp = self.attention_tp_size();

                if enable_alltoall {
                    let latency = db.wideep.query_trtllm_alltoall(
                        num_tokens,
                        self.hidden_size,
                        self.topk,
                        self.num_experts,
                        self.moe_ep_size,
                        self.moe_quant,
                        "uniform",
                    )?;
                    Ok(PerformanceResult::new(latency, Source::Silicon)
                        .clamp_non_negative()
                        .scaled(self.scale_factor))
                } else if self.attention_dp_size > 1 {
                    // Python: query_nccl(half, num_gpus, "all_gather" or
                    // "reduce_scatter", volume * attention_dp_size).
                    // No smoke case exercises this branch today (all smoke
                    // configs use attention_dp_size == 1). The implementation
                    // is intentionally absent rather than added as untested
                    // code; a `PerfDatabase` error will surface and any
                    // future smoke case that hits this path will document
                    // the need.
                    Err(AicError::PerfDatabase(format!(
                        "trtllm MoEDispatch attention_dp_size={}>1 path not yet ported; \
                         add a smoke case to fix.",
                        self.attention_dp_size
                    )))
                } else if attention_tp > 1 {
                    // reduce_results path: Python defaults `_reduce_results`
                    // to True, and the smoke configs do not override it, so
                    // the branch we replicate is the custom_allreduce one.
                    let ar = CustomAllReduceOp::new(
                        &self.name,
                        self.scale_factor,
                        self.hidden_size,
                        attention_tp,
                    );
                    ar.query(db, num_tokens)
                } else {
                    // attn_tp == 1 and attn_dp == 1: no communication needed.
                    Ok(PerformanceResult::new(0.0, Source::Silicon)
                        .clamp_non_negative()
                        .scaled(self.scale_factor))
                }
            }
        }
    }
}
