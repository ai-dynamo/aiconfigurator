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

use serde::{Deserialize, Serialize};

use crate::common::enums::{BackendKind, CommQuantMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::communication::{CustomAllReduceOp, NcclOp};
use crate::perf_database::PerfDatabase;

/// MoE dispatch flavor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// Attention-side context-parallel factor (Python's `_attn_cp_size`,
    /// = `cp_size`). Under CP (sglang, prefill) the pre-dispatch all-gathers
    /// / post-dispatch reduce-scatters the CP-sharded tokens. Defaults to 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub attn_cp_size: u32,
    /// Whether this is a context (prefill) dispatch. CP dispatch comm only runs
    /// in prefill; decode replicates attention across CP ranks (no comm).
    #[serde(default)]
    pub is_context: bool,
    /// DeepEP-normal dispatch SM count (Python `MoEDispatch._sms =
    /// kwargs.get("sms", 12)`), forwarded to the sms-keyed deepep-normal
    /// table with nearest-snap semantics. Default 12 = Python's kwarg
    /// default, so old opspecs keep the same query point.
    #[serde(default = "default_sms")]
    pub sms: u32,
}

fn default_sms() -> u32 {
    12
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
            attn_cp_size: 1,
            is_context: false,
            sms: default_sms(),
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
                // Backend-aware port of Python `MoEDispatch.query` for vLLM and
                // SGLang non-DeepEP paths. Both backends pass through this
                // flavor (set by `models/moe.rs::dispatch_flavor`) but compute
                // dispatch latency very differently when attention_dp > 1.
                //
                // Python (`operations/moe.py`):
                //  * vllm (:1003-1020):
                //      comm = 0
                //      if attn_tp > 1: comm += custom_allreduce(num_gpus, volume)
                //      if attn_dp > 1: comm += nccl(num_gpus, "all_gather" if pre
                //                                  else "reduce_scatter", volume * dp)
                //      (both terms can add; Python asserts moe_tp==1 or moe_ep==1)
                //  * sglang non-deepep pre_dispatch (:1043-1071):
                //      if combined_tp_dp: nccl(attn_tp, "reduce_scatter", volume)
                //                       + nccl(num_gpus, "all_gather", volume*dp)
                //      elif tp > 1:       custom_allreduce(num_gpus, volume)
                //      elif dp > 1:       nccl(num_gpus, "all_gather", volume*dp)
                //      else:              0
                //  * sglang non-deepep combine (:1072-1098): mirrors pre but swaps
                //    reduce_scatter <-> all_gather and inverts the combined order.
                //
                // `num_gpus = moe_tp * moe_ep`; `attn_tp = num_gpus / attn_dp`;
                // `volume = num_tokens * hidden_size` (element count, half-precision).
                // Rust mirrors element-count semantics by passing
                // `num_tokens * attn_dp` to the NCCL sub-op so its internal
                // `message_size = num_tokens * dp * hidden_size = volume * dp`.
                //
                // Sub-ops are constructed with `scale_factor=1.0`; the outer
                // op's `scale_factor` (e.g. layer count) is applied once at the
                // end via `.scaled(self.scale_factor)`.
                let num_gpus = (self.moe_tp_size * self.moe_ep_size).max(1);
                let attn_tp = self.attention_tp_size();
                let attn_dp = self.attention_dp_size.max(1);
                let pre = self.pre_dispatch;

                let comm_latency_ms = match self.backend {
                    BackendKind::Vllm => {
                        let mut total = 0.0;
                        if attn_tp > 1 {
                            let ar = CustomAllReduceOp::new(
                                &self.name,
                                1.0,
                                self.hidden_size,
                                num_gpus,
                            );
                            total += ar.query(db, num_tokens)?.latency_ms;
                        }
                        if attn_dp > 1 {
                            let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                            let nccl = NcclOp::new(
                                &self.name,
                                1.0,
                                self.hidden_size as f64,
                                num_gpus,
                                op_name,
                            );
                            total += nccl.query(db, num_tokens * attn_dp)?.latency_ms;
                        }
                        total
                    }
                    BackendKind::Sglang => {
                        let combined_tp_dp = attn_tp > 1 && attn_dp > 1;
                        if combined_tp_dp {
                            // Two NCCL terms; order/op differs between pre and combine.
                            let (op1, gpus1, tokens1, op2, gpus2, tokens2) = if pre {
                                (
                                    "reduce_scatter",
                                    attn_tp,
                                    num_tokens,
                                    "all_gather",
                                    num_gpus,
                                    num_tokens * attn_dp,
                                )
                            } else {
                                (
                                    "reduce_scatter",
                                    num_gpus,
                                    num_tokens * attn_dp,
                                    "all_gather",
                                    attn_tp,
                                    num_tokens,
                                )
                            };
                            let n1 = NcclOp::new(&self.name, 1.0, self.hidden_size as f64, gpus1, op1);
                            let n2 = NcclOp::new(&self.name, 1.0, self.hidden_size as f64, gpus2, op2);
                            n1.query(db, tokens1)?.latency_ms
                                + n2.query(db, tokens2)?.latency_ms
                        } else if self.attn_cp_size > 1 {
                            // Context parallelism (Python moe.py:1279-1290 pre,
                            // :1318-1330 combine); volume = num_tokens * hidden:
                            //  * prefill: tokens are CP-sharded; all_gather (pre)
                            //    to assemble the full token set / reduce_scatter
                            //    (combine) back.
                            //  * decode pre: CP does not run (attention replicated
                            //    across CP ranks, every rank already holds all
                            //    tokens); the expert selection is local -> no comm.
                            //  * decode combine: each rank computed its owned
                            //    experts' partial outputs for all (replicated)
                            //    tokens; combine into the full per-token sum ->
                            //    custom_allreduce(half, num_gpus, volume).
                            if self.is_context {
                                let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                                let nccl =
                                    NcclOp::new(&self.name, 1.0, self.hidden_size as f64, num_gpus, op_name);
                                nccl.query(db, num_tokens)?.latency_ms
                            } else if pre {
                                0.0
                            } else {
                                let ar = CustomAllReduceOp::new(
                                    &self.name,
                                    1.0,
                                    self.hidden_size,
                                    num_gpus,
                                );
                                ar.query(db, num_tokens)?.latency_ms
                            }
                        } else if attn_tp > 1 {
                            let ar = CustomAllReduceOp::new(
                                &self.name,
                                1.0,
                                self.hidden_size,
                                num_gpus,
                            );
                            ar.query(db, num_tokens)?.latency_ms
                        } else if attn_dp > 1 {
                            let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                            let nccl = NcclOp::new(
                                &self.name,
                                1.0,
                                self.hidden_size as f64,
                                num_gpus,
                                op_name,
                            );
                            nccl.query(db, num_tokens * attn_dp)?.latency_ms
                        } else {
                            0.0
                        }
                    }
                    BackendKind::Trtllm => {
                        // Trtllm should use DispatchFlavor::TrtllmAlltoall, not
                        // CustomAllReduce. Safety fallback: replicate the pre-fix
                        // single-term behavior (custom_allreduce on attn_tp) so
                        // downstream callers don't panic if a model mis-routes.
                        let ar = CustomAllReduceOp::new(
                            &self.name,
                            1.0,
                            self.hidden_size,
                            attn_tp,
                        );
                        ar.query(db, num_tokens)?.latency_ms
                    }
                };

                Ok(PerformanceResult::new(comm_latency_ms, Source::Silicon)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
            DispatchFlavor::DeepEpNormal => {
                let point = db.wideep.query_deepep_normal(
                    spec.node.num_gpus_per_node,
                    self.hidden_size,
                    num_tokens,
                    self.topk,
                    self.num_experts,
                    // Python passes `sms=self._sms` (kwarg default 12).
                    self.sms,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn b200_sglang_db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator/systems");
        PerfDatabase::load(&root, "b200_sxm", "sglang", "0.5.10").expect("db loads")
    }

    fn cp_dispatch(pre_dispatch: bool, is_context: bool) -> MoEDispatchOp {
        let mut op = MoEDispatchOp::new(
            "moe_dispatch",
            7168,
            8,
            256,
            1, // moe_tp
            8, // moe_ep
            1, // attention_dp
            pre_dispatch,
            BackendKind::Sglang,
            DispatchFlavor::CustomAllReduce,
        );
        op.attn_cp_size = 8;
        op.is_context = is_context;
        op
    }

    /// Decode combine under CP mirrors Python moe.py:1324-1330: each rank holds
    /// its owned experts' partial outputs for all (replicated) tokens, combined
    /// via `custom_allreduce(half, num_gpus, num_tokens * hidden)`. It must NOT
    /// be zero — only the decode PRE-dispatch branch (moe.py:1286-1290) is a
    /// local selection with no comm.
    #[test]
    fn cp_decode_combine_is_custom_allreduce_not_zero() {
        let db = b200_sglang_db();
        let num_tokens = 64;

        let combine = cp_dispatch(false, false)
            .query(&db, num_tokens)
            .expect("decode combine query");
        let reference = CustomAllReduceOp::new("moe_dispatch", 1.0, 7168, 8)
            .query(&db, num_tokens)
            .expect("allreduce reference query");
        assert!(
            combine.latency_ms > 0.0,
            "decode combine under CP must not be zeroed, got {}",
            combine.latency_ms
        );
        assert!(
            (combine.latency_ms - reference.latency_ms).abs() < 1e-12,
            "decode combine ({}) must equal custom_allreduce(num_gpus=8, volume=64*7168) ({})",
            combine.latency_ms,
            reference.latency_ms
        );

        // Decode pre-dispatch stays local (moe.py:1286-1290) — still zero.
        let pre = cp_dispatch(true, false)
            .query(&db, num_tokens)
            .expect("decode pre query");
        assert_eq!(pre.latency_ms, 0.0, "decode pre-dispatch under CP is local");
    }
}
