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
//! - **TRT-LLM WideEP**: uses the mode-aware [`query_alltoall_table`]
//!   (silicon arm = `db.wideep.query_trtllm_alltoall`).
//!
//! All paths route through the corresponding tables; the higher-level
//! model is responsible for choosing the dispatch flavor.

use serde::{Deserialize, Serialize};

use crate::common::enums::{BackendKind, CommQuantMode, DatabaseMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::communication::{CustomAllReduceOp, NcclOp};
use crate::operators::util_empirical::{self, UtilGrid};
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
    /// Token-count divisor for the DeepEP branches (Python
    /// `MoEDispatch._scale_num_tokens`, applied as
    /// `num_tokens // scale_num_tokens` before the deepep table lookup).
    /// Defaults to 1 (no scaling) for pre-field opspecs.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub scale_num_tokens: u32,
}

fn default_sms() -> u32 {
    12
}

/// TensorRT-LLM WideEP MoE dispatch (NVLink Two-Sided All2All). Mirrors
/// Python `operations.moe.TrtLLMWideEPMoEDispatch`:
///
/// - pre-dispatch op: `alltoall_prepare` + `alltoall_dispatch` (two queries,
///   summed);
/// - post-dispatch op: `alltoall_combine`, or `alltoall_combine_low_precision`
///   when `use_low_precision_combine` (models set it for nvfp4 generation);
/// - every query passes `moe_backend="wideep"`, so the kernel auto-selection
///   resolves `NVLinkTwoSided` on MNNVL (SM >= 100) and the DeepEP variants
///   on Hopper (see `perf_database::wideep::select_alltoall_kernel`).
///
/// `node_num` is never set by the model builders (Python passes None), so the
/// perf-DB default `1 if ep < 4 else ep // 4` applies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrtllmWideEpMoEDispatchOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
    pub attention_dp_size: u32,
    pub pre_dispatch: bool,
    pub quant_mode: MoeQuantMode,
    #[serde(default)]
    pub use_low_precision_combine: bool,
}

impl TrtllmWideEpMoEDispatchOp {
    /// Mode-aware, like every phase-op query in Python's
    /// `TrtLLMWideEPMoEDispatch.query` (it calls
    /// `database.query_trtllm_alltoall` with `database_mode=None`, which
    /// resolves to the view's mode inside `_query_alltoall_table`):
    /// EMPIRICAL estimates `SOL/util`, HYBRID converts a typed silicon miss
    /// into the estimate, SILICON queries the table. `node_num` stays the
    /// perf-DB default (Python's model builders always pass `None`). The
    /// pre-dispatch sum combines sources exactly like Python's
    /// `PerformanceResult.__add__` (same -> same, mismatch -> mixed).
    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let q = |op_name: &str| {
            query_alltoall_table(
                db,
                op_name,
                num_tokens,
                self.hidden_size,
                self.topk,
                self.num_experts,
                self.moe_ep_size,
                self.quant_mode,
                None,
                Some("wideep"),
            )
        };
        let result = if self.pre_dispatch {
            let prepare = q("alltoall_prepare")?;
            let dispatch = q("alltoall_dispatch")?;
            PerformanceResult::new(
                prepare.latency_ms + dispatch.latency_ms,
                prepare.source.combine(dispatch.source),
            )
        } else if self.use_low_precision_combine {
            q("alltoall_combine_low_precision")?
        } else {
            q("alltoall_combine")?
        };
        Ok(result.clamp_non_negative().scaled(self.scale_factor))
    }
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
            scale_num_tokens: 1,
        }
    }

    fn attention_tp_size(&self) -> u32 {
        let total = self.moe_tp_size * self.moe_ep_size;
        (total / self.attention_dp_size.max(1)).max(1)
    }

    /// Number of NODES the MoE group spans — the DeepEP tables' `node_num`
    /// key. Python (`moe.py:1087-1088`): `_node_num = self.num_gpus /
    /// num_gpus_per_node` with `self.num_gpus = moe_ep * moe_tp`; it is NOT
    /// the per-node GPU count. Python's float ratio only hits the int-keyed
    /// table when the division is whole; a fractional ratio walks into an
    /// empty defaultdict slice and the query fails — mirrored here as an
    /// explicit error rather than a floor-divided wrong slice.
    fn deepep_node_num(&self, spec: &SystemSpec) -> Result<u32, AicError> {
        let num_gpus = (self.moe_tp_size * self.moe_ep_size).max(1);
        let per_node = spec.node.num_gpus_per_node;
        if per_node == 0 || num_gpus % per_node != 0 {
            return Err(AicError::PerfDatabase(format!(
                "DeepEP node_num must be whole: num_gpus={num_gpus} / num_gpus_per_node={per_node} (op {})",
                self.name
            )));
        }
        Ok(num_gpus / per_node)
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
                // Python DeepEP branch: `num_tokens = num_tokens //
                // self._scale_num_tokens` before the table lookup.
                let num_tokens = num_tokens / self.scale_num_tokens.max(1);
                // Python `_query_wideep_deepep_normal_table` has NO empirical
                // path: EMPIRICAL mode raises (`NotImplementedError("WideEP
                // deepep normal operation's empirical is not implemented
                // yet")`), and HYBRID goes STRAIGHT to the silicon interp with
                // no fallback (the method never routes through
                // `_query_silicon_or_hybrid`) — so a silicon miss under HYBRID
                // must propagate unchanged, not convert into an estimate.
                if db.database_mode == DatabaseMode::Empirical {
                    return Err(AicError::EmpiricalNotImplemented(
                        "WideEP deepep normal operation's empirical is not implemented yet"
                            .to_string(),
                    ));
                }
                let point = db.wideep.query_deepep_normal(
                    self.deepep_node_num(spec)?,
                    self.hidden_size,
                    num_tokens,
                    self.topk,
                    self.num_experts,
                    // Python passes `sms=self._sms` (kwarg default 12).
                    self.sms,
                )?;
                // Python (`moe.py:1244-1252`) has NO pre/combine branch on
                // the SGLang DeepEP path: BOTH the pre-dispatch op and the
                // combine op call `query_wideep_deepep_normal`, whose loader
                // stores the FULL round trip per point (`lat =
                // dispatch_transmit + dispatch_notify + combine_transmit +
                // combine_notify`, moe.py:2724). A layer's step total is
                // therefore 2x the point — mirror the double-count exactly;
                // do not split the point into halves.
                let total_us = point.dispatch_transmit_us
                    + point.dispatch_notify_us
                    + point.combine_transmit_us
                    + point.combine_notify_us;
                let latency_ms = total_us / 1000.0;
                Ok(PerformanceResult::new(latency_ms, Source::Silicon)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
            DispatchFlavor::DeepEpLowLatency => {
                // Python DeepEP branch: `num_tokens = num_tokens //
                // self._scale_num_tokens` before the table lookup.
                let num_tokens = num_tokens / self.scale_num_tokens.max(1);
                // Same no-empirical rule as DeepEpNormal (Python
                // `_query_wideep_deepep_ll_table`).
                if db.database_mode == DatabaseMode::Empirical {
                    return Err(AicError::EmpiricalNotImplemented(
                        "WideEP deepep ll operation's empirical is not implemented yet".to_string(),
                    ));
                }
                let point = db.wideep.query_deepep_ll(
                    self.deepep_node_num(spec)?,
                    self.hidden_size,
                    num_tokens,
                    self.topk,
                    self.num_experts,
                )?;
                // Same no-split rule as DeepEpNormal: Python's generation
                // branch (`moe.py:1253-1260`) returns the summed LL point
                // (`lat = combine_avg_t_us + dispatch_avg_t_us`, moe.py:2666)
                // for BOTH the pre-dispatch and the combine op.
                let latency_ms = (point.dispatch_avg_t_us + point.combine_avg_t_us) / 1000.0;
                Ok(PerformanceResult::new(latency_ms, Source::Silicon)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
            DispatchFlavor::TrtllmAlltoall => {
                // Full port of Python `MoEDispatch.query`'s trtllm branch
                // (operations/moe.py:1095-1221). Selecting the *flavor* up
                // front (as the model builder does) cannot encode the gating —
                // it depends on the system's SM version and NVLink topology —
                // so `DispatchFlavor::TrtllmAlltoall` means "trtllm dispatch
                // op; the *path* is picked here at query time with the spec
                // in hand". Two regimes, split on `sm_version == 100`
                // EXACTLY like Python (`.get("sm_version", -1) == 100` —
                // gb300 is sm 103 and takes the else-branch):
                //
                //   sm == 100:
                //     if alltoall (dp>1 && moe_tp==1 && NVL72): alltoall table
                //       (pre -> alltoall_dispatch, combine -> alltoall_combine)
                //     elif dp>1: NCCL all_gather((x+sf volumes)*dp) pre /
                //                reduce_scatter(volume*dp) combine —
                //                quant-compressed volumes on the pre side
                //     elif tp>1 && reduce_results: custom_allreduce(num_gpus)
                //       (the NVL72 tp>4 -> NCCL all_reduce reroute lives in
                //        the operator-level `query_custom_allreduce_table`)
                //     else 0
                //   sm != 100 (checks tp FIRST, mirroring Python):
                //     if tp>1: custom_allreduce(num_gpus, volume)
                //     elif dp>1: all_gather(volume*dp) pre /
                //                reduce_scatter(volume*dp) combine
                //     else 0
                //
                // `reduce_results` defaults True in Python and no model
                // overrides it; `moe_backend` is None in all current callers
                // (backend_supports_alltoall = true). Python raises when the
                // alltoall path is taken with quant_mode=None; the Rust op's
                // `moe_quant` is non-optional so that guard is structural.
                let num_gpus = (self.moe_tp_size * self.moe_ep_size).max(1);
                let attention_tp = self.attention_tp_size();
                let pre = self.pre_dispatch;
                let sm_version = spec.gpu.sm_version.map(i64::from).unwrap_or(-1);

                let comm_latency_ms = if sm_version == 100 {
                    let is_nvl72 = spec.node.num_gpus_per_node >= 72;
                    let enable_alltoall =
                        self.attention_dp_size > 1 && self.moe_tp_size == 1 && is_nvl72;
                    // Quantize-aware dispatch volume factors (elements per
                    // hidden element): nvfp4 -> x/4 + sf/32; fp8/fp8_block ->
                    // x/2; others -> full volume (moe.py:1109-1123).
                    let (x_factor, sf_factor) = match self.moe_quant {
                        MoeQuantMode::Nvfp4 => (0.25, 0.25 / 8.0),
                        MoeQuantMode::Fp8 | MoeQuantMode::Fp8Block => (0.5, 0.0),
                        _ => (1.0, 0.0),
                    };
                    if enable_alltoall {
                        let op_name = if pre { "alltoall_dispatch" } else { "alltoall_combine" };
                        // Mode-dispatched (EMPIRICAL/HYBRID estimate on a
                        // typed silicon miss); the silicon arm is the DB-level
                        // `query_trtllm_alltoall`.
                        query_alltoall_table(
                            db,
                            op_name,
                            num_tokens,
                            self.hidden_size,
                            self.topk,
                            self.num_experts,
                            self.moe_ep_size,
                            self.moe_quant,
                            None,
                            None,
                        )?
                        .latency_ms
                    } else if self.attention_dp_size > 1 {
                        if pre {
                            // all_gather((dispatch_x + dispatch_sf) * dp):
                            // fold the quant compression into the per-token
                            // element count so NcclOp's `tokens * hidden`
                            // reproduces Python's message size exactly.
                            let nccl = NcclOp::new(
                                &self.name,
                                1.0,
                                self.hidden_size as f64 * (x_factor + sf_factor),
                                num_gpus,
                                "all_gather",
                            );
                            nccl.query(db, num_tokens * self.attention_dp_size)?.latency_ms
                        } else {
                            let nccl = NcclOp::new(
                                &self.name,
                                1.0,
                                self.hidden_size as f64,
                                num_gpus,
                                "reduce_scatter",
                            );
                            nccl.query(db, num_tokens * self.attention_dp_size)?.latency_ms
                        }
                    } else if attention_tp > 1 {
                        let ar = CustomAllReduceOp::new(&self.name, 1.0, self.hidden_size, num_gpus);
                        ar.query(db, num_tokens)?.latency_ms
                    } else {
                        0.0
                    }

                } else if attention_tp > 1 {
                    let ar = CustomAllReduceOp::new(&self.name, 1.0, self.hidden_size, num_gpus);
                    ar.query(db, num_tokens)?.latency_ms
                } else if self.attention_dp_size > 1 {
                    let op_name = if pre { "all_gather" } else { "reduce_scatter" };
                    let nccl =
                        NcclOp::new(&self.name, 1.0, self.hidden_size as f64, num_gpus, op_name);
                    nccl.query(db, num_tokens * self.attention_dp_size)?.latency_ms
                } else {
                    0.0
                };

                Ok(PerformanceResult::new(comm_latency_ms, Source::Silicon)
                    .clamp_non_negative()
                    .scaled(self.scale_factor))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TRT-LLM alltoall table (Python
// `TrtLLMWideEPMoEDispatch._query_alltoall_table`, also reached from
// `MoEDispatch.query`'s trtllm SM100 branch via
// `database.query_trtllm_alltoall`). Own-shape empirical only — no transfer
// ladder.
// ---------------------------------------------------------------------------

/// Mirror of `TrtLLMWideEPMoEDispatch._normalize_quant_mode_for_table`:
/// `fp8_block` is a behavioral mode that reuses the `fp8` alltoall tables.
fn normalize_alltoall_quant_for_table(quant: MoeQuantMode) -> MoeQuantMode {
    if quant == MoeQuantMode::Fp8Block {
        MoeQuantMode::Fp8
    } else {
        quant
    }
}

/// Mirror of `TrtLLMWideEPMoEDispatch._select_alltoall_kernel` (aligned with
/// TRT-LLM's per-backend `select_alltoall_method_type`). Pure logic: the
/// Python tail's data-availability check only LOGS a warning and returns the
/// preferred kernel anyway, so it has no behavioral counterpart here.
/// (Python's `quant_mode` parameter is unused by the selection — dropped.)
fn select_alltoall_kernel(
    spec: &SystemSpec,
    moe_ep_size: u32,
    topk: u32,
    moe_backend: Option<&str>,
) -> &'static str {
    if let Some(backend) = moe_backend {
        let upper = backend.to_uppercase();
        if upper == "DEEPGEMM" || upper == "CUTE_DSL" {
            return "NotEnabled";
        }
    }
    let sm_version = spec.gpu.sm_version.unwrap_or(0);
    let num_gpus_per_node = spec.node.num_gpus_per_node;
    let is_inter_node = moe_ep_size > num_gpus_per_node;
    let is_wideep = moe_backend.is_some_and(|b| b.to_uppercase() == "WIDEEP");
    // Python approximates supports_mnnvl() as SM >= 100.
    let supports_mnnvl = sm_version >= 100;

    if is_wideep {
        if supports_mnnvl {
            "NVLinkTwoSided"
        } else {
            let deepep_feasible = moe_ep_size > 1 && topk <= 8;
            if deepep_feasible && is_inter_node {
                "DeepEP"
            } else if deepep_feasible {
                "DeepEPLowLatency"
            } else {
                "NotEnabled"
            }
        }
    } else if supports_mnnvl {
        "NVLinkOneSided"
    } else {
        "NotEnabled"
    }
}

/// Alltoall communication SOL (ms), mirroring `_query_alltoall_table.get_sol`:
/// - prepare: lightweight metadata exchange (`topk * 4` bytes per token);
/// - combine: bfloat16 results (2 B/elem), or fp4 (0.5 B/elem) for the
///   low-precision variant;
/// - dispatch: per-rank deduplication at the RAW quant's precision.
/// `remote_ranks = min(topk, num_experts, ep - 1)`; bandwidth is inter-node
/// when the group spans more than one node. Linear in `num_tokens` with zero
/// intercept — coordinates arrive as f64 from the util-grid engine, and the
/// math is float-exact against Python's (no floor division here).
#[allow(clippy::too_many_arguments)]
fn alltoall_sol_ms(
    spec: &SystemSpec,
    op_name: &str,
    quant: MoeQuantMode,
    node_num: u32,
    num_tokens: f64,
    hidden_size: u32,
    topk: u32,
    num_experts: u32,
    moe_ep_size: u32,
) -> f64 {
    let is_inter_node = node_num > 1;
    let bw = if is_inter_node {
        spec.node.inter_node_bw
    } else {
        spec.node.intra_node_bw
    };
    let remote_ranks = topk.min(num_experts).min(moe_ep_size.saturating_sub(1)) as f64;
    let data_bytes = if op_name == "alltoall_prepare" {
        num_tokens * topk as f64 * 4.0 // token routing indices, ~4 bytes each
    } else if op_name.contains("combine") {
        let bytes_per_element = if op_name.contains("low_precision") { 0.5 } else { 2.0 };
        num_tokens * remote_ranks * hidden_size as f64 * bytes_per_element
    } else {
        // dispatch: per-rank deduplication, use quant_mode precision
        num_tokens * remote_ranks * hidden_size as f64 * quant.mapping().memory
    };
    data_bytes / bw * 1000.0
}

/// Verbatim port of `TrtLLMWideEPMoEDispatch._query_alltoall_table` (minus
/// the SOL diagnostic modes, which never reach the compiled engine):
/// normalize the table quant, default `node_num` from `moe_ep_size`, select
/// the kernel, early-return 0 for `NotEnabled`, then dispatch on the
/// database mode — EMPIRICAL always estimates `SOL(query)/util` from the
/// own-slice token grid; HYBRID converts a typed silicon miss into the
/// estimate; SILICON queries the table.
///
/// KNOWN PYTHON DIVERGENCE: Python's HYBRID fallback closure calls
/// `get_empirical_from_sol` without its `kernel_source` argument and raises
/// `TypeError` (a latent bug — the branch has no working oracle). This port
/// implements the intended fallback (kernel threaded through); when the
/// slice has no data anywhere both languages still end in a typed failure
/// (`EmpiricalNotImplemented` here, `TypeError` there).
#[allow(clippy::too_many_arguments)]
fn query_alltoall_table(
    db: &PerfDatabase,
    op_name: &str,
    num_tokens: u32,
    hidden_size: u32,
    topk: u32,
    num_experts: u32,
    moe_ep_size: u32,
    quant: MoeQuantMode,
    node_num: Option<u32>,
    moe_backend: Option<&str>,
) -> Result<PerformanceResult, AicError> {
    let table_quant = normalize_alltoall_quant_for_table(quant);

    // Python: `node_num = 1 if moe_ep_size < 4 else moe_ep_size // 4` when
    // not provided (no Rust caller provides one today, matching
    // `MoEDispatch.query` which never passes node_num).
    let node_num = node_num.unwrap_or(if moe_ep_size < 4 { 1 } else { moe_ep_size / 4 });

    const VALID_OP_NAMES: [&str; 4] = [
        "alltoall_prepare",
        "alltoall_dispatch",
        "alltoall_combine",
        "alltoall_combine_low_precision",
    ];
    if !VALID_OP_NAMES.contains(&op_name) {
        // Python raises ValueError — a programming error, deliberately NOT a
        // missing-data signal (must not trigger HYBRID/fallback handling).
        return Err(AicError::InvalidEngineConfig(format!(
            "Invalid op_name '{op_name}'. Must be one of {VALID_OP_NAMES:?}"
        )));
    }

    let kernel_source = select_alltoall_kernel(&db.system_spec, moe_ep_size, topk, moe_backend);
    if kernel_source == "NotEnabled" {
        // Python returns PerformanceResult(0.0, source="empirical") for every
        // non-SOL mode.
        return Ok(PerformanceResult::new(0.0, Source::Empirical));
    }

    // The DB-level query redoes kernel selection / normalization / the
    // node_num default internally from the same inputs (deterministic), so
    // delegating keeps one silicon source of truth.
    let silicon = || {
        db.wideep.query_trtllm_alltoall(
            &db.system_spec,
            op_name,
            num_tokens,
            hidden_size,
            topk,
            num_experts,
            moe_ep_size,
            quant,
            moe_backend,
        )
    };
    let empirical = || {
        alltoall_empirical(
            db,
            kernel_source,
            op_name,
            quant,
            table_quant,
            node_num,
            num_tokens,
            hidden_size,
            topk,
            num_experts,
            moe_ep_size,
        )
    };
    match db.database_mode {
        DatabaseMode::Empirical => Ok(PerformanceResult::new(empirical()?, Source::Empirical)),
        DatabaseMode::Hybrid => match silicon() {
            Ok(latency) => Ok(PerformanceResult::new(latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => {
                Ok(PerformanceResult::new(empirical()?, Source::Empirical))
            }
            Err(err) => Err(err),
        },
        _ => Ok(PerformanceResult::new(silicon()?, Source::Silicon)),
    }
}

/// `SOL(query)/util` over the own-slice token curve (depth 1, no transfer
/// ladder). Mirrors `_query_alltoall_table::get_empirical_from_sol`: the SOL
/// uses the RAW quant (only the dispatch op consults its memory width), the
/// table slice uses the NORMALIZED quant.
#[allow(clippy::too_many_arguments)]
fn alltoall_empirical(
    db: &PerfDatabase,
    kernel_source: &str,
    op_name: &str,
    quant: MoeQuantMode,
    table_quant: MoeQuantMode,
    node_num: u32,
    num_tokens: u32,
    hidden_size: u32,
    topk: u32,
    num_experts: u32,
    moe_ep_size: u32,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let sol = |c: &[f64]| {
        alltoall_sol_ms(spec, op_name, quant, node_num, c[0], hidden_size, topk, num_experts, moe_ep_size)
    };
    let sol_time = sol(&[num_tokens as f64]);

    // Python keys on ("alltoall", system, backend, version, kernel, op_name,
    // tqm.name, node_num, hidden, topk, experts, ep) + id(node); the cache
    // here is per-database.
    let key = format!(
        "alltoall:{kernel_source}:{op_name}:{}:{node_num}:{hidden_size}:{topk}:{num_experts}:{moe_ep_size}",
        table_quant.name(),
    );
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.wideep.alltoall_slice_points(
            kernel_source,
            op_name,
            table_quant,
            node_num,
            hidden_size,
            topk,
            num_experts,
            moe_ep_size,
        ) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                points.into_iter().map(|(t, lat)| (vec![t as f64], lat)),
                sol,
            )))),
            // Typed coverage miss -> no grid (estimate() raises the
            // empirical miss); schema/load errors propagate.
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let query = [num_tokens as f64];
    let (latency, _) = util_empirical::estimate(sol_time, &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn b200_sglang_db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
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

    /// Write one synthetic DeepEP-normal parquet. Row tuple: `(node_num,
    /// dispatch_sms, num_token, dispatch_transmit_us)`; the other latency
    /// fields are fixed (`dispatch_notify_us = 1.0`, `combine_transmit_us =
    /// 2.0`, `combine_notify_us = 0.0`), so a point's full sum is
    /// `dispatch_transmit_us + 3.0`. Shape fixed at (hidden=7168, topk=8,
    /// experts=256). Mirrors the writer in `perf_database/wideep.rs` tests.
    fn write_deepep_normal_parquet(path: &std::path::Path, rows: &[(i64, i64, i64, f64)]) {
        use parquet::data_type::{DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;
        use std::sync::Arc;

        let schema = "message schema {
            REQUIRED INT64 node_num;
            REQUIRED INT64 hidden_size;
            REQUIRED INT64 num_token;
            REQUIRED INT64 num_topk;
            REQUIRED INT64 num_experts;
            REQUIRED INT64 dispatch_sms;
            REQUIRED DOUBLE dispatch_transmit_us;
            REQUIRED DOUBLE dispatch_notify_us;
            REQUIRED DOUBLE combine_transmit_us;
            REQUIRED DOUBLE combine_notify_us;
        }";
        let schema = Arc::new(parse_message_type(schema).expect("schema must parse"));
        let file = std::fs::File::create(path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");
        let int_cols: [Vec<i64>; 6] = [
            rows.iter().map(|r| r.0).collect(),  // node_num
            rows.iter().map(|_| 7168).collect(), // hidden_size
            rows.iter().map(|r| r.2).collect(),  // num_token
            rows.iter().map(|_| 8).collect(),    // num_topk
            rows.iter().map(|_| 256).collect(),  // num_experts
            rows.iter().map(|r| r.1).collect(),  // dispatch_sms
        ];
        for values in &int_cols {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(values, None, None).expect("write ints");
            col.close().expect("close col");
        }
        let f64_cols: [Vec<f64>; 4] = [
            rows.iter().map(|r| r.3).collect(), // dispatch_transmit_us
            rows.iter().map(|_| 1.0).collect(), // dispatch_notify_us
            rows.iter().map(|_| 2.0).collect(), // combine_transmit_us
            rows.iter().map(|_| 0.0).collect(), // combine_notify_us
        ];
        for values in &f64_cols {
            let mut col = rg.next_column().expect("next col").expect("f64 col");
            col.typed::<DoubleType>().write_batch(values, None, None).expect("write f64");
            col.close().expect("close col");
        }
        rg.close().expect("close row group");
        writer.close().expect("close writer");
    }

    /// Write one synthetic DeepEP-LL parquet. Row tuple: `(node_num,
    /// num_token, dispatch_avg_t_us, combine_avg_t_us)`; shape fixed at
    /// (hidden=7168, topk=8, experts=256).
    fn write_deepep_ll_parquet(path: &std::path::Path, rows: &[(i64, i64, f64, f64)]) {
        use parquet::data_type::{DoubleType, Int64Type};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;
        use std::sync::Arc;

        let schema = "message schema {
            REQUIRED INT64 node_num;
            REQUIRED INT64 hidden_size;
            REQUIRED INT64 num_token;
            REQUIRED INT64 num_topk;
            REQUIRED INT64 num_experts;
            REQUIRED DOUBLE combine_avg_t_us;
            REQUIRED DOUBLE dispatch_avg_t_us;
        }";
        let schema = Arc::new(parse_message_type(schema).expect("schema must parse"));
        let file = std::fs::File::create(path).expect("create parquet");
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .expect("writer");
        let mut rg = writer.next_row_group().expect("row group");
        let int_cols: [Vec<i64>; 5] = [
            rows.iter().map(|r| r.0).collect(),  // node_num
            rows.iter().map(|_| 7168).collect(), // hidden_size
            rows.iter().map(|r| r.1).collect(),  // num_token
            rows.iter().map(|_| 8).collect(),    // num_topk
            rows.iter().map(|_| 256).collect(),  // num_experts
        ];
        for values in &int_cols {
            let mut col = rg.next_column().expect("next col").expect("int col");
            col.typed::<Int64Type>().write_batch(values, None, None).expect("write ints");
            col.close().expect("close col");
        }
        let f64_cols: [Vec<f64>; 2] = [
            rows.iter().map(|r| r.3).collect(), // combine_avg_t_us
            rows.iter().map(|r| r.2).collect(), // dispatch_avg_t_us
        ];
        for values in &f64_cols {
            let mut col = rg.next_column().expect("next col").expect("f64 col");
            col.typed::<DoubleType>().write_batch(values, None, None).expect("write f64");
            col.close().expect("close col");
        }
        rg.close().expect("close row group");
        writer.close().expect("close writer");
    }

    fn deepep_op(moe_ep: u32, pre: bool, flavor: DispatchFlavor) -> MoEDispatchOp {
        let mut op = MoEDispatchOp::new(
            "moe_dispatch",
            7168,
            8,
            256,
            1, // moe_tp
            moe_ep,
            1, // attention_dp
            pre,
            BackendKind::Sglang,
            flavor,
        );
        op.is_context = flavor == DispatchFlavor::DeepEpNormal;
        op.sms = 16;
        op
    }

    /// Issue #1333 item 4.7-1: `node_num` for the DeepEP tables counts
    /// NODES (`num_gpus / num_gpus_per_node`, Python moe.py:1087-1088 with
    /// `num_gpus = moe_tp * moe_ep`), not GPUs per node. The old code
    /// passed `num_gpus_per_node` (8) straight through, so on a
    /// num_gpus=16 / 8-GPUs-per-node shape it read the node_num=8 slice.
    /// Python oracle (num_gpus=16 -> node_num=2.0 -> the node-2 slice):
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk.operations.moe import MoEDispatch
    /// db = PerfDatabase('h100_sxm','sglang','0.5.6.post2',
    ///                   systems_root='src/aiconfigurator_core/systems', database_mode='SILICON')
    /// db._wideep_deepep_normal_data = {
    ///   2: {7168: {8: {256: {16: {64: {'latency': 103.0, 'energy': 0.0}}}}}},
    ///   8: {7168: {8: {256: {16: {64: {'latency': 903.0, 'energy': 0.0}}}}}}}
    /// op = MoEDispatch('d', 1.0, 7168, 8, 256, 1, 16, 1, True,
    ///                  moe_backend='deepep_moe', is_context=True, sms=16)
    /// print(float(op.query(db, x=64)))  # -> 0.103 (node-2 slice, full sum)"
    /// ```
    #[test]
    fn deepep_node_num_counts_nodes_not_gpus_per_node() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_deepep_normal_parquet(
            &tmp.path().join("wideep_deepep_normal_perf.parquet"),
            // node_num=2 point sums to 103us; node_num=8 (what the old code
            // selected on this shape) to 903us.
            &[(2, 16, 64, 100.0), (8, 16, 64, 900.0)],
        );
        let mut db = b200_sglang_db(); // b200_sxm: num_gpus_per_node = 8
        db.tables_mut().wideep = crate::perf_database::wideep::WideEpTable::new(tmp.path().to_path_buf());

        // num_gpus = moe_tp * moe_ep = 16 -> node_num = 16 / 8 = 2.
        let got = deepep_op(16, true, DispatchFlavor::DeepEpNormal)
            .query(&db, 64)
            .expect("query must succeed");
        assert!(
            (got.latency_ms - 0.103).abs() < 1e-12,
            "must read the node_num=2 slice (103us full sum), got {} ms",
            got.latency_ms
        );

        // num_gpus = 4 on an 8-GPU node: Python's fractional node_num (0.5)
        // never hits the int-keyed table; the Rust mirror errors instead of
        // floor-dividing into the node_num=1 slice.
        assert!(deepep_op(4, true, DispatchFlavor::DeepEpNormal).query(&db, 64).is_err());
    }

    /// Issue #1333 item 4.7-2: Python's SGLang DeepEP branch
    /// (moe.py:1244-1260) has NO pre/combine split — the model builds TWO
    /// MoEDispatch ops per MoE layer (pre_dispatch=True and False, e.g.
    /// deepseek.py:270/308) and EACH returns the FULL summed table point
    /// (normal: dispatch_transmit + dispatch_notify + combine_transmit +
    /// combine_notify, moe.py:2724; ll: dispatch_avg + combine_avg,
    /// moe.py:2666), so the per-step per-layer dispatch total is 2x the
    /// point. The old Rust code split the point into pre/combine halves
    /// (step total = 1x). Python oracle (same synthetic-table pattern as
    /// `deepep_node_num_counts_nodes_not_gpus_per_node`, with both a
    /// pre_dispatch=True and a pre_dispatch=False op):
    ///
    /// ```text
    /// mk = lambda pre: MoEDispatch('d', 1.0, 7168, 8, 256, 1, 8, 1, pre,
    ///                              moe_backend='deepep_moe', is_context=True, sms=16)
    /// print(float(mk(True).query(db, x=64)), float(mk(False).query(db, x=64)))
    /// # -> 0.103 0.103   (step total 0.206)
    /// # generation (is_context=False, ll table {'latency': 50.0}):
    /// # -> 0.05 0.05     (step total 0.1)
    /// ```
    #[test]
    fn deepep_pre_and_combine_each_return_full_point_sum() {
        let tmp = tempfile::tempdir().expect("tmpdir");
        // num_gpus = 8 on b200 (8 GPUs/node) -> node_num = 1.
        write_deepep_normal_parquet(
            &tmp.path().join("wideep_deepep_normal_perf.parquet"),
            &[(1, 16, 64, 100.0)], // full sum = 103us
        );
        write_deepep_ll_parquet(
            &tmp.path().join("wideep_deepep_ll_perf.parquet"),
            &[(1, 64, 30.0, 20.0)], // full sum = 50us
        );
        let mut db = b200_sglang_db();
        db.tables_mut().wideep = crate::perf_database::wideep::WideEpTable::new(tmp.path().to_path_buf());

        // Context (DeepEP normal): pre == combine == full point sum.
        let pre = deepep_op(8, true, DispatchFlavor::DeepEpNormal)
            .query(&db, 64)
            .expect("normal pre query");
        let combine = deepep_op(8, false, DispatchFlavor::DeepEpNormal)
            .query(&db, 64)
            .expect("normal combine query");
        assert!((pre.latency_ms - 0.103).abs() < 1e-12, "pre got {}", pre.latency_ms);
        assert!(
            (combine.latency_ms - 0.103).abs() < 1e-12,
            "combine got {}",
            combine.latency_ms
        );
        assert!(
            (pre.latency_ms + combine.latency_ms - 0.206).abs() < 1e-12,
            "python step total is 2x the point (0.206), got {}",
            pre.latency_ms + combine.latency_ms
        );

        // Generation (DeepEP LL): same no-split rule.
        let pre = deepep_op(8, true, DispatchFlavor::DeepEpLowLatency)
            .query(&db, 64)
            .expect("ll pre query");
        let combine = deepep_op(8, false, DispatchFlavor::DeepEpLowLatency)
            .query(&db, 64)
            .expect("ll combine query");
        assert!((pre.latency_ms - 0.05).abs() < 1e-12, "ll pre got {}", pre.latency_ms);
        assert!(
            (combine.latency_ms - 0.05).abs() < 1e-12,
            "ll combine got {}",
            combine.latency_ms
        );
    }

    // -----------------------------------------------------------------
    // HYBRID / EMPIRICAL parity (util-space empirical port).
    // -----------------------------------------------------------------

    use crate::common::enums::TransferPolicy;

    fn gb200_trtllm_db(mode: DatabaseMode) -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "gb200", "trtllm", "1.3.0rc10")
            .expect("db loads")
            .with_mode(mode, TransferPolicy::ALL)
    }

    fn h100_sglang_db(mode: DatabaseMode) -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "h100_sxm", "sglang", "0.5.6.post2")
            .expect("db loads")
            .with_mode(mode, TransferPolicy::ALL)
    }

    /// Shorthand for the collected gb200 alltoall shape (hidden=7168,
    /// topk=8, experts=256).
    fn a2a(
        db: &PerfDatabase,
        op_name: &str,
        num_tokens: u32,
        quant: MoeQuantMode,
        moe_backend: Option<&str>,
    ) -> Result<PerformanceResult, AicError> {
        query_alltoall_table(db, op_name, num_tokens, 7168, 8, 256, 8, quant, None, moe_backend)
    }

    fn assert_oracle(result: &PerformanceResult, expected: f64, source: Source, label: &str) {
        assert!(
            (result.latency_ms - expected).abs() < 1e-9,
            "{label}: expected {expected}, got {}",
            result.latency_ms
        );
        assert_eq!(result.source, source, "{label}: wrong source");
    }

    /// Oracle values generated from the Python reference on the same data
    /// (shared layer pinned OFF so Python reads exactly the primary parquet):
    ///
    /// ```text
    /// db = perf_database.get_database_view("gb200", "trtllm", "1.3.0rc10",
    ///     allow_missing_data=True, database_mode=..., shared_layer=False)
    /// float(TrtLLMWideEPMoEDispatch._query_alltoall_table(db, op_name=...,
    ///     num_tokens=..., hidden_size=7168, topk=8, num_experts=256,
    ///     moe_ep_size=8, quant_mode=..., node_num=None,
    ///     database_mode=..., moe_backend=...))
    /// ```
    ///
    /// `moe_backend=None` selects NVLinkOneSided (gb200 SM100, non-WideEP;
    /// only nvfp4 collected there); `moe_backend="wideep"` selects
    /// NVLinkTwoSided (bfloat16/fp8/nvfp4 + prepare/combine_low_precision).
    /// ep=8 -> node_num = 8 // 4 = 2 (both computed and, in the loader, the
    /// num-nodes-column default `max(1, ep // 4)`). nt=64 is a collected
    /// point (exact hit); nt=333 is an off-grid interior point.
    #[test]
    fn alltoall_empirical_matches_python_oracles() {
        let db = gb200_trtllm_db(DatabaseMode::Empirical);
        let hit = a2a(&db, "alltoall_dispatch", 64, MoeQuantMode::Nvfp4, None).expect("exact hit");
        assert_oracle(&hit, 0.018886399269104005, Source::Empirical, "emp_dispatch_t64");
        let off = a2a(&db, "alltoall_dispatch", 333, MoeQuantMode::Nvfp4, None).expect("off-grid");
        assert_oracle(&off, 0.033548976838374114, Source::Empirical, "emp_dispatch_t333");
        let combine = a2a(&db, "alltoall_combine", 333, MoeQuantMode::Nvfp4, None).expect("combine");
        assert_oracle(&combine, 0.07118654040018774, Source::Empirical, "emp_combine_t333");

        // WideEP kernel + the ops only that path uses.
        let wideep_dispatch =
            a2a(&db, "alltoall_dispatch", 333, MoeQuantMode::Fp8, Some("wideep")).expect("wideep");
        assert_oracle(&wideep_dispatch, 0.04266341407888801, Source::Empirical, "emp_wideep_fp8");
        let prepare =
            a2a(&db, "alltoall_prepare", 333, MoeQuantMode::Fp8, Some("wideep")).expect("prepare");
        assert_oracle(&prepare, 0.015176159059580488, Source::Empirical, "emp_wideep_prepare");
        let combine_lp =
            a2a(&db, "alltoall_combine_low_precision", 333, MoeQuantMode::Nvfp4, Some("wideep"))
                .expect("combine_lp");
        assert_oracle(&combine_lp, 0.06946014693369004, Source::Empirical, "emp_wideep_combine_lp");
    }

    /// HYBRID with data present stays on silicon; the in-range interpolation
    /// differs from the empirical reconstruction at the same points.
    #[test]
    fn alltoall_hybrid_prefers_silicon_when_covered() {
        let db = gb200_trtllm_db(DatabaseMode::Hybrid);
        let hit = a2a(&db, "alltoall_dispatch", 64, MoeQuantMode::Nvfp4, None).expect("exact hit");
        assert_oracle(&hit, 0.018886399269104005, Source::Silicon, "hyb_dispatch_t64");
        let off = a2a(&db, "alltoall_dispatch", 333, MoeQuantMode::Nvfp4, None).expect("off-grid");
        assert_oracle(&off, 0.033547499962151055, Source::Silicon, "hyb_dispatch_t333");
        let combine = a2a(&db, "alltoall_combine", 333, MoeQuantMode::Nvfp4, None).expect("combine");
        assert_oracle(&combine, 0.07116495203226805, Source::Silicon, "hyb_combine_t333");
        let wideep_dispatch =
            a2a(&db, "alltoall_dispatch", 333, MoeQuantMode::Fp8, Some("wideep")).expect("wideep");
        assert_oracle(&wideep_dispatch, 0.042655749106779696, Source::Silicon, "hyb_wideep_fp8");
        let prepare =
            a2a(&db, "alltoall_prepare", 333, MoeQuantMode::Fp8, Some("wideep")).expect("prepare");
        assert_oracle(&prepare, 0.015222300426103175, Source::Silicon, "hyb_wideep_prepare");
        let combine_lp =
            a2a(&db, "alltoall_combine_low_precision", 333, MoeQuantMode::Nvfp4, Some("wideep"))
                .expect("combine_lp");
        assert_oracle(&combine_lp, 0.069468751270324, Source::Silicon, "hyb_wideep_combine_lp");
    }

    /// The standalone WideEP dispatch op must route through the mode-aware
    /// [`query_alltoall_table`] (Python `TrtLLMWideEPMoEDispatch.query`
    /// passes `database_mode=None`, i.e. the view's mode) — a direct
    /// silicon-table call would return silicon values under EMPIRICAL and
    /// hard-error instead of falling back under HYBRID. Expected values are
    /// the per-phase-op Python oracles already pinned in
    /// `alltoall_empirical_matches_python_oracles` /
    /// `alltoall_hybrid_prefers_silicon_when_covered`, composed per phase
    /// (pre-dispatch = prepare + dispatch).
    #[test]
    fn standalone_wideep_dispatch_is_mode_aware() {
        let wideep_op = |pre_dispatch: bool, quant: MoeQuantMode, low_precision: bool| {
            TrtllmWideEpMoEDispatchOp {
                name: "trtllm_wideep_dispatch".to_string(),
                scale_factor: 1.0,
                hidden_size: 7168,
                topk: 8,
                num_experts: 256,
                moe_tp_size: 1,
                moe_ep_size: 8,
                attention_dp_size: 8,
                pre_dispatch,
                quant_mode: quant,
                use_low_precision_combine: low_precision,
            }
        };

        let emp = gb200_trtllm_db(DatabaseMode::Empirical);
        let pre = wideep_op(true, MoeQuantMode::Fp8, false).query(&emp, 333).expect("emp pre");
        assert_oracle(
            &pre,
            0.015176159059580488 + 0.04266341407888801, // prepare + dispatch
            Source::Empirical,
            "standalone_emp_pre",
        );
        let combine_lp =
            wideep_op(false, MoeQuantMode::Nvfp4, true).query(&emp, 333).expect("emp combine_lp");
        assert_oracle(&combine_lp, 0.06946014693369004, Source::Empirical, "standalone_emp_combine_lp");

        // HYBRID with data present stays on silicon (same values as the
        // covered-slice oracles above).
        let hyb = gb200_trtllm_db(DatabaseMode::Hybrid);
        let pre = wideep_op(true, MoeQuantMode::Fp8, false).query(&hyb, 333).expect("hyb pre");
        assert_oracle(
            &pre,
            0.015222300426103175 + 0.042655749106779696,
            Source::Silicon,
            "standalone_hyb_pre",
        );
    }

    /// fp8 is uncollected under NVLinkOneSided: EMPIRICAL surfaces the typed
    /// empirical miss (own-shape only — no transfer ladder), and fp8_block
    /// normalizes onto the same missing fp8 slice. Under HYBRID, Python's
    /// fallback closure raises `TypeError` (it drops the `kernel_source`
    /// argument — a latent bug, see `query_alltoall_table`); the Rust port
    /// runs the intended fallback, which finds no data and surfaces the same
    /// typed empirical miss.
    #[test]
    fn alltoall_missing_slice_is_typed_empirical_miss() {
        let emp = gb200_trtllm_db(DatabaseMode::Empirical);
        for quant in [MoeQuantMode::Fp8, MoeQuantMode::Fp8Block] {
            let result = a2a(&emp, "alltoall_dispatch", 333, quant, None);
            assert!(
                matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
                "EMPIRICAL {quant:?} must be a typed empirical miss, got {result:?}"
            );
        }
        let hyb = gb200_trtllm_db(DatabaseMode::Hybrid);
        let result = a2a(&hyb, "alltoall_dispatch", 333, MoeQuantMode::Fp8, None);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "HYBRID fallback on a data-less slice must be a typed empirical miss, got {result:?}"
        );
    }

    /// `_select_alltoall_kernel` mirror + the NotEnabled early return
    /// (0.0, source "empirical", regardless of mode — before any table I/O).
    #[test]
    fn alltoall_kernel_selection_matches_python() {
        let db = gb200_trtllm_db(DatabaseMode::Hybrid);
        let spec = &db.system_spec; // SM100, 4 GPUs per node
        assert_eq!(select_alltoall_kernel(spec, 8, 8, None), "NVLinkOneSided");
        assert_eq!(select_alltoall_kernel(spec, 8, 8, Some("wideep")), "NVLinkTwoSided");
        assert_eq!(select_alltoall_kernel(spec, 8, 8, Some("DEEPGEMM")), "NotEnabled");
        assert_eq!(select_alltoall_kernel(spec, 8, 8, Some("cute_dsl")), "NotEnabled");

        // Pre-Blackwell WideEP: DeepEP when feasible, split on inter-node.
        let mut hopper = db.system_spec.clone();
        hopper.gpu.sm_version = Some(90);
        assert_eq!(select_alltoall_kernel(&hopper, 8, 8, Some("wideep")), "DeepEP");
        assert_eq!(select_alltoall_kernel(&hopper, 4, 8, Some("wideep")), "DeepEPLowLatency");
        assert_eq!(select_alltoall_kernel(&hopper, 8, 16, Some("wideep")), "NotEnabled");
        assert_eq!(select_alltoall_kernel(&hopper, 8, 8, None), "NotEnabled");

        let zero = a2a(&db, "alltoall_dispatch", 333, MoeQuantMode::Nvfp4, Some("deepgemm"))
            .expect("NotEnabled early return");
        assert_oracle(&zero, 0.0, Source::Empirical, "not_enabled_zero");
    }

    /// Python `_query_wideep_deepep_{ll,normal}_table` raise
    /// `NotImplementedError` in EMPIRICAL mode — mirrored as the typed
    /// `EmpiricalNotImplemented` (the gate fires before any table I/O, so
    /// any sglang database works).
    #[test]
    fn deepep_empirical_mode_is_typed_not_implemented() {
        let db = b200_sglang_db().with_mode(DatabaseMode::Empirical, TransferPolicy::ALL);
        for flavor in [DispatchFlavor::DeepEpNormal, DispatchFlavor::DeepEpLowLatency] {
            let result = deepep_op(8, true, flavor).query(&db, 64);
            assert!(
                matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
                "{flavor:?} EMPIRICAL must be a typed empirical miss, got {result:?}"
            );
        }
    }

    /// Python's deepep tables have NO hybrid fallback (the `else:` branch
    /// serves both SILICON and HYBRID and never routes through
    /// `_query_silicon_or_hybrid`), so HYBRID answers the silicon interp
    /// value unchanged. Oracles from the shipped h100 data:
    ///
    /// ```text
    /// float(MoEDispatch._query_wideep_deepep_ll_table(db, node_num=1,
    ///     num_tokens=20, num_experts=256, topk=8, hidden_size=7168,
    ///     database_mode=common.DatabaseMode.HYBRID))       # -> 0.03853445
    /// float(MoEDispatch._query_wideep_deepep_normal_table(db, node_num=2,
    ///     num_tokens=64, num_experts=256, topk=8, hidden_size=7168, sms=20,
    ///     database_mode=common.DatabaseMode.HYBRID))       # -> 0.20963
    /// ```
    #[test]
    fn deepep_hybrid_equals_silicon_interp() {
        let db = h100_sglang_db(DatabaseMode::Hybrid);
        // moe_ep=8 on h100 (8 GPUs/node) -> node_num = 1.
        let ll = deepep_op(8, false, DispatchFlavor::DeepEpLowLatency)
            .query(&db, 20)
            .expect("ll hybrid query");
        assert_oracle(&ll, 0.03853445, Source::Silicon, "hyb_deepep_ll_t20");
        // moe_ep=16 -> node_num = 2; sms=20 with node_num != 1 resolves the
        // 2-axis (sms, tokens) grid.
        let mut normal = deepep_op(16, true, DispatchFlavor::DeepEpNormal);
        normal.sms = 20;
        let got = normal.query(&db, 64).expect("normal hybrid query");
        assert_oracle(&got, 0.20963, Source::Silicon, "hyb_deepep_normal_t64");
    }
}
