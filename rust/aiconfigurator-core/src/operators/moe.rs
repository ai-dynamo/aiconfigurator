// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MoE operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.moe.MoE` SILICON path. The perf-DB
//! layer handles workload-distribution fallback to `"uniform"` and resolves
//! the token curve on the perf_interp v2 engine; this operator supplies the
//! MoE roofline SOL closure the engine's beyond-range util-hold anchors on
//! (Python v2 deleted the op-level overflow estimator — the engine's
//! `k_tail=1`, unclamped util-hold replaces it).
//!
//! Weights accounting (per-expert FFN weights + router) is in the model
//! layer; the operator returns latency only.

use serde::{Deserialize, Serialize};
use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// Gated FFN (SwiGLU) when true; non-gated (Relu²) when false.
    /// Mirrors Python's `MoE._is_gated`. The TRT-LLM small-token
    /// `moe_torch_flow_min_latency` kernel is only valid for gated nvfp4
    /// MoE; non-gated paths (e.g. NemotronH) must skip it.
    pub is_gated: bool,
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
            is_gated: true,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // The roofline SOL the perf-DB engine anchors its beyond-range
        // util-hold on (Python `_resolve_tokens` passes the same closure).
        // Coordinates arriving from the engine are always integral (table
        // keys / the u32 query), so rounding to u32 keeps the integer
        // floor-division parity with Python's `get_sol`.
        let sol = |t: f64| self.sol_latency_ms(db, t.round() as u32);

        // Mirrors Python's MoE._query_moe_table TRT-LLM gate: for nvfp4
        // gated MoE at num_tokens <= 128, probe the
        // `moe_torch_flow_min_latency` grid first and fall back to the
        // default grid on a shape miss. Other backends (vLLM, SGLang) never
        // have `kernel_source` populated, so `low_latency_available()`
        // returns false and this short-circuits.
        let latency = if num_tokens <= 128
            && self.quant_mode == MoeQuantMode::Nvfp4
            && self.is_gated
            && db.moe.low_latency_available()?
        {
            if let Some(ll) = db.moe.query_low_latency(
                num_tokens,
                self.hidden_size,
                self.inter_size,
                self.topk,
                self.num_experts,
                self.moe_tp_size,
                self.moe_ep_size,
                self.quant_mode,
                &self.workload_distribution,
                &sol,
            )? {
                ll
            } else {
                db.moe.query(
                    num_tokens,
                    self.hidden_size,
                    self.inter_size,
                    self.topk,
                    self.num_experts,
                    self.moe_tp_size,
                    self.moe_ep_size,
                    self.quant_mode,
                    &self.workload_distribution,
                    &sol,
                )?
            }
        } else {
            db.moe.query(
                num_tokens,
                self.hidden_size,
                self.inter_size,
                self.topk,
                self.num_experts,
                self.moe_tp_size,
                self.moe_ep_size,
                self.quant_mode,
                &self.workload_distribution,
                &sol,
            )?
        };
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// SOL MoE latency (ms) mirroring Python `MoE._query_moe_table`'s
    /// `get_sol` closure (`operations/moe.py:297`). Passed into the perf-DB
    /// engine query as the util-hold roofline; in-grid resolutions never
    /// consult it (1-axis RAW lerp / exact hit).
    fn sol_latency_ms(&self, db: &PerfDatabase, num_tokens: u32) -> f64 {
        // `num_gemms`: 3 for gated SwiGLU (gate + up + down), 2 for
        // non-gated Relu² (up + down). Matches Python `num_gemms = 3 if
        // is_gated else 2` (`operations/moe.py:115, 239`).
        let num_gemms: u64 = if self.is_gated { 3 } else { 2 };
        let total_tokens = num_tokens as u64 * self.topk as u64;
        let moe_ep = (self.moe_ep_size as u64).max(1);
        let moe_tp = (self.moe_tp_size as u64).max(1);
        let h = self.hidden_size as u64;
        let inter = self.inter_size as u64;
        let ne = self.num_experts as u64;

        let ops = total_tokens * h * inter * num_gemms * 2 / moe_ep / moe_tp;
        let mem_bytes_int = total_tokens / moe_ep * h * 2 // input + output
            + total_tokens / moe_ep * inter * num_gemms / moe_tp // intermediate
            + h * inter * num_gemms / moe_tp
                * std::cmp::min(ne / moe_ep, total_tokens / moe_ep);
        let mem_bytes = (mem_bytes_int as f64) * self.quant_mode.mapping().memory;

        let spec = &db.system_spec;
        // Python uses `system_spec["gpu"]["bfloat16_tc_flops"]` directly
        // (KeyError if missing). Rust exposes it as Option; fall back to 1.0
        // to make the math identity (sol_math → ops, sol_mem dominates)
        // rather than dividing by zero. Every shipped system populates it.
        let tc_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(1.0);
        let sol_math = (ops as f64) / (tc_flops * self.quant_mode.mapping().compute) * 1000.0;
        let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }
}
