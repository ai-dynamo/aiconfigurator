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
    /// Attention data-parallel size. sglang all-gathers the DP-sharded tokens
    /// before the MoE, so the compute sees `num_tokens * attention_dp_size`
    /// tokens (mirrors Python `MoE.query`: `x = x * attention_dp_size`).
    /// Absent in pre-existing specs -> treated as 1.
    #[serde(default)]
    pub attention_dp_size: u32,
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
            attention_dp_size: 1,
            quant_mode,
            workload_distribution: workload_distribution.into(),
            is_gated: true,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // sglang all-gathers the attention-DP-sharded tokens before the MoE,
        // so the compute processes the full global token count. Mirror Python
        // `MoE.query` (`operations/moe.py`): `x = x * attention_dp_size`. Do
        // this first so the overflow/interpolation logic below keys off the
        // scaled token count.
        let num_tokens = num_tokens.saturating_mul(self.attention_dp_size.max(1));
        // SOL-anchored overflow: when `num_tokens` exceeds the largest
        // recorded token point for this (quant, distribution, topology),
        // Python switches to a utilization-anchored estimate instead of
        // pure linear extrapolation. The perf-DB layer extrapolates from
        // the last two points, which for sparse tables (e.g. int4_wo MoE
        // with only `num_tokens ∈ {1, 2, 4}`) blows up by 100-1000x at
        // query=512. Mirrors Python `MoE._query_moe_table`'s overflow
        // closure (`operations/moe.py:308-359`).
        let max_point = db.moe.max_token_point(
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.moe_tp_size,
            self.moe_ep_size,
            self.quant_mode,
            &self.workload_distribution,
        )?;
        if let Some(max_tok) = max_point {
            if num_tokens > max_tok {
                let last_latency = db.moe.query(
                    max_tok,
                    self.hidden_size,
                    self.inter_size,
                    self.topk,
                    self.num_experts,
                    self.moe_tp_size,
                    self.moe_ep_size,
                    self.quant_mode,
                    &self.workload_distribution,
                )?;
                let sol_last = self.sol_latency_ms(db, max_tok);
                let sol_query = self.sol_latency_ms(db, num_tokens);
                // Clamp MFU to (0, 1] (Python's `util = min(1.0, sol_last /
                // last_latency); util = max(util, 1e-8)`).
                let util = (sol_last / last_latency).min(1.0).max(1e-8);
                let est = sol_query / util;
                return Ok(PerformanceResult::new(est, Source::Silicon)
                    .clamp_non_negative()
                    .scaled(self.scale_factor));
            }
        }

        // In-grid query. Mirrors Python's MoE._query_moe_table TRT-LLM gate:
        // for nvfp4 gated MoE at num_tokens <= 128, probe the
        // `moe_torch_flow_min_latency` grid first and fall back to the
        // default grid on miss. Other backends (vLLM, SGLang) never have
        // `kernel_source` populated, so `low_latency_available()` returns
        // false and this short-circuits.
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
            )?
        };
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// SOL MoE latency (ms) mirroring Python `MoE._query_moe_table`'s
    /// `get_sol` closure (`operations/moe.py:241`). Used only by the
    /// overflow path; in-grid queries hit silicon perf data directly.
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
