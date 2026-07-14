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
    /// Attention data-parallel size. With attention-dp, every dp rank's
    /// tokens all-gather into the SHARED expert pool, so the MoE compute op
    /// sees `num_tokens * attention_dp_size` tokens (mirrors Python
    /// `MoE.query`: `x = x * attention_dp_size`, operations/moe.py).
    /// Dropping the multiplier under-predicted MoE latency ~4.7x on dp=8
    /// DeepSeek configs. Absent in pre-existing specs -> 0 -> treated as 1
    /// at the query site.
    #[serde(default)]
    pub attention_dp_size: u32,
    pub quant_mode: MoeQuantMode,
    pub workload_distribution: String,
    /// Gated FFN (SwiGLU) when true; non-gated (Relu²) when false.
    /// Mirrors Python's `MoE._is_gated`. The TRT-LLM small-token
    /// `moe_torch_flow_min_latency` kernel is only valid for gated nvfp4
    /// MoE; non-gated paths (e.g. NemotronH) must skip it.
    pub is_gated: bool,
    /// SGLang MoE backend (Python `MoE._moe_backend`). `Some("deepep_moe")`
    /// routes the compute lookup to the wideep context/generation MoE tables
    /// instead of `moe_perf` (operations/moe.py sglang branch). Absent in
    /// pre-existing specs -> None -> the regular table.
    #[serde(default)]
    pub moe_backend: Option<String>,
    /// EPLB enabled (Python `MoE._enable_eplb`). On the sglang branch the
    /// PREFILL token count is corrected to `int(num_tokens * 0.8)`
    /// (operations/moe.py: expert-parallel load balancing evens the
    /// per-expert token distribution).
    #[serde(default)]
    pub enable_eplb: bool,
    /// Context (prefill) op — selects the wideep CONTEXT MoE table under
    /// deepep and gates the EPLB prefill correction (Python `MoE._is_context`).
    #[serde(default)]
    pub is_context: bool,
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
            moe_backend: None,
            enable_eplb: false,
            is_context: false,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // Attention-dp scales up the total input tokens (all dp ranks
        // all-gather into one shared expert pool) -- mirrors Python
        // `MoE.query` (`x = x * attention_dp_size`). Applied exactly once,
        // before the perf-DB resolution keys off the token count.
        let num_tokens = num_tokens.saturating_mul(self.attention_dp_size.max(1));
        let is_sglang = db.backend == "sglang";
        // SGLang EPLB prefill correction (Python operations/moe.py:
        // `num_tokens_corrected = int(num_tokens * 0.8) if enable_eplb and
        // is_context else num_tokens`, sglang branch only).
        let num_tokens = if is_sglang && self.enable_eplb && self.is_context {
            (num_tokens as f64 * 0.8) as u32
        } else {
            num_tokens
        };
        // The roofline SOL the perf-DB engine anchors its beyond-range
        // util-hold on (Python `_resolve_tokens` passes the same closure).
        // Coordinates arriving from the engine are always integral (table
        // keys / the u32 query), so rounding to u32 keeps the integer
        // floor-division parity with Python's `get_sol`. This replaces the
        // deleted op-level SOL-anchored overflow estimator (the engine's
        // `k_tail=1` util-hold handles beyond-range queries).
        let sol = |t: f64| self.sol_latency_ms(db, t.round() as u32);

        // SGLang DeepEP (wideep) routes MoE compute to the wideep
        // context/generation tables (Python operations/moe.py:
        // `if moe_backend == "deepep_moe": moe_data = _wideep_*_moe_data`),
        // resolved through the SAME `_resolve_tokens` semantics (singleton
        // guard + MoE-roofline util-hold, threaded via `sol`).
        if is_sglang && self.moe_backend.as_deref() == Some("deepep_moe") {
            let latency = if self.is_context {
                db.wideep.query_context_moe(
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
            } else {
                db.wideep.query_generation_moe(
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
            return Ok(PerformanceResult::new(latency, Source::Silicon)
                .clamp_non_negative()
                .scaled(self.scale_factor));
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn b200_vllm_db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("src/aiconfigurator/systems");
        PerfDatabase::load(&root, "b200_sxm", "vllm", "0.19.0").expect("db loads")
    }

    fn op(attention_dp_size: u32) -> MoeOp {
        MoeOp {
            name: "moe".into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            inter_size: 2048,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 8,
            quant_mode: MoeQuantMode::Fp8Block,
            workload_distribution: "power_law_1.2".into(),
            attention_dp_size,
            is_gated: true,
            moe_backend: None,
            enable_eplb: false,
            is_context: false,
        }
    }

    /// With attention-dp, all dp ranks' tokens funnel into the shared expert
    /// pool: query(dp=4, t) must equal query(dp=1, 4t). Dropping the
    /// multiplier under-predicted MoE latency ~4.7x on dp=8 DeepSeek configs
    /// (python/rust engine-step divergence, per-op accounted at 81.1 vs
    /// 378.8 ms on the h200 DSV3 tp1/dp8/moe_tp8 case).
    #[test]
    fn moe_query_scales_tokens_by_attention_dp() {
        let db = b200_vllm_db();
        let with_dp = op(4).query(&db, 1000).expect("dp=4 query");
        let equivalent = op(1).query(&db, 4000).expect("dp=1 query");
        assert!(
            (with_dp.latency_ms - equivalent.latency_ms).abs() < 1e-12,
            "dp=4 @ 1000 tokens ({}) must equal dp=1 @ 4000 tokens ({})",
            with_dp.latency_ms,
            equivalent.latency_ms
        );
        assert!(with_dp.latency_ms > op(1).query(&db, 1000).unwrap().latency_ms);
    }
}
