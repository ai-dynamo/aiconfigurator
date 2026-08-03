// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TensorRT-LLM WideEP MoE compute operator.
//!
//! Apple-to-apple port of `aiconfigurator.sdk.operations.moe.TrtLLMWideEPMoE`
//! (`_query_compute_table`). Pure-compute kernel timing (no All2All). The
//! dispatch / combine cost belongs to `MoEDispatchOp` (with the
//! TrtllmAlltoall or DeepEP flavor) or the `wideep` table — depending on
//! which path the model variant exercises.
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
//!
//! Database-mode dispatch follows the gemm.rs reference pattern: EMPIRICAL
//! always estimates `SOL(query)/util` from the op's OWN slice (this table
//! has no transfer ladder — Python's `get_empirical_from_sol` builds a
//! depth-1 grid over the own token curve only); HYBRID queries silicon and
//! converts a typed missing-data error into the estimate; SILICON is
//! unchanged. Both paths resolve the kernel via the `_select_kernel` mirror
//! (architecture-preferred kernel, falling back to whatever the table
//! actually collected).

use serde::{Deserialize, Serialize};
use crate::common::enums::{DatabaseMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid};
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
    /// Wire-carried compile-time kernel hint. NOT used for the lookup:
    /// Python's `_select_kernel(database, quant)` availability fallback only
    /// sees loaded data, so its compile-time value is load-order dependent
    /// (`moe_torch_flow` on a fresh database). The query below re-resolves
    /// at query time exactly like Python's `TrtLLMWideEPMoE.query` does.
    pub kernel_source: String,
}

/// `num_gemms` for the WideEP compute SOL. Python `TrtLLMWideEPMoE.query`
/// never forwards its `is_gated` flag to `query_wideep_moe_compute`, so
/// `_query_compute_table` always runs with its `is_gated=True` default
/// (3 GEMMs: gate + up + down).
const NUM_GEMMS: u64 = 3;

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

        // Database-mode dispatch, mirroring the Python `_query_compute_table`
        // tail (`database._query_silicon_or_hybrid`). The SOL diagnostic
        // modes never reach the compiled engine.
        let (latency, source) = match db.database_mode {
            DatabaseMode::Empirical => (self.empirical_latency(db, scaled)?, Source::Empirical),
            DatabaseMode::Hybrid => match self.silicon_latency(db, scaled) {
                Ok(latency) => (latency, Source::Silicon),
                Err(err) if err.is_missing_perf_data() => {
                    (self.empirical_latency(db, scaled)?, Source::Empirical)
                }
                Err(err) => return Err(err),
            },
            _ => (self.silicon_latency(db, scaled)?, Source::Silicon),
        };
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// SILICON table resolution: query-time kernel selection (Python's
    /// `_select_kernel(database, quant)` — architecture/quant preferred with
    /// the loaded-table availability fallback) + level-wise distribution
    /// fallback in the table layer, token curve anchored on the WideEP MoE
    /// roofline (Python `get_silicon` threads the same `get_sol` closure
    /// through `OpInterpConfig`).
    fn silicon_latency(&self, db: &PerfDatabase, num_tokens: u32) -> Result<f64, AicError> {
        let kernel = self.select_kernel(db)?;
        let spec = &db.system_spec;
        // Engine coordinates are always integral (table keys / the u32
        // query); rounding to u32 keeps integer floor-division parity with
        // Python's `get_sol` (same convention as operators/moe.rs).
        let sol = |t: f64| self.sol_latency_ms(spec, t.round() as u32);
        db.wideep_moe.query_compute(
            num_tokens,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.num_slots,
            self.moe_tp_size,
            self.moe_ep_size,
            self.quant_mode,
            &self.workload_distribution,
            &kernel,
            &sol,
        )
    }

    /// Mirror of Python `TrtLLMWideEPMoE._select_kernel` at QUERY time:
    /// 1. SM >= 100 (Blackwell) with fp8_block -> `deepgemm`;
    /// 2. otherwise -> `moe_torch_flow` (Cutlass);
    /// then keep the preferred kernel if collected, else fall back to the
    /// first collected kernel (BTreeMap order — Python takes the first dict
    /// key; identical for single-kernel shipped tables).
    fn select_kernel(&self, db: &PerfDatabase) -> Result<String, AicError> {
        let is_blackwell = db.system_spec.gpu.sm_version.unwrap_or(0) >= 100;
        // Python: `"fp8_block" in quant_mode_str` (substring, not equality).
        let is_fp8_block = self.quant_mode.name().contains("fp8_block");
        let preferred = if is_blackwell && is_fp8_block {
            "deepgemm"
        } else {
            "moe_torch_flow"
        };
        let available = db.wideep_moe.available_kernels()?;
        if available.iter().any(|k| k == preferred) {
            return Ok(preferred.to_string());
        }
        if let Some(first) = available.into_iter().next() {
            return Ok(first);
        }
        Ok(preferred.to_string())
    }

    /// `SOL(query)/util` over the op's OWN token curve (depth 1, no transfer
    /// ladder). Mirrors Python `_query_compute_table::get_empirical_from_sol`.
    fn empirical_latency(&self, db: &PerfDatabase, num_tokens: u32) -> Result<f64, AicError> {
        let kernel = self.select_kernel(db)?;
        let spec = &db.system_spec;
        let sol = |c: &[f64]| self.sol_latency_ms(spec, c[0].round() as u32);
        let sol_time = self.sol_latency_ms(spec, num_tokens);

        // Python keys on ("wideep_moe", system, backend, version, kernel,
        // quant, topk, num_experts, hidden, inter, num_slots, moe_tp,
        // moe_ep) + id(node); the cache here is per-database, and the
        // requested workload_distribution stands in for the node identity
        // (the resolved slice is a function of it).
        let key = format!(
            "wideep_moe:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            kernel,
            self.quant_mode.name(),
            self.topk,
            self.num_experts,
            self.hidden_size,
            self.inter_size,
            self.num_slots,
            self.moe_tp_size,
            self.moe_ep_size,
            self.workload_distribution,
        );
        let grid = db.util_grids.get_or_try_build(&key, || {
            match db.wideep_moe.slice_points(
                &kernel,
                self.quant_mode,
                &self.workload_distribution,
                self.topk,
                self.num_experts,
                self.hidden_size,
                self.inter_size,
                self.num_slots,
                self.moe_tp_size,
                self.moe_ep_size,
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
        // Own-slice util fired (Python moe.py:1697 estimate()'s default tier).
        db.note_provenance(util_empirical::ProvenanceTier::Empirical);
        Ok(latency)
    }

    /// WideEP MoE roofline SOL (ms) mirroring Python
    /// `_query_compute_table.get_sol`: identical to the plain-MoE roofline
    /// (`operators/moe.rs`) except the weight-read term uses `num_slots`
    /// instead of `num_experts` (WideEP EPLB redundant mode may replicate
    /// experts across slots). `num_experts` never enters the math.
    fn sol_latency_ms(&self, spec: &SystemSpec, num_tokens: u32) -> f64 {
        let total_tokens = num_tokens as u64 * self.topk as u64;
        let moe_ep = (self.moe_ep_size as u64).max(1);
        let moe_tp = (self.moe_tp_size as u64).max(1);
        let h = self.hidden_size as u64;
        let inter = self.inter_size as u64;
        let slots = self.num_slots as u64;

        let ops = total_tokens * h * inter * NUM_GEMMS * 2 / moe_ep / moe_tp;
        let mem_bytes_int = total_tokens / moe_ep * h * 2 // input + output
            + total_tokens / moe_ep * inter * NUM_GEMMS / moe_tp // intermediate
            + h * inter * NUM_GEMMS / moe_tp
                * std::cmp::min(slots / moe_ep, total_tokens / moe_ep); // weights (num_slots)
        let mem_bytes = (mem_bytes_int as f64) * self.quant_mode.mapping().memory;

        // Python indexes `bfloat16_tc_flops` directly (KeyError if absent);
        // every shipped system populates it — same fallback convention as
        // operators/moe.rs.
        let tc_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(1.0);
        let sol_math = (ops as f64) / (tc_flops * self.quant_mode.mapping().compute) * 1000.0;
        let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::enums::TransferPolicy;
    use std::path::PathBuf;

    fn b200_trtllm_db(mode: DatabaseMode) -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "b200_sxm", "trtllm", "1.3.0rc10")
            .expect("db loads")
            .with_mode(mode, TransferPolicy::ALL)
    }

    /// Collected b200 WideEP shape: nvfp4, power_law_1.01, topk=8,
    /// experts=256, hidden=6144, inter=2048, slots=256, tp=1, ep=2.
    fn op() -> WideEpMoeOp {
        WideEpMoeOp {
            name: "wideep_moe".into(),
            scale_factor: 1.0,
            hidden_size: 6144,
            inter_size: 2048,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 2,
            attention_dp_size: 1,
            quant_mode: MoeQuantMode::Nvfp4,
            workload_distribution: "power_law_1.01".into(),
            num_slots: 256,
            // Wire hint only — the query re-resolves the kernel against the
            // loaded table (here: `wideep_compute_cutlass`, the only
            // collected kernel).
            kernel_source: "moe_torch_flow".into(),
        }
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
    /// db = perf_database.get_database_view("b200_sxm", "trtllm", "1.3.0rc10",
    ///     allow_missing_data=True, database_mode=..., shared_layer=False)
    /// float(TrtLLMWideEPMoE._query_compute_table(db, num_tokens=...,
    ///     hidden_size=6144, inter_size=2048, topk=8, num_experts=256,
    ///     num_slots=256, moe_tp_size=1, moe_ep_size=2,
    ///     quant_mode=common.MoEQuantMode.nvfp4,
    ///     workload_distribution=..., database_mode=...))
    /// ```
    ///
    /// nt=1 is a collected point (exact hit, EMPIRICAL == measured);
    /// nt=333 is an off-grid interior point; nt=100000 is beyond the
    /// collected max (16384) — the util-hold anchors on the num_slots-aware
    /// roofline, so it locks the SOL threading through both paths (a linear
    /// token proxy fails this at 1e-9). The kernel resolution also runs the
    /// `_select_kernel` fallback: preferred `moe_torch_flow` is uncollected,
    /// the table only carries `wideep_compute_cutlass`.
    #[test]
    fn wideep_compute_empirical_matches_python_oracles() {
        let db = b200_trtllm_db(DatabaseMode::Empirical);
        let exact = op().query(&db, 1).expect("exact-hit empirical");
        assert_oracle(&exact, 0.08600959777832032, Source::Empirical, "emp_t1");
        let off = op().query(&db, 333).expect("off-grid empirical");
        assert_oracle(&off, 0.4523388956415573, Source::Empirical, "emp_t333");
        let hold = op().query(&db, 100000).expect("beyond-range empirical");
        assert_oracle(&hold, 15.171406557783484, Source::Empirical, "emp_t100000");
        // Python capture: {"empirical"} (own-slice util, no transfer ladder).
        assert_eq!(
            db.worst_provenance(),
            util_empirical::ProvenanceTier::Empirical
        );
    }

    /// HYBRID with data present stays on silicon; the in-range interpolation
    /// differs from the empirical reconstruction at the same point
    /// (0.45251... vs 0.45233...), and the beyond-range hold coincides with
    /// the empirical value (same boundary anchor + same roofline).
    #[test]
    fn wideep_compute_hybrid_prefers_silicon_when_covered() {
        let db = b200_trtllm_db(DatabaseMode::Hybrid);
        let hit = op().query(&db, 1).expect("collected token point");
        assert_oracle(&hit, 0.08600959777832032, Source::Silicon, "hyb_t1");
        let interp = op().query(&db, 333).expect("in-range token interp");
        assert_oracle(&interp, 0.45251359716057776, Source::Silicon, "hyb_t333");
        let hold = op().query(&db, 100000).expect("beyond-range hold");
        assert_oracle(&hold, 15.171406557783484, Source::Silicon, "hyb_t100000");
    }

    /// Distribution fallback: `"balanced"` is uncollected; both silicon and
    /// empirical fall back to the first distribution under (kernel, quant)
    /// (`power_law_1.01`) — nt=64 is a collected point there, so both modes
    /// return the measured value.
    #[test]
    fn wideep_compute_distribution_fallback_matches_python_oracle() {
        let mut fallback_op = op();
        fallback_op.workload_distribution = "balanced".into();
        let emp = fallback_op
            .query(&b200_trtllm_db(DatabaseMode::Empirical), 64)
            .expect("empirical dist fallback");
        assert_oracle(&emp, 0.4135615825653076, Source::Empirical, "emp_dist_fb_t64");
        let hyb = fallback_op
            .query(&b200_trtllm_db(DatabaseMode::Hybrid), 64)
            .expect("hybrid dist fallback");
        assert_oracle(&hyb, 0.4135615825653076, Source::Silicon, "hyb_dist_fb_t64");
    }

    /// HYBRID on a slice with no data anywhere (hidden=9999): the silicon
    /// miss falls to the empirical path, whose own-slice grid is also empty
    /// (this table has NO transfer ladder) -> the typed
    /// EmpiricalNotImplemented miss surfaces, exactly like Python
    /// (`EmpiricalNotImplementedError`).
    #[test]
    fn wideep_compute_hybrid_typed_miss_surfaces_empirical_not_implemented() {
        let db = b200_trtllm_db(DatabaseMode::Hybrid);
        let mut missing = op();
        missing.hidden_size = 9999;
        let result = missing.query(&db, 64);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "expected the typed empirical miss, got {result:?}"
        );

    }

    /// Distribution fallback is FILE-ROW order, not sorted order: gb200's
    /// `wideep_moe_perf.parquet` lists `power_law_1.01_eplb` before
    /// `power_law_1.01` (b200's happens to be sorted, which masked this).
    /// Python's `dists[0]` (dict insertion order — moe.py:1650-1653
    /// empirical, :1755-1765 silicon) therefore answers an uncollected
    /// distribution from the EPLB slice. Oracles:
    ///
    /// ```text
    /// db = perf_database.get_database_view("gb200", "trtllm", "1.3.0rc10",
    ///     allow_missing_data=True, database_mode=..., shared_layer=False)
    /// float(TrtLLMWideEPMoE._query_compute_table(db, num_tokens=...,
    ///     hidden_size=6144, inter_size=2048, topk=8, num_experts=256,
    ///     num_slots=256, moe_tp_size=1, moe_ep_size=2,
    ///     quant_mode=common.MoEQuantMode.nvfp4,
    ///     workload_distribution="balanced", database_mode=...))
    /// ```
    ///
    /// nt=64 is a collected point of the EPLB slice (0.34764...; the sorted
    /// -first `power_law_1.01` slice holds 0.34145... there, so this anchor
    /// fails on lexicographic fallback); nt=333 is an interior lerp,
    /// covering both the silicon and the empirical fallback sites.
    #[test]
    fn wideep_compute_distribution_fallback_uses_file_order() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        let gb200 = |mode: DatabaseMode| {
            PerfDatabase::load(&root, "gb200", "trtllm", "1.3.0rc10")
                .expect("db loads")
                .with_mode(mode, TransferPolicy::ALL)
        };
        let mut fallback_op = op();
        fallback_op.workload_distribution = "balanced".into();

        let hit = fallback_op
            .query(&gb200(DatabaseMode::Hybrid), 64)
            .expect("silicon dist fallback");
        assert_oracle(&hit, 0.34764800071716306, Source::Silicon, "gb200_dist_fb_t64");
        let interp = fallback_op
            .query(&gb200(DatabaseMode::Hybrid), 333)
            .expect("silicon dist fallback interp");
        assert_oracle(&interp, 0.42581739127635954, Source::Silicon, "gb200_dist_fb_t333");
        let emp = fallback_op
            .query(&gb200(DatabaseMode::Empirical), 333)
            .expect("empirical dist fallback");
        assert_oracle(&emp, 0.4259303480537969, Source::Empirical, "gb200_emp_dist_fb_t333");
    }

}
