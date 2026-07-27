// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MHC (Qwen3.5 / DeepSeek-V4 multi-head channel) module operator.
//!
//! Wraps `db.mhc.query_module`, threading the analytic mHC roofline into the
//! table query so beyond-range util-holds anchor on the same SOL Python uses
//! (`dsv4.py::DeepSeekV4MHCModule._query_mhc_table.get_sol`) — the same
//! pattern as `MoeOp` threading `sol_latency_ms` into `MoeTable::query`.
//! The MHC module is collected as a single fused kernel; this operator scales
//! the raw latency by `scale_factor`.

use serde::{Deserialize, Serialize};
use crate::common::enums::{DatabaseMode, GemmQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::gemm::tc_flops_for_compute;
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MhcModuleOp {
    pub name: String,
    pub scale_factor: f64,
    /// Which half of the mHC layer this op models: `pre`, `post`, or `both`.
    /// Part of the table key — pre and post have distinct latencies.
    pub op: String,
    pub hc_mult: u32,
    pub hidden_size: u32,
    /// Emitted by the Python opspec for provenance only. The mHC table is
    /// keyed by compute shape (op, hc_mult, hidden_size) — Python's loader
    /// ignores the architecture column, and so does the Rust one.
    pub architecture: String,
    /// Sinkhorn iteration count (Python `_sinkhorn_iters`, from the model's
    /// `hc_sinkhorn_iters`). Enters the SOL's pre-half op count. Default 20 =
    /// the value every shipped DeepSeek-V4 config carries.
    #[serde(default = "default_sinkhorn_iters")]
    pub sinkhorn_iters: u32,
    /// mHC GEMM quant mode (Python `_quant_mode`; the model always passes
    /// bfloat16 today). Enters the SOL's flops + byte terms.
    #[serde(default = "default_quant_mode")]
    pub quant_mode: GemmQuantMode,
}

fn default_sinkhorn_iters() -> u32 {
    20
}

fn default_quant_mode() -> GemmQuantMode {
    GemmQuantMode::Bfloat16
}

impl MhcModuleOp {
    pub fn new(
        name: impl Into<String>,
        op: impl Into<String>,
        hc_mult: u32,
        hidden_size: u32,
        architecture: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            op: op.into(),
            hc_mult,
            hidden_size,
            architecture: architecture.into(),
            sinkhorn_iters: default_sinkhorn_iters(),
            quant_mode: default_quant_mode(),
        }
    }

    /// Analytic mHC roofline for one RESOLVED op half. Verbatim port of
    /// Python `_query_mhc_table::get_sol` (`operations/dsv4.py`), returning
    /// only the `max(sol_math, sol_mem)` scalar the engine consumes. The
    /// table only ever calls this with `"pre"` / `"post"` (op="both" is
    /// summed at the query level, each half with its own SOL) but the
    /// `"both"` arm is kept for formula completeness.
    fn sol_ms(&self, db: &PerfDatabase, op_name: &str, nt: i64) -> f64 {
        let sites: i128 = 2;
        let nt = nt as i128;
        let hc = self.hc_mult as i128;
        let h = self.hidden_size as i128;
        let sinkhorn = self.sinkhorn_iters as i128;
        let hc_dim = hc * h;
        let mix_hc = (2 + hc) * hc;

        let pre_ops = sites
            * (2 * nt * hc_dim * mix_hc
                + nt * hc_dim * 3
                + nt * (hc * hc + 2 * hc) * sinkhorn
                + 2 * nt * hc * h);
        let post_ops = sites * (2 * nt * hc * hc * h + 2 * nt * hc * h);
        let ops = match op_name {
            "pre" => pre_ops,
            "post" => post_ops,
            _ => pre_ops + post_ops, // "both"
        };

        let mem = self.quant_mode.mapping().memory;
        let param_bytes = (sites * (mix_hc * hc_dim + mix_hc + 3)) as f64 * mem;
        let mut activation_bytes =
            (sites * nt * hc_dim) as f64 * mem * if op_name == "both" { 3.0 } else { 2.0 };
        if op_name == "pre" || op_name == "both" {
            activation_bytes += (sites * nt * (2 * hc + hc * hc)) as f64 * 4.0;
        }

        let spec = &db.system_spec;
        let sol_math =
            ops as f64 / tc_flops_for_compute(spec, self.quant_mode.mapping().compute) * 1000.0;
        let sol_mem = (param_bytes + activation_bytes) / spec.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }

    /// Database-mode dispatch mirroring Python `_query_mhc_table`
    /// (`operations/dsv4.py`): SILICON queries the table; HYBRID converts a
    /// typed silicon miss into the util-space empirical estimate; EMPIRICAL
    /// always estimates. The SOL diagnostic modes never reach the compiled
    /// engine (the routing gate delegates them to the Python step).
    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let sol = |op_name: &str, t: f64| self.sol_ms(db, op_name, t.round() as i64);
        let silicon = || {
            db.mhc
                .query_module(&self.op, num_tokens, self.hc_mult, self.hidden_size, &sol)
        };
        let (latency, source) = match db.database_mode {
            DatabaseMode::Empirical => (self.mhc_empirical(db, num_tokens)?, Source::Empirical),
            DatabaseMode::Hybrid => match silicon() {
                Ok(latency) => (latency, Source::Silicon),
                Err(err) if err.is_missing_perf_data() => {
                    (self.mhc_empirical(db, num_tokens)?, Source::Empirical)
                }
                Err(err) => return Err(err),
            },
            _ => (silicon()?, Source::Silicon),
        };
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// Mirrors Python `_query_mhc_table::get_empirical`: for `op == "both"`
    /// the empirical estimate is the SUM of the two halves' own estimates
    /// (`_emp_for_op("pre") + _emp_for_op("post")`), each half calibrated on
    /// its own token curve with its own SOL.
    fn mhc_empirical(&self, db: &PerfDatabase, num_tokens: u32) -> Result<f64, AicError> {
        if self.op == "both" {
            return Ok(self.emp_for_op(db, "pre", num_tokens)?
                + self.emp_for_op(db, "post", num_tokens)?);
        }
        self.emp_for_op(db, &self.op, num_tokens)
    }

    /// `SOL(query)/util` over one op half's own `(num_tokens,)` curve.
    /// Mirrors Python `_query_mhc_table::get_empirical::_emp_for_op` (grid
    /// depth 1, `sol_fn = lambda c: get_sol(c[0], op_name)[0]`).
    fn emp_for_op(&self, db: &PerfDatabase, op_name: &str, num_tokens: u32) -> Result<f64, AicError> {
        let sol = |c: &[f64]| self.sol_ms(db, op_name, c[0].round() as i64);
        // Python keys the grid on (op_name, hc_mult, hidden_size, quant) —
        // NOT sinkhorn_iters, which is mirrored deliberately.
        let key = format!(
            "dsv4_mhc:{op_name}:{}:{}:{}",
            self.hc_mult,
            self.hidden_size,
            self.quant_mode.name()
        );
        let grid = db.util_grids.get_or_try_build(&key, || {
            match db.mhc.module_points(op_name, self.hc_mult, self.hidden_size) {
                Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
                // Typed coverage miss -> no grid (estimate() raises the
                // empirical miss); schema/load errors propagate.
                Err(err) if err.is_missing_perf_data() => Ok(None),
                Err(err) => Err(err),
            }
        })?;
        let query = [f64::from(num_tokens)];
        let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
        // Own-shape util fired (Python mhc.py, estimate()'s default tier).
        db.note_provenance(util_empirical::ProvenanceTier::Empirical);
        Ok(latency)
    }
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

    fn mhc_op(op: &str) -> MhcModuleOp {
        MhcModuleOp {
            name: "mhc_module".into(),
            scale_factor: 1.0,
            op: op.into(),
            hc_mult: 4,
            hidden_size: 7168,
            architecture: "DeepseekV4ForCausalLM".into(),
            sinkhorn_iters: 20,
            quant_mode: GemmQuantMode::Bfloat16,
        }
    }

    /// Item 1: beyond-range mHC holds must anchor on the mHC ROOFLINE ratio,
    /// mirroring Python `_query_mhc_table` (`sol_fn=lambda t: get_sol(t,
    /// op_name)[0]`). The b200 sglang curve tops out at nt=524288; querying
    /// nt=1048576 exercises the hold. Python oracle generated with:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk import common
    /// db = PerfDatabase('b200_sxm','sglang','0.5.10',
    ///                   systems_root='src/aiconfigurator_core/systems', database_mode='SOL')
    /// for nt, op in [(1048576,'pre'), (1048576,'post'), (1048576,'both')]:
    ///     r = db.query_mhc_module(num_tokens=nt, hidden_size=7168, hc_mult=4,
    ///                             sinkhorn_iters=20, op=op,
    ///                             database_mode=common.DatabaseMode.SILICON)
    ///     print(nt, op, repr(float(r)))"
    /// ```
    ///
    /// The old linear-token-proxy hold returned 2×lat(524288) instead
    /// (pre: 71.5548 vs the roofline 71.55398…), so this fails on the old code.
    #[test]
    fn mhc_beyond_range_hold_matches_python_roofline() {
        let db = b200_sglang_db();
        let cases: &[(&str, f64)] = &[
            ("pre", 71.55398179178216),
            ("post", 40.511536369374085),
            ("both", 112.06551816115625),
        ];
        for &(op, expected) in cases {
            let got = mhc_op(op)
                .query(&db, 1_048_576)
                .expect("query must succeed")
                .latency_ms;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "op={op}: rust {got} vs python {expected}"
            );
        }
    }

    /// In-range queries are SOL-free (RAW lerp / exact hit) and must be
    /// unchanged by the roofline threading. Same oracle command as above with
    /// (3,'pre') and (8,'pre'), sinkhorn_iters irrelevant in range.
    #[test]
    fn mhc_in_range_unchanged_by_roofline() {
        let db = b200_sglang_db();
        for &(nt, expected) in &[(3u32, 0.025050000000000003), (8u32, 0.0251)] {
            let got = mhc_op("pre").query(&db, nt).expect("query must succeed").latency_ms;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "nt={nt}: rust {got} vs python {expected}"
            );
        }
    }

    /// Oracle values generated from the Python reference on the same data:
    ///
    /// ```text
    /// uv run --no-sync python3 -c "
    /// from aiconfigurator.sdk import perf_database, common
    /// from aiconfigurator.sdk.operations.dsv4 import DeepSeekV4MHCModule as MHC
    /// db = perf_database.get_database('b200_sxm', 'sglang', '0.5.10')
    /// for nt, op in [(3000,'pre'), (3000,'post'), (3000,'both'), (8,'pre'), (8,'both'), (1048576,'pre')]:
    ///     r = MHC._query_mhc_table(db, num_tokens=nt, hidden_size=7168, hc_mult=4,
    ///                              sinkhorn_iters=20, op=op, quant_mode=common.GEMMQuantMode.bfloat16,
    ///                              database_mode=common.DatabaseMode.EMPIRICAL)
    ///     print(nt, op, repr(float(r)))"
    /// ```
    ///
    /// Covers: off-grid interior IDW (nt=3000), the exact collected hit
    /// reconstructing the measured value (nt=8), the `both` = emp(pre) +
    /// emp(post) composition at both points, and the beyond-range clamp
    /// (nt=1048576: boundary util frozen, the mHC roofline ratio carries the
    /// growth). Regenerate if the shipped mHC table or the util-empirical
    /// math changes.
    #[test]
    fn mhc_empirical_matches_python_oracles() {
        let mut db = b200_sglang_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        let cases: &[(&str, u32, f64)] = &[
            ("pre", 3000, 0.28677656188924283),
            ("post", 3000, 0.12520766094146765),
            // "both" empirical = emp("pre") + emp("post"), each half on its
            // own curve with its own SOL (Python `_emp_for_op` composition).
            ("both", 3000, 0.41198422283071046),
            ("pre", 8, 0.0251),
            ("both", 8, 0.0357),
            ("pre", 1_048_576, 71.55398179178216),
        ];
        for &(op, nt, expected) in cases {
            let result = mhc_op(op).query(&db, nt).expect("empirical query");
            assert!(
                ((result.latency_ms - expected) / expected).abs() < 1e-9,
                "op={op}, nt={nt}: rust {} vs python {expected}",
                result.latency_ms
            );
            assert_eq!(result.source, Source::Empirical);
        }
    }

    /// HYBRID with silicon data present must stay on the silicon path
    /// (in-range RAW lerp, Source::Silicon) — same oracle as the silicon test.
    #[test]
    fn mhc_hybrid_with_data_stays_silicon() {
        let mut db = b200_sglang_db();
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let result = mhc_op("pre").query(&db, 3).expect("hybrid query");
        let expected = 0.025050000000000003;
        assert!(
            ((result.latency_ms - expected) / expected).abs() < 1e-9,
            "rust {} vs python {expected}",
            result.latency_ms
        );
        assert_eq!(result.source, Source::Silicon);
    }

    /// HYBRID on a slice with NO collected curve (hidden_size=1234 is not in
    /// the mHC table) must surface the terminal EmpiricalNotImplemented miss,
    /// never a fabricated value (mirrors Python: the silicon miss falls to
    /// `get_empirical`, whose own typed miss raises
    /// `EmpiricalNotImplementedError`).
    #[test]
    fn mhc_hybrid_missing_slice_raises_empirical_not_implemented() {
        let mut db = b200_sglang_db();
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let mut op = mhc_op("pre");
        op.hidden_size = 1234;
        let result = op.query(&db, 8);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }

    /// `sinkhorn_iters` / `quant_mode` are new opspec fields; old specs lack
    /// them and must default to (20, bfloat16).
    #[test]
    fn mhc_new_fields_default_in_serde() {
        let mut v = serde_json::to_value(mhc_op("pre")).expect("serialize");
        let obj = v.as_object_mut().expect("object");
        obj.remove("sinkhorn_iters");
        obj.remove("quant_mode");
        let de: MhcModuleOp = serde_json::from_value(v).expect("deserialize");
        assert_eq!(de.sinkhorn_iters, 20);
        assert_eq!(de.quant_mode, GemmQuantMode::Bfloat16);
    }
}
