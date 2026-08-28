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

use crate::common::enums::{DatabaseMode, GemmQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::gemm::quant_tc_flops;
use crate::perf_database::PerfDatabase;
use serde::{Deserialize, Serialize};

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
    /// CP sequence-shard factor (Python's `_seq_split`, = `cp_size` for the
    /// context mHC ops): the mHC module is token-major, so the per-rank
    /// payload is `ceil(num_tokens / seq_split)` (Python
    /// `DeepSeekV4MHCModule.query`). Defaults to 1.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub seq_split: u32,
}

fn default_sinkhorn_iters() -> u32 {
    20
}

fn default_quant_mode() -> GemmQuantMode {
    GemmQuantMode::Bfloat16
}

impl MhcModuleOp {
    /// Python `DeepSeekV4MHCModule` weights × scale_factor: two parameter
    /// sets per decoder block (attention mHC and FFN mHC),
    /// `2 * (mix_hc * hc_dim + mix_hc + 3) * quant.memory` with
    /// `mix_hc = (2 + hc_mult) * hc_mult` and `hc_dim = hc_mult * hidden`.
    pub fn weight_bytes(&self) -> f64 {
        let hc_mult = f64::from(self.hc_mult);
        let mix_hc = (2.0 + hc_mult) * hc_mult;
        let hc_dim = hc_mult * f64::from(self.hidden_size);
        2.0 * (mix_hc * hc_dim + mix_hc + 3.0)
            * self.quant_mode.mapping().memory
            * self.scale_factor
    }

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
            seq_split: 1,
        }
    }

    /// Analytic mHC roofline for one RESOLVED op half. Verbatim port of
    /// Python `_query_mhc_table::get_sol` (`operations/dsv4.py`), returning
    /// only the `max(sol_math, sol_mem)` scalar the engine consumes. The
    /// table only ever calls this with `"pre"` / `"post"` (op="both" is
    /// summed at the query level, each half with its own SOL) but the
    /// `"both"` arm is kept for formula completeness.
    fn sol_ms(&self, db: &PerfDatabase, op_name: &str, nt: i64, tc_flops: f64) -> f64 {
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
        let sol_math = ops as f64 / tc_flops * 1000.0;
        let sol_mem = (param_bytes + activation_bytes) / spec.gpu.mem_bw * 1000.0;
        sol_math.max(sol_mem)
    }

    /// Database-mode dispatch mirroring Python `_query_mhc_table`
    /// (`operations/dsv4.py`): SILICON queries the table; HYBRID converts a
    /// typed silicon miss into the util-space empirical estimate; EMPIRICAL
    /// always estimates; SOL (and the retired SOL_FULL alias) returns the
    /// pure analytic roofline with `Source::Sol` and zero energy.
    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // CP: per-rank token count (ceil = busiest rank). Python divides x
        // BEFORE `_query_mhc_table`, so SOL/silicon/empirical all see the
        // per-rank count (`DeepSeekV4MHCModule.query`, dsv4.py).
        let num_tokens = num_tokens.div_ceil(self.seq_split.max(1));
        let tc_flops = quant_tc_flops(&db.system_spec, self.quant_mode.mapping())?;
        let sol = |op_name: &str, t: f64| self.sol_ms(db, op_name, t.round() as i64, tc_flops);
        let silicon = || {
            db.mhc
                .query_module(&self.op, num_tokens, self.hc_mult, self.hidden_size, &sol)
                .map(|v| PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon))
        };
        let result = match db.database_mode {
            // Python `_query_mhc_table`: `get_sol()[0]` at the pre-bound
            // `(nt=num_tokens, op_name=self.op)` — for op == "both" the SOL
            // is the single fused `pre_ops + post_ops` roofline, NOT the
            // empirical path's pre+post sum of estimates.
            DatabaseMode::Sol | DatabaseMode::SolFull => PerformanceResult::new(
                self.sol_ms(db, &self.op, i64::from(num_tokens), tc_flops),
                Source::Sol,
            ),
            DatabaseMode::Empirical => {
                PerformanceResult::new(self.mhc_empirical(db, num_tokens)?, Source::Empirical)
            }
            DatabaseMode::Hybrid => match silicon() {
                Ok(result) => result,
                Err(err) if err.is_missing_perf_data() => {
                    PerformanceResult::new(self.mhc_empirical(db, num_tokens)?, Source::Empirical)
                }
                Err(err) => return Err(err),
            },
            _ => silicon()?,
        };
        Ok(result.clamp_non_negative().scaled(self.scale_factor))
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
    fn emp_for_op(
        &self,
        db: &PerfDatabase,
        op_name: &str,
        num_tokens: u32,
    ) -> Result<f64, AicError> {
        let tc_flops = quant_tc_flops(&db.system_spec, self.quant_mode.mapping())?;
        let sol = |c: &[f64]| self.sol_ms(db, op_name, c[0].round() as i64, tc_flops);
        // Python keys the grid on (op_name, hc_mult, hidden_size, quant) —
        // NOT sinkhorn_iters, which is mirrored deliberately.
        let key = format!(
            "dsv4_mhc:{op_name}:{}:{}:{}",
            self.hc_mult,
            self.hidden_size,
            self.quant_mode.name()
        );
        let grid = db.util_grids.get_or_try_build(&key, || {
            match db
                .mhc
                .module_points(op_name, self.hc_mult, self.hidden_size)
            {
                Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                    points, sol,
                )))),
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
        PerfDatabase::load(&root, "b200_sxm", "sglang", "0.5.14").expect("db loads")
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
            seq_split: 1,
        }
    }

    /// Beyond the collected token range the hold answers from the mHC
    /// roofline; the fused "both" op equals pre + post exactly (additive
    /// SOLs, same hold regime). Relative pins only (2026-08 test policy);
    /// in-range hold-neutrality is pinned synthetically in `perf_interp`.
    #[test]
    fn mhc_beyond_range_hold_is_sol_additive() {
        let db = b200_sglang_db();
        let q = |op: &str| {
            mhc_op(op)
                .query(&db, 1_048_576)
                .expect("query must succeed")
                .latency_ms
        };
        let (pre, post, both) = (q("pre"), q("post"), q("both"));
        assert!(pre > 0.0 && post > 0.0);
        assert!(
            ((both - (pre + post)) / both).abs() < 1e-9,
            "both must be SOL-additive beyond range: {both} vs {pre}+{post}"
        );
    }

    /// Routing + the "both" = emp(pre) + emp(post) composition, pinned
    /// RELATIVELY (each half resolves on its own curve with its own SOL).
    /// Estimator math lives on synthetic grids; values in the goldens.
    #[test]
    fn mhc_empirical_regime_routing() {
        let mut db = b200_sglang_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        let q = |op: &str, nt: u32| {
            let r = mhc_op(op).query(&db, nt).expect("empirical query");
            assert!(
                r.latency_ms.is_finite() && r.latency_ms > 0.0,
                "op={op}, nt={nt}"
            );
            assert_eq!(r.source, Source::Empirical, "op={op}, nt={nt}");
            r.latency_ms
        };
        for nt in [3000u32, 8, 1_048_576] {
            let (pre, post, both) = (q("pre", nt), q("post", nt), q("both", nt));
            assert!(
                ((both - (pre + post)) / both).abs() < 1e-9,
                "nt={nt}: both must compose pre+post: {both} vs {pre}+{post}"
            );
        }
    }

    /// Cross-mode relative pin: HYBRID replays the SILICON answer on a
    /// covered slice (the empirical layer must not preempt it).
    #[test]
    fn mhc_hybrid_with_data_stays_silicon() {
        let sil = b200_sglang_db();
        let want = mhc_op("pre")
            .query(&sil, 3)
            .expect("silicon query")
            .latency_ms;
        let mut db = b200_sglang_db();
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let result = mhc_op("pre").query(&db, 3).expect("hybrid query");
        assert!(
            (result.latency_ms - want).abs() < 1e-12,
            "hybrid must replay silicon: {} vs {want}",
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

    /// seq_split=k must equal the direct query at ceil(nt/k) tokens —
    /// relative pin, no recorded values (2026-08 test policy).
    #[test]
    fn mhc_seq_split_divides_tokens_like_python_cp() {
        let db = b200_sglang_db();
        let q_split = |op: &str, nt: u32, split: u32| {
            let mut o = mhc_op(op);
            o.seq_split = split;
            o.query(&db, nt).expect("query must succeed").latency_ms
        };
        for &(op, nt, split) in &[
            ("pre", 8192u32, 8u32),
            ("post", 8192, 8),
            ("pre", 8193, 8),
            ("both", 8192, 8),
            ("pre", 8192, 1),
        ] {
            let got = q_split(op, nt, split);
            let want = q_split(op, nt.div_ceil(split), 1);
            assert!(
                (got - want).abs() < 1e-12,
                "op={op}, nt={nt}, split={split}: {got} vs direct {want}"
            );
        }
    }

    /// `sinkhorn_iters` / `quant_mode` are new opspec fields; old specs lack
    /// them and must default to (20, bfloat16).
    #[test]
    fn mhc_new_fields_default_in_serde() {
        let mut v = serde_json::to_value(mhc_op("pre")).expect("serialize");
        let obj = v.as_object_mut().expect("object");
        obj.remove("sinkhorn_iters");
        obj.remove("quant_mode");
        obj.remove("seq_split");
        let de: MhcModuleOp = serde_json::from_value(v).expect("deserialize");
        assert_eq!(de.sinkhorn_iters, 20);
        assert_eq!(de.quant_mode, GemmQuantMode::Bfloat16);
        assert_eq!(de.seq_split, 1);
    }

    /// SOL mode returns the fused mHC roofline tagged `Source::Sol` — for
    /// `op == "both"` the SINGLE `pre_ops + post_ops` formula, NOT the
    /// empirical path's pre+post sum of estimates (Python `_query_mhc_table`
    /// SOL branch calls `get_sol()` once at the bound op name).
    #[test]
    fn mhc_sol_mode_returns_fused_roofline_with_sol_source() {
        let mut db = b200_sglang_db();
        db.database_mode = DatabaseMode::Sol;
        let op = mhc_op("both");
        let tc_flops = quant_tc_flops(&db.system_spec, op.quant_mode.mapping()).unwrap();
        let result = op.query(&db, 512).expect("mhc sol");
        let expected = op.sol_ms(&db, "both", 512, tc_flops);
        assert_eq!(result.latency_ms, expected);
        assert_eq!(result.source, Source::Sol);
        assert_eq!(result.energy_wms, 0.0);
        // Fused "both" == pre + post SOL (linear in ops and bytes), and both
        // halves are individually positive.
        let pre = op.sol_ms(&db, "pre", 512, tc_flops);
        let post = op.sol_ms(&db, "post", 512, tc_flops);
        assert!(pre > 0.0 && post > 0.0);
        assert!(expected <= pre + post + 1e-12);
    }
}
