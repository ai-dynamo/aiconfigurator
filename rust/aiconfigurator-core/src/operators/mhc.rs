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
use crate::common::enums::GemmQuantMode;
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
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

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let sol = |op_name: &str, t: f64| self.sol_ms(db, op_name, t.round() as i64);
        let latency =
            db.mhc
                .query_module(&self.op, num_tokens, self.hc_mult, self.hidden_size, &sol)?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
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
    ///                   systems_root='src/aiconfigurator/systems', database_mode='SOL')
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
