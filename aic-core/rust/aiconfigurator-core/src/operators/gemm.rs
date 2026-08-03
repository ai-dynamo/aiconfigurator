// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GEMM operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.gemm.GEMM`. Config-time fields
//! (`n`, `k`, quant mode, scale_factor, scale_num_tokens) are set once when
//! the model graph is built; the query takes only `x` (the M dimension /
//! number of tokens) and routes through the GEMM perf table.
//!
//! For `fp8_static` quant mode, subtracts `compute_scale` overhead from the
//! GEMM latency (and additionally subtracts `scale_matrix` when the input
//! is also low-precision). Post-subtraction latency is floored at the GEMM's
//! own SOL roofline — mirroring Python's `max(latency_floor, latency)` where
//! `latency_floor = query_gemm(..., DatabaseMode.SOL)` — not at 0.

use serde::{Deserialize, Serialize};
use crate::common::enums::{DatabaseMode, GemmQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid, ZeroAwareDeltaLookup};
use crate::perf_database::gemm::{gemm_sol_latency_ms, normalize_fp8_static_quant};
use crate::perf_database::PerfDatabase;

/// GEMM operation: a dense matrix multiply of shape `M=x, N=n, K=k`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GemmOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub k: u32,
    pub quant_mode: GemmQuantMode,
    pub scale_num_tokens: u32,
    pub low_precision_input: bool,
    /// Context-parallel sequence-shard factor (Python's `_seq_split`, = `cp_size`).
    /// The per-rank token count is `ceil(m / seq_split)`. Defaults to 1 (no CP).
    #[serde(default = "default_seq_split")]
    pub seq_split: u32,
}

pub(crate) fn default_seq_split() -> u32 {
    1
}

impl GemmOp {
    /// Convenience constructor for the most common case (no token scaling,
    /// standard input precision).
    pub fn new(name: impl Into<String>, n: u32, k: u32, quant_mode: GemmQuantMode) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            k,
            quant_mode,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
        }
    }

    /// Query GEMM latency for the given `x` (M / number of tokens) and an
    /// optional quant-mode override.
    ///
    /// Mirrors Python's `GEMM.query`:
    /// 1. `m = x // scale_num_tokens`
    /// 2. Query GEMM table at `(m, n, k, quant_mode)`.
    /// 3. For `fp8_static`: subtract `compute_scale(m, k)` (and
    ///    `scale_matrix(m, k)` when low-precision-input).
    /// 4. Clamp to `>= 0`, scale by `scale_factor`.
    pub fn query(
        &self,
        db: &PerfDatabase,
        x: u32,
        quant_override: Option<GemmQuantMode>,
    ) -> Result<PerformanceResult, AicError> {
        let m = x / self.scale_num_tokens.max(1);
        // CP: per-rank token count = ceil(m / seq_split) (busiest rank).
        let m = m.div_ceil(self.seq_split.max(1));
        let quant = quant_override.unwrap_or(self.quant_mode);

        let (mut latency, mut source) = query_gemm_table(db, quant, m, self.n, self.k)?;
        let mut latency_floor = 0.0_f64;

        if quant == GemmQuantMode::Fp8Static {
            // The component sources are irrelevant: the whole fp8_static
            // path is tagged Estimated below regardless of mode.
            let (cs_latency, _) = query_compute_scale_table(db, quant, m, self.k)?;
            latency -= cs_latency;

            if self.low_precision_input {
                let (sm_latency, _) = query_scale_matrix_table(db, quant, m, self.k)?;
                latency -= sm_latency;
            }
            // Python (`operations/gemm.py`): the subtraction leaves a path
            // that still contains the GEMM; independently interpolated
            // component tables can cross, but that path cannot be faster than
            // the GEMM's own roofline. Floor at the SOL (NOT at 0), and tag
            // the result "estimated" (fp8_static is modeled from dynamic FP8
            // plus overhead tables, not measured directly).
            latency_floor = crate::perf_database::gemm::gemm_sol_latency_ms(
                &db.system_spec,
                quant,
                m as f64,
                self.n as f64,
                self.k as f64,
            );
            source = Source::Estimated;
        }

        Ok(PerformanceResult::new(latency.max(latency_floor), source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// Per-tensor weight count in bytes, matching Python's
    /// `GEMM.get_weights()`: `n * k * quant_mode.value.memory *
    /// scale_factor`. The `scale_factor` multiplier mirrors the way model
    /// builders count weights once at construction and let the op replicate
    /// per layer (e.g. `scale_factor = num_hidden_layers` for body GEMMs).
    pub fn weights_bytes(&self) -> f64 {
        (self.n as f64) * (self.k as f64) * self.quant_mode.mapping().memory * self.scale_factor
    }
}

// ---------------------------------------------------------------------------
// Database-mode dispatch, mirroring the Python `_query_*_table` classmethods
// (`operations/gemm.py`): SILICON queries the table; HYBRID converts a typed
// silicon miss into the util-space empirical estimate; EMPIRICAL always
// estimates. The SOL diagnostic modes never reach the compiled engine (the
// routing gate delegates them to the Python step).
// ---------------------------------------------------------------------------

/// GEMM latency for `(m, n, k, quant)` under the database's query mode.
fn query_gemm_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    n: u32,
    k: u32,
) -> Result<(f64, Source), AicError> {
    match db.database_mode {
        DatabaseMode::Empirical => Ok((gemm_empirical(db, quant, m, n, k)?, Source::Empirical)),
        DatabaseMode::Hybrid => match db.gemm.query(quant, m, n, k) {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => {
                Ok((gemm_empirical(db, quant, m, n, k)?, Source::Empirical))
            }
            Err(err) => Err(err),
        },
        _ => Ok((db.gemm.query(quant, m, n, k)?, Source::Silicon)),
    }
}

/// `SOL(query)/util` over the quant's own collected `(m, n, k)` grid.
/// Mirrors Python `_query_gemm_table::get_empirical` (grid depth 3; the SOL
/// uses the ORIGINAL quant, numerically identical to the fp8_static->fp8
/// table quant since the profiles match).
fn gemm_empirical(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    n: u32,
    k: u32,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let sol = |c: &[f64]| gemm_sol_latency_ms(spec, quant, c[0], c[1], c[2]);
    let key = format!("gemm:{}", normalize_fp8_static_quant(quant).name());
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.gemm.gemm_points(quant) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
            // Typed coverage miss -> no grid (estimate() raises the
            // empirical miss); schema/load errors propagate.
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let query = [m as f64, n as f64, k as f64];
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python estimate()'s default provenance).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// compute_scale delta for `(m, k)` under the database's query mode.
fn query_compute_scale_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<(f64, Source), AicError> {
    match db.database_mode {
        DatabaseMode::Empirical => Ok((compute_scale_empirical(db, quant, m, k)?, Source::Empirical)),
        DatabaseMode::Hybrid => match db.gemm.query_compute_scale(quant, m, k) {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => {
                Ok((compute_scale_empirical(db, quant, m, k)?, Source::Empirical))
            }
            Err(err) => Err(err),
        },
        _ => Ok((db.gemm.query_compute_scale(quant, m, k)?, Source::Silicon)),
    }
}

/// Zero-aware nearest-point delta estimate over the compute_scale grid.
/// Mirrors Python `_query_compute_scale_table::get_empirical`: a typed miss
/// on the slice itself becomes the terminal `EmpiricalNotImplemented`.
fn compute_scale_empirical(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<f64, AicError> {
    let key = format!("compute_scale:{}", normalize_fp8_static_quant(quant).name());
    let lookup = db
        .delta_lookups
        .get_or_try_build(&key, || match db.gemm.compute_scale_points(quant) {
            Ok(points) => Ok(ZeroAwareDeltaLookup::new(points)),
            Err(err) if err.is_missing_perf_data() => Err(AicError::EmpiricalNotImplemented(
                format!("No empirical compute_scale data is available for m={m}, k={k}."),
            )),
            Err(err) => Err(err),
        })?;
    // sol_mem = 2 m k / bw * 1000 (read + write of the activation).
    let spec = &db.system_spec;
    let latency =
        lookup.estimate(&[m as f64, k as f64], |c| 2.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0)?;
    // The delta lookup fired (Python `_ZeroAwareDeltaLookup.estimate` notes
    // "empirical"; zero deltas count — they are measured values).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// scale_matrix latency for `(m, k)` under the database's query mode.
fn query_scale_matrix_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<(f64, Source), AicError> {
    match db.database_mode {
        DatabaseMode::Empirical => Ok((scale_matrix_empirical(db, quant, m, k)?, Source::Empirical)),
        DatabaseMode::Hybrid => match db.gemm.query_scale_matrix(quant, m, k) {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => {
                Ok((scale_matrix_empirical(db, quant, m, k)?, Source::Empirical))
            }
            Err(err) => Err(err),
        },
        _ => Ok((db.gemm.query_scale_matrix(quant, m, k)?, Source::Silicon)),
    }
}

/// `SOL(query)/util` over the scale_matrix `(m, k)` grid (a real memory
/// kernel, unlike the compute_scale delta). Mirrors Python
/// `_query_scale_matrix_table::get_empirical` (grid depth 2,
/// `sol_mem = 3 m k / bw * 1000`).
fn scale_matrix_empirical(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let sol = |c: &[f64]| 3.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0;
    let key = format!("scale_matrix:{}", normalize_fp8_static_quant(quant).name());
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.gemm.scale_matrix_points(quant) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let query = [m as f64, k as f64];
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python estimate()'s default provenance).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0").expect("db must load")
    }

    #[test]
    fn gemm_op_query_exact_hit_matches_table() {
        let db = b200_vllm_db();
        // bf16 GEMM at m=32768 n=65536 k=16384 -> latency=41.5967
        let op = GemmOp::new("test", 65536, 16384, GemmQuantMode::Bfloat16);
        let result = op.query(&db, 32768, None).expect("query must succeed");
        assert!(
            (result.latency_ms - 41.59673055013021).abs() < 1e-9,
            "expected recorded latency, got {}",
            result.latency_ms
        );
        assert_eq!(result.source, Source::Silicon);
    }

    #[test]
    fn gemm_op_scale_factor_multiplies_latency() {
        let db = b200_vllm_db();
        let op = GemmOp {
            name: "scaled".to_string(),
            scale_factor: 0.5,
            n: 65536,
            k: 16384,
            quant_mode: GemmQuantMode::Bfloat16,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
        };
        let result = op.query(&db, 32768, None).expect("query must succeed");
        assert!(
            (result.latency_ms - 41.59673055013021 * 0.5).abs() < 1e-9,
            "scale_factor must be applied to latency: got {}",
            result.latency_ms
        );
    }

    #[test]
    fn gemm_op_scale_num_tokens_divides_x() {
        let db = b200_vllm_db();
        // scale_num_tokens=2 means x=65536 should query at m=32768.
        let op = GemmOp {
            name: "halved".to_string(),
            scale_factor: 1.0,
            n: 65536,
            k: 16384,
            quant_mode: GemmQuantMode::Bfloat16,
            scale_num_tokens: 2,
            low_precision_input: false,
            seq_split: 1,
        };
        let result = op.query(&db, 65536, None).expect("query must succeed");
        assert!(
            (result.latency_ms - 41.59673055013021).abs() < 1e-9,
            "scale_num_tokens must divide x: got {}",
            result.latency_ms
        );
    }

    #[test]
    fn gemm_op_quant_override_routes_to_different_quant() {
        let db = b200_vllm_db();
        let op = GemmOp::new("default-bf16", 65536, 16384, GemmQuantMode::Bfloat16);

        // Override to nvfp4 at the same shape -> latency 20.5387 (recorded).
        let result = op
            .query(&db, 32768, Some(GemmQuantMode::Nvfp4))
            .expect("override query must succeed");
        assert!(
            (result.latency_ms - 20.538665771484375).abs() < 1e-9,
            "quant override must change the lookup: got {}",
            result.latency_ms
        );
    }

    /// Oracle values generated from the Python reference on the same data:
    /// `GEMM._query_gemm_table(db, m, n, k, quant, database_mode=EMPIRICAL)`
    /// on b200_sxm/vllm/0.19.0. Regenerate if the shipped GEMM table or the
    /// util-empirical math changes.
    #[test]
    fn gemm_empirical_matches_python_oracles() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        let cases = [
            // off-grid m on a collected (n, k) site
            (3000u32, 65536u32, 16384u32, GemmQuantMode::Bfloat16, 3.7278025700902204),
            // fully off-site query
            (777, 4000, 5000, GemmQuantMode::Bfloat16, 0.023767651577298037),
            // exact collected hit: util reconstruction returns the measured value
            (32768, 65536, 16384, GemmQuantMode::Nvfp4, 20.538665771484375),
            // small-shape corner
            (1, 129, 130, GemmQuantMode::Fp8, 0.004668885990466126),
        ];
        for (m, n, k, quant, expected) in cases {
            let (latency, source) = query_gemm_table(&db, quant, m, n, k).expect("empirical query");
            assert!(
                (latency - expected).abs() < 1e-9,
                "({m}, {n}, {k}, {quant:?}): expected {expected}, got {latency}"
            );
            assert_eq!(source, Source::Empirical);
        }
    }

    /// HYBRID on a quant with NO collected table (int4_wo on b200/vllm) must
    /// surface the terminal EmpiricalNotImplemented miss, never a fabricated
    /// value (mirrors the Python contract).
    #[test]
    fn gemm_hybrid_missing_quant_raises_empirical_not_implemented() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let result = query_gemm_table(&db, GemmQuantMode::Int4Wo, 64, 64, 64);
        assert!(matches!(result, Err(AicError::EmpiricalNotImplemented(_))), "got {result:?}");
    }

    #[test]
    fn gemm_op_weights_bytes_matches_python_formula() {
        let op = GemmOp::new("w", 1024, 4096, GemmQuantMode::Bfloat16);
        // bfloat16 memory factor is 2.0; weights = 1024 * 4096 * 2.0.
        assert_eq!(op.weights_bytes(), 1024.0 * 4096.0 * 2.0);

        let fp8_op = GemmOp::new("w-fp8", 1024, 4096, GemmQuantMode::Fp8);
        // fp8 memory factor is 1.0.
        assert_eq!(fp8_op.weights_bytes(), 1024.0 * 4096.0 * 1.0);
    }
}
