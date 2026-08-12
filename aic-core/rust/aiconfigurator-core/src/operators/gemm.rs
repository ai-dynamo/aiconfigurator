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

use crate::common::enums::{DatabaseMode, GemmQuantMode, TransferKind};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::moe::policy_fingerprint;
use crate::operators::util_empirical::{self, UtilGrid, ZeroAwareDeltaLookup};
use crate::perf_database::gemm::{
    gemm_quant_by_name, gemm_sol_latency_ms_with_flops, normalize_fp8_static_quant, quant_tc_flops,
};
use crate::perf_database::PerfDatabase;
use serde::{Deserialize, Serialize};

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
    /// Opt-in: a shape outside the collected grid degrades to SOL instead of
    /// erroring (SILICON only; quant-mode misses stay strict, HYBRID keeps
    /// its empirical fallback). Mirrors Python `below_grid_sol`.
    #[serde(default)]
    pub below_grid_sol: bool,
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
            below_grid_sol: false,
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

        // Opt-in below-grid degrade (Python `_query_gemm_table`): a
        // SILICON shape miss falls to SOL when the quant's table exists;
        // quant-mode misses stay strict and HYBRID keeps its empirical
        // fallback. SOL carries no energy (Python passes energy=0.0).
        let base = match query_gemm_table(db, quant, m, self.n, self.k) {
            Err(err)
                if self.below_grid_sol
                    && db.database_mode == DatabaseMode::Silicon
                    && err.is_missing_perf_data()
                    && db.gemm.has_quant(quant)? =>
            {
                let tc_flops = quant_tc_flops(&db.system_spec, quant.mapping())?;
                PerformanceResult::new(
                    gemm_sol_latency_ms_with_flops(
                        &db.system_spec,
                        quant,
                        tc_flops,
                        m as f64,
                        self.n as f64,
                        self.k as f64,
                    ),
                    Source::Sol,
                )
            }
            other => other?,
        };
        let mut latency = base.latency_ms;
        let mut energy = base.energy_wms;
        let mut source = base.source;
        let mut latency_floor = 0.0_f64;

        if quant == GemmQuantMode::Fp8Static {
            // The component sources are irrelevant: the whole fp8_static
            // path is tagged Estimated below regardless of mode. Energy is
            // subtracted alongside latency (Python `GEMM.query`).
            let cs = query_compute_scale_table(db, quant, m, self.k)?;
            latency -= cs.latency_ms;
            energy -= cs.energy_wms;

            if self.low_precision_input {
                let sm = query_scale_matrix_table(db, quant, m, self.k)?;
                latency -= sm.latency_ms;
                energy -= sm.energy_wms;
            }
            // Python (`operations/gemm.py`): the subtraction leaves a path
            // that still contains the GEMM; independently interpolated
            // component tables can cross, but that path cannot be faster than
            // the GEMM's own roofline. Floor at the SOL (NOT at 0), and tag
            // the result "estimated" (fp8_static is modeled from dynamic FP8
            // plus overhead tables, not measured directly). Energy has no
            // analogous SOL model, so it keeps the plain non-negative clamp.
            let tc_flops = quant_tc_flops(&db.system_spec, quant.mapping())?;
            latency_floor = gemm_sol_latency_ms_with_flops(
                &db.system_spec,
                quant,
                tc_flops,
                m as f64,
                self.n as f64,
                self.k as f64,
            );
            source = Source::Estimated;
        }

        Ok(
            PerformanceResult::with_energy(latency.max(latency_floor), energy, source)
                .clamp_non_negative()
                .scaled(self.scale_factor),
        )
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
// estimates; SOL (and the retired SOL_FULL alias) returns the pure
// speed-of-light roofline with `Source::Sol` and zero energy.
// ---------------------------------------------------------------------------

/// GEMM latency + energy for `(m, n, k, quant)` under the database's query
/// mode. Silicon-table hits carry the table's energy (Python `_interp_pr`);
/// empirical fallbacks are energy-0.0 (Python passes `energy=0.0`).
fn query_gemm_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    n: u32,
    k: u32,
) -> Result<PerformanceResult, AicError> {
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon)
    };
    match db.database_mode {
        // Python `_query_gemm_table`: `get_sol(m, n, k, quant_mode)[0]` at the
        // RAW quant mode (not the fp8_static-normalized table quant).
        DatabaseMode::Sol | DatabaseMode::SolFull => {
            let tc_flops = quant_tc_flops(&db.system_spec, quant.mapping())?;
            Ok(PerformanceResult::new(
                gemm_sol_latency_ms_with_flops(
                    &db.system_spec,
                    quant,
                    tc_flops,
                    m as f64,
                    n as f64,
                    k as f64,
                ),
                Source::Sol,
            ))
        }
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            gemm_empirical(db, quant, m, n, k)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.gemm.query(quant, m, n, k) {
            Ok(value) => Ok(silicon(value)),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                gemm_empirical(db, quant, m, n, k)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok(silicon(db.gemm.query(quant, m, n, k)?)),
    }
}

/// Per-quant achieved-util LEVEL `e(q)` for GEMM, keyed by the
/// `(memory, compute)` profile. Mirrors `_GEMM_QUANT_UTIL_LEVEL`
/// (`operations/gemm.py`); consumed ONLY by the cross-profile relation of
/// the quant-transfer ladder, and only as the ratio `e(query)/e(ref)`.
/// Derivation method + LOO evidence documented on the Python table.
const GEMM_QUANT_UTIL_LEVEL: &[(f64, f64, f64)] = &[
    (2.0, 1.0, 0.70),    // w16a16 / bfloat16              [data 0.55-0.79]
    (1.0, 1.0, 0.55),    // w8a16 / int8_wo                [inferred]
    (0.5625, 1.0, 0.45), // w4a16+scales / nvfp4_wo (Marlin FP4) [copies inferred (0.5,1)]
    (0.5, 1.0, 0.45),    // w4a16 / int4_wo                [inferred]
    (1.0, 2.0, 0.45),    // w8a8 / fp8(_block/_ootb), sq   [data 0.28-0.55]
    (0.5, 2.0, 0.35),    // w4a8                           [inferred]
    (1.0, 4.0, 0.30),    // w8a4                           [inferred]
    (0.5, 4.0, 0.30),    // w4a4                           [inferred ≈ nvfp4]
    (0.5625, 4.0, 0.30), // w4a4 / nvfp4                   [data 0.21-0.36]
];
/// Unlisted profile: mid-range relative level (Python `_GEMM_QUANT_UTIL_DEFAULT`).
const GEMM_QUANT_UTIL_DEFAULT: f64 = 0.45;

/// Achieved-util level `e(q)` for a GEMM quant, by `(memory, compute)`
/// profile (mirrors `_gemm_quant_util_level`, `operations/gemm.py`).
fn gemm_quant_util_level(quant: GemmQuantMode) -> f64 {
    let mapping = quant.mapping();
    GEMM_QUANT_UTIL_LEVEL
        .iter()
        .find(|(memory, compute, _)| *memory == mapping.memory && *compute == mapping.compute)
        .map(|(_, _, level)| *level)
        .unwrap_or(GEMM_QUANT_UTIL_DEFAULT)
}

/// Collected quants with a DIFFERENT `(memory, compute)` profile than the
/// query, ordered lexicographically by `(|Δcompute|, |Δmemory|)` — GEMM uses
/// the compute-first metric (Python `xprofile_quant_order(...,
/// prefer_same_compute=True)`): the COMPUTE factor (activation precision)
/// determines the dense kernel's compute family (a weight-only quant runs
/// bf16 MMA after fused dequant), so a same-compute reference must always
/// beat a cross-compute one — under plain L1 a weight-only quant such as
/// (0.5625, 1) TIES between bfloat16 and fp8 and the reference would be a
/// file-order lottery. Stable sort keeps file order on full ties.
fn xprofile_gemm_quants(
    query: GemmQuantMode,
    table_quants: &[GemmQuantMode],
) -> Vec<GemmQuantMode> {
    let qp = query.mapping();
    let mut refs: Vec<GemmQuantMode> = table_quants
        .iter()
        .copied()
        .filter(|q| {
            let mapping = q.mapping();
            *q != query && !(mapping.memory == qp.memory && mapping.compute == qp.compute)
        })
        .collect();
    let dist = |q: GemmQuantMode| {
        let mapping = q.mapping();
        (
            (mapping.compute - qp.compute).abs(),
            (mapping.memory - qp.memory).abs(),
        )
    };
    refs.sort_by(|a, b| {
        dist(*a)
            .partial_cmp(&dist(*b))
            .expect("finite profile distances")
    });
    refs
}

/// Distinct quant names of the loaded GEMM table in first-seen (file row)
/// order, parsed to the enum. A typed table miss yields an empty list —
/// mirroring Python, where the ladder's table accesses happen inside
/// `grid_from_reference`'s typed-miss catch.
fn gemm_table_quants(db: &PerfDatabase) -> Result<Vec<GemmQuantMode>, AicError> {
    match db.gemm.available_quants() {
        Ok(names) => Ok(names.iter().filter_map(|n| gemm_quant_by_name(n)).collect()),
        Err(err) if err.is_missing_perf_data() => Ok(Vec::new()),
        Err(err) => Err(err),
    }
}

/// A borrowed util grid over `source_quant`'s whole `(m, n, k)` table, built
/// with `sol_quant`'s SOL (ReferenceCandidate contract: QUERY quant for the
/// same-profile relation — numerically identical SOL — REFERENCE quant for
/// cross-profile). `None` on a typed data miss; emptiness is the caller's
/// check.
fn gemm_reference_grid(
    db: &PerfDatabase,
    source_quant: GemmQuantMode,
    sol_quant: GemmQuantMode,
    provenance: &'static str,
    key: &str,
) -> Result<Option<std::sync::Arc<UtilGrid>>, AicError> {
    let spec = &db.system_spec;
    db.util_grids.get_or_try_build(key, || {
        let tc_flops = quant_tc_flops(spec, sol_quant.mapping())?;
        let sol =
            |c: &[f64]| gemm_sol_latency_ms_with_flops(spec, sol_quant, tc_flops, c[0], c[1], c[2]);
        match db.gemm.gemm_points(source_quant) {
            Ok(points) => {
                let mut grid = UtilGrid::new(util_empirical::build_samples(points, sol));
                grid.reference_provenance = Some(provenance);
                Ok(Some(grid))
            }
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })
}

/// `SOL(query)/util` over the quant's own collected `(m, n, k)` grid, with
/// the quant-transfer ladder behind an own-data miss. Mirrors Python
/// `_query_gemm_table::get_empirical` + `util_empirical.quant_transfer_grid`
/// (grid depth 3; the own-grid SOL uses the ORIGINAL quant, numerically
/// identical to the fp8_static->fp8 table quant since the profiles match):
///
/// 1. own-quant grid — inherently cross-shape (depth-3 over every collected
///    `(m, n, k)`), so GEMM's **xshape relation class is structurally
///    empty** and no xshape step exists;
/// 2. **xquant** — FIRST same-profile sibling in file order (Python pools
///    one whole-table candidate per sibling with constant features; the
///    stable nearest-candidate argmin selects the first). Not retried on an
///    empty grid — parity with the single pooled selection;
/// 3. **xprofile** — compute-first nearest-profile walk, first quant with a
///    non-empty grid wins, util rescaled by `e(query)/e(ref)`.
fn gemm_empirical(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    n: u32,
    k: u32,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let tc_flops = quant_tc_flops(spec, quant.mapping())?;
    let sol = |c: &[f64]| gemm_sol_latency_ms_with_flops(spec, quant, tc_flops, c[0], c[1], c[2]);
    let tqm = normalize_fp8_static_quant(quant);
    let key = format!("gemm:{}", tqm.name());
    let mut grid = db.util_grids.get_or_try_build(&key, || {
        match db.gemm.gemm_points(quant) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                points, sol,
            )))),
            // Typed coverage miss -> no grid (the ladder below, then
            // estimate(), takes over); schema/load errors propagate.
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;

    let mut util_scale = 1.0;
    if grid.as_deref().is_none_or(UtilGrid::is_empty) {
        let policy = db.transfer_policy;
        let table_quants = gemm_table_quants(db)?;
        let fingerprint = policy_fingerprint(policy);

        if policy.contains(TransferKind::XQuant) {
            let qp = tqm.mapping();
            if let Some(ref_q) = table_quants.iter().copied().find(|q| {
                let mapping = q.mapping();
                *q != tqm && mapping.memory == qp.memory && mapping.compute == qp.compute
            }) {
                let key = format!(
                    "gemm_xquant:{}:policy={}:ref={}",
                    tqm.name(),
                    fingerprint,
                    ref_q.name()
                );
                if let Some(reference) = gemm_reference_grid(db, ref_q, tqm, "xquant", &key)? {
                    grid = Some(reference);
                }
            }
        }

        if grid.as_deref().is_none_or(UtilGrid::is_empty) && policy.contains(TransferKind::XProfile)
        {
            for ref_q in xprofile_gemm_quants(tqm, &table_quants) {
                let key = format!(
                    "gemm_xprofile:{}:policy={}:ref={}",
                    tqm.name(),
                    fingerprint,
                    ref_q.name()
                );
                if let Some(reference) = gemm_reference_grid(db, ref_q, ref_q, "xprofile", &key)? {
                    if !reference.is_empty() {
                        grid = Some(reference);
                        util_scale = gemm_quant_util_level(tqm) / gemm_quant_util_level(ref_q);
                        break;
                    }
                }
            }
        }
    }

    let query = [m as f64, n as f64, k as f64];
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), util_scale)?;
    // Relation fired = the reference grid's transfer kind (xquant/xprofile),
    // or own-quant "empirical" when no borrow happened.
    db.note_provenance(
        grid.as_deref()
            .and_then(|g| g.reference_provenance)
            .and_then(util_empirical::ProvenanceTier::from_tag)
            .unwrap_or(util_empirical::ProvenanceTier::Empirical),
    );
    Ok(latency)
}

/// compute_scale delta (latency + energy) for `(m, k)` under the database's
/// query mode. Empirical estimates are energy-0.0.
fn query_compute_scale_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<PerformanceResult, AicError> {
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon)
    };
    match db.database_mode {
        // Python `_query_compute_scale_table::get_sol`:
        // `sol_mem = 2 m k / bw * 1000` (read + write of the activation).
        DatabaseMode::Sol | DatabaseMode::SolFull => Ok(PerformanceResult::new(
            2.0 * m as f64 * k as f64 / db.system_spec.gpu.mem_bw * 1000.0,
            Source::Sol,
        )),
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            compute_scale_empirical(db, quant, m, k)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.gemm.query_compute_scale(quant, m, k) {
            Ok(value) => Ok(silicon(value)),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                compute_scale_empirical(db, quant, m, k)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok(silicon(db.gemm.query_compute_scale(quant, m, k)?)),
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
    let lookup =
        db.delta_lookups
            .get_or_try_build(&key, || match db.gemm.compute_scale_points(quant) {
                Ok(points) => Ok(ZeroAwareDeltaLookup::new(points)),
                Err(err) if err.is_missing_perf_data() => Err(AicError::EmpiricalNotImplemented(
                    format!("No empirical compute_scale data is available for m={m}, k={k}."),
                )),
                Err(err) => Err(err),
            })?;
    // sol_mem = 2 m k / bw * 1000 (read + write of the activation).
    let spec = &db.system_spec;
    let latency = lookup.estimate(&[m as f64, k as f64], |c| {
        2.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0
    })?;
    // The delta lookup fired (Python `_ZeroAwareDeltaLookup.estimate` notes
    // "empirical"; zero deltas count — they are measured values).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// scale_matrix latency + energy for `(m, k)` under the database's query
/// mode. Empirical estimates are energy-0.0.
fn query_scale_matrix_table(
    db: &PerfDatabase,
    quant: GemmQuantMode,
    m: u32,
    k: u32,
) -> Result<PerformanceResult, AicError> {
    let silicon = |v: crate::perf_database::perf_interp::LeafValue| {
        PerformanceResult::with_energy(v.latency, v.energy, Source::Silicon)
    };
    match db.database_mode {
        // Python `_query_scale_matrix_table::get_sol`:
        // `sol_mem = 3 m k / bw * 1000`.
        DatabaseMode::Sol | DatabaseMode::SolFull => Ok(PerformanceResult::new(
            3.0 * m as f64 * k as f64 / db.system_spec.gpu.mem_bw * 1000.0,
            Source::Sol,
        )),
        DatabaseMode::Empirical => Ok(PerformanceResult::new(
            scale_matrix_empirical(db, quant, m, k)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.gemm.query_scale_matrix(quant, m, k) {
            Ok(value) => Ok(silicon(value)),
            Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                scale_matrix_empirical(db, quant, m, k)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok(silicon(db.gemm.query_scale_matrix(quant, m, k)?)),
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
    let grid =
        db.util_grids
            .get_or_try_build(&key, || match db.gemm.scale_matrix_points(quant) {
                Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                    points, sol,
                )))),
                Err(err) if err.is_missing_perf_data() => Ok(None),
                Err(err) => Err(err),
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
            below_grid_sol: false,
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
            below_grid_sol: false,
        };
        let result = op.query(&db, 65536, None).expect("query must succeed");
        assert!(
            (result.latency_ms - 41.59673055013021).abs() < 1e-9,
            "scale_num_tokens must divide x: got {}",
            result.latency_ms
        );
    }

    /// Oracle from the Python reference on the same data:
    /// `float(db.query_gemm(8, 1, 2048, GEMMQuantMode.bfloat16,
    /// database_mode=DatabaseMode.SOL))` = 4.78961038961039e-06.
    #[test]
    fn gemm_op_below_grid_sol_flag_degrades_shape_miss_to_sol() {
        let db = b200_vllm_db();
        // n=1 is ~5 octaves below the smallest collected n: a strict miss.
        let strict = GemmOp::new("gate", 1, 2048, GemmQuantMode::Bfloat16);
        assert!(strict.query(&db, 8, None).is_err());

        let op = GemmOp {
            below_grid_sol: true,
            ..strict
        };
        let result = op
            .query(&db, 8, None)
            .expect("below-grid opt-in must degrade to SOL");
        assert_eq!(result.source, Source::Sol);
        assert!(
            (result.latency_ms - 4.78961038961039e-06).abs() < 1e-15,
            "expected the Python SOL oracle, got {}",
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
            (
                3000u32,
                65536u32,
                16384u32,
                GemmQuantMode::Bfloat16,
                3.7278025700902204,
            ),
            // fully off-site query
            (
                777,
                4000,
                5000,
                GemmQuantMode::Bfloat16,
                0.023767651577298037,
            ),
            // exact collected hit: util reconstruction returns the measured value
            (
                32768,
                65536,
                16384,
                GemmQuantMode::Nvfp4,
                20.538665771484375,
            ),
            // small-shape corner
            (1, 129, 130, GemmQuantMode::Fp8, 0.004668885990466126),
        ];
        for (m, n, k, quant, expected) in cases {
            let result = query_gemm_table(&db, quant, m, n, k).expect("empirical query");
            let latency = result.latency_ms;
            assert!(
                (latency - expected).abs() < 1e-9,
                "({m}, {n}, {k}, {quant:?}): expected {expected}, got {latency}"
            );
            assert_eq!(result.source, Source::Empirical);
            assert_eq!(
                result.energy_wms, 0.0,
                "empirical fallback carries no energy"
            );
        }
    }

    /// HYBRID on a quant with NO collected table under a policy WITHOUT
    /// XProfile must surface the terminal EmpiricalNotImplemented miss, never
    /// a fabricated value (mirrors the Python contract: cross-profile
    /// borrowing is policy-gated, never implicit). int4_wo (0.5, 1) has no
    /// same-profile sibling anywhere, so "balanced" finds nothing.
    #[test]
    fn gemm_hybrid_missing_quant_raises_when_policy_forbids_transfer() {
        let balanced = crate::common::enums::TransferPolicy {
            xshape: true,
            xquant: true,
            xprofile: false,
            xop: false,
        };
        let db = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, balanced);
        let result = query_gemm_table(&db, GemmQuantMode::Int4Wo, 64, 64, 64);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }

    /// Under the default (all-on) policy the same query resolves via the
    /// xprofile borrow. Python oracle (shared layer OFF, HYBRID):
    ///
    /// ```text
    /// db = perf_database.get_database_view("b200_sxm", "vllm", "0.19.0",
    ///     allow_missing_data=True, database_mode=DatabaseMode.HYBRID,
    ///     shared_layer=False)
    /// float(GEMM._query_gemm_table(db, 64, 64, 64, GEMMQuantMode.int4_wo))
    /// # -> 0.0007006913814463733, provenance {"xprofile"}
    /// ```
    ///
    /// The b200/vllm file order is [nvfp4, bfloat16, fp8, fp8_block]; the
    /// compute-first ordering must borrow bfloat16 (Δcompute=0), not the
    /// file-order-first nvfp4.
    #[test]
    fn gemm_hybrid_missing_quant_borrows_xprofile_under_default_policy() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let result =
            query_gemm_table(&db, GemmQuantMode::Int4Wo, 64, 64, 64).expect("xprofile borrow");
        let latency = result.latency_ms;
        assert!(
            (latency - 0.0007006913814463733).abs() < 1e-9,
            "expected python oracle, got {latency}"
        );
        assert_eq!(result.source, Source::Empirical);
        assert_eq!(
            db.worst_provenance(),
            util_empirical::ProvenanceTier::XProfile
        );
    }

    fn h200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&systems_root, "h200_sxm", "vllm", "0.19.0").expect("db must load")
    }

    /// Quant-transfer ladder oracles on h200/vllm/0.19.0 (collected quants in
    /// file order: [fp8, bfloat16, fp8_block]). Python oracle generation:
    ///
    /// ```text
    /// db = perf_database.get_database_view("h200_sxm", "vllm", "0.19.0",
    ///     allow_missing_data=True, database_mode=DatabaseMode.HYBRID,
    ///     shared_layer=False)
    /// float(GEMM._query_gemm_table(db, m, 4096, 4096, quant))
    /// ```
    ///
    /// - sq (1, 2): xquant borrow from fp8 (first same-profile in file
    ///   order; identical SOL, no rescale).
    /// - int4_wo (0.5, 1): xprofile borrow from bfloat16 (Δcompute=0), NOT
    ///   the file-order-first fp8 — under plain L1 the two TIE at 1.5 and a
    ///   regression to L1/file-order selection changes this value.
    /// - nvfp4 (0.5625, 4): NO LONGER a ladder vehicle on h200 — the strict
    ///   per-dtype resolution (#1398) rejects fp4 at query entry because
    ///   h200 has no fp4 tensor cores (its old xprofile estimate was
    ///   anchored on a fictional bf16*4 SOL). Weight-only fp4 modes (the
    ///   #1392 nvfp4_wo plan) declare the bf16 compute pipeline and keep
    ///   using the ladder like int4_wo does.
    /// Regenerate if the shipped GEMM tables or the util math change.
    #[test]
    fn gemm_quant_transfer_ladder_matches_python_oracles() {
        let mut db = h200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let cases = [
            (
                GemmQuantMode::Sq,
                512u32,
                0.01826755536927117,
                util_empirical::ProvenanceTier::XQuant,
            ),
            (
                GemmQuantMode::Sq,
                8192,
                0.21285422643025717,
                util_empirical::ProvenanceTier::XQuant,
            ),
            (
                GemmQuantMode::Int4Wo,
                512,
                0.036529976276703825,
                util_empirical::ProvenanceTier::XProfile,
            ),
            (
                GemmQuantMode::Int4Wo,
                8192,
                0.5714972813924153,
                util_empirical::ProvenanceTier::XProfile,
            ),
        ];
        for (quant, m, expected, tier) in cases {
            db.reset_provenance();
            let result = query_gemm_table(&db, quant, m, 4096, 4096).expect("ladder query");
            let latency = result.latency_ms;
            assert!(
                (latency - expected).abs() < 1e-9,
                "({quant:?}, m={m}): expected {expected}, got {latency}"
            );
            assert_eq!(
                result.source,
                Source::Empirical,
                "({quant:?}, m={m}): wrong source"
            );
            assert_eq!(
                db.worst_provenance(),
                tier,
                "({quant:?}, m={m}): wrong tier"
            );
        }
        // fp4 on h200: strict resolution fires before the ladder.
        assert!(matches!(
            query_gemm_table(&db, GemmQuantMode::Nvfp4, 512, 4096, 4096),
            Err(AicError::MissingSystemFlops(_))
        ));
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

    /// ENERGY oracle for the fp8_static composition on synthetic
    /// power-carrying fixtures through a full `PerfDatabase` (systems root =
    /// `energy_test_fixtures::write_energy_systems_root`). Python twin
    /// (pandas fixtures with the same rows; `GEMM` op from
    /// `operations/gemm.py`):
    ///
    /// ```text
    /// gemm_perf:         fp8 m=128/256 @ (n=1024, k=1024): lat 1.0/3.0, power 100/200
    /// computescale_perf: fp8 m=128/256 @ k=1024:           lat 0.25/0.75, power 40/80
    /// scale_matrix_perf: fp8 m=128/256 @ k=1024:           lat 0.125/0.375, power 20/60
    /// GEMM("g", 2.0, 1024, 1024, fp8_static, low_precision_input=True).query(db, x=192)
    /// # -> latency=2.5, energy=520.0     (= (2.0-0.5-0.25, 300-30-10) * 2)
    /// GEMM("g1", 1.0, 1024, 1024, fp8_static).query(db, x=192)
    /// # -> latency=1.5, energy=270.0     (= (2.0-0.5, 300-30) * 1)
    /// ```
    ///
    /// Exercises: energy subtraction of the compute_scale and scale_matrix
    /// pieces, the (non-binding here) SOL latency floor + non-negative
    /// energy clamp, and `_scale_factor` scaling of BOTH latency and energy.
    #[test]
    fn gemm_fp8_static_energy_composition_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{
            write_energy_systems_root, write_parquet, Col,
        };
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        write_parquet(
            &data.join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", vec!["fp8", "fp8"]),
                Col::I64("m", vec![128, 256]),
                Col::I64("n", vec![1024, 1024]),
                Col::I64("k", vec![1024, 1024]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        write_parquet(
            &data.join("computescale_perf.parquet"),
            &[
                Col::Str("quant_dtype", vec!["fp8", "fp8"]),
                Col::I64("m", vec![128, 256]),
                Col::I64("k", vec![1024, 1024]),
                Col::F64("latency", vec![0.25, 0.75]),
                Col::F64("power", vec![40.0, 80.0]),
            ],
        );
        write_parquet(
            &data.join("scale_matrix_perf.parquet"),
            &[
                Col::Str("quant_dtype", vec!["fp8", "fp8"]),
                Col::I64("m", vec![128, 256]),
                Col::I64("k", vec![1024, 1024]),
                Col::F64("latency", vec![0.125, 0.375]),
                Col::F64("power", vec![20.0, 60.0]),
            ],
        );
        let db = PerfDatabase::load(tmp.path(), "testsys", "vllm", "1.0").expect("db must load");

        let op = GemmOp {
            name: "g".to_string(),
            scale_factor: 2.0,
            n: 1024,
            k: 1024,
            quant_mode: GemmQuantMode::Fp8Static,
            scale_num_tokens: 1,
            low_precision_input: true,
            seq_split: 1,
            below_grid_sol: false,
        };
        let r = op.query(&db, 192, None).expect("fp8_static query");
        assert!(
            (r.latency_ms - 2.5).abs() < 1e-9,
            "latency {}",
            r.latency_ms
        );
        assert!(
            (r.energy_wms - 520.0).abs() < 1e-9 * 520.0,
            "energy {}",
            r.energy_wms
        );
        assert_eq!(r.source, Source::Estimated);

        let op1 = GemmOp {
            name: "g1".to_string(),
            scale_factor: 1.0,
            n: 1024,
            k: 1024,
            quant_mode: GemmQuantMode::Fp8Static,
            scale_num_tokens: 1,
            low_precision_input: false,
            seq_split: 1,
            below_grid_sol: false,
        };
        let r1 = op1.query(&db, 192, None).expect("fp8_static query");
        assert!(
            (r1.latency_ms - 1.5).abs() < 1e-9,
            "latency {}",
            r1.latency_ms
        );
        assert!(
            (r1.energy_wms - 270.0).abs() < 1e-9 * 270.0,
            "energy {}",
            r1.energy_wms
        );
    }

    /// SOL mode (and the retired SOL_FULL alias) returns the pure roofline
    /// tagged `Source::Sol` with zero energy — Python `_query_gemm_table` /
    /// `_query_compute_scale_table` / `_query_scale_matrix_table` SOL branches.
    #[test]
    fn gemm_sol_mode_returns_roofline_with_sol_source() {
        let mut db = b200_vllm_db();
        db.database_mode = DatabaseMode::Sol;
        let quant = GemmQuantMode::Bfloat16;
        let op = GemmOp::new("gemm", 4096, 4096, quant);
        let result = op.query(&db, 512, None).expect("sol query");
        let tc_flops = quant_tc_flops(&db.system_spec, quant.mapping()).expect("flops");
        let expected =
            gemm_sol_latency_ms_with_flops(&db.system_spec, quant, tc_flops, 512.0, 4096.0, 4096.0);
        assert_eq!(result.latency_ms, expected);
        assert_eq!(result.source, Source::Sol);
        assert_eq!(result.energy_wms, 0.0);

        db.database_mode = DatabaseMode::SolFull;
        let alias = op.query(&db, 512, None).expect("sol_full query");
        assert_eq!(alias.latency_ms, expected);
        assert_eq!(alias.source, Source::Sol);

        // compute_scale / scale_matrix SOL arms: pure mem-bandwidth bounds
        // (`2 m k / bw` and `3 m k / bw`).
        let mem_bw = db.system_spec.gpu.mem_bw;
        let cs = query_compute_scale_table(&db, quant, 512, 4096).expect("cs");
        assert_eq!(cs.latency_ms, 2.0 * 512.0 * 4096.0 / mem_bw * 1000.0);
        assert_eq!(cs.source, Source::Sol);
        let sm = query_scale_matrix_table(&db, quant, 512, 4096).expect("sm");
        assert_eq!(sm.latency_ms, 3.0 * 512.0 * 4096.0 / mem_bw * 1000.0);
        assert_eq!(sm.source, Source::Sol);
    }
}
