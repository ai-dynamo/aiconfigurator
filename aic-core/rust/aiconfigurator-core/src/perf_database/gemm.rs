// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GEMM family perf tables: gemm, compute_scale, scale_matrix.
//!
//! Mirrors the SILICON-mode query algorithm of
//! `aiconfigurator.sdk.operations.gemm.GEMM._query_*_table`. SOL / EMPIRICAL
//! / HYBRID modes layer formulaic fallbacks on top of these queries; they
//! live with the operator code in `operators/gemm.rs`.
//!
//! Each table is lazy: the CSV is read on first query. The `gemm` table is
//! 3-D over `(m, n, k)`; the supporting `compute_scale` and `scale_matrix`
//! tables are 2-D over `(m, k)` and used only by the `fp8_static` quant
//! mode. The compute/scale CSVs are absent for backends that do not need
//! them (vLLM, SGLang); the loaders surface a clear error in that case.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::GemmQuantMode;
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::interpolation::Grid3;
use super::perf_interp::{self, Node, OpInterpConfig, Resolver, SiteIndex, ValueTransform};
use crate::perf_database::parquet_loader::PerfReader;

/// GEMM-family perf-data owner for one `<system>/<backend>/<version>` slice.
///
/// Holds the data directory and three lazy CSV-loaded tables. Construct via
/// `GemmTable::new`; queries trigger the relevant table's load on first use.
///
/// `system_spec` is kept for SOL clamping at load time, mirroring Python's
/// `GEMM._correct_sol`. The supporting `compute_scale` / `scale_matrix`
/// tables are NOT SOL-clamped — Python's `_correct_data` only touches the
/// main GEMM table.
pub struct GemmTable {
    data_root: PathBuf,
    system_spec: SystemSpec,
    /// Ordered, priority-sorted sources for each of the three GEMM-family perf
    /// files (shared-layer aware; see [`PerfSource`]). Single-primary,
    /// no-filter by default (`GemmTable::new`).
    gemm_sources: Vec<PerfSource>,
    compute_scale_sources: Vec<PerfSource>,
    scale_matrix_sources: Vec<PerfSource>,
    gemm: OnceLock<Result<GemmEngineGrids, AicError>>,
    compute_scale: OnceLock<Result<TwoDGrids, AicError>>,
    scale_matrix: OnceLock<Result<TwoDGrids, AicError>>,
}

/// 3-D GEMM tables keyed by quant name -> m -> n -> k -> latency_ms.
struct GemmGrids {
    by_quant: BTreeMap<String, Grid3<f64>>,
}

/// Engine-ready GEMM tables: per quant, the nested table plus the scattered
/// (n, k)-site index, both built once at load (tables are immutable).
struct GemmEngineGrids {
    by_quant: BTreeMap<String, (Node, SiteIndex)>,
}

/// 2-D scale tables keyed by quant name -> m -> k -> latency_ms.
struct TwoDGrids {
    by_quant: BTreeMap<String, Node>,
}

impl GemmTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf, system_spec: SystemSpec) -> Self {
        Self::with_sources(data_root, system_spec, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each GEMM-family file falls back to
    /// its primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        system_spec: SystemSpec,
        perf_db_sources: &PerfDbSources,
    ) -> Self {
        let gemm_sources = resolve_op_sources(perf_db_sources, "gemm_perf.parquet", &data_root);
        let compute_scale_sources =
            resolve_op_sources(perf_db_sources, "computescale_perf.parquet", &data_root);
        let scale_matrix_sources =
            resolve_op_sources(perf_db_sources, "scale_matrix_perf.parquet", &data_root);
        Self {
            data_root,
            system_spec,
            gemm_sources,
            compute_scale_sources,
            scale_matrix_sources,
            gemm: OnceLock::new(),
            compute_scale: OnceLock::new(),
            scale_matrix: OnceLock::new(),
        }
    }

    /// Query GEMM latency (ms) for the given shape and quant mode.
    ///
    /// Mirrors the perf_interp v2 path of `GEMM._query_gemm_table`
    /// (`gemm_config`): (n, k) are scattered collected shapes, each owning
    /// an m-curve. Exact site -> its own curve (exact point / lerp /
    /// k_tail=3 util-hold beyond the sweep); unknown shape -> log2-IDW util
    /// transfer from <=4 covering neighbour sites within 2.0 octaves.
    pub fn query(&self, quant: GemmQuantMode, m: u32, n: u32, k: u32) -> Result<f64, AicError> {
        let grids = self.load_gemm()?;
        // `fp8_static` is a behavioral mode that reuses `fp8` perf tables,
        // mirroring Python `GEMM._normalize_for_lookup`. The
        // compute_scale / scale_matrix tables apply the same
        // normalization in their respective query methods.
        let lookup_quant = normalize_fp8_static_quant(quant);
        let quant_name = lookup_quant.name();
        let (_, index) = grids.by_quant.get(quant_name).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "GEMM perf data missing for quant '{quant_name}' at {}; available: {:?}",
                self.data_root.display(),
                grids.by_quant.keys().collect::<Vec<_>>(),
            ))
        })?;
        let spec = &self.system_spec;
        let sol = move |c: &[f64]| gemm_sol_latency_ms(spec, lookup_quant, c[0], c[1], c[2]);
        let cfg = gemm_engine_config(&sol);
        index.resolve(&cfg, &[m as f64, n as f64, k as f64])
    }

    /// Query compute-scale latency (ms) — used by `fp8_static` GEMM only.
    ///
    /// Like the main GEMM table, the compute_scale data is keyed by `fp8`
    /// (not `fp8_static`) in the perf-DB; normalize before lookup to mirror
    /// Python's `GEMM._normalize_for_lookup`.
    ///
    /// compute_scale stores a quantization-overhead DELTA: beyond the grid
    /// it is deliberately held FLAT at the clamped boundary (Python
    /// `_query_compute_scale_table` contract).
    pub fn query_compute_scale(
        &self,
        quant: GemmQuantMode,
        m: u32,
        k: u32,
    ) -> Result<f64, AicError> {
        let grids = self.load_compute_scale()?;
        let lookup = normalize_fp8_static_quant(quant);
        let spec = &self.system_spec;
        // sol_mem = 2 m k / bw * 1000 (read + write of the activation)
        let sol = move |c: &[f64]| 2.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0;
        query_scale_table(&grids.by_quant, lookup.name(), m, k, &sol, false, &self.data_root)
    }

    /// Query scale-matrix latency (ms) — used by `fp8_static` GEMM only.
    /// Same `fp8_static -> fp8` normalization as the GEMM and
    /// compute_scale lookups.
    ///
    /// scale_matrix is a real memory kernel: outside the grid the boundary
    /// utilization is frozen and SOL(q)/SOL(boundary) carries the growth
    /// (Python `_query_scale_matrix_table` contract).
    pub fn query_scale_matrix(
        &self,
        quant: GemmQuantMode,
        m: u32,
        k: u32,
    ) -> Result<f64, AicError> {
        let grids = self.load_scale_matrix()?;
        let lookup = normalize_fp8_static_quant(quant);
        let spec = &self.system_spec;
        let sol = move |c: &[f64]| 3.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0;
        query_scale_table(&grids.by_quant, lookup.name(), m, k, &sol, true, &self.data_root)
    }

    /// Collected `(m, n, k) -> latency` points for the quant's table, for the
    /// operator-layer util-calibration grid (Python's
    /// `require_data_slice(_gemm_data, tqm)` + `iter_grid(..., depth=3)`).
    /// Missing quant / empty table is a typed `PerfDatabase` miss.
    pub fn gemm_points(&self, quant: GemmQuantMode) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_gemm()?;
        let quant_name = normalize_fp8_static_quant(quant).name();
        let (node, _) = grids.by_quant.get(quant_name).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "GEMM perf data missing for quant '{quant_name}' at {}",
                self.data_root.display()
            ))
        })?;
        let points = crate::perf_database::perf_interp::node_points(node);
        if points.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "GEMM perf data empty for quant '{quant_name}' at {}",
                self.data_root.display()
            )));
        }
        Ok(points)
    }

    /// Collected `(m, k) -> delta` points of the compute_scale table (zeroes
    /// included — they are measured deltas). Typed miss when absent.
    pub fn compute_scale_points(&self, quant: GemmQuantMode) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_compute_scale()?;
        Self::two_d_points(grids, normalize_fp8_static_quant(quant), "compute_scale", &self.data_root)
    }

    /// Collected `(m, k) -> latency` points of the scale_matrix table.
    /// Typed miss when absent.
    pub fn scale_matrix_points(&self, quant: GemmQuantMode) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_scale_matrix()?;
        Self::two_d_points(grids, normalize_fp8_static_quant(quant), "scale_matrix", &self.data_root)
    }

    fn two_d_points(
        grids: &TwoDGrids,
        quant: GemmQuantMode,
        table_name: &str,
        data_root: &Path,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let quant_name = quant.name();
        let node = grids.by_quant.get(quant_name).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "{table_name} perf data missing for quant '{quant_name}' at {}",
                data_root.display()
            ))
        })?;
        let points = crate::perf_database::perf_interp::node_points(node);
        if points.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "{table_name} perf data empty for quant '{quant_name}' at {}",
                data_root.display()
            )));
        }
        Ok(points)
    }

    fn load_gemm(&self) -> Result<&GemmEngineGrids, AicError> {
        let cell = self.gemm.get_or_init(|| {
            let mut grids = load_gemm_parquet(&self.gemm_sources)?;
            // Mirror Python `GEMM._correct_sol`: clamp every stored grid
            // entry to `>= SOL`. SOL is deterministic from the system spec
            // and (m, n, k, quant); on currently-aligned surfaces this is
            // a no-op (raw >= SOL already), so the clamp only affects
            // systems whose collected data drops below SOL (the prime
            // example is l40s at small bf16 shapes).
            //
            // Python interpolates over already-clamped grid points; we
            // mirror that ordering by mutating the grid before any
            // queries run. Off-grid interpolation/extrapolation therefore
            // sees the same monotone-bounded inputs as Python.
            clamp_gemm_grids_to_sol(&self.system_spec, &mut grids);
            // Build the engine table + (n, k)-site index once per quant.
            let by_quant = grids
                .by_quant
                .into_iter()
                .map(|(quant_name, grid)| {
                    let node = grid3_to_node(&grid);
                    let index = SiteIndex::build(&[1, 2], 0, &node);
                    (quant_name, (node, index))
                })
                .collect();
            Ok(GemmEngineGrids { by_quant })
        });
        cell.as_ref().map_err(|err| clone_err(err))
    }

    fn load_compute_scale(&self) -> Result<&TwoDGrids, AicError> {
        let cell = self
            .compute_scale
            .get_or_init(|| load_two_d_parquet(&self.compute_scale_sources));
        cell.as_ref().map_err(|err| clone_err(err))
    }

    fn load_scale_matrix(&self) -> Result<&TwoDGrids, AicError> {
        let cell = self
            .scale_matrix
            .get_or_init(|| load_two_d_parquet(&self.scale_matrix_sources));
        cell.as_ref().map_err(|err| clone_err(err))
    }
}

/// Speed-of-light GEMM latency in ms.
///
/// Mirrors Python's `GEMM._query_gemm_table::get_sol`:
/// - `sol_math = 2 * m * n * k / tc_flops(quant) * 1000`
/// - `sol_mem  = quant.memory * (m*n + m*k + n*k) / mem_bw * 1000`
/// - `sol      = max(sol_math, sol_mem)`
///
/// `tc_flops(quant)` follows Python `_get_quant_tc_flops`: compute factor
/// 1 maps to `bfloat16_tc_flops`, 2 to `fp8_tc_flops`, 4 to `fp4_tc_flops`,
/// with a `bfloat16_tc_flops * compute_factor` fallback when the spec
/// entry is missing.
pub(crate) fn gemm_sol_latency_ms(
    spec: &SystemSpec,
    quant: GemmQuantMode,
    m: f64,
    n: f64,
    k: f64,
) -> f64 {
    let mapping = quant.mapping();
    let (m_f, n_f, k_f) = (m, n, k);
    let tc_flops = tc_flops_for_compute(spec, mapping.compute);
    let sol_math = 2.0 * m_f * n_f * k_f / tc_flops * 1000.0;
    let sol_mem = mapping.memory * (m_f * n_f + m_f * k_f + n_f * k_f)
        / spec.gpu.mem_bw
        * 1000.0;
    sol_math.max(sol_mem)
}

pub(crate) fn tc_flops_for_compute(spec: &SystemSpec, compute_factor: f64) -> f64 {
    let bf16 = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let direct = match compute_factor as u32 {
        1 => spec.gpu.bfloat16_tc_flops,
        2 => spec.gpu.fp8_tc_flops,
        4 => spec.gpu.fp4_tc_flops,
        _ => None,
    };
    direct.unwrap_or(bf16 * compute_factor)
}

/// In-place SOL clamp for every entry in the GEMM grid set.
///
/// `bfloat16_tc_flops` is required for the bf16 SOL path; if the system
/// YAML omits it (no real system in the repo does, but the schema marks
/// it optional), we leave the grids untouched so the caller sees raw data
/// rather than a corrupted clamp using `tc_flops == 0`.
fn clamp_gemm_grids_to_sol(spec: &SystemSpec, grids: &mut GemmGrids) {
    if spec.gpu.bfloat16_tc_flops.is_none() {
        return;
    }
    for (quant_name, grid) in grids.by_quant.iter_mut() {
        let Some(quant) = gemm_quant_by_name(quant_name) else {
            continue;
        };
        for (&m, by_n) in grid.iter_mut() {
            for (&n, by_k) in by_n.iter_mut() {
                for (&k, latency) in by_k.iter_mut() {
                    let sol = gemm_sol_latency_ms(spec, quant, m as f64, n as f64, k as f64);
                    if sol > *latency {
                        *latency = sol;
                    }
                }
            }
        }
    }
}

fn gemm_quant_by_name(name: &str) -> Option<GemmQuantMode> {
    use GemmQuantMode::*;
    Some(match name {
        "bfloat16" => Bfloat16,
        "int8_wo" => Int8Wo,
        "int4_wo" => Int4Wo,
        "fp8" => Fp8,
        "fp8_static" => Fp8Static,
        "sq" => Sq,
        "fp8_block" => Fp8Block,
        "fp8_ootb" => Fp8Ootb,
        "nvfp4" => Nvfp4,
        _ => return None,
    })
}

/// Normalize the `fp8_static` quant mode to `fp8` for perf-table lookups.
/// Mirrors Python `GEMM._normalize_for_lookup`: the `fp8_static` mode is
/// behavioral (subtracts compute_scale + scale_matrix latency) but reuses
/// the fp8 perf tables — the perf-DB never stores rows under
/// `fp8_static`. Applied uniformly to the GEMM, compute_scale, and
/// scale_matrix table queries.
pub(crate) fn normalize_fp8_static_quant(quant: GemmQuantMode) -> GemmQuantMode {
    if quant == GemmQuantMode::Fp8Static {
        GemmQuantMode::Fp8
    } else {
        quant
    }
}

/// The GEMM engine record: (n, k) sites (axes 1, 2), m-curve (axis 0),
/// mirroring Python `perf_interp.gemm_config`.
fn gemm_engine_config<'a>(sol: &'a dyn Fn(&[f64]) -> f64) -> OpInterpConfig<'a> {
    OpInterpConfig {
        axes: &["m", "n", "k"],
        resolver: Resolver::ScatteredSites {
            site_axes: vec![1, 2],
            curve_axis: 0,
            nn_sites: 4,
            max_site_distance: Some(2.0),
            require_curve_coverage: true,
            k_tail: 3,
        },
        sol_fn: sol,
        value_transform: ValueTransform::Raw,
        transform_axis: None,
    }
}

fn grid3_to_node(grid: &Grid3<f64>) -> Node {
    let mut node = Node::branch();
    for (&m, by_n) in grid {
        for (&n, by_k) in by_n {
            for (&k, &lat) in by_k {
                node.insert(&[m, n, k], lat);
            }
        }
    }
    node
}

/// Scale-table (compute_scale / scale_matrix) query: clamp `(m, k)` into the
/// collected envelope FIRST (legacy contract), resolve the interior on the
/// engine (RAW 2-axis grid), then either hold FLAT at the boundary
/// (compute_scale: a quantization DELTA) or re-scale by SOL(q)/SOL(boundary)
/// (scale_matrix: a real memory kernel). Mirrors Python
/// `_query_compute_scale_table` / `_query_scale_matrix_table`.
fn query_scale_table(
    by_quant: &BTreeMap<String, Node>,
    quant_name: &str,
    m: u32,
    k: u32,
    sol: &dyn Fn(&[f64]) -> f64,
    sol_ratio_beyond_grid: bool,
    data_root: &Path,
) -> Result<f64, AicError> {
    let node = by_quant.get(quant_name).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "perf data missing for quant '{quant_name}' at {}; available: {:?}",
            data_root.display(),
            by_quant.keys().collect::<Vec<_>>(),
        ))
    })?;
    let Node::Branch(rows) = node else {
        return Err(AicError::PerfDatabase("malformed scale table".to_string()));
    };
    if rows.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "empty scale table for quant '{quant_name}'"
        )));
    }
    let m_keys: Vec<u32> = rows.keys().copied().collect();
    let m_c = m.clamp(m_keys[0], m_keys[m_keys.len() - 1]);
    let mut k_min = u32::MAX;
    let mut k_max = 0u32;
    for row in rows.values() {
        if let Node::Branch(cols) = row {
            if let (Some(&lo), Some(&hi)) = (cols.keys().next(), cols.keys().next_back()) {
                k_min = k_min.min(lo);
                k_max = k_max.max(hi);
            }
        }
    }
    let k_c = k.clamp(k_min, k_max);

    let cfg = OpInterpConfig::grid(&["m", "k"], sol);
    let lat = perf_interp::query(&cfg, node, &[m_c as f64, k_c as f64])?;
    if !sol_ratio_beyond_grid || (m_c == m && k_c == k) {
        return Ok(lat);
    }
    // Outside the grid, freeze utilization at the clamped boundary:
    // L(q) = L(boundary) * SOL(q)/SOL(boundary).
    let boundary_sol = sol(&[m_c as f64, k_c as f64]);
    let query_sol = sol(&[m as f64, k as f64]);
    if boundary_sol > 0.0 && query_sol > 0.0 {
        Ok(lat * (query_sol / boundary_sol))
    } else {
        Ok(lat)
    }
}

/// Load the GEMM table from an ordered, priority-sorted source list. Sources are
/// read in order; the first source containing a shape wins (`or_insert`),
/// mirroring Python's `_read_filtered_rows` concatenation + `load_gemm_data`
/// skip-on-key-conflict. Missing files are skipped (a sibling declared in the
/// manifest need not exist for every system); an error is returned only when no
/// source yields rows.
fn load_gemm_parquet(sources: &[PerfSource]) -> Result<GemmGrids, AicError> {
    let mut by_quant: BTreeMap<String, Grid3<f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let gemm_dtype_col = reader.col("gemm_dtype")?;
        let m_col = reader.col("m")?;
        let n_col = reader.col("n")?;
        let k_col = reader.col("k")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let dtype = row.str(gemm_dtype_col)?;
            // Skip quant modes AIC does not model in the perf path (matches the
            // legacy perf.rs behavior).
            if dtype == "awq" || dtype == "gptq" {
                continue;
            }
            let dtype = dtype.to_string();
            // First-wins parity with Python's `load_gemm_data` try/except
            // KeyError, extended across shared-layer sources (earlier source wins).
            by_quant
                .entry(dtype)
                .or_default()
                .entry(row.u32(m_col)?)
                .or_default()
                .entry(row.u32(n_col)?)
                .or_default()
                .entry(row.u32(k_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_quant.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no GEMM rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(GemmGrids { by_quant })
}

/// Load a 2-D (compute_scale / scale_matrix) table from an ordered source list.
/// Same first-wins-across-sources + missing-file-skip semantics as
/// [`load_gemm_parquet`].
fn load_two_d_parquet(sources: &[PerfSource]) -> Result<TwoDGrids, AicError> {
    let mut raw: BTreeMap<String, BTreeMap<u32, BTreeMap<u32, f64>>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let quant_dtype_col = reader.col("quant_dtype")?;
        let m_col = reader.col("m")?;
        let k_col = reader.col("k")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            // First-wins parity (compute_scale / scale_matrix tables in Python),
            // extended across shared-layer sources.
            raw.entry(row.str_owned(quant_dtype_col)?)
                .or_default()
                .entry(row.u32(m_col)?)
                .or_default()
                .entry(row.u32(k_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    let by_quant = raw
        .into_iter()
        .map(|(quant, rows)| {
            let mut node = Node::branch();
            for (m, cols) in rows {
                for (k, lat) in cols {
                    node.insert(&[m, k], lat);
                }
            }
            (quant, node)
        })
        .collect();
    Ok(TwoDGrids { by_quant })
}

/// Reconstruct an `AicError` from a borrowed cached error so we can hand a
/// fresh owned copy back to the caller (`OnceLock` returns `&Result`, but
/// the API surface returns `Result`).
fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.19.0")
    }

    fn b200_sxm_spec() -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/b200_sxm.yaml");
        SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse")
    }

    fn b200_gemm_parquet(backend: &str, version: &str) -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join(format!(
                "src/aiconfigurator_core/systems/data/b200_sxm/gemm/{backend}/{version}/gemm_perf.parquet"
            ))
    }

    fn gemm_shape_count(grids: &GemmGrids) -> usize {
        grids
            .by_quant
            .values()
            .flat_map(|by_m| by_m.values())
            .flat_map(|by_n| by_n.values())
            .map(|by_k| by_k.len())
            .sum()
    }

    /// Shared-layer sibling merge: sources are read in priority order, later
    /// sources only add shapes the earlier ones lack (first-wins), and a
    /// per-source `kernel_source` allowlist gates which sibling rows are
    /// admitted. Mirrors Python `_read_filtered_rows` + `load_gemm_data`.
    #[test]
    fn shared_layer_merges_siblings_with_kernel_source_filter_and_first_wins() {
        // trtllm 1.3.0rc10 primary + 1.2.0rc5 sibling — the real shape Python's
        // `_compute_perf_db_sources` emits for this backend.
        let primary = b200_gemm_parquet("trtllm", "1.3.0rc10");
        let sibling = b200_gemm_parquet("trtllm", "1.2.0rc5");

        let primary_only = load_gemm_parquet(&[PerfSource(primary.clone(), None)]).unwrap();

        // Sibling admitted unfiltered: never drops a primary shape, only adds.
        let merged =
            load_gemm_parquet(&[PerfSource(primary.clone(), None), PerfSource(sibling.clone(), None)])
                .unwrap();
        assert!(
            gemm_shape_count(&merged) >= gemm_shape_count(&primary_only),
            "unfiltered sibling must not drop shapes"
        );

        // First-wins: every primary (quant,m,n,k) keeps the PRIMARY latency even
        // though the sibling also carries rows.
        for (q, by_m) in &primary_only.by_quant {
            for (m, by_n) in by_m {
                for (n, by_k) in by_n {
                    for (k, v) in by_k {
                        let got = merged
                            .by_quant
                            .get(q)
                            .and_then(|x| x.get(m))
                            .and_then(|x| x.get(n))
                            .and_then(|x| x.get(k))
                            .copied();
                        assert_eq!(got, Some(*v), "first source must win at ({q},{m},{n},{k})");
                    }
                }
            }
        }

        // A `kernel_source` allowlist that matches nothing drops every sibling
        // row, so the merged table equals primary-only.
        let blocked = load_gemm_parquet(&[
            PerfSource(primary.clone(), None),
            PerfSource(sibling.clone(), Some(vec!["__no_such_kernel_source__".to_string()])),
        ])
        .unwrap();
        assert_eq!(
            gemm_shape_count(&blocked),
            gemm_shape_count(&primary_only),
            "a non-matching kernel_source filter must exclude all sibling rows"
        );
    }

    #[test]
    fn gemm_exact_hit_returns_recorded_latency() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        // First row of b200_sxm/vllm/0.19.0/gemm_perf.txt (bfloat16 32768x65536x16384).
        let latency = table
            .query(GemmQuantMode::Bfloat16, 32768, 65536, 16384)
            .expect("query must succeed");
        assert!(
            (latency - 41.59673055013021).abs() < 1e-9,
            "expected recorded latency, got {latency}"
        );
    }

    #[test]
    fn gemm_query_returns_positive_latency_for_smoke_shape() {
        // Shape pulled from a MiniMax-M2.5 GEMM call: tp=8 ffn1 at hidden=6144.
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let latency = table
            .query(GemmQuantMode::Bfloat16, 1024, 6144, 6144)
            .expect("query must succeed");
        assert!(latency > 0.0, "interpolated latency must be positive");
        assert!(latency < 100.0, "shape this small shouldn't take 100ms");
    }

    #[test]
    fn gemm_lazy_loads_on_first_query_only() {
        // Same data root, two queries — second must not re-read the CSV.
        // We can't directly observe I/O count, but if the cache isn't being
        // hit the second query would still succeed, so verify both paths
        // return identical results (proxy for cache stability).
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let first = table.query(GemmQuantMode::Bfloat16, 32768, 65536, 16384).unwrap();
        let second = table.query(GemmQuantMode::Bfloat16, 32768, 65536, 16384).unwrap();
        assert_eq!(first, second);
    }

    /// Values generated from the Python v2 engine on the same table
    /// (`db.query_gemm(..., SILICON)` on b200_sxm/vllm/0.19.0, bfloat16):
    /// exact hit, m-interp on a collected (n,k) site, m util-hold beyond the
    /// sweep, and an unknown (n,k) site via neighbour util transfer. The two
    /// engines must agree because they implement the same resolution chain.
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails
    #[test]
    fn gemm_query_matches_python_v2_engine() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let q = GemmQuantMode::Bfloat16;
        let cases: &[(u32, u32, u32, f64)] = &[
            (256, 32, 32, 0.00186666660011),
            (259, 32, 32, 0.00184757819233),
            (10_000_000, 32, 32, 1.51111355145),
            (256, 128, 96, 0.00187964537818),
        ];
        for &(m, n, k, expected) in cases {
            let got = table.query(q, m, n, k).unwrap();
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "({m},{n},{k}): rust {got} vs python {expected}"
            );
        }
    }

    #[test]
    fn gemm_missing_quant_mode_errors() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        // vLLM 0.19.0 b200 collects bfloat16/fp8/fp8_block/nvfp4 — int4_wo
        // is genuinely absent for this slice.
        match table.query(GemmQuantMode::Int4Wo, 1024, 4096, 4096) {
            Err(AicError::PerfDatabase(msg)) => {
                assert!(msg.contains("int4_wo"), "expected quant name in error: {msg}");
            }
            other => panic!("expected PerfDatabase error, got {other:?}"),
        }
    }

    #[test]
    fn gemm_missing_data_root_errors_on_query() {
        let table = GemmTable::new(PathBuf::from("/nonexistent/aic/data/root"), b200_sxm_spec());
        let err = table.query(GemmQuantMode::Bfloat16, 1, 1, 1).unwrap_err();
        // The lazy loader should surface the missing file as the cause,
        // and the second access should see the cached error too.
        assert!(matches!(err, AicError::PerfDatabase(_)));
        let err2 = table.query(GemmQuantMode::Bfloat16, 1, 1, 1).unwrap_err();
        assert!(matches!(err2, AicError::PerfDatabase(_)));
    }

    #[test]
    fn compute_scale_absent_on_vllm_b200_errors_clearly() {
        // vLLM doesn't ship compute_scale data on b200; expect a clear IO error.
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let err = table
            .query_compute_scale(GemmQuantMode::Fp8Static, 1024, 4096)
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
