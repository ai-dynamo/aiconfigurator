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
use crate::interpolation::{interp_1d, interp_2d_1d_grid, nearest_neighbors, Grid3};
use crate::perf_database::parquet_loader::PerfReader;

/// GEMM-family perf-data owner for one `<system>/<backend>/<version>` slice.
///
/// Holds the data directory and three lazy CSV-loaded tables. Construct via
/// `GemmTable::new`; queries trigger the relevant table's load on first use.
pub struct GemmTable {
    data_root: PathBuf,
    gemm: OnceLock<Result<GemmGrids, AicError>>,
    compute_scale: OnceLock<Result<TwoDGrids, AicError>>,
    scale_matrix: OnceLock<Result<TwoDGrids, AicError>>,
}

/// 3-D GEMM tables keyed by quant name -> m -> n -> k -> latency_ms.
struct GemmGrids {
    by_quant: BTreeMap<String, Grid3<f64>>,
}

/// 2-D scale tables keyed by quant name -> m -> k -> latency_ms.
struct TwoDGrids {
    by_quant: BTreeMap<String, BTreeMap<u32, BTreeMap<u32, f64>>>,
}

impl GemmTable {
    /// Construct an empty table for the given data directory. No I/O.
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            gemm: OnceLock::new(),
            compute_scale: OnceLock::new(),
            scale_matrix: OnceLock::new(),
        }
    }

    /// Query GEMM latency (ms) for the given shape and quant mode.
    ///
    /// Mirrors the SILICON path of `GEMM._query_gemm_table`:
    /// 1. Exact hit on `(m, n, k)` if present.
    /// 2. 1-D interpolation along `m` when `(n, k)` exists in two or more
    ///    `m` slices.
    /// 3. 3-D fallback via `interp_2d_1d_grid` (bilinear over `(n, k)`,
    ///    linear over `m`).
    pub fn query(&self, quant: GemmQuantMode, m: u32, n: u32, k: u32) -> Result<f64, AicError> {
        let grids = self.load_gemm()?;
        let quant_name = quant.name();
        let grid = grids.by_quant.get(quant_name).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "GEMM perf data missing for quant '{quant_name}' at {}; available: {:?}",
                self.data_root.display(),
                grids.by_quant.keys().collect::<Vec<_>>(),
            ))
        })?;

        // Exact hit.
        if let Some(by_n) = grid.get(&m) {
            if let Some(by_k) = by_n.get(&n) {
                if let Some(&latency) = by_k.get(&k) {
                    return Ok(latency);
                }
            }
        }

        // 1-D interpolation along m when (n, k) lives in >=2 slices.
        let m_with_nk: Vec<u32> = grid
            .iter()
            .filter_map(|(&mv, by_n)| {
                by_n.get(&n)
                    .and_then(|by_k| by_k.get(&k))
                    .map(|_| mv)
            })
            .collect();
        if m_with_nk.len() >= 2 {
            let (m_left, m_right) = nearest_neighbors(m, &m_with_nk, false)?;
            let y_left = grid[&m_left][&n][&k];
            let y_right = grid[&m_right][&n][&k];
            return Ok(interp_1d(
                m_left as f64,
                m_right as f64,
                y_left,
                y_right,
                m as f64,
            ));
        }

        // 3-D fallback.
        interp_2d_1d_grid(grid, m, n, k)
    }

    /// Query compute-scale latency (ms) — used by `fp8_static` GEMM only.
    pub fn query_compute_scale(
        &self,
        quant: GemmQuantMode,
        m: u32,
        k: u32,
    ) -> Result<f64, AicError> {
        let grids = self.load_compute_scale()?;
        query_two_d(&grids.by_quant, quant.name(), m, k, &self.data_root)
    }

    /// Query scale-matrix latency (ms) — used by `fp8_static` GEMM only.
    pub fn query_scale_matrix(
        &self,
        quant: GemmQuantMode,
        m: u32,
        k: u32,
    ) -> Result<f64, AicError> {
        let grids = self.load_scale_matrix()?;
        query_two_d(&grids.by_quant, quant.name(), m, k, &self.data_root)
    }

    fn load_gemm(&self) -> Result<&GemmGrids, AicError> {
        let cell = self
            .gemm
            .get_or_init(|| load_gemm_parquet(&self.data_root.join("gemm_perf.parquet")));
        cell.as_ref().map_err(|err| clone_err(err))
    }

    fn load_compute_scale(&self) -> Result<&TwoDGrids, AicError> {
        let cell = self.compute_scale.get_or_init(|| {
            load_two_d_parquet(&self.data_root.join("computescale_perf.parquet"))
        });
        cell.as_ref().map_err(|err| clone_err(err))
    }

    fn load_scale_matrix(&self) -> Result<&TwoDGrids, AicError> {
        let cell = self.scale_matrix.get_or_init(|| {
            load_two_d_parquet(&self.data_root.join("scale_matrix_perf.parquet"))
        });
        cell.as_ref().map_err(|err| clone_err(err))
    }
}

fn query_two_d(
    by_quant: &BTreeMap<String, BTreeMap<u32, BTreeMap<u32, f64>>>,
    quant_name: &str,
    m: u32,
    k: u32,
    data_root: &Path,
) -> Result<f64, AicError> {
    let grid = by_quant.get(quant_name).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "perf data missing for quant '{quant_name}' at {}; available: {:?}",
            data_root.display(),
            by_quant.keys().collect::<Vec<_>>(),
        ))
    })?;
    if let Some(by_k) = grid.get(&m) {
        if let Some(&latency) = by_k.get(&k) {
            return Ok(latency);
        }
    }
    // 1-D interpolation along m when k matches in >=2 slices.
    let m_with_k: Vec<u32> = grid
        .iter()
        .filter_map(|(&mv, by_k)| by_k.get(&k).map(|_| mv))
        .collect();
    if m_with_k.len() >= 2 {
        let (m_left, m_right) = nearest_neighbors(m, &m_with_k, false)?;
        return Ok(interp_1d(
            m_left as f64,
            m_right as f64,
            grid[&m_left][&k],
            grid[&m_right][&k],
            m as f64,
        ));
    }
    // Bilinear fallback over (m, k).
    let m_keys: Vec<u32> = grid.keys().copied().collect();
    let (m_left, m_right) = nearest_neighbors(m, &m_keys, true)?;
    let left_row = grid.get(&m_left).unwrap();
    let right_row = grid.get(&m_right).unwrap();
    let k_keys: Vec<u32> = left_row
        .keys()
        .copied()
        .filter(|kv| right_row.contains_key(kv))
        .collect();
    let (k_left, k_right) = nearest_neighbors(k, &k_keys, true)?;
    let q00 = left_row[&k_left];
    let q01 = left_row[&k_right];
    let q10 = right_row[&k_left];
    let q11 = right_row[&k_right];
    if m_left == m_right && k_left == k_right {
        return Ok(q00);
    }
    if m_left == m_right {
        return Ok(interp_1d(k_left as f64, k_right as f64, q00, q01, k as f64));
    }
    if k_left == k_right {
        return Ok(interp_1d(m_left as f64, m_right as f64, q00, q10, m as f64));
    }
    Ok(crate::interpolation::bilinear(
        m_left as f64,
        m_right as f64,
        k_left as f64,
        k_right as f64,
        q00,
        q01,
        q10,
        q11,
        m as f64,
        k as f64,
    ))
}

fn load_gemm_parquet(path: &Path) -> Result<GemmGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let gemm_dtype_col = reader.col("gemm_dtype")?;
    let m_col = reader.col("m")?;
    let n_col = reader.col("n")?;
    let k_col = reader.col("k")?;
    let latency_col = reader.col("latency")?;

    let mut by_quant: BTreeMap<String, Grid3<f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let dtype = row.str(gemm_dtype_col)?;
        // Skip quant modes AIC does not model in the perf path (matches the
        // legacy perf.rs behavior).
        if dtype == "awq" || dtype == "gptq" {
            continue;
        }
        let dtype = dtype.to_string();
        // First-wins parity with Python's `load_gemm_data` try/except KeyError.
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
    if by_quant.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no GEMM rows loaded from {}",
            path.display()
        )));
    }
    Ok(GemmGrids { by_quant })
}

fn load_two_d_parquet(path: &Path) -> Result<TwoDGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let quant_dtype_col = reader.col("quant_dtype")?;
    let m_col = reader.col("m")?;
    let k_col = reader.col("k")?;
    let latency_col = reader.col("latency")?;

    let mut by_quant: BTreeMap<String, BTreeMap<u32, BTreeMap<u32, f64>>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        // First-wins parity (compute_scale / scale_matrix tables in Python).
        by_quant
            .entry(row.str_owned(quant_dtype_col)?)
            .or_default()
            .entry(row.u32(m_col)?)
            .or_default()
            .entry(row.u32(k_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_quant.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no rows loaded from {}",
            path.display()
        )));
    }
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
            .join("src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0")
    }

    #[test]
    fn gemm_exact_hit_returns_recorded_latency() {
        let table = GemmTable::new(b200_vllm_data_root());
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
        let table = GemmTable::new(b200_vllm_data_root());
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
        let table = GemmTable::new(b200_vllm_data_root());
        let first = table.query(GemmQuantMode::Bfloat16, 32768, 65536, 16384).unwrap();
        let second = table.query(GemmQuantMode::Bfloat16, 32768, 65536, 16384).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn gemm_missing_quant_mode_errors() {
        let table = GemmTable::new(b200_vllm_data_root());
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
        let table = GemmTable::new(PathBuf::from("/nonexistent/aic/data/root"));
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
        let table = GemmTable::new(b200_vllm_data_root());
        let err = table
            .query_compute_scale(GemmQuantMode::Fp8Static, 1024, 4096)
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}
