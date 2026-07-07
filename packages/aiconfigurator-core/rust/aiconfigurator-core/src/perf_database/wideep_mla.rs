// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang WideEP MLA perf tables (context + generation).
//!
//! Two CSVs with the same column set: `wideep_context_mla_perf.txt` and
//! `wideep_generation_mla_perf.txt`. Columns: framework, version, device,
//! op_name, kernel_source, model, architecture, mla_dtype, kv_cache_dtype,
//! gemm_type, num_heads, batch_size, isl, tp_size, step, latency.
//!
//! Schema-wise the files are nearly identical to the (non-WideEP) MLA
//! module tables, but the nesting in Python's loaders differs:
//!
//! - Context:    `data[kernel_source][fmha/mla_dtype][kv_dtype][num_heads][s][b]`
//! - Generation: `data[kernel_source][kv_dtype][num_heads][b][s = isl + step]`
//!   (Note: generation's `s` collapses `isl + step`, and the `fmha_dtype`
//!   level is absent — generation MLA doesn't tunnel through the fmha
//!   dispatch path the way context does.)
//!
//! Query semantics from Python:
//!
//! - Context: `interp_3d(num_heads, full_s = s + prefix, b)` then multiplied
//!   by `prefix_correction = (full_s^2 - prefix^2) / full_s^2`. The operator
//!   layer applies the correction; the perf-DB query just returns the raw
//!   table value.
//! - Generation: `interp_3d(num_heads, b, s)` — axis order (n, b, s) and no
//!   prefix correction.
//!
//! Python's `_extrapolate` calls `extrapolate_data_grid` with
//! `sqrt_y_value=True` for context (fills the seq axis using a sqrt
//! transform of latency before linear interp). Rust currently relies on
//! linear extrapolation against the original boundary pair, which agrees
//! whenever the per-d_model bucket has a single entry — true for every
//! shipped table at the time of writing.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::{FmhaQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::interpolation::{interp_2d_1d_grid_extrapolate_inner, Grid3};
use crate::perf_database::parquet_loader::PerfReader;

/// Owner for both WideEP MLA tables. Each side is lazily loaded on first
/// query.
pub struct WideEpMlaTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for each WideEP MLA perf file
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`WideEpMlaTable::new`).
    context_sources: Vec<PerfSource>,
    generation_sources: Vec<PerfSource>,
    context: OnceLock<Result<WideEpContextMlaGrids, AicError>>,
    generation: OnceLock<Result<WideEpGenerationMlaGrids, AicError>>,
}

/// Context grids keyed by `(kernel_source, fmha_quant, kv_quant)`.
/// Inner `Grid3` axes: outer = num_heads, middle = s, inner = b.
pub struct WideEpContextMlaGrids {
    pub by_keys: BTreeMap<ContextKey, Grid3<f64>>,
}

/// Generation grids keyed by `(kernel_source, kv_quant)`. Inner `Grid3`
/// axes: outer = num_heads, middle = b, inner = s. The `s` axis here is
/// `isl + step` from the CSV (Python collapses them at load time).
pub struct WideEpGenerationMlaGrids {
    pub by_keys: BTreeMap<GenerationKey, Grid3<f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContextKey {
    pub kernel_source: String,
    pub fmha_quant: String,
    pub kv_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GenerationKey {
    pub kernel_source: String,
    pub kv_quant: String,
}

impl WideEpMlaTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each WideEP MLA file falls back to
    /// its primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let context_sources =
            resolve_op_sources(perf_db_sources, "wideep_context_mla_perf.parquet", &data_root);
        let generation_sources = resolve_op_sources(
            perf_db_sources,
            "wideep_generation_mla_perf.parquet",
            &data_root,
        );
        Self {
            data_root,
            context_sources,
            generation_sources,
            context: OnceLock::new(),
            generation: OnceLock::new(),
        }
    }

    /// Raw context WideEP MLA latency. Caller is responsible for applying
    /// the `prefix_correction = (full_s^2 - prefix^2) / full_s^2`
    /// multiplier; this matches the (non-WideEP) `MlaTable::query_context`
    /// split.
    pub fn query_context(
        &self,
        b: u32,
        full_seq_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        kernel_source: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            kernel_source: kernel_source.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let grid = grids.by_keys.get(&key).ok_or_else(|| {
            missing(
                "WideEP context MLA",
                &self.data_root,
                format!("{key:?}"),
            )
        })?;
        interp_2d_1d_grid_extrapolate_inner(grid, num_heads, full_seq_tokens, b)
    }

    /// Raw generation WideEP MLA latency. `sequence_tokens` is the
    /// pre-collapsed `isl + step` (matching Python's `s = s + step` in
    /// the loader).
    pub fn query_generation(
        &self,
        b: u32,
        sequence_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        kernel_source: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load_generation()?;
        let key = GenerationKey {
            kernel_source: kernel_source.to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let grid = grids.by_keys.get(&key).ok_or_else(|| {
            missing(
                "WideEP generation MLA",
                &self.data_root,
                format!("{key:?}"),
            )
        })?;
        // Python's generation query is interp_3d(num_heads, b, s) — middle
        // axis is batch, inner axis is sequence tokens. Grid is built with
        // that nesting on load.
        interp_2d_1d_grid_extrapolate_inner(grid, num_heads, b, sequence_tokens)
    }

    fn load_context(&self) -> Result<&WideEpContextMlaGrids, AicError> {
        let cell = self
            .context
            .get_or_init(|| load_context_parquet(&self.context_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&WideEpGenerationMlaGrids, AicError> {
        let cell = self
            .generation
            .get_or_init(|| load_generation_parquet(&self.generation_sources));
        cell.as_ref().map_err(clone_err)
    }
}

/// Load the WideEP context MLA table from an ordered, priority-sorted source
/// list. Sources are read in order; the first source containing a shape wins
/// (`or_insert`), mirroring Python's `_read_filtered_rows` concatenation +
/// `load_wideep_context_mla_data` skip-on-key-conflict. Missing files are
/// skipped (a sibling declared in the manifest need not exist for every
/// system); an error is returned only when no source yields rows.
fn load_context_parquet(sources: &[PerfSource]) -> Result<WideEpContextMlaGrids, AicError> {
    let mut by_keys: BTreeMap<ContextKey, Grid3<f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let mla_dtype_col = reader.col("mla_dtype")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let num_heads_col = reader.col("num_heads")?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = ContextKey {
                kernel_source: row.str_owned(kernel_source_col)?,
                fmha_quant: row.str_owned(mla_dtype_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
            };
            // First-wins parity with Python `load_wideep_context_mla_data`,
            // extended across shared-layer sources (earlier source wins).
            by_keys
                .entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(isl_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no WideEP context MLA rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(WideEpContextMlaGrids { by_keys })
}

/// Load the WideEP generation MLA table from an ordered source list. Same
/// first-wins-across-sources + missing-file-skip semantics as
/// [`load_context_parquet`].
fn load_generation_parquet(sources: &[PerfSource]) -> Result<WideEpGenerationMlaGrids, AicError> {
    let mut by_keys: BTreeMap<GenerationKey, Grid3<f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let num_heads_col = reader.col("num_heads")?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let step_col = reader.col("step")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = GenerationKey {
                kernel_source: row.str_owned(kernel_source_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
            };
            // Python collapses `s = isl + step` into the seq axis.
            let seq = row.u32(isl_col)? + row.u32(step_col)?;
            // First-wins parity, extended across shared-layer sources.
            by_keys
                .entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_default()
                .entry(seq)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no WideEP generation MLA rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(WideEpGenerationMlaGrids { by_keys })
}

fn missing(table: &str, data_root: &Path, descriptor: String) -> AicError {
    AicError::PerfDatabase(format!(
        "{table} data missing for {descriptor} at {}",
        data_root.display()
    ))
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_sglang_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/sglang/0.5.10")
    }

    #[test]
    fn wideep_context_mla_exact_hit() {
        // First DSv3 row in b200_sxm/sglang/0.5.10 wideep_context_mla_perf.txt:
        // kernel=trtllm_mla mla=fp8_block kv=fp8 num_heads=128 b=1 isl=1 latency=0.5470
        let table = WideEpMlaTable::new(b200_sglang_data_root());
        let latency = table
            .query_context(
                1,
                1,
                128,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Fp8Block,
                "trtllm_mla",
            )
            .expect("WideEP context MLA query must succeed");
        assert!(
            (latency - 0.5470).abs() < 1e-3,
            "expected recorded latency, got {latency}"
        );
    }

    #[test]
    fn wideep_generation_mla_exact_hit() {
        // First DSv3 row in b200_sxm/sglang/0.5.10 wideep_generation_mla_perf.txt:
        // kernel=trtllm_mla kv=fp8 num_heads=128 b=1 isl=1 step=0 latency=0.1049
        let table = WideEpMlaTable::new(b200_sglang_data_root());
        let latency = table
            .query_generation(1, 1, 128, KvCacheQuantMode::Fp8, "trtllm_mla")
            .expect("WideEP generation MLA query must succeed");
        assert!(
            (latency - 0.1049).abs() < 1e-3,
            "expected recorded latency, got {latency}"
        );
    }
}
