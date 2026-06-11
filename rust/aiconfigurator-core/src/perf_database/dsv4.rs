// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 attention module perf tables.
//!
//! Four primary module CSVs distinguished by `(attn_kind, mode)`:
//! - `dsv4_csa_context_module_perf.txt` — CSA (compressed-sparse) context
//! - `dsv4_hca_context_module_perf.txt` — HCA (hybrid-causal) context
//! - `dsv4_csa_generation_module_perf.txt`
//! - `dsv4_hca_generation_module_perf.txt`
//!
//! ## Indexing
//!
//! Mirrors Python `load_context_dsv4_kind_module_data` /
//! `load_generation_dsv4_kind_module_data`. The latency tables are keyed by:
//!   - `native_heads` (the model's total attention head count, CSV `num_heads`
//!     column) — selects the data slice; and
//!   - `tp_size` — the primary interpolation axis.
//!
//! NOT by the per-rank partitioned head count. Context grids interpolate over
//! `(tp_size, isl, batch)`; generation grids over `(tp_size, batch, s_total)`
//! where `s_total = isl + step` (decode is `q_len=1` with `past_kv=step`).
//!
//! All four primary CSVs share the DSA module column layout. Data is
//! collected only on TRT-LLM / SGLang today; loaders surface a clean error for
//! backends without DSV4 data.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use super::interpolation::{interp_2d_1d_grid, Grid3};
use crate::perf_database::parquet_loader::PerfReader;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnKind {
    Csa,
    Hca,
}

// native_heads -> tp_size -> step -> isl -> batch -> latency
type ByBatch = BTreeMap<u32, f64>;
type ByIsl = BTreeMap<u32, ByBatch>;
type ByStep = BTreeMap<u32, ByIsl>;
type ByTp = BTreeMap<u32, ByStep>;
type ByNative = BTreeMap<u32, ByTp>;

pub struct Dsv4Table {
    data_root: PathBuf,
    csa_context: OnceLock<Result<ModuleGrids, AicError>>,
    hca_context: OnceLock<Result<ModuleGrids, AicError>>,
    csa_generation: OnceLock<Result<ModuleGrids, AicError>>,
    hca_generation: OnceLock<Result<ModuleGrids, AicError>>,
}

struct ModuleGrids {
    by_keys: BTreeMap<ModuleKey, ByNative>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleKey {
    architecture: String,
    fmha_quant: String,
    kv_quant: String,
    gemm_quant: String,
}

impl Dsv4Table {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            csa_context: OnceLock::new(),
            hca_context: OnceLock::new(),
            csa_generation: OnceLock::new(),
            hca_generation: OnceLock::new(),
        }
    }

    /// Context-DSV4 latency at `lookup_s = isl` (the new-token count). Mirrors
    /// Python's context base lookup over `(tp_size, isl, b)` on the
    /// `native_heads` slice. The context CSVs collected to date carry a single
    /// `step=0` anchor, so there is no prefix axis to resolve here (the operator
    /// already supplies the new-token count as `isl`).
    #[allow(clippy::too_many_arguments)]
    pub fn query_context(
        &self,
        attn_kind: AttnKind,
        b: u32,
        isl: u32,
        native_heads: u32,
        tp_size: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_context()?,
            AttnKind::Hca => self.load_hca_context()?,
        };
        let by_tp = select_native(grids, architecture, fmha_quant, kv_quant, gemm_quant, native_heads)?;
        // Context grid: [tp_size][isl][batch]. The `step` (prefix) axis is
        // collapsed because the collected context CSVs carry only step=0;
        // Python's prefix-resolved lookup likewise returns the single anchor.
        // Outer = tp_size, middle = isl, inner = batch.
        let mut grid: Grid3<f64> = BTreeMap::new();
        for (&tp, by_step) in by_tp {
            for by_isl in by_step.values() {
                for (&isl_v, by_batch) in by_isl {
                    for (&bb, &lat) in by_batch {
                        grid.entry(tp).or_default().entry(isl_v).or_default().insert(bb, lat);
                    }
                }
            }
        }
        interp_2d_1d_grid(&grid, tp_size, isl, b)
    }

    /// Generation-DSV4 latency. `sequence_tokens = isl + step` (absolute KV
    /// length). Mirrors Python's generation base
    /// `_dsv4_robust_3d_lookup(dict[native_heads], tp_size, b, s_total, batch_axis="y")`.
    #[allow(clippy::too_many_arguments)]
    pub fn query_generation(
        &self,
        attn_kind: AttnKind,
        b: u32,
        sequence_tokens: u32,
        native_heads: u32,
        tp_size: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_generation()?,
            AttnKind::Hca => self.load_hca_generation()?,
        };
        let by_tp = select_native(grids, architecture, fmha_quant, kv_quant, gemm_quant, native_heads)?;
        // Generation grid: [tp_size][batch][s_total] where s_total = isl + step.
        // Batch is the MIDDLE (y) axis and s_total the inner (z) axis, matching
        // Python's generation lookup `_dsv4_robust_3d_lookup(dict, tp, b, s,
        // batch_axis="y")`. This is the opposite layout from `query_context`
        // (batch inner) and from DSA generation — and it is load-bearing: the
        // DSV4 generation table is RAGGED (e.g. s_total=385 lacks batch=16 that
        // s_total=257/513 have). With batch inner, bracketing on s_total then
        // intersecting batch keys collapses to the sparse batch set and
        // extrapolates the wrong value; with batch middle, the exact b row is
        // selected first and s_total interpolates within it (matches Python).
        let mut grid: Grid3<f64> = BTreeMap::new();
        for (&tp, by_step) in by_tp {
            for (&step, by_isl) in by_step {
                for (&isl_v, by_batch) in by_isl {
                    let s_total = isl_v + step;
                    for (&bb, &lat) in by_batch {
                        grid.entry(tp).or_default().entry(bb).or_default().insert(s_total, lat);
                    }
                }
            }
        }
        interp_2d_1d_grid(&grid, tp_size, b, sequence_tokens)
    }

    fn load_csa_context(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.csa_context.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_csa_context_module_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_context(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.hca_context.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_hca_context_module_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_csa_generation(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.csa_generation.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_csa_generation_module_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_generation(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.hca_generation.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_hca_generation_module_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Resolve the `(quant, architecture)` key and select the `native_heads` slice,
/// returning the `tp_size -> step -> isl -> batch` sub-tree.
fn select_native<'a>(
    grids: &'a ModuleGrids,
    architecture: &str,
    fmha: FmhaQuantMode,
    kv: KvCacheQuantMode,
    gemm: GemmQuantMode,
    native_heads: u32,
) -> Result<&'a ByTp, AicError> {
    let key = ModuleKey {
        architecture: architecture.to_string(),
        fmha_quant: fmha.name().to_string(),
        kv_quant: kv.name().to_string(),
        gemm_quant: gemm.name().to_string(),
    };
    let by_native = grids
        .by_keys
        .get(&key)
        .ok_or_else(|| AicError::PerfDatabase(format!("DSV4 module data missing for {key:?}")))?;
    by_native.get(&native_heads).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "DSV4 module data missing for native_heads={native_heads}, {key:?} (loaded native_heads: {:?})",
            by_native.keys().collect::<Vec<_>>()
        ))
    })
}

/// Canonicalize a DSV4 CSV dtype string to the enum `.name()` form.
///
/// Mirrors Python `_dsv4_normalize_dtype` / `_DSV4_DTYPE_ALIASES`: the only
/// alias is `fp8_e4m3` -> `fp8`. Everything else passes through unchanged.
fn normalize_dsv4_dtype(name: &str) -> String {
    match name {
        "fp8_e4m3" => "fp8".to_string(),
        other => other.to_string(),
    }
}

fn load_module_parquet(path: &Path) -> Result<ModuleGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let arch_col = reader.col("architecture")?;
    let mla_dtype_col = reader.col("mla_dtype")?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let gemm_type_col = reader.col("gemm_type")?;
    let num_heads_col = reader.col("num_heads")?;
    // `tp_size` is the primary interpolation axis. Mirror Python's
    // `row.get("tp_size", 1)`: default to 1 when the column is absent.
    let tp_size_col = reader.col_optional("tp_size");
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<ModuleKey, ByNative> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = ModuleKey {
            architecture: row.str_owned(arch_col)?,
            // CSV columns use sglang dtype naming; the query side builds keys
            // from the enum `.name()` (canonical short names). Normalize on
            // load to match Python `_dsv4_normalize_dtype`, which aliases
            // `fp8_e4m3` -> `fp8` for `mla_dtype` (fmha) and `kv_cache_dtype`
            // (kv). `gemm_type` is intentionally left untouched, matching
            // Python (e.g. `fp8_block` is a real value that must pass through).
            fmha_quant: normalize_dsv4_dtype(&row.str_owned(mla_dtype_col)?),
            kv_quant: normalize_dsv4_dtype(&row.str_owned(kv_cache_dtype_col)?),
            gemm_quant: row.str_owned(gemm_type_col)?,
        };
        let tp_size = row.u32_optional(tp_size_col)?.unwrap_or(1);
        // Last-wins parity with Python `load_*_dsv4_kind_module_data`, which
        // assigns `data[...][b] = {...}` per row so a later duplicate row (same
        // full key, e.g. two appended collection runs) overwrites the earlier
        // one. `BTreeMap::insert` overwrites; do NOT use `or_insert` here.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_heads_col)?) // native_heads (CSV num_heads column)
            .or_default()
            .entry(tp_size)
            .or_default()
            .entry(row.u32(step_col)?)
            .or_default()
            .entry(row.u32(isl_col)?)
            .or_default()
            .insert(row.u32(batch_size_col)?, row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSV4 module rows loaded from {}",
            path.display()
        )));
    }
    Ok(ModuleGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv4_data_absent_errors_cleanly() {
        // DSV4 modules aren't collected for vllm/0.19.0; loader must surface
        // a clean error.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0");
        let table = Dsv4Table::new(root);
        let err = table
            .query_context(
                AttnKind::Csa,
                1,
                1024,
                128, // native_heads
                1,   // tp_size
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV4ForCausalLM",
            )
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn normalize_dsv4_dtype_aliases_fp8_e4m3() {
        assert_eq!(normalize_dsv4_dtype("fp8_e4m3"), "fp8");
        // Non-aliased values must pass through unchanged.
        assert_eq!(normalize_dsv4_dtype("bfloat16"), "bfloat16");
        assert_eq!(normalize_dsv4_dtype("fp8_block"), "fp8_block");
        assert_eq!(normalize_dsv4_dtype("fp8"), "fp8");
    }

    #[test]
    fn dsv4_context_resolves_fp8_e4m3_kv_quant() {
        // Regression for the b200_sxm/sglang/0.5.10 DSV4 context lookup: the
        // CSV stores `kv_cache_dtype=fp8_e4m3`, but the query builds the key
        // from `KvCacheQuantMode::Fp8.name()` = "fp8". Without load-side
        // normalization the lookup misses (Rust-only error vs Python success).
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/sglang/0.5.10");
        if !root.join("dsv4_csa_context_module_perf.parquet").exists() {
            // Data files are git-lfs tracked; skip if not materialized.
            return;
        }
        let table = Dsv4Table::new(root);
        // (native_heads=64, tp_size=1, isl=512, batch=8, step=0) are measured
        // grid points in the CSA context table for this entry, gemm=fp8_block.
        let latency = table
            .query_context(
                AttnKind::Csa,
                8,   // batch
                512, // isl
                64,  // native_heads
                1,   // tp_size
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Fp8Block,
                "DeepseekV4ForCausalLM",
            )
            .expect("DSV4 context lookup must resolve fp8_e4m3 kv_cache_dtype as fp8");
        assert!(latency.is_finite() && latency > 0.0, "unexpected latency: {latency}");
    }
}
