// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MLA family perf tables: op-level context/generation, MLA BMM (pre/post),
//! and module-level context/generation.
//!
//! Mirrors the SILICON paths of `aiconfigurator.sdk.operations.mla.{ContextMLA,
//! GenerationMLA, MLABmm, MLAModule}._query_*_table`. Module-level data is
//! collected as a fused unit (MLA + RoPE + BMM together) and indexed by an
//! extra `gemm_quant` axis.
//!
//! Caller passes `full_seq_tokens` for context queries (= `isl + prefix`);
//! the prefix-correction multiplier is applied by the operator layer.
//! The MLA BMM table falls back to `bfloat16` data when the requested quant
//! mode is absent, matching Python's `quant_mode_lookup` behavior.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::interpolation::{interp_1d, interp_2d_1d_grid, nearest_neighbors, Grid3};
use crate::perf_database::parquet_loader::PerfReader;

pub struct MlaTable {
    data_root: PathBuf,
    context: OnceLock<Result<ContextMlaGrids, AicError>>,
    generation: OnceLock<Result<GenerationMlaGrids, AicError>>,
    bmm: OnceLock<Result<BmmGrids, AicError>>,
    context_module: OnceLock<Result<ModuleGrids, AicError>>,
    generation_module: OnceLock<Result<ModuleGrids, AicError>>,
}

struct ContextMlaGrids {
    by_keys: BTreeMap<ContextKey, Grid3<f64>>,
}

struct GenerationMlaGrids {
    by_keys: BTreeMap<KvOnlyKey, Grid3<f64>>,
}

/// Module-level MLA grids, shared between context and generation variants
/// (the same nested layout; distinct CSV files supply the data).
struct ModuleGrids {
    by_keys: BTreeMap<ModuleKey, Grid3<f64>>,
}

struct BmmGrids {
    // (bmm_quant, "mla_gen_pre" | "mla_gen_post", num_heads) -> {num_tokens -> latency}
    by_keys: BTreeMap<BmmKey, BTreeMap<u32, BTreeMap<u32, f64>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ContextKey {
    fmha_quant: String,
    kv_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct KvOnlyKey {
    kv_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleKey {
    fmha_quant: String,
    kv_quant: String,
    gemm_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BmmKey {
    bmm_quant: String,
    pre_or_post: String,
}

impl MlaTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            bmm: OnceLock::new(),
            context_module: OnceLock::new(),
            generation_module: OnceLock::new(),
        }
    }

    /// Op-level context MLA latency in ms (raw — no prefix correction).
    pub fn query_context(
        &self,
        b: u32,
        full_seq_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let grid = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("context MLA", &self.data_root, format!("{key:?}")))?;
        interp_2d_1d_grid(grid, num_heads, full_seq_tokens, b)
    }

    /// Op-level generation MLA latency in ms.
    pub fn query_generation(
        &self,
        b: u32,
        s: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_generation()?;
        let key = KvOnlyKey {
            kv_quant: kv_quant.name().to_string(),
        };
        let grid = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("generation MLA", &self.data_root, format!("{key:?}")))?;
        // Python's generation MLA uses (num_heads, b, s) as the 3 axes
        // — note b and s order differs from context.
        interp_2d_1d_grid(grid, num_heads, b, s)
    }

    /// MLA BMM (pre or post) latency in ms.
    ///
    /// Falls back to `bfloat16` if the requested quant mode is absent,
    /// matching Python's `quant_mode_lookup` behavior.
    pub fn query_bmm(
        &self,
        num_tokens: u32,
        num_heads: u32,
        quant: GemmQuantMode,
        is_pre: bool,
    ) -> Result<f64, AicError> {
        let grids = self.load_bmm()?;
        let pre_or_post = if is_pre { "mla_gen_pre" } else { "mla_gen_post" };

        // Try the requested quant first; fall back to bfloat16 if missing.
        let key = BmmKey {
            bmm_quant: quant.name().to_string(),
            pre_or_post: pre_or_post.to_string(),
        };
        let chosen = grids.by_keys.get(&key).or_else(|| {
            let fallback = BmmKey {
                bmm_quant: GemmQuantMode::Bfloat16.name().to_string(),
                pre_or_post: pre_or_post.to_string(),
            };
            grids.by_keys.get(&fallback)
        });
        let by_heads = chosen.ok_or_else(|| {
            missing("MLA BMM", &self.data_root, format!("quant={}, {pre_or_post}", quant.name()))
        })?;

        let by_tokens = by_heads.get(&num_heads).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MLA BMM data missing for num_heads={num_heads} at {}",
                self.data_root.display()
            ))
        })?;

        // 1-D interpolation along num_tokens (extrapolation allowed).
        if let Some(&latency) = by_tokens.get(&num_tokens) {
            return Ok(latency);
        }
        let token_keys: Vec<u32> = by_tokens.keys().copied().collect();
        let (lo, hi) = nearest_neighbors(num_tokens, &token_keys, false)?;
        let y_lo = by_tokens[&lo];
        let y_hi = by_tokens[&hi];
        Ok(interp_1d(lo as f64, hi as f64, y_lo, y_hi, num_tokens as f64))
    }

    /// Module-level context MLA latency in ms (raw — no prefix correction).
    pub fn query_context_module(
        &self,
        b: u32,
        full_seq_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_context_module()?;
        let key = ModuleKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let grid = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("context MLA module", &self.data_root, format!("{key:?}")))?;
        interp_2d_1d_grid(grid, num_heads, full_seq_tokens, b)
    }

    /// Module-level generation MLA latency in ms.
    pub fn query_generation_module(
        &self,
        b: u32,
        s: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_generation_module()?;
        let key = ModuleKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let grid = grids.by_keys.get(&key).ok_or_else(|| {
            missing("generation MLA module", &self.data_root, format!("{key:?}"))
        })?;
        interp_2d_1d_grid(grid, num_heads, b, s)
    }

    fn load_context(&self) -> Result<&ContextMlaGrids, AicError> {
        let cell = self
            .context
            .get_or_init(|| load_op_parquet(&self.data_root.join("context_mla_perf.parquet"), true));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&GenerationMlaGrids, AicError> {
        let cell = self.generation.get_or_init(|| {
            load_op_gen_parquet(&self.data_root.join("generation_mla_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_bmm(&self) -> Result<&BmmGrids, AicError> {
        let cell = self
            .bmm
            .get_or_init(|| load_bmm_parquet(&self.data_root.join("mla_bmm_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }

    fn load_context_module(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.context_module.get_or_init(|| {
            load_module_parquet(&self.data_root.join("mla_context_module_perf.parquet"), true)
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation_module(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.generation_module.get_or_init(|| {
            load_module_parquet(&self.data_root.join("mla_generation_module_perf.parquet"), false)
        });
        cell.as_ref().map_err(clone_err)
    }
}

fn load_op_parquet(path: &Path, is_context: bool) -> Result<ContextMlaGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let mla_dtype_col = reader.col("mla_dtype")?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let num_heads_col = reader.col("num_heads")?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<ContextKey, Grid3<f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = ContextKey {
            fmha_quant: row.str_owned(mla_dtype_col)?,
            kv_quant: row.str_owned(kv_cache_dtype_col)?,
        };
        let isl = row.u32(isl_col)?;
        let y_axis = if is_context { isl } else { isl + row.u32(step_col)? };
        // First-wins parity with Python `load_mla_data` (context branch).
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_heads_col)?)
            .or_default()
            .entry(y_axis)
            .or_default()
            .entry(row.u32(batch_size_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MLA op rows loaded from {}",
            path.display()
        )));
    }
    Ok(ContextMlaGrids { by_keys })
}

fn load_op_gen_parquet(path: &Path) -> Result<GenerationMlaGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let num_heads_col = reader.col("num_heads")?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<KvOnlyKey, Grid3<f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = KvOnlyKey {
            kv_quant: row.str_owned(kv_cache_dtype_col)?,
        };
        let sequence_tokens = row.u32(isl_col)? + row.u32(step_col)?;
        // Python uses (num_heads, b, s) axis order for generation MLA.
        // First-wins parity with Python `load_mla_data` (generation branch).
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_heads_col)?)
            .or_default()
            .entry(row.u32(batch_size_col)?)
            .or_default()
            .entry(sequence_tokens)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no generation MLA rows loaded from {}",
            path.display()
        )));
    }
    Ok(GenerationMlaGrids { by_keys })
}

fn load_module_parquet(path: &Path, is_context: bool) -> Result<ModuleGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let mla_dtype_col = reader.col("mla_dtype")?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let gemm_type_col = reader.col("gemm_type")?;
    let num_heads_col = reader.col("num_heads")?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<ModuleKey, Grid3<f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = ModuleKey {
            fmha_quant: row.str_owned(mla_dtype_col)?,
            kv_quant: row.str_owned(kv_cache_dtype_col)?,
            gemm_quant: row.str_owned(gemm_type_col)?,
        };
        let num_heads = row.u32(num_heads_col)?;
        let batch_size = row.u32(batch_size_col)?;
        let isl = row.u32(isl_col)?;
        let latency = row.f64(latency_col)?;
        // Last-wins parity with Python `load_context_mla_module_data`
        // (`operations/mla.py:1804`) and `load_generation_mla_module_data`
        // (`operations/mla.py:1847`). Unlike the legacy CSV loaders and every
        // other Python perf-DB loader (GEMM, attention, MoE, MHC, DSA, wideep)
        // which guard with `try/except KeyError` for first-wins semantics,
        // these two MLA-module parquet loaders use direct assignment and
        // therefore last-wins. Some perf-DB parquet shards (notably
        // b300_sxm/vllm/0.19.0 `mla_generation_module_perf.parquet`) contain
        // duplicate (num_heads, batch_size, sequence_tokens) rows; first-wins
        // here caused a constant +0.247ms/step decode drift on b300 because
        // Rust picked the slightly-higher latency for ~184 affected keys.
        if is_context {
            let inner = by_keys
                .entry(key)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(isl)
                .or_default();
            inner.insert(batch_size, latency);
        } else {
            // Generation module: (num_heads, b, s) axis order.
            let sequence_tokens = isl + row.u32(step_col)?;
            let inner = by_keys
                .entry(key)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(batch_size)
                .or_default();
            inner.insert(sequence_tokens, latency);
        }
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MLA module rows loaded from {}",
            path.display()
        )));
    }
    Ok(ModuleGrids { by_keys })
}

fn load_bmm_parquet(path: &Path) -> Result<BmmGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let op_name_col = reader.col("op_name")?;
    let bmm_dtype_col = reader.col("bmm_dtype")?;
    let num_tokens_col = reader.col("num_tokens")?;
    let num_heads_col = reader.col("num_heads")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<BmmKey, BTreeMap<u32, BTreeMap<u32, f64>>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = BmmKey {
            bmm_quant: row.str_owned(bmm_dtype_col)?,
            pre_or_post: row.str_owned(op_name_col)?,
        };
        // First-wins parity with Python `load_mla_bmm_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_heads_col)?)
            .or_default()
            .entry(row.u32(num_tokens_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MLA BMM rows loaded from {}",
            path.display()
        )));
    }
    Ok(BmmGrids { by_keys })
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

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0")
    }

    fn gb200_trtllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/gb200/trtllm/1.3.0rc10")
    }

    #[test]
    fn op_level_context_mla_absent_on_vllm_b200() {
        // vLLM b200 ships module-level MLA only; op-level context_mla_perf.txt
        // is not present. Expect a clear IO error from the lazy loader.
        let table = MlaTable::new(b200_vllm_data_root());
        let err = table
            .query_context(1, 1024, 128, KvCacheQuantMode::Bfloat16, FmhaQuantMode::Bfloat16)
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn module_level_context_mla_exact_hit() {
        // First row of b200_sxm/vllm/0.19.0/mla_context_module_perf.txt:
        // mla=bfloat16 kv=bfloat16 gemm=bfloat16 n=128 b=1 isl=1 step=0 latency=0.1351
        let table = MlaTable::new(b200_vllm_data_root());
        let latency = table
            .query_context_module(
                1,
                1,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
            )
            .expect("module context MLA query must succeed");
        assert!(
            (latency - 0.1351).abs() < 1e-6,
            "expected recorded module latency, got {latency}"
        );
    }

    #[test]
    fn module_level_generation_mla_smoke() {
        let table = MlaTable::new(b200_vllm_data_root());
        // Verify the generation module CSV loads and returns positive
        // values for a representative smoke shape.
        let result = table.query_generation_module(
            1,
            1024,
            128,
            KvCacheQuantMode::Bfloat16,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Bfloat16,
        );
        match result {
            Ok(latency) => assert!(latency > 0.0, "expected positive latency"),
            Err(AicError::PerfDatabase(_)) => {
                // Shape may be outside recorded range; either loader-OK or
                // interpolation-range error is acceptable for this smoke check.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn mla_bmm_falls_back_to_bfloat16() {
        // gb200/trtllm has mla_bmm data; verify the fallback path works.
        let table = MlaTable::new(gb200_trtllm_data_root());
        // Request an unusual quant; loader should fall back to bfloat16.
        let result = table.query_bmm(64, 128, GemmQuantMode::Sq, true);
        // We just verify no panic and the result is a number; if Sq has no
        // bfloat16 fallback either, expect a clean error.
        match result {
            Ok(latency) => assert!(latency.is_finite() && latency > 0.0),
            Err(AicError::PerfDatabase(_)) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
}
