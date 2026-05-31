// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (DeepSeek-V3.2 Dynamic Sparse Attention) module perf tables.
//!
//! Two CSVs: `dsa_context_module_perf.txt` and
//! `dsa_generation_module_perf.txt`. Both share columns: model,
//! architecture, mla_dtype, kv_cache_dtype, gemm_type, num_heads,
//! batch_size, isl, tp_size, step, latency.
//!
//! Data is nested by (architecture, mla_dtype, kv_cache_dtype, gemm_type)
//! → num_heads → step → isl → batch_size → latency. The `step` axis is the
//! "prefix value" — some architectures (e.g. GlmMoeDsaForCausalLM) need it
//! exposed; others use a single step=0 slice. The query API returns the
//! raw nested slice at a given (key, prefix) so the operator layer can run
//! the topk-piecewise dispatch.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::interpolation::{interp_2d_1d_grid_extrapolate_inner, Grid3};
use crate::perf_database::parquet_loader::PerfReader;

pub struct DsaTable {
    data_root: PathBuf,
    context: OnceLock<Result<DsaGrids, AicError>>,
    generation: OnceLock<Result<DsaGrids, AicError>>,
    /// Lazy per-`(DsaKey, prefix)` Grid3 caches with the exact shape
    /// `pick_prefix_slice` builds, so warm queries skip the per-call walk
    /// over the full `num_heads × step × isl × batch` nested map.
    context_prefix_grids: OnceLock<Result<ContextPrefixCache, AicError>>,
    /// Lazy per-`DsaKey` Grid3 caches with the exact shape the inline
    /// generation collapse builds (axes: num_heads, seq=isl+step, batch),
    /// so warm queries skip the per-call rebuild.
    generation_grids: OnceLock<Result<GenerationGridCache, AicError>>,
}

struct ContextPrefixCache {
    by_keys: BTreeMap<DsaKey, BTreeMap<u32, Grid3<f64>>>,
}

struct GenerationGridCache {
    by_keys: BTreeMap<DsaKey, Grid3<f64>>,
}

/// (arch, fmha, kv, gemm) → num_heads → step → isl → batch → latency.
pub struct DsaGrids {
    pub by_keys: BTreeMap<DsaKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DsaKey {
    pub architecture: String,
    pub fmha_quant: String,
    pub kv_quant: String,
    pub gemm_quant: String,
}

impl DsaTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            context_prefix_grids: OnceLock::new(),
            generation_grids: OnceLock::new(),
        }
    }

    /// Raw context-DSA module latency at a specific prefix/step value.
    /// 3-D interpolation over (num_heads, isl, batch) within the chosen
    /// (key, prefix) slice.
    pub fn query_context(
        &self,
        b: u32,
        full_seq_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let cache = self.load_context_prefix_grids()?;
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let by_prefix = cache.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!("context DSA module data missing for {key:?}"))
        })?;
        let slice = by_prefix.get(&prefix).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "context DSA module data missing for prefix={prefix}, {key:?}"
            ))
        })?;
        interp_2d_1d_grid_extrapolate_inner(slice, num_heads, full_seq_tokens, b)
    }

    /// Raw generation-DSA module latency. `sequence_tokens = isl + step`
    /// from the CSV; query interpolates over (num_heads, batch, seq).
    pub fn query_generation(
        &self,
        b: u32,
        sequence_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let cache = self.load_generation_grids()?;
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let grid = cache
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("generation DSA module", &self.data_root, format!("{key:?}")))?;
        // Grid axis order: outer = num_heads, middle = seq_tokens, inner = batch.
        // Uses the extrapolating variant because Python pre-extends both
        // axes via `_extrapolate` before interpolation; linear extrapolation
        // here from the boundary pair gives the same numeric result for the
        // out-of-envelope queries that show up on sparser backend tables
        // (e.g. SGLang DSv32 at low num_heads).
        interp_2d_1d_grid_extrapolate_inner(grid, num_heads, sequence_tokens, b)
    }

    /// Return the raw nested context grids for a given key, exposed so the
    /// operator layer can run the topk-piecewise interpolation against the
    /// per-prefix slices when sparse-attention semantics require it.
    pub fn raw_context_grids(&self) -> Result<&DsaGrids, AicError> {
        self.load_context()
    }

    fn load_context(&self) -> Result<&DsaGrids, AicError> {
        let cell = self
            .context
            .get_or_init(|| load_dsa_parquet(&self.data_root.join("dsa_context_module_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&DsaGrids, AicError> {
        let cell = self
            .generation
            .get_or_init(|| load_dsa_parquet(&self.data_root.join("dsa_generation_module_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }

    fn load_context_prefix_grids(&self) -> Result<&ContextPrefixCache, AicError> {
        let cell = self.context_prefix_grids.get_or_init(|| {
            let grids = self.load_context()?;
            Ok(build_context_prefix_cache(grids))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation_grids(&self) -> Result<&GenerationGridCache, AicError> {
        let cell = self.generation_grids.get_or_init(|| {
            let grids = self.load_generation()?;
            Ok(build_generation_grid_cache(grids))
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Materialise the per-`(DsaKey, prefix)` `Grid3` shape that `query_context`
/// needs (axes: num_heads, isl, batch). Exact reproduction of the historical
/// `pick_prefix_slice` loop so numerics are identical.
fn build_context_prefix_cache(grids: &DsaGrids) -> ContextPrefixCache {
    let mut by_keys: BTreeMap<DsaKey, BTreeMap<u32, Grid3<f64>>> = BTreeMap::new();
    for (key, by_heads) in &grids.by_keys {
        let entry = by_keys.entry(key.clone()).or_default();
        for (&n, by_step) in by_heads {
            for (&prefix, by_isl) in by_step {
                let slice = entry.entry(prefix).or_default();
                for (&isl, by_batch) in by_isl {
                    for (&bb, &lat) in by_batch {
                        slice.entry(n).or_default().entry(isl).or_default().insert(bb, lat);
                    }
                }
            }
        }
    }
    ContextPrefixCache { by_keys }
}

/// Materialise the per-`DsaKey` `Grid3` shape that `query_generation` needs
/// (axes: num_heads, seq=isl+step, batch). Iterates the same already-loaded
/// nested map in the same BTreeMap-sorted order, so the last-write-wins
/// semantics on `seq` ties (largest step wins) are identical to the pre-fix
/// per-call rebuild.
fn build_generation_grid_cache(grids: &DsaGrids) -> GenerationGridCache {
    let mut by_keys: BTreeMap<DsaKey, Grid3<f64>> = BTreeMap::new();
    for (key, by_heads) in &grids.by_keys {
        let grid = by_keys.entry(key.clone()).or_default();
        for (&n, by_step) in by_heads {
            for (&step, by_isl) in by_step {
                for (&isl, by_batch) in by_isl {
                    let seq = isl + step;
                    for (&bb, &lat) in by_batch {
                        grid.entry(n).or_default().entry(seq).or_default().insert(bb, lat);
                    }
                }
            }
        }
    }
    GenerationGridCache { by_keys }
}

fn load_dsa_parquet(path: &Path) -> Result<DsaGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let arch_col = reader.col("architecture")?;
    let mla_dtype_col = reader.col("mla_dtype")?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let gemm_type_col = reader.col("gemm_type")?;
    let num_heads_col = reader.col("num_heads")?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<DsaKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>> =
        BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = DsaKey {
            architecture: row.str_owned(arch_col)?,
            fmha_quant: row.str_owned(mla_dtype_col)?,
            kv_quant: row.str_owned(kv_cache_dtype_col)?,
            gemm_quant: row.str_owned(gemm_type_col)?,
        };
        // First-wins parity with Python `load_dsa_module_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_heads_col)?)
            .or_default()
            .entry(row.u32(step_col)?)
            .or_default()
            .entry(row.u32(isl_col)?)
            .or_default()
            .entry(row.u32(batch_size_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSA module rows loaded from {}",
            path.display()
        )));
    }
    Ok(DsaGrids { by_keys })
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

    #[test]
    fn dsa_context_module_exact_hit() {
        // First row of dsa_context_module_perf.txt:
        // arch=DeepseekV32ForCausalLM mla=bfloat16 kv=bfloat16 gemm=bfloat16
        // n=128 b=1 isl=1 step=0 latency=1.0972
        let table = DsaTable::new(b200_vllm_data_root());
        let latency = table
            .query_context(
                1,
                1,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV32ForCausalLM",
                0,
            )
            .expect("DSA context query must succeed");
        assert!(
            (latency - 1.0972).abs() < 1e-6,
            "expected recorded latency, got {latency}"
        );
    }

    #[test]
    fn dsa_unknown_architecture_errors() {
        let table = DsaTable::new(b200_vllm_data_root());
        let err = table
            .query_context(
                1,
                1024,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "NotAnArchitecture",
                0,
            )
            .unwrap_err();
        assert!(matches!(err, AicError::PerfDatabase(_)));
    }
}
