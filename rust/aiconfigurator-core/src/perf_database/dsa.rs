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
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::interpolation::{interp_2d_1d_grid_extrapolate_inner, Grid3};

pub struct DsaTable {
    data_root: PathBuf,
    context: OnceLock<Result<DsaGrids, AicError>>,
    generation: OnceLock<Result<DsaGrids, AicError>>,
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

#[derive(Debug, Deserialize)]
struct DsaRow {
    architecture: String,
    mla_dtype: String,
    kv_cache_dtype: String,
    gemm_type: String,
    num_heads: u32,
    batch_size: u32,
    isl: u32,
    step: u32,
    latency: f64,
}

impl DsaTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            context: OnceLock::new(),
            generation: OnceLock::new(),
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
        let grids = self.load_context()?;
        let slice = pick_prefix_slice(grids, architecture, fmha_quant, kv_quant, gemm_quant, prefix)?;
        interp_2d_1d_grid_extrapolate_inner(&slice, num_heads, full_seq_tokens, b)
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
        let grids = self.load_generation()?;
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let by_heads = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("generation DSA module", &self.data_root, format!("{key:?}")))?;
        // Generation: collapse `step` into `seq_tokens = isl + step` then 3D-interp.
        let mut grid: Grid3<f64> = BTreeMap::new();
        for (&n, by_step) in by_heads {
            for (&step, by_isl) in by_step {
                for (&isl, by_batch) in by_isl {
                    let seq = isl + step;
                    for (&bb, &lat) in by_batch {
                        grid.entry(n)
                            .or_default()
                            .entry(seq)
                            .or_default()
                            .insert(bb, lat);
                    }
                }
            }
        }
        // Grid axis order: outer = num_heads, middle = seq_tokens, inner = batch.
        // Uses the extrapolating variant because Python pre-extends both
        // axes via `_extrapolate` before interpolation; linear extrapolation
        // here from the boundary pair gives the same numeric result for the
        // out-of-envelope queries that show up on sparser backend tables
        // (e.g. SGLang DSv32 at low num_heads).
        interp_2d_1d_grid_extrapolate_inner(&grid, num_heads, sequence_tokens, b)
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
            .get_or_init(|| load_dsa_csv(&self.data_root.join("dsa_context_module_perf.txt")));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&DsaGrids, AicError> {
        let cell = self
            .generation
            .get_or_init(|| load_dsa_csv(&self.data_root.join("dsa_generation_module_perf.txt")));
        cell.as_ref().map_err(clone_err)
    }
}

fn pick_prefix_slice(
    grids: &DsaGrids,
    architecture: &str,
    fmha: FmhaQuantMode,
    kv: KvCacheQuantMode,
    gemm: GemmQuantMode,
    prefix: u32,
) -> Result<Grid3<f64>, AicError> {
    let key = DsaKey {
        architecture: architecture.to_string(),
        fmha_quant: fmha.name().to_string(),
        kv_quant: kv.name().to_string(),
        gemm_quant: gemm.name().to_string(),
    };
    let by_heads = grids
        .by_keys
        .get(&key)
        .ok_or_else(|| AicError::PerfDatabase(format!("context DSA module data missing for {key:?}")))?;

    let mut slice: Grid3<f64> = BTreeMap::new();
    for (&n, by_step) in by_heads {
        if let Some(by_isl) = by_step.get(&prefix) {
            for (&isl, by_batch) in by_isl {
                for (&bb, &lat) in by_batch {
                    slice.entry(n).or_default().entry(isl).or_default().insert(bb, lat);
                }
            }
        }
    }
    if slice.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "context DSA module data missing for prefix={prefix}, {key:?}"
        )));
    }
    Ok(slice)
}

fn load_dsa_csv(path: &Path) -> Result<DsaGrids, AicError> {
    let text = read_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());
    let mut by_keys: BTreeMap<DsaKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>> =
        BTreeMap::new();
    for record in reader.deserialize::<DsaRow>() {
        let row = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        let key = DsaKey {
            architecture: row.architecture,
            fmha_quant: row.mla_dtype,
            kv_quant: row.kv_cache_dtype,
            gemm_quant: row.gemm_type,
        };
        // First-wins parity with Python `load_dsa_module_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.num_heads)
            .or_default()
            .entry(row.step)
            .or_default()
            .entry(row.isl)
            .or_default()
            .entry(row.batch_size)
            .or_insert(row.latency);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSA module rows loaded from {}",
            path.display()
        )));
    }
    Ok(DsaGrids { by_keys })
}

fn read_perf_file(path: &Path) -> Result<String, AicError> {
    let text = fs::read_to_string(path).map_err(|source| AicError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if text.starts_with("version https://git-lfs") {
        return Err(AicError::PerfDatabase(format!(
            "perf file is an unresolved git-lfs pointer: {}; run `git lfs pull`",
            path.display()
        )));
    }
    Ok(text)
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
