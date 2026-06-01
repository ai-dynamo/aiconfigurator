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
//! Plus two refinement CSVs retained for prefix-aware corrections of the
//! HCA path: `dsv4_paged_mqa_logits_module_perf.txt` and
//! `dsv4_hca_attn_module_perf.txt`.
//!
//! All four primary CSVs share the DSA module column layout. Data is
//! collected only on TRT-LLM today; loaders surface a clean error for
//! backends without DSV4 data.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::interpolation::{interp_2d_1d_grid, Grid3};
use crate::perf_database::parquet_loader::PerfReader;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKind {
    Csa,
    Hca,
}

pub struct Dsv4Table {
    data_root: PathBuf,
    csa_context: OnceLock<Result<ModuleGrids, AicError>>,
    hca_context: OnceLock<Result<ModuleGrids, AicError>>,
    csa_generation: OnceLock<Result<ModuleGrids, AicError>>,
    hca_generation: OnceLock<Result<ModuleGrids, AicError>>,
}

struct ModuleGrids {
    by_keys: BTreeMap<ModuleKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>>,
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

    /// Raw context-DSV4 latency at a specific prefix value.
    pub fn query_context(
        &self,
        attn_kind: AttnKind,
        b: u32,
        full_seq_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_context()?,
            AttnKind::Hca => self.load_hca_context()?,
        };
        let slice = prefix_slice(grids, architecture, fmha_quant, kv_quant, gemm_quant, prefix)?;
        interp_2d_1d_grid(&slice, num_heads, full_seq_tokens, b)
    }

    /// Raw generation-DSV4 latency. `sequence_tokens = isl + step` from CSV.
    pub fn query_generation(
        &self,
        attn_kind: AttnKind,
        b: u32,
        sequence_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_generation()?,
            AttnKind::Hca => self.load_hca_generation()?,
        };
        let key = ModuleKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let by_heads = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("DSV4 generation module", &self.data_root, format!("{key:?}")))?;
        let mut grid: Grid3<f64> = BTreeMap::new();
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
        // Axis order must match the grid: outer = num_heads, middle =
        // seq_tokens, inner = batch_size. Mirrors `dsa.rs` query_generation.
        interp_2d_1d_grid(&grid, num_heads, sequence_tokens, b)
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

fn prefix_slice(
    grids: &ModuleGrids,
    architecture: &str,
    fmha: FmhaQuantMode,
    kv: KvCacheQuantMode,
    gemm: GemmQuantMode,
    prefix: u32,
) -> Result<Grid3<f64>, AicError> {
    let key = ModuleKey {
        architecture: architecture.to_string(),
        fmha_quant: fmha.name().to_string(),
        kv_quant: kv.name().to_string(),
        gemm_quant: gemm.name().to_string(),
    };
    let by_heads = grids
        .by_keys
        .get(&key)
        .ok_or_else(|| AicError::PerfDatabase(format!("DSV4 context module data missing for {key:?}")))?;
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
            "DSV4 context module data missing for prefix={prefix}, {key:?}"
        )));
    }
    Ok(slice)
}

fn load_module_parquet(path: &Path) -> Result<ModuleGrids, AicError> {
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

    let mut by_keys: BTreeMap<ModuleKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>> =
        BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = ModuleKey {
            architecture: row.str_owned(arch_col)?,
            fmha_quant: row.str_owned(mla_dtype_col)?,
            kv_quant: row.str_owned(kv_cache_dtype_col)?,
            gemm_quant: row.str_owned(gemm_type_col)?,
        };
        // First-wins parity with Python `load_dsv4_module_data`.
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
            "no DSV4 module rows loaded from {}",
            path.display()
        )));
    }
    Ok(ModuleGrids { by_keys })
}

fn missing(table: &str, data_root: &Path, descriptor: String) -> AicError {
    AicError::PerfDatabase(format!("{table} data missing for {descriptor} at {}", data_root.display()))
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv4_data_absent_errors_cleanly() {
        // DSV4 modules aren't collected anywhere yet; loader must surface
        // a clean error.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0");
        let table = Dsv4Table::new(root);
        let err = table
            .query_context(
                AttnKind::Csa,
                1,
                1024,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV4ForCausalLM",
                0,
            )
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
