// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MHC (Qwen3.5 multi-head channel) module perf table.
//!
//! CSV columns: model, architecture, num_tokens, hc_mult, hidden_size,
//! latency. Indexed by (architecture, hc_mult, hidden_size) → num_tokens
//! → latency. Query is 1-D interpolation along num_tokens.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::error::AicError;
use super::interpolation::{interp_1d, nearest_neighbors};
use crate::perf_database::parquet_loader::PerfReader;

pub struct MhcTable {
    data_root: PathBuf,
    module: OnceLock<Result<MhcGrids, AicError>>,
}

struct MhcGrids {
    by_keys: BTreeMap<MhcKey, BTreeMap<u32, f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MhcKey {
    architecture: String,
    hc_mult: u32,
    hidden_size: u32,
}

impl MhcTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            module: OnceLock::new(),
        }
    }

    pub fn query_module(
        &self,
        num_tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load()?;
        let key = MhcKey {
            architecture: architecture.to_string(),
            hc_mult,
            hidden_size,
        };
        let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MHC module data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if let Some(&latency) = by_tokens.get(&num_tokens) {
            return Ok(latency);
        }
        let token_keys: Vec<u32> = by_tokens.keys().copied().collect();
        let (lo, hi) = nearest_neighbors(num_tokens, &token_keys, false)?;
        Ok(interp_1d(
            lo as f64,
            hi as f64,
            by_tokens[&lo],
            by_tokens[&hi],
            num_tokens as f64,
        ))
    }

    fn load(&self) -> Result<&MhcGrids, AicError> {
        let cell = self
            .module
            .get_or_init(|| load_mhc_parquet(&self.data_root.join("mhc_module_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }
}

fn load_mhc_parquet(path: &Path) -> Result<MhcGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let arch_col = reader.col("architecture")?;
    let num_tokens_col = reader.col("num_tokens")?;
    let hc_mult_col = reader.col("hc_mult")?;
    let hidden_size_col = reader.col("hidden_size")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<MhcKey, BTreeMap<u32, f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = MhcKey {
            architecture: row.str_owned(arch_col)?,
            hc_mult: row.u32(hc_mult_col)?,
            hidden_size: row.u32(hidden_size_col)?,
        };
        // First-wins parity with Python `load_mhc_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_tokens_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MHC module rows loaded from {}",
            path.display()
        )));
    }
    Ok(MhcGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mhc_absent_on_vllm_b200_errors_clearly() {
        let table = MhcTable::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0"));
        let err = table
            .query_module(1024, 2, 4096, "Qwen3_5MoeForConditionalGeneration")
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
