// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MHC (Qwen3.5 multi-head channel) module perf table.
//!
//! CSV columns: model, architecture, num_tokens, hc_mult, hidden_size,
//! latency. Indexed by (architecture, hc_mult, hidden_size) → num_tokens
//! → latency. Query is 1-D interpolation along num_tokens.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::common::error::AicError;
use crate::interpolation::{interp_1d, nearest_neighbors};

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

#[derive(Debug, Deserialize)]
struct MhcRow {
    architecture: String,
    num_tokens: u32,
    hc_mult: u32,
    hidden_size: u32,
    latency: f64,
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
            .get_or_init(|| load_mhc_csv(&self.data_root.join("mhc_module_perf.txt")));
        cell.as_ref().map_err(clone_err)
    }
}

fn load_mhc_csv(path: &Path) -> Result<MhcGrids, AicError> {
    let text = read_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());
    let mut by_keys: BTreeMap<MhcKey, BTreeMap<u32, f64>> = BTreeMap::new();
    for record in reader.deserialize::<MhcRow>() {
        let row = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        let key = MhcKey {
            architecture: row.architecture,
            hc_mult: row.hc_mult,
            hidden_size: row.hidden_size,
        };
        // First-wins parity with Python `load_mhc_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.num_tokens)
            .or_insert(row.latency);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MHC module rows loaded from {}",
            path.display()
        )));
    }
    Ok(MhcGrids { by_keys })
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
