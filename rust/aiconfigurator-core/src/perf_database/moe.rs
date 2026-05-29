// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Basic MoE perf table.
//!
//! Mirrors the raw SILICON-path layout of
//! `aiconfigurator.sdk.operations.moe.MoE._query_moe_table`:
//!
//! `moe_data[quant][distribution][topk][num_experts][hidden][inter][moe_tp][moe_ep]`
//! returns a `{num_tokens -> latency_ms}` dict.
//!
//! Query is 1-D linear interpolation along `num_tokens` (extrapolation
//! allowed). Any SOL-anchored overflow estimation for queries beyond the
//! largest recorded token count is the model layer's responsibility; this
//! raw table query simply 1-D-interpolates with extrapolation when needed.
//!
//! `workload_distribution` falls back to `"uniform"` when the requested
//! variant is absent for the given quant, matching Python's behavior.
//!
//! WideEP / DeepEP / TRT-LLM all-to-all variants live in
//! `perf_database::wideep`, `wideep_mla`, and `wideep_moe`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::interpolation::{interp_1d, nearest_neighbors};

pub struct MoeTable {
    data_root: PathBuf,
    moe: OnceLock<Result<MoeGrids, AicError>>,
}

struct MoeGrids {
    by_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MoeKey {
    quant: String,
    distribution: String,
    topk: u32,
    num_experts: u32,
    hidden_size: u32,
    inter_size: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
}

#[derive(Debug, Deserialize)]
struct MoeRow {
    moe_dtype: String,
    num_tokens: u32,
    hidden_size: u32,
    inter_size: u32,
    topk: u32,
    num_experts: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
    distribution: String,
    latency: f64,
}

impl MoeTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            moe: OnceLock::new(),
        }
    }

    /// Raw MoE latency in ms by 1-D interpolation along `num_tokens`.
    ///
    /// Falls back to the `"uniform"` distribution if the requested
    /// distribution is absent for the given quant mode. Extrapolates beyond
    /// the largest recorded `num_tokens` via linear extension; operator
    /// layer is responsible for SOL-anchored utilization correction for
    /// overflow queries.
    pub fn query(
        &self,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load()?;
        let quant_name = quant.name();

        let dist = self.resolve_distribution(grids, quant_name, workload_distribution);
        let key = MoeKey {
            quant: quant_name.to_string(),
            distribution: dist,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MoE data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;

        // Exact hit.
        if let Some(&latency) = by_tokens.get(&num_tokens) {
            return Ok(latency);
        }
        // 1-D interpolation (extrapolation allowed).
        let token_keys: Vec<u32> = by_tokens.keys().copied().collect();
        if token_keys.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "MoE data has no token points for {key:?} at {}",
                self.data_root.display()
            )));
        }
        let (lo, hi) = nearest_neighbors(num_tokens, &token_keys, false)?;
        Ok(interp_1d(
            lo as f64,
            hi as f64,
            by_tokens[&lo],
            by_tokens[&hi],
            num_tokens as f64,
        ))
    }

    /// Largest recorded `num_tokens` for a given (quant, distribution,
    /// topology) tuple. Used by the operator layer to decide whether to
    /// invoke SOL-anchored overflow estimation.
    pub fn max_token_point(
        &self,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
    ) -> Result<Option<u32>, AicError> {
        let grids = self.load()?;
        let quant_name = quant.name();
        let dist = self.resolve_distribution(grids, quant_name, workload_distribution);
        let key = MoeKey {
            quant: quant_name.to_string(),
            distribution: dist,
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        Ok(grids
            .by_keys
            .get(&key)
            .and_then(|m| m.keys().next_back().copied()))
    }

    /// Mirrors Python's:
    /// `dist = workload if workload in moe_data[quant] else "uniform"`
    fn resolve_distribution(
        &self,
        grids: &MoeGrids,
        quant: &str,
        workload_distribution: &str,
    ) -> String {
        // Check whether any key with this (quant, distribution) exists.
        let requested_exists = grids
            .by_keys
            .keys()
            .any(|k| k.quant == quant && k.distribution == workload_distribution);
        if requested_exists {
            workload_distribution.to_string()
        } else {
            "uniform".to_string()
        }
    }

    fn load(&self) -> Result<&MoeGrids, AicError> {
        let cell = self
            .moe
            .get_or_init(|| load_moe_csv(&self.data_root.join("moe_perf.txt")));
        cell.as_ref().map_err(clone_err)
    }
}

fn load_moe_csv(path: &Path) -> Result<MoeGrids, AicError> {
    let text = read_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());

    let mut by_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>> = BTreeMap::new();
    for record in reader.deserialize::<MoeRow>() {
        let row = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        let key = MoeKey {
            quant: row.moe_dtype.clone(),
            distribution: row.distribution.clone(),
            topk: row.topk,
            num_experts: row.num_experts,
            hidden_size: row.hidden_size,
            inter_size: row.inter_size,
            moe_tp_size: row.moe_tp_size,
            moe_ep_size: row.moe_ep_size,
        };
        // Python's `load_moe_data` wraps the leaf insert in a try/except KeyError
        // and skips on conflict, i.e. it keeps the FIRST occurrence of each
        // (shape, num_tokens) tuple. Some perf files contain duplicate rows
        // (same kernel_source, same shape) — preserving first-wins parity here.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.num_tokens)
            .or_insert(row.latency);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MoE rows loaded from {}",
            path.display()
        )));
    }
    Ok(MoeGrids { by_keys })
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

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0")
    }

    #[test]
    fn moe_table_loads_b200_vllm() {
        let table = MoeTable::new(b200_vllm_data_root());
        let _ = table.load().expect("moe_perf.txt must load");
    }

    #[test]
    fn moe_distribution_falls_back_to_uniform() {
        // Pick any common smoke shape; non-existent distribution should
        // fall back without erroring.
        let table = MoeTable::new(b200_vllm_data_root());
        // Use a shape that's likely covered by vLLM b200 data; if not,
        // the error should be about the topology key, not about
        // missing distribution.
        let result = table.query(
            1024,
            4096,
            2048,
            2,
            128,
            1,
            8,
            MoeQuantMode::Bfloat16,
            "nonexistent_distribution",
        );
        // Either succeeds (uniform fallback found a match) or errors
        // with a topology mismatch — but not a distribution-specific
        // error.
        match result {
            Ok(latency) => assert!(latency > 0.0),
            Err(AicError::PerfDatabase(msg)) => {
                assert!(
                    !msg.contains("nonexistent_distribution"),
                    "expected uniform fallback, not literal distribution name in error: {msg}"
                );
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn moe_lazy_loads_once() {
        let table = MoeTable::new(b200_vllm_data_root());
        // Load twice; cached path should produce same outcome.
        let r1 = table.load();
        let r2 = table.load();
        assert_eq!(r1.is_ok(), r2.is_ok());
    }
}
