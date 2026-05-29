// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TensorRT-LLM WideEP MoE *compute* perf table.
//!
//! `wideep_moe_perf.txt` (Python `PerfDataFilename.wideep_moe_compute`).
//! Pure-compute kernel timing (no All2All) for the WideEP execution
//! path. The dispatch / combine cost is modeled separately by the
//! `wideep` (DeepEP / TRT-LLM All2All) table.
//!
//! CSV columns: framework, version, device, op_name, kernel_source,
//! moe_dtype, moe_kernel, num_tokens, dp_num_tokens, rank0_num_tokens,
//! hidden_size, inter_size, topk, num_experts, num_slots, moe_tp_size,
//! moe_ep_size, distribution, simulation_mode, latency.
//!
//! Loader nesting mirrors Python's
//! `data[kernel_source][quant][distribution][topk][num_experts][hidden]`
//! `[inter][num_slots][moe_tp_size][moe_ep_size][num_tokens] = latency`.
//! At query time the leaf `num_tokens` axis is 1-D interpolated.
//!
//! `kernel_source` identifies the WideEP MoE compute kernel:
//!   - `moe_torch_flow` (Cutlass; default for SM < 100)
//!   - `deepgemm` (SM >= 100 with fp8_block)
//! `distribution` carries the workload-distribution string used by the
//! `MoEModel`/`MoeOp`, e.g. `power_law_1.01` or `power_law_1.01_eplb`
//! (the `_eplb` suffix selects the Expert Parallel Load Balancer
//! variants used by the TRT-LLM WideEP path).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::interpolation::{interp_1d, nearest_neighbors};

pub struct WideEpMoeTable {
    data_root: PathBuf,
    compute: OnceLock<Result<WideEpMoeGrids, AicError>>,
}

/// `(kernel_source, quant, distribution, topk, num_experts, hidden, inter,
///   num_slots, moe_tp, moe_ep)` -> `num_tokens -> latency`.
pub struct WideEpMoeGrids {
    pub by_keys: BTreeMap<WideEpMoeKey, BTreeMap<u32, f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct WideEpMoeKey {
    pub kernel_source: String,
    pub quant: String,
    pub distribution: String,
    pub topk: u32,
    pub num_experts: u32,
    pub hidden_size: u32,
    pub inter_size: u32,
    pub num_slots: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
}

#[derive(Debug, Deserialize)]
struct WideEpMoeRow {
    // `kernel_source` is optional in the CSV (defaults to "moe_torch_flow"
    // when absent, per Python's `load_wideep_moe_compute_data`). serde's
    // `default` handles that branch.
    #[serde(default = "default_kernel_source")]
    kernel_source: String,
    moe_dtype: String,
    num_tokens: u32,
    hidden_size: u32,
    inter_size: u32,
    topk: u32,
    num_experts: u32,
    num_slots: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
    distribution: String,
    latency: f64,
}

fn default_kernel_source() -> String {
    "moe_torch_flow".to_string()
}

impl WideEpMoeTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            compute: OnceLock::new(),
        }
    }

    /// Query WideEP MoE compute latency at `num_tokens` along the
    /// `(kernel_source, quant, distribution, topk, num_experts, hidden,
    /// inter, num_slots, moe_tp_size, moe_ep_size)` key. Mirrors Python's
    /// `_query_compute_table`: nearest-neighbour-with-clamp + linear
    /// interp on the token axis. If the exact `distribution` isn't in the
    /// table, falls back to the first distribution available under the
    /// matched quant — same as Python.
    #[allow(clippy::too_many_arguments)]
    pub fn query_compute(
        &self,
        num_tokens: u32,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        num_slots: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        distribution: &str,
        kernel_source: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load_compute()?;

        // Find a matching key. Python falls back to "first distribution
        // under the same (kernel, quant)" when the exact distribution
        // string isn't in the loaded data.
        let exact_key = WideEpMoeKey {
            kernel_source: kernel_source.to_string(),
            quant: quant.name().to_string(),
            distribution: distribution.to_string(),
            topk,
            num_experts,
            hidden_size,
            inter_size,
            num_slots,
            moe_tp_size,
            moe_ep_size,
        };
        let by_tokens = match grids.by_keys.get(&exact_key) {
            Some(t) => t,
            None => {
                let fallback = grids
                    .by_keys
                    .iter()
                    .find(|(k, _)| {
                        k.kernel_source == exact_key.kernel_source
                            && k.quant == exact_key.quant
                            && k.topk == exact_key.topk
                            && k.num_experts == exact_key.num_experts
                            && k.hidden_size == exact_key.hidden_size
                            && k.inter_size == exact_key.inter_size
                            && k.num_slots == exact_key.num_slots
                            && k.moe_tp_size == exact_key.moe_tp_size
                            && k.moe_ep_size == exact_key.moe_ep_size
                    })
                    .map(|(_, t)| t)
                    .ok_or_else(|| {
                        AicError::PerfDatabase(format!(
                            "WideEP MoE compute data missing for {exact_key:?} at {}",
                            self.data_root.display()
                        ))
                    })?;
                fallback
            }
        };

        if let Some(&latency) = by_tokens.get(&num_tokens) {
            return Ok(latency);
        }
        let keys: Vec<u32> = by_tokens.keys().copied().collect();
        let (lo, hi) = nearest_neighbors(num_tokens, &keys, false)?;
        Ok(interp_1d(
            lo as f64,
            hi as f64,
            by_tokens[&lo],
            by_tokens[&hi],
            num_tokens as f64,
        ))
    }

    fn load_compute(&self) -> Result<&WideEpMoeGrids, AicError> {
        let cell = self
            .compute
            .get_or_init(|| load_compute_csv(&self.data_root.join("wideep_moe_perf.txt")));
        cell.as_ref().map_err(clone_err)
    }
}

fn load_compute_csv(path: &Path) -> Result<WideEpMoeGrids, AicError> {
    let text = read_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());
    let mut by_keys: BTreeMap<WideEpMoeKey, BTreeMap<u32, f64>> = BTreeMap::new();
    for record in reader.deserialize::<WideEpMoeRow>() {
        let row = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        let key = WideEpMoeKey {
            kernel_source: row.kernel_source,
            quant: row.moe_dtype,
            distribution: row.distribution,
            topk: row.topk,
            num_experts: row.num_experts,
            hidden_size: row.hidden_size,
            inter_size: row.inter_size,
            num_slots: row.num_slots,
            moe_tp_size: row.moe_tp_size,
            moe_ep_size: row.moe_ep_size,
        };
        // First-wins parity with Python loader.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.num_tokens)
            .or_insert(row.latency);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no WideEP MoE compute rows loaded from {}",
            path.display()
        )));
    }
    Ok(WideEpMoeGrids { by_keys })
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

    fn b200_trtllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/b200_sxm/trtllm/1.3.0rc10")
    }

    #[test]
    fn wideep_moe_compute_exact_hit() {
        // First row of b200_sxm/trtllm/1.3.0rc10/wideep_moe_perf.txt:
        // kernel=wideep_compute_cutlass moe_dtype=nvfp4 num_tokens=1
        // hidden=6144 inter=2048 topk=8 num_experts=256 num_slots=256
        // moe_tp=1 moe_ep=2 distribution=power_law_1.01 latency=0.08600...
        let table = WideEpMoeTable::new(b200_trtllm_data_root());
        let latency = table
            .query_compute(
                1,
                6144,
                2048,
                8,
                256,
                256,
                1,
                2,
                MoeQuantMode::Nvfp4,
                "power_law_1.01",
                "wideep_compute_cutlass",
            )
            .expect("WideEP MoE compute query must succeed");
        assert!(
            (latency - 0.086_009_597_778_320_32).abs() < 1e-6,
            "expected recorded latency, got {latency}"
        );
    }
}
