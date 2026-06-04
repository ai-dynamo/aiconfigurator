// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WideEP / DeepEP / TRT-LLM all-to-all perf tables for distributed MoE.
//!
//! Five CSVs span two shape families:
//!
//! 1. MoE-compute layout (same columns as `moe_perf.txt`):
//!    - `wideep_context_moe_perf.txt`
//!    - `wideep_generation_moe_perf.txt`
//!    - `wideep_moe_perf.txt` (TRT-LLM WideEP MoE compute; extra columns
//!      handled by tolerant deserialization)
//!    - `trtllm_alltoall_perf.txt` (TRT-LLM alltoall dispatch; subset of
//!      MoE columns)
//!
//! 2. DeepEP dispatch layout (separate notify/transmit latencies):
//!    - `wideep_deepep_normal_perf.txt`
//!    - `wideep_deepep_ll_perf.txt`
//!
//! All loaders are lazy. Queries return raw 1-D-interpolated latency along
//! `num_tokens`; SOL/EMPIRICAL wrappers live in `operators/moe.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::interpolation::{interp_1d, nearest_neighbors};
use crate::perf_database::parquet_loader::PerfReader;

pub struct WideEpTable {
    data_root: PathBuf,
    context_moe: OnceLock<Result<MoeGrids, AicError>>,
    generation_moe: OnceLock<Result<MoeGrids, AicError>>,
    trtllm_wideep_moe: OnceLock<Result<MoeGrids, AicError>>,
    trtllm_alltoall: OnceLock<Result<MoeGrids, AicError>>,
    deepep_normal: OnceLock<Result<DispatchGrids, AicError>>,
    deepep_ll: OnceLock<Result<DispatchGrids, AicError>>,
}

struct MoeGrids {
    by_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>>,
}

struct DispatchGrids {
    by_keys: BTreeMap<DispatchKey, BTreeMap<u32, DispatchPoint>>,
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DispatchKey {
    node_num: u32,
    hidden_size: u32,
    num_topk: u32,
    num_experts: u32,
}

/// Dispatch-style latency point. DeepEP normal CSV reports separate
/// notify/transmit times for dispatch and combine; DeepEP LL reports
/// average latencies and bandwidths.
#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchPoint {
    pub dispatch_transmit_us: f64,
    pub dispatch_notify_us: f64,
    pub combine_transmit_us: f64,
    pub combine_notify_us: f64,
    /// LL-only: combine average latency (us).
    pub combine_avg_t_us: f64,
    /// LL-only: dispatch average latency (us).
    pub dispatch_avg_t_us: f64,
}

impl WideEpTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            context_moe: OnceLock::new(),
            generation_moe: OnceLock::new(),
            trtllm_wideep_moe: OnceLock::new(),
            trtllm_alltoall: OnceLock::new(),
            deepep_normal: OnceLock::new(),
            deepep_ll: OnceLock::new(),
        }
    }

    /// WideEP context-phase MoE compute latency (ms).
    pub fn query_context_moe(
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
        let grids = self.load_context_moe()?;
        query_moe(
            grids,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant,
            workload_distribution,
            &self.data_root,
        )
    }

    pub fn query_generation_moe(
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
        let grids = self.load_generation_moe()?;
        query_moe(
            grids,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant,
            workload_distribution,
            &self.data_root,
        )
    }

    pub fn query_trtllm_wideep_moe(
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
        let grids = self.load_trtllm_wideep_moe()?;
        query_moe(
            grids,
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant,
            workload_distribution,
            &self.data_root,
        )
    }

    /// TRT-LLM alltoall dispatch latency. CSV uses `moe_ep_size` for fan-out
    /// (`moe_tp_size`/`inter_size` are not present); the query API still
    /// accepts them for shape symmetry but they're effectively ignored.
    pub fn query_trtllm_alltoall(
        &self,
        num_tokens: u32,
        hidden_size: u32,
        topk: u32,
        num_experts: u32,
        moe_ep_size: u32,
        quant: MoeQuantMode,
        workload_distribution: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load_trtllm_alltoall()?;
        query_moe(
            grids,
            num_tokens,
            hidden_size,
            0,
            topk,
            num_experts,
            1,
            moe_ep_size,
            quant,
            workload_distribution,
            &self.data_root,
        )
    }

    /// DeepEP normal-mode dispatch point.
    pub fn query_deepep_normal(
        &self,
        node_num: u32,
        hidden_size: u32,
        num_tokens: u32,
        num_topk: u32,
        num_experts: u32,
    ) -> Result<DispatchPoint, AicError> {
        let grids = self.load_deepep_normal()?;
        dispatch_lookup(grids, node_num, hidden_size, num_tokens, num_topk, num_experts, &self.data_root)
    }

    /// DeepEP low-latency dispatch point.
    pub fn query_deepep_ll(
        &self,
        node_num: u32,
        hidden_size: u32,
        num_tokens: u32,
        num_topk: u32,
        num_experts: u32,
    ) -> Result<DispatchPoint, AicError> {
        let grids = self.load_deepep_ll()?;
        dispatch_lookup(grids, node_num, hidden_size, num_tokens, num_topk, num_experts, &self.data_root)
    }

    fn load_context_moe(&self) -> Result<&MoeGrids, AicError> {
        let cell = self
            .context_moe
            .get_or_init(|| load_moe_parquet(&self.data_root.join("wideep_context_moe_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }
    fn load_generation_moe(&self) -> Result<&MoeGrids, AicError> {
        let cell = self
            .generation_moe
            .get_or_init(|| load_moe_parquet(&self.data_root.join("wideep_generation_moe_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }
    fn load_trtllm_wideep_moe(&self) -> Result<&MoeGrids, AicError> {
        let cell = self
            .trtllm_wideep_moe
            .get_or_init(|| load_moe_parquet(&self.data_root.join("wideep_moe_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }
    fn load_trtllm_alltoall(&self) -> Result<&MoeGrids, AicError> {
        let cell = self
            .trtllm_alltoall
            .get_or_init(|| load_moe_parquet(&self.data_root.join("trtllm_alltoall_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }
    fn load_deepep_normal(&self) -> Result<&DispatchGrids, AicError> {
        let cell = self.deepep_normal.get_or_init(|| {
            load_deepep_normal_parquet(&self.data_root.join("wideep_deepep_normal_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_deepep_ll(&self) -> Result<&DispatchGrids, AicError> {
        let cell = self
            .deepep_ll
            .get_or_init(|| load_deepep_ll_parquet(&self.data_root.join("wideep_deepep_ll_perf.parquet")));
        cell.as_ref().map_err(clone_err)
    }
}

fn query_moe(
    grids: &MoeGrids,
    num_tokens: u32,
    hidden_size: u32,
    inter_size: u32,
    topk: u32,
    num_experts: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
    quant: MoeQuantMode,
    workload_distribution: &str,
    data_root: &Path,
) -> Result<f64, AicError> {
    let quant_name = quant.name();
    let requested_exists = grids
        .by_keys
        .keys()
        .any(|k| k.quant == quant_name && k.distribution == workload_distribution);
    let distribution = if requested_exists {
        workload_distribution.to_string()
    } else {
        "uniform".to_string()
    };
    let key = MoeKey {
        quant: quant_name.to_string(),
        distribution,
        topk,
        num_experts,
        hidden_size,
        inter_size,
        moe_tp_size,
        moe_ep_size,
    };
    let by_tokens = grids
        .by_keys
        .get(&key)
        .ok_or_else(|| AicError::PerfDatabase(format!("MoE data missing for {key:?} at {}", data_root.display())))?;
    if let Some(&latency) = by_tokens.get(&num_tokens) {
        return Ok(latency);
    }
    let token_keys: Vec<u32> = by_tokens.keys().copied().collect();
    if token_keys.is_empty() {
        return Err(AicError::PerfDatabase("MoE table has no token points".to_string()));
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

fn dispatch_lookup(
    grids: &DispatchGrids,
    node_num: u32,
    hidden_size: u32,
    num_tokens: u32,
    num_topk: u32,
    num_experts: u32,
    data_root: &Path,
) -> Result<DispatchPoint, AicError> {
    let key = DispatchKey {
        node_num,
        hidden_size,
        num_topk,
        num_experts,
    };
    let by_tokens = grids.by_keys.get(&key).ok_or_else(|| {
        AicError::PerfDatabase(format!("dispatch data missing for {key:?} at {}", data_root.display()))
    })?;
    if let Some(point) = by_tokens.get(&num_tokens) {
        return Ok(*point);
    }
    let token_keys: Vec<u32> = by_tokens.keys().copied().collect();
    let (lo, hi) = nearest_neighbors(num_tokens, &token_keys, false)?;
    let p0 = by_tokens[&lo];
    let p1 = by_tokens[&hi];
    Ok(DispatchPoint {
        dispatch_transmit_us: interp_1d(lo as f64, hi as f64, p0.dispatch_transmit_us, p1.dispatch_transmit_us, num_tokens as f64),
        dispatch_notify_us: interp_1d(lo as f64, hi as f64, p0.dispatch_notify_us, p1.dispatch_notify_us, num_tokens as f64),
        combine_transmit_us: interp_1d(lo as f64, hi as f64, p0.combine_transmit_us, p1.combine_transmit_us, num_tokens as f64),
        combine_notify_us: interp_1d(lo as f64, hi as f64, p0.combine_notify_us, p1.combine_notify_us, num_tokens as f64),
        combine_avg_t_us: interp_1d(lo as f64, hi as f64, p0.combine_avg_t_us, p1.combine_avg_t_us, num_tokens as f64),
        dispatch_avg_t_us: interp_1d(lo as f64, hi as f64, p0.dispatch_avg_t_us, p1.dispatch_avg_t_us, num_tokens as f64),
    })
}

fn load_moe_parquet(path: &Path) -> Result<MoeGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let moe_dtype_col = reader.col("moe_dtype")?;
    let num_tokens_col = reader.col("num_tokens")?;
    let hidden_size_col = reader.col("hidden_size")?;
    // `inter_size` / `moe_tp_size` are absent in `trtllm_alltoall_perf.parquet`;
    // shared with the wideep_*_moe parquets which carry them. Optional lookup
    // mirrors the prior `Option<u32>` deserialization plus `unwrap_or` default.
    let inter_size_col = reader.col_optional("inter_size");
    let topk_col = reader.col("topk")?;
    let num_experts_col = reader.col("num_experts")?;
    let moe_tp_size_col = reader.col_optional("moe_tp_size");
    let moe_ep_size_col = reader.col("moe_ep_size")?;
    let distribution_col = reader.col("distribution")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<MoeKey, BTreeMap<u32, f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = MoeKey {
            quant: row.str_owned(moe_dtype_col)?,
            distribution: row.str_owned(distribution_col)?,
            topk: row.u32(topk_col)?,
            num_experts: row.u32(num_experts_col)?,
            hidden_size: row.u32(hidden_size_col)?,
            inter_size: row.u32_optional(inter_size_col)?.unwrap_or(0),
            moe_tp_size: row.u32_optional(moe_tp_size_col)?.unwrap_or(1),
            moe_ep_size: row.u32(moe_ep_size_col)?,
        };
        // First-wins parity with Python `load_wideep_*_moe_data`.
        by_keys
            .entry(key)
            .or_default()
            .entry(row.u32(num_tokens_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MoE-shape rows loaded from {}",
            path.display()
        )));
    }
    Ok(MoeGrids { by_keys })
}

fn load_deepep_normal_parquet(path: &Path) -> Result<DispatchGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let node_num_col = reader.col("node_num")?;
    let hidden_size_col = reader.col("hidden_size")?;
    let num_token_col = reader.col("num_token")?;
    let num_topk_col = reader.col("num_topk")?;
    let num_experts_col = reader.col("num_experts")?;
    let dispatch_transmit_us_col = reader.col_optional("dispatch_transmit_us");
    let dispatch_notify_us_col = reader.col_optional("dispatch_notify_us");
    let combine_transmit_us_col = reader.col_optional("combine_transmit_us");
    let combine_notify_us_col = reader.col_optional("combine_notify_us");

    let mut by_keys: BTreeMap<DispatchKey, BTreeMap<u32, DispatchPoint>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = DispatchKey {
            node_num: row.u32(node_num_col)?,
            hidden_size: row.u32(hidden_size_col)?,
            num_topk: row.u32(num_topk_col)?,
            num_experts: row.u32(num_experts_col)?,
        };
        by_keys.entry(key).or_default().insert(
            row.u32(num_token_col)?,
            DispatchPoint {
                dispatch_transmit_us: row.f64_optional(dispatch_transmit_us_col)?.unwrap_or(0.0),
                dispatch_notify_us: row.f64_optional(dispatch_notify_us_col)?.unwrap_or(0.0),
                combine_transmit_us: row.f64_optional(combine_transmit_us_col)?.unwrap_or(0.0),
                combine_notify_us: row.f64_optional(combine_notify_us_col)?.unwrap_or(0.0),
                combine_avg_t_us: 0.0,
                dispatch_avg_t_us: 0.0,
            },
        );
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DeepEP-normal rows loaded from {}",
            path.display()
        )));
    }
    Ok(DispatchGrids { by_keys })
}

fn load_deepep_ll_parquet(path: &Path) -> Result<DispatchGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let node_num_col = reader.col("node_num")?;
    let hidden_size_col = reader.col("hidden_size")?;
    let num_token_col = reader.col("num_token")?;
    let num_topk_col = reader.col("num_topk")?;
    let num_experts_col = reader.col("num_experts")?;
    let combine_avg_t_us_col = reader.col_optional("combine_avg_t_us");
    let dispatch_avg_t_us_col = reader.col_optional("dispatch_avg_t_us");

    let mut by_keys: BTreeMap<DispatchKey, BTreeMap<u32, DispatchPoint>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = DispatchKey {
            node_num: row.u32(node_num_col)?,
            hidden_size: row.u32(hidden_size_col)?,
            num_topk: row.u32(num_topk_col)?,
            num_experts: row.u32(num_experts_col)?,
        };
        by_keys.entry(key).or_default().insert(
            row.u32(num_token_col)?,
            DispatchPoint {
                dispatch_transmit_us: 0.0,
                dispatch_notify_us: 0.0,
                combine_transmit_us: 0.0,
                combine_notify_us: 0.0,
                combine_avg_t_us: row.f64_optional(combine_avg_t_us_col)?.unwrap_or(0.0),
                dispatch_avg_t_us: row.f64_optional(dispatch_avg_t_us_col)?.unwrap_or(0.0),
            },
        );
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DeepEP-LL rows loaded from {}",
            path.display()
        )));
    }
    Ok(DispatchGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wideep_loaders_smoke() {
        // None of the WideEP/DeepEP data exists on vLLM b200 (TRT-LLM/SGLang
        // territory). Loader must surface clean errors.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0");
        let table = WideEpTable::new(root);
        let err = table
            .query_context_moe(
                1024,
                4096,
                2048,
                2,
                128,
                1,
                8,
                MoeQuantMode::Bfloat16,
                "uniform",
            )
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
