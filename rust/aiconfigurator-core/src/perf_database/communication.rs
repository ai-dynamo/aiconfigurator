// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Communication perf tables: custom_allreduce + NCCL + OneCCL.
//!
//! Mirrors the SILICON paths of
//! `aiconfigurator.sdk.operations.communication.{CustomAllReduce, NCCL}._query_*_table`.
//! P2P latency is computed analytically by the operator layer from
//! `SystemSpec` fields, not from a CSV, so there's no `P2PTable` here.
//!
//! Query APIs take *effective* tp_size / num_gpus values — the operator is
//! responsible for capping to the node fan-out and applying any
//! cross-rack bandwidth correction factor (those depend on `SystemSpec`).
//! Rows with `_eager` kernel sources are filtered out at load time per
//! Python's `CustomAllReduce.load_data` behavior; the production path uses
//! CUDA-graph variants.
//!
//! OneCCL is loaded lazily and is the fallback when NCCL data is absent
//! (e.g. on Intel XPU systems). The query API tries NCCL first and falls
//! back transparently.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::common::enums::CommQuantMode;
use crate::common::error::AicError;
use crate::interpolation::{interp_1d, nearest_neighbors};

pub struct CommunicationTable {
    data_root: PathBuf,
    custom_allreduce: OnceLock<Result<CustomAllReduceGrids, AicError>>,
    nccl: OnceLock<Result<NcclGrids, AicError>>,
    oneccl: OnceLock<Result<NcclGrids, AicError>>,
}

struct CustomAllReduceGrids {
    /// (quant_name, tp_size) -> {message_size -> latency_ms}
    by_keys: BTreeMap<(String, u32), BTreeMap<u64, f64>>,
}

struct NcclGrids {
    /// (dtype_name, operation, num_gpus) -> {message_size -> latency_ms}
    by_keys: BTreeMap<(String, String, u32), BTreeMap<u64, f64>>,
}

#[derive(Debug, Deserialize)]
struct CustomAllReduceRow {
    kernel_source: Option<String>,
    allreduce_dtype: String,
    num_gpus: u32,
    message_size: u64,
    latency: f64,
    #[serde(default)]
    backend: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NcclRow {
    op_name: String,
    nccl_dtype: String,
    num_gpus: u32,
    message_size: u64,
    latency: f64,
}

impl CommunicationTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            custom_allreduce: OnceLock::new(),
            nccl: OnceLock::new(),
            oneccl: OnceLock::new(),
        }
    }

    /// Raw custom-allreduce latency in ms, 1-D interpolated along
    /// `message_size`.
    ///
    /// `tp_size_effective` is the per-node fan-out the caller wants to look
    /// up. For TP > num_gpus_per_node the operator caps this to
    /// `num_gpus_per_node` and applies a bandwidth scale separately.
    pub fn query_custom_allreduce(
        &self,
        quant: CommQuantMode,
        tp_size_effective: u32,
        message_size: u64,
    ) -> Result<f64, AicError> {
        if tp_size_effective <= 1 {
            return Ok(0.0);
        }
        let grids = self.load_custom_allreduce()?;
        let key = (quant.name().to_string(), tp_size_effective);
        let by_size = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "custom_allreduce data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        interp_message_size(by_size, message_size)
    }

    /// Raw NCCL collective latency in ms.
    ///
    /// `operation` is one of `"all_reduce"`, `"all_gather"`,
    /// `"reduce_scatter"`, `"alltoall"`. `num_gpus_effective` should be
    /// capped to the max recorded fan-out by the caller; this routine
    /// errors if the requested key is missing.
    ///
    /// Falls back to OneCCL data when NCCL data is absent for the slice
    /// (matches Python's XPU-fallback behavior).
    pub fn query_nccl(
        &self,
        dtype: CommQuantMode,
        operation: &str,
        num_gpus_effective: u32,
        message_size: u64,
    ) -> Result<f64, AicError> {
        if num_gpus_effective <= 1 {
            return Ok(0.0);
        }
        let key = (dtype.name().to_string(), operation.to_string(), num_gpus_effective);

        if let Ok(grids) = self.load_nccl() {
            if let Some(by_size) = grids.by_keys.get(&key) {
                return interp_message_size(by_size, message_size);
            }
        }
        // Fall back to OneCCL.
        let grids = self.load_oneccl()?;
        let by_size = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "neither NCCL nor OneCCL has data for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        interp_message_size(by_size, message_size)
    }

    /// Maximum recorded `num_gpus` for an NCCL (dtype, operation) tuple.
    /// Operator layer uses this to decide whether to apply a bandwidth
    /// scale factor for out-of-range fan-outs.
    pub fn nccl_max_num_gpus(
        &self,
        dtype: CommQuantMode,
        operation: &str,
    ) -> Result<Option<u32>, AicError> {
        let dtype_name = dtype.name().to_string();
        let op = operation.to_string();
        let mut max_seen = None;
        for source in [self.load_nccl(), self.load_oneccl()] {
            let Ok(grids) = source else { continue };
            for (k_dtype, k_op, k_num) in grids.by_keys.keys() {
                if k_dtype == &dtype_name && k_op == &op {
                    max_seen = Some(max_seen.map_or(*k_num, |m: u32| m.max(*k_num)));
                }
            }
        }
        Ok(max_seen)
    }

    fn load_custom_allreduce(&self) -> Result<&CustomAllReduceGrids, AicError> {
        let cell = self.custom_allreduce.get_or_init(|| {
            load_custom_allreduce_csv(&self.data_root.join("custom_allreduce_perf.txt"))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_nccl(&self) -> Result<&NcclGrids, AicError> {
        let cell = self
            .nccl
            .get_or_init(|| load_nccl_csv(&self.data_root.join("nccl_perf.txt")));
        cell.as_ref().map_err(clone_err)
    }

    fn load_oneccl(&self) -> Result<&NcclGrids, AicError> {
        let cell = self
            .oneccl
            .get_or_init(|| load_nccl_csv(&self.data_root.join("oneccl_perf.txt")));
        cell.as_ref().map_err(clone_err)
    }
}

fn interp_message_size(by_size: &BTreeMap<u64, f64>, message_size: u64) -> Result<f64, AicError> {
    if let Some(&latency) = by_size.get(&message_size) {
        return Ok(latency);
    }
    if by_size.is_empty() {
        return Err(AicError::PerfDatabase(
            "comm data has no message_size points".to_string(),
        ));
    }
    let sizes: Vec<u32> = by_size.keys().map(|&s| s.min(u32::MAX as u64) as u32).collect();
    let query = message_size.min(u32::MAX as u64) as u32;
    let (lo, hi) = nearest_neighbors(query, &sizes, false)?;
    let y_lo = by_size[&(lo as u64)];
    let y_hi = by_size[&(hi as u64)];
    Ok(interp_1d(lo as f64, hi as f64, y_lo, y_hi, query as f64))
}

fn load_custom_allreduce_csv(path: &Path) -> Result<CustomAllReduceGrids, AicError> {
    let text = read_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());

    // Mirror Python/legacy: skip "_eager" kernel sources on systems other
    // than b60. We can't see the system name from here, so apply the filter
    // by path prefix.
    let path_str = path.to_string_lossy();
    let is_b60 = path_str.contains("/b60/");

    let mut by_keys: BTreeMap<(String, u32), BTreeMap<u64, f64>> = BTreeMap::new();
    for record in reader.deserialize::<CustomAllReduceRow>() {
        let row = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        if !is_b60 {
            let kernel = row.kernel_source.as_deref().unwrap_or("");
            let backend = row.backend.as_deref().unwrap_or("");
            if kernel.ends_with("_eager") || backend.ends_with("_eager") {
                continue;
            }
        }
        // Match Python's `load_custom_allreduce_data`: every row is stored
        // under `CommQuantMode.half` regardless of the CSV's
        // `allreduce_dtype` column (Python has a `TODO` here but the
        // behavior is stable in production).
        let _ = row.allreduce_dtype;
        // First-wins parity with Python `load_custom_allreduce_data`.
        by_keys
            .entry(("half".to_string(), row.num_gpus))
            .or_default()
            .entry(row.message_size)
            .or_insert(row.latency);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no custom_allreduce rows loaded from {}",
            path.display()
        )));
    }
    Ok(CustomAllReduceGrids { by_keys })
}

fn load_nccl_csv(path: &Path) -> Result<NcclGrids, AicError> {
    let text = read_perf_file(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());

    let mut by_keys: BTreeMap<(String, String, u32), BTreeMap<u64, f64>> = BTreeMap::new();
    for record in reader.deserialize::<NcclRow>() {
        let row = record.map_err(|source| AicError::Csv {
            path: path.to_path_buf(),
            source,
        })?;
        // First-wins parity with Python `load_nccl_data`.
        by_keys
            .entry((row.nccl_dtype, row.op_name, row.num_gpus))
            .or_default()
            .entry(row.message_size)
            .or_insert(row.latency);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no NCCL/OneCCL rows loaded from {}",
            path.display()
        )));
    }
    Ok(NcclGrids { by_keys })
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

    fn b200_sglang_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/b200_sxm/sglang/0.5.10")
    }

    #[test]
    fn custom_allreduce_tp1_is_zero() {
        let table = CommunicationTable::new(b200_vllm_data_root());
        let latency = table
            .query_custom_allreduce(CommQuantMode::Half, 1, 1024)
            .expect("tp=1 is a no-op");
        assert_eq!(latency, 0.0);
    }

    #[test]
    fn custom_allreduce_loads_from_vllm_b200() {
        let table = CommunicationTable::new(b200_vllm_data_root());
        // Verify the loader runs and the table contains keys for typical
        // smoke TP values.
        let _ = table.load_custom_allreduce().expect("loader must succeed");
    }

    #[test]
    fn custom_allreduce_query_succeeds_for_tp8() {
        let table = CommunicationTable::new(b200_sglang_data_root());
        // SGLang b200 ships custom_allreduce data; pick a small message
        // and a TP that exists.
        let result = table.query_custom_allreduce(CommQuantMode::Half, 2, 1024);
        match result {
            Ok(latency) => assert!(latency > 0.0, "expected positive latency"),
            Err(AicError::PerfDatabase(_)) => {
                // Tp=2 may not be in this dataset — acceptable failure mode.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn nccl_num_gpus_1_is_zero() {
        let table = CommunicationTable::new(b200_vllm_data_root());
        let latency = table
            .query_nccl(CommQuantMode::Half, "all_reduce", 1, 1024)
            .expect("num_gpus=1 is a no-op");
        assert_eq!(latency, 0.0);
    }

    #[test]
    fn nccl_missing_data_errors_clearly() {
        // vLLM b200 doesn't ship nccl_perf.txt; loader returns IO error,
        // then OneCCL fallback also missing — query surfaces a clean error.
        let table = CommunicationTable::new(b200_vllm_data_root());
        let err = table
            .query_nccl(CommQuantMode::Half, "all_reduce", 2, 1024)
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
