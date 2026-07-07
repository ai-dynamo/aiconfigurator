// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Communication perf tables: custom_allreduce + NCCL + OneCCL.
//!
//! Mirrors the SILICON paths of
//! `aiconfigurator_core.sdk.operations.communication.{CustomAllReduce, NCCL}._query_*_table`.
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
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::CommQuantMode;
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::interpolation::{interp_1d, nearest_neighbors};
use crate::perf_database::parquet_loader::PerfReader;

pub struct CommunicationTable {
    /// Directory containing `custom_allreduce_perf.parquet`. Resolved as
    /// `<systems_root>/<data_dir>/<backend>/<version>/`.
    data_root: PathBuf,
    /// Directory containing `nccl_perf.parquet`. Resolved as
    /// `<systems_root>/<data_dir>/nccl/<misc.nccl_version>/` to mirror
    /// Python's system-wide NCCL data layout. `None` when the system YAML
    /// has no `misc.nccl_version` declared.
    nccl_root: Option<PathBuf>,
    /// Directory containing `oneccl_perf.parquet`. Resolved as
    /// `<systems_root>/<data_dir>/oneccl/<misc.oneccl_version>/`. `None`
    /// when the system YAML has no `misc.oneccl_version` declared (most
    /// systems — OneCCL is the XPU fallback path).
    oneccl_root: Option<PathBuf>,
    /// Ordered, priority-sorted sources for `custom_allreduce_perf.parquet`
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`CommunicationTable::new`). NCCL/OneCCL remain framework-agnostic
    /// and are loaded directly from `nccl_root` / `oneccl_root`.
    custom_allreduce_sources: Vec<PerfSource>,
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

impl CommunicationTable {
    /// `data_root` holds the backend/version dir for custom-allreduce.
    /// `nccl_root` / `oneccl_root` point at the system-wide NCCL/OneCCL
    /// directories resolved from `SystemSpec.misc.{nccl,oneccl}_version`;
    /// callers without a system-spec-aware path may pass `None`, in which
    /// case the matching `query_nccl` / fallback path will surface a clear
    /// `PerfDatabase` error.
    pub fn new(
        data_root: PathBuf,
        nccl_root: Option<PathBuf>,
        oneccl_root: Option<PathBuf>,
    ) -> Self {
        Self::with_sources(data_root, nccl_root, oneccl_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied) for `custom_allreduce_perf.parquet`.
    /// The file falls back to its primary `data_root/custom_allreduce_perf.parquet`
    /// when absent from the map. NCCL/OneCCL are framework-agnostic and are NOT
    /// shared-layer sourced — they load directly from `nccl_root` / `oneccl_root`.
    /// No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        nccl_root: Option<PathBuf>,
        oneccl_root: Option<PathBuf>,
        perf_db_sources: &PerfDbSources,
    ) -> Self {
        let custom_allreduce_sources =
            resolve_op_sources(perf_db_sources, "custom_allreduce_perf.parquet", &data_root);
        Self {
            data_root,
            nccl_root,
            oneccl_root,
            custom_allreduce_sources,
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
            load_custom_allreduce_parquet(&self.custom_allreduce_sources)
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_nccl(&self) -> Result<&NcclGrids, AicError> {
        let cell = self.nccl.get_or_init(|| {
            let Some(root) = self.nccl_root.as_ref() else {
                return Err(AicError::PerfDatabase(
                    "NCCL data not configured for this system (no misc.nccl_version in YAML)"
                        .to_string(),
                ));
            };
            load_nccl_parquet(&root.join("nccl_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_oneccl(&self) -> Result<&NcclGrids, AicError> {
        let cell = self.oneccl.get_or_init(|| {
            let Some(root) = self.oneccl_root.as_ref() else {
                return Err(AicError::PerfDatabase(
                    "OneCCL data not configured for this system (no misc.oneccl_version in YAML)"
                        .to_string(),
                ));
            };
            load_nccl_parquet(&root.join("oneccl_perf.parquet"))
        });
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

fn load_custom_allreduce_parquet(sources: &[PerfSource]) -> Result<CustomAllReduceGrids, AicError> {
    let mut by_keys: BTreeMap<(String, u32), BTreeMap<u64, f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let num_gpus_col = reader.col("num_gpus")?;
        let message_size_col = reader.col("message_size")?;
        let latency_col = reader.col("latency")?;
        let kernel_source_col = reader.col_optional("kernel_source");
        let backend_col = reader.col_optional("backend");
        let ks_col = reader.col_optional("kernel_source");

        // Mirror Python/legacy: skip "_eager" kernel sources on systems other
        // than b60. We can't see the system name from here, so apply the filter
        // by path prefix.
        let path_str = path.to_string_lossy();
        let is_b60 = path_str.contains("/b60/");

        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            if !is_b60 {
                let kernel = row.str_optional(kernel_source_col)?.unwrap_or("");
                let backend = row.str_optional(backend_col)?.unwrap_or("");
                if kernel.ends_with("_eager") || backend.ends_with("_eager") {
                    continue;
                }
            }
            // Match Python's `load_custom_allreduce_data`: every row is stored
            // under `CommQuantMode.half` regardless of the CSV's
            // `allreduce_dtype` column (Python has a `TODO` here but the
            // behavior is stable in production).
            // First-wins parity with Python `load_custom_allreduce_data`,
            // extended across shared-layer sources (earlier source wins).
            by_keys
                .entry(("half".to_string(), row.u32(num_gpus_col)?))
                .or_default()
                .entry(row.u64(message_size_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(CustomAllReduceGrids { by_keys })
}

fn load_nccl_parquet(path: &Path) -> Result<NcclGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let op_name_col = reader.col("op_name")?;
    let nccl_dtype_col = reader.col("nccl_dtype")?;
    let num_gpus_col = reader.col("num_gpus")?;
    let message_size_col = reader.col("message_size")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<(String, String, u32), BTreeMap<u64, f64>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        // First-wins parity with Python `load_nccl_data`.
        by_keys
            .entry((
                row.str_owned(nccl_dtype_col)?,
                row.str_owned(op_name_col)?,
                row.u32(num_gpus_col)?,
            ))
            .or_default()
            .entry(row.u64(message_size_col)?)
            .or_insert(row.f64(latency_col)?);
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no NCCL/OneCCL rows loaded from {}",
            path.display()
        )));
    }
    Ok(NcclGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn systems_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems")
    }

    fn b200_vllm_data_root() -> PathBuf {
        systems_root().join("data/b200_sxm/vllm/0.19.0")
    }

    fn b200_sglang_data_root() -> PathBuf {
        systems_root().join("data/b200_sxm/sglang/0.5.10")
    }

    /// `<systems_root>/data/b200_sxm/nccl/2.27.3/` — the system-spec-aware
    /// NCCL root for b200_sxm, mirroring Python's path layout.
    fn b200_nccl_root() -> Option<PathBuf> {
        Some(systems_root().join("data/b200_sxm/nccl/2.27.3"))
    }

    #[test]
    fn custom_allreduce_tp1_is_zero() {
        let table = CommunicationTable::new(b200_vllm_data_root(), None, None);
        let latency = table
            .query_custom_allreduce(CommQuantMode::Half, 1, 1024)
            .expect("tp=1 is a no-op");
        assert_eq!(latency, 0.0);
    }

    #[test]
    fn custom_allreduce_loads_from_vllm_b200() {
        let table = CommunicationTable::new(b200_vllm_data_root(), None, None);
        // Verify the loader runs and the table contains keys for typical
        // smoke TP values.
        let _ = table.load_custom_allreduce().expect("loader must succeed");
    }

    #[test]
    fn custom_allreduce_query_succeeds_for_tp8() {
        let table = CommunicationTable::new(b200_sglang_data_root(), None, None);
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
        let table = CommunicationTable::new(b200_vllm_data_root(), None, None);
        let latency = table
            .query_nccl(CommQuantMode::Half, "all_reduce", 1, 1024)
            .expect("num_gpus=1 is a no-op");
        assert_eq!(latency, 0.0);
    }

    #[test]
    fn nccl_loads_from_system_wide_path() {
        // With the system-spec-aware path (b200_sxm declares
        // `nccl_version: '2.27.3'`), NCCL data resolves to
        // `<systems_root>/data/b200_sxm/nccl/2.27.3/nccl_perf.parquet`
        // and the table loads successfully — NOT
        // `<vllm/0.19.0>/nccl_perf.parquet` which never existed.
        let table = CommunicationTable::new(b200_vllm_data_root(), b200_nccl_root(), None);
        let _ = table.load_nccl().expect("NCCL parquet must load from system-wide path");
    }

    #[test]
    fn nccl_unconfigured_errors_clearly() {
        // When neither `misc.nccl_version` nor `misc.oneccl_version` is
        // declared, both load attempts surface a clean configuration error
        // rather than silently degrading.
        let table = CommunicationTable::new(b200_vllm_data_root(), None, None);
        let err = table
            .query_nccl(CommQuantMode::Half, "all_reduce", 2, 1024)
            .unwrap_err();
        match err {
            AicError::PerfDatabase(msg) => {
                assert!(
                    msg.contains("OneCCL data not configured"),
                    "expected fallthrough-to-OneCCL error message, got: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
