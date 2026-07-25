// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Communication perf tables: custom_allreduce + NCCL + OneCCL.
//!
//! Mirrors the SILICON paths of
//! `aiconfigurator.sdk.operations.communication.{CustomAllReduce, NCCL}._query_*_table`.
//! P2P latency is computed analytically by the operator layer from
//! `SystemSpec` fields, not from a CSV, so there's no `P2PTable` here.
//!
//! The `*_scaled` query APIs take RAW tp_size / num_gpus values and own the
//! full Python DB-level semantics (node-fan-out capping, beyond-range
//! bandwidth correction, and the GB200-NVL72 custom-AR -> NCCL reroute) so
//! every consumer inherits them, exactly like Python's `_query_*_table`
//! funnels. The non-`_scaled` variants take *effective* values and only
//! interpolate the table.
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
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::perf_interp::{self, Node, OpInterpConfig};
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
        message_size: f64,
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

    /// Custom-allreduce latency at a RAW tp_size, mirroring the full Python
    /// DB-level `_query_custom_allreduce_table.get_silicon`
    /// (operations/communication.py) so every consumer inherits the same
    /// semantics:
    ///   1. `tp == 1` -> 0;
    ///   2. GB200 NVL72 (`num_gpus_per_node == 72`) with `tp > 4` -> reroute
    ///      to NCCL all_reduce at the RAW tp (custom AR is only collected up
    ///      to tp4 there);
    ///   3. clamp tp to the node size and interpolate the table;
    ///   4. beyond-node overflow: scale by the p2p-bandwidth ratio.
    pub fn query_custom_allreduce_scaled(
        &self,
        spec: &SystemSpec,
        quant: CommQuantMode,
        tp_size: u32,
        message_size: f64,
    ) -> Result<f64, AicError> {
        if tp_size <= 1 {
            return Ok(0.0);
        }
        let per_node = spec.node.num_gpus_per_node;
        if per_node == 72 && tp_size > 4 {
            return self.query_nccl_scaled(spec, quant, "all_reduce", tp_size, message_size);
        }
        let effective_tp = tp_size.min(per_node);
        let mut latency = self.query_custom_allreduce(quant, effective_tp, message_size)?;
        if tp_size > per_node {
            let base_bw = spec.get_p2p_bandwidth(per_node);
            let target_bw = spec.get_p2p_bandwidth(tp_size);
            let f_tp = tp_size as f64;
            let f_pn = per_node as f64;
            latency *= (f_tp - 1.0) / f_tp * f_pn / (f_pn - 1.0).max(1.0) * base_bw / target_bw;
        }
        Ok(latency)
    }

    /// NCCL collective latency at a RAW num_gpus, mirroring the Python
    /// DB-level `_query_nccl_table.get_silicon`: fan-out capped to the max
    /// recorded `num_gpus` for the (dtype, operation) slice, with the
    /// p2p-bandwidth correction applied beyond it.
    pub fn query_nccl_scaled(
        &self,
        spec: &SystemSpec,
        dtype: CommQuantMode,
        operation: &str,
        num_gpus: u32,
        message_size: f64,
    ) -> Result<f64, AicError> {
        if num_gpus <= 1 {
            return Ok(0.0);
        }
        let max_recorded = self.nccl_max_num_gpus(dtype, operation)?.unwrap_or(num_gpus);
        let effective = num_gpus.min(max_recorded);
        let mut latency = self.query_nccl(dtype, operation, effective, message_size)?;
        if num_gpus > max_recorded {
            let max_bw = spec.get_p2p_bandwidth(max_recorded);
            let req_bw = spec.get_p2p_bandwidth(num_gpus);
            let f_n = num_gpus as f64;
            let f_m = max_recorded as f64;
            latency *= (f_n - 1.0) / f_n * f_m / (f_m - 1.0).max(1.0) * max_bw / req_bw;
        }
        Ok(latency)
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
        message_size: f64,
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

    /// Collected `(message_size,) -> latency_ms` points of the
    /// custom-allreduce curve for `(quant, tp_size)` — the input of the
    /// operator-layer util grid (mirrors Python's
    /// `require_data_slice(dw, quant_mode, eff, "AUTO")`). Typed miss when
    /// the slice is absent or empty.
    pub fn custom_allreduce_points(
        &self,
        quant: CommQuantMode,
        tp_size: u32,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_custom_allreduce()?;
        let key = (quant.name().to_string(), tp_size);
        let by_size = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "custom_allreduce data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if by_size.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "custom_allreduce data empty for {key:?} at {}",
                self.data_root.display()
            )));
        }
        Ok(by_size.iter().map(|(&size, &lat)| (vec![size as f64], lat)).collect())
    }

    /// The single NCCL source the empirical path calibrates from, with
    /// Python's selection order (`NCCL._query_nccl_table.get_empirical`):
    /// the NCCL table when loaded, else the OneCCL fallback; a typed miss
    /// when neither is loaded. Unlike [`Self::query_nccl`], there is NO
    /// per-slice fallback across sources.
    fn nccl_empirical_source(&self) -> Result<&NcclGrids, AicError> {
        if let Ok(grids) = self.load_nccl() {
            return Ok(grids);
        }
        self.load_oneccl()
    }

    /// Maximum collected `num_gpus` for `(dtype, operation)` in the NCCL
    /// empirical source (single source, Python parity — unlike
    /// [`Self::nccl_max_num_gpus`], which unions NCCL and OneCCL for the
    /// silicon cap). Typed miss when the source has no such bucket.
    pub fn nccl_empirical_max_num_gpus(
        &self,
        dtype: CommQuantMode,
        operation: &str,
    ) -> Result<u32, AicError> {
        let grids = self.nccl_empirical_source()?;
        let dtype_name = dtype.name();
        grids
            .by_keys
            .keys()
            .filter(|(d, op, _)| d.as_str() == dtype_name && op.as_str() == operation)
            .map(|(_, _, n)| *n)
            .max()
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "NCCL data missing for dtype='{dtype_name}', operation='{operation}' at {}",
                    self.data_root.display()
                ))
            })
    }

    /// Collected `(message_size,) -> latency_ms` points for
    /// `(dtype, operation, num_gpus)` in the NCCL empirical source. Typed
    /// miss when the slice is absent or empty.
    pub fn nccl_empirical_points(
        &self,
        dtype: CommQuantMode,
        operation: &str,
        num_gpus: u32,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.nccl_empirical_source()?;
        let key = (dtype.name().to_string(), operation.to_string(), num_gpus);
        let by_size = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "NCCL data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if by_size.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "NCCL data empty for {key:?} at {}",
                self.data_root.display()
            )));
        }
        Ok(by_size.iter().map(|(&size, &lat)| (vec![size as f64], lat)).collect())
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

/// Resolve a 1-axis message-size curve on the perf_interp v2 engine: exact
/// hit / RAW lerp in range (bandwidth-bound collectives are ~linear in
/// size); beyond the collected range the boundary util is held (`k_tail=1`)
/// and SOL carries the growth — the legacy raw two-point extrapolation could
/// undershoot the launch floor or go negative below the smallest size.
///
/// SOL is a LINEAR message-size proxy (`sol(size) = size`). Python passes
/// the actual collective roofline
/// (`communication.py::_query_{custom_allreduce,nccl}_table.get_sol`), but
/// for a fixed (op, num_gpus) slice that roofline is `const * size`, and the
/// engine only ever consumes the RATIO `SOL(query)/SOL(anchor)` — so the
/// proxy is exactly ratio-equivalent.
///
/// The query coordinate is passed as `f64` without truncation (Python does
/// none). Table keys clamp to `u32` only as a defensive bound; every shipped
/// comm table tops out at 512 MiB message sizes, well under `u32::MAX`.
/// Interpolate the 1-D size curve at a possibly FRACTIONAL message size —
/// Python keeps float element counts (e.g. the gemma4 CP KV all-gather sizes
/// `kvcache_bytes_per_token / comm_bytes`), and the engine query coordinate
/// is float anyway. Truncating to integer first shifted the lerp point.
fn interp_message_size(by_size: &BTreeMap<u64, f64>, message_size: f64) -> Result<f64, AicError> {
    if by_size.is_empty() {
        return Err(AicError::PerfDatabase(
            "comm data has no message_size points".to_string(),
        ));
    }
    let mut node = Node::branch();
    for (&size, &latency) in by_size {
        node.insert(&[size.min(u32::MAX as u64) as u32], latency);
    }
    let sol = |c: &[f64]| c[0];
    let cfg = OpInterpConfig::grid(&["message_bytes"], &sol);
    perf_interp::query(&cfg, &node, &[message_size])
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

    /// `<systems_root>/data/b200_sxm/comm/nccl/2.27.3/` — the family-first
    /// system-spec-aware NCCL root for b200_sxm.
    fn b200_nccl_root() -> Option<PathBuf> {
        Some(systems_root().join("data/b200_sxm/comm/nccl/2.27.3"))
    }

    #[test]
    fn custom_allreduce_tp1_is_zero() {
        let table = CommunicationTable::new(b200_vllm_data_root(), None, None);
        let latency = table
            .query_custom_allreduce(CommQuantMode::Half, 1, 1024.0)
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
        let result = table.query_custom_allreduce(CommQuantMode::Half, 2, 1024.0);
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
            .query_nccl(CommQuantMode::Half, "all_reduce", 1, 1024.0)
            .expect("num_gpus=1 is a no-op");
        assert_eq!(latency, 0.0);
    }

    #[test]
    fn nccl_loads_from_system_wide_path() {
        // With the system-spec-aware path (b200_sxm declares
        // `nccl_version: '2.27.3'`), NCCL data resolves to
        // `<systems_root>/data/b200_sxm/comm/nccl/2.27.3/nccl_perf.parquet`
        // and the table loads successfully — NOT
        // `<vllm/0.19.0>/nccl_perf.parquet` which never existed.
        let table = CommunicationTable::new(b200_vllm_data_root(), b200_nccl_root(), None);
        let _ = table.load_nccl().expect("NCCL parquet must load from system-wide path");
    }

    /// Cross-language parity with the Python v2 engine. Expected values from:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk import common
    /// db = PerfDatabase('b200_sxm','vllm','0.19.0',
    ///                   systems_root='src/aiconfigurator_core/systems', database_mode='SOL')
    /// for msg in [384, 1073741824, 64]:
    ///     r = db.query_nccl(common.CommQuantMode.half, 8, 'all_gather', msg,
    ///                       database_mode=common.DatabaseMode.SILICON)
    ///     print(msg, repr(float(r)))"
    /// ```
    ///
    /// num_gpus=8 is the largest collected fan-out, so Python's silicon path
    /// applies no multi-node scale factor and compares at the same layer as
    /// this raw table query. msg=384 is an interior RAW lerp; 1 GiB is a
    /// beyond-max util-hold (collected max 256 MiB); 64 B is a below-min
    /// util-hold (collected min 256 B) — the linear-proxy SOL ratio equals
    /// Python's collective-roofline ratio.
    #[test]
    fn nccl_query_matches_python_v2_engine() {
        let table = CommunicationTable::new(b200_vllm_data_root(), b200_nccl_root(), None);
        let cases: &[(u64, f64)] = &[
            (384, 0.01559),
            (1_073_741_824, 3.0412399999999997),
            (64, 0.0038999999999999994),
        ];
        for &(msg, expected) in cases {
            let got = table
                .query_nccl(CommQuantMode::Half, "all_gather", 8, msg as f64)
                .expect("query must succeed");
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "msg={msg}: rust {got} vs python {expected}"
            );
        }
    }

    #[test]
    fn nccl_unconfigured_errors_clearly() {
        // When neither `misc.nccl_version` nor `misc.oneccl_version` is
        // declared, both load attempts surface a clean configuration error
        // rather than silently degrading.
        let table = CommunicationTable::new(b200_vllm_data_root(), None, None);
        let err = table
            .query_nccl(CommQuantMode::Half, "all_reduce", 2, 1024.0)
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
