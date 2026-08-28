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

use super::axis_curve::LeafAxisCurve;
use super::perf_interp::LeafValue;
use super::{kernel_source_ok, SourceResolver};
use crate::common::enums::CommQuantMode;
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use crate::perf_database::parquet_loader::PerfReader;

pub struct CommunicationTable {
    /// Legacy-shaped logical root used to resolve
    /// `comm/<backend>/<version>/custom_allreduce_perf.parquet`.
    data_root: PathBuf,
    /// Directory containing `nccl_perf.parquet`. Preferentially resolved as
    /// `<systems_root>/<data_dir>/comm/nccl/<misc.nccl_version>/`, with the
    /// legacy non-family path retained during the dual-read transition.
    /// `None` when the system YAML has no `misc.nccl_version` declared.
    nccl_root: Option<PathBuf>,
    /// Directory containing `oneccl_perf.parquet`. Preferentially resolved as
    /// `<systems_root>/<data_dir>/comm/oneccl/<misc.oneccl_version>/`, with
    /// the legacy non-family path retained during the dual-read transition.
    /// `None` when the system YAML has no `misc.oneccl_version` declared
    /// (most systems — OneCCL is the XPU fallback path).
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
    /// `(quant_name, tp_size)` -> immutable `u64` message-size leaf curve.
    by_keys: BTreeMap<(String, u32), LeafAxisCurve<u64>>,
}

struct NcclGrids {
    /// `(dtype_name, operation, num_gpus)` -> immutable `u64` message-size leaf curve.
    by_keys: BTreeMap<(String, String, u32), LeafAxisCurve<u64>>,
}

impl CommunicationTable {
    /// `data_root` is the legacy-shaped logical backend/version root used by
    /// the family-aware custom-allreduce resolver.
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
        Self::with_sources(
            data_root,
            nccl_root,
            oneccl_root,
            &SourceResolver::fixed(PerfDbSources::default()),
        )
        .expect("fixed-map resolution is infallible")
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
        resolver: &SourceResolver,
    ) -> Result<Self, AicError> {
        let custom_allreduce_sources =
            resolver.sources_for("custom_allreduce_perf.parquet", &data_root)?;
        Ok(Self {
            data_root,
            nccl_root,
            oneccl_root,
            custom_allreduce_sources,
            custom_allreduce: OnceLock::new(),
            nccl: OnceLock::new(),
            oneccl: OnceLock::new(),
        })
    }

    /// System-wide NCCL data dir (for the table view's primary-only load).
    pub(crate) fn nccl_root(&self) -> Option<&Path> {
        self.nccl_root.as_deref()
    }

    /// System-wide OneCCL data dir (vLLM/XPU systems only).
    pub(crate) fn oneccl_root(&self) -> Option<&Path> {
        self.oneccl_root.as_deref()
    }

    /// Raw custom-allreduce value (latency ms + power/energy), 1-D
    /// interpolated along `message_size`.
    ///
    /// `tp_size_effective` is the fan-out the caller wants to look up. For
    /// TP > num_gpus_per_node the `_scaled` caller passes the largest
    /// recorded beyond-node fan-out when the table has one (multi-node
    /// collection), else caps to `num_gpus_per_node` and applies a
    /// bandwidth scale separately.
    pub fn query_custom_allreduce(
        &self,
        quant: CommQuantMode,
        tp_size_effective: u32,
        message_size: f64,
    ) -> Result<LeafValue, AicError> {
        if tp_size_effective <= 1 {
            return Ok(LeafValue::latency_only(0.0));
        }
        let grids = self.load_custom_allreduce()?;
        let key = (quant.name().to_string(), tp_size_effective);
        let curve = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "custom_allreduce data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        interp_message_size(curve, message_size)
    }

    /// Custom-allreduce latency at a RAW tp_size, mirroring the full Python
    /// DB-level `_query_custom_allreduce_table.get_silicon`
    /// (operations/communication.py) so every consumer inherits the same
    /// semantics:
    ///   1. `tp == 1` -> 0;
    ///   2. GB200 NVL72 (`num_gpus_per_node == 72`) with `tp > 4` -> reroute
    ///      to NCCL all_reduce at the RAW tp (custom AR is only collected up
    ///      to tp4 there);
    ///   3. use the largest recorded fan-out <= tp (beyond-node fan-outs come
    ///      from multi-node collection), else clamp tp to the node size, and
    ///      interpolate the table;
    ///   4. beyond-recorded overflow: scale by the p2p-bandwidth ratio.
    pub fn query_custom_allreduce_scaled(
        &self,
        spec: &SystemSpec,
        quant: CommQuantMode,
        tp_size: u32,
        message_size: f64,
    ) -> Result<LeafValue, AicError> {
        if tp_size <= 1 {
            return Ok(LeafValue::latency_only(0.0));
        }
        let per_node = spec.node.num_gpus_per_node;
        if per_node == 72 && tp_size > 4 {
            return self.query_nccl_scaled(spec, quant, "all_reduce", tp_size, message_size);
        }
        // Recorded beyond-node fan-outs (multi-node collection) take
        // precedence over the clamp+bandwidth extrapolation: use the largest
        // recorded fan-out <= tp_size that exceeds the node size. Tables that
        // stop at or below the node size (the historical single-node
        // collection) keep the legacy clamp+scale behavior bit-for-bit.
        let extended = {
            let grids = self.load_custom_allreduce()?;
            let quant_name = quant.name();
            grids
                .by_keys
                .keys()
                .filter(|(q, n)| q.as_str() == quant_name && *n > per_node && *n <= tp_size)
                .map(|(_, n)| *n)
                .max()
        };
        let effective_tp = extended.unwrap_or_else(|| tp_size.min(per_node));
        let mut value = self.query_custom_allreduce(quant, effective_tp, message_size)?;
        if tp_size > effective_tp {
            let base_bw = spec.get_p2p_bandwidth(effective_tp);
            let target_bw = spec.get_p2p_bandwidth(tp_size);
            let f_tp = tp_size as f64;
            let f_eff = effective_tp as f64;
            let scale = (f_tp - 1.0) / f_tp * f_eff / (f_eff - 1.0).max(1.0) * base_bw / target_bw;
            // Python scales latency AND energy by the beyond-node factor.
            value.latency *= scale;
            value.energy *= scale;
        }
        Ok(value)
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
    ) -> Result<LeafValue, AicError> {
        if num_gpus <= 1 {
            return Ok(LeafValue::latency_only(0.0));
        }
        let max_recorded = self
            .nccl_max_num_gpus(dtype, operation)?
            .unwrap_or(num_gpus);
        let effective = num_gpus.min(max_recorded);
        let mut value = self.query_nccl(dtype, operation, effective, message_size)?;
        if num_gpus > max_recorded {
            let max_bw = spec.get_p2p_bandwidth(max_recorded);
            let req_bw = spec.get_p2p_bandwidth(num_gpus);
            let f_n = num_gpus as f64;
            let f_m = max_recorded as f64;
            let scale = (f_n - 1.0) / f_n * f_m / (f_m - 1.0).max(1.0) * max_bw / req_bw;
            // Python scales latency AND energy by the fan-out correction.
            value.latency *= scale;
            value.energy *= scale;
        }
        Ok(value)
    }

    /// Raw NCCL collective value (latency ms + power/energy).
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
    ) -> Result<LeafValue, AicError> {
        if num_gpus_effective <= 1 {
            return Ok(LeafValue::latency_only(0.0));
        }
        let key = (
            dtype.name().to_string(),
            operation.to_string(),
            num_gpus_effective,
        );

        if let Ok(grids) = self.load_nccl() {
            if let Some(curve) = grids.by_keys.get(&key) {
                return interp_message_size(curve, message_size);
            }
        }
        // Fall back to OneCCL.
        let grids = self.load_oneccl()?;
        let curve = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "neither NCCL nor OneCCL has data for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        interp_message_size(curve, message_size)
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
        let curve = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "custom_allreduce data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if curve.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "custom_allreduce data empty for {key:?} at {}",
                self.data_root.display()
            )));
        }
        Ok(curve
            .iter()
            .map(|(size, leaf)| (vec![size as f64], leaf.latency))
            .collect())
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
        let curve = grids.by_keys.get(&key).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "NCCL data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if curve.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "NCCL data empty for {key:?} at {}",
                self.data_root.display()
            )));
        }
        Ok(curve
            .iter()
            .map(|(size, leaf)| (vec![size as f64], leaf.latency))
            .collect())
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
        let cell = self
            .custom_allreduce
            .get_or_init(|| load_custom_allreduce_parquet(&self.custom_allreduce_sources));
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
/// none), while collected message-size keys retain their original `u64`
/// values. Interpolate the 1-D size curve at a possibly FRACTIONAL message
/// size — Python keeps float element counts (e.g. the gemma4 CP KV all-gather
/// sizes `kvcache_bytes_per_token / comm_bytes`), and the engine query
/// coordinate is float anyway. Truncating to integer first shifted the lerp
/// point.
fn interp_message_size(
    curve: &LeafAxisCurve<u64>,
    message_size: f64,
) -> Result<LeafValue, AicError> {
    curve.query(message_size, &|size| size)
}

fn insert_first_wins_message_point<K: Ord>(
    by_keys: &mut BTreeMap<K, BTreeMap<u64, LeafValue>>,
    key: K,
    message_size: u64,
    leaf: LeafValue,
) {
    by_keys
        .entry(key)
        .or_default()
        .entry(message_size)
        .or_insert(leaf);
}

fn load_custom_allreduce_parquet(sources: &[PerfSource]) -> Result<CustomAllReduceGrids, AicError> {
    let mut by_keys: BTreeMap<(String, u32), BTreeMap<u64, LeafValue>> = BTreeMap::new();
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
        let power_col = reader.col_optional("power");
        let kernel_source_col = reader.col_optional("kernel_source");
        let backend_col = reader.col_optional("backend");

        // Mirror Python/legacy: skip "_eager" kernel sources on systems other
        // than b60. We can't see the system name from here, so apply the filter
        // by path prefix.
        let path_str = path.to_string_lossy();
        let is_b60 = path_str.contains("/b60/");

        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), kernel_source_col, &row)? {
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
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // First-wins parity with Python `load_custom_allreduce_data`,
            // extended across shared-layer sources (earlier source wins).
            insert_first_wins_message_point(
                &mut by_keys,
                ("half".to_string(), row.u32(num_gpus_col)?),
                row.u64(message_size_col)?,
                LeafValue::with_power(latency, power),
            );
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(CustomAllReduceGrids {
        by_keys: by_keys
            .into_iter()
            .map(|(key, points)| (key, LeafAxisCurve::from_map("message_bytes", points)))
            .collect(),
    })
}

fn load_nccl_parquet(path: &Path) -> Result<NcclGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let op_name_col = reader.col("op_name")?;
    let nccl_dtype_col = reader.col("nccl_dtype")?;
    let num_gpus_col = reader.col("num_gpus")?;
    let message_size_col = reader.col("message_size")?;
    let latency_col = reader.col("latency")?;
    let power_col = reader.col_optional("power");

    let mut by_keys: BTreeMap<(String, String, u32), BTreeMap<u64, LeafValue>> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let latency = row.f64(latency_col)?;
        let power = row.f64_optional(power_col)?.unwrap_or(0.0);
        // First-wins parity with Python `load_nccl_data`.
        insert_first_wins_message_point(
            &mut by_keys,
            (
                row.str_owned(nccl_dtype_col)?,
                row.str_owned(op_name_col)?,
                row.u32(num_gpus_col)?,
            ),
            row.u64(message_size_col)?,
            LeafValue::with_power(latency, power),
        );
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no NCCL/OneCCL rows loaded from {}",
            path.display()
        )));
    }
    Ok(NcclGrids {
        by_keys: by_keys
            .into_iter()
            .map(|(key, points)| (key, LeafAxisCurve::from_map("message_bytes", points)))
            .collect(),
    })
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
    fn message_size_curve_matches_python_grid() {
        let points = BTreeMap::from([(256, 1.25), (1024, 2.75), (4096, 5.5)]);
        let curve = latency_curve(points);

        for (message_size, expected) in [
            (64.0_f64, 0.3125_f64),
            (256.0, 1.25),
            (640.5, 2.0009765625),
            (1024.0, 2.75),
            (2048.25, 3.6668904622395835),
            (4096.0, 5.5),
            (8192.0, 11.0),
        ] {
            let actual = interp_message_size(&curve, message_size).unwrap();
            assert_eq!(
                actual.latency.to_bits(),
                expected.to_bits(),
                "message_size={message_size}"
            );
        }

        let curve = latency_curve(BTreeMap::from([(1024, 3.0)]));
        for (message_size, expected) in [(512.0_f64, 1.5_f64), (1024.0, 3.0), (2048.0, 6.0)] {
            let actual = interp_message_size(&curve, message_size).unwrap();
            assert_eq!(actual.latency.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn message_size_curve_preserves_errors_and_u64_coordinates() {
        let empty_curve = latency_curve(BTreeMap::new());
        assert_eq!(
            interp_message_size(&empty_curve, 1024.0)
                .unwrap_err()
                .to_string(),
            "perf database error: perf_interp: no data to anchor query \
             {message_bytes=1024} (empty table)"
        );

        let invalid_curve = latency_curve(BTreeMap::from([(1024_u64, 0.0)]));
        assert_eq!(
            interp_message_size(&invalid_curve, 2048.0)
                .unwrap_err()
                .to_string(),
            "perf database error: perf_interp: no data to anchor query \
             {message_bytes=2048} (no positive-util boundary anchor)"
        );

        let first_oversized = u64::from(u32::MAX) + 1;
        let second_oversized = first_oversized + 1;
        let curve = latency_curve(BTreeMap::from([
            (1024, 1.0),
            (first_oversized, 2.0),
            (second_oversized, 3.0),
        ]));
        assert_eq!(
            curve
                .iter()
                .map(|(size, leaf)| (size, leaf.latency))
                .collect::<Vec<_>>(),
            vec![(1024, 1.0), (first_oversized, 2.0), (second_oversized, 3.0)]
        );

        // Oracle values from Python perf_interp Grid with the same integer
        // coordinates and linear message-size SOL.
        for (message_size, expected) in [
            (first_oversized as f64, 2.0_f64),
            (first_oversized as f64 + 0.5, 2.5),
            (second_oversized as f64, 3.0),
            ((second_oversized * 2) as f64, 6.0),
        ] {
            let actual = interp_message_size(&curve, message_size).unwrap();
            assert_eq!(actual.latency.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn empirical_points_preserve_distinct_u64_coordinates() {
        let first_oversized = u64::from(u32::MAX) + 1;
        let second_oversized = first_oversized + 1;
        let points = BTreeMap::from([(1024, 1.0), (first_oversized, 2.0), (second_oversized, 3.0)]);
        let custom_key = ("half".to_string(), 4);
        let nccl_key = ("half".to_string(), "all_reduce".to_string(), 4);
        let table = table_with_loaded_collectives(
            BTreeMap::from([(custom_key, latency_curve(points.clone()))]),
            BTreeMap::from([(nccl_key, latency_curve(points))]),
            BTreeMap::new(),
        );
        let expected = vec![
            (vec![1024.0], 1.0),
            (vec![first_oversized as f64], 2.0),
            (vec![second_oversized as f64], 3.0),
        ];
        assert_eq!(
            table
                .custom_allreduce_points(CommQuantMode::Half, 4)
                .unwrap(),
            expected
        );
        assert_eq!(
            table
                .nccl_empirical_points(CommQuantMode::Half, "all_reduce", 4)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn custom_allreduce_preserves_first_source_and_first_row_precedence() {
        let key = ("half".to_string(), 4);
        let mut by_keys = BTreeMap::new();
        insert_first_wins_message_point(
            &mut by_keys,
            key.clone(),
            1024,
            LeafValue::with_power(1.0, 10.0),
        );
        insert_first_wins_message_point(
            &mut by_keys,
            key.clone(),
            1024,
            LeafValue::with_power(2.0, 20.0),
        );
        insert_first_wins_message_point(
            &mut by_keys,
            key.clone(),
            1024,
            LeafValue::with_power(3.0, 30.0),
        );
        insert_first_wins_message_point(
            &mut by_keys,
            key.clone(),
            2048,
            LeafValue::with_power(4.0, 40.0),
        );
        let curve = LeafAxisCurve::from_map("message_bytes", by_keys.remove(&key).unwrap());
        assert_eq!(
            interp_message_size(&curve, 1024.0).unwrap(),
            LeafValue::with_power(1.0, 10.0)
        );
        assert_eq!(
            interp_message_size(&curve, 2048.0).unwrap(),
            LeafValue::with_power(4.0, 40.0)
        );
    }

    /// Wrap plain latency points into a leaf message-size curve.
    fn latency_curve(points: BTreeMap<u64, f64>) -> LeafAxisCurve<u64> {
        LeafAxisCurve::from_map(
            "message_bytes",
            points
                .into_iter()
                .map(|(size, latency)| (size, LeafValue::latency_only(latency)))
                .collect(),
        )
    }

    fn table_with_loaded_collectives(
        custom_allreduce: BTreeMap<(String, u32), LeafAxisCurve<u64>>,
        nccl: BTreeMap<(String, String, u32), LeafAxisCurve<u64>>,
        oneccl: BTreeMap<(String, String, u32), LeafAxisCurve<u64>>,
    ) -> CommunicationTable {
        let custom_allreduce_cell = OnceLock::new();
        assert!(custom_allreduce_cell
            .set(Ok(CustomAllReduceGrids {
                by_keys: custom_allreduce
            }))
            .is_ok());
        let nccl_cell = OnceLock::new();
        assert!(nccl_cell.set(Ok(NcclGrids { by_keys: nccl })).is_ok());
        let oneccl_cell = OnceLock::new();
        assert!(oneccl_cell.set(Ok(NcclGrids { by_keys: oneccl })).is_ok());
        CommunicationTable {
            data_root: PathBuf::from("synthetic"),
            nccl_root: None,
            oneccl_root: None,
            custom_allreduce_sources: Vec::new(),
            custom_allreduce: custom_allreduce_cell,
            nccl: nccl_cell,
            oneccl: oneccl_cell,
        }
    }

    #[test]
    fn nccl_primary_and_oneccl_fallback_use_frozen_curves() {
        let key = ("half".to_string(), "all_reduce".to_string(), 4);
        let primary = BTreeMap::from([(key.clone(), latency_curve(BTreeMap::from([(1024, 1.0)])))]);
        let fallback =
            BTreeMap::from([(key.clone(), latency_curve(BTreeMap::from([(1024, 2.0)])))]);
        let table = table_with_loaded_collectives(BTreeMap::new(), primary, fallback.clone());
        assert_eq!(
            table
                .query_nccl(CommQuantMode::Half, "all_reduce", 4, 1024.0)
                .unwrap(),
            LeafValue::latency_only(1.0)
        );

        let table = table_with_loaded_collectives(BTreeMap::new(), BTreeMap::new(), fallback);
        assert_eq!(
            table
                .query_nccl(CommQuantMode::Half, "all_reduce", 4, 1024.0)
                .unwrap(),
            LeafValue::latency_only(2.0)
        );
    }

    #[test]
    fn custom_allreduce_tp1_is_zero() {
        let table = CommunicationTable::new(b200_vllm_data_root(), None, None);
        let value = table
            .query_custom_allreduce(CommQuantMode::Half, 1, 1024.0)
            .expect("tp=1 is a no-op");
        assert_eq!(value.latency, 0.0);
        assert_eq!(value.energy, 0.0);
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
            Ok(value) => assert!(value.latency > 0.0, "expected positive latency"),
            Err(AicError::PerfDatabase(_)) => {
                // Tp=2 may not be in this dataset — acceptable failure mode.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn custom_allreduce_scaled_uses_recorded_beyond_node_fanouts() {
        let spec = SystemSpec::load(&systems_root().join("gb200.yaml"))
            .expect("gb200.yaml parse must succeed");
        assert_eq!(
            spec.node.num_gpus_per_node, 4,
            "test assumes gb200 models 4-GPU nodes"
        );
        let curve = |lat: f64| latency_curve(BTreeMap::from([(1024, lat)]));
        // Table with recorded beyond-node fan-outs (multi-node collection).
        let multi = BTreeMap::from([
            (("half".to_string(), 2), curve(0.002)),
            (("half".to_string(), 4), curve(0.004)),
            (("half".to_string(), 8), curve(0.030)),
            (("half".to_string(), 16), curve(0.048)),
        ]);
        let table = table_with_loaded_collectives(multi, BTreeMap::new(), BTreeMap::new());
        // tp16: exact 16-rank lookup, no scaling.
        let v16 = table
            .query_custom_allreduce_scaled(&spec, CommQuantMode::Half, 16, 1024.0)
            .unwrap();
        assert!((v16.latency - 0.048).abs() < 1e-12);
        // tp12: largest recorded beyond-node fan-out <= 12 is 8, scaled beyond.
        let v12 = table
            .query_custom_allreduce_scaled(&spec, CommQuantMode::Half, 12, 1024.0)
            .unwrap();
        let scale12 =
            11.0 / 12.0 * 8.0 / 7.0 * spec.get_p2p_bandwidth(8) / spec.get_p2p_bandwidth(12);
        assert!((v12.latency - 0.030 * scale12).abs() < 1e-12);
        // Legacy table (fan-outs stop at the node size): clamp+scale unchanged.
        let legacy = BTreeMap::from([
            (("half".to_string(), 2), curve(0.002)),
            (("half".to_string(), 4), curve(0.004)),
        ]);
        let table = table_with_loaded_collectives(legacy, BTreeMap::new(), BTreeMap::new());
        let v_legacy = table
            .query_custom_allreduce_scaled(&spec, CommQuantMode::Half, 16, 1024.0)
            .unwrap();
        let scale_legacy =
            15.0 / 16.0 * 4.0 / 3.0 * spec.get_p2p_bandwidth(4) / spec.get_p2p_bandwidth(16);
        assert!((v_legacy.latency - 0.004 * scale_legacy).abs() < 1e-12);
    }

    #[test]
    fn nccl_num_gpus_1_is_zero() {
        let table = CommunicationTable::new(b200_vllm_data_root(), None, None);
        let value = table
            .query_nccl(CommQuantMode::Half, "all_reduce", 1, 1024.0)
            .expect("num_gpus=1 is a no-op");
        assert_eq!(value.latency, 0.0);
        assert_eq!(value.energy, 0.0);
    }

    #[test]
    fn nccl_loads_from_system_wide_path() {
        // With the system-spec-aware path (b200_sxm declares
        // `nccl_version: '2.27.3'`), NCCL data resolves to
        // `<systems_root>/data/b200_sxm/comm/nccl/2.27.3/nccl_perf.parquet`
        // and the table loads successfully — NOT
        // `<vllm/0.19.0>/nccl_perf.parquet` which never existed.
        let table = CommunicationTable::new(b200_vllm_data_root(), b200_nccl_root(), None);
        let _ = table
            .load_nccl()
            .expect("NCCL parquet must load from system-wide path");
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
                .expect("query must succeed")
                .latency;
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

    /// ENERGY oracle on a synthetic power-carrying fixture. Python twin
    /// (pandas fixture at `data/nccl/test/nccl_perf.parquet`,
    /// `energy_test_fixtures` spec with `misc.nccl_version: test`):
    ///
    /// ```text
    /// db.query_nccl(CommQuantMode.half, 8, "all_gather", 1536, SILICON)
    /// # -> latency=2.0, energy=300.0
    /// ```
    #[test]
    fn nccl_energy_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{write_parquet, Col};
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &tmp.path().join("nccl_perf.parquet"),
            &[
                Col::Str("nccl_dtype", vec!["half", "half"]),
                Col::Str("op_name", vec!["all_gather", "all_gather"]),
                Col::I64("num_gpus", vec![8, 8]),
                Col::I64("message_size", vec![1024, 2048]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        let table = CommunicationTable::new(
            tmp.path().to_path_buf(),
            Some(tmp.path().to_path_buf()),
            None,
        );
        let v = table
            .query_nccl(CommQuantMode::Half, "all_gather", 8, 1536.0)
            .unwrap();
        assert!((v.latency - 2.0).abs() < 1e-9, "latency {}", v.latency);
        assert!(
            (v.energy - 300.0).abs() < 1e-9 * 300.0,
            "energy {}",
            v.energy
        );
    }
}
