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
//! Resolution mirrors Python v2's `_resolve_tokens`: the token curve rides
//! the shared `perf_interp` engine (1-axis Grid, RAW lerp in range; beyond
//! the collected range the boundary util is held with `k_tail=1` and the
//! caller-supplied MoE roofline SOL carries the growth — unclamped util,
//! exactly like Python which deleted the hand-rolled overflow estimator).
//! The SOL closure comes from the operator layer (`operators/moe.rs`),
//! which owns the roofline math.
//!
//! Singleton-underflow contract (Python `_require_moe_token_points`): a
//! curve with a single token point queried BELOW that point is a structured
//! miss — one large-token row cannot define the low-token launch floor.
//!
//! `workload_distribution` falls back to `"uniform"` when the requested
//! variant is absent for the given quant, matching Python's behavior.
//!
//! WideEP / DeepEP / TRT-LLM all-to-all variants live in
//! `perf_database::wideep`, `wideep_mla`, and `wideep_moe`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use super::moe_index::{MoeIndex, MoeShapeKey};
use super::perf_interp::LeafValue;
use super::{kernel_source_ok, resolve_op_sources};
use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use crate::perf_database::parquet_loader::PerfReader;

/// Power-carrying twin of [`super::token_curve::TokenCurve`]: an immutable
/// one-axis token curve over measured `{latency, power, energy}` leaves,
/// specialized away from the generic nested-`Node` engine but preserving
/// `perf_interp::query_value` semantics bit-for-bit (exact hit returns the
/// leaf verbatim; in-range blends lerp latency and blend-power and re-derive
/// `energy = power * latency`; boundary util-holds scale latency by the SOL
/// ratio while power holds at the anchor — "energy scales with latency",
/// mirroring Python `_resolve_tokens`). Shared by the MoE table itself and
/// the mHC/MegaMoE tables; the WideEP token-curve families stay
/// latency-only on `TokenCurve` by design (see the wideep module docs).
#[derive(Clone, Debug, Default)]
pub(crate) struct LeafTokenCurve {
    points: Box<[(u32, LeafValue)]>,
}

impl LeafTokenCurve {
    pub(crate) fn from_map(points: BTreeMap<u32, LeafValue>) -> Self {
        Self {
            points: points.into_iter().collect(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub(crate) fn get(&self, token: u32) -> Option<LeafValue> {
        self.points
            .binary_search_by_key(&token, |&(token, _)| token)
            .ok()
            .map(|index| self.points[index].1)
    }

    pub(crate) fn iter(&self) -> impl DoubleEndedIterator<Item = (u32, LeafValue)> + '_ {
        self.points.iter().copied()
    }

    /// Python `_require_moe_token_points`: a singleton curve queried below
    /// its only measured point is a structured miss (it cannot define the
    /// low-token launch-overhead regime). Multi-point underflow and
    /// singleton overflow go to the engine's util-hold unchanged.
    pub(crate) fn singleton_underflow(&self, num_tokens: u32) -> Option<u32> {
        if self.points.len() == 1 && num_tokens < self.points[0].0 {
            Some(self.points[0].0)
        } else {
            None
        }
    }

    /// Resolve with the one-axis `perf_interp` Grid contract on full
    /// leaves: exact hits verbatim, raw interpolation within the measured
    /// range (latency and blend-power lerped with the same weight, energy
    /// re-derived as `power * latency`), and a boundary-util hold outside
    /// it with `k_tail=1` (latency scales by the SOL ratio; power holds at
    /// the anchor's blend power).
    pub(crate) fn query(
        &self,
        num_tokens: f64,
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        if self.points.is_empty() {
            return Err(miss(num_tokens, "empty table"));
        }

        if let Some(token) = exact_token(num_tokens) {
            if let Some(leaf) = self.get(token) {
                return Ok(leaf);
            }
        }

        let upper = self
            .points
            .partition_point(|&(token, _)| f64::from(token) < num_tokens);
        if upper == 0 || upper == self.points.len() {
            let anchor = if upper == 0 {
                self.points[0]
            } else {
                self.points[self.points.len() - 1]
            };
            let anchor_sol = sol(f64::from(anchor.0));
            if anchor.1.latency.is_nan()
                || anchor.1.latency <= 0.0
                || anchor_sol.is_nan()
                || anchor_sol <= 0.0
            {
                return Err(miss(num_tokens, "no positive-util boundary anchor"));
            }
            let query_sol = sol(num_tokens);
            if query_sol.is_nan() || query_sol <= 0.0 {
                return Err(miss(num_tokens, "non-positive SOL at query"));
            }
            let latency = query_sol / (anchor_sol / anchor.1.latency);
            let power = anchor.1.blend_power();
            return Ok(LeafValue {
                latency,
                power,
                energy: power * latency,
            });
        }

        let lower = self.points[upper - 1];
        let upper = self.points[upper];
        let weight = (num_tokens - f64::from(lower.0)) / f64::from(upper.0 - lower.0);
        let latency = lower.1.latency + (upper.1.latency - lower.1.latency) * weight;
        let lower_power = lower.1.blend_power();
        let power = lower_power + (upper.1.blend_power() - lower_power) * weight;
        Ok(LeafValue {
            latency,
            power,
            energy: power * latency,
        })
    }
}

fn exact_token(value: f64) -> Option<u32> {
    if value >= 0.0 && value <= f64::from(u32::MAX) && value.fract() == 0.0 {
        Some(value as u32)
    } else {
        None
    }
}

fn miss(num_tokens: f64, reason: &str) -> AicError {
    AicError::PerfDatabase(format!(
        "perf_interp: no data to anchor query {{num_tokens={num_tokens}}} ({reason})"
    ))
}

pub struct MoeTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for the MoE perf file (shared-layer
    /// aware; see [`PerfSource`]). Single-primary, no-filter by default
    /// (`MoeTable::new`).
    moe_sources: Vec<PerfSource>,
    moe: OnceLock<Result<LoadedMoeGrids, AicError>>,
}

/// Which kernel grid a MoE accessor addresses: the default table or the
/// TRT-LLM `moe_torch_flow_min_latency` low-latency split (Python's
/// `_moe_data` vs `_moe_low_latency_data`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeKernel {
    Standard,
    LowLatency,
}

/// One collected sibling slice of the MoE table for a fixed
/// `(quant, distribution, moe_tp, moe_ep)`: the categorical shape features
/// plus its `num_tokens -> latency_ms` curve. Consumed by the operator
/// layer's cross-shape/cross-quant transfer ladder (the algorithm lives in
/// `operators/moe.rs`; this is a data accessor payload only).
#[derive(Clone, Debug)]
pub struct MoeSiblingSlice {
    pub topk: u32,
    pub num_experts: u32,
    pub hidden_size: u32,
    pub inter_size: u32,
    /// `(num_tokens, latency_ms)` in ascending token order.
    pub points: Vec<(u32, f64)>,
}

/// Two parallel grids split by `kernel_source`. Mirrors Python's split in
/// `aiconfigurator.sdk.operations.moe.MoE.load_data`, where rows tagged
/// `kernel_source == "moe_torch_flow_min_latency"` route to a separate
/// accumulator that the TRT-LLM SILICON path probes first for small-token
/// nvfp4 gated MoE queries.
struct LoadedMoeGrids {
    default: MoeGrids,
    low_latency: MoeGrids,
}

struct MoeGrids {
    index: MoeIndex<MoeShapeKey, LeafTokenCurve>,
    /// Distinct quant names in first-seen (file row) order. Python's
    /// transfer ladder iterates the table dict in INSERTION order
    /// (`for q in moe_table`), which breaks profile-distance ties by file
    /// order — live on shards whose file order differs from sorted order
    /// (e.g. b200/vllm/0.24.0 lists `fp8_block` before `fp8`).
    quants_in_load_order: Vec<String>,
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

impl MoeKey {
    fn from_shape(quant: &str, distribution: &str, shape: MoeShapeKey) -> Self {
        Self {
            quant: quant.to_string(),
            distribution: distribution.to_string(),
            topk: shape.topk,
            num_experts: shape.num_experts,
            hidden_size: shape.hidden_size,
            inter_size: shape.inter_size,
            moe_tp_size: shape.moe_tp_size,
            moe_ep_size: shape.moe_ep_size,
        }
    }
}

impl MoeTable {
    /// Construct an empty table for the given data directory. No I/O. The MoE
    /// perf file is sourced solely from `data_root/moe_perf.parquet` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). The MoE file falls back to its
    /// primary `data_root/moe_perf.parquet` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let moe_sources = resolve_op_sources(perf_db_sources, "moe_perf.parquet", &data_root);
        Self {
            data_root,
            moe_sources,
            moe: OnceLock::new(),
        }
    }

    /// Raw MoE value (latency ms + power/energy) via the perf_interp v2
    /// engine contract (1-axis token curve): exact hit / RAW lerp in range;
    /// beyond the collected range the boundary util is held (`k_tail=1`,
    /// unclamped) and `sol` — the operator layer's MoE roofline — carries
    /// the growth. Mirrors Python `MoE._query_moe_table._resolve_tokens`.
    ///
    /// Falls back to the `"uniform"` distribution if the requested
    /// distribution is absent for the given quant mode. A singleton curve
    /// queried below its only point is a structured miss (Python
    /// `_require_moe_token_points`).
    #[allow(clippy::too_many_arguments)]
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
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        let loaded = self.load()?;
        let grids = &loaded.default;
        let quant_name = quant.name();

        let shape = MoeShapeKey {
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let (dist, by_tokens) =
            grids
                .index
                .resolve_uniform(quant_name, workload_distribution, &shape);
        let by_tokens = by_tokens.ok_or_else(|| {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            AicError::PerfDatabase(format!(
                "MoE data missing for {key:?} at {}",
                self.data_root.display()
            ))
        })?;
        if by_tokens.is_empty() {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            return Err(AicError::PerfDatabase(format!(
                "MoE data has no token points for {key:?} at {}",
                self.data_root.display()
            )));
        }
        if let Some(only) = by_tokens.singleton_underflow(num_tokens) {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            return Err(AicError::PerfDatabase(format!(
                "MoE silicon token underflow has only one measured point; cannot infer \
                 low-token latency from a singleton. num_tokens={num_tokens}, \
                 measured_token={only}, key={key:?}"
            )));
        }
        by_tokens.query(num_tokens as f64, sol)
    }

    /// Probe the TRT-LLM low-latency NVFP4 MoE kernel table.
    ///
    /// Returns `Ok(Some(value))` when the loaded `low_latency` grid
    /// contains a matching `(quant, distribution-after-uniform-fallback,
    /// topk, num_experts, hidden, inter, moe_tp, moe_ep)` entry, and
    /// `Ok(None)` when the shape is absent — the caller should then fall
    /// through to `query()` (the default grid).
    ///
    /// Mirrors Python's small-token nvfp4 gated-MoE branch in
    /// `MoE._query_moe_table`: the low-latency table is consulted with a
    /// try/except that falls back to `_moe_data` when the SHAPE is absent
    /// (`Ok(None)` here). A singleton-underflow on a present shape is an
    /// `Err` (structured miss), not a fallback — in Python the guard fires
    /// inside `_resolve_tokens`, after the ll table has been selected.
    #[allow(clippy::too_many_arguments)]
    pub fn query_low_latency(
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
        sol: &dyn Fn(f64) -> f64,
    ) -> Result<Option<LeafValue>, AicError> {
        let loaded = self.load()?;
        let grids = &loaded.low_latency;
        if grids.index.is_empty() {
            return Ok(None);
        }
        let quant_name = quant.name();
        let shape = MoeShapeKey {
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let (dist, by_tokens) =
            grids
                .index
                .resolve_uniform(quant_name, workload_distribution, &shape);
        let Some(by_tokens) = by_tokens else {
            return Ok(None);
        };
        if by_tokens.is_empty() {
            return Ok(None);
        }
        if let Some(only) = by_tokens.singleton_underflow(num_tokens) {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            return Err(AicError::PerfDatabase(format!(
                "MoE low-latency token underflow has only one measured point; cannot infer \
                 low-token latency from a singleton. num_tokens={num_tokens}, \
                 measured_token={only}, key={key:?}"
            )));
        }
        by_tokens.query(num_tokens as f64, sol).map(Some)
    }

    /// `true` iff the loaded low-latency grid has any rows.
    ///
    /// Older perf-DB versions predate the `kernel_source` column, so the
    /// low-latency accumulator stays empty and the small-token nvfp4 gate
    /// is short-circuited at the operator layer.
    pub fn low_latency_available(&self) -> Result<bool, AicError> {
        let loaded = self.load()?;
        Ok(!loaded.low_latency.index.is_empty())
    }

    /// Own-slice `num_tokens -> latency_ms` curve for a full MoE key, after
    /// the per-quant `"uniform"` distribution fallback. A typed miss
    /// (`AicError::PerfDatabase`) means the slice is absent or empty —
    /// mirroring Python `util_empirical.require_data_slice` as used by the
    /// empirical own-shape grid (`_slice`) and the low-latency table probe
    /// (`_moe_table`) in `MoE._query_moe_table`.
    #[allow(clippy::too_many_arguments)]
    pub fn slice_points(
        &self,
        kernel: MoeKernel,
        quant_name: &str,
        workload_distribution: &str,
        topk: u32,
        num_experts: u32,
        hidden_size: u32,
        inter_size: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<(u32, f64)>, AicError> {
        let grids = self.grids_for(kernel)?;
        let shape = MoeShapeKey {
            topk,
            num_experts,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let (dist, by_tokens) =
            grids
                .index
                .resolve_uniform(quant_name, workload_distribution, &shape);
        let by_tokens = by_tokens.filter(|curve| !curve.is_empty()).ok_or_else(|| {
            let key = MoeKey::from_shape(quant_name, dist, shape);
            AicError::PerfDatabase(format!(
                "MoE data missing for {key:?} ({kernel:?}) at {}",
                self.data_root.display()
            ))
        })?;
        Ok(by_tokens
            .iter()
            .map(|(t, leaf)| (t, leaf.latency))
            .collect())
    }

    /// All collected sibling slices for `(quant, distribution-after-uniform-
    /// fallback, moe_tp, moe_ep)`; empty curves skipped, an empty result is
    /// data (not an error). Mirrors the enumeration in Python `_collect`
    /// (`MoE._query_moe_table`), which walks the nested
    /// `topk -> num_experts -> hidden -> inter` dicts. NOTE: Python yields
    /// dict insertion (file row) order; the `BTreeMap` yields sorted
    /// `(topk, num_experts, hidden, inter)` order instead — observable only
    /// through exact ties in nearest-candidate selection.
    pub fn sibling_slices(
        &self,
        kernel: MoeKernel,
        quant_name: &str,
        workload_distribution: &str,
        moe_tp_size: u32,
        moe_ep_size: u32,
    ) -> Result<Vec<MoeSiblingSlice>, AicError> {
        let grids = self.grids_for(kernel)?;
        let (_, by_shape) = grids
            .index
            .resolve_uniform_shapes(quant_name, workload_distribution);
        let mut slices = Vec::new();
        let Some(by_shape) = by_shape else {
            return Ok(slices);
        };
        for (shape, curve) in by_shape {
            if shape.moe_tp_size != moe_tp_size
                || shape.moe_ep_size != moe_ep_size
                || curve.is_empty()
            {
                continue;
            }
            slices.push(MoeSiblingSlice {
                topk: shape.topk,
                num_experts: shape.num_experts,
                hidden_size: shape.hidden_size,
                inter_size: shape.inter_size,
                points: curve.iter().map(|(t, leaf)| (t, leaf.latency)).collect(),
            });
        }
        Ok(slices)
    }

    /// Distinct quant names present in the kernel grid, in first-seen
    /// (file row) order — Python iterates the table dict in insertion
    /// order (`for q in moe_table`), and the transfer ladder's stable
    /// profile-distance sort breaks ties by that order.
    pub fn available_quants(&self, kernel: MoeKernel) -> Result<Vec<String>, AicError> {
        Ok(self.grids_for(kernel)?.quants_in_load_order.clone())
    }

    fn grids_for(&self, kernel: MoeKernel) -> Result<&MoeGrids, AicError> {
        let loaded = self.load()?;
        Ok(match kernel {
            MoeKernel::Standard => &loaded.default,
            MoeKernel::LowLatency => &loaded.low_latency,
        })
    }

    fn load(&self) -> Result<&LoadedMoeGrids, AicError> {
        let cell = self.moe.get_or_init(|| load_moe_parquet(&self.moe_sources));
        cell.as_ref().map_err(clone_err)
    }
}

/// Load the MoE table from an ordered, priority-sorted source list. Sources are
/// read in order; the first source containing a `(shape, num_tokens)` tuple wins
/// (`or_insert`), mirroring Python's `_read_filtered_rows` concatenation +
/// `load_moe_data` skip-on-key-conflict. Missing files are skipped (a sibling
/// declared in the manifest need not exist for every system); an error is
/// returned only when no source yields rows.
fn load_moe_parquet(sources: &[PerfSource]) -> Result<LoadedMoeGrids, AicError> {
    let mut default_index: MoeIndex<MoeShapeKey, BTreeMap<u32, LeafValue>> = MoeIndex::default();
    let mut low_latency_index: MoeIndex<MoeShapeKey, BTreeMap<u32, LeafValue>> =
        MoeIndex::default();
    let mut default_quants: Vec<String> = Vec::new();
    let mut low_latency_quants: Vec<String> = Vec::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let moe_dtype_col = reader.col("moe_dtype")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let inter_size_col = reader.col("inter_size")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let moe_tp_size_col = reader.col("moe_tp_size")?;
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let distribution_col = reader.col("distribution")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        // Optional in older perf-DB versions; when absent every row falls into
        // the `default` grid (matching the pre-split behavior). The same column
        // gates the per-source shared-layer `kernel_source` allowlist.
        let kernel_source_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), kernel_source_col, &row)? {
                continue;
            }
            let kernel_source = row
                .str_optional(kernel_source_col)?
                .unwrap_or("")
                .to_string();
            // Kernel-specific mxfp4 remaps (mirror Python `load_moe_data`):
            // the collector logs two distinct kernels under one `moe_dtype`;
            // route them to dedicated quant modes so DeepSeek-V4 modeling can
            // select the right one per GPU generation.
            //  - Blackwell trtllm-gen MXFP4xMXFP8:
            //    w4a8_mxfp4_mxfp8 + sglang_mxfp4_flashinfer_trtllm_moe
            //      -> w4a8_mxfp4_mxfp8_trtllm
            //  - Hopper flashinfer cutlass SM90 mixed-GEMM:
            //    w4a16_mxfp4 + sglang_flashinfer_cutlass_moe
            //      -> w4a16_mxfp4_cutlass
            let raw_quant = row.str_owned(moe_dtype_col)?;
            let quant = match (raw_quant.as_str(), kernel_source.as_str()) {
                ("w4a8_mxfp4_mxfp8", "sglang_mxfp4_flashinfer_trtllm_moe") => {
                    "w4a8_mxfp4_mxfp8_trtllm".to_string()
                }
                ("w4a16_mxfp4", "sglang_flashinfer_cutlass_moe") => {
                    "w4a16_mxfp4_cutlass".to_string()
                }
                _ => raw_quant,
            };
            let distribution = row.str_owned(distribution_col)?;
            let shape = MoeShapeKey {
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                inter_size: row.u32(inter_size_col)?,
                moe_tp_size: row.u32(moe_tp_size_col)?,
                moe_ep_size: row.u32(moe_ep_size_col)?,
            };
            let (target, target_quants) = if kernel_source == "moe_torch_flow_min_latency" {
                (&mut low_latency_index, &mut low_latency_quants)
            } else {
                (&mut default_index, &mut default_quants)
            };
            // First-seen (file row) quant order — Python's dict insertion
            // order, consumed by `available_quants`.
            if !target_quants.iter().any(|q| q == &quant) {
                target_quants.push(quant.clone());
            }
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // Python's `load_moe_data` wraps the leaf insert in a try/except KeyError
            // and skips on conflict, i.e. it keeps the FIRST occurrence of each
            // (shape, num_tokens) tuple. Some perf files contain duplicate rows
            // (same kernel_source, same shape) — preserving first-wins parity here,
            // extended across shared-layer sources (earlier source wins).
            target
                .entry(quant, distribution, shape)
                .entry(row.u32(num_tokens_col)?)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || (default_index.is_empty() && low_latency_index.is_empty()) {
        return Err(AicError::PerfDatabase(format!(
            "no rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(LoadedMoeGrids {
        default: MoeGrids {
            index: default_index.map_values(LeafTokenCurve::from_map),
            quants_in_load_order: default_quants,
        },
        low_latency: MoeGrids {
            index: low_latency_index.map_values(LeafTokenCurve::from_map),
            quants_in_load_order: low_latency_quants,
        },
    })
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
            .join("src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.19.0")
    }

    #[test]
    fn moe_table_loads_b200_vllm() {
        let table = MoeTable::new(b200_vllm_data_root());
        let _ = table.load().expect("moe_perf.parquet must load");
    }

    /// Linear token proxy — fine for key-selection tests where only the
    /// resolution path (not the extrapolated value) matters.
    fn proxy_sol(t: f64) -> f64 {
        t
    }

    #[test]
    fn moe_index_resolves_requested_and_uniform_distributions() {
        let shape = MoeShapeKey {
            topk: 2,
            num_experts: 8,
            hidden_size: 4096,
            inter_size: 2048,
            moe_tp_size: 1,
            moe_ep_size: 4,
        };
        let mut index = MoeIndex::default();
        *index.entry("fp8".into(), "power_law".into(), shape) =
            LeafTokenCurve::from_map(BTreeMap::from([(1, LeafValue::latency_only(1.0))]));
        *index.entry("fp8".into(), "uniform".into(), shape) =
            LeafTokenCurve::from_map(BTreeMap::from([(1, LeafValue::latency_only(2.0))]));
        let grids = MoeGrids {
            index,
            quants_in_load_order: vec!["fp8".to_string()],
        };

        let (dist, curve) = grids.index.resolve_uniform("fp8", "power_law", &shape);
        assert_eq!(dist, "power_law");
        assert_eq!(curve.unwrap().get(1).map(|leaf| leaf.latency), Some(1.0));

        let (dist, curve) = grids.index.resolve_uniform("fp8", "missing", &shape);
        assert_eq!(dist, "uniform");
        assert_eq!(curve.unwrap().get(1).map(|leaf| leaf.latency), Some(2.0));
    }

    /// The specialized curve must be indistinguishable from the generic
    /// engine's `query_value` on power-carrying leaves — bit-exact
    /// latency/power/energy on exact hits, interior lerps, and both
    /// boundary util-holds, and identical error strings on the miss paths
    /// (the LeafValue twin of `token_curve.rs::`
    /// `token_curve_is_bit_exact_with_the_generic_grid`).
    #[test]
    fn leaf_token_curve_is_bit_exact_with_the_generic_engine() {
        use crate::perf_database::perf_interp::{self, Node, OpInterpConfig};

        let points = BTreeMap::from([
            (10, LeafValue::with_power(1.25, 100.0)),
            (20, LeafValue::with_power(2.75, 150.0)),
            (40, LeafValue::with_power(5.5, 275.0)),
        ]);
        let curve = LeafTokenCurve::from_map(points.clone());
        let mut node = Node::branch();
        for (&token, &leaf) in &points {
            node.insert_value(&[token], leaf);
        }
        let sol = |tokens: f64| tokens * tokens + 1.0;
        let generic_sol = |coords: &[f64]| sol(coords[0]);
        let config = OpInterpConfig::grid(&["num_tokens"], &generic_sol);

        for tokens in [5.0, 10.0, 15.5, 20.0, 31.0, 40.0, 80.0] {
            let expected = perf_interp::query_value(&config, &node, &[tokens]).unwrap();
            let actual = curve.query(tokens, &sol).unwrap();
            for (name, actual, expected) in [
                ("latency", actual.latency, expected.latency),
                ("power", actual.power, expected.power),
                ("energy", actual.energy, expected.energy),
            ] {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "tokens={tokens}, field={name}"
                );
            }
        }

        // Miss paths: an empty curve and a non-positive SOL hold must
        // produce the exact generic-engine error strings.
        let empty = LeafTokenCurve::default();
        let empty_node = Node::branch();
        assert_eq!(
            empty.query(10.0, &sol).unwrap_err().to_string(),
            perf_interp::query_value(&config, &empty_node, &[10.0])
                .unwrap_err()
                .to_string()
        );
        let zero_sol = |_: f64| 0.0;
        let generic_zero_sol = |_: &[f64]| 0.0;
        let zero_config = OpInterpConfig::grid(&["num_tokens"], &generic_zero_sol);
        assert_eq!(
            curve.query(80.0, &zero_sol).unwrap_err().to_string(),
            perf_interp::query_value(&zero_config, &node, &[80.0])
                .unwrap_err()
                .to_string()
        );
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
            &proxy_sol,
        );
        // Either succeeds (uniform fallback found a match) or errors
        // with a topology mismatch — but not a distribution-specific
        // error.
        match result {
            Ok(value) => assert!(value.latency > 0.0),
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

    fn b200_trtllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/trtllm/1.2.0rc5")
    }

    /// Cross-language parity with the Python v2 engine. Expected values from:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk import common
    /// db = PerfDatabase('b200_sxm','vllm','0.19.0',
    ///                   systems_root='src/aiconfigurator_core/systems', database_mode='SOL')
    /// for nt in [384, 4096, 7]:
    ///     r = db.query_moe(num_tokens=nt, hidden_size=5120, inter_size=8192, topk=1,
    ///                      num_experts=16, moe_tp_size=1, moe_ep_size=1,
    ///                      quant_mode=common.MoEQuantMode.bfloat16,
    ///                      workload_distribution='power_law_1.01',
    ///                      database_mode=common.DatabaseMode.SILICON)
    ///     print(nt, repr(float(r)))"
    /// ```
    ///
    /// (`database_mode='SOL'` at construction disables the shared layer so
    /// Python loads exactly the same primary parquet the Rust table reads.)
    /// The collected token curve is {128, 256, 512, 1024}: nt=384 is an
    /// interior RAW lerp; nt=4096 / nt=7 are beyond-range util-holds where
    /// the MoE roofline SOL carries the growth (weight-load-dominated regime
    /// — a raw linear extrapolation would give ~2.0 ms at nt=4096, the
    /// roofline hold gives ~0.97 ms).
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if
    // this fails. `MoeTable::new` resolves to the single primary source with no
    // kernel_source filter, so no shared rows should join this curve.
    #[test]
    fn moe_query_matches_python_v2_engine() {
        use crate::common::system_spec::SystemSpec;

        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/b200_sxm.yaml");
        let spec = SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse");

        // The MoE roofline exactly as the operator layer passes it
        // (`operators/moe.rs::sol_latency_ms`, gated => num_gemms = 3),
        // mirroring Python `MoE._query_moe_table.get_sol` incl. its integer
        // floor divisions.
        let quant = MoeQuantMode::Bfloat16;
        let (h, inter, topk, ne, ep, tp) = (5120u64, 8192u64, 1u64, 16u64, 1u64, 1u64);
        let sol = |t: f64| -> f64 {
            let num_gemms = 3u64;
            let total_tokens = t.round() as u64 * topk;
            let ops = total_tokens * h * inter * num_gemms * 2 / ep / tp;
            let mem_bytes_int = total_tokens / ep * h * 2
                + total_tokens / ep * inter * num_gemms / tp
                + h * inter * num_gemms / tp * std::cmp::min(ne / ep, total_tokens / ep);
            let mem_bytes = (mem_bytes_int as f64) * quant.mapping().memory;
            let tc_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(1.0);
            let sol_math = (ops as f64) / (tc_flops * quant.mapping().compute) * 1000.0;
            let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
            sol_math.max(sol_mem)
        };

        let table = MoeTable::new(b200_vllm_data_root());
        let cases: &[(u32, f64)] = &[
            (384, 0.707481598854065),
            (4096, 0.9657080651305716),
            (7, 0.2885776182085593),
        ];
        for &(nt, expected) in cases {
            let got = table
                .query(nt, 5120, 8192, 1, 16, 1, 1, quant, "power_law_1.01", &sol)
                .expect("query must succeed")
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "nt={nt}: rust {got} vs python {expected}"
            );
        }
    }

    #[test]
    fn moe_low_latency_grid_split_on_b200_trtllm() {
        // b200 trtllm 1.2.0rc5 perf-DB carries `moe_torch_flow_min_latency`
        // rows; they must land in the low_latency grid, not the default
        // one. vLLM/SGLang DBs lack the column entirely → low_latency
        // empty → `low_latency_available()` returns false.
        let table = MoeTable::new(b200_trtllm_data_root());
        let available = table
            .low_latency_available()
            .expect("moe_perf.parquet must load");
        assert!(
            available,
            "expected b200/trtllm/1.2.0rc5 to carry moe_torch_flow_min_latency rows"
        );

        let vllm = MoeTable::new(b200_vllm_data_root());
        let vllm_available = vllm
            .low_latency_available()
            .expect("vllm moe_perf.parquet must load");
        assert!(
            !vllm_available,
            "vLLM perf DB lacks kernel_source column → low_latency should be empty"
        );
    }

    /// ENERGY oracle on a synthetic power-carrying fixture. Python twin
    /// (pandas fixture, `energy_test_fixtures` spec):
    ///
    /// ```text
    /// db.query_moe(num_tokens=1536, hidden_size=4096, inter_size=2048,
    ///              topk=2, num_experts=8, moe_tp_size=1, moe_ep_size=1,
    ///              quant_mode=MoEQuantMode.bfloat16,
    ///              workload_distribution="uniform", database_mode=SILICON)
    /// # -> latency=2.0, energy=300.0
    /// ```
    #[test]
    fn moe_energy_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{write_parquet, Col};
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &tmp.path().join("moe_perf.parquet"),
            &[
                Col::Str("moe_dtype", vec!["bfloat16", "bfloat16"]),
                Col::I64("num_tokens", vec![1024, 2048]),
                Col::I64("hidden_size", vec![4096, 4096]),
                Col::I64("inter_size", vec![2048, 2048]),
                Col::I64("topk", vec![2, 2]),
                Col::I64("num_experts", vec![8, 8]),
                Col::I64("moe_tp_size", vec![1, 1]),
                Col::I64("moe_ep_size", vec![1, 1]),
                Col::Str("distribution", vec!["uniform", "uniform"]),
                Col::Str("kernel_source", vec!["moe_torch_flow", "moe_torch_flow"]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        let table = MoeTable::new(tmp.path().to_path_buf());
        let v = table
            .query(
                1536,
                4096,
                2048,
                2,
                8,
                1,
                1,
                MoeQuantMode::Bfloat16,
                "uniform",
                &proxy_sol,
            )
            .unwrap();
        assert!((v.latency - 2.0).abs() < 1e-9, "latency {}", v.latency);
        assert!(
            (v.energy - 300.0).abs() < 1e-9 * 300.0,
            "energy {}",
            v.energy
        );
    }
}
