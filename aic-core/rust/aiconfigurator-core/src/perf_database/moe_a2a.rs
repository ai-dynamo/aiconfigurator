// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified large-EP MoE all-to-all comm table (`moe_a2a_perf.parquet`).
//!
//! Rust port of Python `sdk/operations/moe_comm.py`: `load_moe_a2a_data`
//! (new schema) + `_load_legacy_a2a` (the three per-backend legacy adapters)
//! + the silicon body of `MoEAllToAll._query_a2a_table`.
//!
//! One coordinate serves every inference backend:
//! `[comm_backend][phase][comm_dtype][ep_size][node_num][hidden_size][topk]`
//! `[num_experts][sms]` -> `{num_tokens -> latency_ms}`.
//!
//! Four source files feed it:
//!
//! - `moe_a2a_perf.parquet` — the unified schema. Its `latency` column is in
//!   MICROseconds (collector convention) and is divided by 1000 here; `sms`
//!   is nullable/optional and normalizes to 0 (`_normalize_sms`).
//! - `wideep_deepep_normal_perf.parquet` -> `deepep_ht`. Each legacy row
//!   becomes a dispatch row (`dispatch_transmit_us + dispatch_notify_us`) and
//!   a combine row (`combine_transmit_us + combine_notify_us`), us -> ms,
//!   `comm_dtype = "default"`, `ep_size = node_num * 8` (the legacy tables
//!   were collected on 8-GPU HGX fleets with no dtype axis), `sms =
//!   dispatch_sms`.
//! - `wideep_deepep_ll_perf.parquet` -> `deepep_ll`. Same shape from the two
//!   per-phase average columns; LL rows carry no SM budget -> `sms = 0`.
//! - `trtllm_alltoall_perf.parquet` -> `nvlink_two_sided` /
//!   `nvlink_one_sided`, `op_name` -> phase (`alltoall_combine_low_precision`
//!   -> `combine` keyed under `comm_dtype = "fp4"`; the other three carry the
//!   row's `moe_dtype`). UNITS: the legacy `latency` column is ALREADY in
//!   milliseconds — stored raw, no conversion (see
//!   `_adapt_legacy_trtllm_alltoall`'s docstring).
//!
//! Merge (Python `load_moe_a2a_data`): legacy rows load first, keep-first on
//! an intra-source collision; the FIRST new-schema occurrence of a key then
//! overwrites whatever a legacy adapter stored there, and repeats of that key
//! keep the first new-schema value.
//!
//! Query resolves the comm-dtype chain (exact -> `fp8_block` reusing `fp8`
//! -> the sole collected dtype -> typed miss, `_resolve_comm_dtype_slice`)
//! and then the sms axis: an EXACT `sms` key gets a plain 1-D token curve,
//! anything else a 2-D `(sms, num_tokens)` Grid — the split
//! `_query_a2a_table` makes. Both ride the shared `perf_interp` engine with
//! the linear token proxy SOL (`sol_fn=lambda ..., t: float(t)`): per-slice
//! payload bytes scale ~linearly with tokens, so the proxy is
//! ratio-equivalent to any bandwidth roofline.
//!
//! Backend/phase VALIDATION (`_validate_a2a_request`'s `ValueError`) is the
//! operator layer's job — this file is an algorithm-free accessor, so an
//! unknown backend or phase surfaces here as an ordinary typed data miss.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::OnceLock;

use super::axis_curve::AxisCurve;
use super::perf_interp::{self, Node, OpInterpConfig};
use super::{kernel_source_ok, SourceResolver};
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use crate::perf_database::parquet_loader::{PerfReader, PerfRow};

/// `(comm_backend, phase, comm_dtype, ep_size, node_num, hidden_size, topk,
/// num_experts, sms)` — every level of the Python store above the token axis,
/// in the same order, so a `BTreeMap` range scan over one `sms` span yields
/// the `by_sms` slice the query walks.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MoeA2aKey {
    pub comm_backend: String,
    pub phase: String,
    pub comm_dtype: String,
    pub ep_size: u32,
    pub node_num: u32,
    pub hidden_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub sms: u32,
}

/// `num_tokens -> latency_ms` curves keyed by [`MoeA2aKey`], plus the
/// collected comm-dtypes per `(comm_backend, phase)` the dtype chain needs.
struct MoeA2aGrids {
    by_keys: BTreeMap<MoeA2aKey, BTreeMap<u32, f64>>,
    /// `(comm_backend, phase) -> {comm_dtype}`. Mirrors `len(phase_slice)` /
    /// `next(iter(phase_slice))` in `_resolve_comm_dtype_slice`; the sole-dtype
    /// fallback only fires at size 1, where iteration order cannot matter.
    dtypes_by_phase: BTreeMap<(String, String), BTreeSet<String>>,
}

type A2aGrid = BTreeMap<MoeA2aKey, BTreeMap<u32, f64>>;

pub struct MoeA2aTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources per distinct perf-file basename
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`MoeA2aTable::new`).
    moe_a2a_sources: Vec<PerfSource>,
    legacy_normal_sources: Vec<PerfSource>,
    legacy_ll_sources: Vec<PerfSource>,
    legacy_trtllm_alltoall_sources: Vec<PerfSource>,
    grids: OnceLock<Result<MoeA2aGrids, AicError>>,
}

impl MoeA2aTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &SourceResolver::fixed(PerfDbSources::default()))
            .expect("fixed-map resolution is infallible")
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved
    /// from `perf_db_sources` (Python-supplied). Each perf file falls back to
    /// its primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, resolver: &SourceResolver) -> Result<Self, AicError> {
        let moe_a2a_sources = resolver.sources_for("moe_a2a_perf.parquet", &data_root)?;
        let legacy_normal_sources =
            resolver.sources_for("wideep_deepep_normal_perf.parquet", &data_root)?;
        let legacy_ll_sources =
            resolver.sources_for("wideep_deepep_ll_perf.parquet", &data_root)?;
        let legacy_trtllm_alltoall_sources =
            resolver.sources_for("trtllm_alltoall_perf.parquet", &data_root)?;
        Ok(Self {
            data_root,
            moe_a2a_sources,
            legacy_normal_sources,
            legacy_ll_sources,
            legacy_trtllm_alltoall_sources,
            grids: OnceLock::new(),
        })
    }

    /// Whether an exact A2A shape coordinate has at least one collected SM
    /// curve. Token-axis coverage is deliberately not checked here: an exact
    /// scale with an incomplete curve must fail at its own interpolation
    /// boundary rather than silently falling back to another node scale.
    #[allow(clippy::too_many_arguments)]
    pub fn has_shape(
        &self,
        comm_backend: &str,
        phase: &str,
        comm_dtype: &str,
        ep_size: u32,
        node_num: u32,
        hidden_size: u32,
        topk: u32,
        num_experts: u32,
    ) -> Result<bool, AicError> {
        let grids = self.load()?;
        let phase_slice = (comm_backend.to_string(), phase.to_string());
        let Some(dtypes) = grids.dtypes_by_phase.get(&phase_slice) else {
            return Ok(false);
        };
        let used_dtype = if dtypes.contains(comm_dtype) {
            comm_dtype
        } else if comm_dtype == "fp8_block" && dtypes.contains("fp8") {
            "fp8"
        } else if dtypes.len() == 1 && dtypes.contains("default") {
            "default"
        } else {
            return Ok(false);
        };
        let key_at = |sms: u32| MoeA2aKey {
            comm_backend: comm_backend.to_string(),
            phase: phase.to_string(),
            comm_dtype: used_dtype.to_string(),
            ep_size,
            node_num,
            hidden_size,
            topk,
            num_experts,
            sms,
        };
        Ok(grids
            .by_keys
            .range(key_at(0)..=key_at(u32::MAX))
            .next()
            .is_some())
    }

    /// Unified MoE all-to-all latency (ms) for one comm phase.
    ///
    /// Mirrors the SILICON body of Python `_query_a2a_table`: slice walk
    /// (backend -> phase -> dtype-with-fallback -> ep -> node -> hidden ->
    /// topk -> experts), then an exact-`sms` 1-D token curve or a 2-D
    /// `(sms, num_tokens)` Grid. Argument order matches
    /// `PerfDatabase.query_moe_a2a` (`num_tokens` before `sms`).
    #[allow(clippy::too_many_arguments)]
    pub fn query(
        &self,
        comm_backend: &str,
        phase: &str,
        comm_dtype: &str,
        ep_size: u32,
        node_num: u32,
        hidden_size: u32,
        topk: u32,
        num_experts: u32,
        num_tokens: u32,
        sms: u32,
    ) -> Result<f64, AicError> {
        let grids = self.load()?;
        // `_resolve_comm_dtype_slice`: exact key -> the `fp8_block` -> `fp8`
        // behavioral alias -> the sole collected dtype (the legacy DeepEP
        // tables have no dtype axis and live under "default", so a caller
        // asking for a payload dtype must still reach them) -> typed miss.
        let resolve_dtype = |phase_name: &str| -> Result<String, AicError> {
            let phase_slice = (comm_backend.to_string(), phase_name.to_string());
            let dtypes = grids.dtypes_by_phase.get(&phase_slice).ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "moe_a2a data missing for comm_backend={comm_backend:?} \
                     phase={phase_name:?} at {}",
                    self.data_root.display()
                ))
            })?;
            if dtypes.contains(comm_dtype) {
                Ok(comm_dtype.to_string())
            } else if comm_dtype == "fp8_block" && dtypes.contains("fp8") {
                Ok("fp8".to_string())
            } else if dtypes.len() == 1 && dtypes.contains("default") {
                Ok("default".to_string())
            } else {
                Err(AicError::PerfDatabase(format!(
                    "moe_a2a comm_dtype {comm_dtype:?} is not available for \
                     {comm_backend}/{phase_name} at {}; collected dtypes: {dtypes:?}",
                    self.data_root.display()
                )))
            }
        };
        let used_dtype = resolve_dtype(phase)?;
        let collect_by_sms = |phase_name: &str, dtype: &str| {
            let key_at = |sms: u32| MoeA2aKey {
                comm_backend: comm_backend.to_string(),
                phase: phase_name.to_string(),
                comm_dtype: dtype.to_string(),
                ep_size,
                node_num,
                hidden_size,
                topk,
                num_experts,
                sms,
            };
            grids
                .by_keys
                .range(key_at(0)..=key_at(u32::MAX))
                .map(|(key, curve)| (key.sms, curve))
                .collect::<BTreeMap<_, _>>()
        };
        // `sms` is the last key field, so every collected SM budget of one
        // shape coordinate is one contiguous range — Python's `by_sms` slice.
        let by_sms = collect_by_sms(phase, &used_dtype);
        if by_sms.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "moe_a2a data missing for {comm_backend}/{phase}, dtype={used_dtype}, \
                 ep={ep_size}, nodes={node_num}, hidden={hidden_size}, topk={topk}, \
                 experts={num_experts} at {}",
                self.data_root.display(),
            )));
        }
        // An EXACT sms key collapses that level to a 1-D token curve;
        // anything else resolves the 2-D (sms, num_tokens) Grid. Preserve the
        // Python DeepEP-HT exception: only node=1/sms=20 uses the 1-D path;
        // all other HT requests stay on the 2-D grid even for an exact sms.
        let use_token_curve = by_sms.contains_key(&sms)
            && (comm_backend != "deepep_ht" || (node_num == 1 && sms == 20));
        if use_token_curve {
            let curve = by_sms.get(&sms).expect("contains_key checked");
            return token_axis_curve(curve).query(num_tokens as f64, &|t| t);
        }
        let latency = query_sms_grid(&by_sms, sms, num_tokens)?;

        // The tapered Grid frontier hold is nonlinear in latency. Python
        // preserves the legacy DeepEP round-trip contract by interpolating
        // dispatch+combine together and then apportioning that result by the
        // two independently resolved phase shares.
        if comm_backend == "deepep_ht" && matches!(phase, "dispatch" | "combine") {
            let other_phase = if phase == "dispatch" {
                "combine"
            } else {
                "dispatch"
            };
            let other_dtype = resolve_dtype(other_phase)?;
            let other_by_sms = collect_by_sms(other_phase, &other_dtype);
            let mut combined = BTreeMap::<u32, BTreeMap<u32, f64>>::new();
            for (&sm, curve) in &by_sms {
                let Some(other_curve) = other_by_sms.get(&sm) else {
                    continue;
                };
                for (&tokens, &value) in curve.iter() {
                    if let Some(&other_value) = other_curve.get(&tokens) {
                        combined
                            .entry(sm)
                            .or_default()
                            .insert(tokens, value + other_value);
                    }
                }
            }
            if !combined.is_empty() {
                let other_latency = query_sms_grid(&other_by_sms, sms, num_tokens)?;
                let combined_refs = combined
                    .iter()
                    .map(|(&sm, curve)| (sm, curve))
                    .collect::<BTreeMap<_, _>>();
                let combined_latency = query_sms_grid(&combined_refs, sms, num_tokens)?;
                let phase_sum = latency + other_latency;
                if phase_sum > 0.0 {
                    return Ok(latency * combined_latency / phase_sum);
                }
            }
        }
        Ok(latency)
    }

    fn load(&self) -> Result<&MoeA2aGrids, AicError> {
        let cell = self.grids.get_or_init(|| {
            load_moe_a2a_grids(
                &self.moe_a2a_sources,
                &self.legacy_normal_sources,
                &self.legacy_ll_sources,
                &self.legacy_trtllm_alltoall_sources,
            )
        });
        cell.as_ref().map_err(clone_err)
    }
}

fn query_sms_grid(
    by_sms: &BTreeMap<u32, &BTreeMap<u32, f64>>,
    sms: u32,
    num_tokens: u32,
) -> Result<f64, AicError> {
    let mut node = Node::branch();
    for (&sm, curve) in by_sms {
        for (&tokens, &latency) in curve.iter() {
            node.insert(&[sm, tokens], latency);
        }
    }
    let sol = |c: &[f64]| c[1];
    let cfg = OpInterpConfig::grid(&["sms", "num_tokens"], &sol);
    perf_interp::query(&cfg, &node, &[f64::from(sms), f64::from(num_tokens)])
}

/// `comm_dtype` of every legacy DeepEP row: those tables were collected with
/// no dtype axis (`_adapt_legacy_deepep`). Shared with the table view — the
/// legacy-adaptation vocabulary has ONE home per family.
pub(crate) const LEGACY_DEEPEP_DTYPE: &str = "default";

/// `ep_size` of every legacy DeepEP row: `node_num * 8` — the legacy tables
/// were collected on 8-GPU HGX fleets and carry no ep axis
/// (`_adapt_legacy_deepep`). Saturating only to keep a corrupt row from
/// panicking a debug build; real node counts are single digits. Shared with
/// the table view.
pub(crate) fn legacy_deepep_ep_size(node_num: u32) -> u32 {
    node_num.saturating_mul(8)
}

/// `kernel_source` Python assumes when the legacy trtllm-alltoall file has no
/// such COLUMN (`row.get("kernel_source", "NVLinkTwoSided")`). A column that
/// exists with a NULL cell is a different case — see
/// [`adapt_legacy_trtllm_alltoall`]. Shared with the table view (and the raw
/// `load_trtllm_alltoall_data` twin, which used the same default).
pub(crate) const LEGACY_TRTLLM_DEFAULT_KERNEL_SOURCE: &str = "NVLinkTwoSided";

/// Load the unified table: legacy adapters first (keep-first), then the new
/// schema (first occurrence of a key overwrites, repeats keep first) —
/// Python `load_moe_a2a_data`.
fn load_moe_a2a_grids(
    a2a_sources: &[PerfSource],
    normal_sources: &[PerfSource],
    ll_sources: &[PerfSource],
    trtllm_sources: &[PerfSource],
) -> Result<MoeA2aGrids, AicError> {
    let mut by_keys: A2aGrid = BTreeMap::new();
    let mut any_source = adapt_legacy_deepep_normal(normal_sources, &mut by_keys)?;
    any_source |= adapt_legacy_deepep_ll(ll_sources, &mut by_keys)?;
    any_source |= adapt_legacy_trtllm_alltoall(trtllm_sources, &mut by_keys)?;
    any_source |= load_new_schema(a2a_sources, &mut by_keys)?;
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MoE all-to-all rows loaded from {} source(s) (moe_a2a + 3 legacy tables; \
             first: {})",
            a2a_sources.len() + normal_sources.len() + ll_sources.len() + trtllm_sources.len(),
            a2a_sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_else(|| "<no moe_a2a sources>".to_string())
        )));
    }
    let mut dtypes_by_phase: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for key in by_keys.keys() {
        dtypes_by_phase
            .entry((key.comm_backend.clone(), key.phase.clone()))
            .or_default()
            .insert(key.comm_dtype.clone());
    }
    Ok(MoeA2aGrids {
        by_keys,
        dtypes_by_phase,
    })
}

/// Bridge a sorted token->latency map onto the shared [`AxisCurve`] engine
/// (#1491/#1501 moved the free token-curve helpers onto it). BTreeMap
/// iteration is ascending, so the strict-order constructor holds.
fn token_axis_curve(points: &std::collections::BTreeMap<u32, f64>) -> AxisCurve {
    AxisCurve::from_sorted_iter(
        "num_tokens",
        points
            .iter()
            .map(|(&coordinate, &value)| (coordinate, value)),
    )
}

/// Python `_store_a2a_leaf(..., overwrite=False)`: the first stored leaf at a
/// coordinate wins. Sources are visited in priority order, so an earlier
/// source also outranks later siblings.
fn store_first_wins(by_keys: &mut A2aGrid, key: MoeA2aKey, num_tokens: u32, latency_ms: f64) {
    by_keys
        .entry(key)
        .or_default()
        .entry(num_tokens)
        .or_insert(latency_ms);
}

/// Key of one legacy DeepEP row's phase leaf. `ep_size = node_num * 8` — the
/// legacy tables were collected on 8-GPU HGX fleets and carry no ep axis
/// (`_adapt_legacy_deepep`).
fn legacy_deepep_key(
    comm_backend: &str,
    phase: &str,
    node_num: u32,
    hidden_size: u32,
    topk: u32,
    num_experts: u32,
    sms: u32,
) -> MoeA2aKey {
    MoeA2aKey {
        comm_backend: comm_backend.to_string(),
        phase: phase.to_string(),
        comm_dtype: LEGACY_DEEPEP_DTYPE.to_string(),
        ep_size: legacy_deepep_ep_size(node_num),
        node_num,
        hidden_size,
        topk,
        num_experts,
        sms,
    }
}

/// `_adapt_legacy_deepep_normal`: one legacy row becomes a dispatch leaf
/// (`dispatch_transmit_us + dispatch_notify_us`) and a combine leaf
/// (`combine_transmit_us + combine_notify_us`), us -> ms, both keyed by the
/// row's `dispatch_sms` budget. Returns whether any source file exists —
/// Python's exists-but-empty semantic (`_read_filtered_rows` yields `None`
/// only when EVERY path is missing).
///
/// Column handling follows the retired `wideep.rs::load_deepep_normal_parquet`
/// convention (that loader is gone; the surviving reference for this file is
/// Python `operations/moe_comm.py::_adapt_legacy_deepep_normal`): the four
/// component columns are read optionally and default to 0.0. Python indexes
/// them directly
/// (`float(row[column])`), so an absent column is a hard `KeyError` there;
/// every shipped file carries all four, so the two agree on real data.
fn adapt_legacy_deepep_normal(
    sources: &[PerfSource],
    by_keys: &mut A2aGrid,
) -> Result<bool, AicError> {
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let node_num_col = reader.col("node_num")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let num_token_col = reader.col("num_token")?;
        let num_topk_col = reader.col("num_topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let dispatch_sms_col = reader.col("dispatch_sms")?;
        let dispatch_transmit_us_col = reader.col_optional("dispatch_transmit_us");
        let dispatch_notify_us_col = reader.col_optional("dispatch_notify_us");
        let combine_transmit_us_col = reader.col_optional("combine_transmit_us");
        let combine_notify_us_col = reader.col_optional("combine_notify_us");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let node_num = row.u32(node_num_col)?;
            let hidden_size = row.u32(hidden_size_col)?;
            let topk = row.u32(num_topk_col)?;
            let num_experts = row.u32(num_experts_col)?;
            let sms = row.u32(dispatch_sms_col)?;
            let num_tokens = row.u32(num_token_col)?;
            let dispatch_us = row.f64_optional(dispatch_transmit_us_col)?.unwrap_or(0.0)
                + row.f64_optional(dispatch_notify_us_col)?.unwrap_or(0.0);
            let combine_us = row.f64_optional(combine_transmit_us_col)?.unwrap_or(0.0)
                + row.f64_optional(combine_notify_us_col)?.unwrap_or(0.0);
            for (phase, latency_us) in [("dispatch", dispatch_us), ("combine", combine_us)] {
                store_first_wins(
                    by_keys,
                    legacy_deepep_key(
                        "deepep_ht",
                        phase,
                        node_num,
                        hidden_size,
                        topk,
                        num_experts,
                        sms,
                    ),
                    num_tokens,
                    latency_us / 1000.0,
                );
            }
        }
    }
    Ok(any_source)
}

/// `_adapt_legacy_deepep_ll`: the LL table never had the four-way
/// transmit/notify split — only per-phase averages — and carries no SM
/// budget, so every leaf keys at `sms = 0`. Same optional-column handling as
/// the retired `wideep.rs::load_deepep_ll_parquet` — see
/// [`adapt_legacy_deepep_normal`] and Python
/// `operations/moe_comm.py::_adapt_legacy_deepep_ll`.
fn adapt_legacy_deepep_ll(sources: &[PerfSource], by_keys: &mut A2aGrid) -> Result<bool, AicError> {
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let node_num_col = reader.col("node_num")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let num_token_col = reader.col("num_token")?;
        let num_topk_col = reader.col("num_topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let dispatch_avg_t_us_col = reader.col_optional("dispatch_avg_t_us");
        let combine_avg_t_us_col = reader.col_optional("combine_avg_t_us");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let node_num = row.u32(node_num_col)?;
            let hidden_size = row.u32(hidden_size_col)?;
            let topk = row.u32(num_topk_col)?;
            let num_experts = row.u32(num_experts_col)?;
            let num_tokens = row.u32(num_token_col)?;
            let dispatch_us = row.f64_optional(dispatch_avg_t_us_col)?.unwrap_or(0.0);
            let combine_us = row.f64_optional(combine_avg_t_us_col)?.unwrap_or(0.0);
            for (phase, latency_us) in [("dispatch", dispatch_us), ("combine", combine_us)] {
                store_first_wins(
                    by_keys,
                    legacy_deepep_key(
                        "deepep_ll",
                        phase,
                        node_num,
                        hidden_size,
                        topk,
                        num_experts,
                        0,
                    ),
                    num_tokens,
                    latency_us / 1000.0,
                );
            }
        }
    }
    Ok(any_source)
}

/// `_LEGACY_TRTLLM_KERNEL_TO_BACKEND`; an unmapped kernel skips the row.
/// Shared with the table view.
pub(crate) fn legacy_trtllm_backend(kernel_source: &str) -> Option<&'static str> {
    match kernel_source {
        "NVLinkTwoSided" => Some("nvlink_two_sided"),
        "NVLinkOneSided" => Some("nvlink_one_sided"),
        _ => None,
    }
}

/// `_LEGACY_TRTLLM_OP_TO_PHASE_DTYPE`: `op_name -> (phase, comm_dtype)`,
/// where `None` means the row's own `moe_dtype` passes through. The
/// low-precision combine kernel gets its own `"fp4"` dtype key (an nvfp4
/// run's STANDARD combine still keys as `"nvfp4"`). An unmapped op_name skips
/// the row. Shared with the table view.
pub(crate) fn legacy_trtllm_phase_dtype(
    op_name: &str,
) -> Option<(&'static str, Option<&'static str>)> {
    match op_name {
        "alltoall_prepare" => Some(("prepare", None)),
        "alltoall_dispatch" => Some(("dispatch", None)),
        "alltoall_combine" => Some(("combine", None)),
        "alltoall_combine_low_precision" => Some(("combine", Some("fp4"))),
        _ => None,
    }
}

/// `_adapt_legacy_trtllm_alltoall`. UNITS: the legacy `latency` column is
/// ALREADY in milliseconds (`load_trtllm_alltoall_data` stores it raw and
/// `query_trtllm_alltoall` returns it without the /1000 the DeepEP path
/// applies) — stored raw, no conversion. `node_num` comes from an explicit
/// `num_nodes` column when the file has one, else `max(1, ep // 4)` (the
/// GB200 NVL4 derivation). Legacy alltoall rows carry no SM budget -> sms=0.
fn adapt_legacy_trtllm_alltoall(
    sources: &[PerfSource],
    by_keys: &mut A2aGrid,
) -> Result<bool, AicError> {
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let op_name_col = reader.col("op_name")?;
        let moe_dtype_col = reader.col("moe_dtype")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        let num_nodes_col = reader.col_optional("num_nodes");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            // Python defaults only when the COLUMN is absent; a NULL cell
            // reads back as "" (`_read_perf_rows`) and maps to no backend, so
            // the row is dropped rather than silently attributed to the
            // two-sided kernel.
            let kernel_source = match ks_col {
                None => LEGACY_TRTLLM_DEFAULT_KERNEL_SOURCE,
                Some(_) => row.str_optional(ks_col)?.unwrap_or(""),
            };
            let Some(comm_backend) = legacy_trtllm_backend(kernel_source) else {
                continue;
            };
            let Some((phase, dtype_override)) = legacy_trtllm_phase_dtype(row.str(op_name_col)?)
            else {
                continue;
            };
            let comm_dtype = match dtype_override {
                Some(dtype) => dtype.to_string(),
                None => row.str_owned(moe_dtype_col)?,
            };
            let ep_size = row.u32(moe_ep_size_col)?;
            let node_num = match num_nodes_col {
                // `int(row["num_nodes"])` — a null cell is a hard error in
                // Python (ValueError on ""); surface it as a typed load miss.
                Some(_) => row.u32_optional(num_nodes_col)?.ok_or_else(|| {
                    AicError::PerfDatabase(format!(
                        "legacy trtllm alltoall row has a null num_nodes cell at {}",
                        path.display()
                    ))
                })?,
                None => crate::perf_database::trtllm_alltoall::legacy_num_nodes_fallback(ep_size),
            };
            let key = MoeA2aKey {
                comm_backend: comm_backend.to_string(),
                phase: phase.to_string(),
                comm_dtype,
                ep_size,
                node_num,
                hidden_size: row.u32(hidden_size_col)?,
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                sms: 0,
            };
            store_first_wins(
                by_keys,
                key,
                row.u32(num_tokens_col)?,
                row.f64(latency_col)?,
            );
        }
    }
    Ok(any_source)
}

/// `_normalize_sms`: an absent column, a NULL cell, or NaN all key at 0.
/// The column is INT64 in the collector schema; the f64 arm covers a
/// float-typed collection (Python's `int(float(raw))`).
pub(crate) fn normalize_sms(row: &PerfRow, col: Option<usize>) -> Result<u32, AicError> {
    if let Some(value) = row.u32_optional(col)? {
        return Ok(value);
    }
    match row.f64_optional(col)? {
        Some(value) if value.is_finite() => Ok(value.max(0.0) as u32),
        _ => Ok(0),
    }
}

/// New-schema `moe_a2a_perf.parquet` rows. The `latency` column is in
/// MICROseconds (collector convention) and becomes ms here. The FIRST
/// occurrence of a key overwrites whatever a legacy adapter stored;
/// repeats of that key keep the first new-schema value.
fn load_new_schema(sources: &[PerfSource], by_keys: &mut A2aGrid) -> Result<bool, AicError> {
    let mut any_source = false;
    let mut seen: BTreeSet<(MoeA2aKey, u32)> = BTreeSet::new();
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let comm_backend_col = reader.col("comm_backend")?;
        let phase_col = reader.col("phase")?;
        let comm_dtype_col = reader.col("comm_dtype")?;
        let ep_size_col = reader.col("ep_size")?;
        let node_num_col = reader.col("node_num")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let latency_col = reader.col("latency")?;
        let sms_col = reader.col_optional("sms");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = MoeA2aKey {
                comm_backend: row.str_owned(comm_backend_col)?,
                // Stored as collected; the phase is validated at query time.
                phase: row.str_owned(phase_col)?,
                comm_dtype: row.str_owned(comm_dtype_col)?,
                ep_size: row.u32(ep_size_col)?,
                node_num: row.u32(node_num_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                sms: normalize_sms(&row, sms_col)?,
            };
            let num_tokens = row.u32(num_tokens_col)?;
            let latency_ms = row.f64(latency_col)? / 1000.0;
            if seen.insert((key.clone(), num_tokens)) {
                by_keys
                    .entry(key)
                    .or_default()
                    .insert(num_tokens, latency_ms);
            }
        }
    }
    Ok(any_source)
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
    use parquet::schema::parser::parse_message_type;
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    fn write_column<T: parquet::data_type::DataType>(
        rg: &mut SerializedRowGroupWriter<'_, File>,
        values: &[T::T],
    ) {
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<T>().write_batch(values, None, None).unwrap();
        col.close().unwrap();
    }

    /// One synthetic new-schema row. `sms: None` writes a NULL cell (the LL
    /// convention) when the column is present.
    #[derive(Clone)]
    struct A2aRow {
        comm_backend: &'static str,
        phase: &'static str,
        comm_dtype: &'static str,
        ep_size: i64,
        node_num: i64,
        sms: Option<i64>,
        num_tokens: i64,
        latency_us: f64,
    }

    /// Shape is fixed at (hidden=7168, topk=8, experts=256) for every
    /// synthetic row — the axes under test are backend/phase/dtype/sms/tokens.
    fn a2a_row(
        comm_backend: &'static str,
        phase: &'static str,
        comm_dtype: &'static str,
        ep_size: i64,
        node_num: i64,
        sms: Option<i64>,
        num_tokens: i64,
        latency_us: f64,
    ) -> A2aRow {
        A2aRow {
            comm_backend,
            phase,
            comm_dtype,
            ep_size,
            node_num,
            sms,
            num_tokens,
            latency_us,
        }
    }

    /// Write a synthetic `moe_a2a_perf.parquet`. `with_sms_column = false`
    /// omits the `sms` column entirely (older collections) — the loader must
    /// then key every row at sms=0, same as a NULL cell.
    fn write_a2a_parquet(path: &Path, rows: &[A2aRow], with_sms_column: bool) {
        let sms_decl = if with_sms_column {
            "OPTIONAL INT64 sms;"
        } else {
            ""
        };
        let schema = Arc::new(
            parse_message_type(&format!(
                "message a2a {{
                    REQUIRED BYTE_ARRAY comm_backend (UTF8);
                    REQUIRED BYTE_ARRAY phase (UTF8);
                    REQUIRED BYTE_ARRAY comm_dtype (UTF8);
                    REQUIRED INT64 ep_size;
                    REQUIRED INT64 node_num;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 topk;
                    REQUIRED INT64 num_experts;
                    {sms_decl}
                    REQUIRED INT64 num_tokens;
                    REQUIRED DOUBLE latency;
                }}"
            ))
            .unwrap(),
        );
        let file = File::create(path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut rg = writer.next_row_group().unwrap();
        let n = rows.len();
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.comm_backend))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.phase))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.comm_dtype))
                .collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.ep_size).collect::<Vec<_>>());
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.node_num).collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        if with_sms_column {
            // Optional column: only non-null values go in `values`, with a
            // definition level of 1 (present) / 0 (null) per row.
            let values: Vec<i64> = rows.iter().filter_map(|r| r.sms).collect();
            let def_levels: Vec<i16> = rows
                .iter()
                .map(|r| if r.sms.is_some() { 1 } else { 0 })
                .collect();
            let mut col = rg.next_column().unwrap().unwrap();
            col.typed::<Int64Type>()
                .write_batch(&values, Some(&def_levels), None)
                .unwrap();
            col.close().unwrap();
        }
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.num_tokens).collect::<Vec<_>>(),
        );
        write_column::<DoubleType>(
            &mut rg,
            &rows.iter().map(|r| r.latency_us).collect::<Vec<_>>(),
        );
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Write a synthetic legacy DeepEP-normal parquet. Rows are
    /// `(node_num, dispatch_sms, num_token, dispatch_transmit_us,
    /// dispatch_notify_us, combine_transmit_us, combine_notify_us)`; shape
    /// fixed at (hidden=7168, topk=8, experts=256). Column set is the one
    /// Python `operations/moe_comm.py::_adapt_legacy_deepep_normal` consumes
    /// (it was previously mirrored by `perf_database/wideep.rs`'s writer,
    /// retired with that loader).
    fn write_deepep_normal_parquet(path: &Path, rows: &[(i64, i64, i64, f64, f64, f64, f64)]) {
        let schema = Arc::new(
            parse_message_type(
                "message normal {
                    REQUIRED INT64 node_num;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 num_token;
                    REQUIRED INT64 num_topk;
                    REQUIRED INT64 num_experts;
                    REQUIRED INT64 dispatch_sms;
                    REQUIRED DOUBLE dispatch_transmit_us;
                    REQUIRED DOUBLE dispatch_notify_us;
                    REQUIRED DOUBLE combine_transmit_us;
                    REQUIRED DOUBLE combine_notify_us;
                }",
            )
            .unwrap(),
        );
        let file = File::create(path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut rg = writer.next_row_group().unwrap();
        let n = rows.len();
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.0).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.4).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.5).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.6).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Write a synthetic legacy DeepEP-LL parquet. Rows are `(node_num,
    /// num_token, dispatch_avg_t_us, combine_avg_t_us)`; shape fixed at
    /// (hidden=7168, topk=8, experts=256).
    fn write_deepep_ll_parquet(path: &Path, rows: &[(i64, i64, f64, f64)]) {
        let schema = Arc::new(
            parse_message_type(
                "message ll {
                    REQUIRED INT64 node_num;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 num_token;
                    REQUIRED INT64 num_topk;
                    REQUIRED INT64 num_experts;
                    REQUIRED DOUBLE combine_avg_t_us;
                    REQUIRED DOUBLE dispatch_avg_t_us;
                }",
            )
            .unwrap(),
        );
        let file = File::create(path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut rg = writer.next_row_group().unwrap();
        let n = rows.len();
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.0).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Write a synthetic legacy trtllm-alltoall parquet. Rows are
    /// `(kernel_source, op_name, moe_dtype, moe_ep_size, num_tokens,
    /// latency_ms)`; shape fixed at (hidden=7168, topk=8, experts=256).
    /// `with_num_nodes` adds the explicit `num_nodes` column (absent on the
    /// shipped GB200 NVL4 files, where node_num derives from moe_ep_size).
    fn write_trtllm_alltoall_parquet(
        path: &Path,
        rows: &[(&'static str, &'static str, &'static str, i64, i64, f64)],
        num_nodes: Option<i64>,
    ) {
        let num_nodes_decl = if num_nodes.is_some() {
            "REQUIRED INT64 num_nodes;"
        } else {
            ""
        };
        let schema = Arc::new(
            parse_message_type(&format!(
                "message alltoall {{
                    REQUIRED BYTE_ARRAY op_name (UTF8);
                    REQUIRED BYTE_ARRAY kernel_source (UTF8);
                    REQUIRED BYTE_ARRAY moe_dtype (UTF8);
                    REQUIRED INT64 num_tokens;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 topk;
                    REQUIRED INT64 num_experts;
                    REQUIRED INT64 moe_ep_size;
                    {num_nodes_decl}
                    REQUIRED BYTE_ARRAY distribution (UTF8);
                    REQUIRED DOUBLE latency;
                }}"
            ))
            .unwrap(),
        );
        let file = File::create(path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut rg = writer.next_row_group().unwrap();
        let n = rows.len();
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.1))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.0))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.2))
                .collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.4).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        if let Some(nodes) = num_nodes {
            write_column::<Int64Type>(&mut rg, &vec![nodes; n]);
        }
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("balanced"); n]);
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.5).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Write a synthetic legacy trtllm-alltoall parquet whose `kernel_source`
    /// column is OPTIONAL, so a row can carry a NULL cell (the case Python's
    /// `row.get(...)` default does NOT cover). Rows are `(kernel_source,
    /// moe_ep_size, latency_ms)` at a fixed (op_name=`alltoall_dispatch`,
    /// moe_dtype=`fp8`, hidden=7168, topk=8, experts=256, num_tokens=64)
    /// coordinate.
    fn write_trtllm_alltoall_nullable_ks_parquet(
        path: &Path,
        rows: &[(Option<&'static str>, i64, f64)],
    ) {
        let schema = Arc::new(
            parse_message_type(
                "message alltoall {
                    REQUIRED BYTE_ARRAY op_name (UTF8);
                    OPTIONAL BYTE_ARRAY kernel_source (UTF8);
                    REQUIRED BYTE_ARRAY moe_dtype (UTF8);
                    REQUIRED INT64 num_tokens;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 topk;
                    REQUIRED INT64 num_experts;
                    REQUIRED INT64 moe_ep_size;
                    REQUIRED DOUBLE latency;
                }",
            )
            .unwrap(),
        );
        let file = File::create(path).unwrap();
        let mut writer =
            SerializedFileWriter::new(file, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut rg = writer.next_row_group().unwrap();
        let n = rows.len();
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("alltoall_dispatch"); n]);
        // Optional column: only the non-null values are written, with a
        // definition level of 1 (present) / 0 (null) per row.
        let values: Vec<ByteArray> = rows
            .iter()
            .filter_map(|r| r.0.map(ByteArray::from))
            .collect();
        let def_levels: Vec<i16> = rows.iter().map(|r| i16::from(r.0.is_some())).collect();
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<ByteArrayType>()
            .write_batch(&values, Some(&def_levels), None)
            .unwrap();
        col.close().unwrap();
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("fp8"); n]);
        write_column::<Int64Type>(&mut rg, &vec![64_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    fn approx(got: f64, want: f64) {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "got {got}, want {want}"
        );
    }

    // ------------------------------------------------------------------
    // New-schema loader
    // ------------------------------------------------------------------

    /// R6 units: the unified `latency` column is MICROseconds; leaves are ms.
    /// A NULL `sms` cell and an absent `sms` column both key at sms=0
    /// (`_normalize_sms`).
    #[test]
    fn new_schema_converts_us_to_ms_and_normalizes_sms() {
        let tmp = tempfile::tempdir().unwrap();
        write_a2a_parquet(
            &tmp.path().join("moe_a2a_perf.parquet"),
            &[
                a2a_row("deepep_ht", "dispatch", "fp8", 16, 2, Some(20), 64, 250.0),
                a2a_row("deepep_ht", "combine", "fp8", 16, 2, Some(20), 64, 250.0),
                a2a_row("deepep_ll", "dispatch", "fp8", 16, 2, None, 64, 125.0),
            ],
            true,
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        approx(
            table
                .query("deepep_ht", "dispatch", "fp8", 16, 2, 7168, 8, 256, 64, 20)
                .unwrap(),
            0.25,
        );
        // NULL sms -> key 0.
        approx(
            table
                .query("deepep_ll", "dispatch", "fp8", 16, 2, 7168, 8, 256, 64, 0)
                .unwrap(),
            0.125,
        );

        // Same rows with the `sms` column omitted entirely.
        let tmp2 = tempfile::tempdir().unwrap();
        write_a2a_parquet(
            &tmp2.path().join("moe_a2a_perf.parquet"),
            &[
                a2a_row("deepep_ht", "dispatch", "fp8", 16, 2, Some(20), 64, 250.0),
                a2a_row("deepep_ht", "combine", "fp8", 16, 2, Some(20), 64, 250.0),
            ],
            false,
        );
        let table2 = MoeA2aTable::new(tmp2.path().to_path_buf());
        approx(
            table2
                .query("deepep_ht", "dispatch", "fp8", 16, 2, 7168, 8, 256, 64, 0)
                .unwrap(),
            0.25,
        );
    }

    // ------------------------------------------------------------------
    // Legacy adapters
    // ------------------------------------------------------------------

    /// `_adapt_legacy_deepep_normal`: dispatch = transmit + notify, combine
    /// likewise, us -> ms; `comm_dtype="default"`, `ep_size = node_num * 8`,
    /// `sms = dispatch_sms` on BOTH phases.
    #[test]
    fn legacy_deepep_normal_sums_component_columns() {
        let tmp = tempfile::tempdir().unwrap();
        write_deepep_normal_parquet(
            &tmp.path().join("wideep_deepep_normal_perf.parquet"),
            &[(2, 20, 64, 100.0, 25.0, 300.0, 75.0)],
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        // ep_size = node_num * 8 = 16; sms = dispatch_sms = 20.
        approx(
            table
                .query(
                    "deepep_ht",
                    "dispatch",
                    "default",
                    16,
                    2,
                    7168,
                    8,
                    256,
                    64,
                    20,
                )
                .unwrap(),
            0.125,
        );
        approx(
            table
                .query(
                    "deepep_ht",
                    "combine",
                    "default",
                    16,
                    2,
                    7168,
                    8,
                    256,
                    64,
                    20,
                )
                .unwrap(),
            0.375,
        );
        // ep_size is DERIVED, not the node count: ep=2 must miss.
        assert!(table
            .query(
                "deepep_ht",
                "dispatch",
                "default",
                2,
                2,
                7168,
                8,
                256,
                64,
                20
            )
            .is_err());
    }

    /// `_adapt_legacy_deepep_ll`: the two average columns, `sms = 0` (LL rows
    /// carry no SM budget), `ep_size = node_num * 8`.
    #[test]
    fn legacy_deepep_ll_uses_average_columns_at_sms_zero() {
        let tmp = tempfile::tempdir().unwrap();
        write_deepep_ll_parquet(
            &tmp.path().join("wideep_deepep_ll_perf.parquet"),
            &[(4, 64, 90.0, 210.0)],
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        approx(
            table
                .query(
                    "deepep_ll",
                    "dispatch",
                    "default",
                    32,
                    4,
                    7168,
                    8,
                    256,
                    64,
                    0,
                )
                .unwrap(),
            0.09,
        );
        approx(
            table
                .query(
                    "deepep_ll",
                    "combine",
                    "default",
                    32,
                    4,
                    7168,
                    8,
                    256,
                    64,
                    0,
                )
                .unwrap(),
            0.21,
        );
    }

    /// `_adapt_legacy_trtllm_alltoall`: kernel_source -> comm_backend,
    /// op_name -> phase, `alltoall_combine_low_precision` -> combine keyed
    /// under `"fp4"`, node_num = max(1, ep // 4), latency ALREADY ms, sms=0.
    /// Unmapped kernel_source / op_name rows are dropped.
    #[test]
    fn legacy_trtllm_alltoall_maps_phases_dtypes_and_node_num() {
        let tmp = tempfile::tempdir().unwrap();
        write_trtllm_alltoall_parquet(
            &tmp.path().join("trtllm_alltoall_perf.parquet"),
            &[
                ("NVLinkTwoSided", "alltoall_prepare", "nvfp4", 16, 64, 0.5),
                ("NVLinkTwoSided", "alltoall_dispatch", "nvfp4", 16, 64, 1.5),
                ("NVLinkTwoSided", "alltoall_combine", "nvfp4", 16, 64, 2.5),
                (
                    "NVLinkTwoSided",
                    "alltoall_combine_low_precision",
                    "nvfp4",
                    16,
                    64,
                    3.5,
                ),
                ("NVLinkOneSided", "alltoall_dispatch", "fp8", 2, 64, 4.5),
                // Unmapped: dropped, not stored under some default. Both sit
                // on their OWN (bfloat16, ep=8 -> node=2) coordinate so a
                // leaked row is directly observable rather than masked by the
                // keep-first value of a coordinate that is already asserted.
                (
                    "MnnvlThreeSided",
                    "alltoall_dispatch",
                    "bfloat16",
                    8,
                    64,
                    9.0,
                ),
                (
                    "NVLinkTwoSided",
                    "alltoall_something",
                    "bfloat16",
                    8,
                    64,
                    9.0,
                ),
            ],
            None,
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        let q = |backend: &str, phase: &str, dtype: &str, ep: u32, node: u32| {
            table.query(backend, phase, dtype, ep, node, 7168, 8, 256, 64, 0)
        };
        // node_num = max(1, 16 // 4) = 4; latency is raw ms (no /1000).
        approx(
            q("nvlink_two_sided", "prepare", "nvfp4", 16, 4).unwrap(),
            0.5,
        );
        approx(
            q("nvlink_two_sided", "dispatch", "nvfp4", 16, 4).unwrap(),
            1.5,
        );
        approx(
            q("nvlink_two_sided", "combine", "nvfp4", 16, 4).unwrap(),
            2.5,
        );
        // The low-precision combine keys under "fp4", NOT the run's moe_dtype.
        approx(q("nvlink_two_sided", "combine", "fp4", 16, 4).unwrap(), 3.5);
        // ep=2 -> max(1, 0) = 1 node.
        approx(q("nvlink_one_sided", "dispatch", "fp8", 2, 1).unwrap(), 4.5);
        // Neither unmapped row landed anywhere: their (bfloat16, ep=8 ->
        // node=2) coordinate is absent from every phase of BOTH backends.
        for phase in ["prepare", "dispatch", "combine"] {
            assert!(
                q("nvlink_two_sided", phase, "bfloat16", 8, 2).is_err(),
                "an unmapped row leaked into nvlink_two_sided/{phase}"
            );
            assert!(
                q("nvlink_one_sided", phase, "bfloat16", 8, 2).is_err(),
                "an unmapped row leaked into nvlink_one_sided/{phase}"
            );
        }
        // ...and an unmapped op_name is not passed through as a phase either.
        assert!(q("nvlink_two_sided", "alltoall_something", "bfloat16", 8, 2).is_err());
    }

    /// A present-but-NULL `kernel_source` cell maps to no comm backend, so the
    /// row is DROPPED. Python's `row.get("kernel_source", "NVLinkTwoSided")`
    /// defaults only when the COLUMN is absent; `_read_perf_rows` turns a null
    /// cell into `""`, which matches no backend. This is the one place the
    /// unified adapter deliberately diverges from
    /// `trtllm_alltoall.rs::load_alltoall_parquet`, which treats a null cell
    /// as the two-sided default.
    #[test]
    fn legacy_trtllm_alltoall_null_kernel_source_row_is_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        write_trtllm_alltoall_nullable_ks_parquet(
            &tmp.path().join("trtllm_alltoall_perf.parquet"),
            &[(Some("NVLinkTwoSided"), 16, 1.5), (None, 32, 9.0)],
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        let q = |backend: &str, ep: u32, node: u32| {
            table.query(backend, "dispatch", "fp8", ep, node, 7168, 8, 256, 64, 0)
        };
        // The named row still loads (ep=16 -> node_num = 4).
        approx(q("nvlink_two_sided", 16, 4).unwrap(), 1.5);
        // The NULL-kernel row (ep=32 -> node_num = 8) reached NEITHER backend.
        assert!(
            q("nvlink_two_sided", 32, 8).is_err(),
            "a null kernel_source cell must not default to NVLinkTwoSided"
        );
        assert!(q("nvlink_one_sided", 32, 8).is_err());
    }

    /// An explicit `num_nodes` column wins over the `max(1, ep // 4)` default.
    #[test]
    fn legacy_trtllm_alltoall_num_nodes_column_wins() {
        let tmp = tempfile::tempdir().unwrap();
        write_trtllm_alltoall_parquet(
            &tmp.path().join("trtllm_alltoall_perf.parquet"),
            &[("NVLinkTwoSided", "alltoall_dispatch", "fp8", 16, 64, 1.25)],
            Some(2),
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        approx(
            table
                .query(
                    "nvlink_two_sided",
                    "dispatch",
                    "fp8",
                    16,
                    2,
                    7168,
                    8,
                    256,
                    64,
                    0,
                )
                .unwrap(),
            1.25,
        );
        // The derived default (16 // 4 = 4) must NOT be present.
        assert!(table
            .query(
                "nvlink_two_sided",
                "dispatch",
                "fp8",
                16,
                4,
                7168,
                8,
                256,
                64,
                0
            )
            .is_err());
    }

    // ------------------------------------------------------------------
    // comm_dtype chain (`_resolve_comm_dtype_slice`)
    // ------------------------------------------------------------------

    #[test]
    fn dtype_chain_exact_then_fp8_block_alias_then_sole_then_miss() {
        // Two collected dtypes under (deepep_ht, dispatch): exact + alias
        // resolve; an unrelated dtype is a typed miss (no unambiguous stand-in).
        let tmp = tempfile::tempdir().unwrap();
        write_a2a_parquet(
            &tmp.path().join("moe_a2a_perf.parquet"),
            &[
                a2a_row("deepep_ht", "dispatch", "fp8", 16, 2, Some(20), 64, 100.0),
                a2a_row("deepep_ht", "dispatch", "nvfp4", 16, 2, Some(20), 64, 200.0),
                a2a_row("deepep_ht", "combine", "fp8", 16, 2, Some(20), 64, 100.0),
                a2a_row("deepep_ht", "combine", "nvfp4", 16, 2, Some(20), 64, 200.0),
            ],
            true,
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        let q =
            |dtype: &str| table.query("deepep_ht", "dispatch", dtype, 16, 2, 7168, 8, 256, 64, 20);
        approx(q("fp8").unwrap(), 0.1);
        approx(q("nvfp4").unwrap(), 0.2);
        // fp8_block is a behavioral mode reusing the fp8 comm tables.
        approx(q("fp8_block").unwrap(), 0.1);
        // Two collected dtypes -> no sole-dtype fallback.
        assert!(q("bfloat16").is_err());

        // Sole collected dtype ("default", the legacy DeepEP convention):
        // any requested dtype reaches it.
        let tmp2 = tempfile::tempdir().unwrap();
        write_deepep_ll_parquet(
            &tmp2.path().join("wideep_deepep_ll_perf.parquet"),
            &[(2, 64, 100.0, 300.0)],
        );
        let table2 = MoeA2aTable::new(tmp2.path().to_path_buf());
        approx(
            table2
                .query("deepep_ll", "dispatch", "nvfp4", 16, 2, 7168, 8, 256, 64, 0)
                .unwrap(),
            0.1,
        );
        // A phase that was never collected stays a miss (the chain runs BELOW
        // the phase level).
        assert!(table2
            .query(
                "deepep_ll",
                "prepare",
                "default",
                16,
                2,
                7168,
                8,
                256,
                64,
                0
            )
            .is_err());
    }

    /// The exact key wins over the alias even when both exist: a real
    /// `fp8_block` collection must not be normalized away.
    #[test]
    fn dtype_chain_exact_fp8_block_beats_the_fp8_alias() {
        let tmp = tempfile::tempdir().unwrap();
        write_a2a_parquet(
            &tmp.path().join("moe_a2a_perf.parquet"),
            &[
                a2a_row("deepep_ht", "dispatch", "fp8", 16, 2, Some(20), 64, 100.0),
                a2a_row("deepep_ht", "combine", "fp8", 16, 2, Some(20), 64, 100.0),
                a2a_row(
                    "deepep_ht",
                    "dispatch",
                    "fp8_block",
                    16,
                    2,
                    Some(20),
                    64,
                    700.0,
                ),
                a2a_row(
                    "deepep_ht",
                    "combine",
                    "fp8_block",
                    16,
                    2,
                    Some(20),
                    64,
                    700.0,
                ),
            ],
            true,
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        approx(
            table
                .query(
                    "deepep_ht",
                    "dispatch",
                    "fp8_block",
                    16,
                    2,
                    7168,
                    8,
                    256,
                    64,
                    20,
                )
                .unwrap(),
            0.7,
        );
    }

    // ------------------------------------------------------------------
    // sms resolution: exact -> 1-D token curve, else 2-D (sms, tokens) Grid
    // ------------------------------------------------------------------

    /// An exact `sms` key resolves its OWN token curve (1-D); an off-grid
    /// `sms` resolves the 2-D `(sms, num_tokens)` Grid — interior lerp on the
    /// sms axis, nearest-snap outside it. Python oracle from
    /// `perf_interp.query(OpInterpConfig(axes=("sms","num_tokens"),
    /// resolver=Grid(), sol_fn=lambda _sm, t: float(t)), by_sms, sms, t)`.
    #[test]
    fn sms_exact_is_1d_and_off_grid_is_2d() {
        let tmp = tempfile::tempdir().unwrap();
        write_a2a_parquet(
            &tmp.path().join("moe_a2a_perf.parquet"),
            &[
                a2a_row("deepep_ht", "dispatch", "fp8", 16, 2, Some(16), 64, 100.0),
                a2a_row("deepep_ht", "dispatch", "fp8", 16, 2, Some(16), 128, 200.0),
                a2a_row("deepep_ht", "dispatch", "fp8", 16, 2, Some(32), 64, 500.0),
                a2a_row("deepep_ht", "dispatch", "fp8", 16, 2, Some(32), 128, 900.0),
                a2a_row("deepep_ht", "combine", "fp8", 16, 2, Some(16), 64, 100.0),
                a2a_row("deepep_ht", "combine", "fp8", 16, 2, Some(16), 128, 200.0),
                a2a_row("deepep_ht", "combine", "fp8", 16, 2, Some(32), 64, 500.0),
                a2a_row("deepep_ht", "combine", "fp8", 16, 2, Some(32), 128, 900.0),
            ],
            true,
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        let q = |sms: u32, tokens: u32| {
            table
                .query(
                    "deepep_ht",
                    "dispatch",
                    "fp8",
                    16,
                    2,
                    7168,
                    8,
                    256,
                    tokens,
                    sms,
                )
                .unwrap()
        };
        // Exact sms -> its own curve: exact point and 1-D token lerp.
        approx(q(16, 64), 0.1);
        approx(q(16, 96), 0.15);
        approx(q(32, 128), 0.9);
        // Off-grid sms=24 -> 2-D: token-exact on both bracket slices, then
        // lerp along sms ((0.1 + 0.5) / 2, (0.2 + 0.9) / 2).
        approx(q(24, 64), 0.3);
        approx(q(24, 128), 0.55);
        // Off-grid on BOTH axes: tokens lerp inside each sms slice first.
        approx(q(24, 96), 0.425);
        // Outside the collected sms/token rectangle, current Grid semantics
        // use tapered joint-log transfer across nearby leaves (not a nearest
        // outer-axis snap). These values are pinned by the regenerated
        // same-head Python oracle as well as this synthetic surface.
        approx(q(8, 96), 0.182_754_372_856_303_42);
        approx(q(40, 256), 0.856_666_877_173_728_9);
    }

    // ------------------------------------------------------------------
    // Merge semantics (`_store_a2a_leaf` overwrite / keep-first)
    // ------------------------------------------------------------------

    /// Legacy rows load first; the FIRST new-schema row at the same key
    /// overwrites them, and a repeat of that key keeps the first new-schema
    /// value. Keys the new schema does not cover keep their legacy value.
    #[test]
    fn new_schema_overwrites_legacy_and_repeats_keep_first() {
        let tmp = tempfile::tempdir().unwrap();
        // Legacy LL: (node=2 -> ep=16, sms=0) dispatch 100us, combine 300us.
        write_deepep_ll_parquet(
            &tmp.path().join("wideep_deepep_ll_perf.parquet"),
            &[(2, 64, 100.0, 300.0)],
        );
        // New schema overwrites the dispatch leaf twice; the FIRST wins.
        write_a2a_parquet(
            &tmp.path().join("moe_a2a_perf.parquet"),
            &[
                a2a_row("deepep_ll", "dispatch", "default", 16, 2, None, 64, 700.0),
                a2a_row("deepep_ll", "dispatch", "default", 16, 2, None, 64, 900.0),
            ],
            true,
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        approx(
            table
                .query(
                    "deepep_ll",
                    "dispatch",
                    "default",
                    16,
                    2,
                    7168,
                    8,
                    256,
                    64,
                    0,
                )
                .unwrap(),
            0.7,
        );
        // The combine leaf the new schema never covered keeps its legacy value.
        approx(
            table
                .query(
                    "deepep_ll",
                    "combine",
                    "default",
                    16,
                    2,
                    7168,
                    8,
                    256,
                    64,
                    0,
                )
                .unwrap(),
            0.3,
        );
    }

    /// Intra-legacy collisions keep the FIRST row (Python `overwrite=False`).
    #[test]
    fn legacy_duplicate_rows_keep_first() {
        let tmp = tempfile::tempdir().unwrap();
        write_deepep_normal_parquet(
            &tmp.path().join("wideep_deepep_normal_perf.parquet"),
            &[
                (2, 20, 64, 100.0, 0.0, 0.0, 0.0),
                (2, 20, 64, 999.0, 0.0, 0.0, 0.0),
            ],
        );
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        approx(
            table
                .query(
                    "deepep_ht",
                    "dispatch",
                    "default",
                    16,
                    2,
                    7168,
                    8,
                    256,
                    64,
                    20,
                )
                .unwrap(),
            0.1,
        );
    }

    // ------------------------------------------------------------------
    // Python oracle over the shipped h200_sxm/sglang + gb200/trtllm data
    // ------------------------------------------------------------------

    const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";

    /// Whether the shipped parquet files behind `data_root` are usable: at
    /// least one of the four basenames resolves to a real file and none of
    /// the resolved files is an unresolved git-lfs pointer. `git lfs pull` is
    /// a checkout-time step, so a pointer-only tree skips the data-dependent
    /// oracle instead of failing it (same pointer detection as
    /// `parquet_loader::PerfReader::open`).
    fn shipped_data_ready(data_root: &Path) -> bool {
        use std::io::Read;
        let mut any_file = false;
        for basename in [
            "moe_a2a_perf.parquet",
            "wideep_deepep_normal_perf.parquet",
            "wideep_deepep_ll_perf.parquet",
            "trtllm_alltoall_perf.parquet",
        ] {
            for source in crate::perf_database::resolve_op_sources(
                &PerfDbSources::default(),
                basename,
                data_root,
            ) {
                let path = source.path();
                if !path.exists() {
                    continue;
                }
                let mut head = [0u8; LFS_POINTER_PREFIX.len()];
                let Ok(mut file) = File::open(path) else {
                    return false;
                };
                let Ok(read) = file.read(&mut head) else {
                    return false;
                };
                if read >= LFS_POINTER_PREFIX.len() && head == LFS_POINTER_PREFIX {
                    return false;
                }
                any_file = true;
            }
        }
        any_file
    }

    /// Full-table parity against `PerfDatabase.query_moe_a2a`. The fixture is
    /// generated by `parity_tests/gen_moe_a2a_oracle.py` (regeneration
    /// command in the JSON's `_regenerate` field) from the shipped
    /// h200_sxm/sglang/0.5.6.post2 (legacy DeepEP HT + LL) and
    /// gb200/trtllm/1.3.0rc10 (legacy NVLink alltoall) data, stratified over
    /// exact points, token lerps, token overflow/underflow util-holds, the
    /// 2-D sms grid (interior lerp + off-grid snap) and the comm-dtype chain.
    ///
    /// NOTE(shared-layer merge): the oracle is generated with
    /// `shared_layer=False` and `MoeA2aTable::new` resolves single primary
    /// sources with no kernel_source filter — the four a2a files live under
    /// the `comm` family, which Python hard-excludes from every reuse
    /// channel, so both sides read exactly the same rows.
    #[test]
    fn moe_a2a_matches_python_oracle() {
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("testdata/moe_a2a_oracle.json"))
                .expect("oracle fixture must parse");
        let systems =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems");
        let samples = oracle["samples"].as_array().expect("samples array");
        let mut tables: BTreeMap<String, MoeA2aTable> = BTreeMap::new();
        let mut max_rel = 0.0_f64;
        let mut checked = 0_usize;
        for sample in samples {
            let rel_root = sample["data_root"].as_str().expect("data_root");
            let data_root = systems.join(rel_root);
            if !shipped_data_ready(&data_root) {
                eprintln!(
                    "SKIP moe_a2a_matches_python_oracle: shipped perf data unavailable at {} \
                     (run `git lfs pull`)",
                    data_root.display()
                );
                return;
            }
            let table = tables
                .entry(rel_root.to_string())
                .or_insert_with(|| MoeA2aTable::new(data_root.clone()));
            let u32_of = |field: &str| {
                u32::try_from(sample[field].as_u64().expect(field)).expect("fits in u32")
            };
            let got = table
                .query(
                    sample["comm_backend"].as_str().expect("comm_backend"),
                    sample["phase"].as_str().expect("phase"),
                    sample["comm_dtype"].as_str().expect("comm_dtype"),
                    u32_of("ep_size"),
                    u32_of("node_num"),
                    u32_of("hidden_size"),
                    u32_of("topk"),
                    u32_of("num_experts"),
                    u32_of("num_tokens"),
                    u32_of("sms"),
                )
                .unwrap_or_else(|err| panic!("oracle sample {sample} must resolve: {err}"));
            let want = sample["latency_ms"].as_f64().expect("latency_ms");
            assert!(
                want > 0.0,
                "oracle sample has a non-positive latency: {sample}"
            );
            let rel = ((got - want) / want).abs();
            max_rel = max_rel.max(rel);
            assert!(
                rel <= 1e-9,
                "sample {sample}: rust {got} vs python {want} (rel {rel:e})"
            );
            checked += 1;
        }
        assert!(
            checked >= 150,
            "oracle unexpectedly small: {checked} samples"
        );
        eprintln!("moe_a2a oracle: {checked} samples, max relative error {max_rel:e}");
    }

    /// No source file at all is a typed miss, not a panic.
    #[test]
    fn missing_sources_are_a_typed_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let table = MoeA2aTable::new(tmp.path().to_path_buf());
        match table
            .query(
                "deepep_ht",
                "dispatch",
                "default",
                16,
                2,
                7168,
                8,
                256,
                64,
                20,
            )
            .unwrap_err()
        {
            AicError::PerfDatabase(_) | AicError::Io { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
