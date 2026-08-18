// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified large-EP MoE expert-compute table (`moe_expert_compute_perf.parquet`).
//!
//! Rust port of Python `sdk/operations/moe_comm.py`: `load_moe_expert_compute_data`
//! (new schema) + `_load_legacy_ep` (the three legacy wideep adapters) + the
//! silicon body of `MoEExpertCompute._query_ep_table`.
//!
//! One coordinate serves every inference backend:
//! `[kernel_source][quant][distribution][inference_phase][topk][num_experts]`
//! `[num_slots][hidden_size][inter_size][moe_tp_size][moe_ep_size]`
//! `-> {num_tokens -> latency_ms}`.
//!
//! Four source files feed it:
//!
//! - `moe_expert_compute_perf.parquet` — the unified schema. UNITS: unlike the
//!   us-collected a2a table, its `latency` column is ALREADY in milliseconds
//!   (`load_moe_expert_compute_data`'s docstring / moe_comm.py:916) — stored raw, no
//!   /1000 anywhere. The optional `power` column feeds only the Python
//!   leaves' power/energy ride-alongs, which never influence the latency
//!   resolution; it is not read here.
//! - `wideep_context_moe_perf.parquet` / `wideep_generation_moe_perf.parquet`
//!   -> `kernel_source = "deepep_moe"` (pinned — the legacy column spells it
//!   `deepepmoe` and the oracle loaders never read it), `inference_phase`
//!   from which FILE carried the row, `num_slots = num_experts` (the legacy
//!   sglang tables have no EPLB redundancy axis)
//!   (`_adapt_legacy_sglang_wideep_moe`).
//! - `wideep_moe_perf.parquet` -> native `kernel_source` (default
//!   `"moe_torch_flow"` when the COLUMN is absent), `num_slots` and the
//!   `_eplb` distribution suffixes pass through unchanged. The legacy table
//!   has no context/generation split, so each row registers under BOTH
//!   `inference_phase` values (`_adapt_legacy_trtllm_wideep_moe`).
//!
//! Merge (Python `load_moe_expert_compute_data` / `_store_ep_leaf`): the legacy adapters
//! keep the FIRST row on a collision (`overwrite=False` — their oracle
//! loaders adopted the skip-on-key-conflict shared-layer contract in #1423,
//! same as moe_a2a's keep-first legacy convention). The FIRST new-schema
//! occurrence of a key then overwrites whatever a legacy adapter stored
//! there, and repeats of that key keep the first new-schema value.
//!
//! Query mirrors the silicon body of `_query_ep_table` (moe_comm.py:1160-
//! 1281): `kernel_source` and `quant` resolve EXACTLY
//! (`require_data_slice(data, kernel_source, quant_mode)` — no alias chain
//! on either level, unlike the a2a comm-dtype ladder); the distribution then
//! falls back requested -> `"uniform"` -> FIRST collected distribution that
//! carries the requested `inference_phase`, in dict-INSERTION (file row)
//! order (`_resolve_ep_distribution`; the `dist_order` bookkeeping mirrors
//! the retired `wideep_moe.rs`'s `first_distribution`, extended to a per-(kernel, quant)
//! ordered list because the candidates are phase-filtered); the remaining
//! shape walk is exact. A single-token-point curve queried BELOW its only
//! measurement is a typed miss (the singleton-underflow guard,
//! moe_comm.py:1256-1261); otherwise the token curve rides the shared
//! `perf_interp` engine (1-axis Grid: RAW lerp in range, boundary util-hold
//! beyond it) anchored on the WideEP MoE roofline SOL `get_sol_latency`
//! (moe_comm.py:1221-1237, `num_gemms` from `is_gated`: 3/2) — the same
//! `num_slots`-aware roofline the retired
//! `operators/wideep_moe.rs::sol_latency_ms` implemented.
//!
//! Phase VALIDATION (`_validate_ep_phase`'s `ValueError`) is the operator
//! layer's job — this file is an algorithm-free accessor (the moe_a2a
//! stance), so an unknown `inference_phase` surfaces here as an ordinary
//! typed data miss.
//!
//! Deliberate divergences from Python, both established codebase
//! conventions:
//!
//! - the sglang adapter reads `inter_size` / `moe_tp_size` optionally with
//!   0 / 1 defaults (the convention of the retired
//!   `wideep.rs::load_moe_parquet`, which read the same files; the surviving
//!   reference is Python
//!   `operations/moe_comm.py::_adapt_legacy_sglang_wideep_moe`); Python
//!   indexes them directly (`int(row[...])`), so an
//!   absent column is a hard `KeyError` there — every shipped file carries
//!   both, so the two agree on real data.
//! - `quant` is stored as the raw `moe_dtype` string (the `moe.rs`
//!   convention, shared with the retired `wideep_moe.rs`) and queried via
//!   `MoeQuantMode::name`; Python keys
//!   `common.MoEQuantMode[row["moe_dtype"]]`, so a row with an invalid
//!   moe_dtype CRASHES the Python load (KeyError) but becomes an unreachable
//!   key here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::OnceLock;

use super::axis_curve::AxisCurve;
use super::{kernel_source_ok, SourceResolver};
use crate::common::enums::MoeQuantMode;
use crate::common::error::AicError;
use crate::common::system_spec::{quant_tc_flops, SystemSpec};
use crate::config::{PerfDbSources, PerfSource};
use crate::perf_database::parquet_loader::PerfReader;

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

/// `(kernel_source, quant, distribution, inference_phase, topk, num_experts,
/// num_slots, hidden_size, inter_size, moe_tp_size, moe_ep_size)` — every
/// level of the Python store above the token axis, in the same order, so a
/// `BTreeMap` range scan over a `(kernel, quant, distribution, phase)`
/// prefix answers the distribution chain's carries-phase test.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MoeExpertComputeKey {
    pub kernel_source: String,
    pub quant: String,
    pub distribution: String,
    pub inference_phase: String,
    pub topk: u32,
    pub num_experts: u32,
    pub num_slots: u32,
    pub hidden_size: u32,
    pub inter_size: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
}

/// `num_tokens -> latency_ms` curves keyed by [`MoeExpertComputeKey`], plus the
/// insertion-ordered distribution list per `(kernel_source, quant)` the
/// fallback chain needs. Mirrors the retired `wideep_moe.rs`'s
/// `first_distribution` (Python dict-insertion order is file row order),
/// extended to the FULL
/// ordered list because `_resolve_ep_distribution` filters candidates by
/// inference-phase coverage before taking the first one.
struct MoeEpGrids {
    by_keys: BTreeMap<MoeExpertComputeKey, BTreeMap<u32, f64>>,
    dist_order: BTreeMap<(String, String), Vec<String>>,
}

impl MoeEpGrids {
    fn new() -> Self {
        Self {
            by_keys: BTreeMap::new(),
            dist_order: BTreeMap::new(),
        }
    }

    /// Record the first-seen (file row) order of `key.distribution` under
    /// `(kernel_source, quant)`. Called for every processed row — Python's
    /// defaultdict vivifies the distribution bucket on the store WALK, so
    /// even a keep-first collision cannot reorder (the bucket already
    /// exists whenever a store is skipped).
    fn note_distribution(&mut self, key: &MoeExpertComputeKey) {
        let order = self
            .dist_order
            .entry((key.kernel_source.clone(), key.quant.clone()))
            .or_default();
        if !order.iter().any(|d| d == &key.distribution) {
            order.push(key.distribution.clone());
        }
    }

    /// Python `_store_ep_leaf(..., overwrite=True)`: unconditional
    /// assignment — replaces whatever is there. Used only by the first
    /// new-schema occurrence of a key (precedence over legacy-adapted rows).
    fn store_overwrite(&mut self, key: MoeExpertComputeKey, num_tokens: u32, latency_ms: f64) {
        self.note_distribution(&key);
        self.by_keys
            .entry(key)
            .or_default()
            .insert(num_tokens, latency_ms);
    }

    /// Python `_store_ep_leaf(..., overwrite=False)`: the first stored row
    /// at a coordinate wins. Used by the legacy adapters — their oracle
    /// loaders guard with the skip-on-key-conflict shared-layer contract
    /// (#1423), so an earlier source (or earlier row) takes priority.
    fn store_keep_first(&mut self, key: MoeExpertComputeKey, num_tokens: u32, latency_ms: f64) {
        self.note_distribution(&key);
        self.by_keys
            .entry(key)
            .or_default()
            .entry(num_tokens)
            .or_insert(latency_ms);
    }
}

pub struct MoeExpertComputeTable {
    data_root: PathBuf,
    /// Owns the system spec because the query-time roofline SOL
    /// (`get_sol_latency`) reads `gpu.bfloat16_tc_flops` / `gpu.mem_bw` —
    /// Python's closure captures `database.system_spec` inside
    /// `_query_ep_table`. Same table-owns-spec precedent as
    /// [`super::AttentionTable`].
    spec: SystemSpec,
    /// Ordered, priority-sorted sources per distinct perf-file basename
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`MoeExpertComputeTable::new`).
    moe_ep_sources: Vec<PerfSource>,
    legacy_context_sources: Vec<PerfSource>,
    legacy_generation_sources: Vec<PerfSource>,
    legacy_trtllm_wideep_sources: Vec<PerfSource>,
    grids: OnceLock<Result<MoeEpGrids, AicError>>,
}

impl MoeExpertComputeTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf, spec: SystemSpec) -> Self {
        Self::with_sources(data_root, spec, &SourceResolver::fixed(PerfDbSources::default()))
            .expect("fixed-map resolution is infallible")
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved
    /// from `perf_db_sources` (Python-supplied). Each perf file falls back to
    /// its primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        spec: SystemSpec,
        resolver: &SourceResolver,
    ) -> Result<Self, AicError> {
        let moe_ep_sources = resolver.sources_for("moe_expert_compute_perf.parquet", &data_root)?;
        let legacy_context_sources = resolver.sources_for("wideep_context_moe_perf.parquet", &data_root)?;
        let legacy_generation_sources = resolver.sources_for("wideep_generation_moe_perf.parquet", &data_root)?;
        let legacy_trtllm_wideep_sources =
            resolver.sources_for("wideep_moe_perf.parquet", &data_root)?;
        Ok(Self {
            data_root,
            spec,
            moe_ep_sources,
            legacy_context_sources,
            legacy_generation_sources,
            legacy_trtllm_wideep_sources,
            grids: OnceLock::new(),
        })
    }

    /// Unified EP MoE expert-compute latency (ms).
    ///
    /// Mirrors the SILICON body of Python `MoEExpertCompute._query_ep_table`: exact
    /// kernel/quant walk, distribution chain (requested -> "uniform" ->
    /// first phase-carrying collected distribution in insertion order),
    /// exact shape walk, the singleton-underflow guard, then a 1-D token
    /// curve on the perf_interp engine anchored on the WideEP roofline SOL.
    /// Argument order matches `PerfDatabase.query_moe_expert_compute`.
    #[allow(clippy::too_many_arguments)]
    pub fn query(
        &self,
        kernel_source: &str,
        quant: MoeQuantMode,
        workload_distribution: &str,
        inference_phase: &str,
        topk: u32,
        num_experts: u32,
        num_slots: u32,
        hidden_size: u32,
        inter_size: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        num_tokens: u32,
        is_gated: bool,
    ) -> Result<f64, AicError> {
        // Python's WideEP SOL indexes the bf16 tensor-core rate directly.
        // Resolve it before any perf-data lookup so a missing, non-finite,
        // or non-positive hardware fact is never replaced by a fabricated
        // divisor in the roofline calculation.
        let tc_flops = quant_tc_flops(&self.spec, MoeQuantMode::Bfloat16.mapping())?;
        let grids = self.load()?;
        let quant_name = quant.name();
        // `require_data_slice(data, kernel_source, quant_mode)`: both levels
        // are EXACT — a miss is typed, with no alias/sole-key fallback.
        // `dist_order` carries one entry per (kernel, quant) with any data,
        // so it doubles as the existence index for both levels.
        let slice_key = (kernel_source.to_string(), quant_name.to_string());
        let Some(dist_order) = grids.dist_order.get(&slice_key) else {
            let kernel_seen = grids
                .dist_order
                .keys()
                .any(|(kernel, _)| kernel == kernel_source);
            return Err(AicError::PerfDatabase(if kernel_seen {
                format!(
                    "moe_expert_compute data missing for kernel_source={kernel_source:?} \
                     quant={quant_name:?} at {}",
                    self.data_root.display()
                )
            } else {
                format!(
                    "moe_expert_compute data missing for kernel_source={kernel_source:?} at {}",
                    self.data_root.display()
                )
            }));
        };
        // `_resolve_ep_distribution`: candidates are the collected
        // distributions that carry `inference_phase` data, in insertion
        // order. The key sorts the phase level directly under distribution,
        // so one range probe per candidate answers the coverage test.
        let bound_key = |distribution: &str, fill: u32| MoeExpertComputeKey {
            kernel_source: kernel_source.to_string(),
            quant: quant_name.to_string(),
            distribution: distribution.to_string(),
            inference_phase: inference_phase.to_string(),
            topk: fill,
            num_experts: fill,
            num_slots: fill,
            hidden_size: fill,
            inter_size: fill,
            moe_tp_size: fill,
            moe_ep_size: fill,
        };
        let carries_phase = |distribution: &str| {
            grids
                .by_keys
                .range(bound_key(distribution, 0)..=bound_key(distribution, u32::MAX))
                .next()
                .is_some()
        };
        let used_distribution: &str = if carries_phase(workload_distribution) {
            workload_distribution
        } else if carries_phase("uniform") {
            "uniform"
        } else if let Some(first) = dist_order.iter().find(|dist| carries_phase(dist)) {
            first
        } else {
            return Err(AicError::PerfDatabase(format!(
                "moe_expert_compute workload_distribution {workload_distribution:?} is not available for \
                 {kernel_source}/{quant_name} at {}; no collected distribution carries \
                 {inference_phase:?} data",
                self.data_root.display()
            )));
        };
        // The remaining shape walk is exact (`require_data_slice(quant_slice,
        // used_distribution, inference_phase, topk, ..., moe_ep_size)`) — a
        // shape present only under ANOTHER distribution still misses.
        let key = MoeExpertComputeKey {
            kernel_source: kernel_source.to_string(),
            quant: quant_name.to_string(),
            distribution: used_distribution.to_string(),
            inference_phase: inference_phase.to_string(),
            topk,
            num_experts,
            num_slots,
            hidden_size,
            inter_size,
            moe_tp_size,
            moe_ep_size,
        };
        let curve = grids
            .by_keys
            .get(&key)
            .filter(|curve| !curve.is_empty())
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "moe_expert_compute data missing for {key:?} at {}",
                    self.data_root.display()
                ))
            })?;
        // moe_comm.py:1256-1261: a single measured token point cannot define
        // the low-token launch floor — querying below it is a typed miss.
        if let Some(only) = token_axis_curve(curve).singleton_underflow(num_tokens) {
            return Err(AicError::PerfDatabase(format!(
                "MoE EP silicon token underflow has only one measured point; cannot infer \
                 low-token latency from a singleton. measured_token={only}, requested \
                 num_tokens={num_tokens} for {key:?} at {}",
                self.data_root.display()
            )));
        }
        // `OpInterpConfig(axes=("num_tokens",), resolver=Grid(),
        // sol_fn=get_sol_latency)`: RAW lerp in range; beyond it the boundary
        // util is held and the roofline SOL carries the growth. The engine
        // only evaluates the SOL at integral coordinates (table keys and the
        // integer query), so the round() is parity insurance, the same
        // convention the retired `operators/wideep_moe.rs` used.
        let sol = |tokens: f64| {
            ep_sol_latency_ms(
                &self.spec,
                quant,
                topk,
                num_slots,
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
                tokens.round() as u32,
                is_gated,
                tc_flops,
            )
        };
        token_axis_curve(curve).query(f64::from(num_tokens), &sol)
    }

    /// Distinct `kernel_source` keys present in the loaded table, in sorted
    /// (BTreeMap) order; EMPTY when the table failed to load with a typed
    /// data miss. Mirrors the `if ep_data:` guard of Python
    /// `MoEExpertCompute._resolve_kernel_source` (moe_comm.py:1143-1151) — an
    /// unloaded/empty store is falsy there and the caller keeps its
    /// architecture-preferred kernel. NOTE: Python's `list(ep_data.keys())`
    /// yields dict insertion (file row) order; sorted order is observable
    /// only when several kernels coexist AND the preferred one is absent
    /// (every shipped table collects exactly one kernel). Same accessor and
    /// same caveat as the retired `WideEpMoeTable::available_kernels`
    /// (removed with the wideEP tables, AIC-1601).
    pub fn available_kernels(&self) -> Result<Vec<String>, AicError> {
        let grids = match self.load() {
            Ok(grids) => grids,
            Err(err) if err.is_missing_perf_data() => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };
        let mut names: Vec<String> = Vec::new();
        for key in grids.by_keys.keys() {
            // `MoeExpertComputeKey` sorts by kernel_source first, so consecutive dedup
            // suffices.
            if names.last().map(String::as_str) != Some(key.kernel_source.as_str()) {
                names.push(key.kernel_source.clone());
            }
        }
        Ok(names)
    }

    fn load(&self) -> Result<&MoeEpGrids, AicError> {
        let cell = self.grids.get_or_init(|| {
            load_moe_ep_grids(
                &self.moe_ep_sources,
                &self.legacy_context_sources,
                &self.legacy_generation_sources,
                &self.legacy_trtllm_wideep_sources,
            )
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// `kernel_source` every legacy sglang wideep row is pinned to
/// (`_adapt_legacy_sglang_wideep_moe`; spec §4.2 — the legacy column spells
/// it `deepepmoe` and is never read).
pub(crate) const SGLANG_ADAPTED_KERNEL_SOURCE: &str = "deepep_moe";

/// `kernel_source` Python assumes when the legacy trtllm wideep file has no
/// such COLUMN (`row.get("kernel_source", "moe_torch_flow")`). A column that
/// exists with a NULL cell is a different case — see
/// [`adapt_legacy_trtllm_wideep_moe`].
pub(crate) const LEGACY_TRTLLM_DEFAULT_KERNEL_SOURCE: &str = "moe_torch_flow";

/// `num_gemms` in the roofline SOL follows `is_gated` (3 gated / 2
/// non-gated), mirroring the Python SOL and the legacy oracle
/// (moe.py:309).

/// WideEP MoE roofline SOL (ms) — verbatim `get_sol_latency`
/// (moe_comm.py:1223-1237), the same math the retired
/// `operators/wideep_moe.rs::sol_latency_ms` carried, with `num_slots` (not
/// `num_experts`) sizing the weight-read term. Python evaluates every term
/// in (arbitrary-precision) integer floor division; u64 mirrors it exactly
/// over the collected/queried ranges. `.max(1)` on the divisors only guards
/// a corrupt zero from panicking (Python would ZeroDivisionError there).
#[allow(clippy::too_many_arguments)]
fn ep_sol_latency_ms(
    spec: &SystemSpec,
    quant: MoeQuantMode,
    topk: u32,
    num_slots: u32,
    hidden_size: u32,
    inter_size: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
    tokens: u32,
    is_gated: bool,
    tc_flops: f64,
) -> f64 {
    // moe_comm.py:1224: `total_tokens = tokens * topk`.
    let total_tokens = tokens as u64 * topk as u64;
    let moe_expert_compute = (moe_ep_size as u64).max(1);
    let moe_tp = (moe_tp_size as u64).max(1);
    let h = hidden_size as u64;
    let inter = inter_size as u64;
    let slots = num_slots as u64;
    // moe_comm.py:1225: `total_tokens * hidden_size * inter_size * num_gemms
    // * 2 // moe_ep_size // moe_tp_size`.
    // Gated (SwiGLU) = 3 GEMMs, non-gated (Relu2) = 2 — mirrors the Python
    // SOL's `3 if is_gated else 2` (legacy oracle moe.py:309).
    let num_gemms: u64 = if is_gated { 3 } else { 2 };
    let ops = total_tokens * h * inter * num_gemms * 2 / moe_expert_compute / moe_tp;
    // moe_comm.py:1226-1234: the three integer byte terms, then one float
    // multiply by `quant_mode.value.memory`.
    let mem_bytes_int = total_tokens / moe_expert_compute * h * 2 // input+output (:1227)
        + total_tokens / moe_expert_compute * inter * num_gemms / moe_tp // intermediate activations (:1228)
        + h * inter * num_gemms / moe_tp
            * std::cmp::min(slots / moe_expert_compute, total_tokens / moe_expert_compute); // weights, num_slots-aware (:1229-1233)
    let mem_bytes = (mem_bytes_int as f64) * quant.mapping().memory;
    // moe_comm.py:1235: Python indexes `bfloat16_tc_flops` directly. The
    // caller resolves the same field through the strict shared resolver.
    let sol_math = (ops as f64) / (tc_flops * quant.mapping().compute) * 1000.0;
    // moe_comm.py:1236.
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    // moe_comm.py:1237.
    sol_math.max(sol_mem)
}

/// Load the unified table: legacy adapters first — context, generation,
/// trtllm-wideep, each keeping the first row on a collision (#1423
/// shared-layer contract) — then the
/// new schema (first occurrence of a key overwrites, repeats keep first) —
/// Python `load_moe_expert_compute_data` + `_load_legacy_ep`.
fn load_moe_ep_grids(
    moe_ep_sources: &[PerfSource],
    context_sources: &[PerfSource],
    generation_sources: &[PerfSource],
    trtllm_wideep_sources: &[PerfSource],
) -> Result<MoeEpGrids, AicError> {
    let mut grids = MoeEpGrids::new();
    let mut any_source = adapt_legacy_sglang_wideep_moe(context_sources, "context", &mut grids)?;
    any_source |= adapt_legacy_sglang_wideep_moe(generation_sources, "generation", &mut grids)?;
    any_source |= adapt_legacy_trtllm_wideep_moe(trtllm_wideep_sources, &mut grids)?;
    any_source |= load_new_schema(moe_ep_sources, &mut grids)?;
    if !any_source || grids.by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MoE EP rows loaded from {} source(s) (moe_expert_compute + 3 legacy wideep tables; first: {})",
            moe_ep_sources.len()
                + context_sources.len()
                + generation_sources.len()
                + trtllm_wideep_sources.len(),
            moe_ep_sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_else(|| "<no moe_expert_compute sources>".to_string())
        )));
    }
    Ok(grids)
}

/// `_adapt_legacy_sglang_wideep_moe` (moe_comm.py:743-774): every row keys
/// under the pinned `"deepep_moe"` kernel with `num_slots = num_experts` and
/// the phase of the FILE the row came from; the `latency` column is already
/// ms (stored raw); first row wins on a collision (#1423 contract). Returns
/// whether any source file exists — Python's exists-but-empty semantic
/// (`_read_filtered_rows` yields `None` only when EVERY path is missing).
///
/// `inter_size` / `moe_tp_size` are read optionally with 0 / 1 defaults —
/// the convention of the retired `wideep.rs::load_moe_parquet`, which read
/// the same files; the surviving reference is Python
/// `operations/moe_comm.py::_adapt_legacy_sglang_wideep_moe` (which indexes
/// them directly — see the module doc's divergence note).
fn adapt_legacy_sglang_wideep_moe(
    sources: &[PerfSource],
    inference_phase: &str,
    grids: &mut MoeEpGrids,
) -> Result<bool, AicError> {
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let moe_dtype_col = reader.col("moe_dtype")?;
        let distribution_col = reader.col("distribution")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let inter_size_col = reader.col_optional("inter_size");
        let moe_tp_size_col = reader.col_optional("moe_tp_size");
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            // moe_comm.py:759-772: `num_slots = num_experts` — the legacy
            // sglang tables have no EPLB redundancy axis.
            let num_experts = row.u32(num_experts_col)?;
            let key = MoeExpertComputeKey {
                kernel_source: SGLANG_ADAPTED_KERNEL_SOURCE.to_string(),
                quant: row.str_owned(moe_dtype_col)?,
                distribution: row.str_owned(distribution_col)?,
                inference_phase: inference_phase.to_string(),
                topk: row.u32(topk_col)?,
                num_experts,
                num_slots: num_experts,
                hidden_size: row.u32(hidden_size_col)?,
                inter_size: row.u32_optional(inter_size_col)?.unwrap_or(0),
                moe_tp_size: row.u32_optional(moe_tp_size_col)?.unwrap_or(1),
                moe_ep_size: row.u32(moe_ep_size_col)?,
            };
            grids.store_keep_first(key, row.u32(num_tokens_col)?, row.f64(latency_col)?);
        }
    }
    Ok(any_source)
}

/// `_adapt_legacy_trtllm_wideep_moe` (moe_comm.py:785-816): native
/// `kernel_source` (the `"moe_torch_flow"` default fires only when the
/// COLUMN is absent; a present-but-NULL cell reads back as `""` in Python's
/// `_read_perf_rows` and is stored under that empty key — there is no
/// backend mapping to drop it, unlike moe_a2a's legacy alltoall adapter),
/// `num_slots` and the `_eplb` distributions pass through, latency already
/// ms, first-row-wins assignment (#1423 shared-layer contract). The legacy
/// table has no phase split, so
/// each row registers under BOTH `inference_phase` values.
fn adapt_legacy_trtllm_wideep_moe(
    sources: &[PerfSource],
    grids: &mut MoeEpGrids,
) -> Result<bool, AicError> {
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let ks_col = reader.col_optional("kernel_source");
        let moe_dtype_col = reader.col("moe_dtype")?;
        let distribution_col = reader.col("distribution")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let num_slots_col = reader.col("num_slots")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let inter_size_col = reader.col("inter_size")?;
        let moe_tp_size_col = reader.col("moe_tp_size")?;
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let latency_col = reader.col("latency")?;
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            // Python defaults only when the COLUMN is absent
            // (`row.get("kernel_source", "moe_torch_flow")`); a NULL cell
            // reads back as "" (`_read_perf_rows`) and stays a key.
            let kernel_source = match ks_col {
                None => LEGACY_TRTLLM_DEFAULT_KERNEL_SOURCE.to_string(),
                Some(_) => row.str_optional(ks_col)?.unwrap_or("").to_string(),
            };
            let quant = row.str_owned(moe_dtype_col)?;
            let distribution = row.str_owned(distribution_col)?;
            let topk = row.u32(topk_col)?;
            let num_experts = row.u32(num_experts_col)?;
            let num_slots = row.u32(num_slots_col)?;
            let hidden_size = row.u32(hidden_size_col)?;
            let inter_size = row.u32(inter_size_col)?;
            let moe_tp_size = row.u32(moe_tp_size_col)?;
            let moe_ep_size = row.u32(moe_ep_size_col)?;
            let num_tokens = row.u32(num_tokens_col)?;
            let latency_ms = row.f64(latency_col)?;
            // moe_comm.py:814-816: no context/generation split in the legacy
            // table — one row registers under BOTH phases.
            for inference_phase in ["context", "generation"] {
                let key = MoeExpertComputeKey {
                    kernel_source: kernel_source.clone(),
                    quant: quant.clone(),
                    distribution: distribution.clone(),
                    inference_phase: inference_phase.to_string(),
                    topk,
                    num_experts,
                    num_slots,
                    hidden_size,
                    inter_size,
                    moe_tp_size,
                    moe_ep_size,
                };
                grids.store_keep_first(key, num_tokens, latency_ms);
            }
        }
    }
    Ok(any_source)
}

/// New-schema `moe_expert_compute_perf.parquet` rows (moe_comm.py:900-929). The
/// `latency` column is ALREADY in milliseconds — stored raw, no conversion.
/// The FIRST occurrence of a key overwrites whatever a legacy adapter stored
/// there; repeats of that key keep the first new-schema value.
fn load_new_schema(sources: &[PerfSource], grids: &mut MoeEpGrids) -> Result<bool, AicError> {
    let mut any_source = false;
    let mut seen: BTreeSet<(MoeExpertComputeKey, u32)> = BTreeSet::new();
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let ks_col = reader.col("kernel_source")?;
        let moe_dtype_col = reader.col("moe_dtype")?;
        let distribution_col = reader.col("distribution")?;
        let inference_phase_col = reader.col("inference_phase")?;
        let topk_col = reader.col("topk")?;
        let num_experts_col = reader.col("num_experts")?;
        let num_slots_col = reader.col("num_slots")?;
        let hidden_size_col = reader.col("hidden_size")?;
        let inter_size_col = reader.col("inter_size")?;
        let moe_tp_size_col = reader.col("moe_tp_size")?;
        let moe_ep_size_col = reader.col("moe_ep_size")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let latency_col = reader.col("latency")?;
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), Some(ks_col), &row)? {
                continue;
            }
            let key = MoeExpertComputeKey {
                kernel_source: row.str_owned(ks_col)?,
                quant: row.str_owned(moe_dtype_col)?,
                distribution: row.str_owned(distribution_col)?,
                // Stored as collected; the phase is validated at query time
                // (moe_comm.py:906).
                inference_phase: row.str_owned(inference_phase_col)?,
                topk: row.u32(topk_col)?,
                num_experts: row.u32(num_experts_col)?,
                num_slots: row.u32(num_slots_col)?,
                hidden_size: row.u32(hidden_size_col)?,
                inter_size: row.u32(inter_size_col)?,
                moe_tp_size: row.u32(moe_tp_size_col)?,
                moe_ep_size: row.u32(moe_ep_size_col)?,
            };
            let num_tokens = row.u32(num_tokens_col)?;
            // moe_comm.py:916: `latency` is ALREADY ms — stored raw.
            let latency_ms = row.f64(latency_col)?;
            grids.note_distribution(&key);
            // moe_comm.py:920-929: the first occurrence overwrites a
            // legacy-adapted leaf; repeats keep the first new-schema value.
            if seen.insert((key.clone(), num_tokens)) {
                grids
                    .by_keys
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
    use crate::common::system_spec::{GpuSpec, MiscSpec, NodeSpec};
    use parquet::data_type::{ByteArray, ByteArrayType, DoubleType, Int64Type};
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::{SerializedFileWriter, SerializedRowGroupWriter};
    use parquet::schema::parser::parse_message_type;
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    /// Synthetic spec for the roofline SOL: mem_bw = 1e9 B/s,
    /// bfloat16_tc_flops = 1e12 — round numbers the tests' hand-computed
    /// expected values divide through.
    fn test_spec() -> SystemSpec {
        SystemSpec {
            data_dir: PathBuf::from("data/synthetic"),
            gpu: GpuSpec {
                mem_bw: 1e9,
                mem_bw_empirical_scaling_factor: 1.0,
                mem_empirical_constant_latency: 0.0,
                mem_capacity: None,
                bfloat16_tc_flops: Some(1e12),
                int8_tc_flops: None,
                fp8_tc_flops: None,
                fp4_tc_flops: None,
                power: None,
                sm_version: None,
            },
            node: NodeSpec {
                num_gpus_per_node: 8,
                inter_node_bw: 100.0,
                intra_node_bw: 900.0,
                pcie_bw: None,
                p2p_latency: 0.0,
                num_gpus_per_rack: None,
                inter_rack_bw: None,
            },
            misc: MiscSpec::default(),
        }
    }

    fn write_column<T: parquet::data_type::DataType>(
        rg: &mut SerializedRowGroupWriter<'_, File>,
        values: &[T::T],
    ) {
        let mut col = rg.next_column().unwrap().unwrap();
        col.typed::<T>().write_batch(values, None, None).unwrap();
        col.close().unwrap();
    }

    /// One synthetic new-schema row. Shape is fixed at (topk=8, experts=256,
    /// hidden=7168, inter=2048, moe_tp=1) — the axes under test are
    /// kernel/quant/distribution/phase/slots/ep/tokens.
    #[derive(Clone)]
    struct EpRow {
        kernel_source: &'static str,
        moe_dtype: &'static str,
        distribution: &'static str,
        inference_phase: &'static str,
        num_slots: i64,
        moe_ep_size: i64,
        num_tokens: i64,
        latency_ms: f64,
    }

    #[allow(clippy::too_many_arguments)]
    fn ep_row(
        kernel_source: &'static str,
        moe_dtype: &'static str,
        distribution: &'static str,
        inference_phase: &'static str,
        num_slots: i64,
        moe_ep_size: i64,
        num_tokens: i64,
        latency_ms: f64,
    ) -> EpRow {
        EpRow {
            kernel_source,
            moe_dtype,
            distribution,
            inference_phase,
            num_slots,
            moe_ep_size,
            num_tokens,
            latency_ms,
        }
    }

    /// Write a synthetic new-schema `moe_expert_compute_perf.parquet`.
    fn write_moe_ep_parquet(path: &Path, rows: &[EpRow]) {
        let schema = Arc::new(
            parse_message_type(
                "message ep {
                    REQUIRED BYTE_ARRAY kernel_source (UTF8);
                    REQUIRED BYTE_ARRAY moe_dtype (UTF8);
                    REQUIRED BYTE_ARRAY distribution (UTF8);
                    REQUIRED BYTE_ARRAY inference_phase (UTF8);
                    REQUIRED INT64 topk;
                    REQUIRED INT64 num_experts;
                    REQUIRED INT64 num_slots;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 inter_size;
                    REQUIRED INT64 moe_tp_size;
                    REQUIRED INT64 moe_ep_size;
                    REQUIRED INT64 num_tokens;
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
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.kernel_source))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.moe_dtype))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.distribution))
                .collect::<Vec<_>>(),
        );
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.inference_phase))
                .collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.num_slots).collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![2048_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![1_i64; n]);
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.moe_ep_size).collect::<Vec<_>>(),
        );
        write_column::<Int64Type>(
            &mut rg,
            &rows.iter().map(|r| r.num_tokens).collect::<Vec<_>>(),
        );
        write_column::<DoubleType>(
            &mut rg,
            &rows.iter().map(|r| r.latency_ms).collect::<Vec<_>>(),
        );
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Write a synthetic legacy sglang wideep context/generation parquet.
    /// Rows are `(distribution, moe_ep_size, num_tokens, latency_ms)`; shape
    /// fixed at (moe_dtype=fp8_block, hidden=7168, inter=2048, topk=8,
    /// experts=256, moe_tp=1). The `kernel_source` column carries the legacy
    /// `"deepepmoe"` spelling the adapter must IGNORE (it pins
    /// `"deepep_moe"`). `with_inter_and_tp = false` omits the `inter_size` /
    /// `moe_tp_size` columns entirely — the loader must then key at
    /// inter=0 / tp=1 (the defaults inherited from the retired `wideep.rs`).
    fn write_legacy_sglang_parquet(
        path: &Path,
        rows: &[(&'static str, i64, i64, f64)],
        with_inter_and_tp: bool,
    ) {
        let inter_decl = if with_inter_and_tp {
            "REQUIRED INT64 inter_size;"
        } else {
            ""
        };
        let tp_decl = if with_inter_and_tp {
            "REQUIRED INT64 moe_tp_size;"
        } else {
            ""
        };
        let schema = Arc::new(
            parse_message_type(&format!(
                "message sglang_wideep {{
                    REQUIRED BYTE_ARRAY kernel_source (UTF8);
                    REQUIRED BYTE_ARRAY moe_dtype (UTF8);
                    REQUIRED INT64 num_tokens;
                    REQUIRED INT64 hidden_size;
                    {inter_decl}
                    REQUIRED INT64 topk;
                    REQUIRED INT64 num_experts;
                    {tp_decl}
                    REQUIRED INT64 moe_ep_size;
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
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("deepepmoe"); n]);
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("fp8_block"); n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        if with_inter_and_tp {
            write_column::<Int64Type>(&mut rg, &vec![2048_i64; n]);
        }
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        if with_inter_and_tp {
            write_column::<Int64Type>(&mut rg, &vec![1_i64; n]);
        }
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.0))
                .collect::<Vec<_>>(),
        );
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Write a synthetic legacy trtllm `wideep_moe_perf.parquet`. Rows are
    /// `(kernel_source, distribution, num_slots, moe_ep_size, num_tokens,
    /// latency_ms)`; shape fixed at (moe_dtype=nvfp4, hidden=7168,
    /// inter=2048, topk=8, experts=256, moe_tp=1).
    /// `with_kernel_source_column = false` omits the column entirely (the
    /// Python `row.get` default's ONLY trigger); when present it is OPTIONAL
    /// so a `None` row writes a NULL cell (Python reads that back as `""`).
    fn write_legacy_trtllm_wideep_parquet(
        path: &Path,
        rows: &[(Option<&'static str>, &'static str, i64, i64, i64, f64)],
        with_kernel_source_column: bool,
    ) {
        let ks_decl = if with_kernel_source_column {
            "OPTIONAL BYTE_ARRAY kernel_source (UTF8);"
        } else {
            ""
        };
        let schema = Arc::new(
            parse_message_type(&format!(
                "message trtllm_wideep {{
                    {ks_decl}
                    REQUIRED BYTE_ARRAY moe_dtype (UTF8);
                    REQUIRED INT64 num_tokens;
                    REQUIRED INT64 hidden_size;
                    REQUIRED INT64 inter_size;
                    REQUIRED INT64 topk;
                    REQUIRED INT64 num_experts;
                    REQUIRED INT64 num_slots;
                    REQUIRED INT64 moe_tp_size;
                    REQUIRED INT64 moe_ep_size;
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
        if with_kernel_source_column {
            // Optional column: only non-null values go in `values`, with a
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
        }
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("nvfp4"); n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.4).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &vec![7168_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![2048_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![8_i64; n]);
        write_column::<Int64Type>(&mut rg, &vec![256_i64; n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &vec![1_i64; n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        write_column::<ByteArrayType>(
            &mut rg,
            &rows
                .iter()
                .map(|r| ByteArray::from(r.1))
                .collect::<Vec<_>>(),
        );
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.5).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    fn approx(got: f64, want: f64) {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "got {got}, want {want}"
        );
    }

    /// Query shorthand over the fixed synthetic shape (topk=8, experts=256,
    /// hidden=7168, inter=2048, tp=1).
    #[allow(clippy::too_many_arguments)]
    fn q(
        table: &MoeExpertComputeTable,
        kernel_source: &str,
        quant: MoeQuantMode,
        distribution: &str,
        phase: &str,
        num_slots: u32,
        moe_ep_size: u32,
        num_tokens: u32,
    ) -> Result<f64, AicError> {
        table.query(
            kernel_source,
            quant,
            distribution,
            phase,
            8,
            256,
            num_slots,
            7168,
            2048,
            1,
            moe_ep_size,
            num_tokens,
            true,
        )
    }

    // ------------------------------------------------------------------
    // New-schema loader
    // ------------------------------------------------------------------

    /// R6 units: the unified `latency` column is ALREADY milliseconds —
    /// stored raw (a spurious /1000 would return 0.00025 here). Every key
    /// axis is addressable, and the two phases hold independent leaves.
    #[test]
    fn new_schema_stores_ms_raw_and_keys_all_axes() {
        let tmp = tempfile::tempdir().unwrap();
        write_moe_ep_parquet(
            &tmp.path().join("moe_expert_compute_perf.parquet"),
            &[
                ep_row(
                    "deepep_moe",
                    "fp8_block",
                    "uniform",
                    "context",
                    256,
                    16,
                    128,
                    0.25,
                ),
                ep_row(
                    "deepep_moe",
                    "fp8_block",
                    "uniform",
                    "generation",
                    256,
                    16,
                    128,
                    0.5,
                ),
                ep_row(
                    "deepgemm",
                    "fp8_block",
                    "uniform",
                    "context",
                    288,
                    16,
                    128,
                    0.75,
                ),
            ],
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        approx(
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "uniform",
                "context",
                256,
                16,
                128,
            )
            .unwrap(),
            0.25,
        );
        approx(
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "uniform",
                "generation",
                256,
                16,
                128,
            )
            .unwrap(),
            0.5,
        );
        // num_slots is a real key axis: 288 hits only the deepgemm row.
        approx(
            q(
                &table,
                "deepgemm",
                MoeQuantMode::Fp8Block,
                "uniform",
                "context",
                288,
                16,
                128,
            )
            .unwrap(),
            0.75,
        );
        assert!(q(
            &table,
            "deepgemm",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            256,
            16,
            128
        )
        .is_err());
        // A phase never collected for a slice is a typed miss (validation is
        // the operator layer's job; "prefill" is simply not a key here).
        assert!(q(
            &table,
            "deepgemm",
            MoeQuantMode::Fp8Block,
            "uniform",
            "generation",
            288,
            16,
            128
        )
        .is_err());
        assert!(q(
            &table,
            "deepep_moe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "prefill",
            256,
            16,
            128
        )
        .is_err());
    }

    // ------------------------------------------------------------------
    // Legacy adapters
    // ------------------------------------------------------------------

    /// `_adapt_legacy_sglang_wideep_moe`: kernel pinned to `"deepep_moe"`
    /// (the column's `"deepepmoe"` spelling must NOT be a key),
    /// `num_slots = num_experts`, the phase comes from which FILE carried
    /// the row, and latency is already ms. The two files use DISTINCT ep
    /// coordinates so a row leaking into the other phase would be a
    /// direct key hit rather than masked by token interpolation.
    #[test]
    fn legacy_sglang_context_and_generation_adapters() {
        let tmp = tempfile::tempdir().unwrap();
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_context_moe_perf.parquet"),
            &[("uniform", 2, 32, 0.375)],
            true,
        );
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_generation_moe_perf.parquet"),
            &[("uniform", 4, 2, 0.125)],
            true,
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        approx(
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "uniform",
                "context",
                256,
                2,
                32,
            )
            .unwrap(),
            0.375,
        );
        approx(
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "uniform",
                "generation",
                256,
                4,
                2,
            )
            .unwrap(),
            0.125,
        );
        // The legacy column spelling is not a kernel key.
        assert!(q(
            &table,
            "deepepmoe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            256,
            2,
            32
        )
        .is_err());
        // num_slots defaulted to num_experts: 288 misses.
        assert!(q(
            &table,
            "deepep_moe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            288,
            2,
            32
        )
        .is_err());
        // No both-phase duplication on the sglang adapters: the context
        // row's ep=2 coordinate is absent under "generation", and the
        // generation row's ep=4 coordinate is absent under "context".
        assert!(q(
            &table,
            "deepep_moe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "generation",
            256,
            2,
            32
        )
        .is_err());
        assert!(q(
            &table,
            "deepep_moe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            256,
            4,
            2
        )
        .is_err());
    }

    /// An sglang legacy file without `inter_size` / `moe_tp_size` columns
    /// keys at inter=0 / tp=1 (the defaults inherited from the retired
    /// `wideep.rs::load_moe_parquet`).
    #[test]
    fn legacy_sglang_missing_inter_and_tp_columns_default() {
        let tmp = tempfile::tempdir().unwrap();
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_context_moe_perf.parquet"),
            &[("uniform", 2, 32, 0.25)],
            false,
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        approx(
            table
                .query(
                    "deepep_moe",
                    MoeQuantMode::Fp8Block,
                    "uniform",
                    "context",
                    8,
                    256,
                    256,
                    7168,
                    0, // inter_size default
                    1, // moe_tp_size default
                    2,
                    32,
                    true,
                )
                .unwrap(),
            0.25,
        );
        // The regular inter=2048 coordinate must NOT exist.
        assert!(q(
            &table,
            "deepep_moe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            256,
            2,
            32
        )
        .is_err());
    }

    /// Intra-legacy collisions keep the FIRST row (Python
    /// `_store_ep_leaf(..., overwrite=False)` — the #1423 skip-on-conflict
    /// shared-layer contract, same as moe_a2a's legacy convention).
    #[test]
    fn legacy_sglang_duplicate_rows_first_wins() {
        let tmp = tempfile::tempdir().unwrap();
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_context_moe_perf.parquet"),
            &[("uniform", 2, 32, 0.1), ("uniform", 2, 32, 0.9)],
            true,
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        approx(
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "uniform",
                "context",
                256,
                2,
                32,
            )
            .unwrap(),
            0.1,
        );
    }

    /// `_adapt_legacy_trtllm_wideep_moe`: one legacy row registers under
    /// BOTH phases with the same value, `num_slots` passes through (EPLB
    /// redundant: 288 slots over 256 experts), the native kernel_source is
    /// the key, and a later-loaded trtllm row cannot displace an
    /// sglang-adapted leaf at the same key (adapter order: context,
    /// generation, trtllm — all keep-first per the #1423 contract).
    #[test]
    fn legacy_trtllm_registers_both_phases_and_passes_num_slots() {
        let tmp = tempfile::tempdir().unwrap();
        write_legacy_trtllm_wideep_parquet(
            &tmp.path().join("wideep_moe_perf.parquet"),
            &[(
                Some("wideep_compute_cutlass"),
                "power_law_1.01_eplb",
                288,
                2,
                1,
                0.0611904,
            )],
            true,
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        for phase in ["context", "generation"] {
            approx(
                q(
                    &table,
                    "wideep_compute_cutlass",
                    MoeQuantMode::Nvfp4,
                    "power_law_1.01_eplb",
                    phase,
                    288,
                    2,
                    1,
                )
                .unwrap(),
                0.0611904,
            );
        }
        // num_slots passes through: the num_slots == num_experts coordinate
        // must NOT exist for this row.
        assert!(q(
            &table,
            "wideep_compute_cutlass",
            MoeQuantMode::Nvfp4,
            "power_law_1.01_eplb",
            "context",
            256,
            2,
            1,
        )
        .is_err());

        // Cross-adapter ordering: all three legacy adapters keep the FIRST
        // stored row (#1423), so the trtllm adapter — which loads last —
        // cannot displace an sglang-adapted leaf. The two writers pin
        // different moe_dtype axes (sglang fp8_block, trtllm nvfp4), so the
        // shared `deepep_moe` kernel key below exercises the adapter order
        // while the sglang fp8_block leaf stays untouched. Legacy-vs-new
        // layering is pinned by the dedicated merge test below.
        let tmp2 = tempfile::tempdir().unwrap();
        write_legacy_sglang_parquet(
            &tmp2.path().join("wideep_context_moe_perf.parquet"),
            &[("uniform", 2, 32, 0.1)],
            true,
        );
        // Same unified coordinate as the sglang row above: kernel deepep_moe,
        // dtype fp8_block is NOT expressible here (this writer pins nvfp4) —
        // the collision below therefore uses kernel_source="deepep_moe" with
        // nvfp4, asserting only the adapter ORDER mechanics on a shared
        // kernel key, while the sglang fp8_block leaf stays untouched.
        write_legacy_trtllm_wideep_parquet(
            &tmp2.path().join("wideep_moe_perf.parquet"),
            &[
                (Some("deepep_moe"), "uniform", 256, 2, 32, 0.7),
                (Some("deepep_moe"), "uniform", 256, 2, 32, 0.8),
            ],
            true,
        );
        let table2 = MoeExpertComputeTable::new(tmp2.path().to_path_buf(), test_spec());
        // sglang leaf intact under fp8_block...
        approx(
            q(
                &table2,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "uniform",
                "context",
                256,
                2,
                32,
            )
            .unwrap(),
            0.1,
        );
        // ...and the duplicated trtllm rows resolved first-wins under nvfp4
        // (#1423 skip-on-conflict contract), on both phases.
        for phase in ["context", "generation"] {
            approx(
                q(
                    &table2,
                    "deepep_moe",
                    MoeQuantMode::Nvfp4,
                    "uniform",
                    phase,
                    256,
                    2,
                    32,
                )
                .unwrap(),
                0.7,
            );
        }
    }

    /// The `"moe_torch_flow"` default fires only when the kernel_source
    /// COLUMN is absent; a present-but-NULL cell keys under `""` (Python's
    /// `_read_perf_rows` null -> `""`), NOT under the default.
    #[test]
    fn legacy_trtllm_kernel_source_column_absent_defaults_null_is_empty() {
        // Column absent -> every row keys under "moe_torch_flow".
        let tmp = tempfile::tempdir().unwrap();
        write_legacy_trtllm_wideep_parquet(
            &tmp.path().join("wideep_moe_perf.parquet"),
            &[(None, "uniform", 256, 2, 1, 1.5)],
            false,
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        approx(
            q(
                &table,
                "moe_torch_flow",
                MoeQuantMode::Nvfp4,
                "uniform",
                "context",
                256,
                2,
                1,
            )
            .unwrap(),
            1.5,
        );

        // Column present, one NULL cell -> that row keys under "".
        let tmp2 = tempfile::tempdir().unwrap();
        write_legacy_trtllm_wideep_parquet(
            &tmp2.path().join("wideep_moe_perf.parquet"),
            &[
                (Some("wideep_compute_cutlass"), "uniform", 256, 2, 1, 2.5),
                (None, "uniform", 256, 4, 1, 9.0),
            ],
            true,
        );
        let table2 = MoeExpertComputeTable::new(tmp2.path().to_path_buf(), test_spec());
        approx(
            q(
                &table2,
                "wideep_compute_cutlass",
                MoeQuantMode::Nvfp4,
                "uniform",
                "context",
                256,
                2,
                1,
            )
            .unwrap(),
            2.5,
        );
        approx(
            q(
                &table2,
                "",
                MoeQuantMode::Nvfp4,
                "uniform",
                "context",
                256,
                4,
                1,
            )
            .unwrap(),
            9.0,
        );
        // The null-cell row must NOT have defaulted.
        assert!(q(
            &table2,
            "moe_torch_flow",
            MoeQuantMode::Nvfp4,
            "uniform",
            "context",
            256,
            4,
            1
        )
        .is_err());
    }

    // ------------------------------------------------------------------
    // Merge semantics (`_store_ep_leaf` overwrite / keep-first)
    // ------------------------------------------------------------------

    /// Legacy rows load first; the FIRST new-schema row at the same key
    /// overwrites them, and a repeat of that key keeps the first new-schema
    /// value. Keys the new schema does not cover keep their legacy value.
    #[test]
    fn new_schema_overwrites_legacy_and_repeats_keep_first() {
        let tmp = tempfile::tempdir().unwrap();
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_context_moe_perf.parquet"),
            &[("uniform", 2, 64, 0.1), ("uniform", 2, 128, 0.2)],
            true,
        );
        write_moe_ep_parquet(
            &tmp.path().join("moe_expert_compute_perf.parquet"),
            &[
                ep_row(
                    "deepep_moe",
                    "fp8_block",
                    "uniform",
                    "context",
                    256,
                    2,
                    64,
                    0.7,
                ),
                ep_row(
                    "deepep_moe",
                    "fp8_block",
                    "uniform",
                    "context",
                    256,
                    2,
                    64,
                    0.9,
                ),
            ],
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        approx(
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "uniform",
                "context",
                256,
                2,
                64,
            )
            .unwrap(),
            0.7,
        );
        // The t=128 leaf the new schema never covered keeps its legacy value.
        approx(
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "uniform",
                "context",
                256,
                2,
                128,
            )
            .unwrap(),
            0.2,
        );
    }

    // ------------------------------------------------------------------
    // Distribution chain (`_resolve_ep_distribution`)
    // ------------------------------------------------------------------

    /// requested -> "uniform" -> FIRST phase-carrying collected distribution
    /// in INSERTION (file row) order -> typed miss. Candidates are
    /// phase-scoped: a distribution collected only under the other phase is
    /// not a candidate.
    #[test]
    fn distribution_chain_requested_uniform_first_available_and_phase_scoping() {
        let tmp = tempfile::tempdir().unwrap();
        // Context: "zeta_dist" is inserted FIRST but sorts LAST — the
        // first-available leg must follow insertion order, not BTreeMap
        // order. No "uniform" under context.
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_context_moe_perf.parquet"),
            &[("zeta_dist", 2, 64, 0.111), ("alpha_dist", 2, 64, 0.222)],
            true,
        );
        // Generation carries "uniform" plus its own distribution.
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_generation_moe_perf.parquet"),
            &[("uniform", 2, 64, 0.333), ("gen_only", 2, 64, 0.444)],
            true,
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        let ctx = |dist: &str| {
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                dist,
                "context",
                256,
                2,
                64,
            )
        };
        let gen = |dist: &str| {
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                dist,
                "generation",
                256,
                2,
                64,
            )
        };
        // Requested present -> itself.
        approx(ctx("zeta_dist").unwrap(), 0.111);
        approx(ctx("alpha_dist").unwrap(), 0.222);
        // Requested absent, no uniform under context -> FIRST-INSERTED
        // context distribution ("zeta_dist", though "alpha_dist" sorts
        // first).
        approx(ctx("power_law").unwrap(), 0.111);
        // Phase scoping: "gen_only" exists under this (kernel, quant) but
        // carries no context data -> not a candidate -> same first-available.
        approx(ctx("gen_only").unwrap(), 0.111);
        // Requested absent but "uniform" collected for generation -> uniform
        // (beats first-available even though "uniform" was inserted first
        // here anyway; the uniform leg is order-independent).
        approx(gen("power_law").unwrap(), 0.333);
        // Requested present under generation -> itself.
        approx(gen("gen_only").unwrap(), 0.444);

        // No collected distribution carries the phase at all: a
        // generation-only table queried for context is a typed miss.
        let tmp2 = tempfile::tempdir().unwrap();
        write_legacy_sglang_parquet(
            &tmp2.path().join("wideep_generation_moe_perf.parquet"),
            &[("uniform", 2, 64, 0.5)],
            true,
        );
        let table2 = MoeExpertComputeTable::new(tmp2.path().to_path_buf(), test_spec());
        assert!(q(
            &table2,
            "deepep_moe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            256,
            2,
            64
        )
        .is_err());
    }

    /// The distribution fallback never crosses the shape walk: a shape
    /// collected only under a non-requested distribution still misses when
    /// the requested distribution exists (with the phase) but lacks the
    /// shape.
    #[test]
    fn distribution_fallback_does_not_rescue_a_missing_shape() {
        let tmp = tempfile::tempdir().unwrap();
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_context_moe_perf.parquet"),
            &[("uniform", 2, 64, 0.1), ("power_law_0.8", 4, 64, 0.2)],
            true,
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        // ep=4 exists only under power_law_0.8; requesting uniform at ep=4
        // resolves the distribution to uniform (present + phase-carrying)
        // and then misses the shape.
        assert!(q(
            &table,
            "deepep_moe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            256,
            4,
            64
        )
        .is_err());
        approx(
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Fp8Block,
                "power_law_0.8",
                "context",
                256,
                4,
                64,
            )
            .unwrap(),
            0.2,
        );
    }

    // ------------------------------------------------------------------
    // Kernel / quant exactness
    // ------------------------------------------------------------------

    /// `require_data_slice(data, kernel_source, quant_mode)`: both levels
    /// are exact — no alias chain, no sole-key fallback (unlike the a2a
    /// comm-dtype ladder).
    #[test]
    fn kernel_and_quant_are_exact_typed_misses() {
        let tmp = tempfile::tempdir().unwrap();
        write_legacy_sglang_parquet(
            &tmp.path().join("wideep_context_moe_perf.parquet"),
            &[("uniform", 2, 64, 0.1)],
            true,
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        // fp8_block is the SOLE collected quant, yet nvfp4 must still miss.
        match q(
            &table,
            "deepep_moe",
            MoeQuantMode::Nvfp4,
            "uniform",
            "context",
            256,
            2,
            64,
        )
        .unwrap_err()
        {
            AicError::PerfDatabase(msg) => assert!(msg.contains("quant"), "got: {msg}"),
            other => panic!("unexpected error: {other:?}"),
        }
        // fp8 does NOT alias into fp8_block on this table.
        assert!(q(
            &table,
            "deepep_moe",
            MoeQuantMode::Fp8,
            "uniform",
            "context",
            256,
            2,
            64
        )
        .is_err());
        // Unknown kernel is a typed miss too.
        match q(
            &table,
            "deepgemm",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            256,
            2,
            64,
        )
        .unwrap_err()
        {
            AicError::PerfDatabase(msg) => {
                assert!(msg.contains("kernel_source"), "got: {msg}")
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Token resolution: lerp, roofline util-hold, singleton guard
    // ------------------------------------------------------------------

    /// Expected SOL values for the fixed synthetic shape under
    /// `MoeQuantMode::Bfloat16` (memory=2.0, compute=1.0) and `test_spec()`
    /// (mem_bw=1e9, flops=1e12): hand-folding moe_comm.py:1223-1237 at
    /// topk=8, hidden=7168, inter=2048, tp=1, ep=2, slots=256 gives
    /// `mem_int(t) = 81920*t + 44040192*min(128, 4*t)` bytes
    /// (57344t input/output + 24576t intermediate + the num_slots-capped
    /// weight read), which dwarfs the math term everywhere below — so
    /// `sol(t) = mem_int(t) * 2.0 / 1e9 * 1000.0` ms.
    fn hand_sol_ms(tokens: u64) -> f64 {
        let mem_int = 81920 * tokens + 44040192 * (4 * tokens).min(128);
        (mem_int as f64) * 2.0 / 1e9 * 1000.0
    }

    #[test]
    fn query_rejects_missing_or_invalid_bfloat16_flops() {
        let tmp = tempfile::tempdir().unwrap();
        write_moe_ep_parquet(
            &tmp.path().join("moe_expert_compute_perf.parquet"),
            &[ep_row(
                "deepep_moe",
                "bfloat16",
                "uniform",
                "context",
                256,
                2,
                32,
                0.5,
            )],
        );

        for value in [None, Some(0.0), Some(f64::NAN)] {
            let mut spec = test_spec();
            spec.gpu.bfloat16_tc_flops = value;
            let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), spec);
            match q(
                &table,
                "deepep_moe",
                MoeQuantMode::Bfloat16,
                "uniform",
                "context",
                256,
                2,
                32,
            ) {
                Err(AicError::MissingSystemFlops(message)) => {
                    assert!(message.contains("bfloat16_tc_flops"), "got: {message}");
                }
                other => panic!("expected MissingSystemFlops, got {other:?}"),
            }
        }
    }

    /// In-range token queries are RAW lerp on the measured points (the SOL
    /// never enters); beyond the collected range the boundary util is held
    /// and the ROOFLINE carries the growth — the expected values below are
    /// hand-computed from the Python formula and are decisively different
    /// from a linear-token proxy (which would give 0.8 * 100/64 = 1.25 for
    /// the overflow query, vs ~0.8004 here: the weight term saturates).
    #[test]
    fn token_curve_lerp_and_roofline_util_hold() {
        let tmp = tempfile::tempdir().unwrap();
        write_moe_ep_parquet(
            &tmp.path().join("moe_expert_compute_perf.parquet"),
            &[
                ep_row(
                    "deepep_moe",
                    "bfloat16",
                    "uniform",
                    "context",
                    256,
                    2,
                    32,
                    0.5,
                ),
                ep_row(
                    "deepep_moe",
                    "bfloat16",
                    "uniform",
                    "context",
                    256,
                    2,
                    64,
                    0.8,
                ),
            ],
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        let at = |tokens: u32| {
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Bfloat16,
                "uniform",
                "context",
                256,
                2,
                tokens,
            )
            .unwrap()
        };
        // Exact hits and the interior midpoint lerp.
        approx(at(32), 0.5);
        approx(at(64), 0.8);
        approx(at(48), 0.65);
        // Overflow (q=100 > 64): util-hold on the k_tail=1 boundary point —
        // latency = sol(100) / (sol(64) / lat(64)).
        approx(at(100), hand_sol_ms(100) / (hand_sol_ms(64) / 0.8));
        // Multi-point underflow (q=16 < 32) holds the LOW boundary util —
        // allowed (only the singleton case is a miss); the weight term is
        // unsaturated at t=16 (4t=64 < 128), so the roofline is nonlinear
        // across this hold too.
        approx(at(16), hand_sol_ms(16) / (hand_sol_ms(32) / 0.5));
    }

    /// moe_comm.py:1256-1261: a singleton curve queried BELOW its only
    /// measured point is a typed miss; at the point it answers exactly, and
    /// ABOVE it the ordinary util-hold applies.
    #[test]
    fn singleton_token_curve_underflow_is_a_typed_miss() {
        let tmp = tempfile::tempdir().unwrap();
        write_moe_ep_parquet(
            &tmp.path().join("moe_expert_compute_perf.parquet"),
            &[ep_row(
                "deepep_moe",
                "bfloat16",
                "uniform",
                "context",
                256,
                2,
                64,
                0.8,
            )],
        );
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        let at = |tokens: u32| {
            q(
                &table,
                "deepep_moe",
                MoeQuantMode::Bfloat16,
                "uniform",
                "context",
                256,
                2,
                tokens,
            )
        };
        match at(32).unwrap_err() {
            AicError::PerfDatabase(msg) => {
                assert!(msg.contains("singleton"), "got: {msg}")
            }
            other => panic!("unexpected error: {other:?}"),
        }
        approx(at(64).unwrap(), 0.8);
        approx(at(128).unwrap(), hand_sol_ms(128) / (hand_sol_ms(64) / 0.8));
    }

    // ------------------------------------------------------------------
    // Python oracle over the shipped h200_sxm/sglang + gb200/trtllm data
    // ------------------------------------------------------------------

    const LFS_POINTER_PREFIX: &[u8] = b"version https://git-lfs";

    /// Whether the shipped parquet files behind `data_root` are usable: at
    /// least one of the four basenames resolves to a real file and none of
    /// the resolved files is an unresolved git-lfs pointer. `git lfs pull`
    /// is a checkout-time step, so a pointer-only tree skips the
    /// data-dependent oracle instead of failing it (same pointer detection
    /// as `parquet_loader::PerfReader::open`).
    fn shipped_data_ready(data_root: &Path) -> bool {
        use std::io::Read;
        let mut any_file = false;
        for basename in [
            "moe_expert_compute_perf.parquet",
            "wideep_context_moe_perf.parquet",
            "wideep_generation_moe_perf.parquet",
            "wideep_moe_perf.parquet",
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

    /// Full-table parity against `PerfDatabase.query_moe_expert_compute`. The fixture is
    /// generated by `parity_tests/gen_moe_expert_compute_oracle.py` (regeneration
    /// command in the JSON's `_regenerate` field) from the shipped
    /// h200_sxm/sglang/0.5.6.post2 (legacy sglang wideep pair — both phases,
    /// six distributions incl. the uniform-fallback leg) and
    /// gb200/trtllm/1.3.0rc10 (legacy trtllm wideep — both-phase
    /// duplication, `_eplb` distributions, num_slots=288 EPLB-redundant
    /// slices, and the no-uniform first-available fallback leg) data,
    /// stratified over exact points, token lerps, token overflow/underflow
    /// roofline util-holds, and the distribution chain.
    ///
    /// NOTE(shared-layer merge): the oracle is generated with
    /// `shared_layer=False` and `MoeExpertComputeTable::new` resolves single primary
    /// sources with no kernel_source filter, so both sides read exactly the
    /// same rows.
    #[test]
    fn moe_ep_matches_python_oracle() {
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("testdata/moe_expert_compute_oracle.json"))
                .expect("oracle fixture must parse");
        let systems =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems");
        let samples = oracle["samples"].as_array().expect("samples array");
        let mut tables: BTreeMap<String, MoeExpertComputeTable> = BTreeMap::new();
        let mut max_rel = 0.0_f64;
        let mut checked = 0_usize;
        for sample in samples {
            let rel_root = sample["data_root"].as_str().expect("data_root");
            let data_root = systems.join(rel_root);
            if !shipped_data_ready(&data_root) {
                eprintln!(
                    "SKIP moe_ep_matches_python_oracle: shipped perf data unavailable at {} \
                     (run `git lfs pull`)",
                    data_root.display()
                );
                return;
            }
            let table = tables.entry(rel_root.to_string()).or_insert_with(|| {
                let system = sample["system"].as_str().expect("system");
                let spec = SystemSpec::load(&systems.join(format!("{system}.yaml")))
                    .expect("system yaml must load");
                MoeExpertComputeTable::new(data_root.clone(), spec)
            });
            let u32_of = |field: &str| {
                u32::try_from(sample[field].as_u64().expect(field)).expect("fits in u32")
            };
            let quant: MoeQuantMode = serde_json::from_value(sample["quant"].clone())
                .expect("quant must map to a MoeQuantMode");
            let got = table
                .query(
                    sample["kernel_source"].as_str().expect("kernel_source"),
                    quant,
                    sample["distribution"].as_str().expect("distribution"),
                    sample["inference_phase"].as_str().expect("inference_phase"),
                    u32_of("topk"),
                    u32_of("num_experts"),
                    u32_of("num_slots"),
                    u32_of("hidden_size"),
                    u32_of("inter_size"),
                    u32_of("moe_tp_size"),
                    u32_of("moe_ep_size"),
                    u32_of("num_tokens"),
                    true, // oracle rows were generated at the gated default
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
        eprintln!("moe_expert_compute oracle: {checked} samples, max relative error {max_rel:e}");
    }

    /// No source file at all is a typed miss, not a panic.
    #[test]
    fn missing_sources_are_a_typed_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let table = MoeExpertComputeTable::new(tmp.path().to_path_buf(), test_spec());
        match q(
            &table,
            "deepep_moe",
            MoeQuantMode::Fp8Block,
            "uniform",
            "context",
            256,
            2,
            64,
        )
        .unwrap_err()
        {
            AicError::PerfDatabase(_) | AicError::Io { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
