// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MoE operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.moe.MoE._query_moe_table`. The
//! perf-DB layer handles workload-distribution fallback to `"uniform"` and
//! resolves the token curve on the perf_interp v2 engine; this operator
//! supplies the MoE roofline SOL closure the engine's beyond-range util-hold
//! anchors on (Python v2 deleted the op-level overflow estimator — the
//! engine's `k_tail=1`, unclamped util-hold replaces it).
//!
//! Database-mode dispatch follows the gemm.rs reference pattern: EMPIRICAL
//! always estimates `SOL(query)/util`; HYBRID queries silicon and converts a
//! typed missing-data error into the estimate; SILICON is unchanged. When
//! the op's own `(quant, shape)` slice has no collected data the empirical
//! path walks the transfer ladder (`operations/moe.py:446-570`):
//!
//! 1. **xshape** — nearest collected shape within the query quant;
//! 2. **xquant** — sibling quant with the SAME (memory, compute) profile,
//!    util reconstructed with the QUERY quant's SOL;
//! 3. **xprofile** — nearest-profile collected quant, util reconstructed
//!    with the REFERENCE quant's own SOL, rescaled by the per-quant
//!    util-LEVEL ratio `e(query)/e(ref)`.
//!
//! Policy-disabled tiers are skipped (not an error); the terminal
//! [`AicError::EmpiricalNotImplemented`] only surfaces when every permitted
//! tier found nothing.
//!
//! The SGLang `moe_backend == "deepep_moe"` branch of Python's `_moe_table`
//! routes BOTH the silicon lookup and the empirical calibration (own-shape
//! grid + transfer ladder) to the wideep context/generation MoE tables —
//! mirrored here via [`MoeTableSel`]. (The TRT-LLM WideEP compute table is a
//! different op: `WideEpMoeOp`, `operators/wideep_moe.rs`.)
//!
//! Weights accounting (per-expert FFN weights + router) is in the model
//! layer; the operator returns latency only.

use serde::{Deserialize, Serialize};
use crate::common::enums::{DatabaseMode, MoeQuantMode, TransferKind, TransferPolicy};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::moe::{MoeKernel, MoeSiblingSlice};
use crate::perf_database::PerfDatabase;
use std::sync::Arc;

/// Per-quant achieved-util LEVEL `e(q)` for MoE, keyed by the
/// `(memory, compute)` profile. Mirrors `_MOE_QUANT_UTIL_LEVEL`
/// (`operations/moe.py:87-97`); consumed ONLY by the cross-profile tier,
/// and only as the ratio `e(query)/e(ref)`.
const MOE_QUANT_UTIL_LEVEL: &[(f64, f64, f64)] = &[
    (2.0, 1.0, 0.53),    // w16a16 / bfloat16              [data]
    (1.0, 1.0, 0.45),    // w8a16                          [inferred]
    (0.5, 1.0, 0.07),    // w4a16 (int4_wo, mxfp4)         [data]
    (1.0, 2.0, 0.40),    // w8a8 / fp8(_block)             [data]
    (0.5, 2.0, 0.15),    // w4a8 (w4afp8, mxfp4_mxfp8)     [data]
    (1.0, 4.0, 0.30),    // w8a4                           [inferred]
    (0.5, 4.0, 0.23),    // w4a4                           [data ≈ nvfp4]
    (0.5625, 4.0, 0.23), // w4a4 / nvfp4                   [data]
];
/// Unlisted profile: mid-range relative level (Python `_MOE_QUANT_UTIL_DEFAULT`).
const MOE_QUANT_UTIL_DEFAULT: f64 = 0.30;

/// Achieved-util level `e(q)` for a MoE quant, by `(memory, compute)`
/// profile (mirrors `_moe_quant_util_level`, `operations/moe.py:100-102`).
fn moe_quant_util_level(quant: MoeQuantMode) -> f64 {
    let mapping = quant.mapping();
    MOE_QUANT_UTIL_LEVEL
        .iter()
        .find(|(memory, compute, _)| *memory == mapping.memory && *compute == mapping.compute)
        .map(|(_, _, level)| *level)
        .unwrap_or(MOE_QUANT_UTIL_DEFAULT)
}

/// Every MoE quant variant, for parsing perf-table `moe_dtype` strings back
/// into the enum (Python's table is keyed by enum members directly).
const ALL_MOE_QUANTS: &[MoeQuantMode] = &[
    MoeQuantMode::Bfloat16,
    MoeQuantMode::Fp8,
    MoeQuantMode::Int4Wo,
    MoeQuantMode::Fp8Block,
    MoeQuantMode::W4afp8,
    MoeQuantMode::Nvfp4,
    MoeQuantMode::W4a16Mxfp4,
    MoeQuantMode::W4a8Mxfp4Mxfp8,
    MoeQuantMode::W4a8Mxfp4Mxfp8Trtllm,
    MoeQuantMode::W4a16Mxfp4Cutlass,
];

fn moe_quant_from_name(name: &str) -> Option<MoeQuantMode> {
    ALL_MOE_QUANTS.iter().copied().find(|q| q.name() == name)
}

/// Collected quants with a DIFFERENT `(memory, compute)` profile than the
/// query, nearest-profile first (stable sort by `|Δmemory| + |Δcompute|`;
/// mirrors `_xprofile_moe_quants`, `operations/moe.py:105-117`). Python
/// breaks distance ties by table insertion (file row) order — the accessor
/// feeds that load order in, and the stable sort preserves it.
fn xprofile_moe_quants(query: MoeQuantMode, table_quants: &[MoeQuantMode]) -> Vec<MoeQuantMode> {
    let qp = query.mapping();
    let mut refs: Vec<MoeQuantMode> = table_quants
        .iter()
        .copied()
        .filter(|q| {
            let m = q.mapping();
            *q != query && !(m.memory == qp.memory && m.compute == qp.compute)
        })
        .collect();
    let dist = |q: MoeQuantMode| {
        let m = q.mapping();
        (m.memory - qp.memory).abs() + (m.compute - qp.compute).abs()
    };
    refs.sort_by(|a, b| dist(*a).partial_cmp(&dist(*b)).expect("finite profile distances"));
    refs
}

/// Enabled-tier fingerprint folded into reference-grid cache keys so grids
/// selected under different policies cannot alias (Python's
/// `selection_key`/`identity_key` include the policy frozenset).
fn policy_fingerprint(policy: TransferPolicy) -> String {
    format!(
        "xshape={},xquant={},xprofile={},xop={}",
        policy.xshape as u8, policy.xquant as u8, policy.xprofile as u8, policy.xop as u8
    )
}

/// Which perf table calibrates the EMPIRICAL path. Mirrors Python's
/// `_moe_table()` selection (`operations/moe.py:364-397`): SGLang
/// `moe_backend == "deepep_moe"` routes to the wideep context/generation
/// MoE tables; nvfp4 small-token gated probes the TRT-LLM low-latency
/// split; everything else uses the default table.
#[derive(Clone, Copy, PartialEq)]
enum MoeTableSel {
    Standard,
    LowLatency,
    Wideep { is_context: bool },
}

impl MoeTableSel {
    /// Grid cache-key tag. Python folds `kernel_tag` ("std" / "ll" /
    /// "wideep") plus `id(node)` into the key; the ctx/gen split here plays
    /// the node-identity role, so the two wideep tables cannot alias.
    fn tag(self) -> &'static str {
        match self {
            Self::Standard => "std",
            Self::LowLatency => "ll",
            Self::Wideep { is_context: true } => "wideep_ctx",
            Self::Wideep { is_context: false } => "wideep_gen",
        }
    }
}

/// The perf-DB kernel grid behind a non-wideep selector. Callers match the
/// wideep variants off to `WideEpTable` accessors before reaching this.
fn moe_kernel(table: MoeTableSel) -> MoeKernel {
    match table {
        MoeTableSel::Standard => MoeKernel::Standard,
        MoeTableSel::LowLatency => MoeKernel::LowLatency,
        MoeTableSel::Wideep { .. } => unreachable!("wideep selectors dispatch to WideEpTable"),
    }
}

/// A sibling slice the transfer ladder may borrow: the reference slice's
/// shape + token curve, the quant whose SOL reconstructs its util
/// (`sol_quant`: QUERY quant for same-profile tiers, REFERENCE quant for
/// cross-profile), and the transfer-tier provenance tag. Mirrors
/// `util_empirical.ReferenceCandidate` as built by `_collect`
/// (`operations/moe.py:454-486`).
struct MoeReferenceCandidate {
    slice: MoeSiblingSlice,
    ref_quant: MoeQuantMode,
    sol_quant: MoeQuantMode,
    provenance: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoeOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    pub inter_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
    /// Attention data-parallel size. With attention-dp, every dp rank's
    /// tokens all-gather into the SHARED expert pool, so the MoE compute op
    /// sees `num_tokens * attention_dp_size` tokens (mirrors Python
    /// `MoE.query`: `x = x * attention_dp_size`, operations/moe.py).
    /// Dropping the multiplier under-predicted MoE latency ~4.7x on dp=8
    /// DeepSeek configs. Absent in pre-existing specs -> 0 -> treated as 1
    /// at the query site.
    #[serde(default)]
    pub attention_dp_size: u32,
    pub quant_mode: MoeQuantMode,
    pub workload_distribution: String,
    /// Gated FFN (SwiGLU) when true; non-gated (Relu²) when false.
    /// Mirrors Python's `MoE._is_gated`. The TRT-LLM small-token
    /// `moe_torch_flow_min_latency` kernel is only valid for gated nvfp4
    /// MoE; non-gated paths (e.g. NemotronH) must skip it.
    pub is_gated: bool,
    /// SGLang MoE backend (Python `MoE._moe_backend`). `Some("deepep_moe")`
    /// routes the compute lookup to the wideep context/generation MoE tables
    /// instead of `moe_perf` (operations/moe.py sglang branch). Absent in
    /// pre-existing specs -> None -> the regular table.
    #[serde(default)]
    pub moe_backend: Option<String>,
    /// EPLB enabled (Python `MoE._enable_eplb`). On the sglang branch the
    /// PREFILL token count is corrected to `int(num_tokens * 0.8)`
    /// (operations/moe.py: expert-parallel load balancing evens the
    /// per-expert token distribution).
    #[serde(default)]
    pub enable_eplb: bool,
    /// Context (prefill) op — selects the wideep CONTEXT MoE table under
    /// deepep and gates the EPLB prefill correction (Python `MoE._is_context`).
    #[serde(default)]
    pub is_context: bool,
}

impl MoeOp {
    pub fn new(
        name: impl Into<String>,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: MoeQuantMode,
        workload_distribution: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            attention_dp_size: 1,
            quant_mode,
            workload_distribution: workload_distribution.into(),
            is_gated: true,
            moe_backend: None,
            enable_eplb: false,
            is_context: false,
        }
    }

    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // Attention-dp scales up the total input tokens (all dp ranks
        // all-gather into one shared expert pool) -- mirrors Python
        // `MoE.query` (`x = x * attention_dp_size`). Applied exactly once,
        // before the perf-DB resolution keys off the token count.
        let num_tokens = num_tokens.saturating_mul(self.attention_dp_size.max(1));

        // Database-mode dispatch, mirroring the Python `_query_moe_table`
        // tail (`database._query_silicon_or_hybrid`): EMPIRICAL always
        // estimates; HYBRID converts a typed silicon miss into the estimate;
        // SILICON is unchanged. The SOL diagnostic modes never reach the
        // compiled engine.
        match db.database_mode {
            DatabaseMode::Empirical => Ok(PerformanceResult::new(
                self.empirical_latency(db, num_tokens)?,
                Source::Empirical,
            )
            .clamp_non_negative()
            .scaled(self.scale_factor)),
            DatabaseMode::Hybrid => match self.silicon_pr(db, num_tokens) {
                Ok(result) => Ok(result),
                Err(err) if err.is_missing_perf_data() => Ok(PerformanceResult::new(
                    self.empirical_latency(db, num_tokens)?,
                    Source::Empirical,
                )
                .clamp_non_negative()
                .scaled(self.scale_factor)),
                Err(err) => Err(err),
            },
            _ => self.silicon_pr(db, num_tokens),
        }
    }

    /// SILICON resolution (deepep routing + low-latency probe + the default
    /// grid, scale/clamp applied per branch — the audit-PR body, unchanged).
    fn silicon_pr(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        let is_sglang = db.backend == "sglang";
        // SGLang EPLB prefill correction — INSIDE get_silicon only (Python
        // operations/moe.py:684, sglang branch: `num_tokens_corrected =
        // int(num_tokens * 0.8) if enable_eplb and is_context else
        // num_tokens`). The EMPIRICAL closures receive RAW tokens
        // (moe.py:637-647, 803-813), so the correction must not leak there.
        let num_tokens = if is_sglang && self.enable_eplb && self.is_context {
            (num_tokens as f64 * 0.8) as u32
        } else {
            num_tokens
        };
        // The roofline SOL the perf-DB engine anchors its beyond-range
        // util-hold on (Python `_resolve_tokens` passes the same closure).
        // Coordinates arriving from the engine are always integral (table
        // keys / the u32 query), so rounding to u32 keeps the integer
        // floor-division parity with Python's `get_sol`. This replaces the
        // deleted op-level SOL-anchored overflow estimator (the engine's
        // `k_tail=1` util-hold handles beyond-range queries).
        let sol = |t: f64| self.sol_latency_ms(db, t.round() as u32);

        // SGLang DeepEP (wideep) routes MoE compute to the wideep
        // context/generation tables (Python operations/moe.py:
        // `if moe_backend == "deepep_moe": moe_data = _wideep_*_moe_data`),
        // resolved through the SAME `_resolve_tokens` semantics (singleton
        // guard + MoE-roofline util-hold, threaded via `sol`).
        if is_sglang && self.moe_backend.as_deref() == Some("deepep_moe") {
            let latency = if self.is_context {
                db.wideep.query_context_moe(
                    num_tokens,
                    self.hidden_size,
                    self.inter_size,
                    self.topk,
                    self.num_experts,
                    self.moe_tp_size,
                    self.moe_ep_size,
                    self.quant_mode,
                    &self.workload_distribution,
                    &sol,
                )?
            } else {
                db.wideep.query_generation_moe(
                    num_tokens,
                    self.hidden_size,
                    self.inter_size,
                    self.topk,
                    self.num_experts,
                    self.moe_tp_size,
                    self.moe_ep_size,
                    self.quant_mode,
                    &self.workload_distribution,
                    &sol,
                )?
            };
            return Ok(PerformanceResult::new(latency, Source::Silicon)
                .clamp_non_negative()
                .scaled(self.scale_factor));
        }

        // Mirrors Python's MoE._query_moe_table TRT-LLM gate: for nvfp4
        // gated MoE at num_tokens <= 128, probe the
        // `moe_torch_flow_min_latency` grid first and fall back to the
        // default grid on a shape miss. Other backends (vLLM, SGLang) never
        // have `kernel_source` populated, so `low_latency_available()`
        // returns false and this short-circuits.
        if num_tokens <= 128
            && self.quant_mode == MoeQuantMode::Nvfp4
            && self.is_gated
            && db.moe.low_latency_available()?
        {
            if let Some(ll) = db.moe.query_low_latency(
                num_tokens,
                self.hidden_size,
                self.inter_size,
                self.topk,
                self.num_experts,
                self.moe_tp_size,
                self.moe_ep_size,
                self.quant_mode,
                &self.workload_distribution,
                &sol,
            )? {
                return Ok(PerformanceResult::new(ll, Source::Silicon)
                    .clamp_non_negative()
                    .scaled(self.scale_factor));
            }
        }
        let latency = db.moe.query(
            num_tokens,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.moe_tp_size,
            self.moe_ep_size,
            self.quant_mode,
            &self.workload_distribution,
            &sol,
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// `SOL(query)/util` with the full transfer ladder. Mirrors Python
    /// `MoE._query_moe_table::get_empirical` (`operations/moe.py:327-572`):
    /// own-shape grid → xshape → xquant → xprofile → typed empirical miss.
    fn empirical_latency(&self, db: &PerfDatabase, num_tokens: u32) -> Result<f64, AicError> {
        let spec = &db.system_spec;
        let quant = self.quant_mode;
        let num_gemms: u64 = if self.is_gated { 3 } else { 2 };
        let sol_time = moe_sol_latency_ms(
            spec,
            quant,
            num_gemms,
            num_tokens,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.moe_tp_size,
            self.moe_ep_size,
        );

        // Table selection mirrors get_silicon's (`_moe_table`,
        // `operations/moe.py:364-397`): the SGLang deepep branch comes FIRST
        // and routes the whole calibration (own-shape grid + ladder) to the
        // wideep context/generation tables; otherwise nvfp4 + small tokens +
        // gated probes the low-latency table for the FULL slice and falls
        // back to the default table on a shape miss. Building util from the
        // wrong table would over-estimate by the ~3x kernel gap. The tag
        // folds the choice into every grid cache key so one table's grid
        // can't be served to another's query at the same shape.
        let table = if db.backend == "sglang" && self.moe_backend.as_deref() == Some("deepep_moe")
        {
            MoeTableSel::Wideep {
                is_context: self.is_context,
            }
        } else if num_tokens <= 128
            && quant == MoeQuantMode::Nvfp4
            && self.is_gated
            && db.moe.low_latency_available()?
        {
            match self.slice_points(db, MoeTableSel::LowLatency) {
                Ok(_) => MoeTableSel::LowLatency,
                Err(err) if err.is_missing_perf_data() => MoeTableSel::Standard,
                Err(err) => return Err(err),
            }
        } else {
            MoeTableSel::Standard
        };
        let kernel_tag = table.tag();

        // Own-shape grid over this slice's num_tokens curve (depth 1).
        let own_sol = |c: &[f64]| {
            moe_sol_latency_ms(
                spec,
                quant,
                num_gemms,
                c[0].round() as u32,
                self.hidden_size,
                self.inter_size,
                self.topk,
                self.num_experts,
                self.moe_tp_size,
                self.moe_ep_size,
            )
        };
        let own_key = format!(
            "moe:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
            quant.name(),
            kernel_tag,
            self.topk,
            self.num_experts,
            self.hidden_size,
            self.inter_size,
            self.moe_tp_size,
            self.moe_ep_size,
            self.workload_distribution,
            num_gemms,
        );
        let mut grid = db.util_grids.get_or_try_build(&own_key, || {
            match self.slice_points(db, table) {
                Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(
                    points.into_iter().map(|(t, lat)| (vec![t as f64], lat)),
                    own_sol,
                )))),
                // Typed coverage miss -> no grid (the ladder below, then
                // estimate(), takes over); schema/load errors propagate.
                Err(err) if err.is_missing_perf_data() => Ok(None),
                Err(err) => Err(err),
            }
        })?;

        let mut util_scale = 1.0;
        if grid.as_deref().is_none_or(UtilGrid::is_empty) {
            let policy = db.transfer_policy;

            // Tiers 1+2 flow through ONE reference selection (`_moe_candidates`
            // + a single grid_from_reference, `operations/moe.py:490-532`):
            // xshape candidates win outright when any exist; only an empty
            // xshape set falls through to same-profile xquant siblings.
            let mut candidates: Vec<MoeReferenceCandidate> = Vec::new();
            if policy.contains(TransferKind::XShape) {
                self.collect_candidates(db, table, quant, quant, "xshape", &mut candidates)?;
            }
            if candidates.is_empty() && policy.contains(TransferKind::XQuant) {
                let qp = quant.mapping();
                for name in self.table_available_quants(db, table)? {
                    let Some(sibling) = moe_quant_from_name(&name) else {
                        continue;
                    };
                    let mapping = sibling.mapping();
                    if sibling == quant
                        || mapping.memory != qp.memory
                        || mapping.compute != qp.compute
                    {
                        continue;
                    }
                    self.collect_candidates(db, table, sibling, quant, "xquant", &mut candidates)?;
                }
            }
            if let Some(reference) =
                self.reference_grid(db, kernel_tag, num_gemms, policy, &candidates)?
            {
                grid = Some(reference);
            }

            // Tier 3: cross-PROFILE. No own- or same-profile data at all ->
            // borrow the nearest collected quant's util curve, built with the
            // REFERENCE quant's own SOL, and rescale by the per-quant
            // util-LEVEL ratio e(query)/e(ref) (`operations/moe.py:541-570`).
            if grid.as_deref().is_none_or(UtilGrid::is_empty)
                && policy.contains(TransferKind::XProfile)
            {
                let table_quants: Vec<MoeQuantMode> = self
                    .table_available_quants(db, table)?
                    .iter()
                    .filter_map(|name| moe_quant_from_name(name))
                    .collect();
                for ref_quant in xprofile_moe_quants(quant, &table_quants) {
                    let mut candidates: Vec<MoeReferenceCandidate> = Vec::new();
                    self.collect_candidates(
                        db,
                        table,
                        ref_quant,
                        ref_quant,
                        "xprofile",
                        &mut candidates,
                    )?;
                    if let Some(reference) =
                        self.reference_grid(db, kernel_tag, num_gemms, policy, &candidates)?
                    {
                        if !reference.is_empty() {
                            grid = Some(reference);
                            util_scale =
                                moe_quant_util_level(quant) / moe_quant_util_level(ref_quant);
                            break;
                        }
                    }
                }
            }
        }

        let query = [num_tokens as f64];
        let (latency, _) = util_empirical::estimate(sol_time, &query, grid.as_deref(), util_scale)?;
        // Tier fired = the reference grid's transfer kind (xshape / xquant /
        // xprofile), or own-shape "empirical" when no borrow happened —
        // Python `prov` at moe.py:447/534/569 passed to estimate() at :571.
        db.note_provenance(
            grid.as_deref()
                .and_then(|g| g.reference_provenance)
                .and_then(util_empirical::ProvenanceTier::from_tag)
                .unwrap_or(util_empirical::ProvenanceTier::Empirical),
        );
        Ok(latency)
    }

    /// This op's own-slice token curve on the selected table.
    fn slice_points(&self, db: &PerfDatabase, table: MoeTableSel) -> Result<Vec<(u32, f64)>, AicError> {
        match table {
            MoeTableSel::Standard | MoeTableSel::LowLatency => db.moe.slice_points(
                moe_kernel(table),
                self.quant_mode.name(),
                &self.workload_distribution,
                self.topk,
                self.num_experts,
                self.hidden_size,
                self.inter_size,
                self.moe_tp_size,
                self.moe_ep_size,
            ),
            MoeTableSel::Wideep { is_context } => db.wideep.moe_slice_points(
                is_context,
                self.quant_mode.name(),
                &self.workload_distribution,
                self.topk,
                self.num_experts,
                self.hidden_size,
                self.inter_size,
                self.moe_tp_size,
                self.moe_ep_size,
            ),
        }
    }

    /// Distinct quant names of the selected table, in first-seen (file row)
    /// order — Python's `for q in moe_table` dict-insertion iteration.
    fn table_available_quants(
        &self,
        db: &PerfDatabase,
        table: MoeTableSel,
    ) -> Result<Vec<String>, AicError> {
        match table {
            MoeTableSel::Standard | MoeTableSel::LowLatency => {
                db.moe.available_quants(moe_kernel(table))
            }
            MoeTableSel::Wideep { is_context } => db.wideep.moe_available_quants(is_context),
        }
    }

    /// Enumerate `source_quant`'s collected sibling slices (same table,
    /// same wl-after-fallback / moe_tp / moe_ep) as ladder candidates.
    /// Mirrors `_collect` (`operations/moe.py:454-486`); a typed data miss
    /// (table failed to load) yields no candidates, exactly like Python's
    /// `grid_from_reference` catching the raise from `_collect`.
    fn collect_candidates(
        &self,
        db: &PerfDatabase,
        table: MoeTableSel,
        source_quant: MoeQuantMode,
        sol_quant: MoeQuantMode,
        provenance: &'static str,
        out: &mut Vec<MoeReferenceCandidate>,
    ) -> Result<(), AicError> {
        let slices = match table {
            MoeTableSel::Standard | MoeTableSel::LowLatency => db.moe.sibling_slices(
                moe_kernel(table),
                source_quant.name(),
                &self.workload_distribution,
                self.moe_tp_size,
                self.moe_ep_size,
            ),
            MoeTableSel::Wideep { is_context } => db.wideep.moe_sibling_slices(
                is_context,
                source_quant.name(),
                &self.workload_distribution,
                self.moe_tp_size,
                self.moe_ep_size,
            ),
        };
        let slices = match slices {
            Ok(slices) => slices,
            Err(err) if err.is_missing_perf_data() => return Ok(()),
            Err(err) => return Err(err),
        };
        out.extend(slices.into_iter().map(|slice| MoeReferenceCandidate {
            slice,
            ref_quant: source_quant,
            sol_quant,
            provenance,
        }));
        Ok(())
    }

    /// Nearest-candidate selection + reference-grid build, mirroring
    /// `util_empirical.grid_from_reference`: the candidate nearest to the
    /// query's `(topk, num_experts, hidden, inter)` in normalised log space
    /// wins (first-wins on ties), and its grid is built with the CANDIDATE's
    /// own shape bound into the SOL. `None` when there are no candidates.
    /// The cache key carries the op identity, the query slice, the
    /// enabled-tier fingerprint, the selected reference identity, and the
    /// provenance so differently-policied lookups cannot alias.
    fn reference_grid(
        &self,
        db: &PerfDatabase,
        kernel_tag: &str,
        num_gemms: u64,
        policy: TransferPolicy,
        candidates: &[MoeReferenceCandidate],
    ) -> Result<Option<Arc<UtilGrid>>, AicError> {
        if candidates.is_empty() {
            return Ok(None);
        }
        let query_features = [
            self.topk as f64,
            self.num_experts as f64,
            self.hidden_size as f64,
            self.inter_size as f64,
        ];
        let feature_rows: Vec<Vec<f64>> = candidates
            .iter()
            .map(|c| {
                vec![
                    c.slice.topk as f64,
                    c.slice.num_experts as f64,
                    c.slice.hidden_size as f64,
                    c.slice.inter_size as f64,
                ]
            })
            .collect();
        let chosen = &candidates[util_empirical::nearest_candidate_index(&query_features, &feature_rows)
            .expect("candidate list is non-empty")];

        let spec = &db.system_spec;
        let key = format!(
            "moe_{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:policy={}:ref={}:{}x{}x{}x{}",
            chosen.provenance,
            self.quant_mode.name(),
            kernel_tag,
            self.topk,
            self.num_experts,
            self.hidden_size,
            self.inter_size,
            self.moe_tp_size,
            self.moe_ep_size,
            self.workload_distribution,
            num_gemms,
            policy_fingerprint(policy),
            chosen.ref_quant.name(),
            chosen.slice.topk,
            chosen.slice.num_experts,
            chosen.slice.hidden_size,
            chosen.slice.inter_size,
        );
        db.util_grids.get_or_try_build(&key, || {
            // ReferenceCandidate contract: the SOL uses THE CANDIDATE's
            // shape (not the query's) so util carries only the shared
            // kernel efficiency.
            let sol = |c: &[f64]| {
                moe_sol_latency_ms(
                    spec,
                    chosen.sol_quant,
                    num_gemms,
                    c[0].round() as u32,
                    chosen.slice.hidden_size,
                    chosen.slice.inter_size,
                    chosen.slice.topk,
                    chosen.slice.num_experts,
                    self.moe_tp_size,
                    self.moe_ep_size,
                )
            };
            let mut grid = UtilGrid::new(util_empirical::build_samples(
                chosen.slice.points.iter().map(|&(t, lat)| (vec![t as f64], lat)),
                sol,
            ));
            grid.reference_provenance = Some(chosen.provenance);
            Ok(Some(grid))
        })
    }

    /// SOL MoE latency (ms) mirroring Python `MoE._query_moe_table`'s
    /// `get_sol` closure (`operations/moe.py:297`). Passed into the perf-DB
    /// engine query as the util-hold roofline; in-grid resolutions never
    /// consult it (1-axis RAW lerp / exact hit).
    fn sol_latency_ms(&self, db: &PerfDatabase, num_tokens: u32) -> f64 {
        // `num_gemms`: 3 for gated SwiGLU (gate + up + down), 2 for
        // non-gated Relu² (up + down). Matches Python `num_gemms = 3 if
        // is_gated else 2` (`operations/moe.py:115, 239`).
        let num_gemms: u64 = if self.is_gated { 3 } else { 2 };
        moe_sol_latency_ms(
            &db.system_spec,
            self.quant_mode,
            num_gemms,
            num_tokens,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.moe_tp_size,
            self.moe_ep_size,
        )
    }
}

/// MoE roofline SOL (ms) mirroring Python `MoE._query_moe_table.get_sol`
/// (`operations/moe.py:297-325`), parameterised over the slice's shape and
/// quant so the transfer ladder can bind it to a REFERENCE candidate's shape
/// (`num_experts` folds into the min() weight term; `workload_distribution`
/// never enters the math).
#[allow(clippy::too_many_arguments)]
fn moe_sol_latency_ms(
    spec: &SystemSpec,
    quant: MoeQuantMode,
    num_gemms: u64,
    num_tokens: u32,
    hidden_size: u32,
    inter_size: u32,
    topk: u32,
    num_experts: u32,
    moe_tp_size: u32,
    moe_ep_size: u32,
) -> f64 {
    let total_tokens = num_tokens as u64 * topk as u64;
    let moe_ep = (moe_ep_size as u64).max(1);
    let moe_tp = (moe_tp_size as u64).max(1);
    let h = hidden_size as u64;
    let inter = inter_size as u64;
    let ne = num_experts as u64;

    let ops = total_tokens * h * inter * num_gemms * 2 / moe_ep / moe_tp;
    let mem_bytes_int = total_tokens / moe_ep * h * 2 // input + output
        + total_tokens / moe_ep * inter * num_gemms / moe_tp // intermediate
        + h * inter * num_gemms / moe_tp
            * std::cmp::min(ne / moe_ep, total_tokens / moe_ep);
    let mem_bytes = (mem_bytes_int as f64) * quant.mapping().memory;

    // Python uses `system_spec["gpu"]["bfloat16_tc_flops"]` directly
    // (KeyError if missing). Rust exposes it as Option; fall back to 1.0
    // to make the math identity (sol_math → ops, sol_mem dominates)
    // rather than dividing by zero. Every shipped system populates it.
    let tc_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(1.0);
    let sol_math = (ops as f64) / (tc_flops * quant.mapping().compute) * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn b200_vllm_db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "b200_sxm", "vllm", "0.19.0").expect("db loads")
    }

    fn op(attention_dp_size: u32) -> MoeOp {
        MoeOp {
            name: "moe".into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            inter_size: 2048,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 8,
            quant_mode: MoeQuantMode::Fp8Block,
            workload_distribution: "power_law_1.2".into(),
            attention_dp_size,
            is_gated: true,
            moe_backend: None,
            enable_eplb: false,
            is_context: false,
        }
    }

    fn b200_trtllm_db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "b200_sxm", "trtllm", "1.2.0rc5").expect("db loads")
    }

    /// Python `resolve_transfer_policy("conservative")`: xshape only.
    const CONSERVATIVE: TransferPolicy = TransferPolicy {
        xshape: true,
        xquant: false,
        xprofile: false,
        xop: false,
    };

    /// Qwen3-235B-A22B expert shape on b200/vllm/0.19.0 (collected for
    /// bfloat16 at tp=1/ep=8 under power_law_1.2).
    fn qwen3_op(quant: MoeQuantMode) -> MoeOp {
        MoeOp {
            name: "moe".into(),
            scale_factor: 1.0,
            hidden_size: 4096,
            inter_size: 1536,
            topk: 8,
            num_experts: 128,
            moe_tp_size: 1,
            moe_ep_size: 8,
            quant_mode: quant,
            workload_distribution: "power_law_1.2".into(),
            attention_dp_size: 1,
            is_gated: true,
            moe_backend: None,
            enable_eplb: false,
            is_context: false,
        }
    }

    fn assert_oracle(result: &PerformanceResult, expected: f64, source: Source, label: &str) {
        assert!(
            (result.latency_ms - expected).abs() < 1e-9,
            "{label}: expected {expected}, got {}",
            result.latency_ms
        );
        assert_eq!(result.source, source, "{label}: wrong source");
    }

    /// Oracle values generated from the Python reference on the same data
    /// (shared layer pinned OFF so Python reads exactly the primary parquet
    /// the Rust table loads):
    ///
    /// ```text
    /// db = perf_database.get_database_view("b200_sxm", "vllm", "0.19.0",
    ///     allow_missing_data=True, database_mode=..., transfer_policy=...,
    ///     shared_layer=False)
    /// float(MoE._query_moe_table(db, num_tokens=..., hidden_size=4096,
    ///     inter_size=1536, topk=8, num_experts=128, moe_tp_size=1,
    ///     moe_ep_size=8, quant_mode=..., workload_distribution="power_law_1.2",
    ///     database_mode=...))
    /// ```
    ///
    /// Regenerate if the shipped MoE table or the util-empirical math changes.
    #[test]
    fn moe_empirical_own_shape_matches_python_oracles() {
        let db = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Empirical, TransferPolicy::ALL);
        let op = qwen3_op(MoeQuantMode::Bfloat16);
        let r333 = op.query(&db, 333).expect("own-shape empirical t=333");
        assert_oracle(&r333, 0.19184494219320924, Source::Empirical, "own_emp_t333");
        let r96 = op.query(&db, 96).expect("own-shape empirical t=96");
        assert_oracle(&r96, 0.13852159976959227, Source::Empirical, "own_emp_t96");
        // Python capture: {"empirical"} (own-shape grid, no borrow).
        assert_eq!(db.worst_provenance(), util_empirical::ProvenanceTier::Empirical);
    }

    /// HYBRID with data present must stay on silicon (exact hit and in-range
    /// interpolation), and its interpolated value differs from the empirical
    /// reconstruction at the same point (0.19178... vs 0.19184...).
    #[test]
    fn moe_hybrid_prefers_silicon_when_covered() {
        let db = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, TransferPolicy::ALL);
        let op = qwen3_op(MoeQuantMode::Bfloat16);
        let hit = op.query(&db, 128).expect("collected token point");
        assert_oracle(&hit, 0.146451199054718, Source::Silicon, "hyb_silicon_t128");
        let interp = op.query(&db, 333).expect("in-range token interp");
        assert_oracle(&interp, 0.19178520552814007, Source::Silicon, "hyb_t333");
    }

    /// XQUANT tier: `w4a16_mxfp4_cutlass` is uncollected on b200/vllm/0.19.0
    /// but shares the (0.5, 1) profile with collected int4_wo / w4a16_mxfp4;
    /// the borrowed util curve is reconstructed with the QUERY quant's SOL.
    #[test]
    fn moe_xquant_transfer_matches_python_oracle() {
        let db = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, TransferPolicy::ALL);
        let op = qwen3_op(MoeQuantMode::W4a16Mxfp4Cutlass);
        let r = op.query(&db, 96).expect("xquant transfer");
        assert_oracle(&r, 0.329638409614563, Source::Empirical, "xquant_t96");
        // Python capture: {"xquant"} (reference grid's tier, moe.py:534).
        assert_eq!(db.worst_provenance(), util_empirical::ProvenanceTier::XQuant);
    }

    /// XPROFILE tier: `w4afp8` ((0.5, 2)) has no same-profile sibling in the
    /// table; the nearest-profile quant (fp8, distance 0.5) supplies the util
    /// curve built with ITS own SOL, rescaled by e(w4afp8)/e(fp8) = 0.15/0.40.
    #[test]
    fn moe_xprofile_transfer_matches_python_oracle() {
        let db = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, TransferPolicy::ALL);
        let op = qwen3_op(MoeQuantMode::W4afp8);
        let r = op.query(&db, 96).expect("xprofile transfer");
        assert_oracle(&r, 0.13701972961425785, Source::Empirical, "xprofile_t96");
        // Python capture: {"xprofile"} (tier-3 borrow, moe.py:569).
        assert_eq!(db.worst_provenance(), util_empirical::ProvenanceTier::XProfile);
    }

    /// XSHAPE tier: same quant (bfloat16), uncollected inter_size 1600 →
    /// nearest collected sibling (8, 128, 4096, 1536). Also reachable under
    /// the conservative (xshape-only) policy with the identical value.
    #[test]
    fn moe_xshape_transfer_matches_python_oracle() {
        let mut op = qwen3_op(MoeQuantMode::Bfloat16);
        op.inter_size = 1600;

        let db = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, TransferPolicy::ALL);
        let r = op.query(&db, 96).expect("xshape transfer");
        assert_oracle(&r, 0.14427836344943168, Source::Empirical, "xshape_t96");
        // Python capture: {"xshape"} (tier-1 borrow, moe.py:534).
        assert_eq!(db.worst_provenance(), util_empirical::ProvenanceTier::XShape);

        let conservative = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, CONSERVATIVE);
        let rc = op.query(&conservative, 96).expect("xshape under conservative policy");
        assert_oracle(&rc, 0.14427836344943168, Source::Empirical, "conservative_xshape_t96");
    }

    /// Policy gating: disabled tiers are SKIPPED, and the terminal
    /// EmpiricalNotImplemented only fires when every permitted tier found
    /// nothing — `off` blocks everything for an uncollected quant, and
    /// `conservative` (xshape only) blocks the xquant-needing case.
    #[test]
    fn moe_transfer_policy_gates_tiers() {
        let op = qwen3_op(MoeQuantMode::W4a16Mxfp4Cutlass);

        let off = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, TransferPolicy::OFF);
        let blocked = op.query(&off, 96);
        assert!(
            matches!(blocked, Err(AicError::EmpiricalNotImplemented(_))),
            "off policy must surface the typed empirical miss, got {blocked:?}"
        );

        let conservative = b200_vllm_db().with_mode(crate::common::enums::DatabaseMode::Hybrid, CONSERVATIVE);
        let blocked = op.query(&conservative, 96);
        assert!(
            matches!(blocked, Err(AicError::EmpiricalNotImplemented(_))),
            "conservative policy must not reach the xquant tier, got {blocked:?}"
        );
    }

    /// Low-latency kernel-table selection inside the EMPIRICAL path
    /// (b200/trtllm/1.2.0rc5 carries `moe_torch_flow_min_latency` rows):
    /// nvfp4 gated at t<=128 with the slice present builds the util grid
    /// from the LL table; t>128 uses the regular table at the same shape
    /// (~3x apart); an uncollected shape at t<=128 fails the LL probe and
    /// runs the xshape ladder over the REGULAR table. Oracles:
    ///
    /// ```text
    /// db = perf_database.get_database_view("b200_sxm", "trtllm", "1.2.0rc5",
    ///     allow_missing_data=True, database_mode="EMPIRICAL", shared_layer=False)
    /// float(MoE._query_moe_table(db, num_tokens=..., hidden_size=6144,
    ///     inter_size=..., topk=2, num_experts=8, moe_tp_size=32, moe_ep_size=1,
    ///     quant_mode=common.MoEQuantMode.nvfp4, workload_distribution="balanced",
    ///     database_mode=common.DatabaseMode.EMPIRICAL))
    /// ```
    #[test]
    fn moe_empirical_low_latency_table_selection_matches_python_oracles() {
        let db = b200_trtllm_db().with_mode(crate::common::enums::DatabaseMode::Empirical, TransferPolicy::ALL);
        let op = MoeOp {
            name: "moe-ll".into(),
            scale_factor: 1.0,
            hidden_size: 6144,
            inter_size: 16384,
            topk: 2,
            num_experts: 8,
            moe_tp_size: 32,
            moe_ep_size: 1,
            quant_mode: MoeQuantMode::Nvfp4,
            workload_distribution: "balanced".into(),
            attention_dp_size: 1,
            is_gated: true,
            moe_backend: None,
            enable_eplb: false,
            is_context: false,
        };
        let ll = op.query(&db, 100).expect("ll-table empirical t=100");
        assert_oracle(&ll, 0.023113779703977197, Source::Empirical, "ll_own_t100");
        let std_table = op.query(&db, 200).expect("std-table empirical t=200");
        assert_oracle(&std_table, 0.058452753259364186, Source::Empirical, "std_own_t200");

        let mut off_shape = op.clone();
        off_shape.inter_size = 17000;
        let xshape = off_shape.query(&db, 100).expect("failed ll probe -> std xshape");
        assert_oracle(&xshape, 0.05842286435922407, Source::Empirical, "nvfp4_xshape_t100");
    }

    /// XPROFILE tie-break follows FILE-ROW quant order, not sorted order:
    /// b200/vllm/0.24.0 lists `fp8_block` before `fp8` (both profile (1,2),
    /// distance 0.5 from w4afp8), so Python's stable sort borrows the
    /// `fp8_block` util curve. The shape's `nvfp4` rows (added with the vLLM
    /// 0.24 w4a8 data refresh, #1399) provide a closer XQUANT match that
    /// shadows the tie-break under `TransferPolicy::ALL`, so the test pins a
    /// policy without xquant to keep exercising the XPROFILE path. Oracles:
    ///
    /// ```text
    /// db = perf_database.get_database_view("b200_sxm", "vllm", "0.24.0",
    ///     allow_missing_data=True, database_mode="EMPIRICAL", shared_layer=False,
    ///     transfer_policy=["xshape", "xprofile", "xop"])
    /// float(MoE._query_moe_table(db, num_tokens=..., hidden_size=5120,
    ///     inter_size=8192, topk=1, num_experts=16, moe_tp_size=1,
    ///     moe_ep_size=1, quant_mode=common.MoEQuantMode.w4afp8,
    ///     workload_distribution="power_law_1.01",
    ///     database_mode=common.DatabaseMode.EMPIRICAL))
    /// ```
    ///
    /// The fp8-referenced value (the old sorted-name order) is
    /// 0.42024958928426115 at t=96 — a live ~13% divergence this pins.
    #[test]
    fn moe_xprofile_tie_break_follows_file_order() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("src/aiconfigurator_core/systems");
        let no_xquant = TransferPolicy {
            xshape: true,
            xquant: false,
            xprofile: true,
            xop: true,
        };
        let db = PerfDatabase::load(&root, "b200_sxm", "vllm", "0.24.0")
            .expect("db loads")
            .with_mode(crate::common::enums::DatabaseMode::Empirical, no_xquant);
        let op = MoeOp {
            name: "moe".into(),
            scale_factor: 1.0,
            hidden_size: 5120,
            inter_size: 8192,
            topk: 1,
            num_experts: 16,
            moe_tp_size: 1,
            moe_ep_size: 1,
            quant_mode: MoeQuantMode::W4afp8,
            workload_distribution: "power_law_1.01".into(),
            attention_dp_size: 1,
            is_gated: true,
            moe_backend: None,
            enable_eplb: false,
            is_context: false,
        };
        let r96 = op.query(&db, 96).expect("xprofile tie t=96");
        assert_oracle(&r96, 0.47411200205485027, Source::Empirical, "xprofile_tie_t96");
        let r512 = op.query(&db, 512).expect("xprofile tie t=512");
        assert_oracle(&r512, 0.8249173482259117, Source::Empirical, "xprofile_tie_t512");
    }

    /// SGLang deepep op used by the routing tests below: the h200 sglang
    /// 0.5.10 wideep context/generation MoE tables cover the DSv3 expert
    /// shape (7168, 2048, topk 8, experts 256) at tp=1/ep=8 under
    /// power_law_0.8 — while the REGULAR h200 moe table also carries
    /// fp8_block, so mis-routing to it yields a value, not an error.
    fn h200_deepep_op(is_context: bool) -> MoeOp {
        MoeOp {
            name: "moe".into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            inter_size: 2048,
            topk: 8,
            num_experts: 256,
            moe_tp_size: 1,
            moe_ep_size: 8,
            quant_mode: MoeQuantMode::Fp8Block,
            workload_distribution: "power_law_0.8".into(),
            attention_dp_size: 1,
            is_gated: true,
            moe_backend: Some("deepep_moe".into()),
            enable_eplb: false,
            is_context,
        }
    }

    fn h200_sglang_db(mode: crate::common::enums::DatabaseMode) -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&root, "h200_sxm", "sglang", "0.5.10")
            .expect("db loads")
            .with_mode(mode, TransferPolicy::ALL)
    }

    /// EMPIRICAL under SGLang deepep calibrates from the WIDEEP
    /// context/generation MoE tables, not the regular one — Python
    /// `_moe_table()` (`operations/moe.py:364-397`) routes the util grid by
    /// `moe_backend == "deepep_moe"` + `is_context`. Oracles:
    ///
    /// ```text
    /// db = perf_database.get_database_view("h200_sxm", "sglang", "0.5.10",
    ///     allow_missing_data=True, database_mode="EMPIRICAL", shared_layer=False)
    /// float(MoE._query_moe_table(db, num_tokens=..., hidden_size=7168,
    ///     inter_size=2048, topk=8, num_experts=256, moe_tp_size=1,
    ///     moe_ep_size=8, quant_mode=common.MoEQuantMode.fp8_block,
    ///     workload_distribution="power_law_0.8", is_context=...,
    ///     moe_backend="deepep_moe", database_mode=common.DatabaseMode.EMPIRICAL))
    /// ```
    ///
    /// ctx t=200000 sits beyond the collected range (max 131072) so the
    /// util-hold anchors on the MoE roofline through the wideep grid.
    #[test]
    fn moe_empirical_deepep_routes_to_wideep_tables() {
        let db = h200_sglang_db(crate::common::enums::DatabaseMode::Empirical);
        let ctx = h200_deepep_op(true);
        let r = ctx.query(&db, 300).expect("deepep ctx t=300");
        assert_oracle(&r, 0.6444098182832491, Source::Empirical, "deepep_ctx_t300");
        let r = ctx.query(&db, 200000).expect("deepep ctx t=200000");
        assert_oracle(&r, 21.3889914448373, Source::Empirical, "deepep_ctx_t200000");
        let gen = h200_deepep_op(false);
        let r = gen.query(&db, 100).expect("deepep gen t=100");
        assert_oracle(&r, 0.34094198365735795, Source::Empirical, "deepep_gen_t100");
        let r = gen.query(&db, 3000).expect("deepep gen t=3000");
        assert_oracle(&r, 0.4024570594575049, Source::Empirical, "deepep_gen_t3000");
        // Python capture: {"empirical"} (own-slice wideep calibration).
        assert_eq!(db.worst_provenance(), util_empirical::ProvenanceTier::Empirical);
    }

    /// The EPLB 0.8 prefill token correction applies INSIDE the silicon
    /// path only (Python moe.py:684); the empirical estimate uses RAW
    /// tokens (moe.py:637-647, 803-813). Python oracles (same call shape as
    /// `moe_empirical_deepep_routes_to_wideep_tables`, `enable_eplb=True`):
    /// SILICON eplb-on t=160 = SILICON eplb-off t=128 = 0.6220973747117179
    /// (int(160*0.8) = 128, a collected point); EMPIRICAL eplb-on t=300 =
    /// eplb-off t=300 = 0.6444098182832491.
    #[test]
    fn moe_eplb_correction_scoped_to_silicon_only() {
        let mut eplb_op = h200_deepep_op(true);
        eplb_op.enable_eplb = true;

        let silicon = h200_sglang_db(crate::common::enums::DatabaseMode::Hybrid);
        let corrected = eplb_op.query(&silicon, 160).expect("eplb-on silicon t=160");
        assert_oracle(&corrected, 0.6220973747117179, Source::Silicon, "eplb_sil_t160");
        let baseline = h200_deepep_op(true).query(&silicon, 128).expect("eplb-off silicon t=128");
        assert!(
            (corrected.latency_ms - baseline.latency_ms).abs() < 1e-12,
            "silicon eplb-on(160) ({}) must equal eplb-off(128) ({})",
            corrected.latency_ms,
            baseline.latency_ms
        );

        let empirical = h200_sglang_db(crate::common::enums::DatabaseMode::Empirical);
        let raw = eplb_op.query(&empirical, 300).expect("eplb-on empirical t=300");
        assert_oracle(&raw, 0.6444098182832491, Source::Empirical, "eplb_emp_t300");
    }

    /// With attention-dp, all dp ranks' tokens funnel into the shared expert
    /// pool: query(dp=4, t) must equal query(dp=1, 4t). Dropping the
    /// multiplier under-predicted MoE latency ~4.7x on dp=8 DeepSeek configs
    /// (python/rust engine-step divergence, per-op accounted at 81.1 vs
    /// 378.8 ms on the h200 DSV3 tp1/dp8/moe_tp8 case).
    #[test]
    fn moe_query_scales_tokens_by_attention_dp() {
        let db = b200_vllm_db();
        let with_dp = op(4).query(&db, 1000).expect("dp=4 query");
        let equivalent = op(1).query(&db, 4000).expect("dp=1 query");
        assert!(
            (with_dp.latency_ms - equivalent.latency_ms).abs() < 1e-12,
            "dp=4 @ 1000 tokens ({}) must equal dp=1 @ 4000 tokens ({})",
            with_dp.latency_ms,
            equivalent.latency_ms
        );
        assert!(with_dp.latency_ms > op(1).query(&db, 1000).unwrap().latency_ms);
    }
}
