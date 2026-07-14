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
//! Scope: the SGLang `moe_backend == "deepep_moe"` branch of Python's
//! `_moe_table` does not exist here — the Rust engine routes WideEP MoE
//! through `WideEpMoeOp` (`operators/wideep_moe.rs`), so this op only ever
//! addresses the regular / low-latency tables.
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
/// mirrors `_xprofile_moe_quants`, `operations/moe.py:105-117`). NOTE:
/// Python breaks distance ties by table insertion (file row) order; here
/// the input arrives in the accessor's sorted-name order.
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
        let (latency, source) = match db.database_mode {
            DatabaseMode::Empirical => (self.empirical_latency(db, num_tokens)?, Source::Empirical),
            DatabaseMode::Hybrid => match self.silicon_latency(db, num_tokens) {
                Ok(latency) => (latency, Source::Silicon),
                Err(err) if err.is_missing_perf_data() => {
                    (self.empirical_latency(db, num_tokens)?, Source::Empirical)
                }
                Err(err) => return Err(err),
            },
            _ => (self.silicon_latency(db, num_tokens)?, Source::Silicon),
        };
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// SILICON table resolution (the pre-empirical behaviour, unchanged).
    fn silicon_latency(&self, db: &PerfDatabase, num_tokens: u32) -> Result<f64, AicError> {
        // The roofline SOL the perf-DB engine anchors its beyond-range
        // util-hold on (Python `_resolve_tokens` passes the same closure).
        // Coordinates arriving from the engine are always integral (table
        // keys / the u32 query), so rounding to u32 keeps the integer
        // floor-division parity with Python's `get_sol`. This replaces the
        // deleted op-level SOL-anchored overflow estimator (the engine's
        // `k_tail=1` util-hold handles beyond-range queries).
        let sol = |t: f64| self.sol_latency_ms(db, t.round() as u32);

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
                return Ok(ll);
            }
        }
        db.moe.query(
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
        )
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

        // Kernel-table selection mirrors get_silicon's (`_moe_table`,
        // `operations/moe.py:355-386`): nvfp4 + small tokens + gated probes
        // the low-latency table for the FULL slice and falls back to the
        // default table on a shape miss. Building util from the wrong table
        // would over-estimate by the ~3x kernel gap. The `kernel_tag` folds
        // the choice into every grid cache key so a low-latency grid can't
        // be served to a regular query (or vice versa) at the same shape.
        let kernel = if num_tokens <= 128
            && quant == MoeQuantMode::Nvfp4
            && self.is_gated
            && db.moe.low_latency_available()?
        {
            match self.slice_points(db, MoeKernel::LowLatency) {
                Ok(_) => MoeKernel::LowLatency,
                Err(err) if err.is_missing_perf_data() => MoeKernel::Standard,
                Err(err) => return Err(err),
            }
        } else {
            MoeKernel::Standard
        };
        let kernel_tag = match kernel {
            MoeKernel::LowLatency => "ll",
            MoeKernel::Standard => "std",
        };

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
            match self.slice_points(db, kernel) {
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
                self.collect_candidates(db, kernel, quant, quant, "xshape", &mut candidates)?;
            }
            if candidates.is_empty() && policy.contains(TransferKind::XQuant) {
                let qp = quant.mapping();
                for name in db.moe.available_quants(kernel)? {
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
                    self.collect_candidates(db, kernel, sibling, quant, "xquant", &mut candidates)?;
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
                let table_quants: Vec<MoeQuantMode> = db
                    .moe
                    .available_quants(kernel)?
                    .iter()
                    .filter_map(|name| moe_quant_from_name(name))
                    .collect();
                for ref_quant in xprofile_moe_quants(quant, &table_quants) {
                    let mut candidates: Vec<MoeReferenceCandidate> = Vec::new();
                    self.collect_candidates(
                        db,
                        kernel,
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
        Ok(latency)
    }

    /// This op's own-slice token curve on the chosen kernel table.
    fn slice_points(&self, db: &PerfDatabase, kernel: MoeKernel) -> Result<Vec<(u32, f64)>, AicError> {
        db.moe.slice_points(
            kernel,
            self.quant_mode.name(),
            &self.workload_distribution,
            self.topk,
            self.num_experts,
            self.hidden_size,
            self.inter_size,
            self.moe_tp_size,
            self.moe_ep_size,
        )
    }

    /// Enumerate `source_quant`'s collected sibling slices (same table,
    /// same wl-after-fallback / moe_tp / moe_ep) as ladder candidates.
    /// Mirrors `_collect` (`operations/moe.py:454-486`); a typed data miss
    /// (table failed to load) yields no candidates, exactly like Python's
    /// `grid_from_reference` catching the raise from `_collect`.
    fn collect_candidates(
        &self,
        db: &PerfDatabase,
        kernel: MoeKernel,
        source_quant: MoeQuantMode,
        sol_quant: MoeQuantMode,
        provenance: &'static str,
        out: &mut Vec<MoeReferenceCandidate>,
    ) -> Result<(), AicError> {
        let slices = match db.moe.sibling_slices(
            kernel,
            source_quant.name(),
            &self.workload_distribution,
            self.moe_tp_size,
            self.moe_ep_size,
        ) {
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
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("src/aiconfigurator/systems");
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
        }
    }

    fn b200_trtllm_db() -> PerfDatabase {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("src/aiconfigurator/systems");
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
