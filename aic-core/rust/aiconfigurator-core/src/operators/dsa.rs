// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (Dynamic Sparse Attention) module operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.dsa.ContextDSAModule` /
//! `GenerationDSAModule`. The context lookup evaluates at `isl` (the
//! new-token count) on the raw 4-axis `[heads][prefix][seq][batch]` grid via
//! the perf_interp v2 engine (see `perf_database::dsa::query_context`).
//!
//! `index_topk` is the top-k boundary (per-architecture; 2048 for both
//! DeepSeek-V3.2 and GLM-5). It is plumbed from the Python op-spec emitter.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use crate::common::enums::{DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::communication::NcclOp;
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::dsa::{
    bs_slice, dsa_context_sol_ms, dsa_dims, dsa_generation_sol_ms, dsa_sparse_file_prefix,
    lookup_2d, DsaHeadGrid, DsaKey, DsaSparseTables,
};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DsaModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub num_heads: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    pub architecture: String,
    /// Top-k boundary for the sparse-attention regime split. Sourced from
    /// `DSA_MODEL_DIMS[architecture]["index_topk"]` on the Python side.
    pub index_topk: u32,
    /// Context-parallel size (Python `ContextDSAModule._cp_size`). When > 1
    /// the context query runs the GLM-5/DSA sparse-CP prefill composition
    /// (Python `_query_cp`); 1 (the default) keeps the plain 4-axis lookup.
    /// NOT yet emitted by the Python opspec — `engine.py::_reject_cp` still
    /// guards CP specs; the guard and this field's emission flip atomically
    /// once BOTH the dsa and dsv4 Rust CP paths land.
    #[serde(default = "default_cp_size")]
    pub cp_size: u32,
    /// GLM-5.2 shared-index amortization weight (Python `_full_frac`): the
    /// exact fraction of indexer-computing layers. Per-layer cost is
    /// `full_frac*full + (1-full_frac)*skip` using the directly-collected
    /// skip-indexer table. 1.0 (DeepSeek-V3.2 / GLM-5, and pre-field opspecs)
    /// keeps the pure-full path — the skip table is never touched.
    #[serde(default = "default_full_frac")]
    pub full_frac: f64,
}

fn default_cp_size() -> u32 {
    1
}

fn default_full_frac() -> f64 {
    1.0
}

impl DsaModuleOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        num_heads: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        architecture: impl Into<String>,
        index_topk: u32,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            num_heads,
            kv_cache_dtype,
            fmha_quant_mode,
            gemm_quant_mode,
            architecture: architecture.into(),
            index_topk,
            cp_size: 1,
            full_frac: 1.0,
        }
    }

    pub fn query_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        let w = self.full_frac;
        // CP (round-robin sequence split) prefill takes the sparse-delta
        // composition path (Python `ContextDSAModule.query` -> `_query_cp`
        // when `_cp_size > 1`). GLM-5.2 amortizes full/skip on the CP path
        // too (both carry the same scale_factor, so the weighted sum of the
        // already-scaled results is exact — Python `_amortize`).
        if self.cp_size > 1 {
            let full = self.query_context_cp(db, batch_size, isl, prefix, false)?;
            if w >= 1.0 {
                return Ok(full);
            }
            let skip = self.query_context_cp(db, batch_size, isl, prefix, true)?;
            return Ok(PerformanceResult::new(
                w * full.latency_ms + (1.0 - w) * skip.latency_ms,
                full.source,
            ));
        }
        // Query at `isl` (new-token count) for the exact `prefix` slice — NOT
        // `isl + prefix`. The perf-DB layer resolves one 4-axis RAW grid via
        // the perf_interp v2 engine; there is no multiplicative prefix
        // correction (it had no Python counterpart and under-counted context
        // latency ~75%). `dsa_backend="trtllm"` mirrors Python's non-CP
        // default (`_query_context_dsa_module_table(dsa_backend="trtllm")`).
        let q = |skip_indexer: bool| {
            query_context_table(db, self, batch_size, isl, prefix, "trtllm", skip_indexer)
        };
        let (full, full_source) = q(false)?;
        let (latency, source) = if w >= 1.0 {
            (full, full_source)
        } else {
            // GLM-5.2 shared-index amortization (Python ContextDSAModule.query).
            let (skip, skip_source) = q(true)?;
            (w * full + (1.0 - w) * skip, full_source.combine(skip_source))
        };
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// Context-Parallel (CP) prefill — GLM-5/DSA sparse composition.
    ///
    /// Wires the real data dependencies and delegates to [`Self::query_cp_with`]
    /// (the verbatim mirror of Python `ContextDSAModule._query_cp`); the
    /// caller passes `skip_indexer` through, so GLM-5.2's CP + shared-index
    /// amortization can query both the full and skip-indexer slices:
    /// - base = the existing 4-axis engine query at `(b, per_card, prefix)`
    ///   with `dsa_backend="flashmla_kv"`, exactly like Python's `_query_cp`
    ///   base query;
    /// - AG = `db.query_nccl(half, cp, "all_gather", elems)` via [`NcclOp`],
    ///   which mirrors Python's fan-out cap + multi-node bandwidth scaling.
    fn query_context_cp(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
        skip_indexer: bool,
    ) -> Result<PerformanceResult, AicError> {
        let sparse = db.dsa.load_cp_sparse(&self.architecture, self.num_heads)?;
        let mut base = |per_card: u32| {
            // Python `_query_cp` queries the CP base through the full
            // `query_context_dsa_module` dispatch (no explicit database_mode
            // => the database default), on the flashmla_kv slice (the kernel
            // used under CP); `float(...)` drops the source.
            query_context_table(db, self, batch_size, per_card, prefix, "flashmla_kv", skip_indexer)
                .map(|(latency, _)| latency)
        };
        let mut ag = |elems: u64| {
            NcclOp::new(
                format!("{}_cp_all_gather", self.name),
                1.0,
                elems as f64,
                self.cp_size,
                "all_gather",
            )
            .query(db, 1)
            .map(|r| r.latency_ms)
        };
        self.query_cp_with(&sparse, batch_size, isl, prefix, skip_indexer, &mut base, &mut ag)
    }

    /// CP (round-robin split) per-layer DSA composition. Verbatim mirror of
    /// Python `ContextDSAModule._query_cp` (2026-06-11 strategy):
    ///
    /// ```text
    /// result = dsa(isl/cp, prefix)
    ///        + [mqa(isl, prefix)/cp       - mqa(isl/cp, prefix)]
    ///        + [topk_last(isl, prefix)/cp - topk_flat(isl/cp, prefix)]
    ///        + AG_KV + AG_LSE
    /// ```
    ///
    /// `base(per_card)` supplies the per-card monolithic dsa_module latency;
    /// `ag(elems)` the all-gather latency for an element volume (bf16). Both
    /// are injected so the composition is unit-testable against the same
    /// synthetic inputs as the Python test
    /// (`tests/unit/sdk/test_cp_dsa_modeling.py::test_query_cp_composition`).
    fn query_cp_with(
        &self,
        sparse: &DsaSparseTables,
        b: u32,
        isl: u32,
        prefix: u32,
        skip_indexer: bool,
        base: &mut dyn FnMut(u32) -> Result<f64, AicError>,
        ag: &mut dyn FnMut(u64) -> Result<f64, AicError>,
    ) -> Result<PerformanceResult, AicError> {
        let cp = self.cp_size;
        let per_card = isl.div_ceil(cp).max(1); // ceil: critical path = busiest CP rank
        let file_prefix = dsa_sparse_file_prefix(&self.architecture);
        // Fail fast: CP DSA modeling REQUIRES the sparse mqa/topk tables for
        // the mqa/topk_last deltas. `lookup_2d` clamps isl + interpolates
        // step, so an empty grid below means the table is absent entirely
        // (parquet not collected) — degrading silently to dsa_base would hide
        // that. Message shape mirrors Python's fail-loud contract.
        // skip_indexer layers carry NO indexer -> no mqa/topk deltas needed,
        // so don't require the sparse tables for them (Python dsa.py:835-837).
        let missing: Vec<&str> = if skip_indexer {
            Vec::new()
        } else {
            [
                ("mqa", &sparse.mqa),
                ("topk_last", &sparse.topk_last),
                ("topk_flat", &sparse.topk_flat),
            ]
            .into_iter()
            .filter(|(_, grid)| grid.is_empty())
            .map(|(name, _)| name)
            .collect()
        };
        if !missing.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "DSA CP modeling needs sparse tables ['{}'] for {} (num_heads={}); \
                 collect {file_prefix}_mqa_logits/{file_prefix}_topk first.",
                missing.join("', '"),
                self.architecture,
                self.num_heads
            )));
        }
        // Base: per-card monolithic dsa_module at (per_card, prefix).
        let dsa_base = base(per_card)?;
        // Look the sparse sub-kernels up at the REAL batch b (the bs slice
        // carries the measured bs=b latency), so the delta matches dsa_base
        // (queried at b) WITHOUT an external x b linearity assumption.
        let mqa_tab = bs_slice(&sparse.mqa, b);
        let tl_tab = bs_slice(&sparse.topk_last, b);
        let tf_tab = bs_slice(&sparse.topk_flat, b);
        let empty = std::collections::BTreeMap::new();
        let mqa_tab = mqa_tab.unwrap_or(&empty);
        let tl_tab = tl_tab.unwrap_or(&empty);
        let tf_tab = tf_tab.unwrap_or(&empty);
        let mqa_full = lookup_2d(mqa_tab, isl, prefix)?;
        let mqa_perc = lookup_2d(mqa_tab, per_card, prefix)?;
        let tl_full = lookup_2d(tl_tab, isl, prefix)?;
        let tf_perc = lookup_2d(tf_tab, per_card, prefix)?;
        let mut latency = dsa_base;
        // skip layers reuse a sibling's topk index: no per-layer mqa/topk, so
        // no full/cp deltas — just the per-card skip base + the attention
        // all-gathers (Python dsa.py:871-876).
        if !skip_indexer {
            if let (Some(mqa_full), Some(mqa_perc), Some(tl_full), Some(tf_perc)) =
                (mqa_full, mqa_perc, tl_full, tf_perc)
            {
                let delta_mqa = mqa_full / f64::from(cp) - mqa_perc;
                let delta_topk = tl_full / f64::from(cp) - tf_perc;
                latency += delta_mqa + delta_topk;
            }
        }
        // CP attention all-gathers, per current-chunk tokens (isl, not
        // isl+prefix; prefix KV is already replicated), bf16 (see the Python
        // comment block for the sglang instrumentation provenance):
        //   ag_kv  = DSA indexer key      -> b * isl * index_head_dim  (=128)
        //   ag_lse = compressed KV latent -> b * isl * (kv_lora + rope) (=576)
        // (The hidden_states AG/RS is the MoE token dispatch, modeled by the
        // MoE dispatch ops, not here.)
        let dims = dsa_dims(&self.architecture);
        let tokens = u64::from(b) * u64::from(isl);
        // A skip-indexer (reuse) layer never runs the per-layer indexer, so it
        // does not all-gather the DSA indexer key — only the MLA
        // compressed-KV/LSE gather remains (Python dsa.py:895-902).
        let ag_kv = if skip_indexer {
            0.0
        } else {
            ag(tokens * dims.index_head_dim as u64)?
        };
        let ag_lse = ag(tokens * (dims.kv_lora_rank + dims.qk_rope_head_dim) as u64)?;
        latency += ag_kv + ag_lse;
        Ok(PerformanceResult::new(latency, Source::Estimated).scaled(self.scale_factor))
    }

    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let w = self.full_frac;
        // `dsa_backend="trtllm"` mirrors Python's generation default
        // (`_query_generation_dsa_module_table(dsa_backend="trtllm")`).
        let q = |skip_indexer: bool| {
            query_generation_table(db, self, batch_size, s, "trtllm", skip_indexer)
        };
        let (full, full_source) = q(false)?;
        let (latency, source) = if w >= 1.0 {
            (full, full_source)
        } else {
            // GLM-5.2 shared-index amortization (decode side).
            let (skip, skip_source) = q(true)?;
            (w * full + (1.0 - w) * skip, full_source.combine(skip_source))
        };
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Database-mode dispatch, mirroring the Python `_query_*_dsa_module_table`
// classmethods (`operations/dsa.py`). Python does NOT use the shared
// `_query_silicon_or_hybrid` helper here: it wraps the silicon lookup in an
// explicit try/except over `(PerfDataNotAvailableError,
// InterpolationDataNotAvailableError)` and only falls to `get_empirical` when
// `database_mode == HYBRID` — in Rust that catch set is exactly
// `err.is_missing_perf_data()` (which deliberately excludes
// `EmpiricalNotImplemented`). EMPIRICAL always estimates; the SOL diagnostic
// modes never reach the compiled engine.
// ---------------------------------------------------------------------------

/// Context DSA module latency for the op's slice under the database's mode.
fn query_context_table(
    db: &PerfDatabase,
    op: &DsaModuleOp,
    b: u32,
    isl: u32,
    prefix: u32,
    dsa_backend: &str,
    skip_indexer: bool,
) -> Result<(f64, Source), AicError> {
    let silicon = || {
        db.dsa.query_context(
            &db.system_spec,
            b,
            isl,
            op.num_heads,
            op.kv_cache_dtype,
            op.fmha_quant_mode,
            op.gemm_quant_mode,
            &op.architecture,
            prefix,
            op.index_topk,
            dsa_backend,
            skip_indexer,
        )
    };
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            context_empirical(db, op, b, isl, prefix, dsa_backend, skip_indexer)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match silicon() {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => Ok((
                context_empirical(db, op, b, isl, prefix, dsa_backend, skip_indexer)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok((silicon()?, Source::Silicon)),
    }
}

/// Generation DSA module latency for the op's slice under the database's mode.
fn query_generation_table(
    db: &PerfDatabase,
    op: &DsaModuleOp,
    b: u32,
    s: u32,
    dsa_backend: &str,
    skip_indexer: bool,
) -> Result<(f64, Source), AicError> {
    let silicon = || {
        db.dsa.query_generation(
            &db.system_spec,
            b,
            s,
            op.num_heads,
            op.kv_cache_dtype,
            op.fmha_quant_mode,
            op.gemm_quant_mode,
            &op.architecture,
            dsa_backend,
            skip_indexer,
        )
    };
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            generation_empirical(db, op, b, s, dsa_backend, skip_indexer)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match silicon() {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => Ok((
                generation_empirical(db, op, b, s, dsa_backend, skip_indexer)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok((silicon()?, Source::Silicon)),
    }
}

// ---------------------------------------------------------------------------
// Util-space empirical estimation (`latency = SOL(query) / util`), mirroring
// the `get_empirical` closures of `operations/dsa.py` branch for branch.
// ---------------------------------------------------------------------------

/// Sample-coordinate → SOL-argument mapping for the selected context
/// calibration variant (the Python per-branch `_sol(c)` closures).
#[derive(Clone, Copy)]
enum CtxSolShape {
    /// Samples `(prefix, s, b)` on the exact head slice:
    /// `get_sol(c[2], c[1], c[0], num_heads, ...)`.
    PrefixSeqBatch,
    /// Samples `(num_heads, prefix, s, b)` across heads:
    /// `get_sol(c[3], c[2], c[1], c[0], ...)`.
    HeadPrefixSeqBatch,
    /// Prefix=0-anchored samples `(s, b)` on the exact head slice:
    /// `get_sol(c[1], c[0], 0, num_heads, ...)`.
    SeqBatchP0,
    /// Prefix=0-anchored samples `(num_heads, s, b)` across heads:
    /// `get_sol(c[2], c[1], 0, c[0], ...)`.
    HeadSeqBatchP0,
}

/// Calibration data feeding the selected context variant's util grid.
enum CtxCalibration<'a> {
    /// `[prefix][s][b]` — one head's full prefix-carrying sub-grid.
    HeadPrefix(&'a BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>),
    /// `[num_heads][prefix][s][b]` — the whole backend-selected slice.
    AllHeads(&'a DsaHeadGrid),
    /// `[s][b]` — one head's prefix=0 anchor sub-grid.
    HeadP0(&'a BTreeMap<u32, BTreeMap<u32, f64>>),
    /// `[num_heads] -> [s][b]` at prefix=0, heads without a 0 anchor dropped
    /// (the Python dict comprehension).
    AllHeadsP0(&'a DsaHeadGrid),
    /// Python `calibration_data is None`: `slice_fn` raises the typed
    /// coverage miss, so the grid is `None` and `estimate` reports the gap.
    Missing,
}

/// `SOL(query)/util` for the context DSA module. Mirrors Python
/// `ContextDSAModule._query_context_dsa_module_table::get_empirical`
/// (`skip_indexer=False`): detects the prefix axis, keeps the query on its
/// exact measured head slice when present, then selects the calibration
/// variant keyed by (has_prefix, exact_head, interp_prefix). The prefix=0
/// anchor and boundary-freeze variants compute util at a DIFFERENT anchor
/// coordinate than the query (full_s = s + prefix at prefix=0), while the
/// prefix effect stays in the true query SOL.
fn context_empirical(
    db: &PerfDatabase,
    op: &DsaModuleOp,
    b: u32,
    s: u32,
    prefix: u32,
    dsa_backend: &str,
    skip_indexer: bool,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let dims = dsa_dims(&op.architecture);
    let topk = op.index_topk as i64;
    let (kv, fmha, gemm) = (op.kv_cache_dtype, op.fmha_quant_mode, op.gemm_quant_mode);
    // Python's inner `get_sol(b, s, prefix, num_heads, kv, fmha)` closing
    // over gemm_quant_mode / index_topk / dims.
    let sol = |b: f64, s: f64, prefix: f64, num_heads: f64| {
        dsa_context_sol_ms(
            spec,
            dims,
            topk,
            kv,
            fmha,
            gemm,
            b as i64,
            s as i64,
            prefix as i64,
            num_heads as i64,
            skip_indexer,
        )
    };
    let sol_time = sol(b as f64, s as f64, prefix as f64, op.num_heads as f64);

    // Raw calibration slice; a typed miss mirrors Python's
    // `except PerfDataNotAvailableError: slc = None`.
    let key = DsaKey {
        architecture: op.architecture.clone(),
        fmha_quant: fmha.name().to_string(),
        kv_quant: kv.name().to_string(),
        gemm_quant: gemm.name().to_string(),
    };
    let slice = match db.dsa.context_raw_slice(&key, dsa_backend, skip_indexer) {
        Ok(slice) => Some(slice),
        Err(err) if err.is_missing_perf_data() => None,
        Err(err) => return Err(err),
    };
    // Prefix-axis detection (`_dsa_module_has_prefix_axis`): the Rust loader
    // always materialises the step level (legacy CSVs load as step=0), so any
    // loaded slice carries the prefix axis; with the data unavailable Python
    // falls back to the prior per-arch heuristic.
    let has_prefix = match slice {
        Some(_) => true,
        None => op.architecture == "GlmMoeDsaForCausalLM",
    };

    // num_heads identifies a TP/model shape: stay on the exact measured head
    // slice whenever it exists (`exact_head = isinstance(head_data, dict) and
    // bool(head_data)`), cross-head fallback only when absent.
    let head_grid = slice
        .and_then(|slc| slc.get(&op.num_heads))
        .filter(|head_grid| !head_grid.is_empty());
    let exact_head = head_grid.is_some();

    // Collected prefix values: the exact head's keys, else the sorted union
    // across heads (BTreeMap/BTreeSet iteration == Python `sorted`).
    let prefix_keys: Vec<u32> = if let Some(head_grid) = head_grid {
        head_grid.keys().copied().collect()
    } else if let Some(slc) = slice {
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        for by_prefix in slc.values() {
            seen.extend(by_prefix.keys().copied());
        }
        seen.into_iter().collect()
    } else {
        Vec::new()
    };
    // Genuine prefix interpolation needs >= 2 collected prefix points
    // bracketing the query; otherwise anchor util at the prefix=0 slice at
    // full_s = s + prefix (regime-matched) so the indexer on/off boundary and
    // small-s overhead floor stay in the (true) SOL.
    let interp_prefix = prefix_keys.len() >= 2
        && prefix_keys[0] <= prefix
        && prefix <= *prefix_keys.last().expect("non-empty prefix keys");

    let (heads_f, prefix_f, s_f, b_f) = (
        op.num_heads as f64,
        prefix as f64,
        s as f64,
        b as f64,
    );
    let (tag, depth, query, shape, calibration, head_scoped): (
        &'static str,
        usize,
        Vec<f64>,
        CtxSolShape,
        CtxCalibration,
        bool,
    ) = if has_prefix && interp_prefix {
        // Genuine measured prefix axis. Samples are (prefix, s, b) on an
        // exact head, otherwise (num_heads, prefix, s, b).
        if exact_head {
            (
                "ctx_dsa_exact_head",
                3,
                vec![prefix_f, s_f, b_f],
                CtxSolShape::PrefixSeqBatch,
                CtxCalibration::HeadPrefix(head_grid.expect("exact head slice")),
                true,
            )
        } else {
            (
                "ctx_dsa",
                4,
                vec![heads_f, prefix_f, s_f, b_f],
                CtxSolShape::HeadPrefixSeqBatch,
                CtxCalibration::AllHeads(slice.expect("populated slice")),
                false,
            )
        }
    } else if has_prefix && prefix_keys.contains(&0) {
        // Degenerate/out-of-range prefix axis: anchor utilization at prefix=0
        // and full_s = s + prefix so indexer/top-k regime changes remain in
        // the true query SOL.
        if exact_head {
            (
                "ctx_dsa_p0anchor_exact_head",
                2,
                vec![s_f + prefix_f, b_f],
                CtxSolShape::SeqBatchP0,
                CtxCalibration::HeadP0(&head_grid.expect("exact head slice")[&0]),
                true,
            )
        } else {
            (
                "ctx_dsa_p0anchor",
                3,
                vec![heads_f, s_f + prefix_f, b_f],
                CtxSolShape::HeadSeqBatchP0,
                CtxCalibration::AllHeadsP0(slice.expect("populated slice")),
                false,
            )
        }
    } else if has_prefix {
        // No prefix=0 anchor exists. Preserve coverage by freezing the
        // generic raw-grid utilization at the nearest measured prefix.
        if exact_head {
            (
                "ctx_dsa_prefix_boundary_exact_head",
                3,
                vec![prefix_f, s_f, b_f],
                CtxSolShape::PrefixSeqBatch,
                CtxCalibration::HeadPrefix(head_grid.expect("exact head slice")),
                true,
            )
        } else {
            (
                "ctx_dsa_prefix_boundary",
                4,
                vec![heads_f, prefix_f, s_f, b_f],
                CtxSolShape::HeadPrefixSeqBatch,
                slice.map_or(CtxCalibration::Missing, CtxCalibration::AllHeads),
                false,
            )
        }
    } else {
        // Legacy [heads][s][b] tables. Unreachable with data present in Rust
        // (the loader always materialises the step level), so this only fires
        // for the missing-slice non-GLM heuristic (grid stays None); the tags
        // and query coordinates still mirror Python for the estimate() error.
        if exact_head {
            (
                "ctx_dsa_legacy_exact_head",
                2,
                vec![s_f + prefix_f, b_f],
                CtxSolShape::SeqBatchP0,
                CtxCalibration::Missing,
                true,
            )
        } else {
            (
                "ctx_dsa_legacy",
                3,
                vec![heads_f, s_f + prefix_f, b_f],
                CtxSolShape::HeadSeqBatchP0,
                CtxCalibration::Missing,
                false,
            )
        }
    };

    // Python's per-branch `_sol(c)` closure over the selected variant.
    let sample_sol = |c: &[f64]| match shape {
        CtxSolShape::PrefixSeqBatch => sol(c[2], c[1], c[0], heads_f),
        CtxSolShape::HeadPrefixSeqBatch => sol(c[3], c[2], c[1], c[0]),
        CtxSolShape::SeqBatchP0 => sol(c[1], c[0], 0.0, heads_f),
        CtxSolShape::HeadSeqBatchP0 => sol(c[2], c[1], 0.0, c[0]),
    };

    // Python keys the grid on (tag, system, backend, version, quants, arch,
    // dsa_backend, depth) + id(node); the cache here is per-database, and the
    // exact-head node identity is restored by suffixing num_heads.
    // `skip` disambiguates the GLM-5.2 reuse-layer grids from the full ones.
    let skip_tag = if skip_indexer { ":skip" } else { "" };
    let cache_key = if head_scoped {
        format!(
            "{tag}:{}:{}:{}:{}:{dsa_backend}:{depth}:h{}{skip_tag}",
            fmha.name(),
            kv.name(),
            gemm.name(),
            op.architecture,
            op.num_heads
        )
    } else {
        format!(
            "{tag}:{}:{}:{}:{}:{dsa_backend}:{depth}{skip_tag}",
            fmha.name(),
            kv.name(),
            gemm.name(),
            op.architecture
        )
    };
    let grid = db.util_grids.get_or_try_build(&cache_key, || {
        let points = match calibration {
            CtxCalibration::HeadPrefix(head_grid) => context_points_head(head_grid),
            CtxCalibration::AllHeads(slc) => context_points_all(slc),
            CtxCalibration::HeadP0(p0) => points_2d(p0),
            CtxCalibration::AllHeadsP0(slc) => context_points_all_p0(slc),
            // Typed coverage miss -> no grid (estimate() raises the
            // empirical miss), mirroring Python's slice_fn raising
            // PerfDataNotAvailableError inside grid_for.
            CtxCalibration::Missing => return Ok(None),
        };
        Ok(Some(UtilGrid::new(util_empirical::build_samples(
            points,
            sample_sol,
        ))))
    })?;
    let (latency, _) = util_empirical::estimate(sol_time, &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python dsa.py:620, estimate()'s default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

/// `SOL(query)/util` for the generation DSA module. Mirrors Python
/// `GenerationDSAModule._query_generation_dsa_module_table::get_empirical`
/// (`skip_indexer=False`): exact measured head slice when present
/// (`num_heads in data_slice`), cross-head nearest-neighbour otherwise,
/// typed empirical miss when the slice is absent.
fn generation_empirical(
    db: &PerfDatabase,
    op: &DsaModuleOp,
    b: u32,
    s: u32,
    dsa_backend: &str,
    skip_indexer: bool,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let dims = dsa_dims(&op.architecture);
    let (kv, gemm) = (op.kv_cache_dtype, op.gemm_quant_mode);
    // Python's inner `get_sol(b, s, num_heads, kv_cache_dtype)` (the
    // attention group is hardcoded bfloat16 inside).
    let sol = |b: f64, s: f64, num_heads: f64| {
        dsa_generation_sol_ms(spec, dims, kv, gemm, b as i64, s as i64, num_heads as i64)
    };
    let heads_f = op.num_heads as f64;
    let sol_time = sol(b as f64, s as f64, heads_f);

    // NOTE: Python's generation slice is keyed (kv, gemm, arch) only — no
    // mla_dtype axis. The Rust table retains `mla_dtype` in `DsaKey` (see
    // `query_generation`); collected generation files carry uniformly
    // `bfloat16`, and the empirical slice follows the silicon keying so
    // HYBRID's silicon and empirical stages always agree on the slice.
    let key = DsaKey {
        architecture: op.architecture.clone(),
        fmha_quant: op.fmha_quant_mode.name().to_string(),
        kv_quant: kv.name().to_string(),
        gemm_quant: gemm.name().to_string(),
    };
    let slice = match db.dsa.generation_raw_slice(&key, dsa_backend, skip_indexer) {
        Ok(slice) => Some(slice),
        // Match `grid_for`'s best-effort contract: unavailable table data is
        // reported by `estimate` as an empirical coverage miss.
        Err(err) if err.is_missing_perf_data() => None,
        Err(err) => return Err(err),
    };

    if let Some(head_grid) = slice.and_then(|slc| slc.get(&op.num_heads)) {
        // `num_heads` is a TP/model-shape identity: stay on the exact head
        // slice whenever it exists (Python `num_heads in data_slice`).
        let skip_tag = if skip_indexer { ":skip" } else { "" };
        let cache_key = format!(
            "gen_dsa_exact_heads:{}:{}:{}:{}:{dsa_backend}:h{}{skip_tag}",
            op.fmha_quant_mode.name(),
            kv.name(),
            gemm.name(),
            op.architecture,
            op.num_heads
        );
        let grid = db.util_grids.get_or_try_build(&cache_key, || {
            Ok(Some(UtilGrid::new(util_empirical::build_samples(
                generation_points_head(head_grid),
                // c = (b, s)
                |c: &[f64]| sol(c[0], c[1], heads_f),
            ))))
        })?;
        let (latency, _) =
            util_empirical::estimate(sol_time, &[b as f64, s as f64], grid.as_deref(), 1.0)?;
        // Own-shape util fired (Python dsa.py:1296, estimate()'s default tier).
        db.note_provenance(util_empirical::ProvenanceTier::Empirical);
        Ok(latency)
    } else if let Some(slc) = slice {
        let skip_tag = if skip_indexer { ":skip" } else { "" };
        let cache_key = format!(
            "gen_dsa:{}:{}:{}:{}:{dsa_backend}{skip_tag}",
            op.fmha_quant_mode.name(),
            kv.name(),
            gemm.name(),
            op.architecture
        );
        let grid = db.util_grids.get_or_try_build(&cache_key, || {
            Ok(Some(UtilGrid::new(util_empirical::build_samples(
                generation_points_all(slc),
                // c = (num_heads, b, s)
                |c: &[f64]| sol(c[1], c[2], c[0]),
            ))))
        })?;
        let (latency, _) = util_empirical::estimate(
            sol_time,
            &[heads_f, b as f64, s as f64],
            grid.as_deref(),
            1.0,
        )?;
        // Own-shape util fired (Python dsa.py:1296, estimate()'s default tier).
        db.note_provenance(util_empirical::ProvenanceTier::Empirical);
        Ok(latency)
    } else {
        let (latency, _) = util_empirical::estimate(
            sol_time,
            &[heads_f, b as f64, s as f64],
            None,
            1.0,
        )?;
        Ok(latency)
    }
}

// ---------------------------------------------------------------------------
// Raw-grid point extraction (Python `iter_grid` over the selected variant's
// nested dict). Rust iterates the BTreeMaps in ascending key order — Python
// walks insertion (file first-occurrence) order; the k=2 IDW estimate is
// order-independent except for exact distance ties.
// ---------------------------------------------------------------------------

/// `(s, b)` points of one `[s][b]` sub-grid.
fn points_2d(grid: &BTreeMap<u32, BTreeMap<u32, f64>>) -> Vec<(Vec<f64>, f64)> {
    let mut points = Vec::new();
    for (&s, by_b) in grid {
        for (&b, &latency) in by_b {
            points.push((vec![s as f64, b as f64], latency));
        }
    }
    points
}

/// `(prefix, s, b)` points of one head's `[prefix][s][b]` sub-grid.
fn context_points_head(
    head_grid: &BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>,
) -> Vec<(Vec<f64>, f64)> {
    let mut points = Vec::new();
    for (&prefix, by_s) in head_grid {
        for (&s, by_b) in by_s {
            for (&b, &latency) in by_b {
                points.push((vec![prefix as f64, s as f64, b as f64], latency));
            }
        }
    }
    points
}

/// `(num_heads, prefix, s, b)` points of a whole `[heads][prefix][s][b]` slice.
fn context_points_all(slice: &DsaHeadGrid) -> Vec<(Vec<f64>, f64)> {
    let mut points = Vec::new();
    for (&heads, head_grid) in slice {
        for (&prefix, by_s) in head_grid {
            for (&s, by_b) in by_s {
                for (&b, &latency) in by_b {
                    points.push((
                        vec![heads as f64, prefix as f64, s as f64, b as f64],
                        latency,
                    ));
                }
            }
        }
    }
    points
}

/// `(num_heads, s, b)` points of the prefix=0 anchor across heads; heads
/// without a prefix=0 sub-grid are dropped (the Python dict comprehension).
fn context_points_all_p0(slice: &DsaHeadGrid) -> Vec<(Vec<f64>, f64)> {
    let mut points = Vec::new();
    for (&heads, head_grid) in slice {
        if let Some(by_s) = head_grid.get(&0) {
            for (&s, by_b) in by_s {
                for (&b, &latency) in by_b {
                    points.push((vec![heads as f64, s as f64, b as f64], latency));
                }
            }
        }
    }
    points
}

/// `(b, s)` points of one head's generation sub-grid. The load-time
/// (isl, step) -> seq collapse stores `[0][seq][batch]`; Python's raw view is
/// `[b][s]`, so the coordinate order flips to (batch, seq).
fn generation_points_head(
    head_grid: &BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>,
) -> Vec<(Vec<f64>, f64)> {
    let mut points = Vec::new();
    for by_seq in head_grid.values() {
        for (&seq, by_b) in by_seq {
            for (&b, &latency) in by_b {
                points.push((vec![b as f64, seq as f64], latency));
            }
        }
    }
    points
}

/// `(num_heads, b, s)` points of a whole generation slice.
fn generation_points_all(slice: &DsaHeadGrid) -> Vec<(Vec<f64>, f64)> {
    let mut points = Vec::new();
    for (&heads, head_grid) in slice {
        for by_seq in head_grid.values() {
            for (&seq, by_b) in by_seq {
                for (&b, &latency) in by_b {
                    points.push((vec![heads as f64, b as f64, seq as f64], latency));
                }
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::dsa::SparseGrid;
    use std::path::PathBuf;

    fn glm_cp_op(cp_size: u32) -> DsaModuleOp {
        DsaModuleOp {
            name: "context_dsa".into(),
            scale_factor: 1.0,
            num_heads: 64,
            kv_cache_dtype: KvCacheQuantMode::Bfloat16,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            gemm_quant_mode: GemmQuantMode::Bfloat16,
            architecture: "GlmMoeDsaForCausalLM".into(),
            index_topk: 2048,
            cp_size,
            full_frac: 1.0,
        }
    }

    fn grid(rows: &[(u32, u32, u32, f64)]) -> SparseGrid {
        let mut g = SparseGrid::new();
        for &(bs, isl, step, lat) in rows {
            g.entry(bs).or_default().insert((isl, step), lat);
        }
        g
    }

    /// Composition parity with Python
    /// `tests/unit/sdk/test_cp_dsa_modeling.py::test_query_cp_composition`,
    /// same synthetic inputs (cp=8, isl=16384, prefix=0, b=1; base 4300,
    /// each AG 50):
    ///
    ///   delta_mqa  = mqa_full/cp - mqa_perc  = 1600/8 - 25  = 175
    ///   delta_topk = tl_full/cp  - tf_perc   = 800/8  - 100 = 0
    ///   latency    = 4300 + 175 + 0 + ag_kv 50 + ag_lse 50  = 4575
    ///
    /// AG volumes: indexer key isl*128, compressed latent isl*(512 + 64).
    #[test]
    fn cp_composition_matches_python_synthetic() {
        let (cp, isl, prefix) = (8u32, 16384u32, 0u32);
        let per_card = isl.div_ceil(cp); // 2048
        let sparse = DsaSparseTables {
            mqa: grid(&[(1, isl, 0, 1600.0), (1, per_card, 0, 25.0)]),
            topk_last: grid(&[(1, isl, 0, 800.0), (1, per_card, 0, 190.0)]),
            topk_flat: grid(&[(1, per_card, 0, 100.0)]),
            dsa_attn: SparseGrid::new(),
        };
        let op = glm_cp_op(cp);
        let mut base_calls: Vec<u32> = Vec::new();
        let mut ag_volumes: Vec<u64> = Vec::new();
        let res = op
            .query_cp_with(
                &sparse,
                1,
                isl,
                prefix,
                false,
                &mut |per_card| {
                    base_calls.push(per_card);
                    Ok(4300.0) // per-card monolithic base
                },
                &mut |elems| {
                    ag_volumes.push(elems);
                    Ok(50.0) // each AG
                },
            )
            .expect("CP composition must succeed");
        assert_eq!(res.latency_ms, 4575.0);
        assert_eq!(res.source, Source::Estimated);
        assert_eq!(base_calls, vec![per_card]); // base queried at isl/cp
        ag_volumes.sort_unstable();
        let mut expected = vec![u64::from(isl) * 128, u64::from(isl) * (512 + 64)];
        expected.sort_unstable();
        assert_eq!(ag_volumes, expected);
    }

    /// Mirrors Python `test_query_cp_raises_when_isl_beyond_grid`: the
    /// composition must propagate `lookup_2d`'s fail-loud (no silent
    /// under-estimate) when isl exceeds the collected sparse grid.
    #[test]
    fn cp_propagates_lookup_fail_loud_beyond_grid() {
        let sparse = DsaSparseTables {
            mqa: grid(&[(1, 16384, 0, 1600.0), (1, 4096, 0, 25.0)]), // grid caps at 16384
            topk_last: grid(&[(1, 16384, 0, 800.0), (1, 4096, 0, 190.0)]),
            topk_flat: grid(&[(1, 4096, 0, 100.0)]),
            dsa_attn: SparseGrid::new(),
        };
        let op = glm_cp_op(8);
        let err = op
            .query_cp_with(
                &sparse,
                1,
                32768, // > grid max 16384
                0,
                false,
                &mut |_| Ok(4300.0),
                &mut |_| Ok(50.0),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("exceeds the collected"),
            "unexpected message: {err}"
        );
    }

    /// Fail-loud symmetry with Python's absent-sparse-table contract: the
    /// glm5_* / dsv32_* sparse parquets ship nowhere, so a CP context query
    /// against a real perf DB must error, naming the missing tables and the
    /// files to collect — never degrade silently to the per-card base.
    #[test]
    fn cp_missing_sparse_tables_fail_loud() {
        let systems_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        let db = PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0")
            .expect("b200_sxm/vllm/0.19.0 must load");
        let op = glm_cp_op(8);
        let err = op.query_context(&db, 1, 16384, 0).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("DSA CP modeling needs sparse tables")
                && msg.contains("['mqa', 'topk_last', 'topk_flat']")
                && msg.contains("GlmMoeDsaForCausalLM")
                && msg.contains("num_heads=64")
                && msg.contains("collect glm5_mqa_logits/glm5_topk first."),
            "unexpected message: {msg}"
        );
    }

    /// `cp_size` is absent from every opspec the Python emitter produces
    /// today (`engine.py::_reject_cp` still guards CP) — it must default to
    /// 1 so existing specs keep the plain non-CP lookup.
    #[test]
    fn cp_size_defaults_to_one_in_serde() {
        let mut v = serde_json::to_value(glm_cp_op(3)).expect("serialize");
        v.as_object_mut().expect("object").remove("cp_size");
        let de: DsaModuleOp = serde_json::from_value(v).expect("deserialize");
        assert_eq!(de.cp_size, 1);
    }

    // ------------------------------------------------------------------
    // Util-space empirical (EMPIRICAL / HYBRID) oracle parity.
    //
    // Oracle values generated from the Python reference on the same data
    // (shared layer OFF so both sides read the same single parquet):
    //
    // ```text
    // uv run --no-sync python - <<'PY'
    // from aiconfigurator.sdk import perf_database, common
    // db = perf_database.get_database("b200_sxm", <backend>, <version>,
    //                                 shared_layer=False)
    // r = db.query_context_dsa_module(  # or query_generation_dsa_module
    //     b=..., s=..., prefix=..., num_heads=...,
    //     kvcache_quant_mode=..., fmha_quant_mode=..., gemm_quant_mode=...,
    //     database_mode=common.DatabaseMode.EMPIRICAL, architecture=...)
    // print(float(r))
    // PY
    // ```
    //
    // The fired calibration variant was verified per anchor by instrumenting
    // `util_empirical.grid_for` in the oracle script (not committed).
    // Regenerate if the shipped DSA tables or the util-empirical math change.
    // ------------------------------------------------------------------

    use crate::common::enums::DatabaseMode;
    use crate::operators::base::Source;
    use crate::perf_database::PerfDatabase;

    const DSV32: &str = "DeepseekV32ForCausalLM";
    const GLM: &str = "GlmMoeDsaForCausalLM";

    fn b200_db(backend: &str, version: &str, mode: DatabaseMode) -> PerfDatabase {
        let systems_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        let mut db = PerfDatabase::load(&systems_root, "b200_sxm", backend, version)
            .expect("b200_sxm db must load");
        db.database_mode = mode;
        db
    }

    fn dsa_op(architecture: &str, num_heads: u32, kv: KvCacheQuantMode) -> DsaModuleOp {
        DsaModuleOp::new(
            "dsa_module",
            num_heads,
            kv,
            FmhaQuantMode::Bfloat16,
            GemmQuantMode::Bfloat16,
            architecture,
            2048,
        )
    }

    fn approx_rel_1e9(got: f64, want: f64) {
        assert!(
            ((got - want) / want).abs() < 1e-9,
            "rust {got} vs python {want}"
        );
    }

    /// Context EMPIRICAL parity on b200_sxm/vllm/0.19.0. Fired variants
    /// (verified in Python):
    /// - DSV32 h=128 prefix=0 off-grid s AND prefix=4096: the DSV32 slice
    ///   collects prefix=[0] only, so BOTH anchor util at the prefix=0 slice
    ///   with full_s = s + prefix -> `ctx_dsa_p0anchor_exact_head` (depth 2).
    /// - GLM h=32 prefix=64: bracketed by the collected [0, 128] axis ->
    ///   `ctx_dsa_exact_head` (depth 3, genuine prefix interpolation).
    /// - GLM h=32 prefix=1000: beyond 128 -> `ctx_dsa_p0anchor_exact_head`.
    /// - DSV32 h=96 (uncollected head) -> cross-head `ctx_dsa_p0anchor`
    ///   (depth 3).
    #[test]
    fn context_empirical_matches_python_oracles() {
        let db = b200_db("vllm", "0.19.0", DatabaseMode::Empirical);
        let cases: [(&str, u32, u32, u32, u32, f64); 5] = [
            (DSV32, 128, 4, 3000, 0, 11.266464115749121),
            (DSV32, 128, 2, 2048, 4096, 4.417586773123638),
            (GLM, 32, 1, 512, 64, 2.62689536098129),
            (GLM, 32, 1, 512, 1000, 0.514264970789275),
            (DSV32, 96, 8, 1024, 0, 5.571143850099078),
        ];
        for (arch, heads, b, s, prefix, expected) in cases {
            let op = dsa_op(arch, heads, KvCacheQuantMode::Bfloat16);
            let result = op.query_context(&db, b, s, prefix).expect("empirical query");
            approx_rel_1e9(result.latency_ms, expected);
            assert_eq!(result.source, Source::Empirical, "({arch}, h={heads})");
        }
    }

    /// Context EMPIRICAL parity on b200_sxm/sglang/0.5.14 (GLM-5, bf16 KV;
    /// heads collected: 8/16/32/64 with a rich measured prefix axis):
    /// - h=128 (uncollected) prefix=512 in range -> cross-head `ctx_dsa`
    ///   (depth 4).
    /// - h=64 prefix=10000 at a collected (prefix, s, b) point ->
    ///   `ctx_dsa_exact_head` exact hit returns the measured latency.
    #[test]
    fn context_empirical_sglang_glm_matches_python_oracles() {
        let db = b200_db("sglang", "0.5.14", DatabaseMode::Empirical);
        let cross = dsa_op(GLM, 128, KvCacheQuantMode::Bfloat16);
        let result = cross.query_context(&db, 2, 4096, 512).expect("cross-head query");
        approx_rel_1e9(result.latency_ms, 13.731357702671428);
        assert_eq!(result.source, Source::Empirical);

        let exact = dsa_op(GLM, 64, KvCacheQuantMode::Bfloat16);
        let result = exact.query_context(&db, 2, 4096, 10000).expect("exact-hit query");
        approx_rel_1e9(result.latency_ms, 8.2065);
        assert_eq!(result.source, Source::Empirical);
    }

    /// HYBRID keeps the silicon answer whenever the silicon lookup resolves
    /// (exact collected hit AND an interpolated interior point) — the
    /// empirical layer must not preempt it.
    #[test]
    fn context_hybrid_prefers_silicon() {
        let db = b200_db("vllm", "0.19.0", DatabaseMode::Hybrid);
        let op = dsa_op(DSV32, 128, KvCacheQuantMode::Bfloat16);
        let exact = op.query_context(&db, 4, 2048, 0).expect("silicon exact hit");
        approx_rel_1e9(exact.latency_ms, 7.6471);
        assert_eq!(exact.source, Source::Silicon);
        let interior = op.query_context(&db, 4, 3000, 0).expect("silicon interp");
        approx_rel_1e9(interior.latency_ms, 11.388627343749999);
        assert_eq!(interior.source, Source::Silicon);
    }

    /// An uncollected kv-quant slice (int8 on b200/vllm) is a typed silicon
    /// miss; HYBRID falls to the empirical layer, which also has no
    /// calibration data (DSV32 missing-slice heuristic -> legacy variant,
    /// grid None) and must surface the terminal EmpiricalNotImplemented —
    /// never a fabricated value. Same terminal miss under EMPIRICAL.
    #[test]
    fn context_missing_slice_raises_empirical_not_implemented() {
        for mode in [DatabaseMode::Hybrid, DatabaseMode::Empirical] {
            let db = b200_db("vllm", "0.19.0", mode);
            let op = dsa_op(DSV32, 128, KvCacheQuantMode::Int8);
            let result = op.query_context(&db, 4, 3000, 0);
            assert!(
                matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
                "{mode:?}: got {result:?}"
            );
        }
    }

    /// Generation EMPIRICAL parity on b200_sxm/vllm/0.19.0:
    /// - h=128 off-grid (b=24, s=3000) and an exact collected hit
    ///   (b=16, s=4097 -> measured 0.2698) on the exact head slice
    ///   (`gen_dsa_exact_heads`, depth 2);
    /// - h=96 (uncollected head) -> cross-head `gen_dsa` (depth 3).
    #[test]
    fn generation_empirical_matches_python_oracles() {
        let db = b200_db("vllm", "0.19.0", DatabaseMode::Empirical);
        let cases: [(u32, u32, u32, f64); 3] = [
            (128, 24, 3000, 0.259452522778543),
            (128, 16, 4097, 0.2698),
            (96, 8, 5000, 0.20801465850337103),
        ];
        for (heads, b, s, expected) in cases {
            let op = dsa_op(DSV32, heads, KvCacheQuantMode::Bfloat16);
            let result = op.query_generation(&db, b, s).expect("empirical query");
            approx_rel_1e9(result.latency_ms, expected);
            assert_eq!(result.source, Source::Empirical, "(h={heads}, b={b}, s={s})");
        }
    }

    /// Generation EMPIRICAL parity on b200_sxm/sglang/0.5.14 (GLM-5, bf16 KV,
    /// exact head slice off-grid).
    #[test]
    fn generation_empirical_sglang_glm_matches_python_oracle() {
        let db = b200_db("sglang", "0.5.14", DatabaseMode::Empirical);
        let op = dsa_op(GLM, 64, KvCacheQuantMode::Bfloat16);
        let result = op.query_generation(&db, 48, 10000).expect("empirical query");
        approx_rel_1e9(result.latency_ms, 0.1927939670669063);
        assert_eq!(result.source, Source::Empirical);
    }

    /// Generation HYBRID on an uncollected kv-quant slice: typed silicon miss
    /// -> empirical -> no calibration slice -> terminal
    /// EmpiricalNotImplemented (mirrors the Python contract).
    #[test]
    fn generation_hybrid_missing_slice_raises_empirical_not_implemented() {
        let db = b200_db("vllm", "0.19.0", DatabaseMode::Hybrid);
        let op = dsa_op(DSV32, 128, KvCacheQuantMode::Int8);
        let result = op.query_generation(&db, 24, 3000);
        assert!(
            matches!(result, Err(AicError::EmpiricalNotImplemented(_))),
            "got {result:?}"
        );
    }
}
