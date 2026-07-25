// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 attention module operator.
//!
//! Mirrors `aiconfigurator.sdk.operations.dsv4.DSV4Module`. Each layer of
//! DSv4 picks between CSA (compressed-sparse) and HCA (hybrid-causal)
//! variants depending on the layer index — the model layer decides which
//! `AttnKind` to use; the operator just routes to the right
//! `db.dsv4.query_*` slice.
//!
//! Slice selection resolves the model's rank-LOCAL head count
//! (`native_heads / tp_size`, passed as `num_heads`) against the CSV `num_heads`
//! head keys, mirroring Python `_dsv4_resolve_head_key`. The `tp_size` axis is
//! NOT an interpolation axis — Python's loaders ignore the CSV `tp_size` column,
//! so the table is collapsed to the last (max) tp measurement at load time. See
//! `perf_database::dsv4` and Python `load_context_dsv4_kind_module_data`.

use serde::{Deserialize, Serialize};
use crate::common::enums::{DatabaseMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::communication::NcclOp;
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::dsv4::{dsv4_attention_sol_ms, dsv4_dims, AttnKind, Dsv4SolDims};
use crate::perf_database::PerfDatabase;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv4ModuleOp {
    pub name: String,
    pub scale_factor: f64,
    pub attn_kind: AttnKind,
    /// Per-rank partitioned head count (`native_heads / tp_size`). This is the
    /// value resolved against the CSV `num_heads` head keys for slice selection
    /// (Python `_dsv4_resolve_head_key`).
    pub num_heads: u32,
    /// Model total attention head count. Retained for provenance and SOL
    /// fallbacks; NOT used for table slice selection (see `num_heads`). The CP
    /// sparse lookups key on it (Python passes `self._native_heads` to
    /// `_lookup_sparse_kernel` / `_csa_topk_top_last`).
    pub native_heads: u32,
    /// Tensor-parallel size. Retained for provenance; the DSV4 table collapses
    /// the tp axis at load time, so this does not select a latency.
    pub tp_size: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub gemm_quant_mode: GemmQuantMode,
    pub architecture: String,
    /// Context-parallel size (Python `_cp_size`). When > 1 the context query
    /// runs the DeepSeek-V4 sparse-CP prefill composition (Python
    /// `ContextDeepSeekV4AttentionModule._query_cp`); 1 (the default) keeps
    /// the plain 3-axis lookup. NOT yet emitted by the Python opspec —
    /// `engine.py::_reject_cp` still guards CP specs; the guard and this
    /// field's emission flip atomically once BOTH the dsa and dsv4 Rust CP
    /// paths land.
    #[serde(default = "default_cp_size")]
    pub cp_size: u32,
    /// HCA sliding-window size (Python `_window_size`, from the model spec).
    /// Feeds the analytic SOL's window-pair count and the HCA CP all-gather
    /// (`self._window_size or isl`). Serde default `None` keeps existing
    /// opspecs valid (SOL then falls back to the pinned Pro window, 128).
    #[serde(default)]
    pub window_size: Option<u32>,
    // --- Structural dims for the analytic SOL (Python op fields, sourced
    // --- from the model config). Serde defaults = the pinned DeepSeek-V4-Pro
    // --- values, so old opspecs resolve exactly as before; the Python
    // --- emitter now sends the model's real dims (Flash differs: hidden
    // --- 4096, q_lora 1024, index_topk 512).
    #[serde(default = "default_hidden_size")]
    pub hidden_size: u32,
    #[serde(default = "default_q_lora_rank")]
    pub q_lora_rank: u32,
    #[serde(default = "default_o_lora_rank")]
    pub o_lora_rank: u32,
    #[serde(default = "default_head_dim")]
    pub head_dim: u32,
    #[serde(default = "default_rope_head_dim")]
    pub rope_head_dim: u32,
    #[serde(default = "default_index_n_heads")]
    pub index_n_heads: u32,
    #[serde(default = "default_index_head_dim")]
    pub index_head_dim: u32,
    #[serde(default = "default_index_topk")]
    pub index_topk: u32,
    /// Rank-LOCAL o_groups (Python `_o_groups` = `max(1, o_groups // tp)`,
    /// pre-divided by the model). `None` (old specs) derives it from the
    /// pinned Pro totals via `Dsv4SolDims::from_pinned`, byte-identical to
    /// the pre-override behaviour.
    #[serde(default)]
    pub o_groups: Option<u32>,
}

fn default_cp_size() -> u32 {
    1
}

fn default_hidden_size() -> u32 {
    7168
}
fn default_q_lora_rank() -> u32 {
    1536
}
fn default_o_lora_rank() -> u32 {
    1024
}
fn default_head_dim() -> u32 {
    512
}
fn default_rope_head_dim() -> u32 {
    64
}
fn default_index_n_heads() -> u32 {
    64
}
fn default_index_head_dim() -> u32 {
    128
}
fn default_index_topk() -> u32 {
    1024
}

/// Production chunked-prefill executes mqa as a chunk sequence; the pair
/// count is additive over chunks, so full-isl mqa decomposes EXACTLY into
/// in-grid lookups (Python `_MQA_CHUNK_TOKENS`):
///   mqa(isl, past) = sum_k mqa(chunk_k, past + offset_k)
const MQA_CHUNK_TOKENS: u32 = 8192;

/// Chunk-decomposed mqa latency (Python
/// `ContextDeepSeekV4AttentionModule._mqa_chunked`): walk `isl` in
/// `MQA_CHUNK_TOKENS` chunks, look each up at `(chunk_len, past0 + offset)`,
/// sum. Any `None` part (absent table / no anchor) makes the whole sum `None`
/// (the composition then fails loud).
fn mqa_chunked(
    isl: u32,
    past0: u32,
    lookup: &mut dyn FnMut(u32, u32) -> Result<Option<f64>, AicError>,
) -> Result<Option<f64>, AicError> {
    let mut total = 0.0;
    let mut offset = 0u32;
    while offset < isl {
        let chunk = MQA_CHUNK_TOKENS.min(isl - offset);
        match lookup(chunk, past0 + offset)? {
            Some(part) => total += part,
            None => return Ok(None),
        }
        offset += MQA_CHUNK_TOKENS;
    }
    Ok(Some(total))
}

impl Dsv4ModuleOp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        attn_kind: AttnKind,
        num_heads: u32,
        native_heads: u32,
        tp_size: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
        gemm_quant_mode: GemmQuantMode,
        architecture: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            attn_kind,
            num_heads,
            native_heads,
            tp_size,
            kv_cache_dtype,
            fmha_quant_mode,
            gemm_quant_mode,
            architecture: architecture.into(),
            cp_size: 1,
            window_size: None,
            hidden_size: default_hidden_size(),
            q_lora_rank: default_q_lora_rank(),
            o_lora_rank: default_o_lora_rank(),
            head_dim: default_head_dim(),
            rope_head_dim: default_rope_head_dim(),
            index_n_heads: default_index_n_heads(),
            index_head_dim: default_index_head_dim(),
            index_topk: default_index_topk(),
            o_groups: None,
        }
    }

    /// SOL dims from the opspec fields (Python's op carries them from the
    /// model config). `o_groups: None` (old specs) falls back to the pinned
    /// per-architecture derivation; `window_size: None` falls back to the
    /// pinned Pro window.
    pub(crate) fn sol_dims(&self) -> Dsv4SolDims {
        let pinned = dsv4_dims(&self.architecture);
        let local_o_groups = match self.o_groups {
            Some(groups) => i64::from(groups).max(1),
            None => Dsv4SolDims::from_pinned(pinned, i64::from(self.num_heads)).local_o_groups,
        };
        Dsv4SolDims {
            hidden_size: i64::from(self.hidden_size),
            q_lora_rank: i64::from(self.q_lora_rank),
            o_lora_rank: i64::from(self.o_lora_rank),
            head_dim: i64::from(self.head_dim),
            rope_head_dim: i64::from(self.rope_head_dim),
            index_n_heads: i64::from(self.index_n_heads),
            index_head_dim: i64::from(self.index_head_dim),
            index_topk: i64::from(self.index_topk),
            window_size: self.window_size.map_or(pinned.window_size, i64::from),
            local_o_groups,
        }
    }

    /// The per-(b, s, prefix) module base query shared by the plain and CP
    /// paths (Python `_module_base` -> the standard module query, which
    /// includes the CSA topk DELTA correction and dispatches on the database
    /// mode): SILICON queries the table; HYBRID converts a typed silicon miss
    /// into the util-space empirical estimate; EMPIRICAL always estimates.
    /// The SOL diagnostic modes never reach the compiled engine (the routing
    /// gate delegates them to the Python step).
    fn module_base(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
        prefix: u32,
    ) -> Result<(f64, Source), AicError> {
        match db.database_mode {
            DatabaseMode::Empirical => Ok((
                self.context_empirical(db, batch_size, s, prefix)?,
                Source::Empirical,
            )),
            DatabaseMode::Hybrid => match self.context_silicon(db, batch_size, s, prefix) {
                Ok(latency) => Ok((latency, Source::Silicon)),
                Err(err) if err.is_missing_perf_data() => Ok((
                    self.context_empirical(db, batch_size, s, prefix)?,
                    Source::Empirical,
                )),
                Err(err) => Err(err),
            },
            _ => Ok((self.context_silicon(db, batch_size, s, prefix)?, Source::Silicon)),
        }
    }

    fn context_silicon(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        db.dsv4.query_context(
            &db.system_spec,
            self.attn_kind,
            batch_size,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            self.fmha_quant_mode,
            self.gemm_quant_mode,
            &self.architecture,
            prefix,
            Some(self.sol_dims()),
        )
    }

    /// `SOL(query)/util` over the op's own context slice. Mirrors Python
    /// `_query_context_attn_table::get_empirical`:
    ///
    /// - Genuine prefix interpolation needs >= 2 collected prefix points
    ///   bracketing the query: depth-3 grid over `(prefix, s, b)`, query
    ///   `(prefix, s, b)`.
    /// - Otherwise (degenerate prefix axis or out-of-range query) anchor at
    ///   the prefix=0 slice at `full_s = s + prefix` (regime-matched), with
    ///   the prefix effect carried by `sol_q`: depth-2 grid over `(s, b)`,
    ///   query `(s + prefix, b)`.
    fn context_empirical(
        &self,
        db: &PerfDatabase,
        b: u32,
        s: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let spec = &db.system_spec;
        let dims = self.sol_dims();
        let cr = self.attn_kind.compress_ratio();
        let heads = i64::from(self.num_heads);
        let (kv, fmha, gemm) = (self.kv_cache_dtype, self.fmha_quant_mode, self.gemm_quant_mode);
        let sol_at = move |b_: i64, s_: i64, p_: i64| {
            dsv4_attention_sol_ms(spec, &dims, cr, true, kv, fmha, gemm, b_, s_, p_, heads)
        };
        // True SOL(b, s, prefix) at the query (Python `sol_q = get_sol()[0]`).
        let sol_q = sol_at(i64::from(b), i64::from(s), i64::from(prefix));

        // Own-slice `(prefix, s, b)` points; a typed coverage miss means no
        // prefix keys (Python: `prefix_keys = ()`), so the p0-anchor branch
        // then finds no grid and estimate() raises the empirical miss.
        let points = match db.dsv4.context_points(
            self.attn_kind,
            self.num_heads,
            kv,
            fmha,
            gemm,
        ) {
            Ok(points) => Some(points),
            Err(err) if err.is_missing_perf_data() => None,
            Err(err) => return Err(err),
        };
        let prefix_keys: std::collections::BTreeSet<u32> = points
            .iter()
            .flatten()
            .map(|(coords, _)| coords[0] as u32)
            .collect();
        let interp_prefix = prefix_keys.len() >= 2
            && *prefix_keys.first().expect("non-empty") <= prefix
            && prefix <= *prefix_keys.last().expect("non-empty");

        // Grid cache key mirrors Python's
        // (key_tag, quants, num_heads, cr, depth); `architecture` stands in
        // for Python's `id(node)` identity component (the Rust slice keys it).
        let key_stem = format!(
            "{}:{}:{}:{}:{}:{}",
            self.architecture,
            fmha.name(),
            kv.name(),
            gemm.name(),
            self.num_heads,
            cr
        );
        if interp_prefix {
            let sol3 = |c: &[f64]| sol_at(c[2] as i64, c[1] as i64, c[0] as i64); // c=(prefix, s, b)
            let key = format!("dsv4_ctx_attn:{key_stem}:3");
            let grid = db.util_grids.get_or_try_build(&key, || {
                Ok(points.map(|points| UtilGrid::new(util_empirical::build_samples(points, sol3))))
            })?;
            let query = [f64::from(prefix), f64::from(s), f64::from(b)];
            let (latency, _) = util_empirical::estimate(sol_q, &query, grid.as_deref(), 1.0)?;
            // Own-shape util fired (Python dsv4.py:889, default tier).
            db.note_provenance(util_empirical::ProvenanceTier::Empirical);
            Ok(latency)
        } else {
            let sol2 = |c: &[f64]| sol_at(c[1] as i64, c[0] as i64, 0); // c=(s, b); anchor prefix=0
            let key = format!("dsv4_ctx_attn_p0anchor:{key_stem}:2");
            let grid = db.util_grids.get_or_try_build(&key, || {
                // Python `require_data_slice(_slice(), 0)`: no prefix=0 rows
                // is a typed coverage miss -> no grid.
                let p0_points: Vec<(Vec<f64>, f64)> = points
                    .into_iter()
                    .flatten()
                    .filter(|(coords, _)| coords[0] == 0.0)
                    .map(|(coords, latency)| (vec![coords[1], coords[2]], latency))
                    .collect();
                if p0_points.is_empty() {
                    return Ok(None);
                }
                Ok(Some(UtilGrid::new(util_empirical::build_samples(p0_points, sol2))))
            })?;
            let query = [f64::from(s) + f64::from(prefix), f64::from(b)];
            let (latency, _) = util_empirical::estimate(sol_q, &query, grid.as_deref(), 1.0)?;
            // Own-shape util fired (Python dsv4.py:889, default tier).
            db.note_provenance(util_empirical::ProvenanceTier::Empirical);
            Ok(latency)
        }
    }

    pub fn query_context(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        // CP (round-robin sequence split) prefill takes the sparse-CP
        // composition path (Python `ContextDeepSeekV4AttentionModule.query`
        // -> `_query_cp` when `_cp_size > 1`).
        if self.cp_size > 1 {
            return self.query_context_cp(db, batch_size, isl, prefix);
        }
        // Mirror Python `ContextDeepSeekV4AttentionModule._query_context_attn_table`
        // (SILICON path): a 3-axis perf_interp v2 Grid query over
        // `(prefix, isl, batch)` with the step axis KEPT. The caller supplies
        // the new-token count as `isl` (Python's `s = effective_isl =
        // isl - prefix`, computed in `run_context_ops`); a prefix beyond the
        // collected range resolves via util-hold with the prefix-aware SOL
        // carrying the effect — matching Python. HYBRID/EMPIRICAL route
        // through `module_base`'s mode dispatch.
        let (raw, source) = self.module_base(db, batch_size, isl, prefix)?;
        Ok(PerformanceResult::new(raw, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// Context-Parallel (CP) prefill — DeepSeek-V4 CSA / HCA composition.
    ///
    /// Wires the real data dependencies and delegates to
    /// [`Self::query_cp_with`] (the verbatim mirror of Python
    /// `ContextDeepSeekV4AttentionModule._query_cp`):
    /// - base = the standard module query at `(b, per_card, prefix)`
    ///   (topk-DELTA-corrected, like Python's `_module_base`);
    /// - mqa lookup = `db.dsv4.query_paged_mqa_logits` at the REAL batch `b`
    ///   with the op's `tp_size` / `native_heads` (Python
    ///   `_lookup_sparse_kernel`);
    /// - topk top_last = `db.dsv4.csa_topk_top_last` (the raw top_last rows
    ///   retained by the topk-calib loader);
    /// - AG = `query_nccl(half, cp, "all_gather", elems)` via [`NcclOp`],
    ///   which mirrors Python's fan-out cap + multi-node bandwidth scaling.
    fn query_context_cp(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
    ) -> Result<PerformanceResult, AicError> {
        self.query_cp_with(
            batch_size,
            isl,
            prefix,
            &mut |per_card| {
                self.module_base(db, batch_size, per_card, prefix)
                    .map(|(latency, _)| latency)
            },
            &mut |chunk_isl, past| {
                db.dsv4.query_paged_mqa_logits(
                    batch_size,
                    chunk_isl,
                    past,
                    self.tp_size,
                    self.native_heads,
                )
            },
            &mut |isl_q, step| db.dsv4.csa_topk_top_last(isl_q, step, self.native_heads, batch_size),
            &mut |elems| {
                NcclOp::new(
                    format!("{}_cp_all_gather", self.name),
                    1.0,
                    elems as f64,
                    self.cp_size,
                    "all_gather",
                )
                .query(db, 1)
                .map(|r| r.latency_ms)
            },
        )
    }

    /// CP (round-robin split) per-layer DSV4 composition. Verbatim mirror of
    /// Python `ContextDeepSeekV4AttentionModule._query_cp`:
    ///
    /// ```text
    /// CSA (ratio 4):
    ///   result = module(isl/cp, prefix)                       # per-card base
    ///          + [mqa(isl, prefix)/cp      - mqa(isl/cp, prefix)]
    ///          + [top_last(isl, prefix)/cp - top_last(isl/cp, prefix)]
    ///          + AG(b * isl * index_head_dim)                 # indexer key
    ///          + AG(b * (isl // 4) * head_dim)                # compressed KV
    /// HCA (ratio 128):
    ///   result = module(isl/cp, prefix)
    ///          + AG(b * min(isl, window) * head_dim)          # windowed dense KV
    ///          + AG(b * (isl // 128) * head_dim)              # compressed KV
    /// ```
    ///
    /// mqa is chunk-decomposed ([`mqa_chunked`]); all four CSA sparse values
    /// are REQUIRED — any `None` fails loud naming the tables (Python raises
    /// `PerfDataNotAvailableError` with the same message). The dependencies
    /// are injected so the composition is unit-testable against the same
    /// synthetic inputs as the Python test
    /// (`tests/unit/sdk/test_cp_dsv4_modeling.py`).
    #[allow(clippy::too_many_arguments)]
    fn query_cp_with(
        &self,
        b: u32,
        isl: u32,
        prefix: u32,
        base: &mut dyn FnMut(u32) -> Result<f64, AicError>,
        mqa_lookup: &mut dyn FnMut(u32, u32) -> Result<Option<f64>, AicError>,
        topk_top_last: &mut dyn FnMut(u32, u32) -> Result<Option<f64>, AicError>,
        ag: &mut dyn FnMut(u64) -> Result<f64, AicError>,
    ) -> Result<PerformanceResult, AicError> {
        let cp = self.cp_size;
        let per_card = isl.div_ceil(cp).max(1); // ceil: critical path = busiest CP rank
        // AG element widths come from the op's own dims (Python uses
        // `self._index_head_dim` / `self._head_dim` in `_query_cp`).
        let (index_head_dim, head_dim) = (u64::from(self.index_head_dim), u64::from(self.head_dim));
        // Base: per-card monolithic module at (b, per_card, prefix).
        let mut latency = base(per_card)?;
        // x b throughout: mqa/topk are linear in batch and the all-gather
        // moves b sequences' worth of KV (Python folds b inside `ag`).
        let tokens_of = |elems_per_seq: u64| u64::from(b) * elems_per_seq;
        match self.attn_kind {
            AttnKind::Csa => {
                // CSA: super-linear indexer (mqa) + topk -> full/cp swap.
                // NOTE: Python logs a warning here when
                // AIC_DSV4_TOPK_CORRECTION=0 (dsv4.py ~1071): the composition
                // subtracts top_last(per_card) against a base whose standard
                // module query already had the flat-vs-top_last DELTA removed;
                // disabling the correction leaves a per-card flat-vs-top_last
                // over-estimate. Rust has no logging — behaviour is identical
                // (warn-only in Python), so the branch is a comment.
                let mqa_full = mqa_chunked(isl, prefix, mqa_lookup)?;
                let mqa_perc = mqa_chunked(per_card, prefix, mqa_lookup)?;
                let tl_full = topk_top_last(isl, prefix)?;
                let tl_perc = topk_top_last(per_card, prefix)?;
                // Fail loud (Python parity): the CSA CP deltas REQUIRE the
                // sparse tables — silently dropping them would hide a
                // missing/uncollected parquet behind a too-small base-only
                // estimate. Message mirrors Python's exactly.
                let (Some(mqa_full), Some(mqa_perc), Some(tl_full), Some(tl_perc)) =
                    (mqa_full, mqa_perc, tl_full, tl_perc)
                else {
                    return Err(AicError::PerfDatabase(format!(
                        "DeepSeek-V4 CSA CP modeling needs sparse tables (paged_mqa_logits + \
                         csa_topk_calib top_last) at num_heads={}, b={b}; \
                         collect dsv4_paged_mqa_logits_module / dsv4_csa_topk_calib first.",
                        self.native_heads
                    )));
                };
                let cp_f = f64::from(cp);
                latency += (mqa_full / cp_f - mqa_perc) + (tl_full / cp_f - tl_perc);
                // AG indexer key (mqa stage).
                latency += ag(tokens_of(u64::from(isl) * index_head_dim))?;
            }
            AttnKind::Hca => {
                // HCA (128) / SWA: windowed dense; no indexer/topk selection.
                let window = match self.window_size {
                    Some(w) if w > 0 => w,
                    _ => isl, // Python `self._window_size or isl`
                };
                // AG windowed dense KV (the HCA "+1").
                latency += ag(tokens_of(u64::from(isl.min(window)) * head_dim))?;
            }
        }
        // Compressed-KV all-gather: both CSA and HCA gather the isl//ratio
        // compressed entries (the fmha-stage KV). Python guards `if ratio:`;
        // in Rust the kind IS the ratio (4 or 128), always truthy.
        let ratio = self.attn_kind.compress_ratio() as u32;
        latency += ag(tokens_of(u64::from(isl / ratio) * head_dim))?;
        Ok(PerformanceResult::new(latency, Source::Estimated).scaled(self.scale_factor))
    }

    /// Database-mode dispatch mirroring Python
    /// `_query_generation_attn_table` (`operations/dsv4.py`): SILICON queries
    /// the table; HYBRID converts a typed silicon miss into the util-space
    /// empirical estimate; EMPIRICAL always estimates.
    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let (latency, source) = match db.database_mode {
            DatabaseMode::Empirical => (
                self.generation_empirical(db, batch_size, s)?,
                Source::Empirical,
            ),
            DatabaseMode::Hybrid => match self.generation_silicon(db, batch_size, s) {
                Ok(latency) => (latency, Source::Silicon),
                Err(err) if err.is_missing_perf_data() => (
                    self.generation_empirical(db, batch_size, s)?,
                    Source::Empirical,
                ),
                Err(err) => return Err(err),
            },
            _ => (self.generation_silicon(db, batch_size, s)?, Source::Silicon),
        };
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }

    /// No fmha argument: the generation table keys on kv dtype only and the
    /// SOL dtype is derived from kv inside `query_generation` (PR #1337).
    /// `self.fmha_quant_mode` stays on the op for the context path.
    fn generation_silicon(&self, db: &PerfDatabase, batch_size: u32, s: u32) -> Result<f64, AicError> {
        db.dsv4.query_generation(

            &db.system_spec,
            self.attn_kind,
            batch_size,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            self.gemm_quant_mode,
            &self.architecture,
            Some(self.sol_dims()),
        )
    }

    /// `SOL(query)/util` over the op's own `(b, s_total)` generation slice.
    /// Mirrors Python `_query_generation_attn_table::get_empirical` (grid
    /// depth 2, `sol_fn = lambda c: get_sol(c[0], c[1])[0]`, query `(b, s)`).
    fn generation_empirical(&self, db: &PerfDatabase, b: u32, s: u32) -> Result<f64, AicError> {
        let spec = &db.system_spec;
        let dims = self.sol_dims();
        let cr = self.attn_kind.compress_ratio();
        let heads = i64::from(self.num_heads);
        let (kv, gemm) = (self.kv_cache_dtype, self.gemm_quant_mode);
        // Python derives the decode SOL dtype from the kv-cache dtype at the
        // top of `_query_generation_attn_table` (the fmha label is inert for
        // generation; the table keys on kv dtype).
        let fmha = if kv == KvCacheQuantMode::Fp8 {
            FmhaQuantMode::Fp8
        } else {
            FmhaQuantMode::Bfloat16
        };
        let sol = move |c: &[f64]| {
            // c=(b, s_total); is_context=false, prefix=0.
            dsv4_attention_sol_ms(
                spec,
                &dims,
                cr,
                false,
                kv,
                fmha,
                gemm,
                c[0] as i64,
                c[1] as i64,
                0,
                heads,
            )
        };
        // Python keys the grid on (kv, gemm, num_heads, cr) — no fmha level
        // (derived from kv above); `architecture` stands in for Python's
        // `id(node)` identity component.
        let key = format!(
            "dsv4_gen_attn:{}:{}:{}:{}:{}",
            self.architecture,
            kv.name(),
            gemm.name(),
            self.num_heads,
            cr
        );
        let grid = db.util_grids.get_or_try_build(&key, || {
            match db
                .dsv4
                .generation_points(self.attn_kind, self.num_heads, kv, gemm)
            {
                Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
                // Typed coverage miss -> no grid (estimate() raises the
                // empirical miss); schema/load errors propagate.
                Err(err) if err.is_missing_perf_data() => Ok(None),
                Err(err) => Err(err),
            }
        })?;
        let query = [f64::from(b), f64::from(s)];
        let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
        // Own-shape util fired (Python dsv4.py:1313, default tier).
        db.note_provenance(util_empirical::ProvenanceTier::Empirical);
        Ok(latency)
    }
}

/// SGLang DeepSeek-V4 MegaMoE routed module.
///
/// Mirrors Python `operations/dsv4.py::DeepSeekV4MegaMoEModule`: the measured
/// routed MegaMoE module boundary (prepared hidden states + top-k tensors ->
/// SGLang pre-dispatch -> `deep_gemm.fp8_fp4_mega_moe` -> routed output
/// scaling). Gate/top-k and shared experts are modeled outside this op. One
/// class serves both phases via `is_context` (same as Python), so a single
/// `Op::Dsv4MegaMoe` variant dispatches on `ctx.num_tokens`.
///
/// Mode contract (Python `_query_megamoe_table`): SILICON/HYBRID only. The op
/// has NO empirical path — EMPIRICAL (and the SOL diagnostics) raise a typed
/// perf-data error, and under HYBRID a silicon miss propagates RAW (Python
/// never routes this op through `_query_silicon_or_hybrid`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dsv4MegaMoeOp {
    pub name: String,
    pub scale_factor: f64,
    pub hidden_size: u32,
    /// FULL (un-partitioned) MoE inter size — Python passes
    /// `self._moe_inter_size` (models/deepseek_v4.py), not the tp-local one.
    pub inter_size: u32,
    pub topk: u32,
    pub num_experts: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
    pub quant_mode: MoeQuantMode,
    /// Normalized by the Python op ctor (`uniform` -> `balanced`); the query
    /// re-applies the normalization defensively (see `query`).
    pub workload_distribution: String,
    pub is_context: bool,
    pub source_policy: String,
    pub pre_dispatch: String,
    pub num_fused_shared_experts: u32,
    pub kernel_source: String,
    pub kernel_dtype: String,
}

impl Dsv4MegaMoeOp {
    /// Query measured MegaMoE routed-module latency at the rank-LOCAL token
    /// count `num_tokens` (Python `query`'s `x`; the perf rows are indexed by
    /// local-rank tokens — do NOT pre-multiply by attention_dp_size).
    pub fn query(&self, db: &PerfDatabase, num_tokens: u32) -> Result<PerformanceResult, AicError> {
        // Python `DeepSeekV4MegaMoEModule.query`: Blackwell-only guard
        // (`ValueError`, not a perf-data miss).
        let sm_version = db
            .system_spec
            .gpu
            .sm_version
            .map_or(-1_i64, i64::from);
        if sm_version < 100 {
            return Err(AicError::ModelConfig(format!(
                "DeepSeek-V4 MegaMoE is only supported on Blackwell-class GPUs (SM >= 100); \
                 got sm_version={sm_version}."
            )));
        }
        // Python `_query_megamoe_table`: `database_mode not in (SILICON,
        // HYBRID)` -> PerfDataNotAvailableError. HYBRID takes the SAME
        // silicon table path as SILICON, and a miss propagates raw.
        match db.database_mode {
            DatabaseMode::Silicon | DatabaseMode::Hybrid => {}
            mode => {
                return Err(AicError::PerfDatabase(format!(
                    "DSv4 MegaMoE module only supports measured SILICON data, got \
                     database_mode={mode:?}."
                )))
            }
        }
        // Python `_normalize_distribution` (op ctor): uniform -> balanced.
        // The Python emitter sends the already-normalized value; re-apply for
        // hand-written specs.
        let distribution = if self.workload_distribution == "uniform" {
            "balanced"
        } else {
            self.workload_distribution.as_str()
        };
        let latency = db.dsv4_megamoe.query_module(
            num_tokens,
            self.hidden_size,
            self.inter_size,
            self.topk,
            self.num_experts,
            self.moe_tp_size,
            self.moe_ep_size,
            self.quant_mode,
            distribution,
            self.is_context,
            &self.source_policy,
            &self.pre_dispatch,
            self.num_fused_shared_experts,
            &self.kernel_source,
            &self.kernel_dtype,
        )?;
        // Python: `PerformanceResult(float(result) * scale, source="silicon")`
        // — no clamp (nothing is subtracted on this path).
        Ok(PerformanceResult::new(latency, Source::Silicon).scaled(self.scale_factor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_database::dsv4::Dsv4Table;
    use std::path::PathBuf;

    fn dsv4_cp_op(attn_kind: AttnKind, cp_size: u32, window_size: Option<u32>) -> Dsv4ModuleOp {
        Dsv4ModuleOp {
            name: "context_dsv4".into(),
            scale_factor: 1.0,
            attn_kind,
            num_heads: 64,
            native_heads: 64,
            tp_size: 1,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            gemm_quant_mode: GemmQuantMode::Fp8Block,
            architecture: "DeepseekV4ForCausalLM".into(),
            cp_size,
            window_size,
            hidden_size: default_hidden_size(),
            q_lora_rank: default_q_lora_rank(),
            o_lora_rank: default_o_lora_rank(),
            head_dim: default_head_dim(),
            rope_head_dim: default_rope_head_dim(),
            index_n_heads: default_index_n_heads(),
            index_head_dim: default_index_head_dim(),
            index_topk: default_index_topk(),
            o_groups: None,
        }
    }

    /// CSA composition parity with Python
    /// `tests/unit/sdk/test_cp_dsv4_modeling.py::test_query_cp_csa_composition`,
    /// same synthetic inputs (cp=8, isl=16384, prefix=0, b=1; base 4300, each
    /// AG 50; mqa stubs keyed (isl, past); topk keyed isl):
    ///
    ///   mqa_full   = mqa(8192, 0) + mqa(8192, 8192) = 900 + 700 = 1600
    ///   mqa_perc   = mqa(2048, 0)                   = 25
    ///   delta_mqa  = 1600/8 - 25  = 175
    ///   delta_topk = 800/8  - 100 = 0
    ///   latency    = 4300 + 175 + 0 + ag(indexer) 50 + ag(compressed) 50 = 4575
    ///
    /// AG volumes: indexer key b*isl*index_head_dim(128); compressed KV
    /// b*(isl//4)*head_dim(512).
    #[test]
    fn cp_csa_composition_matches_python_synthetic() {
        let (cp, isl, b) = (8u32, 16384u32, 1u32);
        let per_card = isl.div_ceil(cp); // 2048
        let op = dsv4_cp_op(AttnKind::Csa, cp, None);
        let mut base_calls: Vec<u32> = Vec::new();
        let mut ag_volumes: Vec<u64> = Vec::new();
        let res = op
            .query_cp_with(
                b,
                isl,
                0,
                &mut |pc| {
                    base_calls.push(pc);
                    Ok(4300.0) // per-card monolithic base
                },
                &mut |chunk_isl, past| {
                    Ok(Some(match (chunk_isl, past) {
                        (8192, 0) => 900.0,
                        (8192, 8192) => 700.0,
                        (2048, 0) => 25.0,
                        other => panic!("unexpected mqa lookup {other:?}"),
                    }))
                },
                &mut |isl_q, step| {
                    assert_eq!(step, 0);
                    Ok(Some(match isl_q {
                        16384 => 800.0,
                        2048 => 100.0,
                        other => panic!("unexpected topk lookup isl={other}"),
                    }))
                },
                &mut |elems| {
                    ag_volumes.push(elems);
                    Ok(50.0) // each AG
                },
            )
            .expect("CP composition must succeed");
        assert_eq!(res.latency_ms, 4575.0);
        assert_eq!(res.source, Source::Estimated);
        assert_eq!(base_calls, vec![per_card]); // base queried at ceil(isl/cp)
        ag_volumes.sort_unstable();
        let mut expected = vec![
            u64::from(b) * u64::from(isl) * 128,
            u64::from(b) * u64::from(isl / 4) * 512,
        ];
        expected.sort_unstable();
        assert_eq!(ag_volumes, expected);
    }

    /// HCA composition parity with Python `test_query_cp_hca_composition`
    /// (cp=4, isl=8192, b=2, window=2048; base 1000, each AG 30): no
    /// indexer/topk swap — base + AG(windowed dense KV) + AG(compressed KV)
    /// = 1000 + 30 + 30. AG volumes: b*min(isl,window)*512, b*(isl//128)*512.
    #[test]
    fn cp_hca_composition_matches_python_synthetic() {
        let (cp, isl, b, window) = (4u32, 8192u32, 2u32, 2048u32);
        let per_card = isl.div_ceil(cp); // 2048
        let op = dsv4_cp_op(AttnKind::Hca, cp, Some(window));
        let mut base_calls: Vec<u32> = Vec::new();
        let mut ag_volumes: Vec<u64> = Vec::new();
        let res = op
            .query_cp_with(
                b,
                isl,
                0,
                &mut |pc| {
                    base_calls.push(pc);
                    Ok(1000.0)
                },
                &mut |_, _| panic!("HCA must not touch the mqa table"),
                &mut |_, _| panic!("HCA must not touch the topk table"),
                &mut |elems| {
                    ag_volumes.push(elems);
                    Ok(30.0)
                },
            )
            .expect("CP composition must succeed");
        assert_eq!(res.latency_ms, 1060.0);
        assert_eq!(base_calls, vec![per_card]);
        ag_volumes.sort_unstable();
        let mut expected = vec![
            u64::from(b) * u64::from(isl.min(window)) * 512,
            u64::from(b) * u64::from(isl / 128) * 512,
        ];
        expected.sort_unstable();
        assert_eq!(ag_volumes, expected);
    }

    /// Fail-loud symmetry with Python
    /// `test_query_cp_fails_loud_without_sparse_tables`: any missing CSA
    /// sparse value (mqa or topk top_last) must raise naming the tables and
    /// the files to collect — never degrade silently to the per-card base.
    #[test]
    fn cp_missing_sparse_tables_fail_loud() {
        let op = dsv4_cp_op(AttnKind::Csa, 2, None);
        let err = op
            .query_cp_with(
                1,
                4096,
                0,
                &mut |_| Ok(100.0),
                &mut |_, _| Ok(None),
                &mut |_, _| Ok(None),
                &mut |_| Ok(0.0),
            )
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(
                "DeepSeek-V4 CSA CP modeling needs sparse tables (paged_mqa_logits + \
                 csa_topk_calib top_last)"
            ) && msg.contains("num_heads=64, b=1")
                && msg.contains("collect dsv4_paged_mqa_logits_module / dsv4_csa_topk_calib first."),
            "unexpected message: {msg}"
        );
    }

    /// Real-DB fail-loud: b200_sxm/sglang/0.5.10 ships the DSV4 module +
    /// paged_mqa_logits parquets but NO `dsv4_csa_topk_calib` (no system
    /// ships it), so a CSA CP context query resolves the base + mqa and then
    /// fails loud on the missing top_last rows — exactly Python's end-to-end
    /// behaviour today.
    #[test]
    fn cp_missing_calib_fails_loud_on_real_db() {
        let systems_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        let data_dir = systems_root.join("data/b200_sxm/sglang/0.5.10");
        if !data_dir.join("dsv4_csa_context_module_perf.parquet").exists()
            || !data_dir.join("dsv4_paged_mqa_logits_module_perf.parquet").exists()
        {
            return; // git-lfs data not materialized
        }
        let db = PerfDatabase::load(&systems_root, "b200_sxm", "sglang", "0.5.10")
            .expect("b200_sxm/sglang/0.5.10 must load");
        let op = dsv4_cp_op(AttnKind::Csa, 8, None);
        let err = op.query_context(&db, 1, 16384, 0).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("DeepSeek-V4 CSA CP modeling needs sparse tables")
                && msg.contains("csa_topk_calib top_last"),
            "unexpected message: {msg}"
        );
    }

    /// Real-data parity anchor for the chunk-decomposed mqa lookup on the
    /// SHIPPED gb200 sparse table (in-grid: isl=16384 walks two chunks,
    /// (8192, past=0) + (8192, past=8192), both exact grid hits). Python
    /// oracle generated with:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import get_database
    /// from aiconfigurator.sdk.operations.dsv4 import ContextDeepSeekV4AttentionModule as M
    /// db = get_database('gb200', 'sglang', '0.5.10')
    /// M.load_data(db)
    /// print(M._mqa_chunked(db, 1, 16384, 0, 1, 64))"
    /// # -> 0.45932399999999995  (= 0.194236 + 0.265088)
    /// ```
    #[test]
    fn mqa_chunked_matches_python_on_gb200_data() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/gb200/sglang/0.5.10");
        if !root.join("dsv4_paged_mqa_logits_module_perf.parquet").exists() {
            return; // git-lfs data not materialized
        }
        let table = Dsv4Table::new(root);
        let mut lookup =
            |chunk_isl: u32, past: u32| table.query_paged_mqa_logits(1, chunk_isl, past, 1, 64);
        let got = mqa_chunked(16384, 0, &mut lookup)
            .expect("lookup must not error")
            .expect("in-grid chunks must resolve");
        let want = 0.45932399999999995;
        assert!(
            ((got - want) / want).abs() < 1e-9,
            "rust {got} vs python {want}"
        );
    }

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems")
    }

    fn b200_sglang_root() -> PathBuf {
        systems_root().join("data/b200_sxm/sglang/0.5.10")
    }

    /// DeepSeek-V4-Pro op at tp=8: rank-local num_heads = 128/8 = 16,
    /// o_groups: None derives local 2 from the pinned totals — matching the
    /// Python oracle calls (num_heads=16, o_groups=2, window_size=128).
    fn dsv4_pro_op(attn_kind: AttnKind) -> Dsv4ModuleOp {
        Dsv4ModuleOp {
            name: "dsv4_pro".into(),
            scale_factor: 1.0,
            attn_kind,
            num_heads: 16,
            native_heads: 128,
            tp_size: 8,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            gemm_quant_mode: GemmQuantMode::Fp8Block,
            architecture: "DeepseekV4ForCausalLM".into(),
            cp_size: 1,
            window_size: None,
            hidden_size: default_hidden_size(),
            q_lora_rank: default_q_lora_rank(),
            o_lora_rank: default_o_lora_rank(),
            head_dim: default_head_dim(),
            rope_head_dim: default_rope_head_dim(),
            index_n_heads: default_index_n_heads(),
            index_head_dim: default_index_head_dim(),
            index_topk: default_index_topk(),
            o_groups: None,
        }
    }

    fn approx(got: f64, want: f64) {
        assert!(
            ((got - want) / want).abs() < 1e-9,
            "rust {got} vs python {want}"
        );
    }

    /// Oracle values generated from the Python reference on the same data:
    ///
    /// ```text
    /// uv run --no-sync python3 -c "
    /// from aiconfigurator.sdk import perf_database, common
    /// from aiconfigurator.sdk.operations.dsv4 import (
    ///     ContextDeepSeekV4AttentionModule as CTX,
    ///     GenerationDeepSeekV4AttentionModule as GEN)
    /// db = perf_database.get_database('b200_sxm', 'sglang', '0.5.10')
    /// pro = dict(num_heads=16, native_heads=128, tp_size=8, hidden_size=7168,
    ///            q_lora_rank=1536, o_lora_rank=1024, head_dim=512, rope_head_dim=64,
    ///            index_n_heads=64, index_head_dim=128, index_topk=1024, window_size=128,
    ///            o_groups=2, kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    ///            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    ///            gemm_quant_mode=common.GEMMQuantMode.fp8_block)
    /// EMP = common.DatabaseMode.EMPIRICAL
    /// for b, s, p, cr in [(8,512,0,4), (8,700,0,4), (8,512,1024,4), (3,8192,4096,4),
    ///                     (1,128,0,128), (8,512,1024,128)]:
    ///     print(float(CTX._query_context_attn_table(db, b=b, s=s, prefix=p,
    ///           compress_ratio=cr, database_mode=EMP, **pro)))
    /// for b, s, cr in [(16,385,4), (16,200,4), (13,100000,4), (16,385,128)]:
    ///     print(float(GEN._query_generation_attn_table(db, b=b, s=s,
    ///           compress_ratio=cr, database_mode=EMP, **pro)))"
    /// ```
    ///
    /// The 0.5.10 context tables carry a single step=0 anchor, so every
    /// context case exercises the p0-anchor branch (depth-2 grid, query
    /// `(s + prefix, b)`, the prefix effect carried by sol_q). Covers per
    /// phase: exact collected hit, off-grid interior IDW, prefix>0 anchored
    /// at full_s, and the generation kv->fmha SOL derivation. Regenerate if
    /// the shipped tables or the util-empirical math change.
    #[test]
    fn dsv4_empirical_matches_python_oracles() {
        let root = b200_sglang_root();
        if !root.join("dsv4_csa_context_module_perf.parquet").exists() {
            return; // git-lfs data not materialized
        }
        let mut db = PerfDatabase::load(&systems_root(), "b200_sxm", "sglang", "0.5.10")
            .expect("b200_sxm/sglang/0.5.10 must load");
        db.database_mode = DatabaseMode::Empirical;

        let ctx_cases: &[(AttnKind, u32, u32, u32, f64)] = &[
            // exact collected hit (prefix=0) reconstructs the measured value
            (AttnKind::Csa, 8, 512, 0, 0.9819),
            // off-grid interior isl
            (AttnKind::Csa, 8, 700, 0, 1.3345620964544214),
            // prefix>0: p0-anchor at full_s = s + prefix, SOL carries prefix
            (AttnKind::Csa, 8, 512, 1024, 1.0674237160346303),
            (AttnKind::Csa, 3, 8192, 4096, 11.05984646800717),
            (AttnKind::Hca, 1, 128, 0, 0.0802),
            (AttnKind::Hca, 8, 512, 1024, 0.5717204610218155),
        ];
        for &(kind, b, s, prefix, expected) in ctx_cases {
            let result = dsv4_pro_op(kind)
                .query_context(&db, b, s, prefix)
                .expect("empirical context query");
            approx(result.latency_ms, expected);
            assert_eq!(result.source, Source::Empirical);
        }

        let gen_cases: &[(AttnKind, u32, u32, f64)] = &[
            (AttnKind::Csa, 16, 385, 0.11423651225442899),
            (AttnKind::Csa, 16, 200, 0.11330535071492717),
            // beyond-range s_total: boundary util frozen, decode SOL carries it
            (AttnKind::Csa, 13, 100_000, 0.16844612496931166),
            (AttnKind::Hca, 16, 385, 0.07241533398068029),
        ];
        for &(kind, b, s, expected) in gen_cases {
            let result = dsv4_pro_op(kind)
                .query_generation(&db, b, s)
                .expect("empirical generation query");
            approx(result.latency_ms, expected);
            assert_eq!(result.source, Source::Empirical);
        }
    }

    /// The genuine prefix-interpolation branch (>= 2 collected prefix keys
    /// bracketing the query -> depth-3 grid over `(prefix, s, b)`), on the
    /// shipped b200_sxm/sglang/0.5.14 tables (28 prefix keys, 0..1048575).
    /// Same Python oracle command as above with '0.5.14':
    ///
    /// ```text
    /// for b, s, p, cr in [(8,512,1024,4), (8,700,1000,4), (8,512,2097150,4), (8,700,1000,128)]: ...
    /// ```
    ///
    /// Covers: exact 3-axis hit, off-grid (prefix, s) IDW, HCA, and a prefix
    /// beyond the collected range falling back to the p0-anchor branch
    /// (query `(s + prefix, b)` = (2097662, 8)).
    #[test]
    fn dsv4_empirical_prefix_interp_matches_python_oracles() {
        let systems_root = systems_root();
        let data_root = systems_root.join("data/b200_sxm/sglang/0.5.14");
        if !data_root.join("dsv4_csa_context_module_perf.parquet").exists() {
            return; // git-lfs data not materialized
        }
        let mut db = PerfDatabase::load(&systems_root, "b200_sxm", "sglang", "0.5.14")
            .expect("b200_sxm/sglang/0.5.14 must load");
        db.database_mode = DatabaseMode::Empirical;

        let cases: &[(AttnKind, u32, u32, u32, f64)] = &[
            (AttnKind::Csa, 8, 512, 1024, 1.0183),
            (AttnKind::Csa, 8, 700, 1000, 1.429314779309641),
            (AttnKind::Csa, 8, 512, 2_097_150, 32.64516668240808),
            (AttnKind::Hca, 8, 700, 1000, 0.836433276456113),
        ];
        for &(kind, b, s, prefix, expected) in cases {
            let result = dsv4_pro_op(kind)
                .query_context(&db, b, s, prefix)
                .expect("empirical context query");
            approx(result.latency_ms, expected);
            assert_eq!(result.source, Source::Empirical);
        }
    }

    /// HYBRID with silicon data present must stay on the silicon path; a kv
    /// dtype with NO collected slice (bfloat16 — only fp8 ships) must fall
    /// through the empirical path and surface the terminal
    /// EmpiricalNotImplemented miss, never a fabricated value (Python: the
    /// silicon miss falls to `get_empirical`, whose own typed miss raises
    /// `EmpiricalNotImplementedError`).
    #[test]
    fn dsv4_hybrid_silicon_passthrough_and_terminal_miss() {
        let root = b200_sglang_root();
        if !root.join("dsv4_csa_context_module_perf.parquet").exists() {
            return; // git-lfs data not materialized
        }
        let mut db = PerfDatabase::load(&systems_root(), "b200_sxm", "sglang", "0.5.10")
            .expect("b200_sxm/sglang/0.5.10 must load");
        db.database_mode = DatabaseMode::Hybrid;

        // Data present: silicon passthrough (exact grid point, same value as
        // the silicon parity test).
        let result = dsv4_pro_op(AttnKind::Csa)
            .query_context(&db, 8, 512, 0)
            .expect("hybrid context query");
        approx(result.latency_ms, 0.9819);
        assert_eq!(result.source, Source::Silicon);

        // kv=bfloat16 is not collected: typed silicon miss -> empirical miss.
        let mut op = dsv4_pro_op(AttnKind::Csa);
        op.kv_cache_dtype = KvCacheQuantMode::Bfloat16;
        let ctx = op.query_context(&db, 8, 512, 0);
        assert!(
            matches!(ctx, Err(AicError::EmpiricalNotImplemented(_))),
            "got {ctx:?}"
        );
        let gen = op.query_generation(&db, 16, 385);
        assert!(
            matches!(gen, Err(AicError::EmpiricalNotImplemented(_))),
            "got {gen:?}"
        );
    }

    /// DeepSeek-V4-Pro MegaMoE routed-module op mirroring the model builder
    /// (`models/deepseek_v4.py::_moe_ops` with `use_megamoe`): full
    /// moe_inter_size, defaults for source_policy / pre_dispatch / fused
    /// shared experts / kernel identity.
    fn megamoe_op(is_context: bool, distribution: &str) -> Dsv4MegaMoeOp {
        Dsv4MegaMoeOp {
            name: if is_context { "context_megamoe" } else { "generation_megamoe" }.into(),
            scale_factor: 1.0,
            hidden_size: 7168,
            inter_size: 3072,
            topk: 6,
            num_experts: 384,
            moe_tp_size: 1,
            moe_ep_size: 8,
            quant_mode: MoeQuantMode::W4a8Mxfp4Mxfp8,
            workload_distribution: distribution.into(),
            is_context,
            source_policy: "random".into(),
            pre_dispatch: "sglang_jit".into(),
            num_fused_shared_experts: 0,
            kernel_source: "deepgemm_megamoe".into(),
            kernel_dtype: "fp8_fp4".into(),
        }
    }

    fn gb200_db() -> Option<PerfDatabase> {
        let systems_root = systems_root();
        let data_root = systems_root.join("data/gb200/sglang/0.5.10");
        if !data_root.join("dsv4_megamoe_module_perf.parquet").exists() {
            return None; // git-lfs data not materialized
        }
        Some(
            PerfDatabase::load(&systems_root, "gb200", "sglang", "0.5.10")
                .expect("gb200/sglang/0.5.10 must load"),
        )
    }

    /// SILICON parity anchors on the shipped gb200/sglang/0.5.10 MegaMoE
    /// table. Python oracle:
    ///
    /// ```text
    /// uv run --no-sync python -c "
    /// from aiconfigurator.sdk import common
    /// from aiconfigurator.sdk.perf_database import get_database
    /// from aiconfigurator.sdk.operations.dsv4 import DeepSeekV4MegaMoEModule as M
    /// from aiconfigurator.sdk.common import DatabaseMode
    /// db = get_database('gb200', 'sglang', '0.5.10')
    /// base = dict(hidden_size=7168, inter_size=3072, topk=6, num_experts=384,
    ///             moe_tp_size=1, moe_ep_size=8,
    ///             quant_mode=common.MoEQuantMode.w4a8_mxfp4_mxfp8)
    /// for tok, dist, ctx in [(1024,'balanced',True), (3000,'power_law_1.2',True),
    ///                        (100,'balanced',True), (64,'power_law_1.01',False),
    ///                        (2000,'balanced',False)]:
    ///     print(float(M._query_megamoe_table(db, num_tokens=tok,
    ///           workload_distribution=dist, is_context=ctx,
    ///           database_mode=DatabaseMode.SILICON, **base)))"
    /// ```
    ///
    /// Covers: exact collected hit, off-grid in-range lerp, below-range and
    /// beyond-range boundary util-hold (linear token-proxy SOL), both phases.
    #[test]
    fn megamoe_silicon_matches_python_oracles() {
        let Some(db) = gb200_db() else { return };
        assert_eq!(db.database_mode, DatabaseMode::Silicon);

        let cases: &[(bool, &str, u32, f64)] = &[
            // exact collected hit (context, tokens=1024, balanced, ep=8)
            (true, "balanced", 1024, 0.508394),
            // off-grid in-range lerp (context, tokens=3000 between 2048/4096)
            (true, "power_law_1.2", 3000, 2.01460530859375),
            // below the collected range (context, tokens=100 < 1024)
            (true, "balanced", 100, 0.0496478515625),
            // exact collected hit (generation, tokens=64)
            (false, "power_law_1.01", 64, 0.312203),
            // beyond the collected range (generation, tokens=2000 > 512)
            (false, "balanced", 2000, 1.5932929687500001),
        ];
        for &(is_context, dist, tokens, expected) in cases {
            let result = megamoe_op(is_context, dist)
                .query(&db, tokens)
                .expect("silicon megamoe query");
            approx(result.latency_ms, expected);
            assert_eq!(result.source, Source::Silicon);
        }
    }

    /// Mode + miss contract, mirroring Python `_query_megamoe_table`:
    /// - EMPIRICAL -> typed PerfDataNotAvailableError (the op has NO
    ///   empirical path);
    /// - an absent shape is a typed miss under SILICON;
    /// - HYBRID == SILICON: hit passes through, a miss propagates RAW
    ///   (never converted into an empirical estimate).
    /// - `uniform` normalizes to `balanced` (Python ctor normalization).
    #[test]
    fn megamoe_mode_contract_and_miss() {
        let Some(mut db) = gb200_db() else { return };

        // EMPIRICAL -> error, Python message mirrored.
        db.database_mode = DatabaseMode::Empirical;
        let err = megamoe_op(true, "balanced").query(&db, 1024).unwrap_err();
        assert!(err.is_missing_perf_data(), "got {err:?}");
        assert!(
            err.to_string()
                .contains("DSv4 MegaMoE module only supports measured SILICON data"),
            "unexpected message: {err}"
        );

        // SILICON miss: absent shape is a typed miss naming the key.
        db.database_mode = DatabaseMode::Silicon;
        let mut op = megamoe_op(true, "balanced");
        op.num_experts = 999;
        let err = op.query(&db, 1024).unwrap_err();
        assert!(err.is_missing_perf_data(), "got {err:?}");
        assert!(
            err.to_string().contains("No DSv4 MegaMoE context module data")
                && err.to_string().contains("num_experts=999"),
            "unexpected message: {err}"
        );

        // HYBRID: hit == silicon value; miss propagates raw (same typed
        // error, no empirical conversion).
        db.database_mode = DatabaseMode::Hybrid;
        let result = megamoe_op(true, "balanced")
            .query(&db, 1024)
            .expect("hybrid megamoe hit");
        approx(result.latency_ms, 0.508394);
        assert_eq!(result.source, Source::Silicon);
        let err = op.query(&db, 1024).unwrap_err();
        assert!(err.is_missing_perf_data(), "got {err:?}");

        // `uniform` -> `balanced` (Python `_normalize_distribution`).
        db.database_mode = DatabaseMode::Silicon;
        let result = megamoe_op(true, "uniform")
            .query(&db, 1024)
            .expect("uniform must normalize to balanced");
        approx(result.latency_ms, 0.508394);
    }

    /// The exact externally-tagged JSON the Python emitter
    /// (`engine.py::_dsv4_megamoe`) produces must decode into the op —
    /// pins the wire field names.
    #[test]
    fn megamoe_python_wire_json_decodes() {
        let json = serde_json::json!({
            "Dsv4MegaMoe": {
                "name": "context_megamoe",
                "scale_factor": 61.0,
                "hidden_size": 7168,
                "inter_size": 3072,
                "topk": 6,
                "num_experts": 384,
                "moe_tp_size": 1,
                "moe_ep_size": 8,
                "quant_mode": "w4a8_mxfp4_mxfp8",
                "workload_distribution": "balanced",
                "is_context": true,
                "source_policy": "random",
                "pre_dispatch": "sglang_jit",
                "num_fused_shared_experts": 0,
                "kernel_source": "deepgemm_megamoe",
                "kernel_dtype": "fp8_fp4",
            }
        });
        let op: crate::operators::op::Op = serde_json::from_value(json).expect("decode");
        let crate::operators::op::Op::Dsv4MegaMoe(op) = op else {
            panic!("expected Dsv4MegaMoe variant");
        };
        assert_eq!(op.quant_mode, MoeQuantMode::W4a8Mxfp4Mxfp8);
        assert_eq!(op.scale_factor, 61.0);
        assert!(op.is_context);
    }

    /// `cp_size` / `window_size` are absent from every opspec the Python
    /// emitter produces today (`engine.py::_reject_cp` still guards CP) —
    /// they must default to 1 / None so existing specs keep the plain non-CP
    /// lookup.
    #[test]
    fn cp_fields_default_in_serde() {
        let mut v = serde_json::to_value(dsv4_cp_op(AttnKind::Csa, 3, Some(2048))).expect("serialize");
        let obj = v.as_object_mut().expect("object");
        obj.remove("cp_size");
        obj.remove("window_size");
        let de: Dsv4ModuleOp = serde_json::from_value(v).expect("deserialize");
        assert_eq!(de.cp_size, 1);
        assert_eq!(de.window_size, None);
    }
}
