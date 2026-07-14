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
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::communication::NcclOp;
use crate::perf_database::dsv4::{dsv4_dims, AttnKind, Dsv4SolDims};
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
    /// includes the CSA topk DELTA correction).
    fn module_base(
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
        // isl - prefix`, computed in `run_context_ops`); the context CSVs
        // collected to date carry a single `step=0` anchor, so `prefix=0`
        // collapses that level exactly and `prefix>0` resolves via util-hold
        // with the prefix-aware SOL carrying the effect — matching Python.
        let raw = self.module_base(db, batch_size, isl, prefix)?;
        Ok(PerformanceResult::new(raw, Source::Silicon)
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
            &mut |per_card| self.module_base(db, batch_size, per_card, prefix),
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

    pub fn query_generation(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        // No fmha argument: the generation table keys on kv dtype only and the
        // SOL dtype is derived from kv inside `query_generation` (PR #1337).
        // `self.fmha_quant_mode` stays on the op for the context path.
        let latency = db.dsv4.query_generation(
            &db.system_spec,
            self.attn_kind,
            batch_size,
            s,
            self.num_heads,
            self.kv_cache_dtype,
            self.gemm_quant_mode,
            &self.architecture,
            Some(self.sol_dims()),
        )?;
        Ok(PerformanceResult::new(latency, Source::Silicon)
            .clamp_non_negative()
            .scaled(self.scale_factor))
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
            .join("src/aiconfigurator/systems");
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
            .join("src/aiconfigurator/systems/data/gb200/sglang/0.5.10");
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
