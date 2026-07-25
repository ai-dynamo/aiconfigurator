// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 attention module perf tables.
//!
//! Four primary module CSVs distinguished by `(attn_kind, mode)`:
//! - `dsv4_csa_context_module_perf.txt` — CSA (compressed-sparse) context
//! - `dsv4_hca_context_module_perf.txt` — HCA (hybrid-causal) context
//! - `dsv4_csa_generation_module_perf.txt`
//! - `dsv4_hca_generation_module_perf.txt`
//!
//! Each file loads from an ordered, shared-layer-aware source list (see
//! [`PerfSource`]).
//!
//! ## Indexing
//!
//! Mirrors Python `load_context_dsv4_kind_module_data` /
//! `load_generation_dsv4_kind_module_data` (SCHEME A): the head axis is the
//! rank-LOCAL head count straight from the CSV `num_heads` column; the CSV
//! `tp_size` column is collapsed at load (see the loader note below).
//!
//! ## Resolution (perf_interp v2)
//!
//! Queries resolve on the RAW tables through the shared engine, mirroring
//! Python `operations/dsv4.py`:
//! - context: 3-axis Grid RAW `[prefix(step)][isl][batch]` — the step axis is
//!   KEPT (a prefix beyond the collected range is util-hold with the
//!   prefix-aware SOL carrying the effect, replacing the legacy
//!   fold-to-last-anchor + robust batch-scaling lookup);
//! - generation: 2-axis Grid RAW `[batch][s_total]` where
//!   `s_total = isl + step` (decode is q_len=1 with past_kv=step).
//!
//! All four primary CSVs share the DSA module column layout. Data is
//! collected only on TRT-LLM / SGLang today; loaders surface a clean error for
//! backends without DSV4 data.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use super::dsa::{bs_slice, lookup_2d, SparseGrid};
use super::perf_interp::{self, Node, OpInterpConfig};
use super::{kernel_source_ok, resolve_op_sources};
use crate::perf_database::parquet_loader::PerfReader;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnKind {
    Csa,
    Hca,
}

impl AttnKind {
    /// The compress_ratio each split file was collected at. Python keeps
    /// compress_ratio as a dict key inside one merged table; the Rust port
    /// keeps the files separate, so the kind IS the ratio.
    pub(crate) fn compress_ratio(self) -> i64 {
        match self {
            AttnKind::Csa => 4,
            AttnKind::Hca => 128,
        }
    }
}

// head -> step -> isl -> batch -> latency
//
// NOTE: the CSV `tp_size` column is intentionally COLLAPSED at load time, NOT
// kept as an interpolation axis. Python's loaders (`load_*_dsv4_kind_module_data`)
// key only on `(num_heads, compress_ratio, step, isl, batch)` and never on
// `tp_size`, so when several tp_size rows share a cell the last parquet row
// wins (a plain dict overwrite). The collected files are gemm-then-tp-ascending,
// so the survivor is the largest measured tp. We reproduce that by inserting in
// file order and overwriting, dropping the tp axis entirely. The head axis here
// is the CSV `num_heads` value {64, 128}; the query resolves the model's
// rank-LOCAL head count against it (see `resolve_head_key`), mirroring Python's
// `_dsv4_resolve_head_key`.
type ByBatch = BTreeMap<u32, f64>;
type ByIsl = BTreeMap<u32, ByBatch>;
type ByStep = BTreeMap<u32, ByIsl>;
type ByNative = BTreeMap<u32, ByStep>;

pub struct Dsv4Table {
    /// Ordered, priority-sorted sources for each of the four DSV4 module perf
    /// files (shared-layer aware; see [`PerfSource`]). Single-primary,
    /// no-filter by default (`Dsv4Table::new`).
    csa_context_sources: Vec<PerfSource>,
    hca_context_sources: Vec<PerfSource>,
    csa_generation_sources: Vec<PerfSource>,
    hca_generation_sources: Vec<PerfSource>,
    /// CSA topk DELTA calibration (`dsv4_csa_topk_calib_perf.parquet`).
    /// Same source resolution as the module files; an absent file loads as
    /// `None` and the correction is a no-op (Python parity).
    topk_calib_sources: Vec<PerfSource>,
    /// Sparse-kernel table (`dsv4_paged_mqa_logits_module_perf.parquet`) for
    /// the CP prefill composition's mqa full/per-card deltas. Same source
    /// resolution as the module files (Python `_load_sparse` runs the file
    /// through `database._build_op_sources` too); an absent file loads as
    /// `None` and the operator fails loud.
    paged_mqa_sources: Vec<PerfSource>,
    csa_context: OnceLock<Result<ModuleNodes, AicError>>,
    hca_context: OnceLock<Result<ModuleNodes, AicError>>,
    csa_generation: OnceLock<Result<ModuleNodes, AicError>>,
    hca_generation: OnceLock<Result<ModuleNodes, AicError>>,
    topk_calib: OnceLock<Result<Option<TopkCalib>, AicError>>,
    paged_mqa: OnceLock<Result<Option<SparseKernelNodes>, AicError>>,
}

/// Raw per-key nested grids straight from the parquet (loader output).
struct ModuleGrids {
    by_keys: BTreeMap<ModuleKey, ByNative>,
}

/// Engine-ready tables: per (key, head), the phase-shaped `Node`
/// (context: `[step][isl][batch]`; generation: `[batch][s_total]`).
struct ModuleNodes {
    by_keys: BTreeMap<ModuleKey, BTreeMap<u32, Node>>,
}

/// Table key mirroring the Python loaders (PR #1337 alignment):
/// - context modules key `[fmha][kv][gemm]` (`load_context_dsv4_kind_module_data`);
/// - generation modules key `[kv][gemm]` only — the `mla_dtype` column is
///   ignored at load (`load_generation_dsv4_kind_module_data`), so
///   `fmha_quant` is the empty sentinel for generation keys.
/// Neither keys on `architecture` (Python never reads that column).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleKey {
    fmha_quant: String,
    kv_quant: String,
    gemm_quant: String,
}

/// DeepSeek-V4 structural dims used as the DEFAULT source for the analytic
/// SOL when the caller does not supply per-op dims. Python receives these
/// from the model config (`DeepSeekV4Config`) via the op fields; the Rust
/// opspec now carries them too (`Dsv4ModuleOp` -> [`Dsv4SolDims`]), so this
/// pinned table only backs old specs and direct perf-database queries.
///
/// NOTE: DeepSeek-V4-Pro and DeepSeek-V4-Flash share the architecture string
/// `DeepseekV4ForCausalLM` but differ in shape (Flash: hidden 4096,
/// q_lora 1024, o_groups 8, index_topk 512, native heads 64). The pinned
/// table carries the PRO dims — op-level queries override it from the spec.
pub(crate) struct Dsv4Dims {
    pub(crate) hidden_size: i64,
    pub(crate) q_lora_rank: i64,
    pub(crate) o_lora_rank: i64,
    pub(crate) head_dim: i64,
    pub(crate) rope_head_dim: i64,
    pub(crate) index_n_heads: i64,
    pub(crate) index_head_dim: i64,
    pub(crate) index_topk: i64,
    pub(crate) window_size: i64,
    pub(crate) o_groups: i64,
    pub(crate) native_heads: i64,
}

const DSV4_PRO_DIMS: Dsv4Dims = Dsv4Dims {
    hidden_size: 7168,
    q_lora_rank: 1536,
    o_lora_rank: 1024,
    head_dim: 512,
    rope_head_dim: 64,
    index_n_heads: 64,
    index_head_dim: 128,
    index_topk: 1024,
    window_size: 128,
    o_groups: 16,
    native_heads: 128,
};

pub(crate) fn dsv4_dims(_architecture: &str) -> &'static Dsv4Dims {
    // Only DeepseekV4ForCausalLM exists today; see the Pro-vs-Flash note on
    // `Dsv4Dims`.
    &DSV4_PRO_DIMS
}

/// SOL-ready structural dims, matching the parameter list Python's
/// `_deepseek_v4_attention_sol` receives from the op fields (model config).
/// `local_o_groups` is the rank-LOCAL group count — Python's op carries it
/// pre-divided (`max(1, o_groups // tp_size)` in `models/deepseek_v4.py`).
#[derive(Clone, Copy, Debug)]
pub struct Dsv4SolDims {
    pub(crate) hidden_size: i64,
    pub(crate) q_lora_rank: i64,
    pub(crate) o_lora_rank: i64,
    pub(crate) head_dim: i64,
    pub(crate) rope_head_dim: i64,
    pub(crate) index_n_heads: i64,
    pub(crate) index_head_dim: i64,
    pub(crate) index_topk: i64,
    pub(crate) window_size: i64,
    pub(crate) local_o_groups: i64,
}

impl Dsv4SolDims {
    /// Resolve from the pinned per-architecture table (the pre-override
    /// behaviour): the rank-local o_groups is derived from
    /// `tp = native_heads / local_heads`, mirroring Python's
    /// `max(1, o_groups // tp)` with the pinned totals.
    pub(crate) fn from_pinned(dims: &Dsv4Dims, local_heads: i64) -> Self {
        let tp = (dims.native_heads / local_heads.max(1)).max(1);
        Dsv4SolDims {
            hidden_size: dims.hidden_size,
            q_lora_rank: dims.q_lora_rank,
            o_lora_rank: dims.o_lora_rank,
            head_dim: dims.head_dim,
            rope_head_dim: dims.rope_head_dim,
            index_n_heads: dims.index_n_heads,
            index_head_dim: dims.index_head_dim,
            index_topk: dims.index_topk,
            window_size: dims.window_size,
            local_o_groups: (dims.o_groups / tp).max(1),
        }
    }
}

impl Dsv4Table {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each DSV4 file falls back to its
    /// primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let csa_context_sources =
            resolve_op_sources(perf_db_sources, "dsv4_csa_context_module_perf.parquet", &data_root);
        let hca_context_sources =
            resolve_op_sources(perf_db_sources, "dsv4_hca_context_module_perf.parquet", &data_root);
        let csa_generation_sources = resolve_op_sources(
            perf_db_sources,
            "dsv4_csa_generation_module_perf.parquet",
            &data_root,
        );
        let hca_generation_sources = resolve_op_sources(
            perf_db_sources,
            "dsv4_hca_generation_module_perf.parquet",
            &data_root,
        );
        let topk_calib_sources =
            resolve_op_sources(perf_db_sources, "dsv4_csa_topk_calib_perf.parquet", &data_root);
        let paged_mqa_sources = resolve_op_sources(
            perf_db_sources,
            "dsv4_paged_mqa_logits_module_perf.parquet",
            &data_root,
        );
        Self {
            csa_context_sources,
            hca_context_sources,
            csa_generation_sources,
            hca_generation_sources,
            topk_calib_sources,
            paged_mqa_sources,
            csa_context: OnceLock::new(),
            hca_context: OnceLock::new(),
            csa_generation: OnceLock::new(),
            hca_generation: OnceLock::new(),
            topk_calib: OnceLock::new(),
            paged_mqa: OnceLock::new(),
        }
    }

    /// Context-DSV4 latency at `lookup_s = isl` (the new-token count).
    ///
    /// Mirrors Python `ContextDeepSeekV4AttentionModule._query_context_attn_table`
    /// (SILICON path): resolve the `(quant, arch)` key, resolve the model's
    /// rank-LOCAL head count (`local_heads = native // tp`) against the CSV
    /// head keys via [`resolve_head_key`], then one 3-axis Grid RAW engine
    /// query over `(prefix, isl, batch)` — the step axis is KEPT. The context
    /// CSVs collected to date carry a single `step=0` anchor, so `prefix=0`
    /// collapses that level exactly and `prefix>0` is out-of-range util-hold
    /// with the prefix-aware SOL carrying the effect (matching Python).
    /// `sol_dims`: per-op structural dims for the analytic SOL (from the
    /// opspec, mirroring Python's op fields). `None` falls back to the pinned
    /// per-architecture table ([`Dsv4SolDims::from_pinned`]) — the
    /// pre-override behaviour kept for direct perf-database queries.
    #[allow(clippy::too_many_arguments)]
    pub fn query_context(
        &self,
        spec: &SystemSpec,
        attn_kind: AttnKind,
        b: u32,
        isl: u32,
        local_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        prefix: u32,
        sol_dims: Option<Dsv4SolDims>,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_context()?,
            AttnKind::Hca => self.load_hca_context()?,
        };
        let node = select_resolved(grids, Some(fmha_quant), kv_quant, gemm_quant, local_heads)?;

        let dims = sol_dims
            .unwrap_or_else(|| Dsv4SolDims::from_pinned(dsv4_dims(architecture), local_heads as i64));
        let cr = attn_kind.compress_ratio();
        let heads = local_heads as i64;
        // Engine coordinates are (prefix, seq, batch); Python's sol_fn is
        // lambda p, s, b: get_sol(b, s, p).
        let sol = move |c: &[f64]| {
            dsv4_attention_sol_ms(
                spec,
                &dims,
                cr,
                true,
                kv_quant,
                fmha_quant,
                gemm_quant,
                c[2] as i64, // b
                c[1] as i64, // s
                c[0] as i64, // prefix
                heads,
            )
        };
        let cfg = OpInterpConfig::grid(&["prefix", "seq_len", "batch"], &sol);
        let latency = perf_interp::query(&cfg, node, &[prefix as f64, isl as f64, b as f64])?;
        // Mirrors Python `ContextDeepSeekV4AttentionModule` get_silicon
        // (operations/dsv4.py): for CSA (compress_ratio==4) ONLY, subtract the
        // measured topK DELTA = flat_ms - top_last_ms at the ORIGINAL query
        // point (prefix, s, b) and clamp at 0. HCA (cr==128) is left untouched.
        if attn_kind == AttnKind::Csa {
            return self.topk_corrected(latency, TopkPhase::Context, prefix, isl, b);
        }
        Ok(latency)
    }

    /// Generation-DSV4 latency. `sequence_tokens = isl + step` (absolute KV
    /// length). Mirrors Python
    /// `GenerationDeepSeekV4AttentionModule._query_generation_attn_table`
    /// (SILICON path): one 2-axis Grid RAW engine query over the
    /// `{b: {s_total}}` table. The DSV4 generation table is RAGGED (e.g.
    /// `s_total=385` measured only at some batches); the engine's
    /// single-survivor SOL-ratio correction / util-hold replaces the legacy
    /// batch-scaling fallback.
    /// `sol_dims`: see [`Dsv4Table::query_context`].
    #[allow(clippy::too_many_arguments)]
    pub fn query_generation(
        &self,
        spec: &SystemSpec,
        attn_kind: AttnKind,
        b: u32,
        sequence_tokens: u32,
        local_heads: u32,
        kv_quant: KvCacheQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        sol_dims: Option<Dsv4SolDims>,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_generation()?,
            AttnKind::Hca => self.load_hca_generation()?,
        };
        // PR #1337: decode attention compute dtype follows the kv-cache dtype;
        // the fmha label is inert for generation (the table keys on kv dtype).
        // Derive the SOL dtype from kv so label changes cannot move decode SOL
        // — mirrors `GenerationDeepSeekV4AttentionModule` (operations/dsv4.py).
        // The table key carries no fmha level at all (see `ModuleKey`), so this
        // query takes no fmha parameter.
        let fmha_quant = if kv_quant == KvCacheQuantMode::Fp8 {
            FmhaQuantMode::Fp8
        } else {
            FmhaQuantMode::Bfloat16
        };
        let node = select_resolved(grids, None, kv_quant, gemm_quant, local_heads)?;

        let dims = sol_dims
            .unwrap_or_else(|| Dsv4SolDims::from_pinned(dsv4_dims(architecture), local_heads as i64));
        let cr = attn_kind.compress_ratio();
        let heads = local_heads as i64;
        // Engine coordinates are (batch, seq); Python's sol_fn is
        // lambda b, s: get_sol(b, s) with prefix=0 and is_context=False.
        let sol = move |c: &[f64]| {
            dsv4_attention_sol_ms(
                spec,
                &dims,
                cr,
                false,
                kv_quant,
                fmha_quant,
                gemm_quant,
                c[0] as i64, // b
                c[1] as i64, // s_total
                0,
                heads,
            )
        };
        let cfg = OpInterpConfig::grid(&["batch", "seq_len"], &sol);
        let latency = perf_interp::query(&cfg, node, &[b as f64, sequence_tokens as f64])?;
        // Mirrors Python `GenerationDeepSeekV4AttentionModule` get_silicon
        // (operations/dsv4.py): subtract the topK DELTA for CSA (cr==4) only.
        // Decode is q_len=1 with past_kv = s_total - 1, so the DELTA keys at
        // (prefix = max(s_total - 1, 0), isl = 1, bs = b).
        if attn_kind == AttnKind::Csa {
            return self.topk_corrected(latency, TopkPhase::Generation, sequence_tokens.saturating_sub(1), 1, b);
        }
        Ok(latency)
    }

    fn load_csa_context(&self) -> Result<&ModuleNodes, AicError> {
        let cell = self.csa_context.get_or_init(|| {
            load_module_parquet(&self.csa_context_sources, true).map(context_nodes)
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_context(&self) -> Result<&ModuleNodes, AicError> {
        let cell = self.hca_context.get_or_init(|| {
            load_module_parquet(&self.hca_context_sources, true).map(context_nodes)
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_csa_generation(&self) -> Result<&ModuleNodes, AicError> {
        let cell = self.csa_generation.get_or_init(|| {
            load_module_parquet(&self.csa_generation_sources, false).map(generation_nodes)
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_generation(&self) -> Result<&ModuleNodes, AicError> {
        let cell = self.hca_generation.get_or_init(|| {
            load_module_parquet(&self.hca_generation_sources, false).map(generation_nodes)
        });
        cell.as_ref().map_err(clone_err)
    }

    /// Apply the CSA topK DELTA correction to a module latency, mirroring the
    /// Python apply sites in `operations/dsv4.py` (context and generation
    /// `get_silicon`): `latency = max(0, latency - DELTA(prefix, isl, bs))`.
    /// No-op when the correction is disabled (`AIC_DSV4_TOPK_CORRECTION=0`)
    /// or when no calibration file exists — Python gates the calib LOAD behind
    /// the env var too, so keep the load lazy-skipped when disabled.
    fn topk_corrected(
        &self,
        latency: f64,
        phase: TopkPhase,
        prefix: u32,
        isl: u32,
        bs: u32,
    ) -> Result<f64, AicError> {
        if !topk_correction_enabled() {
            return Ok(latency);
        }
        let exact = self.load_topk_calib()?.map(|calib| match phase {
            TopkPhase::Context => &calib.exact_v1,
            TopkPhase::Generation => &calib.exact_v2,
        });
        Ok(apply_topk_delta(latency, exact, prefix, isl, bs))
    }

    /// Lazy-load the topK DELTA calibration (Python `_get_dsv4_topk_calib`,
    /// which caches on the database object). `Ok(None)` when every source
    /// file is absent or no usable rows exist.
    fn load_topk_calib(&self) -> Result<Option<&TopkCalib>, AicError> {
        let cell = self
            .topk_calib
            .get_or_init(|| load_topk_calib_parquet(&self.topk_calib_sources));
        match cell {
            Ok(calib) => Ok(calib.as_ref()),
            Err(err) => Err(clone_err(err)),
        }
    }

    /// Lazy-load the paged_mqa_logits sparse-kernel table (Python
    /// `load_dsv4_sparse_kernel_data` under the `"paged_mqa_logits"` key of
    /// `_dsv4_sparse_kernel_data`). `Ok(None)` when every source file is
    /// absent (Python: `_read_filtered_rows` -> None).
    fn load_paged_mqa(&self) -> Result<Option<&SparseKernelNodes>, AicError> {
        let cell = self
            .paged_mqa
            .get_or_init(|| load_sparse_kernel_parquet(&self.paged_mqa_sources));
        match cell {
            Ok(nodes) => Ok(nodes.as_ref()),
            Err(err) => Err(clone_err(err)),
        }
    }

    /// Sparse-kernel latency at `(b, isl, past_kv)` on the paged_mqa_logits
    /// table. Mirror of Python
    /// `ContextDeepSeekV4AttentionModule._lookup_sparse_kernel` with the
    /// kernel FIXED to `"paged_mqa_logits"` — the quadratic pair-count SOL
    /// below is valid only for the causal indexer logits; the windowed
    /// sidecars (hca_attn / csa_attn) must NOT route here (Python raises on
    /// any other kernel name; Rust simply has no other entry point).
    ///
    /// Table shape `[native_heads][tp][past_kv][isl][bs]`; `tp` falls back to
    /// 1 (the kernel is collected at tp=1 only and its work is
    /// TP-independent). Resolves on the perf_interp 3-axis Grid
    /// `(past_kv, seq_len, batch)` with SOL `b * (past * isl + isl^2 / 2)`.
    /// `Ok(None)` when the table/slice is absent or the engine cannot anchor
    /// (Python catches `InterpolationDataNotAvailableError` -> None) or the
    /// resolved value is non-finite.
    pub fn query_paged_mqa_logits(
        &self,
        b: u32,
        isl: u32,
        past_kv: u32,
        tp_size: u32,
        native_heads: u32,
    ) -> Result<Option<f64>, AicError> {
        let Some(nodes) = self.load_paged_mqa()? else {
            return Ok(None);
        };
        let Some(per_tp) = nodes.by_heads.get(&native_heads) else {
            return Ok(None);
        };
        let Some(node) = per_tp.get(&tp_size).or_else(|| per_tp.get(&1)) else {
            return Ok(None);
        };
        let sol = |c: &[f64]| c[2] * (c[0] * c[1] + c[1] * c[1] / 2.0);
        let cfg = OpInterpConfig::grid(&["past_kv", "seq_len", "batch"], &sol);
        match perf_interp::query(&cfg, node, &[past_kv as f64, isl as f64, b as f64]) {
            Ok(latency) if latency.is_finite() => Ok(Some(latency)),
            Ok(_) | Err(_) => Ok(None),
        }
    }

    /// Collected context-module points `(prefix, s, b) -> latency` for the
    /// operator-layer util-calibration grid (Python
    /// `_query_context_attn_table::get_empirical`'s `_slice()`:
    /// `require_data_slice(data, fmha, kv, gemm)` -> head resolution ->
    /// `require_data_slice(quant_data, head, compress_ratio)`). Slices exactly
    /// like the silicon query ([`select_resolved`]). Missing slice / empty
    /// node is a typed miss.
    pub fn context_points(
        &self,
        attn_kind: AttnKind,
        local_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_context()?,
            AttnKind::Hca => self.load_hca_context()?,
        };
        let node = select_resolved(grids, Some(fmha_quant), kv_quant, gemm_quant, local_heads)?;
        let points = perf_interp::node_points(node);
        if points.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "DSV4 context module data empty for local_heads={local_heads}, \
                 attn_kind={attn_kind:?}"
            )));
        }
        Ok(points)
    }

    /// Collected generation-module points `(b, s_total) -> latency` for the
    /// operator-layer util-calibration grid (Python
    /// `_query_generation_attn_table::get_empirical`'s `_slice()`:
    /// `require_data_slice(data, kv, gemm)` -> head resolution). The
    /// generation table keys `[kv][gemm]` with no fmha level (PR #1337),
    /// matching the silicon query's `select_resolved(fmha=None, ...)`.
    pub fn generation_points(
        &self,
        attn_kind: AttnKind,
        local_heads: u32,
        kv_quant: KvCacheQuantMode,
        gemm_quant: GemmQuantMode,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_generation()?,
            AttnKind::Hca => self.load_hca_generation()?,
        };
        let node = select_resolved(grids, None, kv_quant, gemm_quant, local_heads)?;
        let points = perf_interp::node_points(node);
        if points.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "DSV4 generation module data empty for local_heads={local_heads}, \
                 attn_kind={attn_kind:?}"
            )));
        }
        Ok(points)
    }

    /// Absolute top_last topk latency at `(isl, step)` for batch `b`. CP
    /// composition only — reads the RAW top_last rows the calib loader
    /// retains alongside the DELTA pairs (Python `_csa_topk_top_last` /
    /// `_load_csa_topk_top_last`; single load, no re-read of the parquet).
    /// The num_heads filter applies only when the CSV carries the column
    /// (Python: `df[df["num_heads"] == native_heads] if "num_heads" in df`).
    /// Missing calib / head slice / batch grid -> `Ok(None)` (the operator
    /// fails loud); an `isl` beyond the collected grid errors via
    /// [`lookup_2d`] — same fail-loud as Python, which reuses
    /// `ContextDSAModule._lookup_2d` here.
    pub fn csa_topk_top_last(
        &self,
        isl: u32,
        step: u32,
        native_heads: u32,
        b: u32,
    ) -> Result<Option<f64>, AicError> {
        let Some(calib) = self.load_topk_calib()? else {
            return Ok(None);
        };
        let Some(grid) = calib
            .top_last
            .get(&Some(native_heads))
            .or_else(|| calib.top_last.get(&None))
        else {
            return Ok(None);
        };
        let Some(bs_grid) = bs_slice(grid, b) else {
            return Ok(None);
        };
        lookup_2d(bs_grid, isl, step)
    }
}

/// Convert loaded grids into context-phase engine tables: per (key, head),
/// a `[step][isl][batch]` Node (the raw nesting, step axis KEPT).
fn context_nodes(grids: ModuleGrids) -> ModuleNodes {
    let mut by_keys: BTreeMap<ModuleKey, BTreeMap<u32, Node>> = BTreeMap::new();
    for (key, by_native) in grids.by_keys {
        let per_head = by_keys.entry(key).or_default();
        for (head, by_step) in by_native {
            let node = per_head.entry(head).or_insert_with(Node::branch);
            for (step, by_isl) in by_step {
                for (isl, by_batch) in by_isl {
                    for (bb, lat) in by_batch {
                        node.insert(&[step, isl, bb], lat);
                    }
                }
            }
        }
    }
    ModuleNodes { by_keys }
}

/// Convert loaded grids into generation-phase engine tables: per (key, head),
/// a `[batch][s_total]` Node where `s_total = isl + step`. The generation
/// CSVs use isl=1, so s_total = 1 + step. If multiple (step, isl) pairs map
/// to the same s_total the last write wins, mirroring Python's flat
/// `{b: {s_total: leaf}}` dict overwrite.
fn generation_nodes(grids: ModuleGrids) -> ModuleNodes {
    let mut by_keys: BTreeMap<ModuleKey, BTreeMap<u32, Node>> = BTreeMap::new();
    for (key, by_native) in grids.by_keys {
        let per_head = by_keys.entry(key).or_default();
        for (head, by_step) in by_native {
            let node = per_head.entry(head).or_insert_with(Node::branch);
            for (step, by_isl) in by_step {
                for (isl, by_batch) in by_isl {
                    let s_total = isl + step;
                    for (bb, lat) in by_batch {
                        node.insert(&[bb, s_total], lat);
                    }
                }
            }
        }
    }
    ModuleNodes { by_keys }
}

// ---------------------------------------------------------------------------
// DSV4 CSA topk DELTA calibration — port of Python `operations/dsv4.py`
// (`_TOPK_CORRECTION_ENABLED`, `_get_dsv4_topk_calib`,
// `_build_topk_calib_from_rows`, `_dsv4_interp_1d_from_points`,
// `_dsv4_topk_delta_ms`).
//
// The CSA context-module collector runs the topK kernel on DEGENERATE scores
// (near-constant logits -> the Small topK path's O(n^2) tie-break), inflating
// the measured module latency vs real silicon. The calibration file stores
// phase-qualified rows (context runs the v1 topk selector, generation v2:
// score_mode = v{1,2}_{flat,top_last}) per (step, isl, batch_size) shape;
// DELTA = flat.latency - top_last.latency per variant. At query time the
// matching variant's DELTA is SUBTRACTED from the CSA (compress_ratio==4)
// module latency only.
// Gate: AIC_DSV4_TOPK_CORRECTION (default on; set "0" to disable).
// ---------------------------------------------------------------------------

/// Python `_TOPK_CORRECTION_ENABLED = os.environ.get("AIC_DSV4_TOPK_CORRECTION", "1") != "0"`
/// — evaluated once at module import; mirrored here with a process-wide
/// OnceLock so both engines toggle together.
fn topk_correction_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_topk_correction_env(std::env::var("AIC_DSV4_TOPK_CORRECTION").ok().as_deref())
    })
}

/// Enabled unless the value is exactly `"0"` (Python `!= "0"`; unset -> on).
fn parse_topk_correction_env(value: Option<&str>) -> bool {
    value != Some("0")
}

/// Which topk selector variant a query needs: context runs v1, generation v2
/// (the producer phase-qualifies score_mode accordingly).
#[derive(Clone, Copy)]
enum TopkPhase {
    Context,
    Generation,
}

/// The paired topK DELTA tables, one per selector variant:
/// `(step, isl, batch_size) -> max(0, flat - top_last)`. Python keeps a
/// `by_pi` sibling in the calib dict, but only `exact` is read by
/// `_dsv4_topk_delta_ms`, so only `exact` is ported.
struct TopkCalib {
    exact_v1: BTreeMap<(u32, u32, u32), f64>,
    exact_v2: BTreeMap<(u32, u32, u32), f64>,
    /// RAW top_last rows for the CP composition (Python
    /// `_load_csa_topk_top_last` reads the same parquet; here they are
    /// retained in the ONE calib load pass instead of a second read). Keyed
    /// by the row's `num_heads` (`None` when the CSV lacks the column) ->
    /// `{bs -> {(isl, step) -> top_last latency}}`.
    top_last: BTreeMap<Option<u32>, SparseGrid>,
}

/// Python apply formula shared by both apply sites:
/// `corrected = max(0.0, latency - _dsv4_topk_delta_ms(calib, prefix, isl, bs))`,
/// with a missing calib meaning DELTA = 0 (no-op).
fn apply_topk_delta(
    latency: f64,
    exact: Option<&BTreeMap<(u32, u32, u32), f64>>,
    prefix: u32,
    isl: u32,
    bs: u32,
) -> f64 {
    match exact {
        Some(exact) => (latency - topk_delta_ms(exact, prefix, isl, bs)).max(0.0),
        None => latency,
    }
}

/// Load `dsv4_csa_topk_calib_perf.parquet` from an ordered source list and
/// pair the flat/top_last rows into the DELTA table, ALSO retaining the raw
/// top_last rows for the CP composition (`csa_topk_top_last`) — one load
/// pass serves both consumers.
///
/// Mirrors Python `load_dsv4_sparse_op_data(sources, _TOPK_CALIB_KEYS)` +
/// `_build_topk_calib_from_rows`: rows nest under
/// `(step, isl, batch_size, score_mode)` with last-write-wins per leaf; a
/// shape missing either mode is skipped; `DELTA = max(0, flat - top_last)`.
/// The retained top_last grid mirrors `_load_csa_topk_top_last`'s
/// `{bs: {(isl, step): latency}}` with the same last-row-wins overwrite.
/// Returns `Ok(None)` when every source file is absent (Python: rows is
/// None) or no usable row exists (no DELTA pair AND no top_last row —
/// behaviourally identical to Python's two separate None/{} outcomes).
fn load_topk_calib_parquet(sources: &[PerfSource]) -> Result<Option<TopkCalib>, AicError> {
    let mut by_mode: BTreeMap<(u32, u32, u32), BTreeMap<String, f64>> = BTreeMap::new();
    let mut top_last: BTreeMap<Option<u32>, SparseGrid> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let step_col = reader.col("step")?;
        let isl_col = reader.col("isl")?;
        let batch_size_col = reader.col("batch_size")?;
        let score_mode_col = reader.col("score_mode")?;
        let latency_col = reader.col("latency")?;
        let num_heads_col = reader.col_optional("num_heads");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let (step, isl, bs) =
                (row.u32(step_col)?, row.u32(isl_col)?, row.u32(batch_size_col)?);
            let mode = row.str_owned(score_mode_col)?;
            let latency = row.f64(latency_col)?;
            // The CP composition consumes the context (v1) selector's raw
            // top_last latencies.
            if mode == "v1_top_last" {
                top_last
                    .entry(row.u32_optional(num_heads_col)?)
                    .or_default()
                    .entry(bs)
                    .or_default()
                    .insert((isl, step), latency);
            }
            by_mode.entry((step, isl, bs)).or_default().insert(mode, latency);
        }
    }
    if !any_source {
        return Ok(None);
    }
    let mut exact_v1 = BTreeMap::new();
    let mut exact_v2 = BTreeMap::new();
    for (key, modes) in by_mode {
        if let (Some(flat), Some(tl)) = (modes.get("v1_flat"), modes.get("v1_top_last")) {
            exact_v1.insert(key, (flat - tl).max(0.0));
        }
        if let (Some(flat), Some(tl)) = (modes.get("v2_flat"), modes.get("v2_top_last")) {
            exact_v2.insert(key, (flat - tl).max(0.0));
        }
    }
    if exact_v1.is_empty() && exact_v2.is_empty() && top_last.is_empty() {
        return Ok(None);
    }
    Ok(Some(TopkCalib { exact_v1, exact_v2, top_last }))
}

/// Paged-mqa-logits sparse-kernel table: `[native_heads][tp]` -> engine Node
/// `[past_kv(step)][isl][batch]` (Python `load_dsv4_sparse_kernel_data`'s
/// `_SPARSE_KERNEL_KEYS = (num_heads, tp_size, step, isl, batch_size)`).
struct SparseKernelNodes {
    by_heads: BTreeMap<u32, BTreeMap<u32, Node>>,
}

/// Load one sparse-kernel parquet. Rows are nested with plain overwrite in
/// read order — within a file the LAST row wins, and across shared-layer
/// sources a later source overwrites an earlier one at a shared cell,
/// matching Python (`_read_filtered_rows` concatenates sources in order and
/// `load_dsv4_sparse_op_data` does a per-row dict overwrite; the module
/// loader above follows the same policy). `Ok(None)` only when every source
/// file is absent.
fn load_sparse_kernel_parquet(sources: &[PerfSource]) -> Result<Option<SparseKernelNodes>, AicError> {
    let mut by_heads: BTreeMap<u32, BTreeMap<u32, Node>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let num_heads_col = reader.col("num_heads")?;
        let tp_size_col = reader.col("tp_size")?;
        let step_col = reader.col("step")?;
        let isl_col = reader.col("isl")?;
        let batch_size_col = reader.col("batch_size")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            by_heads
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(tp_size_col)?)
                .or_insert_with(Node::branch)
                .insert(
                    &[row.u32(step_col)?, row.u32(isl_col)?, row.u32(batch_size_col)?],
                    row.f64(latency_col)?,
                );
        }
    }
    if !any_source {
        return Ok(None);
    }
    Ok(Some(SparseKernelNodes { by_heads }))
}

/// Python `_dsv4_interp_1d_from_points`: linear interpolation with
/// nearest-value extrapolation; duplicate coordinates are mean-merged.
fn topk_interp_1d(points: &[(u32, f64)], x: u32) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let mut merged: BTreeMap<u32, (f64, u32)> = BTreeMap::new();
    for &(coord, value) in points {
        let entry = merged.entry(coord).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }
    let vals: BTreeMap<u32, f64> =
        merged.into_iter().map(|(k, (sum, n))| (k, sum / n as f64)).collect();
    if let Some(&v) = vals.get(&x) {
        return Some(v);
    }
    let (&first_x, &first_v) = vals.iter().next().unwrap();
    let (&last_x, &last_v) = vals.iter().next_back().unwrap();
    if vals.len() == 1 || x <= first_x {
        return Some(first_v);
    }
    if x >= last_x {
        return Some(last_v);
    }
    let (&left_x, &left_v) = vals.range(..x).next_back().unwrap();
    let (&right_x, &right_v) = vals.range(x + 1..).next().unwrap();
    let t = (x as f64 - left_x as f64) / (right_x as f64 - left_x as f64);
    Some(left_v * (1.0 - t) + right_v * t)
}

/// Python `_dsv4_topk_delta_ms`: exact `(prefix, isl, bs)` rows are preferred;
/// off-grid shapes are interpolated prefix-first within a fixed `(isl, bs)`,
/// then isl, then bs. Returns 0.0 when nothing resolves.
fn topk_delta_ms(exact: &BTreeMap<(u32, u32, u32), f64>, prefix: u32, isl: u32, bs: u32) -> f64 {
    if let Some(&direct) = exact.get(&(prefix, isl, bs)) {
        return direct.max(0.0);
    }
    let prefix_interp = |query_prefix: u32, anchor_isl: u32, anchor_bs: u32| -> Option<f64> {
        let points: Vec<(u32, f64)> = exact
            .iter()
            .filter(|(&(_, i, b), _)| i == anchor_isl && b == anchor_bs)
            .map(|(&(p, _, _), &d)| (p, d))
            .collect();
        topk_interp_1d(&points, query_prefix)
    };
    let isl_interp = |query_prefix: u32, query_isl: u32, anchor_bs: u32| -> Option<f64> {
        let isl_values: BTreeSet<u32> =
            exact.keys().filter(|(_, _, b)| *b == anchor_bs).map(|(_, i, _)| *i).collect();
        let points: Vec<(u32, f64)> = isl_values
            .into_iter()
            .filter_map(|i| prefix_interp(query_prefix, i, anchor_bs).map(|v| (i, v)))
            .collect();
        topk_interp_1d(&points, query_isl)
    };
    let bs_values: BTreeSet<u32> = exact.keys().map(|(_, _, b)| *b).collect();
    let points: Vec<(u32, f64)> = bs_values
        .into_iter()
        .filter_map(|b| isl_interp(prefix, isl, b).map(|v| (b, v)))
        .collect();
    match topk_interp_1d(&points, bs) {
        Some(interpolated) => interpolated.max(0.0),
        None => 0.0,
    }
}

/// Resolve the quant key ([fmha][kv][gemm] for context, [kv][gemm] for
/// generation — pass `fmha = None`), then resolve the model's rank-LOCAL
/// head count against the CSV head keys, returning the engine table for that
/// head.
fn select_resolved<'a>(
    grids: &'a ModuleNodes,
    fmha: Option<FmhaQuantMode>,
    kv: KvCacheQuantMode,
    gemm: GemmQuantMode,
    local_heads: u32,
) -> Result<&'a Node, AicError> {
    let key = ModuleKey {
        fmha_quant: fmha.map(|f| f.name().to_string()).unwrap_or_default(),
        kv_quant: kv.name().to_string(),
        gemm_quant: gemm.name().to_string(),
    };
    let by_native = grids
        .by_keys
        .get(&key)
        .ok_or_else(|| AicError::PerfDatabase(format!("DSV4 module data missing for {key:?}")))?;
    let head = resolve_head_key(by_native, local_heads).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "DSV4 module data missing for local_heads={local_heads}, {key:?} (loaded heads: {:?})",
            by_native.keys().collect::<Vec<_>>()
        ))
    })?;
    Ok(&by_native[&head])
}

/// Resolve the model's rank-LOCAL head count against the available CSV head
/// keys. Mirrors Python `operations.dsv4._dsv4_resolve_head_key`:
///   1. exact match on the requested local-head value;
///   2. if only one head key is loaded, use it (the b300 universal-sweep case);
///   3. otherwise the nearest head key `<=` request, else the smallest key.
fn resolve_head_key<T>(by_native: &BTreeMap<u32, T>, local_heads: u32) -> Option<u32> {
    if by_native.is_empty() {
        return None;
    }
    if by_native.contains_key(&local_heads) {
        return Some(local_heads);
    }
    if by_native.len() == 1 {
        return by_native.keys().next().copied();
    }
    // nearest <= request, else the smallest available.
    by_native
        .range(..=local_heads)
        .next_back()
        .map(|(&k, _)| k)
        .or_else(|| by_native.keys().next().copied())
}

// ---------------------------------------------------------------------------
// Analytic SOL — verbatim port of Python `_deepseek_v4_attention_sol`
// ---------------------------------------------------------------------------

/// Python `GEMM._get_quant_tc_flops` (compute factor -> spec TC-flops entry,
/// bf16-scaled fallback).
fn tc_flops(spec: &SystemSpec, compute_factor: f64) -> f64 {
    let bf16 = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    let direct = match compute_factor as u32 {
        1 => spec.gpu.bfloat16_tc_flops,
        2 => spec.gpu.fp8_tc_flops,
        4 => spec.gpu.fp4_tc_flops,
        _ => None,
    };
    direct.unwrap_or(bf16 * compute_factor)
}

/// Python `PerfDatabase._causal_limited_pairs`: sum over queries of
/// `min(prefix + query_index + 1, limit)`, times batch.
fn causal_limited_pairs(batch: i128, query_len: i128, prefix: i128, limit: i128) -> i128 {
    if limit <= 0 || query_len <= 0 {
        return 0;
    }
    let full_s = prefix + query_len;
    if prefix >= limit {
        return batch * query_len * limit;
    }
    if full_s <= limit {
        return batch * (full_s * (full_s + 1) - prefix * (prefix + 1)) / 2;
    }
    let ramp = batch * (limit * (limit + 1) - prefix * (prefix + 1)) / 2;
    let saturated = batch * (full_s - limit) * limit;
    ramp + saturated
}

/// Python `PerfDatabase._sum_floor_upto`: `sum_{i=0..n} floor(i / divisor)`.
fn sum_floor_upto(n: i128, divisor: i128) -> i128 {
    if n < 0 {
        return 0;
    }
    let q = n / divisor;
    let r = n % divisor;
    divisor * q * (q - 1) / 2 + q * (r + 1)
}

/// Python `PerfDatabase._compressed_context_pairs`.
fn compressed_context_pairs(batch: i128, query_len: i128, prefix: i128, ratio: i128, limit: i128) -> i128 {
    if ratio <= 0 || query_len <= 0 || limit <= 0 {
        return 0;
    }
    let start = prefix + 1;
    let end = prefix + query_len;
    let saturation_start = limit * ratio;
    let total = if end < saturation_start {
        sum_floor_upto(end, ratio) - sum_floor_upto(start - 1, ratio)
    } else if start >= saturation_start {
        query_len * limit
    } else {
        let ramp = sum_floor_upto(saturation_start - 1, ratio) - sum_floor_upto(start - 1, ratio);
        ramp + (end - saturation_start + 1) * limit
    };
    batch * total
}

/// Shared SOL formula for both context and generation phases. Verbatim port
/// of Python `operations.dsv4._deepseek_v4_attention_sol` (returns only the
/// `max(sol_math, sol_mem)` scalar the engine consumes).
///
/// `local_heads` is the rank-local head count Python passes as `num_heads`.
/// `dims.local_o_groups` is the rank-local group count Python passes as
/// `o_groups` (see [`Dsv4SolDims`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsv4_attention_sol_ms(
    spec: &SystemSpec,
    dims: &Dsv4SolDims,
    compress_ratio: i64,
    is_context: bool,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    gemm_quant: GemmQuantMode,
    b: i64,
    s: i64,
    prefix: i64,
    local_heads: i64,
) -> f64 {
    let local_o_groups = dims.local_o_groups.max(1);

    let (b, s, prefix) = (b as i128, s as i128, prefix as i128);
    let nh = local_heads as i128;
    let h = dims.hidden_size as i128;
    let qlr = dims.q_lora_rank as i128;
    let olr = dims.o_lora_rank as i128;
    let hd = dims.head_dim as i128;
    let rope_hd = dims.rope_head_dim as i128;
    let inh = dims.index_n_heads as i128;
    let ihd = dims.index_head_dim as i128;
    let topk = dims.index_topk as i128;
    let window = dims.window_size as i128;
    let cr = compress_ratio as i128;
    let lg = local_o_groups as i128; // Python local_groups = max(1, o_groups)

    let tokens = if is_context { b * s } else { b };
    let kv_len = if is_context { prefix + s } else { (s - 1).max(0) };

    let gemm_projection_ops =
        2 * tokens * h * qlr + 2 * tokens * qlr * nh * hd + 2 * tokens * h * hd + 2 * tokens * lg * olr * h;
    let output_absorption_ops = 2 * tokens * nh * hd * olr;

    let compressor_mult: i128 = if cr == 4 { 2 } else { 1 };
    let mut compressor_ops: i128 = 0;
    if cr != 0 {
        compressor_ops = 4 * tokens * h * compressor_mult * hd + 2 * tokens * compressor_mult * hd;
        if cr == 4 {
            let indexer_compressor_mult: i128 = 2;
            compressor_ops += 4 * tokens * h * indexer_compressor_mult * ihd;
            compressor_ops += 2 * tokens * indexer_compressor_mult * ihd;
        }
    }

    let (window_pairs, compressed_pairs) = if is_context {
        let wp = causal_limited_pairs(b, s, prefix, window);
        let cp = if cr != 0 {
            let limit = if cr == 4 { topk } else { (kv_len / cr).max(0) };
            compressed_context_pairs(b, s, prefix, cr, limit)
        } else {
            0
        };
        (wp, cp)
    } else {
        let wp = b * kv_len.min(window);
        let cp = if cr != 0 {
            let limit = if cr == 4 { topk } else { (kv_len / cr).max(0) };
            b * (kv_len / cr).min(limit)
        } else {
            0
        };
        (wp, cp)
    };

    let attention_pairs = window_pairs + compressed_pairs;
    let attention_ops = 4 * nh * hd * attention_pairs;

    let mut indexer_ops: i128 = 0;
    let mut indexer_bfloat16_ops: i128 = 0;
    let mut indexer_cache_bytes: f64 = 0.0;
    if cr == 4 {
        let compressed_len = kv_len / cr;
        let indexer_query_tokens = if is_context { b * s } else { b };
        let indexer_pairs = indexer_query_tokens * compressed_len;
        indexer_ops = 2 * indexer_query_tokens * qlr * inh * ihd + 2 * indexer_pairs * inh * ihd;
        indexer_bfloat16_ops = 2 * indexer_query_tokens * h * inh;
        // Python: b * compressed_len * deepseek_v4_indexer_cache_entry_bytes(ihd)
        // where the entry is index_head_dim * 0.5 (FP4).
        indexer_cache_bytes = (b * compressed_len) as f64 * (dims.index_head_dim as f64 * 0.5);
    }

    let gemm_mem = gemm_quant.mapping().memory;
    let bf16_mem = GemmQuantMode::Bfloat16.mapping().memory;
    let mut gemm_weight_bytes =
        (h * qlr + qlr * nh * hd + h * hd + lg * olr * h) as f64 * gemm_mem;
    let mut bfloat16_weight_bytes = (nh * hd * olr) as f64 * bf16_mem;
    if cr != 0 {
        gemm_weight_bytes += (2 * h * compressor_mult * hd) as f64 * gemm_mem;
    }
    if cr == 4 {
        gemm_weight_bytes += (qlr * inh * ihd) as f64 * gemm_mem;
        bfloat16_weight_bytes += (h * inh) as f64 * bf16_mem;
    }

    let activation_bytes = (tokens * (h + qlr + nh * hd + hd + lg * olr)) as f64 * gemm_mem;
    let kv_cache_bytes = (attention_pairs * hd) as f64 * kv_quant.mapping().memory;
    let rope_bytes = (tokens * nh * rope_hd) as f64 * fmha_quant.mapping().memory;

    let sol_math = ((gemm_projection_ops + compressor_ops) as f64 / tc_flops(spec, gemm_quant.mapping().compute)
        + (output_absorption_ops + indexer_bfloat16_ops) as f64
            / tc_flops(spec, GemmQuantMode::Bfloat16.mapping().compute)
        + indexer_ops as f64 / tc_flops(spec, GemmQuantMode::Fp8.mapping().compute)
        + attention_ops as f64 / tc_flops(spec, fmha_quant.mapping().compute))
        * 1000.0;
    let sol_mem = (gemm_weight_bytes
        + bfloat16_weight_bytes
        + activation_bytes
        + kv_cache_bytes
        + indexer_cache_bytes
        + rope_bytes)
        / spec.gpu.mem_bw
        * 1000.0;
    sol_math.max(sol_mem)
}

/// Canonicalize a DSV4 CSV dtype string to the enum `.name()` form.
///
/// Mirrors Python `_dsv4_normalize_dtype` / `_DSV4_DTYPE_ALIASES`: the only
/// alias is `fp8_e4m3` -> `fp8`. Everything else passes through unchanged.
fn normalize_dsv4_dtype(name: &str) -> String {
    match name {
        "fp8_e4m3" => "fp8".to_string(),
        other => other.to_string(),
    }
}

/// Load a DSV4 module table from an ordered, priority-sorted source list.
/// Sources are read in order; missing files are skipped (a sibling declared in
/// the manifest need not exist for every system). Within the DSV4 dict the
/// tp_size axis is collapsed with last-write-wins, so a later source overwrites
/// an earlier one at a shared cell (mirroring Python's flat-dict overwrite). An
/// error is returned only when no source yields rows.
///
/// `key_on_fmha` selects the Python keying (PR #1337): context loaders key
/// `[fmha][kv][gemm]`, generation loaders ignore the `mla_dtype` column and
/// key `[kv][gemm]` (the fmha level of `ModuleKey` stays the empty sentinel,
/// and duplicate rows that differed only in `mla_dtype` collapse last-wins —
/// exactly what Python's direct-assign loader does).
fn load_module_parquet(sources: &[PerfSource], key_on_fmha: bool) -> Result<ModuleGrids, AicError> {
    let mut by_keys: BTreeMap<ModuleKey, ByNative> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let mla_dtype_col = if key_on_fmha { Some(reader.col("mla_dtype")?) } else { None };
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let gemm_type_col = reader.col("gemm_type")?;
        let num_heads_col = reader.col("num_heads")?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let step_col = reader.col("step")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");

        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = ModuleKey {
                // CSV columns use sglang dtype naming; the query side builds keys
                // from the enum `.name()` (canonical short names). Normalize on
                // load to match Python `_dsv4_normalize_dtype`, which aliases
                // `fp8_e4m3` -> `fp8` for `mla_dtype` (fmha) and `kv_cache_dtype`
                // (kv). `gemm_type` is intentionally left untouched, matching
                // Python (e.g. `fp8_block` is a real value that must pass through).
                fmha_quant: match mla_dtype_col {
                    Some(col) => normalize_dsv4_dtype(&row.str_owned(col)?),
                    None => String::new(),
                },
                kv_quant: normalize_dsv4_dtype(&row.str_owned(kv_cache_dtype_col)?),
                gemm_quant: row.str_owned(gemm_type_col)?,
            };
            // Last-wins parity with Python `load_*_dsv4_kind_module_data`, which
            // assigns `data[...][b][s] = {...}` per row keyed on
            // `(num_heads, compress_ratio, step, isl, batch)` but NOT on `tp_size`,
            // so a later row (here: a higher tp_size, since the file is
            // gemm-then-tp-ascending) overwrites the earlier one. We drop the tp
            // axis and let `BTreeMap::insert` overwrite; do NOT use `or_insert`.
            by_keys
                .entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?) // CSV `num_heads` column (head axis)
                .or_default()
                .entry(row.u32(step_col)?)
                .or_default()
                .entry(row.u32(isl_col)?)
                .or_default()
                .insert(row.u32(batch_size_col)?, row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSV4 module rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(ModuleGrids { by_keys })
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b200_sxm_spec() -> SystemSpec {
        let systems_yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("src/aiconfigurator_core/systems/b200_sxm.yaml");
        SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse")
    }

    fn b200_sglang_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/data/b200_sxm/sglang/0.5.10")
    }

    #[test]
    fn dsv4_data_absent_errors_cleanly() {
        // DSV4 modules aren't collected for vllm/0.19.0; loader must surface
        // a clean error.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.19.0");
        let table = Dsv4Table::new(root);
        let spec = b200_sxm_spec();
        let err = table
            .query_context(
                &spec,
                AttnKind::Csa,
                1,
                1024,
                128, // local_heads
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV4ForCausalLM",
                0,
                None,
            )
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// Cross-language parity with the Python v2 engine on the real
    /// b200_sxm/sglang/0.5.10 tables. Oracle values generated with
    /// `PYTHONPATH=src AIC_DSV4_TOPK_CORRECTION=0 python3` via
    /// `PerfDatabase.query_{context,generation}_deepseek_v4_attention_module`
    /// (DatabaseMode.SILICON, shared layer off, DSV4-Pro dims with rank-local
    /// num_heads=16 / o_groups=2). Covers, per phase: exact hit, interior
    /// blend, and util-hold beyond the collected range (incl. the ragged
    /// batch row and the step=0-only prefix axis).
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails.
    #[test]
    fn dsv4_query_matches_python_v2_engine() {
        let root = b200_sglang_root();
        if !root.join("dsv4_csa_context_module_perf.parquet").exists() {
            return; // git-lfs data not materialized
        }
        let table = Dsv4Table::new(root);
        let spec = b200_sxm_spec();
        let q_ctx = |kind, b, isl, prefix| {
            table
                .query_context(
                    &spec, kind, b, isl, 16, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM", prefix, None,
                )
                .unwrap()
        };
        let q_gen = |kind, b, s| {
            table
                .query_generation(
                    &spec, kind, b, s, 16, KvCacheQuantMode::Fp8,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM",
                    None,
                )
                .unwrap()
        };
        let approx = |got: f64, want: f64| {
            assert!(
                ((got - want) / want).abs() < 1e-9,
                "rust {got} vs python {want}"
            );
        };
        // Context CSA: exact / interior isl / interior batch / isl util-hold /
        // prefix util-hold (step axis has only the 0 anchor).
        approx(q_ctx(AttnKind::Csa, 8, 512, 0), 0.9819);
        approx(q_ctx(AttnKind::Csa, 8, 768, 0), 1.46555);
        approx(q_ctx(AttnKind::Csa, 12, 512, 0), 1.4142000000000001);
        approx(q_ctx(AttnKind::Csa, 8, 8192, 0), 20.84104587973274);
        approx(q_ctx(AttnKind::Csa, 8, 512, 1024), 1.0928707937592828);
        // Context HCA: exact / isl util-hold.
        approx(q_ctx(AttnKind::Hca, 1, 128, 0), 0.0802);
        approx(q_ctx(AttnKind::Hca, 8, 8192, 0), 9.0088);
        // Generation CSA: exact / interior s / s util-hold / ragged batch.
        // Util-hold oracle regenerated post-#1337: the generation SOL now
        // derives its fmha dtype from the kv dtype (fp8 here), not the label.
        approx(q_gen(AttnKind::Csa, 16, 385), 0.1142);
        approx(q_gen(AttnKind::Csa, 16, 200), 0.11328828125);
        approx(q_gen(AttnKind::Csa, 16, 100000), 0.17550461244541488);
        approx(q_gen(AttnKind::Csa, 15, 385), 0.1129625);
        // Generation HCA: exact.
        approx(q_gen(AttnKind::Hca, 16, 385), 0.07239999999999999);
    }

    /// Parity regression for the DeepSeek-V4-Pro b200_sxm/sglang/0.5.10 lookup.
    /// The model passes rank-LOCAL `num_heads = 128 / tp(8) = 16`, which must
    /// resolve to the CSV head key 64 (Python `_dsv4_resolve_head_key`).
    /// Oracle values regenerated from the Python v2 engine (perf_interp):
    /// exact grid points return the measured leaves; the ragged
    /// `q_gen(Csa, 15, 385)` row now resolves through the engine
    /// (single-survivor SOL-ratio correction) instead of the deleted
    /// batch-scaling fallback.
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails.
    #[test]
    fn dsv4_pro_head_resolution_and_ragged_generation() {
        let root = b200_sglang_root();
        if !root.join("dsv4_csa_generation_module_perf.parquet").exists() {
            return; // git-lfs data not materialized
        }
        let table = Dsv4Table::new(root);
        let spec = b200_sxm_spec();
        let q_gen = |kind, b, s| {
            table
                .query_generation(
                    &spec, kind, b, s, 16, KvCacheQuantMode::Fp8,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM",
                    None,
                )
                .unwrap()
        };
        let q_ctx = |kind, b, isl| {
            table
                .query_context(
                    &spec, kind, b, isl, 16, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM", 0, None,
                )
                .unwrap()
        };
        let approx = |got: f64, want: f64| {
            assert!(
                ((got - want) / want).abs() < 1e-9,
                "rust {got} vs python {want}"
            );
        };
        // local=16 resolves to head-64; b=16/s=385 is an exact grid point.
        approx(q_gen(AttnKind::Csa, 16, 385), 0.1142);
        approx(q_gen(AttnKind::Hca, 16, 385), 0.0724);
        // RAGGED batch row: engine semantics (regenerated from Python v2;
        // the deleted batch-scaling fallback returned 0.19556 here).
        approx(q_gen(AttnKind::Csa, 15, 385), 0.1129625);
        // Context single-anchor lookups (exact grid points).
        approx(q_ctx(AttnKind::Csa, 1, 128), 0.132);
        approx(q_ctx(AttnKind::Hca, 1, 128), 0.0802);
    }

    #[test]
    fn normalize_dsv4_dtype_aliases_fp8_e4m3() {
        assert_eq!(normalize_dsv4_dtype("fp8_e4m3"), "fp8");
        // Non-aliased values must pass through unchanged.
        assert_eq!(normalize_dsv4_dtype("bfloat16"), "bfloat16");
        assert_eq!(normalize_dsv4_dtype("fp8_block"), "fp8_block");
        assert_eq!(normalize_dsv4_dtype("fp8"), "fp8");
    }

    #[test]
    fn dsv4_context_resolves_fp8_e4m3_kv_quant() {
        // Regression for the b200_sxm/sglang/0.5.10 DSV4 context lookup: the
        // CSV stores `kv_cache_dtype=fp8_e4m3`, but the query builds the key
        // from `KvCacheQuantMode::Fp8.name()` = "fp8". Without load-side
        // normalization the lookup misses (Rust-only error vs Python success).
        let root = b200_sglang_root();
        if !root.join("dsv4_csa_context_module_perf.parquet").exists() {
            // Data files are git-lfs tracked; skip if not materialized.
            return;
        }
        let table = Dsv4Table::new(root);
        let spec = b200_sxm_spec();
        // (head=64, isl=512, batch=8, step=0) are measured grid points in the
        // CSA context table for this entry, gemm=fp8_block.
        let latency = table
            .query_context(
                &spec,
                AttnKind::Csa,
                8,   // batch
                512, // isl
                64,  // local_heads (exact head key)
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Fp8Block,
                "DeepseekV4ForCausalLM",
                0, // prefix
                None,
            )
            .expect("DSV4 context lookup must resolve fp8_e4m3 kv_cache_dtype as fp8");
        assert!(latency.is_finite() && latency > 0.0, "unexpected latency: {latency}");
    }

    // ------------------------------------------------------------------
    // CSA topk DELTA calibration (port of Python operations/dsv4.py:
    // _get_dsv4_topk_calib / _build_topk_calib_from_rows / _dsv4_topk_delta_ms)
    // ------------------------------------------------------------------

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

    /// Rows are `(score_mode, step, isl, batch_size, latency)` — the calib
    /// columns the loader reads (collector writes more; extras are ignored).
    fn write_calib_parquet(path: &Path, rows: &[(&str, i64, i64, i64, f64)]) {
        let schema = Arc::new(
            parse_message_type(
                "message calib {
                    REQUIRED BYTE_ARRAY score_mode (UTF8);
                    REQUIRED INT64 step;
                    REQUIRED INT64 isl;
                    REQUIRED INT64 batch_size;
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
        let modes: Vec<ByteArray> = rows.iter().map(|r| ByteArray::from(r.0)).collect();
        write_column::<ByteArrayType>(&mut rg, &modes);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.4).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Rows are `(num_heads, batch_size, isl, step, latency)` under the fixed
    /// (DeepseekV4ForCausalLM, bfloat16, fp8, fp8_block) key the tests query.
    fn write_module_parquet(path: &Path, rows: &[(i64, i64, i64, i64, f64)]) {
        let schema = Arc::new(
            parse_message_type(
                "message module {
                    REQUIRED BYTE_ARRAY architecture (UTF8);
                    REQUIRED BYTE_ARRAY mla_dtype (UTF8);
                    REQUIRED BYTE_ARRAY kv_cache_dtype (UTF8);
                    REQUIRED BYTE_ARRAY gemm_type (UTF8);
                    REQUIRED INT64 num_heads;
                    REQUIRED INT64 batch_size;
                    REQUIRED INT64 isl;
                    REQUIRED INT64 step;
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
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("DeepseekV4ForCausalLM"); n]);
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("bfloat16"); n]);
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("fp8"); n]);
        write_column::<ByteArrayType>(&mut rg, &vec![ByteArray::from("fp8_block"); n]);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.0).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.4).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Item 5: the analytic SOL must use the OP's structural dims (Python
    /// sources them from the model config), not a table pinned to the
    /// DeepSeek-V4-Pro shape — Pro and Flash share the architecture string
    /// but differ in (hidden, q_lora, index_topk, o_groups, native heads),
    /// so a Flash op spec must yield a DIFFERENT beyond-grid hold. Synthetic
    /// HCA table {isl 1024: 1.0, 2048: 2.0} at (n=64, b=1, step=0); querying
    /// isl=8192 holds at the isl=2048 anchor:
    /// `hold = 2.0 * sol(8192) / sol(2048)`. Oracles hand-computed from the
    /// Python formula:
    ///
    /// ```text
    /// PYTHONPATH=src python3 -c "
    /// from aiconfigurator.sdk.perf_database import PerfDatabase
    /// from aiconfigurator.sdk.operations.dsv4 import _deepseek_v4_attention_sol
    /// from aiconfigurator.sdk import common
    /// db = PerfDatabase('b200_sxm','sglang','0.5.10',
    ///                   systems_root='src/aiconfigurator_core/systems', database_mode='SOL')
    /// def sol(s, hidden, q_lora, index_topk):
    ///     return _deepseek_v4_attention_sol(db, is_context=True, b=1, s=s, prefix=0,
    ///         num_heads=64, hidden_size=hidden, q_lora_rank=q_lora, o_lora_rank=1024,
    ///         head_dim=512, rope_head_dim=64, index_n_heads=64, index_head_dim=128,
    ///         index_topk=index_topk, window_size=128, compress_ratio=128, o_groups=8,
    ///         kvcache_quant_mode=common.KVCacheQuantMode.fp8,
    ///         fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    ///         gemm_quant_mode=common.GEMMQuantMode.fp8_block)[0]
    /// for name, hidden, qlr, topk in [('flash',4096,1024,512), ('pro',7168,1536,1024)]:
    ///     print(name, repr(2.0 * sol(8192, hidden, qlr, topk) / sol(2048, hidden, qlr, topk)))"
    /// # -> flash 8.17467016968122 / pro 8.131309314148407
    /// ```
    ///
    /// The old pinned-dims code returned the PRO hold for the Flash spec.
    #[test]
    fn dsv4_flash_op_spec_dims_change_beyond_grid_hold() {
        use crate::operators::dsv4::Dsv4ModuleOp;

        let dir = tempfile::tempdir().unwrap();
        write_module_parquet(
            &dir.path().join("dsv4_hca_context_module_perf.parquet"),
            &[(64, 1, 1024, 0, 1.0), (64, 1, 2048, 0, 2.0)],
        );
        let table = Dsv4Table::new(dir.path().to_path_buf());
        let spec = b200_sxm_spec();

        // Flash op spec: dims come through the SERDE path, like the engine.
        let flash: Dsv4ModuleOp = serde_json::from_value(serde_json::json!({
            "name": "context_attention",
            "scale_factor": 1.0,
            "attn_kind": "Hca",
            "num_heads": 64,
            "native_heads": 64,
            "tp_size": 1,
            "kv_cache_dtype": "fp8",
            "fmha_quant_mode": "bfloat16",
            "gemm_quant_mode": "fp8_block",
            "architecture": "DeepseekV4ForCausalLM",
            "window_size": 128,
            "hidden_size": 4096,
            "q_lora_rank": 1024,
            "o_lora_rank": 1024,
            "head_dim": 512,
            "rope_head_dim": 64,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "o_groups": 8,
        }))
        .expect("Flash op spec must deserialize");
        let q = |sol_dims, isl: u32| {
            table
                .query_context(
                    &spec,
                    AttnKind::Hca,
                    1,
                    isl,
                    64,
                    KvCacheQuantMode::Fp8,
                    FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block,
                    "DeepseekV4ForCausalLM",
                    0,
                    sol_dims,
                )
                .unwrap()
        };
        let approx = |got: f64, want: f64| {
            assert!(
                ((got - want) / want).abs() < 1e-9,
                "rust {got} vs python {want}"
            );
        };
        // isl=8192 is beyond the frontier -> util-hold on the SOL ratio.
        let flash_hold = q(Some(flash.sol_dims()), 8192);
        let pro_hold = q(None, 8192); // pinned default (old specs / direct queries)
        approx(flash_hold, 8.17467016968122);
        approx(pro_hold, 8.131309314148407);
        assert!(
            (flash_hold - pro_hold).abs() > 1e-3,
            "Flash dims must change the hold ({flash_hold} vs {pro_hold})"
        );
        // In-range resolution is SOL-free and identical for both.
        approx(q(Some(flash.sol_dims()), 1536), 1.5);
        approx(q(None, 1536), 1.5);
    }

    /// Old op specs carry none of the dim fields; serde must default them to
    /// the Pro values and `sol_dims()` must reproduce the pinned derivation
    /// (`from_pinned`) exactly, including the tp-derived local o_groups.
    #[test]
    fn dsv4_op_spec_dims_default_to_pinned_pro() {
        use crate::operators::dsv4::Dsv4ModuleOp;

        let old_spec: Dsv4ModuleOp = serde_json::from_value(serde_json::json!({
            "name": "context_attention",
            "scale_factor": 1.0,
            "attn_kind": "Csa",
            "num_heads": 16,
            "native_heads": 128,
            "tp_size": 8,
            "kv_cache_dtype": "fp8",
            "fmha_quant_mode": "bfloat16",
            "gemm_quant_mode": "fp8_block",
            "architecture": "DeepseekV4ForCausalLM",
        }))
        .expect("old op spec must deserialize");
        let got = old_spec.sol_dims();
        let want = Dsv4SolDims::from_pinned(dsv4_dims("DeepseekV4ForCausalLM"), 16);
        assert_eq!(got.hidden_size, want.hidden_size);
        assert_eq!(got.q_lora_rank, want.q_lora_rank);
        assert_eq!(got.o_lora_rank, want.o_lora_rank);
        assert_eq!(got.head_dim, want.head_dim);
        assert_eq!(got.rope_head_dim, want.rope_head_dim);
        assert_eq!(got.index_n_heads, want.index_n_heads);
        assert_eq!(got.index_head_dim, want.index_head_dim);
        assert_eq!(got.index_topk, want.index_topk);
        assert_eq!(got.window_size, want.window_size);
        assert_eq!(got.local_o_groups, want.local_o_groups); // 16 / (128/16) = 2
        assert_eq!(got.local_o_groups, 2);
    }

    /// Loader + delta parity against a Python oracle. Oracle generated with
    /// `PYTHONPATH=src python3` by feeding the SAME rows through
    /// `_build_topk_calib_from_rows` + `_dsv4_topk_delta_ms`
    /// (operations/dsv4.py); values below are the printed reprs.
    #[test]
    fn topk_calib_loader_and_delta_match_python_oracle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dsv4_csa_topk_calib_perf.parquet");
        write_calib_parquet(
            &path,
            &[
                ("v1_flat", 0, 512, 1, 1.0),
                ("v1_top_last", 0, 512, 1, 0.4),
                ("v1_flat", 0, 512, 4, 2.0),
                ("v1_top_last", 0, 512, 4, 0.9),
                ("v1_flat", 0, 2048, 1, 3.0),
                ("v1_top_last", 0, 2048, 1, 1.0),
                ("v1_flat", 0, 2048, 4, 5.0),
                ("v1_top_last", 0, 2048, 4, 2.2),
                ("v1_flat", 1024, 512, 1, 1.5),
                ("v1_top_last", 1024, 512, 1, 0.7),
                ("v1_flat", 1024, 512, 4, 2.5),
                ("v1_top_last", 1024, 512, 4, 1.0),
                ("v1_flat", 4096, 512, 1, 0.5),
                ("v1_top_last", 4096, 512, 1, 0.9), // flat < top_last -> DELTA clamps to 0
                ("v1_flat", 8192, 512, 1, 9.9),     // no top_last -> shape skipped
            ],
        );
        let calib = load_topk_calib_parquet(&[PerfSource(path, None)])
            .unwrap()
            .expect("calib must load");
        // Pairing (Python _build_topk_calib_from_rows): DELTA = max(0, flat - top_last).
        let expected_exact = [
            ((0u32, 512u32, 1u32), 0.6),
            ((0, 512, 4), 1.1),
            ((0, 2048, 1), 2.0),
            ((0, 2048, 4), 2.8),
            ((1024, 512, 1), 0.8),
            ((1024, 512, 4), 1.5),
            ((4096, 512, 1), 0.0),
        ];
        assert_eq!(calib.exact_v1.len(), expected_exact.len());
        assert!(calib.exact_v2.is_empty());
        for (key, want) in expected_exact {
            let got = calib.exact_v1[&key];
            assert!((got - want).abs() < 1e-12, "exact[{key:?}] = {got} vs {want}");
        }
        let oracle = [
            ((0u32, 512u32, 1u32), 0.6),          // exact hit
            ((512, 512, 1), 0.7),                 // prefix interp
            ((2048, 512, 1), 0.5333333333333334), // prefix interp across the clamped-0 anchor
            ((99999, 512, 4), 1.5),               // prefix extrapolation clamps to nearest
            ((8192, 512, 1), 0.0),                // skipped shape resolves via prefix clamp
            ((0, 1024, 1), 1.0666666666666667),   // isl interp
            ((0, 512, 2), 0.7666666666666667),    // bs interp
            ((512, 1024, 2), 1.3555555555555556), // prefix+isl+bs all off-grid
            ((384, 1, 16), 1.25),                 // generation-shaped: isl below range, bs above
            ((0, 512, 4), 1.1),                   // second exact hit
        ];
        for ((prefix, isl, bs), want) in oracle {
            let got = topk_delta_ms(&calib.exact_v1, prefix, isl, bs);
            let tol = if want == 0.0 { 1e-12 } else { want * 1e-9 };
            assert!(
                (got - want).abs() <= tol,
                "delta({prefix},{isl},{bs}) = {got} vs python {want}"
            );
        }
    }

    #[test]
    fn topk_calib_absent_file_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("dsv4_csa_topk_calib_perf.parquet");
        let calib = load_topk_calib_parquet(&[PerfSource(missing, None)]).unwrap();
        assert!(calib.is_none(), "absent file must load as None");
        // Missing calib -> DELTA machinery is a no-op (Python
        // `_dsv4_topk_delta_ms(None, ...) == 0.0`).
        assert_eq!(apply_topk_delta(1.25, None, 0, 512, 8), 1.25);
    }

    #[test]
    fn topk_correction_env_gate_matches_python() {
        // Python: os.environ.get("AIC_DSV4_TOPK_CORRECTION", "1") != "0" —
        // ONLY the literal "0" disables; anything else (or unset) enables.
        assert!(parse_topk_correction_env(None));
        assert!(parse_topk_correction_env(Some("1")));
        assert!(parse_topk_correction_env(Some("")));
        assert!(parse_topk_correction_env(Some("false")));
        assert!(!parse_topk_correction_env(Some("0")));
    }

    #[test]
    fn topk_delta_clamps_corrected_latency_at_zero() {
        let mut exact = BTreeMap::new();
        exact.insert((0u32, 512u32, 8u32), 0.12);
        // Python: corrected = max(0.0, latency - delta).
        assert_eq!(apply_topk_delta(0.05, Some(&exact), 0, 512, 8), 0.0);
        assert!((apply_topk_delta(1.0, Some(&exact), 0, 512, 8) - 0.88).abs() < 1e-12);
    }

    /// End-to-end apply-site check on synthetic module + calib parquets:
    /// context CSA subtracts the DELTA at the ORIGINAL query point
    /// (prefix, isl, b); generation CSA keys at (s_total - 1, isl=1, b); HCA
    /// is untouched; the same root WITHOUT the calib file returns the
    /// uncorrected leaves (absent-file no-op).
    #[test]
    fn dsv4_query_applies_topk_delta_end_to_end() {
        if !topk_correction_enabled() {
            return; // suite launched with AIC_DSV4_TOPK_CORRECTION=0
        }
        let spec = b200_sxm_spec();
        // (num_heads, batch_size, isl, step, latency)
        let csa_ctx_rows = [(64, 8, 512, 0, 1.0)];
        let hca_ctx_rows = [(64, 8, 512, 0, 0.7)];
        let csa_gen_rows = [(64, 16, 1, 384, 0.5)]; // s_total = 385
        let calib_rows = [
            ("v1_flat", 0, 512, 8, 0.30),
            ("v1_top_last", 0, 512, 8, 0.18), // context (v1) DELTA = 0.12
            ("v2_flat", 384, 1, 16, 0.05),
            ("v2_top_last", 384, 1, 16, 0.02), // generation (v2) DELTA = 0.03
        ];
        let make_root = |with_calib: bool| {
            let dir = tempfile::tempdir().unwrap();
            write_module_parquet(&dir.path().join("dsv4_csa_context_module_perf.parquet"), &csa_ctx_rows);
            write_module_parquet(&dir.path().join("dsv4_hca_context_module_perf.parquet"), &hca_ctx_rows);
            write_module_parquet(
                &dir.path().join("dsv4_csa_generation_module_perf.parquet"),
                &csa_gen_rows,
            );
            if with_calib {
                write_calib_parquet(&dir.path().join("dsv4_csa_topk_calib_perf.parquet"), &calib_rows);
            }
            dir
        };
        let q_ctx = |table: &Dsv4Table, kind| {
            table
                .query_context(
                    &spec, kind, 8, 512, 64, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM", 0, None,
                )
                .unwrap()
        };
        let q_gen = |table: &Dsv4Table, kind| {
            table
                .query_generation(
                    &spec, kind, 16, 385, 64, KvCacheQuantMode::Fp8,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM", None,
                )
                .unwrap()
        };

        let root = make_root(true);
        let table = Dsv4Table::new(root.path().to_path_buf());
        assert!((q_ctx(&table, AttnKind::Csa) - 0.88).abs() < 1e-12); // 1.0 - 0.12
        assert!((q_gen(&table, AttnKind::Csa) - 0.47).abs() < 1e-12); // 0.5 - 0.03
        assert!((q_ctx(&table, AttnKind::Hca) - 0.7).abs() < 1e-12); // HCA untouched

        let bare_root = make_root(false);
        let table = Dsv4Table::new(bare_root.path().to_path_buf());
        assert!((q_ctx(&table, AttnKind::Csa) - 1.0).abs() < 1e-12);
        assert!((q_gen(&table, AttnKind::Csa) - 0.5).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // CP sparse-kernel table (paged_mqa_logits) + raw top_last calib rows
    // (DeepSeek-V4 sparse-CP prefill composition)
    // ------------------------------------------------------------------

    /// Rows are `(num_heads, batch_size, isl, tp_size, step, latency)` — the
    /// sparse-kernel columns the loader reads (collector writes more; extras
    /// are ignored).
    fn write_sparse_kernel_parquet(path: &Path, rows: &[(i64, i64, i64, i64, i64, f64)]) {
        let schema = Arc::new(
            parse_message_type(
                "message sparse {
                    REQUIRED INT64 num_heads;
                    REQUIRED INT64 batch_size;
                    REQUIRED INT64 isl;
                    REQUIRED INT64 tp_size;
                    REQUIRED INT64 step;
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
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.0).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.4).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.5).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Like `write_calib_parquet` but with the optional `num_heads` column,
    /// exercising the Python `df[df["num_heads"] == native_heads]` filter.
    /// Rows are `(score_mode, step, isl, batch_size, num_heads, latency)`.
    fn write_calib_parquet_with_heads(path: &Path, rows: &[(&str, i64, i64, i64, i64, f64)]) {
        let schema = Arc::new(
            parse_message_type(
                "message calib {
                    REQUIRED BYTE_ARRAY score_mode (UTF8);
                    REQUIRED INT64 step;
                    REQUIRED INT64 isl;
                    REQUIRED INT64 batch_size;
                    REQUIRED INT64 num_heads;
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
        let modes: Vec<ByteArray> = rows.iter().map(|r| ByteArray::from(r.0)).collect();
        write_column::<ByteArrayType>(&mut rg, &modes);
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.1).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.2).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.3).collect::<Vec<_>>());
        write_column::<Int64Type>(&mut rg, &rows.iter().map(|r| r.4).collect::<Vec<_>>());
        write_column::<DoubleType>(&mut rg, &rows.iter().map(|r| r.5).collect::<Vec<_>>());
        rg.close().unwrap();
        writer.close().unwrap();
    }

    /// Mirrors Python `_lookup_sparse_kernel` resolution: exact grid hit; tp
    /// falls back to 1 when the requested tp is uncollected (paged_mqa_logits
    /// is collected at tp=1 only); missing head slice -> None; absent file ->
    /// None (the operator's fail-loud happens on top).
    #[test]
    fn paged_mqa_lookup_exact_tp_fallback_and_missing() {
        let dir = tempfile::tempdir().unwrap();
        write_sparse_kernel_parquet(
            &dir.path().join("dsv4_paged_mqa_logits_module_perf.parquet"),
            &[
                (64, 1, 8192, 1, 0, 0.2),
                (64, 1, 8192, 1, 8192, 0.3),
                (64, 1, 2048, 1, 0, 0.05),
            ],
        );
        let table = Dsv4Table::new(dir.path().to_path_buf());
        // Exact 3-axis hits.
        assert_eq!(table.query_paged_mqa_logits(1, 8192, 0, 1, 64).unwrap(), Some(0.2));
        assert_eq!(table.query_paged_mqa_logits(1, 8192, 8192, 1, 64).unwrap(), Some(0.3));
        // tp=8 not collected -> falls back to the tp=1 slice.
        assert_eq!(table.query_paged_mqa_logits(1, 8192, 0, 8, 64).unwrap(), Some(0.2));
        // Missing head slice -> None.
        assert_eq!(table.query_paged_mqa_logits(1, 8192, 0, 1, 32).unwrap(), None);
        // Absent file -> None.
        let empty = tempfile::tempdir().unwrap();
        let bare = Dsv4Table::new(empty.path().to_path_buf());
        assert_eq!(bare.query_paged_mqa_logits(1, 8192, 0, 1, 64).unwrap(), None);
    }

    /// Raw top_last retention + lookup (Python `_load_csa_topk_top_last` +
    /// `ContextDSAModule._lookup_2d`): top_last rows resolve by (isl, step)
    /// at the bs slice; flat rows stay out of the grid (they only feed the
    /// DELTA, which coexists in the same single load); an isl beyond the
    /// collected grid fails loud via `lookup_2d`; absent calib -> None.
    #[test]
    fn csa_topk_top_last_raw_rows_lookup() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dsv4_csa_topk_calib_perf.parquet");
        write_calib_parquet(
            &path,
            &[
                ("v1_top_last", 0, 16384, 1, 800.0),
                ("v1_top_last", 0, 2048, 1, 100.0),
                ("v1_flat", 0, 2048, 1, 130.0),
            ],
        );
        let table = Dsv4Table::new(dir.path().to_path_buf());
        // num_heads column absent -> no filter (any native_heads resolves).
        assert_eq!(table.csa_topk_top_last(16384, 0, 64, 1).unwrap(), Some(800.0));
        // flat row (130.0) must not shadow the top_last value.
        assert_eq!(table.csa_topk_top_last(2048, 0, 64, 1).unwrap(), Some(100.0));
        // The DELTA table coexists in the same load: (0, 2048, 1) pairs up.
        let calib = table.load_topk_calib().unwrap().expect("calib must load");
        assert!((topk_delta_ms(&calib.exact_v1, 0, 2048, 1) - 30.0).abs() < 1e-12);
        // isl beyond the collected grid -> fail loud (dsa::lookup_2d contract).
        let err = table.csa_topk_top_last(32768, 0, 64, 1).unwrap_err();
        assert!(err.to_string().contains("exceeds the collected"), "unexpected: {err}");
        // Absent calib -> None (operator fails loud on top).
        let empty = tempfile::tempdir().unwrap();
        let bare = Dsv4Table::new(empty.path().to_path_buf());
        assert_eq!(bare.csa_topk_top_last(2048, 0, 64, 1).unwrap(), None);
    }

    /// num_heads filter parity: when the calib file carries the column,
    /// only rows matching the queried native_heads resolve (Python
    /// `df[df["num_heads"] == native_heads]`); a head count with no rows
    /// yields None (-> operator fail-loud), never another head's latency.
    #[test]
    fn csa_topk_top_last_filters_num_heads_when_present() {
        let dir = tempfile::tempdir().unwrap();
        write_calib_parquet_with_heads(
            &dir.path().join("dsv4_csa_topk_calib_perf.parquet"),
            &[
                ("v1_top_last", 0, 2048, 1, 64, 42.0),
                ("v1_top_last", 0, 2048, 1, 128, 77.0),
            ],
        );
        let table = Dsv4Table::new(dir.path().to_path_buf());
        assert_eq!(table.csa_topk_top_last(2048, 0, 64, 1).unwrap(), Some(42.0));
        assert_eq!(table.csa_topk_top_last(2048, 0, 128, 1).unwrap(), Some(77.0));
        assert_eq!(table.csa_topk_top_last(2048, 0, 32, 1).unwrap(), None);
    }
}
