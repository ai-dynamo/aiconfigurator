// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State-space layer perf tables: Mamba2, Gated Delta Network (GDN) and
//! Kimi Delta Attention (KDA).
//!
//! Used by hybrid models such as Nemotron-H, Qwen3.5 and Kimi-K3. The files
//! share a similar shape: a `phase` discriminator (`context` /
//! `generation`, plus `verify` for KDA), a model-name key, and several
//! layer-specific dimension columns.
//!
//! Resolution mirrors Python v2 (`operations/mamba.py` + the perf_interp
//! engine): after shape-key selection, context queries are a 2-axis Grid
//! RAW engine query over `[batch][seq_len]`, generation queries a 1-axis
//! Grid RAW query over `[batch]` (generation rows are collected at a single
//! seq, and the query's seq only feeds SOL — where generation formulas
//! ignore it). The operator layer owns the SOL closure and the SOL
//! degradation contract: any `PerfDatabase` error here makes the op fall
//! back to its analytic `sol_latency_ms` (`source="sol"`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::perf_interp::{self, LeafValue, Node, OpInterpConfig};
use super::{kernel_source_ok, SourceResolver};
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use crate::perf_database::parquet_loader::PerfReader;

pub struct StateSpaceTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for each state-space perf file
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`StateSpaceTable::new`).
    mamba2_sources: Vec<PerfSource>,
    gdn_sources: Vec<PerfSource>,
    kda_sources: Vec<PerfSource>,
    vllm_024_gdn_aliases: bool,
    /// SM-major-10 (sm 100/103, NOT sm 120) sglang serving auto-selects the
    /// FlashInfer bf16-state GDN decode kernel when the model's
    /// `mamba_ssm_dtype` is bfloat16 (see `query_gdn`'s alias branch below —
    /// the dtype half of the predicate is per-query). Computed once at
    /// construction, same convention as `vllm_024_gdn_aliases`.
    sglang_sm100_gdn_flashinfer_lane: bool,
    mamba2: OnceLock<Result<Mamba2Grids, AicError>>,
    gdn: OnceLock<Result<GdnGrids, AicError>>,
    kda: OnceLock<Result<KdaGrids, AicError>>,
}

/// Per shape-key engine table. Context keys hold a 2-level `[batch][seq]`
/// node; generation keys hold a 1-level `[batch]` node (Python v2 keys
/// generation leaves by batch only).
struct Mamba2Grids {
    by_keys: BTreeMap<Mamba2Key, Node>,
}

struct GdnGrids {
    by_keys: BTreeMap<GdnKey, Node>,
}

struct KdaGrids {
    by_keys: BTreeMap<KdaKey, Node>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Mamba2Key {
    /// Kernel routine name (e.g. `causal_conv1d_fn` /
    /// `causal_conv1d_update`); discriminates between context and
    /// generation kernels that share the rest of the shape.
    kernel_source: String,
    phase: String,
    d_model: u32,
    d_state: u32,
    d_conv: u32,
    nheads: u32,
    head_dim: u32,
    n_groups: u32,
    chunk_size: u32,
    // Note: Python keys by SHAPE tuple, not by `model_name`. The CSV's
    // `model_name` column is metadata identifying which model the row
    // was collected against; the lookup itself is shape-based, so a
    // matching shape is reused across model names. We mirror that.
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GdnKey {
    kernel_source: String,
    phase: String,
    d_model: u32,
    d_conv: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    num_v_heads: u32,
    head_v_dim: u32,
    // See `Mamba2Key`: shape is the key, `model_name` is metadata.
}

/// Same structural key as [`GdnKey`] — KDA shares the GDN shape tuple but is
/// a distinct kernel family collected into `kda_perf.parquet`. Its `phase`
/// also spans "verify" (2-axis `[batch][draft_tokens]` leaves).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct KdaKey {
    kernel_source: String,
    phase: String,
    d_model: u32,
    d_conv: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    num_v_heads: u32,
    head_v_dim: u32,
    // See `Mamba2Key`: shape is the key, `model_name` is metadata.
}

impl StateSpaceTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf, backend: &str, version: &str) -> Self {
        Self::with_sources(
            data_root,
            backend,
            version,
            None,
            &SourceResolver::fixed(PerfDbSources::default()),
        )
        .expect("fixed-map resolution is infallible")
    }

    /// Construct with shared-layer (sibling/cross-version) sources supplied by the
    /// engine's `SourceResolver` (live resolution owns the shared-layer walk;
    /// a fixed source map is the test-only path). Each state-space file falls back to
    /// its primary `data_root/<basename>` when the resolver names no override. No I/O.
    ///
    /// `sm_version` is the system spec's `gpu.sm_version` (absent on systems
    /// that don't declare one) — needed to gate the SM-major-10 sglang GDN
    /// flashinfer-lane branch in `query_gdn`.
    pub fn with_sources(
        data_root: PathBuf,
        backend: &str,
        version: &str,
        sm_version: Option<u32>,
        resolver: &SourceResolver,
    ) -> Result<Self, AicError> {
        let mamba2_sources = resolver.sources_for("mamba2_perf.parquet", &data_root)?;
        let gdn_sources = resolver.sources_for("gdn_perf.parquet", &data_root)?;
        let kda_sources = resolver.sources_for("kda_perf.parquet", &data_root)?;
        Ok(Self {
            data_root,
            mamba2_sources,
            gdn_sources,
            kda_sources,
            vllm_024_gdn_aliases: backend == "vllm" && version == "0.24.0",
            // Mirror sglang's `is_sm100_supported` (utils/common.py @ pinned
            // v0.5.14 clone): device capability major EXACTLY 10, so sm
            // 100/103 qualify and sm 120 (rtx_pro_6000_server) does not.
            sglang_sm100_gdn_flashinfer_lane: backend == "sglang"
                && matches!(sm_version, Some(v) if (100..110).contains(&v)),
            mamba2: OnceLock::new(),
            gdn: OnceLock::new(),
            kda: OnceLock::new(),
        })
    }

    /// Mamba2 latency for a layer instance, resolved on the perf_interp
    /// engine (context: 2-axis `(batch, seq_len)` Grid RAW; see module
    /// doc). `sol` is the operator's analytic SOL in `(batch, seq)` order
    /// — it anchors util-hold beyond the collected range.
    #[allow(clippy::too_many_arguments)]
    pub fn query_mamba2(
        &self,
        kernel_source: &str,
        phase: &str,
        batch_size: u32,
        seq_len: u32,
        d_model: u32,
        d_state: u32,
        d_conv: u32,
        nheads: u32,
        head_dim: u32,
        n_groups: u32,
        chunk_size: u32,
        sol: &dyn Fn(f64, f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        // Mirror Python v2's `load_mamba2_data` defaultdict bug (still
        // present today, verified 2026-07-09): the row-population pattern
        // `try { data[ks][ph][mk][bs] } except KeyError: ... = entry` never
        // reaches the `except` branch for generation rows because the
        // fourth level is a `defaultdict(dict)` that lazily materialises an
        // empty dict on `[bs]` access (no KeyError). Generation leaves end
        // up as empty `{}`; `_query_mamba2_table`'s generation branch then
        // normalizes them to an empty curve, the perf_interp engine raises,
        // and the op falls back to SOL (`db.query_mamba2(generation)`
        // returns `source="sol"` for every query). The Rust parquet loader
        // populates the rows correctly, so returning silicon here would
        // give a numerically different (and arguably "more correct")
        // answer — but for apple-to-apple parity we mirror Python by
        // returning a PerfDatabase error so the operator-layer SOL branch
        // fires. GDN's loader is fine (uses explicit `in` checks), so this
        // workaround is Mamba2-generation-only.
        if phase == "generation" {
            return Err(AicError::PerfDatabase(format!(
                "Mamba2 generation data intentionally not used (matches Python v2 \
                 `load_mamba2_data` defaultdict bug in operations/mamba.py — generation \
                 leaves load empty, so every generation query degrades to SOL); \
                 operator must fall to SOL. ks={kernel_source}, d_model={d_model}"
            )));
        }
        let grids = self.load_mamba2()?;
        let key = Mamba2Key {
            kernel_source: kernel_source.to_string(),
            phase: phase.to_string(),
            d_model,
            d_state,
            d_conv,
            nheads,
            head_dim,
            n_groups,
            chunk_size,
        };
        // Mirror Python `_query_mamba2_table`: on exact-shape miss, fall back
        // to the first table entry sharing the same `d_model` (insertion order
        // in Python; sorted order here — which agrees whenever the per-d_model
        // bucket has a single entry, as in all current matrices). If no entry
        // shares d_model, surface as `PerfDatabase` so the operator layer's
        // SOL fallback applies.
        //
        // Only the context phase reaches this point — generation queries are
        // short-circuited above to match Python's degenerate behaviour.
        let node = match grids.by_keys.get(&key) {
            Some(node) => node,
            None => grids
                .by_keys
                .iter()
                .find(|(k, _)| {
                    k.kernel_source == key.kernel_source
                        && k.phase == key.phase
                        && k.d_model == key.d_model
                })
                .map(|(_, node)| node)
                .ok_or_else(|| missing("Mamba2", &self.data_root, format!("{key:?}")))?,
        };
        engine_query(node, phase, batch_size, seq_len, sol)
    }

    /// GDN latency for a layer instance. Same engine resolution as Mamba2
    /// (and, unlike Mamba2, generation queries really resolve: 1-axis
    /// `(batch,)` Grid RAW over the generation curve, per Python v2).
    #[allow(clippy::too_many_arguments)]
    pub fn query_gdn(
        &self,
        kernel_source: &str,
        phase: &str,
        batch_size: u32,
        seq_len: u32,
        d_model: u32,
        d_conv: u32,
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
        mamba_ssm_dtype: &str,
        sol: &dyn Fn(f64, f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        let grids = self.load_gdn()?;
        let key = GdnKey {
            kernel_source: kernel_source.to_string(),
            phase: phase.to_string(),
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
        };
        // Mirror Python `_query_gdn_table`: exact geometry (or an exact
        // physical-alias hit) only; any miss surfaces as `PerfDatabase` so
        // the operator degrades to SOL.
        //
        // The framework's own persisted physical kernels (vLLM 0.24 names its
        // context scan chunk_gated_delta_rule_*; SM100+ sglang decode prefers
        // flashinfer_gated_delta_rule_decode) take precedence: after the
        // shared-layer merge the logical lane can hold cross-backend donor
        // rows, which only serve as gap fill when no own physical lane covers
        // the shape. Ambiguous physical data fails closed.
        let aliases: &[&str] = if self.vllm_024_gdn_aliases {
            match (key.kernel_source.as_str(), key.phase.as_str()) {
                ("chunk_gated_delta_rule", "context") => &[
                    "chunk_gated_delta_rule_flashinfer",
                    "chunk_gated_delta_rule_triton",
                    "chunk_gated_delta_rule_cutedsl",
                ],
                ("fused_sigmoid_gating_delta_rule_update", "generation") => {
                    &["fused_recurrent_gated_delta_rule_packed_decode"]
                }
                _ => &[],
            }
        } else if self.sglang_sm100_gdn_flashinfer_lane
            && mamba_ssm_dtype == "bfloat16"
            && key.phase == "generation"
            && key.kernel_source == "fused_sigmoid_gating_delta_rule_update"
        {
            // SM-major-10 sglang serving auto-selects the FlashInfer
            // bf16-state GDN decode kernel ONLY when the model's
            // mamba_ssm_dtype is bfloat16 (server_args.py's
            // _handle_linear_attn_backend, server_args.py:4884-4915 @ pinned
            // v0.5.14 clone: `is_sm100_supported()` — capability major
            // exactly 10 — AND `mamba_ssm_dtype == "bfloat16"`): prefer its
            // own rows over the fla/triton fp32-state lane when they cover
            // this shape. Every bundled Qwen3.5/3.6 config pins
            // mamba_ssm_dtype=float32, so the default query stays on the fla
            // lane; tables collected before AIC-1745's collector change carry
            // no flashinfer_gated_delta_rule_decode rows anyway --
            // alias_matches stays empty below and this degrades to the fla
            // lane exactly as before. SM90 and SM120 are untouched (fla lane
            // always wins there, matching serving's default).
            &["flashinfer_gated_delta_rule_decode"]
        } else {
            &[]
        };
        let alias_matches: Vec<_> = aliases
            .iter()
            .filter_map(|alias| {
                let mut alias_key = key.clone();
                alias_key.kernel_source = (*alias).to_string();
                grids.by_keys.get_key_value(&alias_key)
            })
            .collect();
        if alias_matches.len() > 1 {
            let sources: Vec<_> = alias_matches
                .iter()
                .map(|(alias_key, _)| alias_key.kernel_source.as_str())
                .collect();
            return Err(AicError::PerfDatabase(format!(
                "ambiguous vLLM 0.24.0 GDN physical kernels for {key:?}: {}",
                sources.join(", ")
            )));
        }
        if let Some((_, node)) = alias_matches.first() {
            // TRIPWIRE: `sol` still closes over the CALLER's original
            // kernel_source (the operator's own `sol_latency_ms`/`kernel_source`
            // field), never the winning alias -- a flashinfer-lane row served
            // here still extrapolates beyond-range with the fla lane's 4-byte
            // state formula, not the flashinfer lane's 2-byte one. Deliberate,
            // not a bug: GDN decode SOL is linear/homogeneous in batch with no
            // additive constant, so the state_bytes constant cancels exactly in
            // perf_interp's util-hold ratio (sol_q / median(sol(cv_i)/lat(cv_i))
            // -- verified algebraically, see AIC-1745 Task 3 (Python commit
            // 87e7c03, mamba.py). Revisit if a future GDN-family SOL formula
            // stops being linear in the query axis.
            return engine_query(node, phase, batch_size, seq_len, sol);
        }
        let node = match grids.by_keys.get(&key) {
            Some(node) => node,
            None => return Err(missing("GDN", &self.data_root, format!("{key:?}"))),
        };
        engine_query(node, phase, batch_size, seq_len, sol)
    }

    /// KDA (Kimi Delta Attention) latency for a layer instance. Same engine
    /// resolution as GDN, plus a "verify" phase that — like context — is a
    /// 2-axis `(batch, seq_len)` Grid RAW query (the caller passes
    /// `seq_len = draft_tokens` and the verify-normalized batch; see
    /// `operators/mamba.rs::KdaOp::effective_coords`). Generation is the
    /// 1-axis `(batch,)` curve. Unlike GDN there is NO physical
    /// kernel-source alias map — only the same-`d_model` nearest-shard
    /// fallback (collector rows are per-TP shard).
    #[allow(clippy::too_many_arguments)]
    pub fn query_kda(
        &self,
        kernel_source: &str,
        phase: &str,
        batch_size: u32,
        seq_len: u32,
        d_model: u32,
        d_conv: u32,
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
        sol: &dyn Fn(f64, f64) -> f64,
    ) -> Result<LeafValue, AicError> {
        let grids = self.load_kda()?;
        let key = KdaKey {
            kernel_source: kernel_source.to_string(),
            phase: phase.to_string(),
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
        };
        // Mirror Python `_query_kda_table`: on exact-shape miss, fall back to
        // any same-d_model entry within the SAME kernel source and phase,
        // breaking ties by minimum `|num_v_heads - query.num_v_heads|`.
        // Surface as `PerfDatabase` if no d_model match exists so the
        // operator's SOL fallback fires.
        let node = match grids.by_keys.get(&key) {
            Some(node) => node,
            None => {
                let nearest = grids
                    .by_keys
                    .iter()
                    .filter(|(k, _)| {
                        k.kernel_source == key.kernel_source
                            && k.phase == key.phase
                            && k.d_model == key.d_model
                    })
                    .min_by_key(|(k, _)| (k.num_v_heads as i64 - key.num_v_heads as i64).abs());
                match nearest {
                    Some((_, node)) => node,
                    None => return Err(missing("KDA", &self.data_root, format!("{key:?}"))),
                }
            }
        };
        engine_query(node, phase, batch_size, seq_len, sol)
    }

    fn load_mamba2(&self) -> Result<&Mamba2Grids, AicError> {
        let cell = self
            .mamba2
            .get_or_init(|| load_mamba2_parquet(&self.mamba2_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_gdn(&self) -> Result<&GdnGrids, AicError> {
        let cell = self.gdn.get_or_init(|| load_gdn_parquet(&self.gdn_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_kda(&self) -> Result<&KdaGrids, AicError> {
        let cell = self.kda.get_or_init(|| load_kda_parquet(&self.kda_sources));
        cell.as_ref().map_err(clone_err)
    }

    /// Whether the loaded KDA table holds any verify-phase rows for
    /// `kernel_source`, across every model key. Mirrors the Python
    /// `_has_verify_rows` probe in `KDAKernel._query_kda_table` (detects
    /// fused-CuTeDSL SM100 verify datasets); `false` when the KDA table
    /// itself is absent.
    pub fn kda_has_verify_rows(&self, kernel_source: &str) -> bool {
        match self.load_kda() {
            Ok(grids) => grids
                .by_keys
                .keys()
                .any(|k| k.kernel_source == kernel_source && k.phase == "verify"),
            Err(_) => false,
        }
    }

    /// Whether the loaded KDA table holds rows for exactly this
    /// (kernel_source, phase, model dims) key. Python twin: the per-model-key
    /// membership checks in `KDAKernel._query_kda_table` that route the
    /// fused-decode shard (the 12-head TP8 shard has a single
    /// `kda_fused_decode` generation row and no Triton pair).
    #[allow(clippy::too_many_arguments)]
    pub fn kda_has_key(
        &self,
        kernel_source: &str,
        phase: &str,
        d_model: u32,
        d_conv: u32,
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
    ) -> bool {
        match self.load_kda() {
            Ok(grids) => grids.by_keys.contains_key(&KdaKey {
                kernel_source: kernel_source.to_string(),
                phase: phase.to_string(),
                d_model,
                d_conv,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim,
            }),
            Err(_) => false,
        }
    }
}

/// One perf_interp engine query per phase, mirroring Python v2's
/// `_query_mamba2_table` / `_query_gdn_table`:
///
/// - context (and KDA's verify, which shares the 2-axis layout with
///   `seq_len = draft_tokens`): `axes=("batch", "seq_len")`, Grid RAW,
///   coords `(b, s)` — the same axis order as the Python record.
/// - generation: `axes=("batch",)`, Grid RAW over the per-batch curve. The
///   query's `seq_len` is forwarded to `sol` only (Python passes the op's
///   `seq_len=None` there; generation SOL formulas ignore it either way).
fn engine_query(
    node: &Node,
    phase: &str,
    batch_size: u32,
    seq_len: u32,
    sol: &dyn Fn(f64, f64) -> f64,
) -> Result<LeafValue, AicError> {
    if phase == "generation" {
        let s = seq_len as f64;
        let sol1 = move |c: &[f64]| sol(c[0], s);
        let cfg = OpInterpConfig::grid(&["batch"], &sol1);
        perf_interp::query_value(&cfg, node, &[batch_size as f64])
    } else {
        // Python: `if seq_len is None or seq_len <= 0: return SOL` — surface
        // as a PerfDatabase error so the operator's SOL branch fires.
        if seq_len == 0 {
            return Err(AicError::PerfDatabase(
                "state-space context/verify query needs seq_len > 0".to_string(),
            ));
        }
        let sol2 = move |c: &[f64]| sol(c[0], c[1]);
        let cfg = OpInterpConfig::grid(&["batch", "seq_len"], &sol2);
        perf_interp::query_value(&cfg, node, &[batch_size as f64, seq_len as f64])
    }
}

/// First-wins measured-leaf insert (Python loaders skip rows whose
/// coordinate is already populated; `Node::insert_value` would overwrite).
fn insert_first_wins(root: &mut Node, path: &[u32], value: LeafValue) {
    let Node::Branch(map) = root else {
        return; // malformed nesting; keep the earlier row
    };
    if path.len() == 1 {
        map.entry(path[0]).or_insert(Node::Leaf(value));
    } else {
        let child = map.entry(path[0]).or_insert_with(Node::branch);
        insert_first_wins(child, &path[1..], value);
    }
}

/// Load the Mamba2 table from an ordered, priority-sorted source list. Sources
/// are read in order; the first source containing a coordinate wins
/// (`insert_first_wins`), mirroring Python's `_read_filtered_rows` concatenation
/// + `load_mamba2_data` skip-on-key-conflict. Missing files are skipped (a
/// sibling declared in the manifest need not exist for every system); an error
/// is returned only when no source yields rows.
fn load_mamba2_parquet(sources: &[PerfSource]) -> Result<Mamba2Grids, AicError> {
    let mut by_keys: BTreeMap<Mamba2Key, Node> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let phase_col = reader.col("phase")?;
        let batch_size_col = reader.col("batch_size")?;
        let seq_len_col = reader.col("seq_len")?;
        let d_model_col = reader.col("d_model")?;
        let d_state_col = reader.col("d_state")?;
        let d_conv_col = reader.col("d_conv")?;
        let nheads_col = reader.col("nheads")?;
        let head_dim_col = reader.col("head_dim")?;
        let n_groups_col = reader.col("n_groups")?;
        let chunk_size_col = reader.col("chunk_size")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let phase = row.str_owned(phase_col)?;
            let key = Mamba2Key {
                kernel_source: row.str_owned(kernel_source_col)?,
                phase: phase.clone(),
                d_model: row.u32(d_model_col)?,
                d_state: row.u32(d_state_col)?,
                d_conv: row.u32(d_conv_col)?,
                nheads: row.u32(nheads_col)?,
                head_dim: row.u32(head_dim_col)?,
                n_groups: row.u32(n_groups_col)?,
                chunk_size: row.u32(chunk_size_col)?,
            };
            // First-wins parity with Python `load_mamba2_data`, extended across
            // shared-layer sources (earlier source wins). Generation rows are
            // keyed by batch only (Python drops seq for generation). Note the
            // stored generation leaves are never read: `query_mamba2` mirrors
            // Python's empty-generation-leaves bug by erroring first.
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            let batch = row.u32(batch_size_col)?;
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            let leaf = LeafValue::with_power(latency, power);
            if phase == "generation" {
                insert_first_wins(node, &[batch], leaf);
            } else {
                insert_first_wins(node, &[batch, row.u32(seq_len_col)?], leaf);
            }
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no Mamba2 rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(Mamba2Grids { by_keys })
}

/// Load the GDN table from an ordered, priority-sorted source list. Same
/// first-wins-across-sources + missing-file-skip semantics as
/// [`load_mamba2_parquet`].
/// The GDN decode-recurrence kernel name drifted across sglang releases
/// (0.5.10: `fused_recurrent_gated_delta_rule`; 0.5.14 records the executed
/// `fused_recurrent_gated_delta_rule_packed_decode`). Consumers query one
/// canonical modeling identity; normalize the LOOKUP key here (mirrors
/// Python `_GDN_DECODE_RECURRENCE_ALIASES`) — the parquet keeps the
/// executed-kernel truth.
pub(crate) fn normalize_gdn_kernel_source(kernel_source: String) -> String {
    match kernel_source.as_str() {
        "fused_recurrent_gated_delta_rule" | "fused_recurrent_gated_delta_rule_packed_decode" => {
            "fused_sigmoid_gating_delta_rule_update".to_string()
        }
        _ => kernel_source,
    }
}

fn load_gdn_parquet(sources: &[PerfSource]) -> Result<GdnGrids, AicError> {
    let mut by_keys: BTreeMap<GdnKey, Node> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let phase_col = reader.col("phase")?;
        let batch_size_col = reader.col("batch_size")?;
        let seq_len_col = reader.col("seq_len")?;
        let d_model_col = reader.col("d_model")?;
        let d_conv_col = reader.col("d_conv")?;
        let num_k_heads_col = reader.col("num_k_heads")?;
        let head_k_dim_col = reader.col("head_k_dim")?;
        let num_v_heads_col = reader.col("num_v_heads")?;
        let head_v_dim_col = reader.col("head_v_dim")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let phase = row.str_owned(phase_col)?;
            let key = GdnKey {
                kernel_source: normalize_gdn_kernel_source(row.str_owned(kernel_source_col)?),
                phase: phase.clone(),
                d_model: row.u32(d_model_col)?,
                d_conv: row.u32(d_conv_col)?,
                num_k_heads: row.u32(num_k_heads_col)?,
                head_k_dim: row.u32(head_k_dim_col)?,
                num_v_heads: row.u32(num_v_heads_col)?,
                head_v_dim: row.u32(head_v_dim_col)?,
            };
            // First-wins parity with Python `load_gdn_data`, extended across
            // shared-layer sources (earlier source wins): context leaves at
            // `[batch][seq]`, generation leaves at `[batch]`.
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            let batch = row.u32(batch_size_col)?;
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            let leaf = LeafValue::with_power(latency, power);
            if phase == "generation" {
                insert_first_wins(node, &[batch], leaf);
            } else {
                insert_first_wins(node, &[batch, row.u32(seq_len_col)?], leaf);
            }
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no GDN rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(GdnGrids { by_keys })
}

/// Load the KDA table from an ordered, priority-sorted source list. Same
/// first-wins-across-sources + missing-file-skip semantics as
/// [`load_gdn_parquet`], and the same column layout as `gdn_perf.parquet`.
/// Unlike GDN there is NO kernel-source normalization on load (Python
/// `load_kda_data` keeps the recorded name; SOL-side aliasing lives in the
/// operator's byte model only). "context" AND "verify" rows nest
/// `[batch][seq]` (`seq_len` carries the per-request draft-token count for
/// verify rows); "generation" rows nest `[batch]` only.
fn load_kda_parquet(sources: &[PerfSource]) -> Result<KdaGrids, AicError> {
    let mut by_keys: BTreeMap<KdaKey, Node> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let phase_col = reader.col("phase")?;
        let batch_size_col = reader.col("batch_size")?;
        let seq_len_col = reader.col("seq_len")?;
        let d_model_col = reader.col("d_model")?;
        let d_conv_col = reader.col("d_conv")?;
        let num_k_heads_col = reader.col("num_k_heads")?;
        let head_k_dim_col = reader.col("head_k_dim")?;
        let num_v_heads_col = reader.col("num_v_heads")?;
        let head_v_dim_col = reader.col("head_v_dim")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let phase = row.str_owned(phase_col)?;
            let key = KdaKey {
                kernel_source: row.str_owned(kernel_source_col)?,
                phase: phase.clone(),
                d_model: row.u32(d_model_col)?,
                d_conv: row.u32(d_conv_col)?,
                num_k_heads: row.u32(num_k_heads_col)?,
                head_k_dim: row.u32(head_k_dim_col)?,
                num_v_heads: row.u32(num_v_heads_col)?,
                head_v_dim: row.u32(head_v_dim_col)?,
            };
            // First-wins parity with Python `load_kda_data`, extended across
            // shared-layer sources (earlier source wins): context AND verify
            // leaves at `[batch][seq]`, generation leaves at `[batch]`.
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            let batch = row.u32(batch_size_col)?;
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            let leaf = LeafValue::with_power(latency, power);
            if phase == "context" || phase == "verify" {
                insert_first_wins(node, &[batch, row.u32(seq_len_col)?], leaf);
            } else {
                insert_first_wins(node, &[batch], leaf);
            }
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no KDA rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(KdaGrids { by_keys })
}

fn missing(table: &str, data_root: &Path, descriptor: String) -> AicError {
    AicError::PerfDatabase(format!(
        "{table} data missing for {descriptor} at {}",
        data_root.display()
    ))
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::system_spec::SystemSpec;

    fn data_root(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/data")
            .join(rel)
    }

    fn h100_sxm_mem_bw() -> f64 {
        let yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator_core/systems/h100_sxm.yaml");
        SystemSpec::load(&yaml)
            .expect("h100_sxm.yaml must parse")
            .gpu
            .mem_bw
    }

    fn dummy_sol(_b: f64, _s: f64) -> f64 {
        1.0
    }

    /// In-memory GDN table over one fixed model shape (d_model=5120,
    /// heads 16/128, v-dim 128), varying only `(kernel_source, phase,
    /// num_v_heads)`. Context rows land at `[batch=1][seq=1024]`, generation
    /// rows at `[batch=1]`, matching the loader's leaf layout — queries below
    /// hit those coordinates exactly, so engine RAW returns the stored value.
    fn in_memory_gdn_table(
        backend: &str,
        version: &str,
        rows: &[(&str, &str, u32, f64)],
    ) -> StateSpaceTable {
        let mut by_keys: BTreeMap<GdnKey, Node> = BTreeMap::new();
        for &(kernel_source, phase, num_v_heads, latency) in rows {
            let key = GdnKey {
                kernel_source: kernel_source.to_string(),
                phase: phase.to_string(),
                d_model: 5120,
                d_conv: 4,
                num_k_heads: 16,
                head_k_dim: 128,
                num_v_heads,
                head_v_dim: 128,
            };
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            let leaf = LeafValue::latency_only(latency);
            if phase == "generation" {
                insert_first_wins(node, &[1], leaf);
            } else {
                insert_first_wins(node, &[1, 1024], leaf);
            }
        }
        let table = StateSpaceTable::new(PathBuf::from("test-data"), backend, version);
        assert!(table.gdn.set(Ok(GdnGrids { by_keys })).is_ok());
        table
    }

    fn query_gdn_test_shape(
        table: &StateSpaceTable,
        kernel_source: &str,
        phase: &str,
        num_v_heads: u32,
    ) -> Result<f64, AicError> {
        // Default state dtype (every bundled Qwen3.5/3.6 config pins
        // float32); the flashinfer-lane tests below pass "bfloat16"
        // explicitly via query_gdn_test_shape_with_dtype.
        query_gdn_test_shape_with_dtype(table, kernel_source, phase, num_v_heads, "float32")
    }

    fn query_gdn_test_shape_with_dtype(
        table: &StateSpaceTable,
        kernel_source: &str,
        phase: &str,
        num_v_heads: u32,
        mamba_ssm_dtype: &str,
    ) -> Result<f64, AicError> {
        table
            .query_gdn(
                kernel_source,
                phase,
                1,
                1024,
                5120,
                4,
                16,
                128,
                num_v_heads,
                128,
                mamba_ssm_dtype,
                &dummy_sol,
            )
            .map(|v| v.latency)
    }

    #[test]
    fn vllm_024_gdn_resolves_context_and_generation_physical_aliases() {
        for source in [
            "chunk_gated_delta_rule_flashinfer",
            "chunk_gated_delta_rule_triton",
            "chunk_gated_delta_rule_cutedsl",
        ] {
            let table = in_memory_gdn_table("vllm", "0.24.0", &[(source, "context", 48, 2.0)]);
            assert_eq!(
                query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).unwrap(),
                2.0
            );
        }

        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[(
                "fused_recurrent_gated_delta_rule_packed_decode",
                "generation",
                48,
                3.0,
            )],
        );
        assert_eq!(
            query_gdn_test_shape(
                &table,
                "fused_sigmoid_gating_delta_rule_update",
                "generation",
                48,
            )
            .unwrap(),
            3.0
        );
    }

    #[test]
    fn vllm_024_gdn_own_physical_lane_wins_over_logical_lane() {
        // The logical lane can hold cross-backend donor rows after the
        // shared-layer merge; the own physical lane must beat it.
        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[
                ("chunk_gated_delta_rule", "context", 48, 1.0),
                ("chunk_gated_delta_rule_flashinfer", "context", 48, 2.0),
            ],
        );
        assert_eq!(
            query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).unwrap(),
            2.0
        );
    }

    #[test]
    fn gdn_physical_aliases_are_vllm_024_only() {
        for (backend, version) in [("vllm", "0.23.0"), ("sglang", "0.24.0")] {
            let table = in_memory_gdn_table(
                backend,
                version,
                &[("chunk_gated_delta_rule_flashinfer", "context", 48, 2.0)],
            );
            assert!(query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).is_err());
        }
    }

    #[test]
    fn vllm_024_gdn_ambiguous_exact_aliases_error() {
        // The logical-lane row must not mask the ambiguity between physical lanes.
        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[
                ("chunk_gated_delta_rule", "context", 48, 1.0),
                ("chunk_gated_delta_rule_flashinfer", "context", 48, 2.0),
                ("chunk_gated_delta_rule_triton", "context", 48, 3.0),
            ],
        );
        match query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48) {
            Err(AicError::PerfDatabase(message)) => {
                assert!(message.contains("ambiguous vLLM 0.24.0 GDN physical kernels"));
                assert!(message.contains("chunk_gated_delta_rule_flashinfer"));
                assert!(message.contains("chunk_gated_delta_rule_triton"));
            }
            other => panic!("expected an explicit ambiguity error, got {other:?}"),
        }
    }

    #[test]
    fn vllm_024_gdn_alias_has_no_nearest_shape_fallback() {
        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[("chunk_gated_delta_rule_flashinfer", "context", 32, 2.0)],
        );
        assert!(query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).is_err());
    }

    #[test]
    fn vllm_024_gdn_does_not_borrow_nearest_shape_within_logical_source() {
        // Exact geometry only: nearest-num_v_heads rows are never returned as
        // silicon (mirrors the Python twin test).
        let table = in_memory_gdn_table(
            "vllm",
            "0.24.0",
            &[
                ("chunk_gated_delta_rule_flashinfer", "context", 32, 2.0),
                ("chunk_gated_delta_rule", "context", 16, 4.0),
                ("chunk_gated_delta_rule", "context", 64, 5.0),
            ],
        );
        assert!(query_gdn_test_shape(&table, "chunk_gated_delta_rule", "context", 48).is_err());
    }

    /// Like `in_memory_gdn_table`, but threads an explicit `sm_version`
    /// through `StateSpaceTable::with_sources` -- needed to exercise the
    /// SM100+ sglang GDN flashinfer-lane branch below (`in_memory_gdn_table`'s
    /// `StateSpaceTable::new` always passes `sm_version=None`, so it can't
    /// reach that branch).
    fn in_memory_gdn_table_with_sm(
        backend: &str,
        version: &str,
        sm_version: Option<u32>,
        rows: &[(&str, &str, u32, f64)],
    ) -> StateSpaceTable {
        let mut by_keys: BTreeMap<GdnKey, Node> = BTreeMap::new();
        for &(kernel_source, phase, num_v_heads, latency) in rows {
            let key = GdnKey {
                kernel_source: kernel_source.to_string(),
                phase: phase.to_string(),
                d_model: 5120,
                d_conv: 4,
                num_k_heads: 16,
                head_k_dim: 128,
                num_v_heads,
                head_v_dim: 128,
            };
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            let leaf = LeafValue::latency_only(latency);
            if phase == "generation" {
                insert_first_wins(node, &[1], leaf);
            } else {
                insert_first_wins(node, &[1, 1024], leaf);
            }
        }
        let table = StateSpaceTable::with_sources(
            PathBuf::from("test-data"),
            backend,
            version,
            sm_version,
            &SourceResolver::fixed(PerfDbSources::default()),
        )
        .expect("fixed-map resolution is infallible");
        assert!(table.gdn.set(Ok(GdnGrids { by_keys })).is_ok());
        table
    }

    #[test]
    fn sglang_sm100_gdn_prefers_flashinfer_decode_lane_for_bf16_state() {
        // Serving predicate (server_args.py:4884-4915 @ pinned v0.5.14):
        // capability major exactly 10 AND mamba_ssm_dtype == "bfloat16".
        let table = in_memory_gdn_table_with_sm(
            "sglang",
            "0.5.14",
            Some(103),
            &[
                (
                    "fused_sigmoid_gating_delta_rule_update",
                    "generation",
                    48,
                    4.0,
                ),
                ("flashinfer_gated_delta_rule_decode", "generation", 48, 2.1),
            ],
        );
        assert_eq!(
            query_gdn_test_shape_with_dtype(
                &table,
                "fused_sigmoid_gating_delta_rule_update",
                "generation",
                48,
                "bfloat16"
            )
            .unwrap(),
            2.1
        );
    }

    #[test]
    fn sglang_sm100_gdn_keeps_fla_lane_for_fp32_state_even_if_flashinfer_present() {
        // Every bundled Qwen3.5/3.6 config pins mamba_ssm_dtype=float32;
        // serving auto-selects FlashInfer only for bfloat16 state, so the
        // default query must resolve the fla lane's own row even when a
        // flashinfer sibling row covers the shape.
        let table = in_memory_gdn_table_with_sm(
            "sglang",
            "0.5.14",
            Some(103),
            &[
                (
                    "fused_sigmoid_gating_delta_rule_update",
                    "generation",
                    48,
                    4.0,
                ),
                ("flashinfer_gated_delta_rule_decode", "generation", 48, 2.1),
            ],
        );
        assert_eq!(
            query_gdn_test_shape(
                &table,
                "fused_sigmoid_gating_delta_rule_update",
                "generation",
                48
            )
            .unwrap(),
            4.0
        );
    }

    #[test]
    fn sglang_sm120_gdn_never_selects_flashinfer_lane_regardless_of_dtype() {
        // sglang's is_sm100_supported is capability major EXACTLY 10; sm 120
        // (rtx_pro_6000_server) never auto-selects the FlashInfer decode
        // backend, bf16 state or not.
        let table = in_memory_gdn_table_with_sm(
            "sglang",
            "0.5.14",
            Some(120),
            &[
                (
                    "fused_sigmoid_gating_delta_rule_update",
                    "generation",
                    48,
                    4.0,
                ),
                ("flashinfer_gated_delta_rule_decode", "generation", 48, 2.1),
            ],
        );
        for dtype in ["float32", "bfloat16"] {
            assert_eq!(
                query_gdn_test_shape_with_dtype(
                    &table,
                    "fused_sigmoid_gating_delta_rule_update",
                    "generation",
                    48,
                    dtype
                )
                .unwrap(),
                4.0
            );
        }
    }

    #[test]
    fn sglang_sm100_gdn_falls_back_to_fla_lane_when_flashinfer_absent() {
        // Tables collected before AIC-1745's collector change carry no
        // flashinfer_gated_delta_rule_decode rows: degrade to the fla lane
        // exactly as before, never a hard failure — even for a bf16-state
        // model whose serving predicate WOULD select the FlashInfer backend.
        let table = in_memory_gdn_table_with_sm(
            "sglang",
            "0.5.14",
            Some(103),
            &[(
                "fused_sigmoid_gating_delta_rule_update",
                "generation",
                48,
                4.0,
            )],
        );
        assert_eq!(
            query_gdn_test_shape_with_dtype(
                &table,
                "fused_sigmoid_gating_delta_rule_update",
                "generation",
                48,
                "bfloat16"
            )
            .unwrap(),
            4.0
        );
    }

    #[test]
    fn sglang_sm90_gdn_never_selects_flashinfer_lane_even_if_present() {
        // Hopper unchanged: serving never runs the FlashInfer decode kernel on
        // SM90, so the modeling layer must not reach for it there even when a
        // flashinfer row exists (mirrors the Python twin test) — bf16 state
        // included (the dtype half of the predicate cannot override the SM
        // half).
        let table = in_memory_gdn_table_with_sm(
            "sglang",
            "0.5.10",
            Some(90),
            &[
                (
                    "fused_sigmoid_gating_delta_rule_update",
                    "generation",
                    48,
                    4.0,
                ),
                ("flashinfer_gated_delta_rule_decode", "generation", 48, 2.1),
            ],
        );
        for dtype in ["float32", "bfloat16"] {
            assert_eq!(
                query_gdn_test_shape_with_dtype(
                    &table,
                    "fused_sigmoid_gating_delta_rule_update",
                    "generation",
                    48,
                    dtype
                )
                .unwrap(),
                4.0
            );
        }
    }

    /// In-memory KDA table over one fixed model shape (d_model=4096, heads
    /// 16/128, v-dim 128). Verify rows land at `[batch=1][seq=4]` (seq is the
    /// draft-token count), generation rows at `[batch=1]`.
    fn in_memory_kda_table(rows: &[(&str, &str, u32, f64)]) -> StateSpaceTable {
        let mut by_keys: BTreeMap<KdaKey, Node> = BTreeMap::new();
        for &(kernel_source, phase, num_v_heads, latency) in rows {
            let key = KdaKey {
                kernel_source: kernel_source.to_string(),
                phase: phase.to_string(),
                d_model: 4096,
                d_conv: 4,
                num_k_heads: 16,
                head_k_dim: 128,
                num_v_heads,
                head_v_dim: 128,
            };
            let node = by_keys.entry(key).or_insert_with(Node::branch);
            let leaf = LeafValue::latency_only(latency);
            if phase == "generation" {
                insert_first_wins(node, &[1], leaf);
            } else {
                insert_first_wins(node, &[1, 4], leaf);
            }
        }
        let table = StateSpaceTable::new(PathBuf::from("test-data"), "sglang", "0.5.14");
        assert!(table.kda.set(Ok(KdaGrids { by_keys })).is_ok());
        table
    }

    fn query_kda_test_shape(
        table: &StateSpaceTable,
        kernel_source: &str,
        phase: &str,
        num_v_heads: u32,
    ) -> Result<f64, AicError> {
        table
            .query_kda(
                kernel_source,
                phase,
                1,
                4,
                4096,
                4,
                16,
                128,
                num_v_heads,
                128,
                &dummy_sol,
            )
            .map(|v| v.latency)
    }

    #[test]
    fn kda_verify_is_a_two_axis_grid_and_generation_one_axis() {
        let table = in_memory_kda_table(&[
            ("fused_sigmoid_gating_delta_rule_update", "verify", 16, 2.5),
            ("fused_recurrent_kda_packed_decode", "generation", 16, 1.5),
        ]);
        // Verify resolves at [batch=1][seq=draft_tokens=4].
        assert_eq!(
            query_kda_test_shape(
                &table,
                "fused_sigmoid_gating_delta_rule_update",
                "verify",
                16
            )
            .unwrap(),
            2.5
        );
        // Generation resolves on the 1-axis batch curve (seq feeds SOL only).
        assert_eq!(
            query_kda_test_shape(
                &table,
                "fused_recurrent_kda_packed_decode",
                "generation",
                16
            )
            .unwrap(),
            1.5
        );
    }

    #[test]
    fn kda_nearest_shard_fallback_has_no_alias_sources() {
        // Same logical source, other num_v_heads shards: nearest wins.
        let table = in_memory_kda_table(&[
            ("chunk_kda", "context", 8, 4.0),
            ("chunk_kda", "context", 32, 5.0),
        ]);
        assert_eq!(
            query_kda_test_shape(&table, "chunk_kda", "context", 24).unwrap(),
            5.0
        );
        // Unlike GDN, a physical vLLM kernel name is NOT aliased at lookup:
        // querying the logical name against physical-only rows must miss
        // (the SOL byte model in the operator handles those names instead).
        let table = in_memory_kda_table(&[("chunk_kda_with_fused_gate", "context", 16, 2.0)]);
        assert!(query_kda_test_shape(&table, "chunk_kda", "context", 16).is_err());
    }

    #[test]
    fn state_space_loaders_smoke() {
        // GDN data exists on vLLM b200 (Nemotron-H slice); Mamba2 may not.
        let root = data_root("b200_sxm/vllm/0.19.0");
        let table = StateSpaceTable::new(root, "vllm", "0.19.0");
        // Just verify loader doesn't panic on missing-key path; we don't
        // assert a specific value here.
        let _ = table
            .query_gdn(
                "causal_conv1d_fn",
                "prefill",
                1,
                1024,
                4096,
                4,
                16,
                128,
                32,
                128,
                "float32",
                &dummy_sol,
            )
            .err();
    }

    #[test]
    fn gdn_table_finds_qwen35_27b_conv1d_update() {
        let root = data_root("b200_sxm/vllm/0.19.0");
        let table = StateSpaceTable::new(root, "vllm", "0.19.0");
        let r = table.query_gdn(
            "causal_conv1d_update",
            "generation",
            1,
            1,
            5120,
            4,
            16,
            128,
            48,
            128,
            "float32",
            &dummy_sol,
        );
        eprintln!("query: {r:?}");
        assert!(r.is_ok(), "expected silicon lookup to succeed: {r:?}");
        let latency = r.unwrap().latency;
        assert!(latency > 0.0, "non-zero latency: {latency}");
        eprintln!("latency: {latency}");
    }

    /// Values generated from Python v2 on the same tables
    /// (`db.query_gdn` / `db.query_mamba2` via `get_database(...)`, default
    /// SILICON mode; the shared layer was verified to contribute no rows to
    /// these slices, so both engines see identical data; the KDA oracle
    /// below was generated against `get_database(..., shared_layer=False)`
    /// to match this table's single-source view). Covers, per table:
    /// exact hit, in-range interpolation, and beyond-range util-hold (which
    /// exercises the SOL closure). SOL closures replicate the operators'
    /// `sol_latency_ms` for the queried kernels. Latencies compared at the
    /// table/db layer — Python applies no extra factors there (the op-layer
    /// `scale_factor` sits above `db.query_*`).
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate
    // if this fails (the multi-source loaders may now merge sibling/shared
    // rows into these curves).
    #[test]
    fn state_space_queries_match_python_v2_engine() {
        let bw = h100_sxm_mem_bw();

        // GDN causal_conv1d kernels: read = x*conv_channels*(d_conv+1)*2,
        // write = x*conv_channels*2; x = b*s (context) or b (generation);
        // conv_channels is the packed q/k/v width (2K + V).
        let conv_channels = (2 * 16 * 128 + 32 * 128) as f64;
        let gdn_conv_sol = move |x: f64| {
            (x * conv_channels * (4.0 + 1.0) * 2.0 + x * conv_channels * 2.0) / bw * 1000.0
        };
        let gdn_ctx_sol = move |b: f64, s: f64| gdn_conv_sol(b * s);
        let gdn_gen_sol = move |b: f64, _s: f64| gdn_conv_sol(b);

        let gdn = StateSpaceTable::new(data_root("h100_sxm/sglang/0.5.10"), "sglang", "0.5.10");
        // (batch, seq, expected): exact / seq-interp / batch-interp / beyond-seq.
        let ctx_cases: &[(u32, u32, f64)] = &[
            (8, 1024, 0.03154560029506683),
            (8, 1536, 0.04624959975481033),
            (3, 1024, 0.011241600289940833),
            (8, 65536, 1.8143790228535068),
        ];
        for &(b, s, expected) in ctx_cases {
            let got = gdn
                .query_gdn(
                    "causal_conv1d_fn",
                    "context",
                    b,
                    s,
                    2048,
                    4,
                    16,
                    128,
                    32,
                    128,
                    "float32",
                    &gdn_ctx_sol,
                )
                .unwrap()
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "gdn ctx (b={b}, s={s}): rust {got} vs python {expected}"
            );
        }
        // Generation: exact / interp / beyond-range on the 1-axis batch curve.
        let gen_cases: &[(u32, f64)] = &[
            (8, 0.0052767999470233916),
            (48, 0.005246399901807308),
            (2048, 0.04643200039863586),
        ];
        for &(b, expected) in gen_cases {
            let got = gdn
                .query_gdn(
                    "causal_conv1d_update",
                    "generation",
                    b,
                    1,
                    2048,
                    4,
                    16,
                    128,
                    32,
                    128,
                    "float32",
                    &gdn_gen_sol,
                )
                .unwrap()
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "gdn gen (b={b}): rust {got} vs python {expected}"
            );
        }

        // Mamba2 causal_conv1d: conv_dim = d_inner + 2*n_groups*d_state.
        let conv_dim = (64 * 64 + 2 * 8 * 128) as f64;
        let m2_ctx_sol = move |b: f64, s: f64| {
            let x = b * s;
            (x * conv_dim * (4.0 + 1.0) * 2.0 + x * conv_dim * 2.0) / bw * 1000.0
        };
        let mamba2 = StateSpaceTable::new(
            data_root("h100_sxm/trtllm/1.3.0rc10"),
            "trtllm",
            "1.3.0rc10",
        );
        let m2_cases: &[(u32, u32, f64)] = &[
            (4, 1024, 0.058057600259780885),
            (4, 1536, 0.07725920081138611),
            (4, 65536, 2.53553341830743),
        ];
        for &(b, s, expected) in m2_cases {
            let got = mamba2
                .query_mamba2(
                    "causal_conv1d_fn",
                    "context",
                    b,
                    s,
                    2688,
                    128,
                    4,
                    64,
                    64,
                    8,
                    128,
                    &m2_ctx_sol,
                )
                .unwrap()
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "mamba2 ctx (b={b}, s={s}): rust {got} vs python {expected}"
            );
        }
        // Mamba2 generation stays SOL-degraded (Python v2 loader bug parity):
        // the table layer must error so the operator falls to plain SOL.
        assert!(mamba2
            .query_mamba2(
                "causal_conv1d_update",
                "generation",
                4,
                1,
                2688,
                128,
                4,
                64,
                64,
                8,
                128,
                &dummy_sol,
            )
            .is_err());

        // KDA (Kimi-K3, review Blocker 1 anchor): the TP8 12-head shard
        // (7168, 12, 128, 12, 128, 4) on b300_sxm/kda/sglang/0.5.16 —
        // Python oracle generated with
        // `get_database("b300_sxm", "sglang", "0.5.16", shared_layer=False)`
        // + `KDAKernel._query_kda_table` at these exact coordinates. Covers
        // the fused-dispatch kernels the K3 per-key routing depends on:
        // context chunk_kda, generation kda_fused_decode (the key whose
        // ABSENCE-load-bearing Triton twin must never be donor-filled), and
        // the fused CuTeDSL DSPARK verify kernel. SOL closures replicate
        // `KdaOp::sol_total_bytes_with`; b300_sxm mem_bw = 7.75e12
        // (systems/b300_sxm.yaml).
        let kda_bw = 7.75e12_f64;
        let kda_proj = (12 * 128) as f64;
        let kda_state = (12 * 128 * 128 * 4) as f64;
        let kda_conv_ch = 3.0 * kda_proj;
        let kda_ctx_sol = move |b: f64, s: f64| {
            // chunk_kda: chunked scan with per-chunk fp32 states.
            let x = b * s;
            let num_chunks = if s > 0.0 { (s / 64.0).floor() } else { 0.0 };
            let h_chunks = num_chunks * kda_state * b;
            let read = x * 4.0 * kda_proj * 2.0 + kda_state * b + h_chunks;
            let write = x * kda_proj * 2.0 + kda_state * b + h_chunks;
            (read + write) / kda_bw * 1000.0
        };
        let kda_gen_sol = move |b: f64, _s: f64| {
            // kda_fused_decode = conv update + packed recurrence (+ folded norm).
            let x = b;
            let read =
                x * kda_conv_ch * (4.0 + 1.0) * 2.0 + (x * 4.0 * kda_proj * 2.0 + kda_state * b);
            let write = x * kda_conv_ch * 2.0 + (x * kda_proj * 2.0 + kda_state * b);
            (read + write) / kda_bw * 1000.0
        };
        let kda_verify_sol = move |b: f64, s: f64| {
            // fused_kda_decode_mtp_dspark = conv update + chain-verify recurrence.
            let x = b * s;
            let read =
                x * kda_conv_ch * (4.0 + 1.0) * 2.0 + (x * 4.0 * kda_proj * 2.0 + kda_state * b);
            let write = x * kda_conv_ch * 2.0 + (x * kda_proj * 2.0 + kda_state * x);
            (read + write) / kda_bw * 1000.0
        };

        let kda = StateSpaceTable::new(data_root("b300_sxm/sglang/0.5.16"), "sglang", "0.5.16");
        // context chunk_kda: exact / seq-interp / batch-interp / beyond-seq.
        let kda_ctx_cases: &[(u32, u32, f64)] = &[
            (8, 1024, 0.35404798984527586),
            (8, 1536, 0.49525119066238404),
            (3, 1024, 0.16356800198554994),
            (8, 65536, 19.15999071181899),
        ];
        for &(b, s, expected) in kda_ctx_cases {
            let got = kda
                .query_kda(
                    "chunk_kda",
                    "context",
                    b,
                    s,
                    7168,
                    4,
                    12,
                    128,
                    12,
                    128,
                    &kda_ctx_sol,
                )
                .unwrap()
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "kda ctx (b={b}, s={s}): rust {got} vs python {expected}"
            );
        }
        // generation kda_fused_decode: exact / batch-interp / beyond-batch.
        let kda_gen_cases: &[(u32, f64)] = &[
            (8, 0.006656000018119812),
            (48, 0.01388160027563572),
            (4096, 1.1138175964355468),
        ];
        for &(b, expected) in kda_gen_cases {
            let got = kda
                .query_kda(
                    "kda_fused_decode",
                    "generation",
                    b,
                    0,
                    7168,
                    4,
                    12,
                    128,
                    12,
                    128,
                    &kda_gen_sol,
                )
                .unwrap()
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "kda gen (b={b}): rust {got} vs python {expected}"
            );
        }
        // verify fused CuTeDSL DSPARK: exact / batch-interp / draft-interp /
        // beyond-batch (seq axis carries the draft-token width).
        let kda_verify_cases: &[(u32, u32, f64)] = &[
            (8, 8, 0.020873600244522096),
            (12, 4, 0.018801599740982056),
            (8, 6, 0.01780159994959831),
            (1024, 8, 1.5144814803344375),
        ];
        for &(b, s, expected) in kda_verify_cases {
            let got = kda
                .query_kda(
                    "fused_kda_decode_mtp_dspark",
                    "verify",
                    b,
                    s,
                    7168,
                    4,
                    12,
                    128,
                    12,
                    128,
                    &kda_verify_sol,
                )
                .unwrap()
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "kda verify (b={b}, draft={s}): rust {got} vs python {expected}"
            );
        }
    }

    /// ENERGY oracle on a synthetic power-carrying fixture. Python twin
    /// (pandas fixture, `energy_test_fixtures` spec):
    ///
    /// ```text
    /// db.query_gdn(phase="context", kernel_source="causal_conv1d_fn",
    ///              batch_size=1, seq_len=1536, d_model=2048, num_k_heads=16,
    ///              head_k_dim=128, num_v_heads=32, head_v_dim=128, d_conv=4)
    /// # -> latency=2.0, energy=300.0
    /// ```
    #[test]
    fn gdn_energy_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{write_parquet, Col};
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &tmp.path().join("gdn_perf.parquet"),
            &[
                Col::Str("kernel_source", vec!["causal_conv1d_fn"; 2]),
                Col::Str("phase", vec!["context", "context"]),
                Col::I64("batch_size", vec![1, 1]),
                Col::I64("seq_len", vec![1024, 2048]),
                Col::I64("d_model", vec![2048, 2048]),
                Col::I64("d_conv", vec![4, 4]),
                Col::I64("num_k_heads", vec![16, 16]),
                Col::I64("head_k_dim", vec![128, 128]),
                Col::I64("num_v_heads", vec![32, 32]),
                Col::I64("head_v_dim", vec![128, 128]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        let table = StateSpaceTable::new(tmp.path().to_path_buf(), "vllm", "1.0");
        let v = table
            .query_gdn(
                "causal_conv1d_fn",
                "context",
                1,
                1536,
                2048,
                4,
                16,
                128,
                32,
                128,
                "float32",
                &dummy_sol,
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
