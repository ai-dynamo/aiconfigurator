// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! MLA family perf tables: op-level context/generation, MLA BMM (pre/post),
//! and module-level context/generation.
//!
//! Mirrors the SILICON paths of `aiconfigurator.sdk.operations.mla.{ContextMLA,
//! GenerationMLA, MLABmm, MLAModule}._query_*_table` on the perf_interp v2
//! engine: context-type tables ([num_heads][seq][batch], latency ~ seq^2) use
//! a Grid resolver with SQRT blending on the seq axis; generation-type tables
//! ([num_heads][batch][seq], ~linear in seq) and the 1-D BMM tokens curve use
//! RAW Grid blending. Beyond the collected range every query util-holds on the
//! boundary anchored by the op's SOL (ported verbatim from each Python
//! `get_sol`). Module-level data is collected as a fused unit (MLA + RoPE +
//! BMM together) and indexed by an extra `gemm_quant` axis.
//!
//! Each perf file loads from an ordered, shared-layer-aware source list (see
//! [`PerfSource`]); `MlaTable::new` degrades to the single primary
//! `data_root/<basename>` with no `kernel_source` filter.
//!
//! Caller passes `full_seq_tokens` for context queries (= `isl + prefix`);
//! the prefix-correction multiplier is applied by the operator layer, and the
//! context SOL is therefore evaluated at `prefix = 0` exactly like the
//! `sol_fn` Python wires into `perf_interp.context_grid_config`.
//! The MLA BMM table falls back to `bfloat16` data when the requested quant
//! mode is absent, matching Python's `quant_mode_lookup` behavior; the BMM
//! SOL keeps using the REQUESTED quant (Python passes `quant_mode`, not the
//! lookup fallback, into its `get_sol`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::attention::generation_attn_flops;
use super::gemm::quant_tc_flops;
use super::interpolation::Grid3;
use super::perf_interp::{self, LeafValue, Node, OpInterpConfig};
use super::{kernel_source_ok, SourceResolver};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use crate::operators::base::SolComponents;
use crate::perf_database::parquet_loader::PerfReader;

/// Axes for context-type MLA tables (op-level and module-level).
const CONTEXT_AXES: &[&str] = &["num_heads", "seq_len", "batch"];
/// Axes for generation-type MLA tables (op-level and module-level).
const GENERATION_AXES: &[&str] = &["num_heads", "batch", "seq_len"];
/// Axes for the 1-D MLA BMM tokens curve.
const BMM_AXES: &[&str] = &["num_tokens"];

pub struct MlaTable {
    data_root: PathBuf,
    system_spec: SystemSpec,
    /// Ordered, priority-sorted sources for each MLA-family perf file
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`MlaTable::new`).
    context_mla_sources: Vec<PerfSource>,
    generation_mla_sources: Vec<PerfSource>,
    mla_bmm_sources: Vec<PerfSource>,
    mla_context_module_sources: Vec<PerfSource>,
    mla_generation_module_sources: Vec<PerfSource>,
    context: OnceLock<Result<ContextMlaGrids, AicError>>,
    generation: OnceLock<Result<GenerationMlaGrids, AicError>>,
    bmm: OnceLock<Result<BmmGrids, AicError>>,
    context_module: OnceLock<Result<ModuleGrids, AicError>>,
    generation_module: OnceLock<Result<GenModuleGrids, AicError>>,
}

struct ContextMlaGrids {
    by_keys: BTreeMap<ContextKey, Node>,
}

struct GenerationMlaGrids {
    by_keys: BTreeMap<KvOnlyKey, Node>,
}

/// Module-level context MLA grids (fmha is a real axis for context). Inner
/// level: model-native head identity from the `model` column via
/// [`mla_module_native_heads`] — never `num_heads * tp_size` (#1458; module
/// rows are single-GPU rank-local head sweeps, tp is provenance).
struct ModuleGrids {
    by_keys: BTreeMap<ModuleKey, BTreeMap<u32, Node>>,
}

/// Generation twin of [`ModuleGrids`]; keyed (kv, gemm) only — the parquet's
/// `mla_dtype` column is degenerate and dropped, mirroring Python.
struct GenModuleGrids {
    by_keys: BTreeMap<GenModuleKey, BTreeMap<u32, Node>>,
}

/// Native-head pin for MLA module tables — byte-equal with Python's
/// `_MLA_MODULE_NATIVE_HEADS` (operations/mla.py). Unknown models fail the
/// load: extending the pin is part of landing new module data.
pub(crate) fn mla_module_native_heads(model: &str) -> Option<u32> {
    match model {
        "deepseek-ai/DeepSeek-V3" => Some(128),
        // vllm 0.22.0 provenance aliases of the same 128-native geometry.
        "deepseek-ai/DeepSeek-R1" => Some(128),
        "nvidia/DeepSeek-V3.1-NVFP4" => Some(128),
        _ => None,
    }
}

/// Byte-equal twin of Python's `_resolve_mla_module_native_key`: exact ->
/// sole bucket -> nearest <= -> smallest; `None` resolves only a
/// single-bucket table.
fn resolve_module_native<T>(buckets: &BTreeMap<u32, T>, native_heads: Option<u32>) -> Option<&T> {
    if buckets.is_empty() {
        return None;
    }
    let native = match native_heads {
        None => {
            return if buckets.len() == 1 {
                buckets.values().next()
            } else {
                None
            };
        }
        Some(native) => native,
    };
    if let Some(node) = buckets.get(&native) {
        return Some(node);
    }
    if buckets.len() == 1 {
        return buckets.values().next();
    }
    buckets
        .range(..=native)
        .next_back()
        .map(|(_, v)| v)
        .or_else(|| buckets.values().next())
}

struct BmmGrids {
    // (bmm_quant, "mla_gen_pre" | "mla_gen_post") -> num_heads -> 1-D tokens curve
    by_keys: BTreeMap<BmmKey, BTreeMap<u32, Node>>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ContextKey {
    fmha_quant: String,
    kv_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct KvOnlyKey {
    kv_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleKey {
    fmha_quant: String,
    kv_quant: String,
    gemm_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GenModuleKey {
    kv_quant: String,
    gemm_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct BmmKey {
    bmm_quant: String,
    pre_or_post: String,
}

impl MlaTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf, system_spec: SystemSpec) -> Self {
        Self::with_sources(
            data_root,
            system_spec,
            &SourceResolver::fixed(PerfDbSources::default()),
        )
        .expect("fixed-map resolution is infallible")
    }

    /// Construct with shared-layer (sibling/cross-version) sources supplied by the
    /// engine's `SourceResolver` (live resolution owns the shared-layer walk;
    /// a fixed source map is the test-only path). Each MLA-family file falls back to
    /// its primary `data_root/<basename>` when the resolver names no override. No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        system_spec: SystemSpec,
        resolver: &SourceResolver,
    ) -> Result<Self, AicError> {
        let context_mla_sources = resolver.sources_for("context_mla_perf.parquet", &data_root)?;
        let generation_mla_sources =
            resolver.sources_for("generation_mla_perf.parquet", &data_root)?;
        let mla_bmm_sources = resolver.sources_for("mla_bmm_perf.parquet", &data_root)?;
        let mla_context_module_sources =
            resolver.sources_for("mla_context_module_perf.parquet", &data_root)?;
        let mla_generation_module_sources =
            resolver.sources_for("mla_generation_module_perf.parquet", &data_root)?;
        Ok(Self {
            data_root,
            system_spec,
            context_mla_sources,
            generation_mla_sources,
            mla_bmm_sources,
            mla_context_module_sources,
            mla_generation_module_sources,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            bmm: OnceLock::new(),
            context_module: OnceLock::new(),
            generation_module: OnceLock::new(),
        })
    }

    /// Op-level context MLA value (latency ms + power/energy; raw — no
    /// prefix correction, which the operator layer applies to latency AND
    /// energy).
    pub fn query_context(
        &self,
        b: u32,
        full_seq_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
    ) -> Result<LeafValue, AicError> {
        // Resolve flops BEFORE any perf-data lookup: a missing dtype entry
        // must classify as MissingSystemFlops on both engines, in every mode
        // (mirrors Python's query-entry resolution and GemmTable::query).
        let attn_flops = quant_tc_flops(&self.system_spec, fmha_quant.mapping())?;
        let grids = self.load_context()?;
        let key = ContextKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("context MLA", &self.data_root, format!("{key:?}")))?;
        let spec = &self.system_spec;
        // c = (num_heads, seq_len, batch), prefix = 0 (see module docs).
        let sol = move |c: &[f64]| context_mla_sol_ms(spec, kv_quant, c[0], c[1], c[2], attn_flops);
        let cfg = OpInterpConfig::grid_sqrt_axis(CONTEXT_AXES, 1, &sol);
        perf_interp::query_value(
            &cfg,
            node,
            &[num_heads as f64, full_seq_tokens as f64, b as f64],
        )
    }

    /// Op-level generation MLA value (latency ms + power/energy).
    pub fn query_generation(
        &self,
        b: u32,
        s: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
    ) -> Result<LeafValue, AicError> {
        // Resolve flops BEFORE any perf-data lookup: a missing dtype entry
        // must classify as MissingSystemFlops on both engines, in every mode
        // (mirrors Python's query-entry resolution and GemmTable::query).
        let attn_flops = generation_attn_flops(&self.system_spec, kv_quant)?;
        let grids = self.load_generation()?;
        let key = KvOnlyKey {
            kv_quant: kv_quant.name().to_string(),
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("generation MLA", &self.data_root, format!("{key:?}")))?;
        let spec = &self.system_spec;
        // Python's generation MLA uses (num_heads, b, s) as the 3 axes
        // — note b and s order differs from context. RAW blending (~linear in s).
        let sol =
            move |c: &[f64]| generation_mla_sol_ms(spec, kv_quant, c[0], c[1], c[2], attn_flops);
        let cfg = OpInterpConfig::grid(GENERATION_AXES, &sol);
        perf_interp::query_value(&cfg, node, &[num_heads as f64, b as f64, s as f64])
    }

    /// MLA BMM (pre or post) value (latency ms + power/energy; raw — the
    /// operator layer applies the exact-head-routing `head_scale` to latency
    /// AND energy, mirroring Python's `_interp_pr(lat * head_scale,
    /// energy=energy * head_scale)`).
    ///
    /// Falls back to `bfloat16` if the requested quant mode is absent,
    /// matching Python's `quant_mode_lookup` behavior. The SOL keeps using
    /// the requested quant mode either way (parity with Python's `get_sol`).
    pub fn query_bmm(
        &self,
        num_tokens: u32,
        num_heads: u32,
        quant: GemmQuantMode,
        is_pre: bool,
    ) -> Result<LeafValue, AicError> {
        // Resolve flops BEFORE any perf-data lookup: a missing dtype entry
        // must classify as MissingSystemFlops on both engines, in every mode
        // (mirrors Python's query-entry resolution and GemmTable::query).
        let bmm_flops = quant_tc_flops(&self.system_spec, quant.mapping())?;
        let grids = self.load_bmm()?;
        let pre_or_post = if is_pre {
            "mla_gen_pre"
        } else {
            "mla_gen_post"
        };

        // Try the requested quant first; fall back to bfloat16 if missing.
        let key = BmmKey {
            bmm_quant: quant.name().to_string(),
            pre_or_post: pre_or_post.to_string(),
        };
        let chosen = grids.by_keys.get(&key).or_else(|| {
            let fallback = BmmKey {
                bmm_quant: GemmQuantMode::Bfloat16.name().to_string(),
                pre_or_post: pre_or_post.to_string(),
            };
            grids.by_keys.get(&fallback)
        });
        let by_heads = chosen.ok_or_else(|| {
            missing(
                "MLA BMM",
                &self.data_root,
                format!("quant={}, {pre_or_post}", quant.name()),
            )
        })?;

        let node = by_heads.get(&num_heads).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "MLA BMM data missing for num_heads={num_heads} at {}",
                self.data_root.display()
            ))
        })?;

        // 1-D tokens curve: RAW lerp in range (BMM is ~linear in tokens);
        // boundary util-hold beyond it via the BMM SOL.
        let spec = &self.system_spec;
        let sol = move |c: &[f64]| mla_bmm_sol_ms(spec, quant, num_heads as f64, c[0], bmm_flops);
        let cfg = OpInterpConfig::grid(BMM_AXES, &sol);
        perf_interp::query_value(&cfg, node, &[num_tokens as f64])
    }

    /// Module-level context MLA value (latency ms + power/energy; raw — no
    /// prefix correction, which the operator layer applies to latency AND
    /// energy).
    pub fn query_context_module(
        &self,
        b: u32,
        full_seq_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        native_heads: Option<u32>,
    ) -> Result<LeafValue, AicError> {
        // Resolve flops BEFORE any perf-data lookup: a missing dtype entry
        // must classify as MissingSystemFlops on both engines, in every mode
        // (mirrors Python's query-entry resolution and GemmTable::query).
        let attn_flops = quant_tc_flops(&self.system_spec, fmha_quant.mapping())?;
        let grids = self.load_context_module()?;
        let key = ModuleKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let buckets = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("context MLA module", &self.data_root, format!("{key:?}")))?;
        let node = resolve_module_native(buckets, native_heads).ok_or_else(|| {
            missing(
                "context MLA module",
                &self.data_root,
                format!("{key:?} native_heads={native_heads:?}"),
            )
        })?;
        let spec = &self.system_spec;
        // Python's module-context get_sol reuses the op-level context SOL
        // verbatim (the module fuses MLA + RoPE + BMM but the SOL refinement
        // was deliberately deferred there too).
        let sol = move |c: &[f64]| context_mla_sol_ms(spec, kv_quant, c[0], c[1], c[2], attn_flops);
        let cfg = OpInterpConfig::grid_sqrt_axis(CONTEXT_AXES, 1, &sol);
        perf_interp::query_value(
            &cfg,
            node,
            &[num_heads as f64, full_seq_tokens as f64, b as f64],
        )
    }

    /// Module-level generation MLA value (latency ms + power/energy). No
    /// fmha axis: decode compute dtype follows the kv-cache dtype (see
    /// [`GenModuleGrids`]).
    pub fn query_generation_module(
        &self,
        b: u32,
        s: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        gemm_quant: GemmQuantMode,
        native_heads: Option<u32>,
    ) -> Result<LeafValue, AicError> {
        // Resolve flops BEFORE any perf-data lookup: a missing dtype entry
        // must classify as MissingSystemFlops on both engines, in every mode
        // (mirrors Python's query-entry resolution and GemmTable::query).
        let attn_flops = generation_attn_flops(&self.system_spec, kv_quant)?;
        let bmm_flops = quant_tc_flops(&self.system_spec, gemm_quant.mapping())?;
        let grids = self.load_generation_module()?;
        let key = GenModuleKey {
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let buckets = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("generation MLA module", &self.data_root, format!("{key:?}")))?;
        let node = resolve_module_native(buckets, native_heads).ok_or_else(|| {
            missing(
                "generation MLA module",
                &self.data_root,
                format!("{key:?} native_heads={native_heads:?}"),
            )
        })?;
        let spec = &self.system_spec;
        // Generation module SOL = generation MLA SOL + BMM pre/post terms
        // (Python's module get_sol folds the BMM into sol_math/sol_mem before
        // the max).
        let sol = move |c: &[f64]| {
            generation_mla_module_sol_ms(
                spec, kv_quant, gemm_quant, c[0], c[1], c[2], attn_flops, bmm_flops,
            )
        };
        let cfg = OpInterpConfig::grid(GENERATION_AXES, &sol);
        perf_interp::query_value(&cfg, node, &[num_heads as f64, b as f64, s as f64])
    }

    // -----------------------------------------------------------------------
    // Point accessors for the util-space empirical layer (algorithm-free:
    // typed `AicError::PerfDatabase` miss on absent slice / empty node, no
    // estimation logic). Coordinate order matches the Python `depth=3`
    // iteration of each `require_data_slice` slice.
    // -----------------------------------------------------------------------

    /// Collected `(num_heads, seq, batch) -> latency` points of the op-level
    /// context MLA `(fmha, kv)` slice. Typed miss when absent/empty.
    pub fn context_points(
        &self,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("context MLA", &self.data_root, format!("{key:?}")))?;
        non_empty_points(node, "context MLA", &self.data_root)
    }

    /// Collected `(num_heads, batch, seq) -> latency` points of the op-level
    /// generation MLA kv slice. Typed miss when absent/empty.
    pub fn generation_points(
        &self,
        kv_quant: KvCacheQuantMode,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_generation()?;
        let key = KvOnlyKey {
            kv_quant: kv_quant.name().to_string(),
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("generation MLA", &self.data_root, format!("{key:?}")))?;
        non_empty_points(node, "generation MLA", &self.data_root)
    }

    /// The BMM quant slice Python's `quant_mode in wrapper` membership check
    /// selects: the requested quant when ANY BMM data exists for it (either
    /// op_name), otherwise `bfloat16` — even when bfloat16 has no data
    /// either (the follow-up [`Self::bmm_points`] then reports the typed
    /// miss, matching Python's `require_data_slice`). Errs only when the
    /// whole BMM table failed to load.
    pub fn bmm_selected_quant(&self, quant: GemmQuantMode) -> Result<GemmQuantMode, AicError> {
        let grids = self.load_bmm()?;
        let has_quant = grids
            .by_keys
            .keys()
            .any(|key| key.bmm_quant == quant.name());
        Ok(if has_quant {
            quant
        } else {
            GemmQuantMode::Bfloat16
        })
    }

    /// Collected `(num_tokens,) -> latency` points of the
    /// `(quant, op_name, num_heads)` BMM slice. No fallback here — the caller
    /// resolves the quant via [`Self::bmm_selected_quant`] first (mirroring
    /// Python, where the membership check is on the quant level only).
    /// Typed miss when the slice is absent/empty.
    pub fn bmm_points(
        &self,
        quant: GemmQuantMode,
        is_pre: bool,
        num_heads: u32,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_bmm()?;
        let pre_or_post = if is_pre {
            "mla_gen_pre"
        } else {
            "mla_gen_post"
        };
        let key = BmmKey {
            bmm_quant: quant.name().to_string(),
            pre_or_post: pre_or_post.to_string(),
        };
        let node = grids
            .by_keys
            .get(&key)
            .and_then(|by_heads| by_heads.get(&num_heads))
            .ok_or_else(|| {
                missing(
                    "MLA BMM",
                    &self.data_root,
                    format!(
                        "quant={}, {pre_or_post}, num_heads={num_heads}",
                        quant.name()
                    ),
                )
            })?;
        non_empty_points(node, "MLA BMM", &self.data_root)
    }

    /// Whether the `(quant, op_name, num_heads)` BMM slice has rows —
    /// data-presence probe for the exact-head-first routing
    /// (`operators/mla.rs::resolve_bmm_slice_heads`). No bfloat16 quant
    /// fallback here; the caller resolves the quant via
    /// [`Self::bmm_selected_quant`] first. Errs only when the whole BMM
    /// table failed to load.
    pub fn bmm_has_heads(
        &self,
        quant: GemmQuantMode,
        is_pre: bool,
        num_heads: u32,
    ) -> Result<bool, AicError> {
        let grids = self.load_bmm()?;
        let key = BmmKey {
            bmm_quant: quant.name().to_string(),
            pre_or_post: if is_pre {
                "mla_gen_pre"
            } else {
                "mla_gen_post"
            }
            .to_string(),
        };
        Ok(grids
            .by_keys
            .get(&key)
            .is_some_and(|by_heads| by_heads.contains_key(&num_heads)))
    }

    /// Collected `(num_heads, seq, batch) -> latency` points of the
    /// module-level context MLA `(fmha, kv, gemm)` slice. Typed miss when
    /// absent/empty.
    pub fn context_module_points(
        &self,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        native_heads: Option<u32>,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_context_module()?;
        let key = ModuleKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let buckets = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("context MLA module", &self.data_root, format!("{key:?}")))?;
        let node = resolve_module_native(buckets, native_heads).ok_or_else(|| {
            missing(
                "context MLA module",
                &self.data_root,
                format!("{key:?} native_heads={native_heads:?}"),
            )
        })?;
        non_empty_points(node, "context MLA module", &self.data_root)
    }

    /// Collected `(num_heads, batch, seq) -> latency` points of the
    /// module-level generation MLA `(kv, gemm)` slice. Typed miss when
    /// absent/empty.
    pub fn generation_module_points(
        &self,
        kv_quant: KvCacheQuantMode,
        gemm_quant: GemmQuantMode,
        native_heads: Option<u32>,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_generation_module()?;
        let key = GenModuleKey {
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let buckets = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("generation MLA module", &self.data_root, format!("{key:?}")))?;
        let node = resolve_module_native(buckets, native_heads).ok_or_else(|| {
            missing(
                "generation MLA module",
                &self.data_root,
                format!("{key:?} native_heads={native_heads:?}"),
            )
        })?;
        non_empty_points(node, "generation MLA module", &self.data_root)
    }

    fn load_context(&self) -> Result<&ContextMlaGrids, AicError> {
        let cell = self
            .context
            .get_or_init(|| load_op_parquet(&self.context_mla_sources, true));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&GenerationMlaGrids, AicError> {
        let cell = self
            .generation
            .get_or_init(|| load_op_gen_parquet(&self.generation_mla_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_bmm(&self) -> Result<&BmmGrids, AicError> {
        let cell = self
            .bmm
            .get_or_init(|| load_bmm_parquet(&self.mla_bmm_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_context_module(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self
            .context_module
            .get_or_init(|| load_context_module_parquet(&self.mla_context_module_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation_module(&self) -> Result<&GenModuleGrids, AicError> {
        let cell = self
            .generation_module
            .get_or_init(|| load_generation_module_parquet(&self.mla_generation_module_sources));
        cell.as_ref().map_err(clone_err)
    }
}

// ---------------------------------------------------------------------------
// SOL formulas — verbatim ports of the Python `get_sol` closures in
// `operations/mla.py`. Each keeps Python's `max(compute, mem)` structure and
// arithmetic ordering so cross-language parity holds to float precision.
// ---------------------------------------------------------------------------

/// Context MLA SOL in ms, evaluated at prefix = 0 (the perf_interp `sol_fn`
/// contract — samples are prefix=0 and the operator layer owns the prefix
/// correction). See [`context_mla_sol_prefix_ms`] for the formula.
pub(crate) fn context_mla_sol_ms(
    spec: &SystemSpec,
    kv_quant: KvCacheQuantMode,
    n: f64,
    s: f64,
    b: f64,
    attn_flops: f64,
) -> f64 {
    context_mla_sol_prefix_ms(spec, kv_quant, n, s, 0.0, b, attn_flops)
}

/// Prefix-aware context MLA SOL in ms (`s` is the chunk / isl length; the
/// util-empirical query SOL carries prefix natively, unlike the prefix=0
/// sample SOL). Mirrors `ContextMLA._query_context_mla_table::get_sol` and
/// the identical `MLAModule._query_context_mla_module_table::get_sol`.
/// `attn_flops` is the caller-resolved TC-FLOPS for the op's fmha quant
/// (`quant_tc_flops(spec, fmha_quant.mapping())`):
/// - `full_s   = s + prefix`
/// - `ops      = b * n * 2/2 * (192 + 128) * (full_s^2 - prefix^2)`
/// - `mem      = b * n * (kv.memory * full_s * (192+128) + 2 * s * (192+128))`
/// - `sol_math = ops / attn_flops * 1000`
/// - `sol_mem  = mem / mem_bw * 1000`
/// - `sol      = max(sol_math, sol_mem)`
pub(crate) fn context_mla_sol_prefix(
    spec: &SystemSpec,
    kv_quant: KvCacheQuantMode,
    n: f64,
    s: f64,
    prefix: f64,
    b: f64,
    attn_flops: f64,
) -> SolComponents {
    let full_s = s + prefix;
    let ops = b * n * 2.0 / 2.0 * (192.0 + 128.0) * (full_s * full_s - prefix * prefix);
    let mem_bytes =
        b * n * (kv_quant.mapping().memory * full_s * (192.0 + 128.0) + 2.0 * s * (192.0 + 128.0));
    let sol_math = ops / attn_flops * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    SolComponents::new(sol_math, sol_mem)
}

pub(crate) fn context_mla_sol_prefix_ms(
    spec: &SystemSpec,
    kv_quant: KvCacheQuantMode,
    n: f64,
    s: f64,
    prefix: f64,
    b: f64,
    attn_flops: f64,
) -> f64 {
    context_mla_sol_prefix(spec, kv_quant, n, s, prefix, b, attn_flops).time_ms()
}

/// Generation MLA SOL in ms. Mirrors
/// `GenerationMLA._query_generation_mla_table::get_sol`. `attn_flops` is
/// the caller-resolved decode-attention TC-FLOPS
/// (`generation_attn_flops(spec, kv_quant)` — fp8 KV -> fp8, else bf16):
/// - `ops      = 2 * b * n * 1088 * s`
/// - `mem      = b * (n * 1088 * 2 + (s - 1) * 576 * kv.memory)`
/// - `sol_math = ops / attn_flops * 1000`
/// - `sol_mem  = mem / mem_bw * 1000`
/// - `sol      = max(sol_math, sol_mem)`
pub(crate) fn generation_mla_sol(
    spec: &SystemSpec,
    kv_quant: KvCacheQuantMode,
    n: f64,
    b: f64,
    s: f64,
    attn_flops: f64,
) -> SolComponents {
    let ops = 2.0 * b * n * 1088.0 * s;
    let mem_bytes = b * (n * 1088.0 * 2.0 + (s - 1.0) * 576.0 * kv_quant.mapping().memory);
    let sol_math = ops / attn_flops * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    SolComponents::new(sol_math, sol_mem)
}

pub(crate) fn generation_mla_sol_ms(
    spec: &SystemSpec,
    kv_quant: KvCacheQuantMode,
    n: f64,
    b: f64,
    s: f64,
    attn_flops: f64,
) -> f64 {
    generation_mla_sol(spec, kv_quant, n, b, s, attn_flops).time_ms()
}

/// MLA BMM SOL in ms. Mirrors `MLABmm._query_mla_bmm_table::get_sol` (uses
/// the REQUESTED quant, even when the data lookup fell back to bfloat16).
/// `bmm_flops` is the caller-resolved TC-FLOPS for the REQUESTED quant
/// (`quant_tc_flops(spec, quant.mapping())`):
/// - `ops      = 2 * t * n * 128 * 512`
/// - `mem      = n * (t * 640 + 128 * 512) * quant.memory`
/// - `sol_math = ops / bmm_flops * 1000`
/// - `sol_mem  = mem / mem_bw * 1000`
/// - `sol      = max(sol_math, sol_mem)`
pub(crate) fn mla_bmm_sol(
    spec: &SystemSpec,
    quant: GemmQuantMode,
    n: f64,
    t: f64,
    bmm_flops: f64,
) -> SolComponents {
    let ops = 2.0 * t * n * 128.0 * 512.0;
    let mem_bytes = n * (t * 640.0 + 128.0 * 512.0) * quant.mapping().memory;
    let sol_math = ops / bmm_flops * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    SolComponents::new(sol_math, sol_mem)
}

pub(crate) fn mla_bmm_sol_ms(
    spec: &SystemSpec,
    quant: GemmQuantMode,
    n: f64,
    t: f64,
    bmm_flops: f64,
) -> f64 {
    mla_bmm_sol(spec, quant, n, t, bmm_flops).time_ms()
}

/// Generation MLA module SOL in ms. Mirrors
/// `MLAModule._query_generation_mla_module_table::get_sol`: the generation
/// MLA SOL plus BMM pre+post terms folded into sol_math / sol_mem BEFORE the
/// max (NOT `max(attn) + max(bmm)`). `attn_flops` is the caller-resolved
/// decode-attention TC-FLOPS (`generation_attn_flops(spec, kv_quant)`);
/// `bmm_flops` the gemm-quant TC-FLOPS
/// (`quant_tc_flops(spec, gemm_quant.mapping())`):
/// - attn: `ops = 2*b*n*1088*s`, `mem = b*(n*1088*2 + (s-1)*576*kv.memory)`
/// - bmm:  `ops = 2*2*b*n*128*512`, `mem = 2*n*(b*640 + 128*512)*gemm.memory`
/// - `sol_math = attn_ops/attn_flops + bmm_ops/bmm_flops`
/// - `sol_mem  = (attn_mem + bmm_mem) / mem_bw`
/// - `sol      = max(sol_math, sol_mem)`
#[allow(clippy::too_many_arguments)]
pub(crate) fn generation_mla_module_sol(
    spec: &SystemSpec,
    kv_quant: KvCacheQuantMode,
    gemm_quant: GemmQuantMode,
    n: f64,
    b: f64,
    s: f64,
    attn_flops: f64,
    bmm_flops: f64,
) -> SolComponents {
    // MLA attention ops
    let attn_ops = 2.0 * b * n * 1088.0 * s;
    let mem_bytes = b * (n * 1088.0 * 2.0 + (s - 1.0) * 576.0 * kv_quant.mapping().memory);
    let mut sol_math = attn_ops / attn_flops * 1000.0;
    let mut sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    // Add BMM pre + post SOL (same as query_mla_bmm)
    let bmm_ops = 2.0 * 2.0 * b * n * 128.0 * 512.0; // pre + post
    let bmm_mem = 2.0 * n * (b * 640.0 + 128.0 * 512.0) * gemm_quant.mapping().memory;
    let bmm_math = bmm_ops / bmm_flops * 1000.0;
    let bmm_mem_time = bmm_mem / spec.gpu.mem_bw * 1000.0;
    sol_math += bmm_math;
    sol_mem += bmm_mem_time;
    SolComponents::new(sol_math, sol_mem)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generation_mla_module_sol_ms(
    spec: &SystemSpec,
    kv_quant: KvCacheQuantMode,
    gemm_quant: GemmQuantMode,
    n: f64,
    b: f64,
    s: f64,
    attn_flops: f64,
    bmm_flops: f64,
) -> f64 {
    generation_mla_module_sol(spec, kv_quant, gemm_quant, n, b, s, attn_flops, bmm_flops).time_ms()
}

fn grid3_to_node(grid: &Grid3<LeafValue>) -> Node {
    let mut node = Node::branch();
    for (&a, by_b) in grid {
        for (&b, by_c) in by_b {
            for (&c, &leaf) in by_c {
                node.insert_value(&[a, b, c], leaf);
            }
        }
    }
    node
}

fn curve_to_node(curve: &BTreeMap<u32, LeafValue>) -> Node {
    let mut node = Node::branch();
    for (&t, &leaf) in curve {
        node.insert_value(&[t], leaf);
    }
    node
}

/// Load the op-level context MLA table from an ordered, priority-sorted
/// source list. Sources are read in order; the first source containing a
/// shape wins (`or_insert`), mirroring Python's `_read_filtered_rows`
/// concatenation + `load_mla_data` skip-on-key-conflict. Missing files are
/// skipped (a sibling declared in the manifest need not exist for every
/// system); an error is returned only when no source yields rows.
fn load_op_parquet(sources: &[PerfSource], is_context: bool) -> Result<ContextMlaGrids, AicError> {
    let mut raw: BTreeMap<ContextKey, Grid3<LeafValue>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let mla_dtype_col = reader.col("mla_dtype")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let num_heads_col = reader.col("num_heads")?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let step_col = reader.col("step")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = ContextKey {
                fmha_quant: row.str_owned(mla_dtype_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
            };
            let isl = row.u32(isl_col)?;
            let y_axis = if is_context {
                isl
            } else {
                isl + row.u32(step_col)?
            };
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // First-wins parity with Python `load_mla_data` (context branch),
            // extended across shared-layer sources (earlier source wins).
            raw.entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(y_axis)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MLA op rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    let by_keys = raw
        .into_iter()
        .map(|(key, grid)| (key, grid3_to_node(&grid)))
        .collect();
    Ok(ContextMlaGrids { by_keys })
}

/// Load the op-level generation MLA table from an ordered source list. Same
/// first-wins-across-sources + missing-file-skip semantics as
/// [`load_op_parquet`].
fn load_op_gen_parquet(sources: &[PerfSource]) -> Result<GenerationMlaGrids, AicError> {
    let mut raw: BTreeMap<KvOnlyKey, Grid3<LeafValue>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let num_heads_col = reader.col("num_heads")?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let step_col = reader.col("step")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = KvOnlyKey {
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
            };
            let sequence_tokens = row.u32(isl_col)? + row.u32(step_col)?;
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // Python uses (num_heads, b, s) axis order for generation MLA.
            // First-wins parity with Python `load_mla_data` (generation branch).
            raw.entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_default()
                .entry(sequence_tokens)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no generation MLA rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    let by_keys = raw
        .into_iter()
        .map(|(key, grid)| (key, grid3_to_node(&grid)))
        .collect();
    Ok(GenerationMlaGrids { by_keys })
}

/// Byte-equal twin of Python's `_mla_module_native_heads`: model pin lookup
/// plus the per-row rank-local guard (tp > 1 with `heads * tp != native` is
/// the #1429 stale fingerprint).
fn module_row_native_heads(
    model: &str,
    num_heads: u32,
    tp_size: u32,
    path: &std::path::Path,
) -> Result<u32, AicError> {
    if model.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "MLA module row in {} carries no model value; the module table keys its \
             native-head identity off the model pin (#1458)",
            path.display()
        )));
    }
    let native_heads = mla_module_native_heads(model).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "MLA module row in {} names unpinned model {model:?}; add its native head \
             count to the module native-head pin when landing the data (#1458)",
            path.display()
        ))
    })?;
    if tp_size > 1 && num_heads * tp_size != native_heads {
        return Err(AicError::PerfDatabase(format!(
            "MLA module row in {} for model {model:?} has num_heads={num_heads} at \
             tp_size={tp_size}, inconsistent with native {native_heads} (num_heads must \
             be rank-local, #1429/#1458)",
            path.display()
        )));
    }
    Ok(native_heads)
}

fn load_context_module_parquet(sources: &[PerfSource]) -> Result<ModuleGrids, AicError> {
    let mut raw: BTreeMap<ModuleKey, BTreeMap<u32, Grid3<LeafValue>>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let mla_dtype_col = reader.col("mla_dtype")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let gemm_type_col = reader.col("gemm_type")?;
        let model_col = reader.col("model")?;
        let num_heads_col = reader.col("num_heads")?;
        let tp_size_col = reader.col_optional("tp_size");
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = ModuleKey {
                fmha_quant: row.str_owned(mla_dtype_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
                gemm_quant: row.str_owned(gemm_type_col)?,
            };
            let num_heads = row.u32(num_heads_col)?;
            let tp_size = match tp_size_col {
                Some(col) => row.u32(col)?.max(1),
                None => 1,
            };
            let model = row.str_owned(model_col)?;
            let native_heads = module_row_native_heads(&model, num_heads, tp_size, path)?;
            let batch_size = row.u32(batch_size_col)?;
            let isl = row.u32(isl_col)?;
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // First-wins parity with Python `load_context_mla_module_data`,
            // which now guards with the standard skip-on-key-conflict idiom
            // (shared-layer contract: the primary version's rows must not be
            // shadowed by later sibling-version sources — design §6.1). Note
            // this also flips intra-file duplicate resolution to first-row-
            // wins; duplicate rows are re-measurements of the same shape and
            // a file-hygiene issue, not a semantic axis.
            let inner = raw
                .entry(key)
                .or_default()
                .entry(native_heads)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(isl)
                .or_default();
            inner
                .entry(batch_size)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MLA module rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    let by_keys = raw
        .into_iter()
        .map(|(key, by_native)| {
            (
                key,
                by_native
                    .into_iter()
                    .map(|(native, grid)| (native, grid3_to_node(&grid)))
                    .collect(),
            )
        })
        .collect();
    Ok(ModuleGrids { by_keys })
}

fn load_generation_module_parquet(sources: &[PerfSource]) -> Result<GenModuleGrids, AicError> {
    let mut raw: BTreeMap<GenModuleKey, BTreeMap<u32, Grid3<LeafValue>>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        // The `mla_dtype` column is intentionally not read: it is degenerate
        // for generation (collectors hardcode `bfloat16`; decode compute
        // dtype follows the kv-cache dtype).
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let gemm_type_col = reader.col("gemm_type")?;
        let model_col = reader.col("model")?;
        let num_heads_col = reader.col("num_heads")?;
        let tp_size_col = reader.col_optional("tp_size");
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let step_col = reader.col("step")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = GenModuleKey {
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
                gemm_quant: row.str_owned(gemm_type_col)?,
            };
            let num_heads = row.u32(num_heads_col)?;
            let tp_size = match tp_size_col {
                Some(col) => row.u32(col)?.max(1),
                None => 1,
            };
            let model = row.str_owned(model_col)?;
            let native_heads = module_row_native_heads(&model, num_heads, tp_size, path)?;
            let batch_size = row.u32(batch_size_col)?;
            let isl = row.u32(isl_col)?;
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // Generation module: (num_heads, b, s) axis order. First-wins on
            // duplicate rows -- see the note in `load_context_module_parquet`.
            let sequence_tokens = isl + row.u32(step_col)?;
            let inner = raw
                .entry(key)
                .or_default()
                .entry(native_heads)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(batch_size)
                .or_default();
            inner
                .entry(sequence_tokens)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MLA module rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    let by_keys = raw
        .into_iter()
        .map(|(key, by_native)| {
            (
                key,
                by_native
                    .into_iter()
                    .map(|(native, grid)| (native, grid3_to_node(&grid)))
                    .collect(),
            )
        })
        .collect();
    Ok(GenModuleGrids { by_keys })
}

fn load_bmm_parquet(sources: &[PerfSource]) -> Result<BmmGrids, AicError> {
    let mut raw: BTreeMap<BmmKey, BTreeMap<u32, BTreeMap<u32, LeafValue>>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let op_name_col = reader.col("op_name")?;
        let bmm_dtype_col = reader.col("bmm_dtype")?;
        let num_tokens_col = reader.col("num_tokens")?;
        let num_heads_col = reader.col("num_heads")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = BmmKey {
                bmm_quant: row.str_owned(bmm_dtype_col)?,
                pre_or_post: row.str_owned(op_name_col)?,
            };
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // First-wins parity with Python `load_mla_bmm_data`.
            raw.entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(num_tokens_col)?)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no MLA BMM rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    let by_keys = raw
        .into_iter()
        .map(|(key, by_heads)| {
            let converted = by_heads
                .into_iter()
                .map(|(heads, curve)| (heads, curve_to_node(&curve)))
                .collect();
            (key, converted)
        })
        .collect();
    Ok(BmmGrids { by_keys })
}

fn missing(table: &str, data_root: &Path, descriptor: String) -> AicError {
    AicError::PerfDatabase(format!(
        "{table} data missing for {descriptor} at {}",
        data_root.display()
    ))
}

/// Flatten a slice node into `(coords, latency)` points, treating an empty
/// node as a typed coverage miss (mirrors `require_data_slice`'s empty-node
/// check).
fn non_empty_points(
    node: &Node,
    table: &str,
    data_root: &Path,
) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
    let points = perf_interp::node_points(node);
    if points.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "{table} perf data empty for the requested slice at {}",
            data_root.display()
        )));
    }
    Ok(points)
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    /// Byte-equal twin of Python `test_resolve_native_key_ladder` (#1458).
    #[test]
    fn resolve_module_native_ladder() {
        let mut two: BTreeMap<u32, &str> = BTreeMap::new();
        two.insert(64, "a");
        two.insert(128, "b");
        let mut one: BTreeMap<u32, &str> = BTreeMap::new();
        one.insert(128, "b");
        let empty: BTreeMap<u32, &str> = BTreeMap::new();
        assert_eq!(resolve_module_native(&two, Some(128)), Some(&"b")); // exact
        assert_eq!(resolve_module_native(&two, Some(96)), Some(&"a")); // nearest <=
        assert_eq!(resolve_module_native(&two, Some(32)), Some(&"a")); // below all -> smallest
        assert_eq!(resolve_module_native(&one, Some(64)), Some(&"b")); // sole bucket
        assert_eq!(resolve_module_native(&one, None), Some(&"b")); // legacy caller, one bucket
        assert_eq!(resolve_module_native(&two, None), None); // legacy caller, ambiguous
        assert_eq!(resolve_module_native(&empty, Some(128)), None);
    }

    /// Twin of the Python module-loader model-pin tests (#1458).
    #[test]
    fn module_row_native_heads_pins_and_guards() {
        let path = Path::new("test.parquet");
        assert_eq!(
            module_row_native_heads("deepseek-ai/DeepSeek-V3", 16, 1, path).unwrap(),
            128
        );
        // Rank-local tp chain consistent with the pin.
        assert_eq!(
            module_row_native_heads("deepseek-ai/DeepSeek-V3", 64, 2, path).unwrap(),
            128
        );
        let unpinned = module_row_native_heads("unknown/NewModel", 16, 1, path);
        assert!(
            matches!(&unpinned, Err(AicError::PerfDatabase(msg)) if msg.contains("unpinned model")),
            "got {unpinned:?}"
        );
        let stale = module_row_native_heads("deepseek-ai/DeepSeek-V3", 128, 2, path);
        assert!(
            matches!(&stale, Err(AicError::PerfDatabase(msg)) if msg.contains("rank-local")),
            "got {stale:?}"
        );
        let empty = module_row_native_heads("", 16, 1, path);
        assert!(
            matches!(&empty, Err(AicError::PerfDatabase(msg)) if msg.contains("no model value")),
            "got {empty:?}"
        );
    }

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.24.0")
    }

    fn gb200_trtllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/gb200/trtllm/1.3.0rc20")
    }

    fn h200_trtllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/h200_sxm/trtllm/1.3.0rc20")
    }

    fn load_spec(name: &str) -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join(format!("src/aiconfigurator_core/systems/{name}.yaml"));
        SystemSpec::load(&systems_yaml).unwrap_or_else(|_| panic!("{name}.yaml must parse"))
    }

    #[test]
    fn op_level_context_mla_absent_on_vllm_b200() {
        // vLLM b200 ships module-level MLA only; op-level
        // context_mla_perf.parquet
        // is not present. Expect a clear IO error from the lazy loader.
        let table = MlaTable::new(b200_vllm_data_root(), load_spec("b200_sxm"));
        let err = table
            .query_context(
                1,
                1024,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
            )
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn module_level_context_mla_exact_hit() {
        // First row of
        // b200_sxm/mla/vllm/0.24.0/mla_context_module_perf.parquet:
        // mla=bfloat16 kv=bfloat16 gemm=bfloat16 n=128 b=1 isl=1 step=0 latency=0.0955
        let table = MlaTable::new(b200_vllm_data_root(), load_spec("b200_sxm"));
        let latency = table
            .query_context_module(
                1,
                1,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                None,
            )
            .expect("module context MLA query must succeed")
            .latency;
        assert!(
            (latency - 0.0955).abs() < 1e-6,
            "expected recorded module latency, got {latency}"
        );
    }

    #[test]
    fn module_level_generation_mla_smoke() {
        let table = MlaTable::new(b200_vllm_data_root(), load_spec("b200_sxm"));
        // Verify the generation module CSV loads and returns positive
        // values for a representative smoke shape.
        let result = table.query_generation_module(
            1,
            1024,
            128,
            KvCacheQuantMode::Bfloat16,
            GemmQuantMode::Bfloat16,
            None,
        );
        match result {
            Ok(value) => assert!(value.latency > 0.0, "expected positive latency"),
            Err(AicError::PerfDatabase(_)) => {
                // Shape may be outside recorded range; either loader-OK or
                // interpolation-range error is acceptable for this smoke check.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn module_level_generation_mla_fp8_kv_lane_is_live() {
        // Routing contract: the fp8-KV decode module lane (kv=fp8,
        // gemm=fp8_block) must resolve — dropping the degenerate fmha axis
        // made this lane live for fp8-checkpoint V3/R1/Kimi decode (was:
        // FallbackOp -> granular path). No value pins (2026-08 test policy).
        let table = MlaTable::new(h200_trtllm_data_root(), load_spec("h200_sxm"));
        for &(b, s) in &[(8u32, 4097u32), (64, 4096)] {
            let got = table
                .query_generation_module(
                    b,
                    s,
                    16,
                    KvCacheQuantMode::Fp8,
                    GemmQuantMode::Fp8Block,
                    None,
                )
                .unwrap()
                .latency;
            assert!(
                got.is_finite() && got > 0.0,
                "gen_mla_module_fp8kv(b={b}, s={s}): {got}"
            );
        }
    }

    #[test]
    fn mla_bmm_falls_back_to_bfloat16() {
        // gb200/trtllm has mla_bmm data; verify the fallback path works.
        let table = MlaTable::new(gb200_trtllm_data_root(), load_spec("gb200"));
        // Request an unusual quant; loader should fall back to bfloat16.
        let result = table.query_bmm(64, 128, GemmQuantMode::Sq, true);
        // We just verify no panic and the result is a number; if Sq has no
        // bfloat16 fallback either, expect a clean error.
        match result {
            Ok(value) => assert!(value.latency.is_finite() && value.latency > 0.0),
            Err(AicError::PerfDatabase(_)) => {}
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn generation_mla_energy_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{energy_test_spec, write_parquet, Col};
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &tmp.path().join("generation_mla_perf.parquet"),
            &[
                Col::Str("kv_cache_dtype", vec!["bfloat16", "bfloat16"]),
                Col::I64("num_heads", vec![128, 128]),
                Col::I64("batch_size", vec![1, 1]),
                Col::I64("isl", vec![1023, 2047]),
                Col::I64("step", vec![1, 1]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        let table = MlaTable::new(tmp.path().to_path_buf(), energy_test_spec());
        let v = table
            .query_generation(1, 1536, 128, KvCacheQuantMode::Bfloat16)
            .unwrap();
        assert!((v.latency - 2.0).abs() < 1e-9, "latency {}", v.latency);
        assert!(
            (v.energy - 300.0).abs() < 1e-9 * 300.0,
            "energy {}",
            v.energy
        );
    }
}
