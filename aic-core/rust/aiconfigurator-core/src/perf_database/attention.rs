// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attention family perf tables: context, generation, encoder.
//!
//! Mirrors the raw table layout used by Python's
//! `aiconfigurator.sdk.operations.attention.{ContextAttention,
//! GenerationAttention, EncoderAttention}._query_*_table` SILICON paths.
//!
//! Each variant nests its data as `(discrete keys) -> 3-D grid` where the
//! 3-D grid is keyed by the three continuous interpolation axes:
//! - context attention: `(num_heads, full_seq_tokens, batch_size)`
//! - generation attention: `(num_heads, kv_seq_tokens, batch_size)` where
//!   `kv_seq_tokens = isl + step`
//! - encoder attention: `(num_heads, seq_tokens, batch_size)`
//!
//! `n_kv` is normalized to `0` when `num_heads == num_key_value_heads`
//! (MHA sentinel), matching Python's `n_kv_lookup` rule. `window_size`
//! defaults to `0` for backends whose collectors don't record it.
//!
//! Context and generation carry one extra, OUTERMOST axis: the kernel-source
//! LANE (`kernel_source`, `"default"` when the column is absent/empty),
//! mirroring Python's `load_{context,generation}_attention_data`
//! (`data[kernel_source][...]`). Callers pass the resolved lane precedence
//! (`lane_order`) and the FIRST lane carrying the requested slice serves the
//! query in full — see [`lane_slice`].
//!
//! Queries resolve on the RAW grids via the shared perf_interp v2 engine
//! (`perf_interp.rs`, mirroring Python `sdk/perf_interp`): context/encoder
//! use the Grid resolver with sqrt-space blending on the seq axis
//! (`context_attention_config`); generation uses the RAW Grid resolver
//! (`generation_attention_config`). Past the collected range — including the
//! truncated large-seq x large-batch staircase corner — the engine holds the
//! boundary util and lets the analytic SOL carry the growth.
//!
//! The query methods on this table return the raw interpolated measured
//! value (`{latency ms, power W, energy W·ms}`, mirroring the Python loader
//! leaves). The operator layer wraps these with prefix correction,
//! SOL/EMPIRICAL fallbacks, and extra fused-op accounting (qk_norm, rope,
//! kv writes).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::gemm::quant_tc_flops;
use super::interpolation::Grid3;
use super::perf_interp::{self, LeafValue, Node, OpInterpConfig};
use super::{kernel_source_ok, SourceResolver};
use crate::common::enums::{FmhaQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use crate::operators::base::SolComponents;
use crate::perf_database::parquet_loader::{PerfReader, PerfRow};

/// Lane key for rows without a usable `kernel_source`. Mirrors Python's
/// `ks = row.get("kernel_source") or "default"` (absent column, null, and
/// empty string all collapse here).
pub const DEFAULT_LANE: &str = "default";

pub struct AttentionTable {
    data_root: PathBuf,
    system_spec: SystemSpec,
    /// Ordered, priority-sorted sources for each attention perf file
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`AttentionTable::new`).
    context_sources: Vec<PerfSource>,
    generation_sources: Vec<PerfSource>,
    encoder_sources: Vec<PerfSource>,
    context: OnceLock<Result<ContextGrids, AicError>>,
    generation: OnceLock<Result<GenerationGrids, AicError>>,
    encoder: OnceLock<Result<EncoderGrids, AicError>>,
}

/// Engine-ready tables: per kernel-source LANE, per discrete key, the raw
/// nested grid as a `Node`. The lane axis is outermost, mirroring Python's
/// `data[kernel_source][...]` nesting.
struct ContextGrids {
    by_lane: BTreeMap<String, BTreeMap<ContextKey, Node>>,
}

struct GenerationGrids {
    by_lane: BTreeMap<String, BTreeMap<GenerationKey, Node>>,
}

/// Encoder attention has no kernel-source axis on the Python side either
/// (`load_encoder_attention_data` keys straight from the quant mode), so it
/// stays lane-free.
struct EncoderGrids {
    by_keys: BTreeMap<EncoderKey, Node>,
}

/// First lane in `lane_order` that carries `key`.
///
/// Mirrors Python `operations/attention.py::_lane_data_slice`: own-lane-first,
/// whole-slice donor gap-fill. A lane whose slice EXISTS serves the query in
/// full — interpolation happens strictly inside that lane's node and a
/// cross-lane point merge WITHIN a slice is intentionally not performed.
///
/// `lane_order` is the walk order resolved python-side
/// (`attention.lane_walk_order`, serialized on the op spec) and is tried
/// FIRST, in order — pinned lanes and density-ranked donors take precedence
/// exactly as resolved. Lanes absent from the table are skipped.
///
/// AIC-1715/1716 follow-up: `lane_order`'s donor/leftover tiers are computed
/// against the ENUMERATION table view (`engine_table_view.py`,
/// `perf_database/table_view.rs`), a lane-blind fold kept for
/// charts/support-matrix; it cannot see this QUERY table's real
/// `kernel_source` values, so a collected lane outside the resolver's static
/// vocabulary (trtllm ships e.g. `torch_flow*`, vllm `vllm_*`) never appears
/// in `lane_order`. Once every given lane has been tried, fall back to any
/// OTHER lane still present in `by_lane` — BTreeMap iteration is sorted-key
/// order, so this fallback is deterministic. A future table-view lane axis
/// would let Python compute this instead and could retire the fallback.
fn lane_slice<'a, K: Ord, V>(
    by_lane: &'a BTreeMap<String, BTreeMap<K, V>>,
    lane_order: &[String],
    key: &K,
) -> Option<&'a V> {
    if let Some(v) = lane_order
        .iter()
        .find_map(|lane| by_lane.get(lane).and_then(|slices| slices.get(key)))
    {
        return Some(v);
    }
    by_lane
        .iter()
        .filter(|(lane, _)| !lane_order.iter().any(|tried| tried == *lane))
        .find_map(|(_, slices)| slices.get(key))
}

/// Lane key for one parquet row (see [`DEFAULT_LANE`]).
fn row_lane(ks_col: Option<usize>, row: &PerfRow) -> Result<String, AicError> {
    Ok(match row.str_optional(ks_col)? {
        Some(ks) if !ks.is_empty() => ks.to_string(),
        _ => DEFAULT_LANE.to_string(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ContextKey {
    fmha_quant: String,
    kv_quant: String,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GenerationKey {
    kv_quant: String,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct EncoderKey {
    fmha_quant: String,
    head_size: u32,
}

impl AttentionTable {
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
    /// a fixed source map is the test-only path). Each attention file falls back to
    /// its primary `data_root/<basename>` when the resolver names no override. No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        system_spec: SystemSpec,
        resolver: &SourceResolver,
    ) -> Result<Self, AicError> {
        let context_sources = resolver.sources_for("context_attention_perf.parquet", &data_root)?;
        let generation_sources =
            resolver.sources_for("generation_attention_perf.parquet", &data_root)?;
        let encoder_sources = resolver.sources_for("encoder_attention_perf.parquet", &data_root)?;
        Ok(Self {
            data_root,
            system_spec,
            context_sources,
            generation_sources,
            encoder_sources,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            encoder: OnceLock::new(),
        })
    }

    /// Raw interpolated context attention value (latency ms + power/energy).
    ///
    /// `full_seq_tokens = isl + prefix` from the caller's perspective. The
    /// operator layer applies the prefix correction multiplier
    /// `(full_s² - prefix²) / full_s²` to latency AND energy (mirroring
    /// Python's `get_silicon`). `lane_order` is the op's serialized
    /// kernel-lane walk order (see [`lane_slice`]).
    #[allow(clippy::too_many_arguments)]
    pub fn query_context(
        &self,
        lane_order: &[String],
        b: u32,
        full_seq_tokens: u32,
        n: u32,
        n_kv: u32,
        head_size: u32,
        window_size: u32,
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
            n_kv_lookup: normalize_kv(n, n_kv),
            head_size,
            window_size,
        };
        let node = lane_slice(&grids.by_lane, lane_order, &key)
            .ok_or_else(|| missing_key(&self.data_root, lane_order, &key))?;
        // Python `perf_interp.context_attention_config`: Grid resolver,
        // sqrt-space blend on the seq axis only (~seq^2 curvature; heads and
        // batch are ~linear). Past the staircase frontier (large seq x large
        // batch, uncollected) the engine holds the boundary util and lets SOL
        // carry the growth. The sol_fn mirrors the Python wiring: samples are
        // full attention, so it is evaluated at prefix=0 with the slice's own
        // kv-head/window/head-size setup; c = [n, full_s, b].
        let spec = &self.system_spec;
        let n_kv_lookup = key.n_kv_lookup;
        let sol = move |c: &[f64]| {
            context_attention_sol_ms(
                spec,
                n_kv_lookup,
                head_size,
                window_size,
                kv_quant,
                c[0],
                c[1],
                c[2],
                attn_flops,
            )
        };
        let cfg = OpInterpConfig::grid_sqrt_axis(&["num_heads", "seq_len", "batch"], 1, &sol);
        perf_interp::query_value(&cfg, node, &[n as f64, full_seq_tokens as f64, b as f64])
    }

    /// Raw interpolated generation attention value (latency ms + energy,
    /// each averaged over the 5 seq samples — Python sums
    /// `get_value(r, "latency")` / `get_value(r, "energy")` and divides by
    /// `sample_cnt`; the per-sample power is dropped at this boundary just
    /// like Python's `_interp_pr(latency, energy=energy)`).
    ///
    /// `kv_seq_tokens` is the total decode context length (Python passes
    /// `s` from the caller; the CSV stores `isl + step`). `lane_order` is the
    /// op's serialized kernel-lane walk order (see [`lane_slice`]).
    #[allow(clippy::too_many_arguments)]
    pub fn query_generation(
        &self,
        lane_order: &[String],
        b: u32,
        kv_seq_tokens: u32,
        n: u32,
        n_kv: u32,
        head_size: u32,
        window_size: u32,
        kv_quant: KvCacheQuantMode,
    ) -> Result<LeafValue, AicError> {
        // Resolve flops BEFORE any perf-data lookup: a missing dtype entry
        // must classify as MissingSystemFlops on both engines, in every mode
        // (mirrors Python's query-entry resolution and GemmTable::query).
        let attn_flops = generation_attn_flops(&self.system_spec, kv_quant)?;
        let grids = self.load_generation()?;
        let key = GenerationKey {
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup: normalize_kv(n, n_kv),
            head_size,
            window_size,
        };
        let node = lane_slice(&grids.by_lane, lane_order, &key)
            .ok_or_else(|| missing_gen_key(&self.data_root, lane_order, &key))?;
        // Python `perf_interp.generation_attention_config`: Grid resolver, RAW
        // blend everywhere (~linear in seq), axes [num_heads][batch][seq_len].
        // The ±10% 5-sample seq averaging is op-level smoothing (decode s
        // drifts across a request) and lives at this wrapper level in Python
        // too — each sample resolves independently via the engine:
        //   s_min = max(1, int(s*0.9)); s_max = max(s_min, int(s*1.1))
        //   s_samples[i] = s_min + (s_max - s_min) * i // (sample_cnt - 1)
        let spec = &self.system_spec;
        let n_kv_lookup = key.n_kv_lookup;
        let sol = move |c: &[f64]| {
            generation_attention_sol_ms(
                spec,
                n_kv_lookup,
                head_size,
                window_size,
                kv_quant,
                c[0],
                c[1],
                c[2],
                attn_flops,
            )
        };
        let cfg = OpInterpConfig::grid(&["num_heads", "batch", "seq_len"], &sol);
        let s = kv_seq_tokens;
        let s_min = ((s as f64 * 0.9) as u32).max(1);
        let s_max = ((s as f64 * 1.1) as u32).max(s_min);
        const SAMPLE_CNT: u32 = 5;
        let mut latency_sum = 0.0_f64;
        let mut energy_sum = 0.0_f64;
        for i in 0..SAMPLE_CNT {
            // Match Python integer arithmetic: multiply before integer divide.
            let s_i = s_min
                + ((u64::from(s_max - s_min) * u64::from(i)) / u64::from(SAMPLE_CNT - 1)) as u32;
            let sample = perf_interp::query_value(&cfg, node, &[n as f64, b as f64, s_i as f64])?;
            latency_sum += sample.latency;
            energy_sum += sample.energy;
        }
        Ok(LeafValue {
            latency: latency_sum / SAMPLE_CNT as f64,
            power: 0.0, // dropped at this boundary, like Python `_interp_pr`
            energy: energy_sum / SAMPLE_CNT as f64,
        })
    }

    /// Raw interpolated encoder (non-causal) attention value
    /// (latency ms + power/energy).
    pub fn query_encoder(
        &self,
        b: u32,
        s: u32,
        n: u32,
        head_size: u32,
        fmha_quant: FmhaQuantMode,
    ) -> Result<LeafValue, AicError> {
        // Resolve flops BEFORE any perf-data lookup: a missing dtype entry
        // must classify as MissingSystemFlops on both engines, in every mode
        // (mirrors Python's query-entry resolution and GemmTable::query).
        let attn_flops = quant_tc_flops(&self.system_spec, fmha_quant.mapping())?;
        let grids = self.load_encoder()?;
        let key = EncoderKey {
            fmha_quant: fmha_quant.name().to_string(),
            head_size,
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing_encoder_key(&self.data_root, &key))?;
        // Encoder is full N^2 (~seq^2 along seq, ~linear along heads/batch);
        // Python reuses `context_attention_config` = sqrt on the seq axis
        // only, raw elsewhere. The SOL differs from context: non-causal (no
        // /2) and no KV-cache read.
        let spec = &self.system_spec;
        let sol = move |c: &[f64]| {
            encoder_attention_sol_ms(spec, head_size, c[0], c[1], c[2], attn_flops)
        };
        let cfg = OpInterpConfig::grid_sqrt_axis(&["num_heads", "seq_len", "batch"], 1, &sol);
        perf_interp::query_value(&cfg, node, &[n as f64, s as f64, b as f64])
    }

    /// Collected `(num_heads, full_seq, batch) -> latency` points of one
    /// context slice, for the operator-layer util-calibration grid (Python's
    /// `_lane_data_slice(_context_attention_data, lane_order, fmha, kv, n_kv,
    /// hs, w)` + `iter_grid(..., depth=3)`). `n_kv_lookup` is the
    /// MHA-normalized kv-head count (`0` == MHA); the slice is served by the
    /// first lane in `lane_order` that carries it. Missing slice / empty node
    /// is a typed `PerfDatabase` miss. No estimation logic here — callers own
    /// the SOL/util math.
    pub fn context_points(
        &self,
        lane_order: &[String],
        fmha_quant: FmhaQuantMode,
        kv_quant: KvCacheQuantMode,
        n_kv_lookup: u32,
        head_size: u32,
        window_size: u32,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup,
            head_size,
            window_size,
        };
        let node = lane_slice(&grids.by_lane, lane_order, &key)
            .ok_or_else(|| missing_key(&self.data_root, lane_order, &key))?;
        let points = perf_interp::node_points(node);
        if points.is_empty() {
            return Err(missing_key(&self.data_root, lane_order, &key));
        }
        Ok(points)
    }

    /// Whether ONE lane carries the full context slice. Mirrors the second
    /// `require_data_slice(table, lane, *prefix_keys, ref_hs, window_size)`
    /// probe in Python `_ref_lane_and_head_size`.
    pub fn context_has_slice(
        &self,
        lane: &str,
        fmha_quant: FmhaQuantMode,
        kv_quant: KvCacheQuantMode,
        n_kv_lookup: u32,
        head_size: u32,
        window_size: u32,
    ) -> Result<bool, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup,
            head_size,
            window_size,
        };
        Ok(grids
            .by_lane
            .get(lane)
            .and_then(|slices| slices.get(&key))
            .is_some())
    }

    /// Distinct collected `head_size` keys of ONE lane under
    /// `(fmha, kv, n_kv_lookup)`, any window — the cross-head_size (XSHAPE)
    /// candidate list (Python's `require_data_slice(wrapper, lane, fmha, kv,
    /// n_kv).keys()`, which is per-lane because the lane axis is outermost).
    /// Returned in ascending order; Python yields CSV insertion order instead,
    /// which is observable only on exact log-distance ties in the reference
    /// pick. Typed `PerfDatabase` miss when nothing matches.
    pub fn context_head_sizes(
        &self,
        lane: &str,
        fmha_quant: FmhaQuantMode,
        kv_quant: KvCacheQuantMode,
        n_kv_lookup: u32,
    ) -> Result<Vec<u32>, AicError> {
        let grids = self.load_context()?;
        let fmha = fmha_quant.name();
        let kv = kv_quant.name();
        let mut sizes: Vec<u32> = Vec::new();
        for key in grids.by_lane.get(lane).into_iter().flat_map(|s| s.keys()) {
            if key.fmha_quant == fmha
                && key.kv_quant == kv
                && key.n_kv_lookup == n_kv_lookup
                && !sizes.contains(&key.head_size)
            {
                sizes.push(key.head_size);
            }
        }
        if sizes.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "context attention data missing for lane={lane}, fmha={fmha}, kv={kv}, \
                 n_kv={n_kv_lookup} at {}",
                self.data_root.display()
            )));
        }
        Ok(sizes)
    }

    /// Every lane this table actually carries (sorted — `by_lane` is a
    /// `BTreeMap`). AIC-1715/1716 follow-up: the XSHAPE reference-grid walk
    /// (`operators::attention::ctx_headsize_ref_grid`) tries the resolved
    /// `lane_order` first, then falls back to every OTHER real lane here —
    /// see `lane_slice`'s doc comment for why the resolved order alone
    /// cannot see a collected `kernel_source` outside its static vocabulary.
    pub fn context_lanes(&self) -> Result<Vec<String>, AicError> {
        Ok(self.load_context()?.by_lane.keys().cloned().collect())
    }

    /// Collected `(num_heads, batch, seq) -> latency` points of one
    /// generation slice. Python calibrates from
    /// `_raw_generation_attention_data`, which in v2 is an alias of the
    /// SOL-clamped working table (`_correct_sol` runs before the alias is
    /// taken) — exactly what [`AttentionTable::load_generation`] produces, so
    /// this IS the RAW-table equivalent. Typed `PerfDatabase` miss when the
    /// slice is absent/empty.
    pub fn generation_points(
        &self,
        lane_order: &[String],
        kv_quant: KvCacheQuantMode,
        n_kv_lookup: u32,
        head_size: u32,
        window_size: u32,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_generation()?;
        let key = GenerationKey {
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup,
            head_size,
            window_size,
        };
        let node = lane_slice(&grids.by_lane, lane_order, &key)
            .ok_or_else(|| missing_gen_key(&self.data_root, lane_order, &key))?;
        let points = perf_interp::node_points(node);
        if points.is_empty() {
            return Err(missing_gen_key(&self.data_root, lane_order, &key));
        }
        Ok(points)
    }

    /// Whether ONE lane carries the full generation slice. Decode twin of
    /// [`AttentionTable::context_has_slice`].
    pub fn generation_has_slice(
        &self,
        lane: &str,
        kv_quant: KvCacheQuantMode,
        n_kv_lookup: u32,
        head_size: u32,
        window_size: u32,
    ) -> Result<bool, AicError> {
        let grids = self.load_generation()?;
        let key = GenerationKey {
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup,
            head_size,
            window_size,
        };
        Ok(grids
            .by_lane
            .get(lane)
            .and_then(|slices| slices.get(&key))
            .is_some())
    }

    /// Distinct collected `head_size` keys of ONE lane under
    /// `(kv, n_kv_lookup)`, any window — the decode XSHAPE candidate list.
    /// Same per-lane scoping and ordering note as
    /// [`AttentionTable::context_head_sizes`].
    pub fn generation_head_sizes(
        &self,
        lane: &str,
        kv_quant: KvCacheQuantMode,
        n_kv_lookup: u32,
    ) -> Result<Vec<u32>, AicError> {
        let grids = self.load_generation()?;
        let kv = kv_quant.name();
        let mut sizes: Vec<u32> = Vec::new();
        for key in grids.by_lane.get(lane).into_iter().flat_map(|s| s.keys()) {
            if key.kv_quant == kv
                && key.n_kv_lookup == n_kv_lookup
                && !sizes.contains(&key.head_size)
            {
                sizes.push(key.head_size);
            }
        }
        if sizes.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "generation attention data missing for lane={lane}, kv={kv}, \
                 n_kv={n_kv_lookup} at {}",
                self.data_root.display()
            )));
        }
        Ok(sizes)
    }

    /// Decode twin of [`AttentionTable::context_lanes`].
    pub fn generation_lanes(&self) -> Result<Vec<String>, AicError> {
        Ok(self.load_generation()?.by_lane.keys().cloned().collect())
    }

    /// Collected `(num_heads, seq, batch) -> latency` points of one encoder
    /// slice (own-shape only; encoder has no transfer ladder). Typed
    /// `PerfDatabase` miss when the slice is absent/empty.
    pub fn encoder_points(
        &self,
        fmha_quant: FmhaQuantMode,
        head_size: u32,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_encoder()?;
        let key = EncoderKey {
            fmha_quant: fmha_quant.name().to_string(),
            head_size,
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing_encoder_key(&self.data_root, &key))?;
        let points = perf_interp::node_points(node);
        if points.is_empty() {
            return Err(missing_encoder_key(&self.data_root, &key));
        }
        Ok(points)
    }

    fn load_context(&self) -> Result<&ContextGrids, AicError> {
        let cell = self.context.get_or_init(|| {
            let raw = load_context_parquet(&self.context_sources)?;
            // No load-time SOL clamp: Python's `_correct_data` historically
            // skipped context attention, and v2 keeps that.
            Ok(ContextGrids {
                by_lane: raw
                    .into_iter()
                    .map(|(lane, slices)| {
                        (
                            lane,
                            slices.into_iter().map(|(k, g)| (k, grid3_to_node(&g))).collect(),
                        )
                    })
                    .collect(),
            })
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&GenerationGrids, AicError> {
        let cell = self.generation.get_or_init(|| {
            let mut raw = load_generation_parquet(&self.generation_sources)?;
            // Mirror Python `GenerationAttention.load_data`: clamp the raw
            // measured rows to SOL (`_correct_sol`) — and nothing else. The
            // v1 load-time grid densification is gone; queries resolve on the
            // RAW table via the perf_interp engine, so the table IS the raw
            // (clamped) data.
            clamp_generation_attention_grids_to_sol(&self.system_spec, &mut raw);
            Ok(GenerationGrids {
                by_lane: raw
                    .into_iter()
                    .map(|(lane, slices)| {
                        (
                            lane,
                            slices.into_iter().map(|(k, g)| (k, grid3_to_node(&g))).collect(),
                        )
                    })
                    .collect(),
            })
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_encoder(&self) -> Result<&EncoderGrids, AicError> {
        let cell = self.encoder.get_or_init(|| {
            let raw = load_encoder_parquet(&self.encoder_sources)?;
            Ok(EncoderGrids {
                by_keys: raw
                    .into_iter()
                    .map(|(k, g)| (k, grid3_to_node(&g)))
                    .collect(),
            })
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Mirror Python's `n_kv_lookup = 0 if n == n_kv else n_kv` (MHA sentinel).
fn normalize_kv(n: u32, n_kv: u32) -> u32 {
    if n_kv == n {
        0
    } else {
        n_kv
    }
}

fn grid3_to_node(grid: &Grid3<LeafValue>) -> Node {
    let mut node = Node::branch();
    for (&x, by_y) in grid {
        for (&y, by_z) in by_y {
            for (&z, &leaf) in by_z {
                node.insert_value(&[x, y, z], leaf);
            }
        }
    }
    node
}

/// Load the context-attention table from an ordered, priority-sorted source
/// list, keyed by kernel-source LANE first. Sources are read in order; the
/// first source containing a key wins (`or_insert`), per lane — Python's
/// dedup key carries `kernel_source` too, so an identical shape measured on
/// two kernels lands in two lanes rather than colliding. `kernel_source`
/// filtering ([`kernel_source_ok`], shared-layer inheritance) still gates
/// which ROWS load; the lane axis only structures what is KEPT. Missing files
/// are skipped; an error is returned only when no source yields rows.
fn load_context_parquet(
    sources: &[PerfSource],
) -> Result<BTreeMap<String, BTreeMap<ContextKey, Grid3<LeafValue>>>, AicError> {
    let mut by_lane: BTreeMap<String, BTreeMap<ContextKey, Grid3<LeafValue>>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let num_heads_col = reader.col("num_heads")?;
        let num_kv_col = reader.col("num_key_value_heads")?;
        let head_dim_col = reader.col("head_dim")?;
        let attn_dtype_col = reader.col("attn_dtype")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let window_size_col = reader.col_optional("window_size");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let num_heads = row.u32(num_heads_col)?;
            let num_kv = row.u32(num_kv_col)?;
            let key = ContextKey {
                fmha_quant: row.str_owned(attn_dtype_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
                n_kv_lookup: normalize_kv(num_heads, num_kv),
                head_size: row.u32(head_dim_col)?,
                window_size: row.u32_optional(window_size_col)?.unwrap_or(0),
            };
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // First-wins parity with Python `load_context_attention_data`,
            // extended across shared-layer sources (earlier source wins).
            by_lane
                .entry(row_lane(ks_col, &row)?)
                .or_default()
                .entry(key)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(row.u32(isl_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || by_lane.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no context-attention rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(by_lane)
}

/// Load the generation-attention table from an ordered, priority-sorted source
/// list, keyed by kernel-source LANE first. Same first-wins-across-sources +
/// missing-file-skip + lane semantics as [`load_context_parquet`].
fn load_generation_parquet(
    sources: &[PerfSource],
) -> Result<BTreeMap<String, BTreeMap<GenerationKey, Grid3<LeafValue>>>, AicError> {
    let mut by_lane: BTreeMap<String, BTreeMap<GenerationKey, Grid3<LeafValue>>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let num_heads_col = reader.col("num_heads")?;
        let num_kv_col = reader.col("num_key_value_heads")?;
        let head_dim_col = reader.col("head_dim")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let step_col = reader.col("step")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let window_size_col = reader.col_optional("window_size");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let num_heads = row.u32(num_heads_col)?;
            let num_kv = row.u32(num_kv_col)?;
            let key = GenerationKey {
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
                n_kv_lookup: normalize_kv(num_heads, num_kv),
                head_size: row.u32(head_dim_col)?,
                window_size: row.u32_optional(window_size_col)?.unwrap_or(0),
            };
            let sequence_tokens = row.u32(isl_col)? + row.u32(step_col)?;
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // First-wins parity with Python `load_generation_attention_data`,
            // extended across shared-layer sources.
            // Grid axis order is `[n][b][s]` to match Python's `interp_3d(n, b, s)`
            // (1-D over n, bilinear over (b, s)). Nesting: num_heads -> batch_size
            // -> sequence_tokens.
            by_lane
                .entry(row_lane(ks_col, &row)?)
                .or_default()
                .entry(key)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_default()
                .entry(sequence_tokens)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || by_lane.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no generation-attention rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(by_lane)
}

/// Speed-of-light context-attention latency in ms, at prefix=0.
///
/// Mirrors Python's `ContextAttention._query_context_attention_table::get_sol`
/// as wired into the perf_interp sol_fn: table rows are full attention, so the
/// formula is evaluated at prefix=0 with `full_s == s` (the table's seq axis).
/// `n_kv_lookup == 0` means MHA (n_kv tracks n); a positive `window_size`
/// smaller than the seq cuts the O(s^2) causal work to O(s*w).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn context_attention_sol(
    spec: &SystemSpec,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    n: f64,
    s: f64,
    b: f64,
    attn_flops: f64,
) -> SolComponents {
    let h = head_size as f64;
    let w = window_size as f64;
    let n_kv = if n_kv_lookup == 0 {
        n
    } else {
        n_kv_lookup as f64
    };
    let ops = if window_size > 0 && s > w {
        2.0 * b * s * w * n * h * 2.0
    } else {
        // the /2 is the causal-mask halving of the s^2 score matrix
        2.0 * b * (s * s) * n * h * 2.0 / 2.0
    };
    // Q read + output write (bf16) + KV write at kv-cache precision.
    let mem_bytes =
        2.0 * b * (n * s * h + n * s * h) + kv_quant.mapping().memory * b * (2.0 * n_kv * s * h);
    let sol_math = ops / attn_flops * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    SolComponents::new(sol_math, sol_mem)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn context_attention_sol_ms(
    spec: &SystemSpec,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    n: f64,
    s: f64,
    b: f64,
    attn_flops: f64,
) -> f64 {
    context_attention_sol(
        spec,
        n_kv_lookup,
        head_size,
        window_size,
        kv_quant,
        n,
        s,
        b,
        attn_flops,
    )
    .time_ms()
}

/// Speed-of-light context-attention latency in ms for a QUERY with a prefix.
///
/// Mirrors Python's `ContextAttention._query_context_attention_table::get_sol`
/// verbatim: `full_s = s + prefix`; the windowed branch fires when `w > 0 &&
/// full_s > w`; the causal branch discounts already-computed prefix work
/// (`full_s² − prefix²`) and the Q/output traffic covers only the new tokens
/// (`full_s − prefix`) while the KV write spans the full sequence.
/// [`context_attention_sol_ms`] is the `prefix = 0` specialization used as
/// the per-sample sol_fn (table rows are full attention); this variant feeds
/// the empirical query SOL. `n_kv` is the REAL kv-head count (not the MHA
/// sentinel).
#[allow(clippy::too_many_arguments)]
pub(crate) fn context_attention_sol_with_prefix(
    spec: &SystemSpec,
    b: f64,
    s: f64,
    prefix: f64,
    n: f64,
    n_kv: f64,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    attn_flops: f64,
) -> SolComponents {
    let h = head_size as f64;
    let w = window_size as f64;
    let full_s = s + prefix;
    let ops = if window_size > 0 && full_s > w {
        2.0 * b * (full_s - prefix) * w * n * h * 2.0
    } else {
        2.0 * b * (full_s * full_s - prefix * prefix) * n * h * 2.0 / 2.0
    };
    let mem_bytes = 2.0 * b * (n * (full_s - prefix) * h + n * (full_s - prefix) * h)
        + kv_quant.mapping().memory * b * (2.0 * n_kv * full_s * h);
    let sol_math = ops / attn_flops * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    SolComponents::new(sol_math, sol_mem)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn context_attention_sol_with_prefix_ms(
    spec: &SystemSpec,
    b: f64,
    s: f64,
    prefix: f64,
    n: f64,
    n_kv: f64,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    attn_flops: f64,
) -> f64 {
    context_attention_sol_with_prefix(
        spec,
        b,
        s,
        prefix,
        n,
        n_kv,
        head_size,
        window_size,
        kv_quant,
        attn_flops,
    )
    .time_ms()
}

/// Speed-of-light encoder-attention latency in ms.
///
/// Mirrors Python's `EncoderAttention._query_encoder_attention_table::get_sol`:
/// non-causal full N^2 (no /2), no KV-cache read — Q/K/V read + output write
/// in bf16 only.
pub(crate) fn encoder_attention_sol(
    spec: &SystemSpec,
    head_size: u32,
    n: f64,
    s: f64,
    b: f64,
    attn_flops: f64,
) -> SolComponents {
    let h = head_size as f64;
    let ops = 2.0 * b * s * s * n * h * 2.0; // 2 for fma, 2 for q*k^t + *v
    let mem_bytes = 2.0 * b * (3.0 * n * s * h + n * s * h); // Q/K/V read + output write, bf16
    let sol_math = ops / attn_flops * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    SolComponents::new(sol_math, sol_mem)
}

pub(crate) fn encoder_attention_sol_ms(
    spec: &SystemSpec,
    head_size: u32,
    n: f64,
    s: f64,
    b: f64,
    attn_flops: f64,
) -> f64 {
    encoder_attention_sol(spec, head_size, n, s, b, attn_flops).time_ms()
}

/// Speed-of-light generation-attention latency in ms.
///
/// Mirrors Python's `GenerationAttention._query_generation_attention_table::get_sol`
/// as wired into the perf_interp sol_fn: c = [n, b, s]. `n_kv_lookup == 0`
/// means MHA (n_kv tracks n); `window_size > 0` clamps `kv_len` to
/// `min(s-1, window_size)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generation_attention_sol(
    spec: &SystemSpec,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    n: f64,
    b: f64,
    s: f64,
    attn_flops: f64,
) -> SolComponents {
    let n_kv = if n_kv_lookup == 0 {
        n
    } else {
        n_kv_lookup as f64
    };
    let kv_len = if window_size > 0 {
        (s - 1.0).min(window_size as f64)
    } else {
        s - 1.0
    };
    let h = head_size as f64;
    let kv_mem = kv_quant.mapping().memory;
    let ops = 2.0 * b * n * h * 2.0 * kv_len;
    let mem_bytes = b * (n * h * 2.0 + 2.0 * n_kv * kv_len * h * kv_mem + n * h * 2.0);
    let sol_math = ops / attn_flops * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    SolComponents::new(sol_math, sol_mem)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generation_attention_sol_ms(
    spec: &SystemSpec,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    n: f64,
    b: f64,
    s: f64,
    attn_flops: f64,
) -> f64 {
    generation_attention_sol(
        spec,
        n_kv_lookup,
        head_size,
        window_size,
        kv_quant,
        n,
        b,
        s,
        attn_flops,
    )
    .time_ms()
}

/// Decode-attention TC-FLOPS: Python derives the FMHA mode from the kv-cache
/// dtype (`fp8` kv -> fp8 FMHA, else bf16) inside `get_sol`; resolve it
/// strictly via `quant_tc_flops`.
/// Decode-attention FMHA mode implied by the kv-cache dtype. fp8 KV implies
/// an fp8-MMA decode kernel only where fp8 tensor cores exist (SM >= 89,
/// Ada and newer); on pre-89 hardware — and on specs without `sm_version`,
/// e.g. XPU — the kernel dequantizes KV and issues the MMA on the bf16
/// pipeline. That is how a100's shipped fp8-kv generation data was collected
/// in the first place, so gating here keeps that silicon usable under the
/// strict per-dtype resolution.
pub(crate) fn generation_attn_mode(spec: &SystemSpec, kv_quant: KvCacheQuantMode) -> FmhaQuantMode {
    let has_fp8_mma = spec.gpu.sm_version.is_some_and(|sm| sm >= 89);
    if kv_quant == KvCacheQuantMode::Fp8 && has_fp8_mma {
        FmhaQuantMode::Fp8
    } else {
        FmhaQuantMode::Bfloat16
    }
}

pub(crate) fn generation_attn_flops(
    spec: &SystemSpec,
    kv_quant: KvCacheQuantMode,
) -> Result<f64, AicError> {
    quant_tc_flops(spec, generation_attn_mode(spec, kv_quant).mapping())
}

/// In-place SOL clamp for every raw row in the generation-attention grid set.
/// Mirrors Python `GenerationAttention._correct_sol` (which v2 keeps) —
/// including its EVERY-LANE scope: any lane may serve a query (own lane or
/// donor gap-fill), so a sub-SOL row is just as wrong in a donor.
fn clamp_generation_attention_grids_to_sol(
    spec: &SystemSpec,
    grids: &mut BTreeMap<String, BTreeMap<GenerationKey, Grid3<LeafValue>>>,
) {
    for (key, grid) in grids.values_mut().flat_map(|slices| slices.iter_mut()) {
        let Some(kv_quant) = kv_cache_quant_by_name(&key.kv_quant) else {
            continue;
        };
        // Silicon data can exist for a dtype whose `*_tc_flops` entry is
        // missing from the system YAML (e.g. b60 fp8): leave that slice
        // unclamped rather than failing the load (mirrors Python
        // `Attention._correct_sol`).
        let Ok(attn_flops) = generation_attn_flops(spec, kv_quant) else {
            continue;
        };
        // Grid order is `[n][b][s]`: outer=n, middle=b, inner=s.
        for (&n, by_b) in grid.iter_mut() {
            for (&b, by_s) in by_b.iter_mut() {
                for (&s, leaf) in by_s.iter_mut() {
                    let sol = generation_attention_sol_ms(
                        spec,
                        key.n_kv_lookup,
                        key.head_size,
                        key.window_size,
                        kv_quant,
                        n as f64,
                        b as f64,
                        s as f64,
                        attn_flops,
                    );
                    if sol > leaf.latency {
                        // Python `_correct_sol` raises only the "latency"
                        // field, preserving power/energy unchanged.
                        leaf.latency = sol;
                    }
                }
            }
        }
    }
}

fn kv_cache_quant_by_name(name: &str) -> Option<KvCacheQuantMode> {
    use KvCacheQuantMode::*;
    Some(match name {
        "bfloat16" => Bfloat16,
        "int8" => Int8,
        "fp8" => Fp8,
        _ => return None,
    })
}

/// Load the encoder-attention table from an ordered, priority-sorted source
/// list. Same first-wins-across-sources + missing-file-skip semantics as
/// [`load_context_parquet`].
fn load_encoder_parquet(
    sources: &[PerfSource],
) -> Result<BTreeMap<EncoderKey, Grid3<LeafValue>>, AicError> {
    let mut by_keys: BTreeMap<EncoderKey, Grid3<LeafValue>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let num_heads_col = reader.col("num_heads")?;
        let head_dim_col = reader.col("head_dim")?;
        let attn_dtype_col = reader.col("attn_dtype")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = EncoderKey {
                fmha_quant: row.str_owned(attn_dtype_col)?,
                head_size: row.u32(head_dim_col)?,
            };
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // First-wins parity with Python `load_encoder_attention_data`,
            // extended across shared-layer sources.
            by_keys
                .entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(isl_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no encoder-attention rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(by_keys)
}

fn missing_key(data_root: &Path, lane_order: &[String], key: &ContextKey) -> AicError {
    AicError::PerfDatabase(format!(
        "context attention data missing for {key:?} in lanes {lane_order:?} at {}",
        data_root.display()
    ))
}

fn missing_gen_key(data_root: &Path, lane_order: &[String], key: &GenerationKey) -> AicError {
    AicError::PerfDatabase(format!(
        "generation attention data missing for {key:?} in lanes {lane_order:?} at {}",
        data_root.display()
    ))
}

fn missing_encoder_key(data_root: &Path, key: &EncoderKey) -> AicError {
    AicError::PerfDatabase(format!(
        "encoder attention data missing for {key:?} at {}",
        data_root.display()
    ))
}

fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.24.0")
    }

    fn b200_sxm_spec() -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/b200_sxm.yaml");
        SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse")
    }
    fn lanes(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    /// Walk order Python serializes for a no-override op on vllm/0.24.0.
    /// The canonical lanes miss and the table's raw vLLM lanes follow in
    /// measured-density order (`resolve_lane_order` + `lane_walk_order`).
    fn vllm_lanes() -> Vec<String> {
        lanes(&[
            "fa3",
            "fla",
            "flashinfer",
            "triton",
            "trtllm_mha",
            "default",
            "vllm_flashinfer_trtllmprefill",
            "vllm_flashinfer_trtllmdecode",
            "vllm_triton_attn",
        ])
    }

    #[test]
    fn context_attention_mha_normalizes_n_kv_to_zero() {
        // Caller passes n_kv=64 (== n, MHA); the loader normalizes the lookup
        // to the stored n_kv=0 lane. Pinned RELATIVELY: the n_kv=64 query
        // must resolve to exactly the n_kv=0 query's row — no recorded
        // constant to re-mint on data refreshes.
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let via_mha = table
            .query_context(
                &vllm_lanes(),
                4,
                16384,
                64,
                64,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .expect("MHA lookup must normalize and find the row")
            .latency;
        let via_zero = table
            .query_context(
                &vllm_lanes(),
                4,
                16384,
                64,
                0,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .expect("stored-lane lookup must succeed")
            .latency;
        assert!(
            (via_mha - via_zero).abs() < 1e-12,
            "n_kv=64 must normalize onto the stored n_kv=0 lane: {via_mha} vs {via_zero}"
        );
        assert!(via_mha.is_finite() && via_mha > 0.0);
    }

    #[test]
    fn context_attention_missing_quant_combo_errors() {
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        // vLLM b200 context attention has fmha=bfloat16 only; Fp8 fmha
        // is genuinely absent.
        match table.query_context(
            &vllm_lanes(),
            1,
            1024,
            64,
            1,
            128,
            0,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Fp8,
        ) {
            Err(AicError::PerfDatabase(_)) => {}
            other => panic!("expected PerfDatabase error, got {other:?}"),
        }
    }

    #[test]
    fn encoder_attention_missing_head_size_errors() {
        // vLLM b200 encoder attention collects head_dim 64/72/80 only; an
        // uncollected head size is a genuine missing key.
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        match table.query_encoder(1, 1024, 16, 128, FmhaQuantMode::Bfloat16) {
            Err(AicError::PerfDatabase(_)) => {}
            other => panic!("expected PerfDatabase error, got {other:?}"),
        }
    }

    #[test]
    fn context_attention_lazy_loads_once() {
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let first = table
            .query_context(
                &vllm_lanes(),
                8,
                16384,
                64,
                1,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .unwrap();
        let second = table
            .query_context(
                &vllm_lanes(),
                8,
                16384,
                64,
                1,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .unwrap();
        assert_eq!(first, second);
    }

    /// ENERGY oracle on a synthetic power-carrying fixture. Python twin
    /// (pandas fixture, `energy_test_fixtures` spec):
    ///
    /// ```text
    /// db.query_context_attention(2, 1536, 0, 16, 16, bfloat16, bfloat16,
    ///                            SILICON, window_size=0, head_size=128)
    /// # -> latency=1.8660254037844386, energy=279.9038105676658
    /// ```
    ///
    /// s=1536 sqrt-blends the seq axis between (isl 1024, lat 1.0, power
    /// 100) and (isl 2048, lat 3.0, power 200): latency = ((1+sqrt 3)/2)^2,
    /// POWER lerps linearly to 150, energy = 150 * latency.
    #[test]
    fn context_attention_energy_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{energy_test_spec, write_parquet, Col};
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &tmp.path().join("context_attention_perf.parquet"),
            &[
                Col::Str("attn_dtype", vec!["bfloat16", "bfloat16"]),
                Col::Str("kv_cache_dtype", vec!["bfloat16", "bfloat16"]),
                Col::I64("batch_size", vec![2, 2]),
                Col::I64("isl", vec![1024, 2048]),
                Col::I64("num_heads", vec![16, 16]),
                Col::I64("num_key_value_heads", vec![16, 16]),
                Col::I64("head_dim", vec![128, 128]),
                Col::I64("step", vec![0, 0]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        let table = AttentionTable::new(tmp.path().to_path_buf(), energy_test_spec());
        let v = table
            .query_context(
                &lanes(&[DEFAULT_LANE]),
                2,
                1536,
                16,
                16,
                128,
                0,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
            )
            .unwrap();
        assert!(
            ((v.latency - 1.8660254037844386) / 1.8660254037844386).abs() < 1e-9,
            "latency {}",
            v.latency
        );
        assert!(
            ((v.energy - 279.9038105676658) / 279.9038105676658).abs() < 1e-9,
            "energy {}",
            v.energy
        );
    }

    /// Hand-derived synthetic pin for the decode 5-sample seq averaging —
    /// the one piece of attention-wrapper math not covered by the shared
    /// `perf_interp` synthetic suites. Grid: seq {50, 150} with latency
    /// LINEAR in seq (0.5 -> 1.5). Query s=100: s_min=90, s_max=110,
    /// samples [90, 95, 100, 105, 110]; each resolves by the raw linear
    /// blend to s/100, so the average is exactly 1.0. No production data.
    #[test]
    fn generation_five_sample_averaging_matches_hand_derivation() {
        use crate::perf_database::energy_test_fixtures::{
            write_energy_systems_root, write_parquet, Col,
        };
        let tmp = tempfile::tempdir().expect("tmpdir");
        let data = write_energy_systems_root(tmp.path());
        write_parquet(
            &data.join("generation_attention_perf.parquet"),
            &[
                Col::Str("kv_cache_dtype", vec!["bfloat16", "bfloat16"]),
                Col::I64("batch_size", vec![1, 1]),
                Col::I64("isl", vec![50, 150]),
                Col::I64("num_heads", vec![8, 8]),
                Col::I64("num_key_value_heads", vec![1, 1]),
                Col::I64("head_dim", vec![128, 128]),
                Col::I64("step", vec![0, 0]),
                Col::F64("latency", vec![0.5, 1.5]),
            ],
        );
        let spec = SystemSpec::load(&tmp.path().join("testsys.yaml")).expect("testsys spec");
        let table = AttentionTable::new(data, spec);
        let latency = table
            .query_generation(
                &lanes(&[DEFAULT_LANE]),
                1,
                100,
                8,
                1,
                128,
                0,
                KvCacheQuantMode::Bfloat16,
            )
            .expect("query must succeed")
            .latency;
        assert!(
            (latency - 1.0).abs() < 1e-9,
            "5-sample average over a linear grid must be exactly 1.0, got {latency}"
        );
    }
    // ------------------------------------------------------------------
    // Kernel-source lanes (AIC-1715/1716)
    // ------------------------------------------------------------------

    fn b200_sglang_0514_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/sglang/0.5.14")
    }

    /// Walk order Python serializes for a no-override op on
    /// b200_sxm/sglang/0.5.14 (`resolve_lane_order` + `lane_walk_order`):
    /// `triton` is the framework-default map lane and the donor tier is
    /// density-ranked, so `trtllm_mha` (66 slices) precedes `flashinfer` (10).
    fn sglang_default_lanes() -> Vec<String> {
        lanes(&["triton", "trtllm_mha", "flashinfer", "fa3", "fla", "default"])
    }

    /// Same table, `attention_backend="flashinfer"`: the override pins
    /// `flashinfer` ahead of the map lane `triton`.
    fn sglang_flashinfer_lanes() -> Vec<String> {
        lanes(&["flashinfer", "triton", "trtllm_mha", "fa3", "fla", "default"])
    }

    /// Two-lane in-memory context table: `head` carries only the `hs=128`
    /// slice, `donor` only the `hs=64` slice, and both carry `hs=256` with
    /// DIFFERENT latencies. Rows land on exact grid coordinates so the engine
    /// returns the stored value verbatim.
    fn in_memory_two_lane_context_table() -> AttentionTable {
        let key = |head_size: u32| ContextKey {
            fmha_quant: "bfloat16".to_string(),
            kv_quant: "bfloat16".to_string(),
            n_kv_lookup: 0,
            head_size,
            window_size: 0,
        };
        let leaf = |latency: f64| {
            let mut node = Node::branch();
            node.insert(&[8, 1024, 1], latency);
            node
        };
        let mut by_lane: BTreeMap<String, BTreeMap<ContextKey, Node>> = BTreeMap::new();
        by_lane
            .entry("head".to_string())
            .or_default()
            .extend([(key(128), leaf(1.0)), (key(256), leaf(2.0))]);
        by_lane
            .entry("donor".to_string())
            .or_default()
            .extend([(key(64), leaf(3.0)), (key(256), leaf(4.0))]);
        let table = AttentionTable::new(PathBuf::from("test-data"), b200_sxm_spec());
        assert!(table.context.set(Ok(ContextGrids { by_lane })).is_ok());
        table
    }

    fn two_lane_query(table: &AttentionTable, lane_order: &[String], head_size: u32) -> Result<f64, AicError> {
        table
            .query_context(
                lane_order,
                1,
                1024,
                8,
                8,
                head_size,
                0,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
            )
            .map(|v| v.latency)
    }

    /// The lane-walk rules, on a synthetic table so the assertions are about
    /// the walk and nothing else (mirrors Python `_lane_data_slice`): the head
    /// lane serves a slice it owns; a slice it LACKS falls through to the
    /// donor at whole-slice granularity; a NAMED `lane_order` is tried first,
    /// in order; and once it is exhausted, `lane_slice` falls back to any
    /// OTHER lane still in the table (AIC-1715/1716 follow-up — the resolved
    /// order's leftover tier is computed against a lane-blind Python view, see
    /// `lane_slice`'s doc comment) — never a re-sort of the given order, only
    /// a last-resort scan in `by_lane`'s BTreeMap (sorted-key) order. Only
    /// when NO lane anywhere in the table carries the slice is it a true miss.
    #[test]
    fn context_lane_walk_serves_head_lane_then_donor_then_misses() {
        let table = in_memory_two_lane_context_table();
        let order = lanes(&["head", "donor"]);

        // Own-lane serve: both lanes carry hs=256, the head lane wins. NO
        // cross-lane merge — the donor's 4.0 never enters the interpolation.
        assert_eq!(two_lane_query(&table, &order, 256).unwrap(), 2.0);
        // Head-lane-only slice.
        assert_eq!(two_lane_query(&table, &order, 128).unwrap(), 1.0);
        // Donor gap-fill: the head lane has no hs=64 slice at all.
        assert_eq!(two_lane_query(&table, &order, 64).unwrap(), 3.0);
        // Reversing the order flips which lane serves the shared slice.
        assert_eq!(two_lane_query(&table, &lanes(&["donor", "head"]), 256).unwrap(), 4.0);

        // Named-lane walk exhausted (neither "fa3" nor "default" exists in
        // this table): falls back to any other lane still present. Both
        // "donor" and "head" carry hs=256; BTreeMap order ("donor" < "head")
        // picks "donor" — not a miss.
        let fallback = two_lane_query(&table, &lanes(&["fa3", "default"]), 256);
        assert_eq!(fallback.unwrap(), 4.0, "fallback must scan the table's OTHER lanes, not just miss");
        // No lane anywhere in the table carries hs=999 — a genuine miss the
        // fallback cannot paper over.
        let miss = two_lane_query(&table, &order, 999);
        assert!(matches!(miss, Err(AicError::PerfDatabase(_))), "got {miss:?}");
        // An empty order still falls back to every table lane: "donor" lacks
        // hs=128 (head-lane-only slice, see above) but "head" carries it.
        let fallback = two_lane_query(&table, &[], 128);
        assert_eq!(fallback.unwrap(), 1.0, "empty order must still fall back to the table's own lanes");
    }

    /// The loader KEEPS every `kernel_source` as its own lane instead of
    /// collapsing them (Python `load_context_attention_data` ->
    /// `data[kernel_source][...]`). b200_sxm/sglang/0.5.14 collects
    /// `trtllm_mha`, `triton` and `flashinfer`; a collapsed table would let
    /// whichever kernel happened to be read first answer for all three.
    #[test]
    fn loader_keeps_every_kernel_source_lane() {
        let table = AttentionTable::new(b200_sglang_0514_data_root(), b200_sxm_spec());
        let ctx_lanes: Vec<&str> = table
            .load_context()
            .expect("context table must load")
            .by_lane
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(ctx_lanes, vec!["flashinfer", "triton", "trtllm_mha"]);
        let gen_lanes: Vec<&str> = table
            .load_generation()
            .expect("generation table must load")
            .by_lane
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(gen_lanes, vec!["flashinfer", "triton", "trtllm_mha"]);
    }

    /// Lane selection changes the ANSWER, anchored on Python. Oracle
    /// generation (shared layer OFF, SILICON) on b200_sxm/sglang/0.5.14:
    ///
    /// ```text
    /// db = perf_database.get_database_view("b200_sxm", "sglang", "0.5.14",
    ///     allow_missing_data=True, database_mode=DatabaseMode.SILICON,
    ///     shared_layer=False)
    /// order = attention.lane_walk_order(db._context_attention_data,
    ///     attention.resolve_lane_order(db, override), attention._CONTEXT_SLICE_DEPTH)
    /// float(ContextAttention._query_context_attention_table(db, b, s, 0, 64, 8,
    ///     KVCacheQuantMode.bfloat16, FMHAQuantMode.bfloat16,
    ///     database_mode=DatabaseMode.SILICON, window_size=0, head_size=128,
    ///     lane_order=order))
    /// ```
    ///
    /// The `(bfloat16, bfloat16, n_kv=8, hs=128, w=0)` slice is collected in
    /// `trtllm_mha` and `flashinfer` but NOT in `triton`, so the no-override
    /// walk gap-fills from `trtllm_mha` while the `flashinfer` override is
    /// served by its own lane — ~26% apart, which is exactly the kernel
    /// difference the lane axis exists to preserve.
    #[test]
    fn context_lane_selection_matches_python_oracles() {
        let table = AttentionTable::new(b200_sglang_0514_data_root(), b200_sxm_spec());
        // (lane_order, b, full_s, expected)
        let cases: &[(Vec<String>, u32, u32, f64)] = &[
            (sglang_default_lanes(), 4, 4096, 0.9642000198364258),
            (sglang_default_lanes(), 2, 2000, 0.15175898359817958),
            (sglang_flashinfer_lanes(), 4, 4096, 1.2178943634033204),
            (sglang_flashinfer_lanes(), 2, 2000, 0.2179272220499033),
        ];
        for (order, b, s, expected) in cases {
            let got = table
                .query_context(
                    order,
                    *b,
                    *s,
                    64,
                    8,
                    128,
                    0,
                    KvCacheQuantMode::Bfloat16,
                    FmhaQuantMode::Bfloat16,
                )
                .expect("lane-served query must succeed")
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "({order:?}, b={b}, s={s}): rust {got} vs python {expected}"
            );
        }

        // Donor gap-fill past TWO lanes: hs=64 is collected only in
        // `trtllm_mha` (Python oracle, same call with head_size=64, n_kv=64).
        // Oracle regenerated post-rebase onto post-#1479 main (d221111
        // replaced the grid-hold nearest-path snap with a tapered joint-log
        // util transfer); the old snap-based constant (3.8185985565185545)
        // is stale — re-verified live-Python == live-Rust on this exact call.
        let got = table
            .query_context(
                &sglang_default_lanes(),
                4,
                4096,
                64,
                64,
                64,
                0,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
            )
            .expect("donor gap-fill must succeed")
            .latency;
        let expected = 3.749649873014346;
        assert!(
            ((got - expected) / expected).abs() < 1e-9,
            "donor gap-fill: rust {got} vs python {expected}"
        );
    }

    /// Decode twin of [`context_lane_selection_matches_python_oracles`].
    /// Oracle: `GenerationAttention._query_generation_attention_table(db, b, s,
    /// 64, 8, KVCacheQuantMode.bfloat16, database_mode=DatabaseMode.SILICON,
    /// window_size=0, head_size=128, lane_order=order)` on the same view.
    #[test]
    fn generation_lane_selection_matches_python_oracles() {
        let table = AttentionTable::new(b200_sglang_0514_data_root(), b200_sxm_spec());
        let cases: &[(Vec<String>, u32, u32, f64)] = &[
            (sglang_default_lanes(), 8, 4096, 0.028358187839476155),
            (sglang_default_lanes(), 16, 2048, 0.027373899476369846),
            (sglang_flashinfer_lanes(), 8, 4096, 0.03659885138535173),
            (sglang_flashinfer_lanes(), 16, 2048, 0.038325441025081085),
        ];
        for (order, b, s, expected) in cases {
            let got = table
                .query_generation(order, *b, *s, 64, 8, 128, 0, KvCacheQuantMode::Bfloat16)
                .expect("lane-served query must succeed")
                .latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "({order:?}, b={b}, s={s}): rust {got} vs python {expected}"
            );
        }
    }
}
