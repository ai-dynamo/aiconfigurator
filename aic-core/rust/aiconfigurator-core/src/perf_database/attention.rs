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
//! Queries resolve on the RAW grids via the shared perf_interp v2 engine
//! (`perf_interp.rs`, mirroring Python `sdk/perf_interp`): context/encoder
//! use the Grid resolver with sqrt-space blending on the seq axis
//! (`context_attention_config`); generation uses the RAW Grid resolver
//! (`generation_attention_config`). Past the collected range — including the
//! truncated large-seq x large-batch staircase corner — the engine holds the
//! boundary util and lets the analytic SOL carry the growth.
//!
//! The query methods on this table return raw interpolated latency in ms.
//! The operator layer wraps these with prefix correction, SOL/EMPIRICAL
//! fallbacks, and extra fused-op accounting (qk_norm, rope, kv writes).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::{FmhaQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::interpolation::Grid3;
use super::perf_interp::{self, Node, OpInterpConfig};
use crate::perf_database::parquet_loader::PerfReader;

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

/// Engine-ready tables: per discrete key, the raw nested grid as a `Node`.
struct ContextGrids {
    by_keys: BTreeMap<ContextKey, Node>,
}

struct GenerationGrids {
    by_keys: BTreeMap<GenerationKey, Node>,
}

struct EncoderGrids {
    by_keys: BTreeMap<EncoderKey, Node>,
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
        Self::with_sources(data_root, system_spec, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each attention file falls back to
    /// its primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        system_spec: SystemSpec,
        perf_db_sources: &PerfDbSources,
    ) -> Self {
        let context_sources =
            resolve_op_sources(perf_db_sources, "context_attention_perf.parquet", &data_root);
        let generation_sources = resolve_op_sources(
            perf_db_sources,
            "generation_attention_perf.parquet",
            &data_root,
        );
        let encoder_sources =
            resolve_op_sources(perf_db_sources, "encoder_attention_perf.parquet", &data_root);
        Self {
            data_root,
            system_spec,
            context_sources,
            generation_sources,
            encoder_sources,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            encoder: OnceLock::new(),
        }
    }

    /// Raw interpolated context attention latency in ms.
    ///
    /// `full_seq_tokens = isl + prefix` from the caller's perspective. The
    /// operator layer applies the prefix correction multiplier
    /// `(full_s² - prefix²) / full_s²`.
    pub fn query_context(
        &self,
        b: u32,
        full_seq_tokens: u32,
        n: u32,
        n_kv: u32,
        head_size: u32,
        window_size: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup: normalize_kv(n, n_kv),
            head_size,
            window_size,
        };
        let node = grids.by_keys.get(&key).ok_or_else(|| missing_key(&self.data_root, &key))?;
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
                spec, n_kv_lookup, head_size, window_size, kv_quant, fmha_quant, c[0], c[1], c[2],
            )
        };
        let cfg = OpInterpConfig::grid_sqrt_axis(&["num_heads", "seq_len", "batch"], 1, &sol);
        perf_interp::query(&cfg, node, &[n as f64, full_seq_tokens as f64, b as f64])
    }

    /// Raw interpolated generation attention latency in ms.
    ///
    /// `kv_seq_tokens` is the total decode context length (Python passes
    /// `s` from the caller; the CSV stores `isl + step`).
    pub fn query_generation(
        &self,
        b: u32,
        kv_seq_tokens: u32,
        n: u32,
        n_kv: u32,
        head_size: u32,
        window_size: u32,
        kv_quant: KvCacheQuantMode,
    ) -> Result<f64, AicError> {
        let grids = self.load_generation()?;
        let key = GenerationKey {
            kv_quant: kv_quant.name().to_string(),
            n_kv_lookup: normalize_kv(n, n_kv),
            head_size,
            window_size,
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing_gen_key(&self.data_root, &key))?;
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
                spec, n_kv_lookup, head_size, window_size, kv_quant, c[0], c[1], c[2],
            )
        };
        let cfg = OpInterpConfig::grid(&["num_heads", "batch", "seq_len"], &sol);
        let s = kv_seq_tokens;
        let s_min = ((s as f64 * 0.9) as u32).max(1);
        let s_max = ((s as f64 * 1.1) as u32).max(s_min);
        const SAMPLE_CNT: u32 = 5;
        let mut latency_sum = 0.0_f64;
        for i in 0..SAMPLE_CNT {
            // Match Python integer arithmetic: multiply before integer divide.
            let s_i = s_min + ((u64::from(s_max - s_min) * u64::from(i)) / u64::from(SAMPLE_CNT - 1)) as u32;
            latency_sum += perf_interp::query(&cfg, node, &[n as f64, b as f64, s_i as f64])?;
        }
        Ok(latency_sum / SAMPLE_CNT as f64)
    }

    /// Raw interpolated encoder (non-causal) attention latency in ms.
    pub fn query_encoder(
        &self,
        b: u32,
        s: u32,
        n: u32,
        head_size: u32,
        fmha_quant: FmhaQuantMode,
    ) -> Result<f64, AicError> {
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
        let sol =
            move |c: &[f64]| encoder_attention_sol_ms(spec, head_size, fmha_quant, c[0], c[1], c[2]);
        let cfg = OpInterpConfig::grid_sqrt_axis(&["num_heads", "seq_len", "batch"], 1, &sol);
        perf_interp::query(&cfg, node, &[n as f64, s as f64, b as f64])
    }

    /// Collected `(num_heads, full_seq, batch) -> latency` points of one
    /// context slice, for the operator-layer util-calibration grid (Python's
    /// `require_data_slice(_context_attention_data, fmha, kv, n_kv, hs, w)` +
    /// `iter_grid(..., depth=3)`). `n_kv_lookup` is the MHA-normalized
    /// kv-head count (`0` == MHA). Missing slice / empty node is a typed
    /// `PerfDatabase` miss. No estimation logic here — callers own the
    /// SOL/util math.
    pub fn context_points(
        &self,
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
        let node = grids.by_keys.get(&key).ok_or_else(|| missing_key(&self.data_root, &key))?;
        let points = perf_interp::node_points(node);
        if points.is_empty() {
            return Err(missing_key(&self.data_root, &key));
        }
        Ok(points)
    }

    /// Distinct collected `head_size` keys under `(fmha, kv, n_kv_lookup)`,
    /// any window — the cross-head_size (XSHAPE) candidate list (Python's
    /// `require_data_slice(wrapper, fmha, kv, n_kv).keys()`). Returned in
    /// ascending order; Python yields CSV insertion order instead, which is
    /// observable only on exact log-distance ties in the reference pick.
    /// Typed `PerfDatabase` miss when nothing matches.
    pub fn context_head_sizes(
        &self,
        fmha_quant: FmhaQuantMode,
        kv_quant: KvCacheQuantMode,
        n_kv_lookup: u32,
    ) -> Result<Vec<u32>, AicError> {
        let grids = self.load_context()?;
        let fmha = fmha_quant.name();
        let kv = kv_quant.name();
        let mut sizes: Vec<u32> = Vec::new();
        for key in grids.by_keys.keys() {
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
                "context attention data missing for fmha={fmha}, kv={kv}, \
                 n_kv={n_kv_lookup} at {}",
                self.data_root.display()
            )));
        }
        Ok(sizes)
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
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing_gen_key(&self.data_root, &key))?;
        let points = perf_interp::node_points(node);
        if points.is_empty() {
            return Err(missing_gen_key(&self.data_root, &key));
        }
        Ok(points)
    }

    /// Distinct collected `head_size` keys under `(kv, n_kv_lookup)`, any
    /// window — the decode XSHAPE candidate list. Same ordering note as
    /// [`AttentionTable::context_head_sizes`].
    pub fn generation_head_sizes(
        &self,
        kv_quant: KvCacheQuantMode,
        n_kv_lookup: u32,
    ) -> Result<Vec<u32>, AicError> {
        let grids = self.load_generation()?;
        let kv = kv_quant.name();
        let mut sizes: Vec<u32> = Vec::new();
        for key in grids.by_keys.keys() {
            if key.kv_quant == kv && key.n_kv_lookup == n_kv_lookup && !sizes.contains(&key.head_size) {
                sizes.push(key.head_size);
            }
        }
        if sizes.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "generation attention data missing for kv={kv}, n_kv={n_kv_lookup} at {}",
                self.data_root.display()
            )));
        }
        Ok(sizes)
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
                by_keys: raw.into_iter().map(|(k, g)| (k, grid3_to_node(&g))).collect(),
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
                by_keys: raw.into_iter().map(|(k, g)| (k, grid3_to_node(&g))).collect(),
            })
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_encoder(&self) -> Result<&EncoderGrids, AicError> {
        let cell = self.encoder.get_or_init(|| {
            let raw = load_encoder_parquet(&self.encoder_sources)?;
            Ok(EncoderGrids {
                by_keys: raw.into_iter().map(|(k, g)| (k, grid3_to_node(&g))).collect(),
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

fn grid3_to_node(grid: &Grid3<f64>) -> Node {
    let mut node = Node::branch();
    for (&x, by_y) in grid {
        for (&y, by_z) in by_y {
            for (&z, &lat) in by_z {
                node.insert(&[x, y, z], lat);
            }
        }
    }
    node
}

/// Load the context-attention table from an ordered, priority-sorted source
/// list. Sources are read in order; the first source containing a key wins
/// (`or_insert`). Missing files are skipped; an error is returned only when no
/// source yields rows.
fn load_context_parquet(
    sources: &[PerfSource],
) -> Result<BTreeMap<ContextKey, Grid3<f64>>, AicError> {
    let mut by_keys: BTreeMap<ContextKey, Grid3<f64>> = BTreeMap::new();
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
            // First-wins parity with Python `load_context_attention_data`,
            // extended across shared-layer sources (earlier source wins).
            by_keys
                .entry(key)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(row.u32(isl_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no context-attention rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(by_keys)
}

/// Load the generation-attention table from an ordered, priority-sorted source
/// list. Same first-wins-across-sources + missing-file-skip semantics as
/// [`load_context_parquet`].
fn load_generation_parquet(
    sources: &[PerfSource],
) -> Result<BTreeMap<GenerationKey, Grid3<f64>>, AicError> {
    let mut by_keys: BTreeMap<GenerationKey, Grid3<f64>> = BTreeMap::new();
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
            // First-wins parity with Python `load_generation_attention_data`,
            // extended across shared-layer sources.
            // Grid axis order is `[n][b][s]` to match Python's `interp_3d(n, b, s)`
            // (1-D over n, bilinear over (b, s)). Nesting: num_heads -> batch_size
            // -> sequence_tokens.
            by_keys
                .entry(key)
                .or_default()
                .entry(num_heads)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_default()
                .entry(sequence_tokens)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no generation-attention rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(by_keys)
}

/// Speed-of-light context-attention latency in ms, at prefix=0.
///
/// Mirrors Python's `ContextAttention._query_context_attention_table::get_sol`
/// as wired into the perf_interp sol_fn: table rows are full attention, so the
/// formula is evaluated at prefix=0 with `full_s == s` (the table's seq axis).
/// `n_kv_lookup == 0` means MHA (n_kv tracks n); a positive `window_size`
/// smaller than the seq cuts the O(s^2) causal work to O(s*w).
#[allow(clippy::too_many_arguments)]
pub(crate) fn context_attention_sol_ms(
    spec: &SystemSpec,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
    n: f64,
    s: f64,
    b: f64,
) -> f64 {
    let bf16_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    if bf16_flops <= 0.0 {
        return 0.0;
    }
    let h = head_size as f64;
    let w = window_size as f64;
    let n_kv = if n_kv_lookup == 0 { n } else { n_kv_lookup as f64 };
    let ops = if window_size > 0 && s > w {
        2.0 * b * s * w * n * h * 2.0
    } else {
        // the /2 is the causal-mask halving of the s^2 score matrix
        2.0 * b * (s * s) * n * h * 2.0 / 2.0
    };
    // Q read + output write (bf16) + KV write at kv-cache precision.
    let mem_bytes =
        2.0 * b * (n * s * h + n * s * h) + kv_quant.mapping().memory * b * (2.0 * n_kv * s * h);
    let sol_math = ops / bf16_flops * 1000.0 / fmha_quant.mapping().compute;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
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
    fmha_quant: FmhaQuantMode,
) -> f64 {
    let bf16_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    if bf16_flops <= 0.0 {
        return 0.0;
    }
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
    let sol_math = ops / bf16_flops * 1000.0 / fmha_quant.mapping().compute;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// Speed-of-light encoder-attention latency in ms.
///
/// Mirrors Python's `EncoderAttention._query_encoder_attention_table::get_sol`:
/// non-causal full N^2 (no /2), no KV-cache read — Q/K/V read + output write
/// in bf16 only.
pub(crate) fn encoder_attention_sol_ms(
    spec: &SystemSpec,
    head_size: u32,
    fmha_quant: FmhaQuantMode,
    n: f64,
    s: f64,
    b: f64,
) -> f64 {
    let bf16_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    if bf16_flops <= 0.0 {
        return 0.0;
    }
    let h = head_size as f64;
    let ops = 2.0 * b * s * s * n * h * 2.0; // 2 for fma, 2 for q*k^t + *v
    let mem_bytes = 2.0 * b * (3.0 * n * s * h + n * s * h); // Q/K/V read + output write, bf16
    let sol_math = ops / bf16_flops * 1000.0 / fmha_quant.mapping().compute;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// Speed-of-light generation-attention latency in ms.
///
/// Mirrors Python's `GenerationAttention._query_generation_attention_table::get_sol`
/// as wired into the perf_interp sol_fn: c = [n, b, s]. `n_kv_lookup == 0`
/// means MHA (n_kv tracks n); `window_size > 0` clamps `kv_len` to
/// `min(s-1, window_size)`.
pub(crate) fn generation_attention_sol_ms(
    spec: &SystemSpec,
    n_kv_lookup: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    n: f64,
    b: f64,
    s: f64,
) -> f64 {
    let bf16_flops = spec.gpu.bfloat16_tc_flops.unwrap_or(0.0);
    if bf16_flops <= 0.0 {
        return 0.0;
    }
    let n_kv = if n_kv_lookup == 0 { n } else { n_kv_lookup as f64 };
    let kv_len = if window_size > 0 {
        (s - 1.0).min(window_size as f64)
    } else {
        s - 1.0
    };
    let quant_mode_gen_compute = if kv_quant == KvCacheQuantMode::Fp8 {
        FmhaQuantMode::Fp8.mapping().compute
    } else {
        FmhaQuantMode::Bfloat16.mapping().compute
    };
    let h = head_size as f64;
    let kv_mem = kv_quant.mapping().memory;
    let ops = 2.0 * b * n * h * 2.0 * kv_len;
    let mem_bytes = b * (n * h * 2.0 + 2.0 * n_kv * kv_len * h * kv_mem + n * h * 2.0);
    let sol_math = ops / bf16_flops * 1000.0 / quant_mode_gen_compute;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// In-place SOL clamp for every raw row in the generation-attention grid set.
/// Mirrors Python `GenerationAttention._correct_sol` (which v2 keeps).
fn clamp_generation_attention_grids_to_sol(
    spec: &SystemSpec,
    grids: &mut BTreeMap<GenerationKey, Grid3<f64>>,
) {
    if spec.gpu.bfloat16_tc_flops.unwrap_or(0.0) <= 0.0 {
        return;
    }
    for (key, grid) in grids.iter_mut() {
        let Some(kv_quant) = kv_cache_quant_by_name(&key.kv_quant) else {
            continue;
        };
        // Grid order is `[n][b][s]`: outer=n, middle=b, inner=s.
        for (&n, by_b) in grid.iter_mut() {
            for (&b, by_s) in by_b.iter_mut() {
                for (&s, latency) in by_s.iter_mut() {
                    let sol = generation_attention_sol_ms(
                        spec,
                        key.n_kv_lookup,
                        key.head_size,
                        key.window_size,
                        kv_quant,
                        n as f64,
                        b as f64,
                        s as f64,
                    );
                    if sol > *latency {
                        *latency = sol;
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
) -> Result<BTreeMap<EncoderKey, Grid3<f64>>, AicError> {
    let mut by_keys: BTreeMap<EncoderKey, Grid3<f64>> = BTreeMap::new();
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
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no encoder-attention rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    Ok(by_keys)
}

fn missing_key(data_root: &Path, key: &ContextKey) -> AicError {
    AicError::PerfDatabase(format!(
        "context attention data missing for {key:?} at {}",
        data_root.display()
    ))
}

fn missing_gen_key(data_root: &Path, key: &GenerationKey) -> AicError {
    AicError::PerfDatabase(format!(
        "generation attention data missing for {key:?} at {}",
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
            .join("src/aiconfigurator_core/systems/data/b200_sxm/vllm/0.19.0")
    }

    fn b200_sxm_spec() -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/b200_sxm.yaml");
        SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse")
    }

    fn gb200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/gb200/vllm/0.19.0")
    }

    fn gb200_spec() -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/gb200.yaml");
        SystemSpec::load(&systems_yaml).expect("gb200.yaml must parse")
    }

    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails
    #[test]
    fn generation_query_ragged_corner_matches_python_v2_engine() {
        // Ragged-corner regime: large batch x long kv, off-measured-grid —
        // v2 resolves it on the RAW [n][b][s] grid (no densification), with
        // the truncated corner handled by boundary-util hold, then 5-sample
        // s-averaging at the wrapper level. Expected value generated from
        // Python `db.query_generation_attention(256, 2561, 32, 8, bfloat16,
        // SILICON, window_size=0, head_size=128)` on gb200/vllm/0.19.0.
        let table = AttentionTable::new(gb200_vllm_data_root(), gb200_spec());
        let latency = table
            .query_generation(256, 2561, 32, 8, 128, 0, KvCacheQuantMode::Bfloat16)
            .expect("ragged-corner query must succeed");
        let expected = 0.4923998240128304;
        assert!(
            ((latency - expected) / expected).abs() < 1e-9,
            "rust {latency} vs python {expected}"
        );
    }

    #[test]
    fn context_attention_exact_hit() {
        // First row of b200_sxm/vllm/0.19.0/context_attention_perf.txt:
        // batch=8 isl=16384 n=64 n_kv=1 head_dim=128 attn=bfloat16 kv=fp8 step=0 latency=19.82
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let latency = table
            .query_context(
                8,
                16384,
                64,
                1,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .expect("query must succeed");
        assert!(
            (latency - 19.820667266845703).abs() < 1e-9,
            "expected recorded latency, got {latency}"
        );
    }

    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails
    #[test]
    fn generation_attention_query_matches_python_v2_engine() {
        // batch=32 isl=1 n=64 n_kv=4 head_dim=128 kv=fp8 step=1 (stored
        // sequence_tokens = isl + step = 2). The 5-sample averaging spans
        // s ∈ [max(1,int(2*0.9)), max(..,int(2*1.1))] = [1, 2], i.e.
        // s_samples = [1, 1, 1, 1, 2]; s=1 is below the collected range, so
        // it resolves via boundary-util hold on the RAW grid. Expected value
        // generated from Python `db.query_generation_attention(32, 2, 64, 4,
        // fp8, SILICON, 0, 128)` on b200_sxm/vllm/0.19.0.
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let latency = table
            .query_generation(32, 2, 64, 4, 128, 0, KvCacheQuantMode::Fp8)
            .expect("query must succeed");
        let expected = 0.008451361751014535;
        assert!(
            ((latency - expected) / expected).abs() < 1e-9,
            "rust {latency} vs python {expected}"
        );
    }

    /// Values generated from the Python v2 engine on the same table
    /// (`db.query_context_attention(b, s, 0, 64, 1, fp8, bfloat16, SILICON,
    /// window_size=0, head_size=128)` on b200_sxm/vllm/0.19.0): an exact hit,
    /// a seq interpolation (sqrt-space blend between s=10240 and s=12288),
    /// and a batch past the staircase frontier (b=64 where s=16384 collects
    /// only up to b=8 -> boundary-util hold). The two engines must agree
    /// because they implement the same resolution chain.
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails
    #[test]
    fn context_attention_query_matches_python_v2_engine() {
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let cases: &[(u32, u32, f64)] = &[
            (8, 16384, 19.820667266845703),  // exact hit
            (8, 12000, 11.515825737734879),  // seq interp (sqrt blend)
            (64, 16384, 158.56533813476562), // batch beyond staircase (util-hold)
        ];
        for &(b, s, expected) in cases {
            let got = table
                .query_context(
                    b,
                    s,
                    64,
                    1,
                    128,
                    0,
                    KvCacheQuantMode::Fp8,
                    FmhaQuantMode::Bfloat16,
                )
                .unwrap();
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "(b={b},s={s}): rust {got} vs python {expected}"
            );
        }
    }

    #[test]
    fn context_attention_mha_normalizes_n_kv_to_zero() {
        // Real MHA row from vLLM b200 context attention:
        // b=4 isl=16384 n=64 n_kv=64 head=128 fmha=bfloat16 kv=fp8 latency=9.98
        // Caller passes n_kv=64; loader normalizes to n_kv_lookup=0 since
        // n==n_kv (MHA). Query should hit the same recorded row.
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let latency = table
            .query_context(
                4,
                16384,
                64,
                64,
                128,
                0,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
            )
            .expect("MHA lookup must normalize and find the row");
        assert!(
            (latency - 9.983466466267904).abs() < 1e-9,
            "expected recorded MHA latency, got {latency}"
        );
    }

    #[test]
    fn context_attention_missing_quant_combo_errors() {
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        // vLLM b200 context attention has fmha=bfloat16 only; Fp8 fmha
        // is genuinely absent.
        match table.query_context(
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

    /// Values generated from the Python v2 engine on the same table
    /// (`db.query_encoder_attention(b, s, 16, 64, bfloat16, SILICON)` on
    /// b200_sxm/vllm/0.19.0): an exact hit, a seq interpolation (sqrt-space
    /// blend between s=1296 and s=1500), and a batch past the staircase
    /// frontier (b=64 where s=65536 collects only up to b=2 -> boundary-util
    /// hold).
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails
    #[test]
    fn encoder_attention_query_matches_python_v2_engine() {
        let table = AttentionTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let cases: &[(u32, u32, f64)] = &[
            (1, 1024, 0.03258133431275686), // exact hit
            (2, 1400, 0.0779337721462867),  // seq interp (sqrt blend)
            (64, 65536, 9775.049479166666), // batch beyond staircase (util-hold)
        ];
        for &(b, s, expected) in cases {
            let got = table
                .query_encoder(b, s, 16, 64, FmhaQuantMode::Bfloat16)
                .unwrap();
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "(b={b},s={s}): rust {got} vs python {expected}"
            );
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
}
