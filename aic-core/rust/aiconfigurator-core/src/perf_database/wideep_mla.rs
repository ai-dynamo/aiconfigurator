// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang WideEP MLA perf tables (context + generation).
//!
//! Two CSVs with the same column set: `wideep_context_mla_perf.txt` and
//! `wideep_generation_mla_perf.txt`. Columns: framework, version, device,
//! op_name, kernel_source, model, architecture, mla_dtype, kv_cache_dtype,
//! gemm_type, num_heads, batch_size, isl, tp_size, step, latency.
//!
//! Schema-wise the files are nearly identical to the (non-WideEP) MLA
//! module tables, but the nesting in Python's loaders differs:
//!
//! - Context:    `data[kernel_source][fmha/mla_dtype][kv_dtype][num_heads][s][b]`
//! - Generation: `data[kernel_source][kv_dtype][num_heads][b][s = isl + step]`
//!   (Note: generation's `s` collapses `isl + step`, and the `fmha_dtype`
//!   level is absent — generation MLA doesn't tunnel through the fmha
//!   dispatch path the way context does.)
//!
//! Each perf file loads from an ordered, shared-layer-aware source list (see
//! [`PerfSource`]); `WideEpMlaTable::new` degrades to the single primary
//! `data_root/<basename>` with no `kernel_source` filter.
//!
//! Query semantics from Python (perf_interp v2):
//!
//! - Context: `perf_interp.context_grid_config` — Grid resolver over
//!   (num_heads, full_s = s + prefix, b) with SQRT blending on the seq axis
//!   (latency ~ seq^2; the sqrt-on-seq Grid is the principled replacement
//!   for the legacy `extrapolate_data_grid(sqrt_y_value=True)` load-time
//!   pre-expansion). The query returns the raw table value; the operator
//!   layer applies `prefix_correction = (full_s^2 - prefix^2) / full_s^2`.
//! - Generation: `perf_interp.generation_grid_config` — Grid resolver over
//!   (num_heads, b, s), RAW blending (~linear in s), no prefix correction.
//!
//! Beyond the collected range both queries util-hold on the boundary using
//! the WideEP DeepSeek SOL formulas ported from the Python `get_sol`
//! closures. The Python SOLs take `tp_size` while these tables key by
//! `num_heads`; the sol closures map `tp = 128 // num_heads` exactly as the
//! Python `sol_fn` lambdas do (128 total heads for DeepSeek).

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

/// Axes for the context table (sqrt-on-seq Grid).
const CONTEXT_AXES: &[&str] = &["num_heads", "seq_len", "batch"];
/// Axes for the generation table (RAW Grid; seq is innermost).
const GENERATION_AXES: &[&str] = &["num_heads", "batch", "seq_len"];

/// Owner for both WideEP MLA tables. Each side is lazily loaded on first
/// query.
pub struct WideEpMlaTable {
    data_root: PathBuf,
    system_spec: SystemSpec,
    /// Ordered, priority-sorted sources for each WideEP MLA perf file
    /// (shared-layer aware; see [`PerfSource`]). Single-primary, no-filter by
    /// default (`WideEpMlaTable::new`).
    context_sources: Vec<PerfSource>,
    generation_sources: Vec<PerfSource>,
    context: OnceLock<Result<WideEpContextMlaGrids, AicError>>,
    generation: OnceLock<Result<WideEpGenerationMlaGrids, AicError>>,
}

/// Context grids keyed by `(kernel_source, fmha_quant, kv_quant)`.
/// Inner node axes: outer = num_heads, middle = s, inner = b.
pub struct WideEpContextMlaGrids {
    pub by_keys: BTreeMap<ContextKey, Node>,
}

/// Generation grids keyed by `(kernel_source, kv_quant)`. Inner node
/// axes: outer = num_heads, middle = b, inner = s. The `s` axis here is
/// `isl + step` from the CSV (Python collapses them at load time).
pub struct WideEpGenerationMlaGrids {
    pub by_keys: BTreeMap<GenerationKey, Node>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContextKey {
    pub kernel_source: String,
    pub fmha_quant: String,
    pub kv_quant: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GenerationKey {
    pub kernel_source: String,
    pub kv_quant: String,
}

impl WideEpMlaTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf, system_spec: SystemSpec) -> Self {
        Self::with_sources(data_root, system_spec, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each WideEP MLA file falls back to
    /// its primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        system_spec: SystemSpec,
        perf_db_sources: &PerfDbSources,
    ) -> Self {
        let context_sources =
            resolve_op_sources(perf_db_sources, "wideep_context_mla_perf.parquet", &data_root);
        let generation_sources = resolve_op_sources(
            perf_db_sources,
            "wideep_generation_mla_perf.parquet",
            &data_root,
        );
        Self {
            data_root,
            system_spec,
            context_sources,
            generation_sources,
            context: OnceLock::new(),
            generation: OnceLock::new(),
        }
    }

    /// Raw context WideEP MLA latency. Caller is responsible for applying
    /// the `prefix_correction = (full_s^2 - prefix^2) / full_s^2`
    /// multiplier; this matches the (non-WideEP) `MlaTable::query_context`
    /// split. The SOL is evaluated at prefix = 0 accordingly (the Python
    /// `sol_fn` passes prefix=0; samples are prefix=0).
    pub fn query_context(
        &self,
        b: u32,
        full_seq_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        kernel_source: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            kernel_source: kernel_source.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let node = grids.by_keys.get(&key).ok_or_else(|| {
            missing(
                "WideEP context MLA",
                &self.data_root,
                format!("{key:?}"),
            )
        })?;
        // kv_quant keys the table slice only; the Python context SOL never
        // reads it (memory scales by fmha.memory).
        let _ = kv_quant;
        let spec = &self.system_spec;
        // Silicon sol_fn: `tp = 128 // n` then `num_head = 128 // tp`
        // (Python `get_silicon`'s lambda), prefix = 0 (samples are prefix=0).
        let sol = move |c: &[f64]| {
            wideep_context_mla_sol_ms(spec, fmha_quant, wideep_num_head(c[0]), c[1], 0.0, c[2])
        };
        let cfg = OpInterpConfig::grid_sqrt_axis(CONTEXT_AXES, 1, &sol);
        perf_interp::query(
            &cfg,
            node,
            &[num_heads as f64, full_seq_tokens as f64, b as f64],
        )
    }

    /// Raw generation WideEP MLA latency. `sequence_tokens` is the
    /// pre-collapsed `isl + step` (matching Python's `s = s + step` in
    /// the loader).
    pub fn query_generation(
        &self,
        b: u32,
        sequence_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        kernel_source: &str,
    ) -> Result<f64, AicError> {
        let grids = self.load_generation()?;
        let key = GenerationKey {
            kernel_source: kernel_source.to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let node = grids.by_keys.get(&key).ok_or_else(|| {
            missing(
                "WideEP generation MLA",
                &self.data_root,
                format!("{key:?}"),
            )
        })?;
        // Python's generation query is (num_heads, b, s) — middle axis is
        // batch, inner axis is sequence tokens; the node is built with that
        // nesting on load.
        //
        // The Python generation SOL takes an `fmha_quant_mode` that this
        // query surface doesn't carry (the generation table isn't keyed by
        // it and the operator doesn't pass it down). Derive it from the KV
        // mode the way the non-WideEP generation SOL does: fp8 KV -> fp8,
        // else bfloat16. This is exact for every shipped configuration —
        // the collected WideEP data is fp8-KV with fp8_block fmha, and
        // fp8 / fp8_block share the same (memory=1, compute=2) mapping.
        let fmha_quant = if kv_quant == KvCacheQuantMode::Fp8 {
            FmhaQuantMode::Fp8
        } else {
            FmhaQuantMode::Bfloat16
        };
        let spec = &self.system_spec;
        // Silicon sol_fn: `tp = 128 // n` then `num_head = 128 // tp`.
        let sol = move |c: &[f64]| {
            wideep_generation_mla_sol_ms(spec, fmha_quant, wideep_num_head(c[0]), c[1], c[2])
        };
        let cfg = OpInterpConfig::grid(GENERATION_AXES, &sol);
        perf_interp::query(
            &cfg,
            node,
            &[num_heads as f64, b as f64, sequence_tokens as f64],
        )
    }

    // -----------------------------------------------------------------------
    // Point accessors for the util-space empirical layer (algorithm-free:
    // typed `AicError::PerfDatabase` miss on absent slice / empty node, no
    // estimation logic). Coordinate order matches the Python `depth=3`
    // iteration of each `require_data_slice` slice.
    // -----------------------------------------------------------------------

    /// Collected `(num_heads, seq, batch) -> latency` points of the
    /// `(kernel_source, fmha, kv)` context slice. Typed miss when
    /// absent/empty.
    pub fn context_points(
        &self,
        kernel_source: &str,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_context()?;
        let key = ContextKey {
            kernel_source: kernel_source.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("WideEP context MLA", &self.data_root, format!("{key:?}")))?;
        non_empty_points(node, "WideEP context MLA", &self.data_root)
    }

    /// Collected `(num_heads, batch, seq) -> latency` points of the
    /// `(kernel_source, kv)` generation slice. Typed miss when absent/empty.
    pub fn generation_points(
        &self,
        kernel_source: &str,
        kv_quant: KvCacheQuantMode,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_generation()?;
        let key = GenerationKey {
            kernel_source: kernel_source.to_string(),
            kv_quant: kv_quant.name().to_string(),
        };
        let node = grids
            .by_keys
            .get(&key)
            .ok_or_else(|| missing("WideEP generation MLA", &self.data_root, format!("{key:?}")))?;
        non_empty_points(node, "WideEP generation MLA", &self.data_root)
    }

    /// Probe the context table load. Typed missing-data error when the
    /// perf file is absent — Python `get_silicon`'s `raise_if_not_loaded()`
    /// step, which PRECEDES the attn-backend whitelist (`mla.py:1449-1461`).
    pub fn ensure_context_loaded(&self) -> Result<(), AicError> {
        self.load_context().map(|_| ())
    }

    /// Generation-table counterpart of [`Self::ensure_context_loaded`]
    /// (`mla.py:1188-1192`).
    pub fn ensure_generation_loaded(&self) -> Result<(), AicError> {
        self.load_generation().map(|_| ())
    }

    fn load_context(&self) -> Result<&WideEpContextMlaGrids, AicError> {
        let cell = self
            .context
            .get_or_init(|| load_context_parquet(&self.context_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&WideEpGenerationMlaGrids, AicError> {
        let cell = self
            .generation
            .get_or_init(|| load_generation_parquet(&self.generation_sources));
        cell.as_ref().map_err(clone_err)
    }
}

// ---------------------------------------------------------------------------
// SOL formulas — verbatim ports of the Python `get_sol` closures in
// `WideEPContextMLA._query_wideep_context_mla_table` and
// `WideEPGenerationMLA._query_wideep_generation_mla_table` (DeepSeek
// constants: hidden 7168, q_lora 1536, kv_lora 512, rope 64, nope 128,
// v_head 128). Arithmetic ordering mirrors Python for float parity.
// ---------------------------------------------------------------------------

/// See `mla.rs::bf16_tc_flops` — Python indexes the spec directly.
fn bf16_tc_flops(spec: &SystemSpec) -> f64 {
    spec.gpu.bfloat16_tc_flops.unwrap_or(0.0)
}

/// The tables key by `num_heads`; the Python SILICON `sol_fn` lambdas take
/// `tp_size` derived as `tp = 128 // n` and the SOLs then use
/// `num_head = 128 // tp_size`. Compose both floor divisions exactly. (The
/// util-empirical sample mapping differs — Python rounds there:
/// `tp = round(128 / n)`; see `operators/wideep_mla.rs`.)
pub(crate) fn wideep_num_head(n: f64) -> f64 {
    let tp_size = (128.0 / n).floor();
    (128.0 / tp_size).floor()
}

/// WideEP context MLA SOL in ms. `num_head` is the per-rank head count
/// (Python's `128 // tp_size`; the `n -> num_head` mapping lives at the
/// call sites because silicon and empirical map differently). `s` is the
/// chunk / isl length; silicon sol_fns pass `prefix = 0` (samples are
/// prefix=0), the util-empirical query SOL carries the real prefix.
/// Structure (per Python):
/// - q_b / kv_b projections + attention output projection -> `ops`
///   (divided by `bf16_tc_flops * fmha.compute`)
/// - attention flops `2 * nh * (nope*2 + rope) * b * (full_s^2 - prefix^2) // 2`
///   added at full bf16 throughput (no fmha compute factor)
/// - `mem = (q_b_mem + kv_b_mem + attn_mem * 2 + attn_out_mem) * fmha.memory`
/// - `sol = max(sol_math, sol_mem)`
pub(crate) fn wideep_context_mla_sol_ms(
    spec: &SystemSpec,
    fmha_quant: FmhaQuantMode,
    num_head: f64,
    s: f64,
    prefix: f64,
    b: f64,
) -> f64 {
    let hidden_size = 7168.0_f64;
    let q_lora_rank = 1536.0_f64;
    let kv_lora_rank = 512.0_f64;
    let qk_rope_head_dim = 64.0_f64;
    let qk_nope_head_dim = 128.0_f64;
    let v_head_dim = 128.0_f64;

    // q_b projection
    let q_b_flop = 2.0 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b * s;
    let q_b_mem = b * q_lora_rank * s
        + q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim)
        + 2.0 * b * num_head * (qk_rope_head_dim + qk_nope_head_dim) * s;

    // kv_b projection
    let kv_b_flop = 2.0 * kv_lora_rank * num_head * (qk_nope_head_dim + v_head_dim) * b * s;
    let kv_b_mem = b * s * kv_lora_rank
        + num_head * (qk_nope_head_dim + v_head_dim) * kv_lora_rank
        + 2.0 * b * num_head * (qk_nope_head_dim + v_head_dim) * s;

    // attention computation (prefill mode). Python floor-divides by 2; the
    // numerator's leading 2 keeps that exact for integer-valued inputs.
    let full_s = s + prefix;
    let attn_flop = (2.0
        * num_head
        * (qk_nope_head_dim * 2.0 + qk_rope_head_dim)
        * b
        * (full_s * full_s - prefix * prefix)
        / 2.0)
        .floor();
    let attn_mem = b * s * num_head * (qk_nope_head_dim + qk_rope_head_dim) // q read
        + b * full_s * num_head * (qk_nope_head_dim + qk_rope_head_dim) // k read
        + b * full_s * num_head * qk_nope_head_dim // v read
        + b * s * num_head * qk_nope_head_dim; // write

    // attention output projection
    let attn_out_flop = 2.0 * num_head * v_head_dim * hidden_size * b * s;
    let attn_out_mem =
        b * num_head * v_head_dim * s + num_head * v_head_dim * hidden_size + 2.0 * b * hidden_size * s;

    let ops = q_b_flop + kv_b_flop + attn_out_flop;
    let mem_bytes = (q_b_mem + kv_b_mem + attn_mem * 2.0 + attn_out_mem) * fmha_quant.mapping().memory;
    let mut sol_math = ops / (bf16_tc_flops(spec) * fmha_quant.mapping().compute) * 1000.0;
    sol_math += attn_flop / bf16_tc_flops(spec) * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

/// WideEP generation MLA SOL in ms. `num_head` is the per-rank head count
/// (see [`wideep_context_mla_sol_ms`] for the mapping split between
/// silicon and empirical call sites). Structure (per Python): q_b, q_w_kc,
/// s_w_vc and attention-output projections -> `ops` (divided by
/// `bf16_tc_flops * fmha.compute`); the MQA attention flops
/// `2 * b * s * nh * (rope + kv_lora*2)` added at full bf16 throughput;
/// `mem = (q_b + q_w_kc + attn*2 + s_w_vc + attn_out) * fmha.memory`;
/// `sol = max(sol_math, sol_mem)`.
pub(crate) fn wideep_generation_mla_sol_ms(
    spec: &SystemSpec,
    fmha_quant: FmhaQuantMode,
    num_head: f64,
    b: f64,
    s: f64,
) -> f64 {
    let hidden_size = 7168.0_f64;
    let q_lora_rank = 1536.0_f64;
    let kv_lora_rank = 512.0_f64;
    let qk_rope_head_dim = 64.0_f64;
    let qk_nope_head_dim = 128.0_f64;
    let v_head_dim = 128.0_f64;

    // NOTE: qkv_a projection is modeled as a standalone GEMM op
    // (generation_qkv_a_proj_gemm) outside the MLA attention forward path,
    // matching sglang >= 0.5.6 (same note as the Python get_sol).

    // q_b projection
    let q_b_flop = 2.0 * q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim) * b;
    let q_b_mem = b * q_lora_rank
        + q_lora_rank * num_head * (qk_rope_head_dim + qk_nope_head_dim)
        + 2.0 * b * num_head * (qk_rope_head_dim + qk_nope_head_dim);

    // q_w_kc (attention computation)
    let q_w_kc_flop = 2.0 * num_head * qk_nope_head_dim * kv_lora_rank * b;
    let q_w_kc_mem = b * num_head * qk_nope_head_dim
        + num_head * kv_lora_rank * qk_nope_head_dim
        + 2.0 * b * num_head * kv_lora_rank;

    let attn_flop = 2.0 * b * s * num_head * (qk_rope_head_dim + kv_lora_rank * 2.0);
    let attn_mem = b * num_head * (kv_lora_rank + qk_rope_head_dim)
        + b * s * (qk_rope_head_dim + kv_lora_rank)
        + b * num_head * kv_lora_rank;

    // s_w_vc (attention output projection)
    let s_w_vc_flop = 2.0 * b * num_head * kv_lora_rank * v_head_dim;
    let s_w_vc_mem = b * num_head * kv_lora_rank
        + num_head * v_head_dim * kv_lora_rank
        + 2.0 * b * num_head * v_head_dim;

    // attention output projection
    let attn_out_flop = 2.0 * num_head * v_head_dim * hidden_size * b;
    let attn_out_mem =
        b * num_head * v_head_dim + num_head * v_head_dim * hidden_size + 2.0 * b * hidden_size;

    let ops = q_b_flop + q_w_kc_flop + s_w_vc_flop + attn_out_flop;
    let mem_bytes =
        (q_b_mem + q_w_kc_mem + attn_mem * 2.0 + s_w_vc_mem + attn_out_mem) * fmha_quant.mapping().memory;
    let mut sol_math = ops / (bf16_tc_flops(spec) * fmha_quant.mapping().compute) * 1000.0;
    sol_math += attn_flop / bf16_tc_flops(spec) * 1000.0;
    let sol_mem = mem_bytes / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

fn grid3_to_node(grid: &Grid3<f64>) -> Node {
    let mut node = Node::branch();
    for (&a, by_b) in grid {
        for (&b, by_c) in by_b {
            for (&c, &lat) in by_c {
                node.insert(&[a, b, c], lat);
            }
        }
    }
    node
}

/// Load the WideEP context MLA table from an ordered, priority-sorted source
/// list. Sources are read in order; the first source containing a shape wins
/// (`or_insert`), mirroring Python's `_read_filtered_rows` concatenation +
/// `load_wideep_context_mla_data` skip-on-key-conflict. Missing files are
/// skipped (a sibling declared in the manifest need not exist for every
/// system); an error is returned only when no source yields rows.
fn load_context_parquet(sources: &[PerfSource]) -> Result<WideEpContextMlaGrids, AicError> {
    let mut raw: BTreeMap<ContextKey, Grid3<f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let mla_dtype_col = reader.col("mla_dtype")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
        let num_heads_col = reader.col("num_heads")?;
        let batch_size_col = reader.col("batch_size")?;
        let isl_col = reader.col("isl")?;
        let latency_col = reader.col("latency")?;
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = ContextKey {
                kernel_source: row.str_owned(kernel_source_col)?,
                fmha_quant: row.str_owned(mla_dtype_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
            };
            // First-wins parity with Python `load_wideep_context_mla_data`,
            // extended across shared-layer sources (earlier source wins).
            raw.entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(isl_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no WideEP context MLA rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    let by_keys = raw
        .into_iter()
        .map(|(key, grid)| (key, grid3_to_node(&grid)))
        .collect();
    Ok(WideEpContextMlaGrids { by_keys })
}

/// Load the WideEP generation MLA table from an ordered source list. Same
/// first-wins-across-sources + missing-file-skip semantics as
/// [`load_context_parquet`].
fn load_generation_parquet(sources: &[PerfSource]) -> Result<WideEpGenerationMlaGrids, AicError> {
    let mut raw: BTreeMap<GenerationKey, Grid3<f64>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let kernel_source_col = reader.col("kernel_source")?;
        let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
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
            let key = GenerationKey {
                kernel_source: row.str_owned(kernel_source_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
            };
            // Python collapses `s = isl + step` into the seq axis.
            let seq = row.u32(isl_col)? + row.u32(step_col)?;
            // First-wins parity, extended across shared-layer sources.
            raw.entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_default()
                .entry(seq)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no WideEP generation MLA rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
        )));
    }
    let by_keys = raw
        .into_iter()
        .map(|(key, grid)| (key, grid3_to_node(&grid)))
        .collect();
    Ok(WideEpGenerationMlaGrids { by_keys })
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

    fn b200_sglang_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/b200_sxm/sglang/0.5.10")
    }

    fn h200_sglang_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems/data/h200_sxm/sglang/0.5.10")
    }

    fn load_spec(name: &str) -> SystemSpec {
        let systems_yaml = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join(format!("src/aiconfigurator_core/systems/{name}.yaml"));
        SystemSpec::load(&systems_yaml).unwrap_or_else(|_| panic!("{name}.yaml must parse"))
    }

    #[test]
    fn wideep_context_mla_exact_hit() {
        // First DSv3 row in b200_sxm/sglang/0.5.10 wideep_context_mla_perf.txt:
        // kernel=trtllm_mla mla=fp8_block kv=fp8 num_heads=128 b=1 isl=1 latency=0.5470
        let table = WideEpMlaTable::new(b200_sglang_data_root(), load_spec("b200_sxm"));
        let latency = table
            .query_context(
                1,
                1,
                128,
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Fp8Block,
                "trtllm_mla",
            )
            .expect("WideEP context MLA query must succeed");
        assert!(
            (latency - 0.5470).abs() < 1e-3,
            "expected recorded latency, got {latency}"
        );
    }

    #[test]
    fn wideep_generation_mla_exact_hit() {
        // First DSv3 row in b200_sxm/sglang/0.5.10 wideep_generation_mla_perf.txt:
        // kernel=trtllm_mla kv=fp8 num_heads=128 b=1 isl=1 step=0 latency=0.1049
        let table = WideEpMlaTable::new(b200_sglang_data_root(), load_spec("b200_sxm"));
        let latency = table
            .query_generation(1, 1, 128, KvCacheQuantMode::Fp8, "trtllm_mla")
            .expect("WideEP generation MLA query must succeed");
        assert!(
            (latency - 0.1049).abs() < 1e-3,
            "expected recorded latency, got {latency}"
        );
    }

    /// Cross-language parity with the Python v2 engine.
    ///
    /// h200_sxm/sglang/0.5.10 is the root whose wideep tables carry the
    /// `flashinfer` kernel_source Python's query accepts (b200's are
    /// trtllm_mla-only, which `_query_wideep_*_table` rejects at the
    /// attn-backend check). Expected values generated with
    /// `PYTHONPATH=src python3` via `get_database('h200_sxm', 'sglang',
    /// '0.5.10', database_mode="SOL")` (shared layer disabled so Python
    /// loads exactly this primary parquet) and per-query
    /// `database_mode=DatabaseMode.SILICON`, `tp_size=1` (= 128 heads),
    /// `prefix=0`, `fmha=fp8_block`, `kv=fp8`,
    /// `attention_backend='flashinfer'`. Cases: exact hit, interior interp,
    /// beyond-range util-hold.
    ///
    /// NOTE(shared-layer merge): oracle generated pre-shared-layer;
    /// regenerate if this fails (Python's default `get_database` now merges
    /// shared-layer rows, which can add points to these curves; the Rust
    /// side here uses the single-primary `new` constructor).
    #[test]
    fn wideep_mla_queries_match_python_v2_engine() {
        let table = WideEpMlaTable::new(h200_sglang_data_root(), load_spec("h200_sxm"));
        let assert_rel = |got: f64, expected: f64, what: &str| {
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "{what}: rust {got} vs python {expected}"
            );
        };

        // db.query_wideep_context_mla(b, s, prefix=0, tp_size=1, ...)
        let ctx_cases: &[(u32, u32, f64)] = &[
            (4, 4096, 9.6274),             // exact hit
            (4, 6000, 16.671686220608603), // seq interior (sqrt blend)
            (4, 50000, 699.5122474936111), // beyond seq range (util-hold)
        ];
        for &(b, s, expected) in ctx_cases {
            let got = table
                .query_context(
                    b,
                    s,
                    128,
                    KvCacheQuantMode::Fp8,
                    FmhaQuantMode::Fp8Block,
                    "flashinfer",
                )
                .unwrap();
            assert_rel(got, expected, &format!("wideep_context_mla(b={b}, s={s})"));
        }

        // db.query_wideep_generation_mla(b, s, tp_size=1, kv=fp8,
        // fmha=fp8_block, 'flashinfer'). The Rust query derives the SOL's
        // fmha mode from the fp8 KV cache (same mapping as fp8_block).
        let gen_cases: &[(u32, u32, f64)] = &[
            (1, 4096, 0.1017),               // exact hit
            (1, 3000, 0.09988046874999999),  // seq interior (raw blend)
            (1, 100000, 0.17286183638702504), // beyond seq range (util-hold)
        ];
        for &(b, s, expected) in gen_cases {
            let got = table
                .query_generation(b, s, 128, KvCacheQuantMode::Fp8, "flashinfer")
                .unwrap();
            assert_rel(got, expected, &format!("wideep_generation_mla(b={b}, s={s})"));
        }
    }
}
