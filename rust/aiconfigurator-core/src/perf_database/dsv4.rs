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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use super::perf_interp::{self, Node, OpInterpConfig};
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
    fn compress_ratio(self) -> i64 {
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
    data_root: PathBuf,
    csa_context: OnceLock<Result<ModuleNodes, AicError>>,
    hca_context: OnceLock<Result<ModuleNodes, AicError>>,
    csa_generation: OnceLock<Result<ModuleNodes, AicError>>,
    hca_generation: OnceLock<Result<ModuleNodes, AicError>>,
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleKey {
    architecture: String,
    fmha_quant: String,
    kv_quant: String,
    gemm_quant: String,
}

/// DeepSeek-V4 structural dims for the analytic SOL. Python receives these
/// from the model config (`DeepSeekV4Config`); the Rust perf-database query
/// only carries the architecture string, so the dims are pinned here.
///
/// NOTE: DeepSeek-V4-Pro and DeepSeek-V4-Flash share the architecture string
/// `DeepseekV4ForCausalLM` but differ in shape (Flash: hidden 4096,
/// q_lora 1024, o_groups 8, index_topk 512). The table pins the PRO dims —
/// the SOL only enters as a ratio (util-hold / single-survivor correction),
/// so measured-range resolution is unaffected; only beyond-grid ratios for
/// Flash would drift.
struct Dsv4Dims {
    hidden_size: i64,
    q_lora_rank: i64,
    o_lora_rank: i64,
    head_dim: i64,
    rope_head_dim: i64,
    index_n_heads: i64,
    index_head_dim: i64,
    index_topk: i64,
    window_size: i64,
    o_groups: i64,
    native_heads: i64,
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

fn dsv4_dims(_architecture: &str) -> &'static Dsv4Dims {
    // Only DeepseekV4ForCausalLM exists today; see the Pro-vs-Flash note on
    // `Dsv4Dims`.
    &DSV4_PRO_DIMS
}

impl Dsv4Table {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            csa_context: OnceLock::new(),
            hca_context: OnceLock::new(),
            csa_generation: OnceLock::new(),
            hca_generation: OnceLock::new(),
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
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_context()?,
            AttnKind::Hca => self.load_hca_context()?,
        };
        let node = select_resolved(grids, architecture, fmha_quant, kv_quant, gemm_quant, local_heads)?;

        let dims = dsv4_dims(architecture);
        let cr = attn_kind.compress_ratio();
        let heads = local_heads as i64;
        // Engine coordinates are (prefix, seq, batch); Python's sol_fn is
        // lambda p, s, b: get_sol(b, s, p).
        let sol = move |c: &[f64]| {
            dsv4_attention_sol_ms(
                spec,
                dims,
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
        perf_interp::query(&cfg, node, &[prefix as f64, isl as f64, b as f64])
    }

    /// Generation-DSV4 latency. `sequence_tokens = isl + step` (absolute KV
    /// length). Mirrors Python
    /// `GenerationDeepSeekV4AttentionModule._query_generation_attn_table`
    /// (SILICON path): one 2-axis Grid RAW engine query over the
    /// `{b: {s_total}}` table. The DSV4 generation table is RAGGED (e.g.
    /// `s_total=385` measured only at some batches); the engine's
    /// single-survivor SOL-ratio correction / util-hold replaces the legacy
    /// batch-scaling fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn query_generation(
        &self,
        spec: &SystemSpec,
        attn_kind: AttnKind,
        b: u32,
        sequence_tokens: u32,
        local_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_generation()?,
            AttnKind::Hca => self.load_hca_generation()?,
        };
        let node = select_resolved(grids, architecture, fmha_quant, kv_quant, gemm_quant, local_heads)?;

        let dims = dsv4_dims(architecture);
        let cr = attn_kind.compress_ratio();
        let heads = local_heads as i64;
        // Engine coordinates are (batch, seq); Python's sol_fn is
        // lambda b, s: get_sol(b, s) with prefix=0 and is_context=False.
        let sol = move |c: &[f64]| {
            dsv4_attention_sol_ms(
                spec,
                dims,
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
        perf_interp::query(&cfg, node, &[b as f64, sequence_tokens as f64])
    }

    fn load_csa_context(&self) -> Result<&ModuleNodes, AicError> {
        let cell = self.csa_context.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_csa_context_module_perf.parquet"))
                .map(context_nodes)
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_context(&self) -> Result<&ModuleNodes, AicError> {
        let cell = self.hca_context.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_hca_context_module_perf.parquet"))
                .map(context_nodes)
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_csa_generation(&self) -> Result<&ModuleNodes, AicError> {
        let cell = self.csa_generation.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_csa_generation_module_perf.parquet"))
                .map(generation_nodes)
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_generation(&self) -> Result<&ModuleNodes, AicError> {
        let cell = self.hca_generation.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_hca_generation_module_perf.parquet"))
                .map(generation_nodes)
        });
        cell.as_ref().map_err(clone_err)
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

/// Resolve the `(quant, architecture)` key, then resolve the model's rank-LOCAL
/// head count against the CSV head keys, returning the engine table for that
/// head.
fn select_resolved<'a>(
    grids: &'a ModuleNodes,
    architecture: &str,
    fmha: FmhaQuantMode,
    kv: KvCacheQuantMode,
    gemm: GemmQuantMode,
    local_heads: u32,
) -> Result<&'a Node, AicError> {
    let key = ModuleKey {
        architecture: architecture.to_string(),
        fmha_quant: fmha.name().to_string(),
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
/// `local_heads` is the rank-local head count Python passes as `num_heads`;
/// the rank-local `o_groups` Python receives from the model
/// (`max(1, o_groups // tp)`) is derived here from
/// `tp = native_heads / local_heads`.
#[allow(clippy::too_many_arguments)]
fn dsv4_attention_sol_ms(
    spec: &SystemSpec,
    dims: &Dsv4Dims,
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
    let tp = (dims.native_heads / local_heads.max(1)).max(1);
    let local_o_groups = (dims.o_groups / tp).max(1);

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

fn load_module_parquet(path: &Path) -> Result<ModuleGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let arch_col = reader.col("architecture")?;
    let mla_dtype_col = reader.col("mla_dtype")?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let gemm_type_col = reader.col("gemm_type")?;
    let num_heads_col = reader.col("num_heads")?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<ModuleKey, ByNative> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let key = ModuleKey {
            architecture: row.str_owned(arch_col)?,
            // CSV columns use sglang dtype naming; the query side builds keys
            // from the enum `.name()` (canonical short names). Normalize on
            // load to match Python `_dsv4_normalize_dtype`, which aliases
            // `fp8_e4m3` -> `fp8` for `mla_dtype` (fmha) and `kv_cache_dtype`
            // (kv). `gemm_type` is intentionally left untouched, matching
            // Python (e.g. `fp8_block` is a real value that must pass through).
            fmha_quant: normalize_dsv4_dtype(&row.str_owned(mla_dtype_col)?),
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
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSV4 module rows loaded from {}",
            path.display()
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
            .join("src/aiconfigurator/systems/b200_sxm.yaml");
        SystemSpec::load(&systems_yaml).expect("b200_sxm.yaml must parse")
    }

    fn b200_sglang_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/sglang/0.5.10")
    }

    #[test]
    fn dsv4_data_absent_errors_cleanly() {
        // DSV4 modules aren't collected for vllm/0.19.0; loader must surface
        // a clean error.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0");
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
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM", prefix,
                )
                .unwrap()
        };
        let q_gen = |kind, b, s| {
            table
                .query_generation(
                    &spec, kind, b, s, 16, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM",
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
        approx(q_gen(AttnKind::Csa, 16, 385), 0.1142);
        approx(q_gen(AttnKind::Csa, 16, 200), 0.11328828125);
        approx(q_gen(AttnKind::Csa, 16, 100000), 0.17550076017464525);
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
                    &spec, kind, b, s, 16, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM",
                )
                .unwrap()
        };
        let q_ctx = |kind, b, isl| {
            table
                .query_context(
                    &spec, kind, b, isl, 16, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block, "DeepseekV4ForCausalLM", 0,
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
            )
            .expect("DSV4 context lookup must resolve fp8_e4m3 kv_cache_dtype as fp8");
        assert!(latency.is_finite() && latency > 0.0, "unexpected latency: {latency}");
    }
}
