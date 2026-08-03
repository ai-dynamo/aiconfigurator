// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Attention operators: context, generation, encoder.
//!
//! Mirrors `aiconfigurator.sdk.operations.attention.{ContextAttention,
//! GenerationAttention, EncoderAttention}`. Each holds its config-time
//! attention shape (n, n_kv, head_size, window_size, quant modes) and
//! wraps the raw `AttentionTable` query with:
//!
//! - prefix correction `(full_s² − prefix²) / full_s²` for context paths
//! - fused-op extras for context: qk_norm (optional), apply_rope, kv_write
//!   via the analytic `mem_op` formula
//! - 1.1× correction factor on the extras (matches Python)
//! - `seq_imbalance_correction_scale` / `gen_seq_imbalance_correction_scale`
//!   multiplier for unbalanced sequence distributions
//! - `scale_factor` scaling at the end
//! - database-mode dispatch (SILICON / HYBRID / EMPIRICAL) with the
//!   util-space empirical layer: exact-window then window=0 util carriers,
//!   plus the cross-head_size (XSHAPE) transfer ladder

use serde::{Deserialize, Serialize};
use crate::common::enums::{DatabaseMode, FmhaQuantMode, KvCacheQuantMode, TransferKind};
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::operators::base::{PerformanceResult, Source};
use crate::operators::util_empirical::{self, UtilGrid};
use crate::perf_database::attention::{
    context_attention_sol_ms, context_attention_sol_with_prefix_ms, encoder_attention_sol_ms,
    generation_attention_sol_ms,
};
use crate::perf_database::PerfDatabase;

/// Analytic memory-op latency (ms). Matches Python's
/// `PerfDatabase.query_mem_op` empirical path (the only path used for
/// SILICON queries — there's no perf table for raw memory ops).
pub(crate) fn mem_op_latency_ms(spec: &SystemSpec, mem_bytes: f64) -> f64 {
    let mem_bw = spec.gpu.mem_bw.max(1.0);
    let scaling = spec.gpu.mem_bw_empirical_scaling_factor.max(1e-9);
    let constant = spec.gpu.mem_empirical_constant_latency;
    (mem_bytes / (mem_bw * scaling) + constant) * 1000.0
}

fn prefix_correction(full_s: u32, prefix: u32) -> f64 {
    if full_s == 0 {
        return 0.0;
    }
    let f = full_s as f64;
    let p = prefix as f64;
    (f * f - p * p) / (f * f)
}

// ---------------------------------------------------------------------------
// Cross-head_size (XSHAPE) transfer support, mirroring the module-level
// helpers of `operations/attention.py`.
// ---------------------------------------------------------------------------

/// Prefill-attention util vs head_size, relative to head_size=128 (Python's
/// `_ATTN_PREFILL_HS_RATIO`). Used to rescale a borrowed util curve when the
/// exact head_size has no collected data. DECODE util is ~head_size-
/// independent (memory-bound KV read), so decode transfer uses no table.
const ATTN_PREFILL_HS_RATIO_TRTLLM: &[(u32, f64)] =
    &[(64, 0.58), (128, 1.00), (192, 1.10), (256, 1.17), (512, 1.20)];
const ATTN_PREFILL_HS_RATIO_SGLANG: &[(u32, f64)] =
    &[(64, 0.60), (128, 1.00), (192, 1.18), (256, 1.32), (512, 1.38)];
const ATTN_PREFILL_HS_RATIO_VLLM: &[(u32, f64)] =
    &[(64, 0.60), (128, 1.00), (192, 1.27), (256, 1.51), (512, 1.60)];

/// Prefill-attention util ratio vs head_size=128, log2-interpolated between
/// table points and clamped at the ends. Unknown backend -> 1.0 (no
/// correction). Mirrors Python `_attn_prefill_hs_ratio`.
fn attn_prefill_hs_ratio(backend: &str, head_size: u32) -> f64 {
    let table = match backend {
        "trtllm" => ATTN_PREFILL_HS_RATIO_TRTLLM,
        "sglang" => ATTN_PREFILL_HS_RATIO_SGLANG,
        "vllm" => ATTN_PREFILL_HS_RATIO_VLLM,
        _ => return 1.0,
    };
    if let Some(&(_, ratio)) = table.iter().find(|&&(h, _)| h == head_size) {
        return ratio;
    }
    let (first, last) = (table[0], table[table.len() - 1]);
    if head_size <= first.0 {
        return first.1;
    }
    if head_size >= last.0 {
        return last.1;
    }
    // Bracketing keys exist by the checks above (table is sorted ascending).
    let (lo, lo_ratio) = *table.iter().rev().find(|&&(h, _)| h < head_size).expect("lower bracket");
    let (hi, hi_ratio) = *table.iter().find(|&&(h, _)| h > head_size).expect("upper bracket");
    let t = ((head_size as f64).log2() - (lo as f64).log2()) / ((hi as f64).log2() - (lo as f64).log2());
    lo_ratio + t * (hi_ratio - lo_ratio)
}

/// Pick the reference head_size to transfer util from. Prefer 128 — the
/// canonical, most densely collected head_size and the ratio table's
/// reference point; otherwise the nearest collected head_size in log space
/// (1-D normalised-log argmin == `|log2 h − log2 target|` argmin; ties keep
/// the first candidate). Mirrors Python `_ref_head_size`.
fn ref_head_size(available: &[u32], target: u32) -> Option<u32> {
    let avail: Vec<u32> = available.iter().copied().filter(|&h| h != 0).collect();
    if avail.is_empty() {
        return None;
    }
    if avail.contains(&128) {
        return Some(128);
    }
    let features: Vec<Vec<f64>> = avail.iter().map(|&h| vec![h as f64]).collect();
    let idx = util_empirical::nearest_candidate_index(&[target as f64], &features)?;
    Some(avail[idx])
}

// ---------------------------------------------------------------------------
// Context attention
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub n_kv: u32,
    pub head_size: u32,
    pub window_size: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
    pub fmha_quant_mode: FmhaQuantMode,
    pub use_qk_norm: bool,
    /// Context-parallel factor (Python's `_cp_size`, = `cp_size`). When `>1`,
    /// prefill FMHA is modeled as rank-0's two zigzag chunks:
    /// `ctx(c, prefix) + ctx(c, prefix + isl - c)` with `c = ceil(isl / 2cp)`.
    /// Defaults to 1 (no CP). The fused rope/kv_write/qk_norm extras are still
    /// added once, not per chunk.
    #[serde(default = "crate::operators::gemm::default_seq_split")]
    pub cp_size: u32,
}

impl ContextAttentionOp {
    pub fn new(
        name: impl Into<String>,
        n: u32,
        n_kv: u32,
        head_size: u32,
        kv_cache_dtype: KvCacheQuantMode,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            n_kv,
            head_size,
            window_size: 0,
            kv_cache_dtype,
            fmha_quant_mode,
            use_qk_norm: false,
            cp_size: 1,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        isl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
    ) -> Result<PerformanceResult, AicError> {
        // Mirror Python's `ContextAttention._ctx(s, pfx)`: each chunk
        // dispatches through the database mode — the silicon table at the
        // full sequence `s + pfx` with the prefix correction, or the
        // util-space empirical estimate (whose SOL already discounts prefix).
        let ctx = |s: u32, pfx: u32| -> Result<(f64, Source), AicError> {
            query_context_attention_table(
                db,
                batch_size,
                s,
                pfx,
                self.n,
                self.n_kv,
                self.head_size,
                self.window_size,
                self.kv_cache_dtype,
                self.fmha_quant_mode,
            )
        };

        // Context parallelism (SGLang AllGather / zigzag): model rank 0's two
        // balanced chunks. c = ceil(isl / 2cp); rank 0 owns chunk 0 (prefix
        // unchanged) and chunk 2cp-1 (attends almost the full sequence). Only
        // the FMHA table term is split; the fused extras below are added once.
        let (mut latency, source) = if self.cp_size > 1 {
            let c = isl.div_ceil(2 * self.cp_size).max(1);
            let (first, first_source) = ctx(c, prefix)?;
            let (second, second_source) = ctx(c, prefix + isl - c)?;
            (first + second, first_source.combine(second_source))
        } else {
            ctx(isl, prefix)?
        };

        // Fused-op extras (qk_norm optional, rope + kv_write mandatory).
        let q_num = (self.n * self.head_size) as f64;
        let k_num = (self.n_kv * self.head_size) as f64;
        let v_num = (self.n_kv * self.head_size) as f64;
        let spec = &db.system_spec;

        let mut extra = 0.0;
        if self.use_qk_norm {
            let qk_norm = 2.0 * mem_op_latency_ms(spec, q_num * 2.0)
                + 2.0 * mem_op_latency_ms(spec, k_num * 2.0);
            extra += qk_norm * 2.0; // elementwise before norm
        }
        let apply_rope = 2.0 * mem_op_latency_ms(spec, q_num * 2.0 + k_num * 2.0);
        let kv_write = mem_op_latency_ms(spec, k_num * self.fmha_quant_mode.mapping().memory)
            + mem_op_latency_ms(spec, v_num * self.fmha_quant_mode.mapping().memory);
        extra += apply_rope + kv_write;

        latency += extra * 1.1; // Python's correction factor for the fused extras.

        if seq_imbalance_correction_scale != 1.0 {
            latency *= seq_imbalance_correction_scale;
        }

        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Generation attention
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub n_kv: u32,
    pub head_size: u32,
    pub window_size: u32,
    pub kv_cache_dtype: KvCacheQuantMode,
}

impl GenerationAttentionOp {
    pub fn new(
        name: impl Into<String>,
        n: u32,
        n_kv: u32,
        head_size: u32,
        kv_cache_dtype: KvCacheQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            n_kv,
            head_size,
            window_size: 0,
            kv_cache_dtype,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        kv_seq_tokens: u32,
        gen_seq_imbalance_correction_scale: f64,
    ) -> Result<PerformanceResult, AicError> {
        let (table_latency, source) = query_generation_attention_table(
            db,
            batch_size,
            kv_seq_tokens,
            self.n,
            self.n_kv,
            self.head_size,
            self.window_size,
            self.kv_cache_dtype,
        )?;
        let mut latency = table_latency;
        if gen_seq_imbalance_correction_scale != 1.0 {
            latency *= gen_seq_imbalance_correction_scale;
        }
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Encoder attention (non-causal; vision encoder path)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncoderAttentionOp {
    pub name: String,
    pub scale_factor: f64,
    pub n: u32,
    pub head_size: u32,
    pub fmha_quant_mode: FmhaQuantMode,
    /// Partial-RoPE fraction (Python `_partial_rotary_factor`): 1.0 = full
    /// rotation, 0.5 = half head_dim rotated (Qwen3-VL), 0.0 = no RoPE.
    /// Adds `factor * 2 * mem_op(Q+K bytes) * 1.1` on top of the table
    /// latency. Defaults to 0.0 for pre-field opspecs.
    #[serde(default)]
    pub partial_rotary_factor: f64,
}

impl EncoderAttentionOp {
    pub fn new(
        name: impl Into<String>,
        n: u32,
        head_size: u32,
        fmha_quant_mode: FmhaQuantMode,
    ) -> Self {
        Self {
            name: name.into(),
            scale_factor: 1.0,
            n,
            head_size,
            fmha_quant_mode,
            partial_rotary_factor: 0.0,
        }
    }

    pub fn query(
        &self,
        db: &PerfDatabase,
        batch_size: u32,
        s: u32,
    ) -> Result<PerformanceResult, AicError> {
        let (mut latency, source) = query_encoder_attention_table(
            db,
            batch_size,
            s,
            self.n,
            self.head_size,
            self.fmha_quant_mode,
        )?;
        // Partial RoPE extra (Python `EncoderAttention.query`,
        // operations/attention.py): Q + K bytes (bf16) over all tokens,
        // rotated fractionally, with the 1.1 correction factor. Added on top
        // of the mode-dispatched table value (Python applies it after the
        // table query in every mode).
        if self.partial_rotary_factor > 0.0 {
            let qk_num = (self.n as u64) * (self.head_size as u64); // MHA: q == k
            let qk_bytes = 2 * (qk_num * 2) * (batch_size as u64) * (s as u64);
            let apply_rope =
                self.partial_rotary_factor * 2.0 * mem_op_latency_ms(&db.system_spec, qk_bytes as f64);
            latency += apply_rope * 1.1;
        }
        Ok(PerformanceResult::new(latency, source)
            .clamp_non_negative()
            .scaled(self.scale_factor))
    }
}

// ---------------------------------------------------------------------------
// Database-mode dispatch, mirroring the Python `_query_*_attention_table`
// classmethods (`operations/attention.py`): SILICON queries the table; HYBRID
// converts a typed silicon miss into the util-space empirical estimate;
// EMPIRICAL always estimates. The SOL diagnostic modes never reach the
// compiled engine (the routing gate delegates them to the Python step).
// ---------------------------------------------------------------------------

/// Context attention latency for `(b, s, prefix, shape)` under the database's
/// query mode. The silicon path queries the table at `full_s = s + prefix`
/// and applies the prefix correction; the empirical path bakes the prefix
/// into the query SOL instead (mirroring Python's `get_sol`).
#[allow(clippy::too_many_arguments)]
fn query_context_attention_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    prefix: u32,
    n: u32,
    n_kv: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
) -> Result<(f64, Source), AicError> {
    let silicon = || -> Result<f64, AicError> {
        let full_s = s + prefix;
        let latency = db.attention.query_context(
            b, full_s, n, n_kv, head_size, window_size, kv_quant, fmha_quant,
        )?;
        Ok(latency * prefix_correction(full_s, prefix))
    };
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            context_attention_empirical(
                db, b, s, prefix, n, n_kv, head_size, window_size, kv_quant, fmha_quant,
            )?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match silicon() {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => Ok((
                context_attention_empirical(
                    db, b, s, prefix, n, n_kv, head_size, window_size, kv_quant, fmha_quant,
                )?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok((silicon()?, Source::Silicon)),
    }
}

/// `SOL(query)/util` for context (prefill) attention. Mirrors Python
/// `_query_context_attention_table::get_empirical`: the query SOL always uses
/// the real window/prefix; the UTIL carrier is borrowed by slice — exact
/// window first, then window=0 (full attention), each trying the own
/// head_size grid and then the cross-head_size (XSHAPE) ladder before moving
/// to the next window. No basis at all surfaces the typed empirical miss.
#[allow(clippy::too_many_arguments)]
fn context_attention_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    prefix: u32,
    n: u32,
    n_kv: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
    fmha_quant: FmhaQuantMode,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let sol_time = context_attention_sol_with_prefix_ms(
        spec,
        b as f64,
        s as f64,
        prefix as f64,
        n as f64,
        n_kv as f64,
        head_size,
        window_size,
        kv_quant,
        fmha_quant,
    );
    let n_kv_lookup = if n == n_kv { 0 } else { n_kv };
    let query = [n as f64, (s + prefix) as f64, b as f64];

    let windows: Vec<u32> = if window_size > 0 { vec![window_size, 0] } else { vec![window_size] };
    for &slice_window in &windows {
        // Own-slice grid: samples are full attention (prefix=0), so the
        // per-sample SOL is the prefix=0 specialization at the slice's own
        // head_size/window (c = [n, full_s, b]).
        let key = format!(
            "ctx_attn:{}:{}:{}:{}:{}",
            fmha_quant.name(),
            kv_quant.name(),
            n_kv_lookup,
            head_size,
            slice_window
        );
        let grid = db.util_grids.get_or_try_build(&key, || {
            match db.attention.context_points(fmha_quant, kv_quant, n_kv_lookup, head_size, slice_window) {
                Ok(points) => {
                    let sol = |c: &[f64]| {
                        context_attention_sol_ms(
                            spec, n_kv_lookup, head_size, slice_window, kv_quant, fmha_quant, c[0],
                            c[1], c[2],
                        )
                    };
                    Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol))))
                }
                // Typed coverage miss -> no grid (fall through the ladder);
                // schema/load errors propagate.
                Err(err) if err.is_missing_perf_data() => Ok(None),
                Err(err) => Err(err),
            }
        })?;
        if grid.as_deref().is_some_and(|g| !g.is_empty()) {
            let (latency, _) = util_empirical::estimate(sol_time, &query, grid.as_deref(), 1.0)?;
            // Own-shape util fired (Python attention.py:406, default tier).
            db.note_provenance(util_empirical::ProvenanceTier::Empirical);
            return Ok(latency);
        }
        // Cross-head_size transfer (XSHAPE): this head_size has no data, but
        // num_heads is already an in-grid axis, so only head_size differs.
        // Borrow the nearest collected head_size's util and rescale by the
        // prefill head_size-util ratio (SOL still uses the query's own
        // head_size).
        if db.transfer_policy.contains(TransferKind::XShape) {
            if let Some((ref_grid, ref_hs)) =
                ctx_headsize_ref_grid(db, fmha_quant, kv_quant, n_kv_lookup, head_size, slice_window)?
            {
                let scale = attn_prefill_hs_ratio(&db.backend, head_size)
                    / attn_prefill_hs_ratio(&db.backend, ref_hs);
                let (latency, _) =
                    util_empirical::estimate(sol_time, &query, Some(&ref_grid), scale)?;
                // Cross-head_size borrow (Python attention.py:424 "xshape").
                db.note_provenance(util_empirical::ProvenanceTier::XShape);
                return Ok(latency);
            }
        }
    }

    // No own-window, full-attention, or cross-head basis -> typed miss.
    util_empirical::estimate(sol_time, &query, None, 1.0).map(|(latency, _)| latency)
}

/// Reference util grid for context attention borrowed from the nearest
/// collected head_size (same fmha/kv/n_kv/window). Mirrors Python
/// `_ctx_headsize_ref_grid`: built with the REFERENCE slice's own SOL
/// (reference head_size in the formula). `Ok(None)` when nothing usable is
/// collected.
fn ctx_headsize_ref_grid(
    db: &PerfDatabase,
    fmha_quant: FmhaQuantMode,
    kv_quant: KvCacheQuantMode,
    n_kv_lookup: u32,
    target_hs: u32,
    window_size: u32,
) -> Result<Option<(std::sync::Arc<UtilGrid>, u32)>, AicError> {
    let head_sizes = match db.attention.context_head_sizes(fmha_quant, kv_quant, n_kv_lookup) {
        Ok(sizes) => sizes,
        Err(err) if err.is_missing_perf_data() => return Ok(None),
        Err(err) => return Err(err),
    };
    let Some(ref_hs) = ref_head_size(&head_sizes, target_hs) else {
        return Ok(None);
    };
    let spec = &db.system_spec;
    // Reference identity (ref_hs) + provenance in the key, so a policy that
    // later reuses the same slice as own-shape cannot alias this grid.
    let key = format!(
        "ctx_attn_xhs:{}:{}:{}:{}:{}:xshape",
        fmha_quant.name(),
        kv_quant.name(),
        n_kv_lookup,
        ref_hs,
        window_size
    );
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.attention.context_points(fmha_quant, kv_quant, n_kv_lookup, ref_hs, window_size) {
            Ok(points) => {
                let sol = |c: &[f64]| {
                    context_attention_sol_ms(
                        spec, n_kv_lookup, ref_hs, window_size, kv_quant, fmha_quant, c[0], c[1],
                        c[2],
                    )
                };
                let mut grid = UtilGrid::new(util_empirical::build_samples(points, sol));
                grid.reference_provenance = Some("xshape");
                Ok(Some(grid))
            }
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    Ok(grid.filter(|g| !g.is_empty()).map(|g| (g, ref_hs)))
}

/// Generation attention latency for `(b, s, shape)` under the database's
/// query mode.
#[allow(clippy::too_many_arguments)]
fn query_generation_attention_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    n: u32,
    n_kv: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
) -> Result<(f64, Source), AicError> {
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            generation_attention_empirical(db, b, s, n, n_kv, head_size, window_size, kv_quant)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => {
            match db.attention.query_generation(b, s, n, n_kv, head_size, window_size, kv_quant) {
                Ok(latency) => Ok((latency, Source::Silicon)),
                Err(err) if err.is_missing_perf_data() => Ok((
                    generation_attention_empirical(
                        db, b, s, n, n_kv, head_size, window_size, kv_quant,
                    )?,
                    Source::Empirical,
                )),
                Err(err) => Err(err),
            }
        }
        _ => Ok((
            db.attention.query_generation(b, s, n, n_kv, head_size, window_size, kv_quant)?,
            Source::Silicon,
        )),
    }
}

/// `SOL(query)/util` for generation (decode) attention. Mirrors Python
/// `_query_generation_attention_table::get_empirical`: the query SOL uses the
/// real window (`kv_len` capped at `w`); the UTIL carrier is borrowed by
/// slice — exact window then window=0 — calibrated from the RAW (SOL-clamped)
/// generation table. Decode util is ~head_size-independent, so the XSHAPE
/// transfer keeps `util_scale = 1.0`.
#[allow(clippy::too_many_arguments)]
fn generation_attention_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    n: u32,
    n_kv: u32,
    head_size: u32,
    window_size: u32,
    kv_quant: KvCacheQuantMode,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let n_kv_lookup = if n_kv == n { 0 } else { n_kv };
    let sol_time = generation_attention_sol_ms(
        spec, n_kv_lookup, head_size, window_size, kv_quant, n as f64, b as f64, s as f64,
    );
    let query = [n as f64, b as f64, s as f64];

    let windows: Vec<u32> = if window_size > 0 { vec![window_size, 0] } else { vec![window_size] };
    for &slice_window in &windows {
        let key = format!(
            "gen_attn:{}:{}:{}:{}",
            kv_quant.name(),
            n_kv_lookup,
            head_size,
            slice_window
        );
        let grid = db.util_grids.get_or_try_build(&key, || {
            match db.attention.generation_points(kv_quant, n_kv_lookup, head_size, slice_window) {
                Ok(points) => {
                    let sol = |c: &[f64]| {
                        generation_attention_sol_ms(
                            spec, n_kv_lookup, head_size, slice_window, kv_quant, c[0], c[1], c[2],
                        )
                    };
                    Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol))))
                }
                Err(err) if err.is_missing_perf_data() => Ok(None),
                Err(err) => Err(err),
            }
        })?;
        if grid.as_deref().is_some_and(|g| !g.is_empty()) {
            let (latency, _) = util_empirical::estimate(sol_time, &query, grid.as_deref(), 1.0)?;
            // Own-shape util fired (Python attention.py:791, default tier).
            db.note_provenance(util_empirical::ProvenanceTier::Empirical);
            return Ok(latency);
        }
        if db.transfer_policy.contains(TransferKind::XShape) {
            if let Some((ref_grid, _ref_hs)) =
                gen_headsize_ref_grid(db, kv_quant, n_kv_lookup, head_size, slice_window)?
            {
                let (latency, _) =
                    util_empirical::estimate(sol_time, &query, Some(&ref_grid), 1.0)?;
                // Cross-head_size borrow (Python attention.py:802 "xshape").
                db.note_provenance(util_empirical::ProvenanceTier::XShape);
                return Ok(latency);
            }
        }
    }

    util_empirical::estimate(sol_time, &query, None, 1.0).map(|(latency, _)| latency)
}

/// Reference util grid for generation attention borrowed from the nearest
/// collected head_size (same kv/n_kv/window). Mirrors Python
/// `_gen_headsize_ref_grid` (reference head_size in the sample SOL).
fn gen_headsize_ref_grid(
    db: &PerfDatabase,
    kv_quant: KvCacheQuantMode,
    n_kv_lookup: u32,
    target_hs: u32,
    window_size: u32,
) -> Result<Option<(std::sync::Arc<UtilGrid>, u32)>, AicError> {
    let head_sizes = match db.attention.generation_head_sizes(kv_quant, n_kv_lookup) {
        Ok(sizes) => sizes,
        Err(err) if err.is_missing_perf_data() => return Ok(None),
        Err(err) => return Err(err),
    };
    let Some(ref_hs) = ref_head_size(&head_sizes, target_hs) else {
        return Ok(None);
    };
    let spec = &db.system_spec;
    let key = format!(
        "gen_attn_xhs:{}:{}:{}:{}:xshape",
        kv_quant.name(),
        n_kv_lookup,
        ref_hs,
        window_size
    );
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.attention.generation_points(kv_quant, n_kv_lookup, ref_hs, window_size) {
            Ok(points) => {
                let sol = |c: &[f64]| {
                    generation_attention_sol_ms(
                        spec, n_kv_lookup, ref_hs, window_size, kv_quant, c[0], c[1], c[2],
                    )
                };
                let mut grid = UtilGrid::new(util_empirical::build_samples(points, sol));
                grid.reference_provenance = Some("xshape");
                Ok(Some(grid))
            }
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    Ok(grid.filter(|g| !g.is_empty()).map(|g| (g, ref_hs)))
}

/// Encoder attention latency for `(b, s, shape)` under the database's query
/// mode.
fn query_encoder_attention_table(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    n: u32,
    head_size: u32,
    fmha_quant: FmhaQuantMode,
) -> Result<(f64, Source), AicError> {
    match db.database_mode {
        DatabaseMode::Empirical => Ok((
            encoder_attention_empirical(db, b, s, n, head_size, fmha_quant)?,
            Source::Empirical,
        )),
        DatabaseMode::Hybrid => match db.attention.query_encoder(b, s, n, head_size, fmha_quant) {
            Ok(latency) => Ok((latency, Source::Silicon)),
            Err(err) if err.is_missing_perf_data() => Ok((
                encoder_attention_empirical(db, b, s, n, head_size, fmha_quant)?,
                Source::Empirical,
            )),
            Err(err) => Err(err),
        },
        _ => Ok((
            db.attention.query_encoder(b, s, n, head_size, fmha_quant)?,
            Source::Silicon,
        )),
    }
}

/// `SOL(query)/util` over the encoder slice's own `(n, s, b)` grid. Mirrors
/// Python `_query_encoder_attention_table::get_empirical`: own-shape only, no
/// window ladder, no transfer.
fn encoder_attention_empirical(
    db: &PerfDatabase,
    b: u32,
    s: u32,
    n: u32,
    head_size: u32,
    fmha_quant: FmhaQuantMode,
) -> Result<f64, AicError> {
    let spec = &db.system_spec;
    let sol = |c: &[f64]| encoder_attention_sol_ms(spec, head_size, fmha_quant, c[0], c[1], c[2]);
    let query = [n as f64, s as f64, b as f64];
    let key = format!("encoder_attn:{}:{}", fmha_quant.name(), head_size);
    let grid = db.util_grids.get_or_try_build(&key, || {
        match db.attention.encoder_points(fmha_quant, head_size) {
            Ok(points) => Ok(Some(UtilGrid::new(util_empirical::build_samples(points, sol)))),
            Err(err) if err.is_missing_perf_data() => Ok(None),
            Err(err) => Err(err),
        }
    })?;
    let (latency, _) = util_empirical::estimate(sol(&query), &query, grid.as_deref(), 1.0)?;
    // Own-shape util fired (Python attention.py:1044, default tier).
    db.note_provenance(util_empirical::ProvenanceTier::Empirical);
    Ok(latency)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_db() -> PerfDatabase {
        let systems_root = PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator_core/systems");
        PerfDatabase::load(&systems_root, "b200_sxm", "vllm", "0.19.0").expect("db must load")
    }

    #[test]
    fn context_attention_smoke() {
        let db = b200_vllm_db();
        let op = ContextAttentionOp::new(
            "ctx",
            64,
            1,
            128,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Bfloat16,
        );
        // prefix=0 means prefix_correction=1.0, table latency is consumed full.
        let result = op
            .query(&db, 8, 16384, 0, 1.0)
            .expect("context attention query must succeed");
        // Table value at exact hit is 19.82; mem_op extras add ~0-1ms on top.
        assert!(result.latency_ms > 19.0 && result.latency_ms < 30.0);
        assert_eq!(result.source, Source::Silicon);
    }

    #[test]
    fn context_attention_prefix_correction_shrinks_latency() {
        let db = b200_vllm_db();
        let op = ContextAttentionOp::new(
            "ctx",
            64,
            1,
            128,
            KvCacheQuantMode::Fp8,
            FmhaQuantMode::Bfloat16,
        );
        // prefix=8192 -> prefix_correction = (16384^2 - 8192^2)/16384^2 = 0.75
        let with_prefix = op
            .query(&db, 8, 8192, 8192, 1.0)
            .expect("query must succeed")
            .latency_ms;
        let no_prefix = op.query(&db, 8, 16384, 0, 1.0).expect("query must succeed").latency_ms;
        assert!(
            with_prefix < no_prefix,
            "prefix correction must shrink latency: {with_prefix} vs {no_prefix}"
        );
    }

    #[test]
    fn generation_attention_smoke() {
        let db = b200_vllm_db();
        let op = GenerationAttentionOp::new("gen", 64, 4, 128, KvCacheQuantMode::Fp8);
        // b=32 isl+step=2 n=64 n_kv=4. The query averages 5 interp samples
        // over s ∈ [1, 2] (s_samples = [1,1,1,1,2]) on the densified grid,
        // matching Python's `_query_generation_attention_table`. Verified
        // against `PerfDatabase.query_generation_attention` (= 0.0086442669).
        let result = op
            .query(&db, 32, 2, 1.0)
            .expect("gen attention query must succeed");
        assert!(
            // Python v2 engine value (raw table, no densified lattice); the
            // pre-perf_interp expectation was 0.008644266923268636.
            (result.latency_ms - 0.008451361751014535).abs() < 1e-9,
            "expected 5-sample-averaged gen latency, got {}",
            result.latency_ms
        );
    }

    #[test]
    fn mem_op_latency_uses_empirical_formula() {
        let db = b200_vllm_db();
        // Formula check against the LIVE spec values (hardcoding mem_bw went
        // stale when PR #1246 corrected b200 to 7.7 TB/s):
        // latency = (bytes / (mem_bw * scaling) + constant) * 1000.
        let spec = &db.system_spec;
        let latency = mem_op_latency_ms(spec, 1_000_000.0);
        let expected = (1_000_000.0_f64
            / (spec.gpu.mem_bw * spec.gpu.mem_bw_empirical_scaling_factor)
            + spec.gpu.mem_empirical_constant_latency)
            * 1000.0;
        assert!((latency - expected).abs() < 1e-12);
    }

    /// Oracle values generated from the Python reference on the same data:
    /// `ContextAttention._query_context_attention_table(db, b, s, prefix, n,
    /// n_kv, kv, fmha, database_mode=EMPIRICAL, window_size=w, head_size=hs)`
    /// on b200_sxm/vllm/0.19.0. Regenerate if the shipped attention tables or
    /// the util-empirical math changes.
    #[test]
    fn context_attention_empirical_matches_python_oracles() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        // (b, s, prefix, n, n_kv, hs, w, kv, expected)
        let cases: &[(u32, u32, u32, u32, u32, u32, u32, KvCacheQuantMode, f64)] = &[
            // own-shape off-grid query on the collected hs=128 slice
            (7, 3000, 0, 64, 1, 128, 0, KvCacheQuantMode::Fp8, 0.771381792089557),
            // exact collected hit: util reconstruction returns the measured value
            (8, 16384, 0, 64, 1, 128, 0, KvCacheQuantMode::Fp8, 19.820667266845703),
            // prefix baked into the query SOL (util from the full-seq point)
            (4, 8192, 8192, 64, 1, 128, 0, KvCacheQuantMode::Fp8, 7.964372158050536),
            // head_size=192 XSHAPE transfer (collected head sizes are {128, 256};
            // ref=128, util_scale = ratio(vllm,192)/ratio(vllm,128) = 1.27)
            (4, 4096, 0, 48, 8, 192, 0, KvCacheQuantMode::Fp8, 0.7588535312592514),
            // collected windowed slice (bfloat16 kv, w=8192) as its own carrier
            (2, 10000, 0, 32, 1, 128, 8192, KvCacheQuantMode::Bfloat16, 6.254832211751053),
            // uncollected window (w=4096) -> window=0 slice as the util carrier
            (2, 10000, 0, 32, 1, 128, 4096, KvCacheQuantMode::Bfloat16, 1.0547865593548398),
        ];
        for &(b, s, prefix, n, n_kv, hs, w, kv, expected) in cases {
            let (latency, source) = query_context_attention_table(
                &db, b, s, prefix, n, n_kv, hs, w, kv, FmhaQuantMode::Bfloat16,
            )
            .expect("empirical query");
            assert!(
                (latency - expected).abs() < 1e-9,
                "(b={b}, s={s}, p={prefix}, n={n}, n_kv={n_kv}, hs={hs}, w={w}): \
                 expected {expected}, got {latency}"
            );
            assert_eq!(source, Source::Empirical);
        }
    }

    /// Oracle values generated from the Python reference:
    /// `GenerationAttention._query_generation_attention_table(db, b, s, n,
    /// n_kv, kv, database_mode=EMPIRICAL, window_size=w, head_size=hs)` on
    /// b200_sxm/vllm/0.19.0.
    #[test]
    fn generation_attention_empirical_matches_python_oracles() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        // (b, s, n, n_kv, hs, w, kv, expected)
        let cases: &[(u32, u32, u32, u32, u32, u32, KvCacheQuantMode, f64)] = &[
            // own-shape off-grid query on the collected hs=128 slice
            (48, 7777, 64, 8, 128, 0, KvCacheQuantMode::Fp8, 0.1302149492334821),
            // exact collected hit (isl=1 + step=1 -> stored s=2), calibrated
            // from the RAW (SOL-clamped) table -- NOT the 5-sample silicon avg
            (32, 2, 64, 4, 128, 0, KvCacheQuantMode::Fp8, 0.008661333471536636),
            // head_size=192 XSHAPE transfer (decode util_scale stays 1.0)
            (16, 4096, 48, 8, 192, 0, KvCacheQuantMode::Fp8, 0.03992800042033196),
            // collected windowed slice (bfloat16 kv, w=8192) as its own carrier
            (8, 12000, 32, 1, 128, 8192, KvCacheQuantMode::Bfloat16, 0.07096281754412269),
            // uncollected window (w=2048) -> window=0 slice as the util carrier
            (8, 12000, 32, 1, 128, 2048, KvCacheQuantMode::Bfloat16, 0.0023706380832401778),
        ];
        for &(b, s, n, n_kv, hs, w, kv, expected) in cases {
            let (latency, source) =
                query_generation_attention_table(&db, b, s, n, n_kv, hs, w, kv)
                    .expect("empirical query");
            assert!(
                (latency - expected).abs() < 1e-9,
                "(b={b}, s={s}, n={n}, n_kv={n_kv}, hs={hs}, w={w}): \
                 expected {expected}, got {latency}"
            );
            assert_eq!(source, Source::Empirical);
        }
    }

    /// Oracle values generated from the Python reference:
    /// `EncoderAttention._query_encoder_attention_table(db, 3, 900, 16, 64,
    /// bfloat16, database_mode=...)` on b200_sxm/vllm/0.19.0. EMPIRICAL
    /// estimates from the util grid; HYBRID resolves on silicon (the slice is
    /// collected) and must NOT detour through the empirical layer.
    #[test]
    fn encoder_attention_empirical_and_hybrid_match_python_oracles() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        let (latency, source) =
            query_encoder_attention_table(&db, 3, 900, 16, 64, FmhaQuantMode::Bfloat16)
                .expect("empirical query");
        assert!((latency - 0.03625488888618745).abs() < 1e-9, "got {latency}");
        assert_eq!(source, Source::Empirical);

        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let (latency, source) =
            query_encoder_attention_table(&db, 3, 900, 16, 64, FmhaQuantMode::Bfloat16)
                .expect("hybrid query");
        assert!((latency - 0.038151752523614205).abs() < 1e-9, "got {latency}");
        assert_eq!(source, Source::Silicon);
    }

    /// HYBRID: an uncollected head_size (192) misses silicon and falls back
    /// to the XSHAPE empirical estimate (same value as EMPIRICAL mode), while
    /// a collected slice keeps resolving on silicon. Oracle from Python
    /// `_query_context_attention_table(..., database_mode=HYBRID)`.
    #[test]
    fn context_attention_hybrid_dispatch_matches_python() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Hybrid;
        let (latency, source) = query_context_attention_table(
            &db, 4, 4096, 0, 48, 8, 192, 0, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
        )
        .expect("hybrid query");
        assert!((latency - 0.7588535312592514).abs() < 1e-9, "got {latency}");
        assert_eq!(source, Source::Empirical);

        // Collected slice: silicon exact hit, untouched by the fallback.
        let (latency, source) = query_context_attention_table(
            &db, 8, 16384, 0, 64, 1, 128, 0, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
        )
        .expect("hybrid query");
        assert!((latency - 19.820667266845703).abs() < 1e-9, "got {latency}");
        assert_eq!(source, Source::Silicon);
    }

    /// With XSHAPE disabled and no own-slice data (head_size=192), the
    /// estimate must surface the terminal EmpiricalNotImplemented miss —
    /// verified against Python `db.set_transfer_policy("off")` raising
    /// `EmpiricalNotImplementedError` for both ctx and gen.
    #[test]
    fn attention_xshape_disabled_raises_empirical_not_implemented() {
        let mut db = b200_vllm_db();
        db.database_mode = crate::common::enums::DatabaseMode::Empirical;
        db.transfer_policy = crate::common::enums::TransferPolicy::OFF;
        let ctx = query_context_attention_table(
            &db, 4, 4096, 0, 48, 8, 192, 0, KvCacheQuantMode::Fp8, FmhaQuantMode::Bfloat16,
        );
        assert!(matches!(ctx, Err(AicError::EmpiricalNotImplemented(_))), "got {ctx:?}");
        let gen = query_generation_attention_table(&db, 16, 4096, 48, 8, 192, 0, KvCacheQuantMode::Fp8);
        assert!(matches!(gen, Err(AicError::EmpiricalNotImplemented(_))), "got {gen:?}");
    }
}
