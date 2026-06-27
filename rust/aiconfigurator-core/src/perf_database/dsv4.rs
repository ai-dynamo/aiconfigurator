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
//! `load_generation_dsv4_kind_module_data`. The module slice is keyed by the
//! physical `(profile, tp_size, local_heads)` tuple. This is deliberately more
//! specific than a bare local-head count: Flash TP1 and Pro TP2 both have 64
//! local heads but benchmark different kernels and must not overwrite or borrow
//! one another's rows.
//!
//! All four primary CSVs share the DSA module column layout. Data is
//! collected only on TRT-LLM / SGLang today; loaders surface a clean error for
//! backends without DSV4 data.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::interpolation::{interp_1d, nearest_neighbors};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::perf_database::parquet_loader::PerfReader;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttnKind {
    Csa,
    Hca,
}

// module-head key -> step -> isl -> batch -> latency
type ByBatch = BTreeMap<u32, f64>;
type ByIsl = BTreeMap<u32, ByBatch>;
type ByStep = BTreeMap<u32, ByIsl>;
type ByModuleHead = BTreeMap<ModuleHeadKey, ByStep>;

pub struct Dsv4Table {
    data_root: PathBuf,
    csa_context: OnceLock<Result<ModuleGrids, AicError>>,
    hca_context: OnceLock<Result<ModuleGrids, AicError>>,
    csa_generation: OnceLock<Result<ModuleGrids, AicError>>,
    hca_generation: OnceLock<Result<ModuleGrids, AicError>>,
}

struct ModuleGrids {
    by_keys: BTreeMap<ModuleKey, ByModuleHead>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleHeadKey {
    profile: String,
    tp_size: u32,
    local_heads: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleKey {
    architecture: String,
    fmha_quant: String,
    kv_quant: String,
    gemm_quant: String,
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

    /// Context-DSV4 latency at `lookup_s = isl` (the new-token count). Mirrors
    /// Python's context base lookup over `(tp_size, isl, b)` on the
    /// `native_heads` slice. The context CSVs collected to date carry a single
    /// `step=0` anchor, so there is no prefix axis to resolve here (the operator
    /// already supplies the new-token count as `isl`).
    ///
    /// `local_heads` is the model's rank-LOCAL head count (`native // tp`); it is
    /// resolved against the CSV head keys via [`resolve_head_key`] (Python
    /// `_dsv4_resolve_head_key`). The lookup mirrors Python's context path
    /// `_query_context_attn_table -> _dsv4_lookup_prefix_resolved ->
    /// _dsv4_robust_3d_lookup(..., batch_axis="z")`: exact `(isl, b)` hit, else
    /// the sampled-batch-scaling fallback (batch is the inner axis). The single
    /// `step=0` anchor means the cubic 3-D path is degenerate, so Python always
    /// falls through to exact-or-batch-scaling here.
    #[allow(clippy::too_many_arguments)]
    pub fn query_context(
        &self,
        attn_kind: AttnKind,
        b: u32,
        isl: u32,
        local_heads: u32,
        native_heads: u32,
        tp_size: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_context()?,
            AttnKind::Hca => self.load_hca_context()?,
        };
        let by_step = select_resolved(
            grids,
            architecture,
            fmha_quant,
            kv_quant,
            gemm_quant,
            local_heads,
            native_heads,
            tp_size,
        )?;
        // Single step=0 anchor: fold the step axis to the `[isl][batch]` slice
        // (last anchor wins, matching Python's prefix-resolved single anchor).
        let slice = by_step.values().next_back().ok_or_else(|| {
            AicError::PerfDatabase("DSV4 context slice has no step anchor".into())
        })?;
        // batch_axis="z": batch is the inner key, isl the outer.
        robust_lookup_batch_inner(slice, isl, b)
    }

    /// Generation-DSV4 latency. `sequence_tokens = isl + step` (absolute KV
    /// length). Mirrors Python's generation path
    /// `_query_generation_attn_table -> _dsv4_robust_3d_lookup(dict[head], head,
    /// b, s_total, batch_axis="y")`: exact `(b, s_total)` hit, else
    /// sampled-batch-scaling (batch is the MIDDLE axis, s_total the inner).
    ///
    /// `local_heads` is resolved against the CSV head keys via
    /// [`resolve_head_key`]. The DSV4 generation table is RAGGED (e.g.
    /// `s_total=385` is measured only at `batch=2`); the batch-scaling fallback
    /// — take the largest measured `bp <= b`, interpolate along `s_total`, then
    /// scale by `b/bp` — is what Python does and what a plain grid interpolation
    /// would get wrong (it would smoothly interpolate the batch axis instead).
    #[allow(clippy::too_many_arguments)]
    pub fn query_generation(
        &self,
        attn_kind: AttnKind,
        b: u32,
        sequence_tokens: u32,
        local_heads: u32,
        native_heads: u32,
        tp_size: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_generation()?,
            AttnKind::Hca => self.load_hca_generation()?,
        };
        let by_step = select_resolved(
            grids,
            architecture,
            fmha_quant,
            kv_quant,
            gemm_quant,
            local_heads,
            native_heads,
            tp_size,
        )?;
        // Build the `[batch][s_total]` slice where s_total = isl + step. The
        // generation CSVs use isl=1, so s_total = 1 + step. If multiple
        // (step, isl) pairs map to the same s_total the last write wins, which
        // mirrors Python's flat `{b: {s_total: leaf}}` dict overwrite.
        let mut slice: BTreeMap<u32, BTreeMap<u32, f64>> = BTreeMap::new();
        for (&step, by_isl) in by_step {
            for (&isl_v, by_batch) in by_isl {
                let s_total = isl_v + step;
                for (&bb, &lat) in by_batch {
                    slice.entry(bb).or_default().insert(s_total, lat);
                }
            }
        }
        // batch_axis="y": batch is the outer key of `slice`, s_total the inner.
        robust_lookup_batch_outer(&slice, b, sequence_tokens)
    }

    fn load_csa_context(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.csa_context.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_csa_context_module_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_context(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.hca_context.get_or_init(|| {
            load_module_parquet(&self.data_root.join("dsv4_hca_context_module_perf.parquet"))
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_csa_generation(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.csa_generation.get_or_init(|| {
            load_module_parquet(
                &self
                    .data_root
                    .join("dsv4_csa_generation_module_perf.parquet"),
            )
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_generation(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.hca_generation.get_or_init(|| {
            load_module_parquet(
                &self
                    .data_root
                    .join("dsv4_hca_generation_module_perf.parquet"),
            )
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Resolve the `(quant, architecture, profile, tp, local-head)` key.
fn select_resolved<'a>(
    grids: &'a ModuleGrids,
    architecture: &str,
    fmha: FmhaQuantMode,
    kv: KvCacheQuantMode,
    gemm: GemmQuantMode,
    local_heads: u32,
    native_heads: u32,
    tp_size: u32,
) -> Result<&'a ByStep, AicError> {
    let key = ModuleKey {
        architecture: architecture.to_string(),
        fmha_quant: fmha.name().to_string(),
        kv_quant: kv.name().to_string(),
        gemm_quant: gemm.name().to_string(),
    };
    let by_module_head = grids
        .by_keys
        .get(&key)
        .ok_or_else(|| AicError::PerfDatabase(format!("DSV4 module data missing for {key:?}")))?;
    let profile = profile_from_native_heads(native_heads);
    let requested = ModuleHeadKey {
        profile: profile.to_string(),
        tp_size,
        local_heads,
    };
    if let Some(by_step) = by_module_head.get(&requested) {
        return Ok(by_step);
    }

    // Match Python's one narrow universal-sweep fallback. It may borrow a
    // different local-head bucket only within the same profile and TP, and
    // only when that physical slice has exactly one candidate.
    let mut candidates = by_module_head
        .iter()
        .filter(|(head, _)| head.profile == profile && head.tp_size == tp_size);
    let candidate = candidates.next();
    if let Some((_, by_step)) = candidate {
        if candidates.next().is_none() {
            return Ok(by_step);
        }
    }

    Err(AicError::PerfDatabase(format!(
        "DSV4 module data missing for {requested:?}, {key:?} (loaded module heads: {:?})",
        by_module_head.keys().collect::<Vec<_>>()
    )))
}

fn profile_from_native_heads(native_heads: u32) -> &'static str {
    match native_heads {
        64 => "flash",
        128 => "pro",
        _ => "unknown",
    }
}

fn profile_from_model(model: &str) -> &'static str {
    let model = model.to_ascii_lowercase();
    if model.contains("deepseek-v4-flash") {
        "flash"
    } else if model.contains("deepseek-v4-pro") {
        "pro"
    } else {
        "unknown"
    }
}

/// DSV4 robust lookup with batch as the INNER axis (Python `batch_axis="z"`).
/// `slice` is `[outer (isl)][inner (batch)]`. Used by the context path where the
/// outer axis is the new-token count and the inner axis is the batch.
///
/// Exact `(outer, b)` hit, else the sampled-batch-scaling fallback: take the
/// largest measured batch `bp <= b` (across all outer rows), interpolate along
/// the outer axis at that batch (interpolation-only first, then extrapolation),
/// and scale the result by `b / bp`.
fn robust_lookup_batch_inner(slice: &ByIsl, outer: u32, b: u32) -> Result<f64, AicError> {
    // Step 1: exact (outer, b).
    if let Some(&lat) = slice.get(&outer).and_then(|by_b| by_b.get(&b)) {
        return Ok(lat);
    }
    // Step 2: sampled-batch scaling. Candidate batches <= b across all rows.
    let mut batch_points: Vec<u32> = slice
        .values()
        .flat_map(|by_b| by_b.keys().copied())
        .filter(|&bp| bp <= b)
        .collect();
    batch_points.sort_unstable();
    batch_points.dedup();
    for allow_extrapolate in [false, true] {
        for &bp in batch_points.iter().rev() {
            if let Some(leaf) = interp_along_outer_at_batch(slice, outer, bp, allow_extrapolate) {
                let scaled = leaf * (b as f64) / (bp as f64);
                if scaled.is_finite() {
                    return Ok(scaled);
                }
            }
        }
    }
    Err(AicError::PerfDatabase(format!(
        "DSV4 robust lookup (batch inner) failed (outer={outer}, b={b})"
    )))
}

/// Interpolate along the outer axis for a fixed batch `bp` within an
/// `[outer][batch]` slice. Mirrors Python `_lookup_at_batch(batch_axis="z")`.
fn interp_along_outer_at_batch(
    slice: &ByIsl,
    outer: u32,
    bp: u32,
    allow_extrapolate: bool,
) -> Option<f64> {
    // Exact (outer, bp).
    if let Some(by_b) = slice.get(&outer) {
        if let Some(&leaf) = by_b.get(&bp) {
            return Some(leaf);
        }
    }
    // Outer points that carry this batch.
    let outer_points: Vec<u32> = slice
        .iter()
        .filter(|(_, by_b)| by_b.contains_key(&bp))
        .map(|(&o, _)| o)
        .collect();
    interp_1d_over(&outer_points, outer, allow_extrapolate, |o| slice[&o][&bp])
}

/// DSV4 robust lookup with batch as the OUTER axis (Python `batch_axis="y"`).
/// `slice` is `[outer (batch)][inner (s_total)]`. Used by the generation path.
///
/// Exact `(b, s)` hit, else the sampled-batch-scaling fallback: take the largest
/// measured batch `bp <= b`, interpolate along `s_total` within that batch row
/// (interpolation-only first, then extrapolation), and scale by `b / bp`. This
/// is the branch that reproduces Python's ragged-grid behaviour (a query batch
/// with no measured `s_total` row scales up from the nearest smaller batch
/// rather than smoothly interpolating the batch axis).
fn robust_lookup_batch_outer(
    slice: &BTreeMap<u32, BTreeMap<u32, f64>>,
    b: u32,
    s: u32,
) -> Result<f64, AicError> {
    // Step 1: exact (b, s).
    if let Some(&lat) = slice.get(&b).and_then(|by_s| by_s.get(&s)) {
        return Ok(lat);
    }
    // Step 2: sampled-batch scaling over batch rows <= b, descending.
    let batch_points: Vec<u32> = slice.keys().copied().filter(|&bp| bp <= b).collect();
    for allow_extrapolate in [false, true] {
        for &bp in batch_points.iter().rev() {
            if let Some(leaf) = interp_along_seq_at_batch(slice, bp, s, allow_extrapolate) {
                let scaled = leaf * (b as f64) / (bp as f64);
                if scaled.is_finite() {
                    return Ok(scaled);
                }
            }
        }
    }
    Err(AicError::PerfDatabase(format!(
        "DSV4 robust lookup (batch outer) failed (b={b}, s={s})"
    )))
}

/// Interpolate along `s_total` for a fixed batch row `bp` within a
/// `[batch][s_total]` slice. Mirrors Python `_lookup_at_batch(batch_axis="y")`.
fn interp_along_seq_at_batch(
    slice: &BTreeMap<u32, BTreeMap<u32, f64>>,
    bp: u32,
    s: u32,
    allow_extrapolate: bool,
) -> Option<f64> {
    let by_s = slice.get(&bp)?;
    // Exact (bp, s).
    if let Some(&leaf) = by_s.get(&s) {
        return Some(leaf);
    }
    let seq_points: Vec<u32> = by_s.keys().copied().collect();
    interp_1d_over(&seq_points, s, allow_extrapolate, |sp| by_s[&sp])
}

/// Shared 1-D interpolation helper: bracket `query` within `points` and linearly
/// interpolate `value_at(point)`. With `allow_extrapolate=false`, returns `None`
/// when `query` is outside the measured range. Needs at least two points.
fn interp_1d_over(
    points: &[u32],
    query: u32,
    allow_extrapolate: bool,
    value_at: impl Fn(u32) -> f64,
) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let (lo_bound, hi_bound) = (points[0], points[points.len() - 1]);
    if !allow_extrapolate && !(lo_bound <= query && query <= hi_bound) {
        return None;
    }
    let (lo, hi) = nearest_neighbors(query, points, !allow_extrapolate).ok()?;
    Some(interp_1d(
        lo as f64,
        hi as f64,
        value_at(lo),
        value_at(hi),
        query as f64,
    ))
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
    let model_col = reader.col("model")?;
    let mla_dtype_col = reader.col("mla_dtype")?;
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let gemm_type_col = reader.col("gemm_type")?;
    let num_heads_col = reader.col("num_heads")?;
    let tp_size_col = reader.col("tp_size")?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<ModuleKey, ByModuleHead> = BTreeMap::new();
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
        let model = row.str_owned(model_col)?;
        let profile = profile_from_model(&model);
        let tp_size = row.u32(tp_size_col)?;
        let logged_heads = row.u32(num_heads_col)?;
        // Historical collectors disagree on whether `num_heads` is native or
        // rank-local. The model profile and TP give an unambiguous local value
        // for known V4 variants, matching the Python loader.
        let local_heads = match (profile, tp_size) {
            ("flash", 1..) => 64 / tp_size,
            ("pro", 1..) => 128 / tp_size,
            _ => logged_heads,
        };
        by_keys
            .entry(key)
            .or_default()
            .entry(ModuleHeadKey {
                profile: profile.to_string(),
                tp_size,
                local_heads,
            })
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

    #[test]
    fn dsv4_data_absent_errors_cleanly() {
        // DSV4 modules aren't collected for vllm/0.19.0; loader must surface
        // a clean error.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0");
        let table = Dsv4Table::new(root);
        let err = table
            .query_context(
                AttnKind::Csa,
                1,
                1024,
                128, // local_heads
                128, // native_heads
                1,   // tp_size
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV4ForCausalLM",
            )
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// Parity regression for the DeepSeek-V4-Pro b200_sxm/sglang/0.5.10 lookup.
    /// The model passes rank-local `num_heads = 128 / tp(8) = 16`; the physical
    /// key must resolve Pro/TP8 and must not borrow Flash/TP1's overlapping
    /// 64-head bucket. Oracle values come directly from the Pro/TP8 rows:
    /// (`query_{context,generation}_deepseek_v4_attention_module`):
    ///   gen CSA source b=2 s=385 = 0.1381, scaled to b=16 -> 1.1048
    ///   gen CSA b=15 s=385 = 1.03575 (same ragged b=2 anchor)
    ///   gen HCA source b=2 s=385 = 0.0848, scaled to b=16 -> 0.6784
    ///   ctx CSA b=1 isl=128 = 0.1659 ; ctx HCA b=1 isl=128 = 0.1104
    #[test]
    fn dsv4_pro_head_resolution_and_ragged_generation() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/sglang/0.5.10");
        if !root
            .join("dsv4_csa_generation_module_perf.parquet")
            .exists()
        {
            return; // git-lfs data not materialized
        }
        let table = Dsv4Table::new(root);
        let q_gen = |kind, b, s| {
            table
                .query_generation(
                    kind,
                    b,
                    s,
                    16,
                    128,
                    8,
                    KvCacheQuantMode::Fp8,
                    FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block,
                    "DeepseekV4ForCausalLM",
                )
                .unwrap()
        };
        let q_ctx = |kind, b, isl| {
            table
                .query_context(
                    kind,
                    b,
                    isl,
                    16,
                    128,
                    8,
                    KvCacheQuantMode::Fp8,
                    FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block,
                    "DeepseekV4ForCausalLM",
                )
                .unwrap()
        };
        let approx = |got: f64, want: f64| {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        };
        // Only b=2 carries s=385 in the Pro/TP8 slice; robust lookup scales it.
        approx(q_gen(AttnKind::Csa, 16, 385), 1.1048);
        approx(q_gen(AttnKind::Hca, 16, 385), 0.6784);
        // RAGGED batch-scaling: b=15 has no measured s=385 row except at b=2.
        approx(q_gen(AttnKind::Csa, 15, 385), 1.03575);
        // Context single-anchor lookups.
        approx(q_ctx(AttnKind::Csa, 1, 128), 0.1659);
        approx(q_ctx(AttnKind::Hca, 1, 128), 0.1104);
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
    fn module_head_key_separates_flash_tp1_from_pro_tp2() {
        fn by_step(latency: f64) -> ByStep {
            BTreeMap::from([(0, BTreeMap::from([(8192, BTreeMap::from([(1, latency)]))]))])
        }

        let key = ModuleKey {
            architecture: "DeepseekV4ForCausalLM".into(),
            fmha_quant: "bfloat16".into(),
            kv_quant: "fp8".into(),
            gemm_quant: "fp8_block".into(),
        };
        let heads = BTreeMap::from([
            (
                ModuleHeadKey {
                    profile: "flash".into(),
                    tp_size: 1,
                    local_heads: 64,
                },
                by_step(11.0),
            ),
            (
                ModuleHeadKey {
                    profile: "pro".into(),
                    tp_size: 2,
                    local_heads: 64,
                },
                by_step(19.0),
            ),
        ]);
        let grids = ModuleGrids {
            by_keys: BTreeMap::from([(key, heads)]),
        };

        let flash = select_resolved(
            &grids,
            "DeepseekV4ForCausalLM",
            FmhaQuantMode::Bfloat16,
            KvCacheQuantMode::Fp8,
            GemmQuantMode::Fp8Block,
            64,
            64,
            1,
        )
        .unwrap();
        let pro = select_resolved(
            &grids,
            "DeepseekV4ForCausalLM",
            FmhaQuantMode::Bfloat16,
            KvCacheQuantMode::Fp8,
            GemmQuantMode::Fp8Block,
            64,
            128,
            2,
        )
        .unwrap();

        assert_eq!(flash[&0][&8192][&1], 11.0);
        assert_eq!(pro[&0][&8192][&1], 19.0);
    }

    #[test]
    fn dsv4_context_resolves_fp8_e4m3_kv_quant() {
        // Regression for the b200_sxm/sglang/0.5.10 DSV4 context lookup: the
        // CSV stores `kv_cache_dtype=fp8_e4m3`, but the query builds the key
        // from `KvCacheQuantMode::Fp8.name()` = "fp8". Without load-side
        // normalization the lookup misses (Rust-only error vs Python success).
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../src/aiconfigurator/systems/data/b200_sxm/sglang/0.5.10");
        if !root.join("dsv4_csa_context_module_perf.parquet").exists() {
            // Data files are git-lfs tracked; skip if not materialized.
            return;
        }
        let table = Dsv4Table::new(root);
        // (head=64, isl=512, batch=8, step=0) are measured grid points in the
        // CSA context table for this entry, gemm=fp8_block.
        let latency = table
            .query_context(
                AttnKind::Csa,
                8,   // batch
                512, // isl
                64,  // local_heads (exact head key)
                64,  // native_heads (Flash)
                1,   // tp_size
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Fp8Block,
                "DeepseekV4ForCausalLM",
            )
            .expect("DSV4 context lookup must resolve fp8_e4m3 kv_cache_dtype as fp8");
        assert!(
            latency.is_finite() && latency > 0.0,
            "unexpected latency: {latency}"
        );
    }
}
