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
//! `load_generation_dsv4_kind_module_data`. Context quant slices are keyed by
//! `(fmha, kv, gemm)`, while generation slices are keyed by `(kv, gemm)` because
//! the Python generation consumer intentionally ignores FMHA dtype. Both phases
//! ignore the provenance-only `architecture` column. The module-head slice is
//! keyed by the physical `(profile, tp_size, local_heads)` tuple. This is
//! deliberately more specific than a bare local-head count: Flash TP1 and Pro
//! TP2 both have 64 local heads but benchmark different kernels and must not
//! overwrite or borrow one another's rows.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModulePhase {
    Context,
    Generation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModuleKind {
    Csa,
    Hca,
}

impl ModuleKind {
    fn expected_compress_ratio(self) -> u32 {
        match self {
            Self::Csa => 4,
            Self::Hca => 128,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Csa => "CSA",
            Self::Hca => "HCA",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleHeadKey {
    /// Legacy rows without a model use `(None, None, logged_heads)`.
    profile: Option<String>,
    tp_size: Option<u32>,
    local_heads: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ModuleKey {
    // Python's context loader indexes FMHA dtype; generation does not.
    fmha_quant: Option<String>,
    kv_quant: String,
    gemm_quant: String,
}

impl ModuleKey {
    fn context(
        fmha_quant: impl Into<String>,
        kv_quant: impl Into<String>,
        gemm_quant: impl Into<String>,
    ) -> Self {
        Self {
            fmha_quant: Some(fmha_quant.into()),
            kv_quant: kv_quant.into(),
            gemm_quant: gemm_quant.into(),
        }
    }

    fn generation(kv_quant: impl Into<String>, gemm_quant: impl Into<String>) -> Self {
        Self {
            fmha_quant: None,
            kv_quant: kv_quant.into(),
            gemm_quant: gemm_quant.into(),
        }
    }
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
    /// Python's context lookup over `(prefix, isl, b)` on the resolved module
    /// head slice. Exact prefix anchors are preferred; off-grid prefixes are
    /// linearly interpolated between anchors that can answer `(isl, b)`, with
    /// clamping outside the collected prefix range.
    ///
    /// `local_heads` is the model's rank-LOCAL head count (`native // tp`); it is
    /// resolved against the CSV head keys via [`resolve_head_key`] (Python
    /// `_dsv4_resolve_head_key`). The lookup mirrors Python's context path
    /// `_query_context_attn_table -> _dsv4_lookup_prefix_resolved ->
    /// _dsv4_robust_3d_lookup(..., batch_axis="z")`: exact `(isl, b)` hit, else
    /// the sampled-batch-scaling fallback (batch is the inner axis), independently
    /// at each usable prefix anchor.
    #[allow(clippy::too_many_arguments)]
    pub fn query_context(
        &self,
        attn_kind: AttnKind,
        b: u32,
        isl: u32,
        prefix: u32,
        local_heads: u32,
        native_heads: u32,
        tp_size: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_context()?,
            AttnKind::Hca => self.load_hca_context()?,
        };
        let by_step = select_resolved(
            grids,
            ModuleKey::context(fmha_quant.name(), kv_quant.name(), gemm_quant.name()),
            local_heads,
            native_heads,
            tp_size,
        )?;
        context_prefix_resolved(by_step, prefix, isl, b)
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
        gemm_quant: GemmQuantMode,
    ) -> Result<f64, AicError> {
        let grids = match attn_kind {
            AttnKind::Csa => self.load_csa_generation()?,
            AttnKind::Hca => self.load_hca_generation()?,
        };
        let by_step = select_resolved(
            grids,
            ModuleKey::generation(kv_quant.name(), gemm_quant.name()),
            local_heads,
            native_heads,
            tp_size,
        )?;
        // Generation rows are normalized during load to
        // `[step=0][s_total=isl+step][batch]`, resolving physical-key
        // collisions in file order. Build the query's `[batch][s_total]`
        // slice without reintroducing a BTreeMap step/isl winner.
        let by_sequence = by_step.get(&0).ok_or_else(|| {
            AicError::PerfDatabase("DSV4 generation slice has no normalized sequence rows".into())
        })?;
        let mut slice: BTreeMap<u32, BTreeMap<u32, f64>> = BTreeMap::new();
        for (&s_total, by_batch) in by_sequence {
            for (&bb, &lat) in by_batch {
                slice.entry(bb).or_default().insert(s_total, lat);
            }
        }
        // batch_axis="y": batch is the outer key of `slice`, s_total the inner.
        robust_lookup_batch_outer(&slice, b, sequence_tokens)
    }

    fn load_csa_context(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.csa_context.get_or_init(|| {
            load_module_parquet(
                &self.data_root.join("dsv4_csa_context_module_perf.parquet"),
                ModulePhase::Context,
                ModuleKind::Csa,
            )
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_hca_context(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.hca_context.get_or_init(|| {
            load_module_parquet(
                &self.data_root.join("dsv4_hca_context_module_perf.parquet"),
                ModulePhase::Context,
                ModuleKind::Hca,
            )
        });
        cell.as_ref().map_err(clone_err)
    }
    fn load_csa_generation(&self) -> Result<&ModuleGrids, AicError> {
        let cell = self.csa_generation.get_or_init(|| {
            load_module_parquet(
                &self
                    .data_root
                    .join("dsv4_csa_generation_module_perf.parquet"),
                ModulePhase::Generation,
                ModuleKind::Csa,
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
                ModulePhase::Generation,
                ModuleKind::Hca,
            )
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Resolve the phase-specific quant key plus `(profile, tp, local-head)`.
fn select_resolved<'a>(
    grids: &'a ModuleGrids,
    key: ModuleKey,
    local_heads: u32,
    native_heads: u32,
    tp_size: u32,
) -> Result<&'a ByStep, AicError> {
    let by_module_head = grids
        .by_keys
        .get(&key)
        .ok_or_else(|| AicError::PerfDatabase(format!("DSV4 module data missing for {key:?}")))?;
    let profile = profile_from_native_heads(native_heads);
    let requested = ModuleHeadKey {
        profile: Some(profile),
        tp_size: Some(tp_size),
        local_heads,
    };
    if let Some(by_step) = by_module_head.get(&requested) {
        return Ok(by_step);
    }

    // Match Python's one narrow universal-sweep fallback. It may borrow a
    // different local-head bucket only within the same profile and TP, and
    // only when that physical slice has exactly one candidate.
    if by_module_head.keys().any(|head| head.profile.is_some()) {
        let mut candidates = by_module_head.iter().filter(|(head, _)| {
            head.profile.as_deref() == requested.profile.as_deref()
                && head.tp_size == requested.tp_size
        });
        let candidate = candidates.next();
        if let Some((_, by_step)) = candidate {
            if candidates.next().is_none() {
                return Ok(by_step);
            }
        }
    } else {
        // Python's legacy local-head-only fallback: exact head, otherwise the
        // nearest head not greater than the request, else the smallest head.
        let legacy_exact = ModuleHeadKey {
            profile: None,
            tp_size: None,
            local_heads,
        };
        if let Some(by_step) = by_module_head.get(&legacy_exact) {
            return Ok(by_step);
        }
        let candidate = by_module_head
            .iter()
            .filter(|(head, _)| head.profile.is_none() && head.tp_size.is_none())
            .filter(|(head, _)| head.local_heads <= local_heads)
            .next_back()
            .or_else(|| {
                by_module_head
                    .iter()
                    .find(|(head, _)| head.profile.is_none() && head.tp_size.is_none())
            });
        if let Some((_, by_step)) = candidate {
            return Ok(by_step);
        }
    }

    Err(AicError::PerfDatabase(format!(
        "DSV4 module data missing for {requested:?}, {key:?} (loaded module heads: {:?})",
        by_module_head.keys().collect::<Vec<_>>()
    )))
}

fn profile_from_native_heads(native_heads: u32) -> String {
    match native_heads {
        64 => "flash".to_string(),
        128 => "pro".to_string(),
        _ => format!("heads={native_heads}:hidden=0:topk=0"),
    }
}

fn profile_from_model(model: &str) -> String {
    let model = model.to_ascii_lowercase();
    if model.contains("deepseek-v4-flash") {
        "flash".to_string()
    } else if model.contains("deepseek-v4-pro") {
        "pro".to_string()
    } else {
        "heads=0:hidden=0:topk=0".to_string()
    }
}

/// Resolve a context query across the collected prefix (`step`) anchors.
///
/// Mirrors Python `_dsv4_lookup_prefix_resolved`: query each anchor using the
/// robust `(isl, batch)` lookup, return an exact usable anchor when present,
/// clamp outside the usable prefix range, and linearly interpolate in between.
fn context_prefix_resolved(
    by_prefix: &ByStep,
    prefix: u32,
    isl: u32,
    b: u32,
) -> Result<f64, AicError> {
    if let Some(slice) = by_prefix.get(&prefix) {
        if let Ok(latency) = robust_lookup_batch_inner(slice, isl, b) {
            if latency.is_finite() {
                return Ok(latency);
            }
        }
    }

    let values: BTreeMap<u32, f64> = by_prefix
        .iter()
        .filter_map(|(&anchor, slice)| {
            robust_lookup_batch_inner(slice, isl, b)
                .ok()
                .filter(|latency| latency.is_finite())
                .map(|latency| (anchor, latency))
        })
        .collect();
    let (&first, &first_latency) = values.first_key_value().ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "DSV4 context prefix lookup failed (prefix={prefix}, isl={isl}, b={b})"
        ))
    })?;
    let (&last, &last_latency) = values
        .last_key_value()
        .expect("non-empty after first_key_value");

    if prefix <= first {
        return Ok(first_latency);
    }
    if prefix >= last {
        return Ok(last_latency);
    }

    let (&left, &left_latency) = values
        .range(..prefix)
        .next_back()
        .expect("prefix above first usable anchor");
    let (&right, &right_latency) = values
        .range((
            std::ops::Bound::Excluded(prefix),
            std::ops::Bound::Unbounded,
        ))
        .next()
        .expect("prefix below last usable anchor");
    Ok(interp_1d(
        left as f64,
        right as f64,
        left_latency,
        right_latency,
        prefix as f64,
    ))
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

fn module_storage_coordinates(
    phase: ModulePhase,
    step: u32,
    isl: u32,
) -> Result<(u32, u32), AicError> {
    match phase {
        ModulePhase::Context => Ok((step, isl)),
        ModulePhase::Generation => isl
            .checked_add(step)
            .map(|sequence_tokens| (0, sequence_tokens))
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "DSV4 generation sequence length overflow: isl={isl}, step={step}"
                ))
            }),
    }
}

fn insert_module_measurement(
    by_step: &mut ByStep,
    phase: ModulePhase,
    step: u32,
    isl: u32,
    batch_size: u32,
    latency: f64,
) -> Result<(), AicError> {
    let (stored_step, stored_isl) = module_storage_coordinates(phase, step, isl)?;
    // Python assigns directly into its nested dict, so an identical physical
    // row key keeps the last row in file order.
    by_step
        .entry(stored_step)
        .or_default()
        .entry(stored_isl)
        .or_default()
        .insert(batch_size, latency);
    Ok(())
}

fn module_head_identity(
    model: Option<&str>,
    tp_size: Option<u32>,
    logged_heads: u32,
) -> Result<ModuleHeadKey, AicError> {
    let Some(model) = model.filter(|value| !value.trim().is_empty()) else {
        return Ok(ModuleHeadKey {
            profile: None,
            tp_size: None,
            local_heads: logged_heads,
        });
    };

    let profile = profile_from_model(model);
    let tp_size = tp_size.unwrap_or(1);
    if tp_size == 0 {
        return Err(AicError::PerfDatabase(
            "invalid DSV4 tp_size=0; expected a positive value".into(),
        ));
    }
    let native_heads = match profile.as_str() {
        "flash" => Some(64),
        "pro" => Some(128),
        _ => None,
    };
    if native_heads.is_some_and(|heads| heads % tp_size != 0) {
        return Err(AicError::PerfDatabase(format!(
            "invalid DSV4 tp_size={tp_size} for profile={profile:?}"
        )));
    }
    Ok(ModuleHeadKey {
        profile: Some(profile),
        tp_size: Some(tp_size),
        local_heads: native_heads.map_or(logged_heads, |heads| heads / tp_size),
    })
}

fn validate_compress_ratio(kind: ModuleKind, actual: u32, path: &Path) -> Result<(), AicError> {
    let expected = kind.expected_compress_ratio();
    if actual == expected {
        return Ok(());
    }
    Err(AicError::PerfDatabase(format!(
        "invalid DSV4 {} split row at {}: compress_ratio={actual}, expected {expected}",
        kind.label(),
        path.display()
    )))
}

fn load_module_parquet(
    path: &Path,
    phase: ModulePhase,
    kind: ModuleKind,
) -> Result<ModuleGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let model_col = reader.col_optional("model");
    let mla_dtype_col = match phase {
        ModulePhase::Context => Some(reader.col("mla_dtype")?),
        ModulePhase::Generation => None,
    };
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let gemm_type_col = reader.col("gemm_type")?;
    let compress_ratio_col = reader.col("compress_ratio")?;
    let num_heads_col = reader.col("num_heads")?;
    let tp_size_col = reader.col_optional("tp_size");
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<ModuleKey, ByModuleHead> = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        validate_compress_ratio(kind, row.u32(compress_ratio_col)?, path)?;
        // CSV columns use SGLang dtype naming; query keys use enum `.name()`
        // values. Normalize the one historical alias on load. Context keeps
        // FMHA dtype as an index dimension; generation deliberately drops it,
        // exactly like Python's phase-specific loaders. Architecture is
        // provenance-only in both Python loaders and is not read here.
        let kv_quant = normalize_dsv4_dtype(&row.str_owned(kv_cache_dtype_col)?);
        let gemm_quant = row.str_owned(gemm_type_col)?;
        let key =
            match phase {
                ModulePhase::Context => ModuleKey::context(
                    normalize_dsv4_dtype(&row.str_owned(
                        mla_dtype_col.expect("context phase resolves mla_dtype column"),
                    )?),
                    kv_quant,
                    gemm_quant,
                ),
                ModulePhase::Generation => ModuleKey::generation(kv_quant, gemm_quant),
            };
        let logged_heads = row.u32(num_heads_col)?;
        let head_key = module_head_identity(
            row.str_optional(model_col)?,
            row.u32_optional(tp_size_col)?,
            logged_heads,
        )?;
        let by_step = by_keys.entry(key).or_default().entry(head_key).or_default();
        insert_module_measurement(
            by_step,
            phase,
            row.u32(step_col)?,
            row.u32(isl_col)?,
            row.u32(batch_size_col)?,
            row.f64(latency_col)?,
        )?;
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
                0,   // prefix
                128, // local_heads
                128, // native_heads
                1,   // tp_size
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
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
                    GemmQuantMode::Fp8Block,
                )
                .unwrap()
        };
        let q_ctx = |kind, b, isl| {
            table
                .query_context(
                    kind,
                    b,
                    isl,
                    0,
                    16,
                    128,
                    8,
                    KvCacheQuantMode::Fp8,
                    FmhaQuantMode::Bfloat16,
                    GemmQuantMode::Fp8Block,
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
    fn context_prefix_lookup_interpolates_and_clamps() {
        let by_prefix = BTreeMap::from([
            (0, BTreeMap::from([(128, BTreeMap::from([(1, 10.0)]))])),
            (512, BTreeMap::from([(128, BTreeMap::from([(1, 30.0)]))])),
        ]);

        assert_eq!(
            context_prefix_resolved(&by_prefix, 0, 128, 1).unwrap(),
            10.0
        );
        assert_eq!(
            context_prefix_resolved(&by_prefix, 256, 128, 1).unwrap(),
            20.0
        );
        assert_eq!(
            context_prefix_resolved(&by_prefix, 512, 128, 1).unwrap(),
            30.0
        );
        assert_eq!(
            context_prefix_resolved(&by_prefix, 1024, 128, 1).unwrap(),
            30.0
        );
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
    fn module_phase_keys_match_python_architecture_and_fmha_axes() {
        #[derive(Clone, Copy)]
        struct Row<'a> {
            architecture: &'a str,
            fmha: &'a str,
            latency: f64,
        }

        let architecture_rows = [
            Row {
                architecture: "ArchitectureA",
                fmha: "bfloat16",
                latency: 11.0,
            },
            Row {
                architecture: "ArchitectureB",
                fmha: "bfloat16",
                latency: 19.0,
            },
        ];
        assert_ne!(
            architecture_rows[0].architecture,
            architecture_rows[1].architecture
        );

        // Python context ignores architecture, so the second otherwise-equal
        // row overwrites the first. FMHA remains a real context key axis.
        let mut context_rows = BTreeMap::new();
        for row in architecture_rows {
            context_rows.insert(
                ModuleKey::context(row.fmha, "fp8", "fp8_block"),
                row.latency,
            );
        }
        assert_eq!(context_rows.len(), 1);
        assert_eq!(context_rows.values().copied().next(), Some(19.0));
        context_rows.insert(ModuleKey::context("fp8", "fp8", "fp8_block"), 23.0);
        assert_eq!(context_rows.len(), 2);

        // Python generation ignores both architecture and FMHA dtype. These
        // two physically identical consumer rows must therefore share one key
        // and retain the loader's last-row-wins behavior.
        let generation_rows = [
            architecture_rows[0],
            Row {
                architecture: "ArchitectureB",
                fmha: "fp8",
                latency: 29.0,
            },
        ];
        assert_ne!(generation_rows[0].fmha, generation_rows[1].fmha);
        let mut generation = BTreeMap::new();
        for row in generation_rows {
            generation.insert(ModuleKey::generation("fp8", "fp8_block"), row.latency);
        }
        assert_eq!(generation.len(), 1);
        assert_eq!(generation.values().copied().next(), Some(29.0));
    }

    #[test]
    fn generation_rows_normalize_total_sequence_and_last_file_row_wins() {
        let mut by_step = BTreeMap::new();
        insert_module_measurement(&mut by_step, ModulePhase::Generation, 0, 1024, 2, 11.0).unwrap();
        // Same physical `(batch, isl+step)` point via different raw metadata.
        insert_module_measurement(&mut by_step, ModulePhase::Generation, 512, 512, 2, 19.0)
            .unwrap();

        assert_eq!(by_step.len(), 1);
        assert_eq!(by_step[&0].len(), 1);
        assert_eq!(by_step[&0][&1024][&2], 19.0);
    }

    #[test]
    fn module_head_identity_matches_profile_aware_and_legacy_keys() {
        assert_eq!(
            module_head_identity(None, Some(8), 16).unwrap(),
            ModuleHeadKey {
                profile: None,
                tp_size: None,
                local_heads: 16,
            }
        );
        assert_eq!(
            module_head_identity(Some("custom/checkpoint"), Some(2), 48).unwrap(),
            ModuleHeadKey {
                profile: Some("heads=0:hidden=0:topk=0".into()),
                tp_size: Some(2),
                local_heads: 48,
            }
        );
        assert_eq!(
            module_head_identity(Some("sgl-project/DeepSeek-V4-Pro-FP8"), Some(2), 128).unwrap(),
            ModuleHeadKey {
                profile: Some("pro".into()),
                tp_size: Some(2),
                local_heads: 64,
            }
        );
    }

    #[test]
    fn split_compress_ratio_validation_rejects_misfiled_rows() {
        let path = Path::new("dsv4_csa_context_module_perf.parquet");
        validate_compress_ratio(ModuleKind::Csa, 4, path).unwrap();
        let err = validate_compress_ratio(ModuleKind::Csa, 128, path).unwrap_err();
        assert!(err.to_string().contains("compress_ratio=128, expected 4"));
        assert!(err
            .to_string()
            .contains("dsv4_csa_context_module_perf.parquet"));
        validate_compress_ratio(ModuleKind::Hca, 128, path).unwrap();
    }

    #[test]
    fn module_head_key_separates_flash_tp1_from_pro_tp2() {
        fn by_step(latency: f64) -> ByStep {
            BTreeMap::from([(0, BTreeMap::from([(8192, BTreeMap::from([(1, latency)]))]))])
        }

        let key = ModuleKey::context("bfloat16", "fp8", "fp8_block");
        let heads = BTreeMap::from([
            (
                ModuleHeadKey {
                    profile: Some("flash".into()),
                    tp_size: Some(1),
                    local_heads: 64,
                },
                by_step(11.0),
            ),
            (
                ModuleHeadKey {
                    profile: Some("pro".into()),
                    tp_size: Some(2),
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
            ModuleKey::context("bfloat16", "fp8", "fp8_block"),
            64,
            64,
            1,
        )
        .unwrap();
        let pro = select_resolved(
            &grids,
            ModuleKey::context("bfloat16", "fp8", "fp8_block"),
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
                0,   // prefix
                64,  // local_heads (exact head key)
                64,  // native_heads (Flash)
                1,   // tp_size
                KvCacheQuantMode::Fp8,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Fp8Block,
            )
            .expect("DSV4 context lookup must resolve fp8_e4m3 kv_cache_dtype as fp8");
        assert!(
            latency.is_finite() && latency > 0.0,
            "unexpected latency: {latency}"
        );
    }
}
