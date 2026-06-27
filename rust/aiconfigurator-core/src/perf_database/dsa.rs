// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (DeepSeek-V3.2 Dynamic Sparse Attention) module perf tables.
//!
//! Two parquet files: `dsa_context_module_perf.parquet` and
//! `dsa_generation_module_perf.parquet`. Both share columns: model,
//! architecture, mla_dtype, kv_cache_dtype, gemm_type, kernel_source,
//! num_heads, batch_size, isl, tp_size, step, latency.
//!
//! Context data is nested by
//! `(mla_dtype, kv_cache_dtype, gemm_type, architecture, dsa_backend)`, while
//! generation intentionally drops `mla_dtype`, matching Python and the
//! collector's physical-row key. `dsa_backend` is derived from
//! `kernel_source`: sources containing `trtllm` use `trtllm`; every other
//! source uses `flashmla_kv`. The remaining axes are
//! `num_heads → step → isl → batch_size → latency`. The `step` axis is
//! the "prefix value" — some architectures (e.g. GlmMoeDsaForCausalLM) need
//! it exposed; others use a single step=0 slice.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::interpolation::{
    interp_1d, interp_2d_1d_grid_extrapolate_inner, interp_2d_1d_grid_strict,
    interp_context_topk_piecewise, nearest_neighbors, Grid3,
};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::perf_database::parquet_loader::PerfReader;

const DEFAULT_DSA_ARCHITECTURE: &str = "DeepseekV32ForCausalLM";

pub struct DsaTable {
    data_root: PathBuf,
    context: OnceLock<Result<DsaGrids, AicError>>,
    generation: OnceLock<Result<DsaGrids, AicError>>,
    /// Lazy per-`(DsaKey, prefix)` Grid3 caches with the exact shape
    /// `pick_prefix_slice` builds, so warm queries skip the per-call walk
    /// over the full `num_heads × step × isl × batch` nested map.
    context_prefix_grids: OnceLock<Result<ContextPrefixCache, AicError>>,
    /// Lazy per-`DsaKey` Grid3 caches with the exact shape the inline
    /// generation collapse builds (axes: num_heads, seq=isl+step, batch),
    /// so warm queries skip the per-call rebuild.
    generation_grids: OnceLock<Result<GenerationGridCache, AicError>>,
}

struct ContextPrefixCache {
    by_keys: BTreeMap<DsaKey, BTreeMap<u32, Grid3<f64>>>,
}

struct GenerationGridCache {
    by_keys: BTreeMap<DsaKey, Grid3<f64>>,
}

/// (arch, fmha, kv, gemm) → num_heads → step → isl → batch → latency.
pub struct DsaGrids {
    pub by_keys: BTreeMap<DsaKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DsaPhase {
    Context,
    Generation,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DsaKey {
    pub architecture: String,
    /// Context indexes FMHA; generation deliberately does not.
    pub fmha_quant: Option<String>,
    pub kv_quant: String,
    pub gemm_quant: String,
    pub dsa_backend: String,
}

impl DsaKey {
    fn context(
        architecture: impl Into<String>,
        fmha_quant: impl Into<String>,
        kv_quant: impl Into<String>,
        gemm_quant: impl Into<String>,
        dsa_backend: impl Into<String>,
    ) -> Self {
        Self {
            architecture: architecture.into(),
            fmha_quant: Some(fmha_quant.into()),
            kv_quant: kv_quant.into(),
            gemm_quant: gemm_quant.into(),
            dsa_backend: dsa_backend.into(),
        }
    }

    fn generation(
        architecture: impl Into<String>,
        kv_quant: impl Into<String>,
        gemm_quant: impl Into<String>,
        dsa_backend: impl Into<String>,
    ) -> Self {
        Self {
            architecture: architecture.into(),
            fmha_quant: None,
            kv_quant: kv_quant.into(),
            gemm_quant: gemm_quant.into(),
            dsa_backend: dsa_backend.into(),
        }
    }

    fn with_backend(&self, dsa_backend: impl Into<String>) -> Self {
        let mut key = self.clone();
        key.dsa_backend = dsa_backend.into();
        key
    }

    fn same_shape_except_backend(&self, other: &Self) -> bool {
        self.architecture == other.architecture
            && self.fmha_quant == other.fmha_quant
            && self.kv_quant == other.kv_quant
            && self.gemm_quant == other.gemm_quant
    }
}

impl DsaTable {
    pub fn new(data_root: PathBuf) -> Self {
        Self {
            data_root,
            context: OnceLock::new(),
            generation: OnceLock::new(),
            context_prefix_grids: OnceLock::new(),
            generation_grids: OnceLock::new(),
        }
    }

    /// Context-DSA module latency for the sparse-attention block.
    ///
    /// Mirrors Python `ContextDSAModule._lookup_prefix_module_at(prefix)`:
    /// the lookup is keyed by the exact `prefix` slice and evaluated at
    /// `isl` (the new-token count), NOT `isl + prefix`. The dispatch is:
    ///
    ///   1. top-k regime-aware piecewise interpolation over the sequence
    ///      axis (`boundary = index_topk - prefix`); returns `None` when
    ///      the exact `(num_heads, b)` curve has < 2 same-regime anchors;
    ///   2. on `None`, the DSv4 robust 3-D lookup (exact -> strict 3-D ->
    ///      sampled-batch scaling) over the `(num_heads, isl, batch)` slice.
    ///
    /// The previous multiplicative `(s² - prefix²)/s²` "prefix correction"
    /// had no Python counterpart and under-counted context latency by
    /// ~75% on disagg DSA shapes; it has been removed.
    #[allow(clippy::too_many_arguments)]
    pub fn query_context(
        &self,
        b: u32,
        isl: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        dsa_backend: &str,
        prefix: u32,
        index_topk: u32,
    ) -> Result<f64, AicError> {
        let cache = self.load_context_prefix_grids()?;
        let key = DsaKey::context(
            architecture,
            fmha_quant.name(),
            kv_quant.name(),
            gemm_quant.name(),
            dsa_backend,
        );
        let by_prefix = select_dsa_backend(&cache.by_keys, &key).ok_or_else(|| {
            AicError::PerfDatabase(format!("context DSA module data missing for {key:?}"))
        })?;
        let slice = by_prefix.get(&prefix).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "context DSA module data missing for prefix={prefix}, {key:?}"
            ))
        })?;

        // Branch 1: top-k piecewise over the exact (num_heads, b) seq curve.
        // `boundary = index_topk - prefix` (saturates as None at prefix>topk
        // because subtraction underflow can't happen for u32 -> guard it).
        if index_topk >= prefix {
            let boundary = index_topk - prefix;
            if let Some(curve) = build_exact_seq_curve(slice, num_heads, b) {
                if let Some(latency) = interp_context_topk_piecewise(&curve, isl, boundary) {
                    return Ok(latency);
                }
            }
        }

        // Branch 2: DSv4 robust 3-D lookup over (num_heads, isl, batch).
        dsv4_robust_3d_lookup(slice, num_heads, isl, b)
    }

    /// Raw generation-DSA module latency. `sequence_tokens = isl + step`
    /// from the CSV; query interpolates over (num_heads, batch, seq).
    pub fn query_generation(
        &self,
        b: u32,
        sequence_tokens: u32,
        num_heads: u32,
        kv_quant: KvCacheQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
        dsa_backend: &str,
    ) -> Result<f64, AicError> {
        let cache = self.load_generation_grids()?;
        let key = DsaKey::generation(
            architecture,
            kv_quant.name(),
            gemm_quant.name(),
            dsa_backend,
        );
        let grid = select_dsa_backend(&cache.by_keys, &key)
            .ok_or_else(|| missing("generation DSA module", &self.data_root, format!("{key:?}")))?;
        // Grid axis order: outer = num_heads, middle = seq_tokens, inner = batch.
        // Uses the extrapolating variant because Python pre-extends both
        // axes via `_extrapolate` before interpolation; linear extrapolation
        // here from the boundary pair gives the same numeric result for the
        // out-of-envelope queries that show up on sparser backend tables
        // (e.g. SGLang DSv32 at low num_heads).
        interp_2d_1d_grid_extrapolate_inner(grid, num_heads, sequence_tokens, b)
    }

    /// Return the raw nested context grids for a given key, exposed so the
    /// operator layer can run the topk-piecewise interpolation against the
    /// per-prefix slices when sparse-attention semantics require it.
    pub fn raw_context_grids(&self) -> Result<&DsaGrids, AicError> {
        self.load_context()
    }

    fn load_context(&self) -> Result<&DsaGrids, AicError> {
        let cell = self.context.get_or_init(|| {
            load_dsa_parquet(
                &self.data_root.join("dsa_context_module_perf.parquet"),
                DsaPhase::Context,
            )
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&DsaGrids, AicError> {
        let cell = self.generation.get_or_init(|| {
            load_dsa_parquet(
                &self.data_root.join("dsa_generation_module_perf.parquet"),
                DsaPhase::Generation,
            )
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_context_prefix_grids(&self) -> Result<&ContextPrefixCache, AicError> {
        let cell = self.context_prefix_grids.get_or_init(|| {
            let grids = self.load_context()?;
            Ok(build_context_prefix_cache(grids))
        });
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation_grids(&self) -> Result<&GenerationGridCache, AicError> {
        let cell = self.generation_grids.get_or_init(|| {
            let grids = self.load_generation()?;
            Ok(build_generation_grid_cache(grids))
        });
        cell.as_ref().map_err(clone_err)
    }
}

/// Match Python `_select_dsa_backend`: prefer the requested backend, then the
/// portable FlashMLA bucket, then TRT-LLM, then any remaining backend bucket
/// for the same physical shape. This keeps legacy single-backend files usable
/// without collapsing distinct backend measurements at load time.
fn select_dsa_backend<'a, T>(
    by_keys: &'a BTreeMap<DsaKey, T>,
    requested: &DsaKey,
) -> Option<&'a T> {
    for backend in [requested.dsa_backend.as_str(), "flashmla_kv", "trtllm"] {
        if let Some(value) = by_keys.get(&requested.with_backend(backend)) {
            return Some(value);
        }
    }
    by_keys
        .iter()
        .find(|(key, _)| requested.same_shape_except_backend(key))
        .map(|(_, value)| value)
}

/// Materialise the per-`(DsaKey, prefix)` `Grid3` shape that `query_context`
/// needs (axes: num_heads, isl, batch). Exact reproduction of the historical
/// `pick_prefix_slice` loop so numerics are identical.
fn build_context_prefix_cache(grids: &DsaGrids) -> ContextPrefixCache {
    let mut by_keys: BTreeMap<DsaKey, BTreeMap<u32, Grid3<f64>>> = BTreeMap::new();
    for (key, by_heads) in &grids.by_keys {
        let entry = by_keys.entry(key.clone()).or_default();
        for (&n, by_step) in by_heads {
            for (&prefix, by_isl) in by_step {
                let slice = entry.entry(prefix).or_default();
                for (&isl, by_batch) in by_isl {
                    for (&bb, &lat) in by_batch {
                        slice
                            .entry(n)
                            .or_default()
                            .entry(isl)
                            .or_default()
                            .insert(bb, lat);
                    }
                }
            }
        }
    }
    ContextPrefixCache { by_keys }
}

/// Materialise the per-`DsaKey` `Grid3` shape that `query_generation` needs
/// (axes: num_heads, sequence tokens, batch). Generation rows are normalized
/// to `(step=0, isl=isl+step)` during load, so physical-key collisions have
/// already been resolved in file order before this cache is built.
fn build_generation_grid_cache(grids: &DsaGrids) -> GenerationGridCache {
    let mut by_keys: BTreeMap<DsaKey, Grid3<f64>> = BTreeMap::new();
    for (key, by_heads) in &grids.by_keys {
        let grid = by_keys.entry(key.clone()).or_default();
        for (&n, by_step) in by_heads {
            for (&step, by_isl) in by_step {
                for (&isl, by_batch) in by_isl {
                    let seq = isl + step;
                    for (&bb, &lat) in by_batch {
                        grid.entry(n)
                            .or_default()
                            .entry(seq)
                            .or_default()
                            .insert(bb, lat);
                    }
                }
            }
        }
    }
    GenerationGridCache { by_keys }
}

/// Extract the exact `(num_heads, b)` sequence curve (`isl -> latency`) from a
/// per-prefix `[num_heads][isl][batch]` slice. Mirrors the `curve` dict in
/// Python `interp_context_topk_piecewise_from_raw` — only `seq_len`s that have
/// the exact requested `b` are included. Returns `None` when the head is absent
/// (so the caller skips the piecewise branch).
fn build_exact_seq_curve(slice: &Grid3<f64>, num_heads: u32, b: u32) -> Option<BTreeMap<u32, f64>> {
    let head_slice = slice.get(&num_heads)?;
    let mut curve: BTreeMap<u32, f64> = BTreeMap::new();
    for (&isl_v, by_batch) in head_slice {
        if let Some(&lat) = by_batch.get(&b) {
            curve.insert(isl_v, lat);
        }
    }
    Some(curve)
}

/// Port of Python `operations.dsv4._dsv4_robust_3d_lookup(dict_, x, y, z,
/// batch_axis="z")` for the DSA context prefix slice.
///
/// `slice` is `[num_heads][isl][batch]`; we look up `(num_heads, isl, b)`:
///   1. exact `slice[num_heads][isl][b]`;
///   2. strict 3-D interpolation — `interp_2d_1d_grid_strict` errors on a
///      degenerate axis, mirroring Python's cubic `interp_3d` raising
///      `QhullError` and falling through;
///   3. sampled-batch scaling: take the largest sampled batch `bp <= b`,
///      interpolate (then, on a second pass, extrapolate) along `isl`, and
///      scale the leaf by `b / bp`.
fn dsv4_robust_3d_lookup(
    slice: &Grid3<f64>,
    num_heads: u32,
    isl: u32,
    b: u32,
) -> Result<f64, AicError> {
    // Step 1: exact leaf.
    if let Some(value) = slice
        .get(&num_heads)
        .and_then(|m| m.get(&isl))
        .and_then(|r| r.get(&b))
    {
        return Ok(*value);
    }

    // Step 2: strict 3-D interpolation over (num_heads, isl, batch). Errors on
    // a degenerate axis (single num_heads, single isl, or single batch) — that
    // is the signal to fall through to batch scaling, matching Python.
    if let Ok(value) = interp_2d_1d_grid_strict(slice, num_heads, isl, b, "DSA context 3-D lookup")
    {
        if value.is_finite() {
            return Ok(value);
        }
    }

    // Step 3: sampled-batch scaling on the exact-head sub-slice `[isl][batch]`.
    let sub = slice.get(&num_heads).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "DSA context robust lookup failed: missing num_heads={num_heads}"
        ))
    })?;

    // Candidate batch points <= query batch, across all isl rows, descending.
    let batch_set: std::collections::BTreeSet<u32> = sub
        .values()
        .flat_map(|by_batch| by_batch.keys().copied())
        .filter(|&bp| bp <= b)
        .collect();
    let batch_points: Vec<u32> = batch_set.into_iter().rev().collect();

    // Two passes: first interpolation-only (inner_only), then allow extrapolation.
    for allow_extrapolate in [false, true] {
        for &bp in &batch_points {
            if let Some(leaf) = lookup_at_batch(sub, isl, bp, allow_extrapolate) {
                let scaled = leaf * (b as f64) / (bp as f64);
                if scaled.is_finite() {
                    return Ok(scaled);
                }
            }
        }
    }

    Err(AicError::PerfDatabase(format!(
        "DSA context robust lookup failed (num_heads={num_heads}, isl={isl}, b={b})"
    )))
}

/// Resolve the latency at `(isl, bp)` within a `[isl][batch]` sub-slice,
/// interpolating along `isl` for the fixed batch `bp`. Mirrors Python
/// `_lookup_at_batch` with `batch_axis="z"`.
fn lookup_at_batch(
    sub: &BTreeMap<u32, BTreeMap<u32, f64>>,
    isl: u32,
    bp: u32,
    allow_extrapolate: bool,
) -> Option<f64> {
    // Exact (isl, bp).
    if let Some(by_batch) = sub.get(&isl) {
        if let Some(&leaf) = by_batch.get(&bp) {
            return Some(leaf);
        }
    }
    // Sequence points that have this exact batch.
    let ss: Vec<u32> = sub
        .iter()
        .filter(|(_, by_batch)| by_batch.contains_key(&bp))
        .map(|(&s, _)| s)
        .collect();
    if ss.len() < 2 {
        return None;
    }
    if !allow_extrapolate && !(ss[0] <= isl && isl <= *ss.last().unwrap()) {
        return None;
    }
    let (sl, sr) = nearest_neighbors(isl, &ss, !allow_extrapolate).ok()?;
    let left = *sub.get(&sl)?.get(&bp)?;
    let right = *sub.get(&sr)?.get(&bp)?;
    Some(interp_1d(sl as f64, sr as f64, left, right, isl as f64))
}

fn dsa_backend_from_kernel_source(kernel_source: &str) -> &'static str {
    if kernel_source.to_ascii_lowercase().contains("trtllm") {
        "trtllm"
    } else {
        "flashmla_kv"
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_dsa_measurement(
    by_keys: &mut BTreeMap<DsaKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>>,
    key: DsaKey,
    num_heads: u32,
    step: u32,
    isl: u32,
    batch_size: u32,
    latency: f64,
) {
    // Python's nested-dict assignment is last-wins for an identical physical
    // row key. Keeping that rule here also makes FMHA provenance
    // collisions in generation deterministic after those non-key dimensions
    // are dropped.
    by_keys
        .entry(key)
        .or_default()
        .entry(num_heads)
        .or_default()
        .entry(step)
        .or_default()
        .entry(isl)
        .or_default()
        .insert(batch_size, latency);
}

fn storage_coordinates(phase: DsaPhase, step: u32, isl: u32) -> Result<(u32, u32), AicError> {
    match phase {
        DsaPhase::Context => Ok((step, isl)),
        // Python generation flattens `(isl, step)` to the physical sequence
        // key `isl + step` while reading rows. Store that normalized key
        // directly so duplicate rows preserve file-order last-wins semantics;
        // a later BTreeMap traversal must not decide the winner.
        DsaPhase::Generation => isl
            .checked_add(step)
            .map(|sequence_tokens| (0, sequence_tokens))
            .ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "DSA generation sequence length overflow: isl={isl}, step={step}"
                ))
            }),
    }
}

fn load_dsa_parquet(path: &Path, phase: DsaPhase) -> Result<DsaGrids, AicError> {
    let reader = PerfReader::open(path)?;
    let arch_col = reader.col("architecture")?;
    let mla_dtype_col = match phase {
        DsaPhase::Context => Some(reader.col("mla_dtype")?),
        DsaPhase::Generation => None,
    };
    let kv_cache_dtype_col = reader.col("kv_cache_dtype")?;
    let gemm_type_col = reader.col("gemm_type")?;
    let kernel_source_col = reader.col_optional("kernel_source");
    let num_heads_col = reader.col("num_heads")?;
    let batch_size_col = reader.col("batch_size")?;
    let isl_col = reader.col("isl")?;
    let step_col = reader.col("step")?;
    let latency_col = reader.col("latency")?;

    let mut by_keys: BTreeMap<
        DsaKey,
        BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>,
    > = BTreeMap::new();
    for row in reader.rows()? {
        let row = row?;
        let architecture = row
            .str_optional(Some(arch_col))?
            .unwrap_or(DEFAULT_DSA_ARCHITECTURE)
            .to_string();
        let kv_quant = row.str_owned(kv_cache_dtype_col)?;
        let gemm_quant = row.str_owned(gemm_type_col)?;
        let dsa_backend =
            dsa_backend_from_kernel_source(row.str_optional(kernel_source_col)?.unwrap_or(""));
        let key = match phase {
            DsaPhase::Context => DsaKey::context(
                architecture,
                row.str_owned(mla_dtype_col.expect("context phase resolves mla_dtype column"))?,
                kv_quant,
                gemm_quant,
                dsa_backend,
            ),
            DsaPhase::Generation => {
                DsaKey::generation(architecture, kv_quant, gemm_quant, dsa_backend)
            }
        };
        let (stored_step, stored_isl) =
            storage_coordinates(phase, row.u32(step_col)?, row.u32(isl_col)?)?;
        insert_dsa_measurement(
            &mut by_keys,
            key,
            row.u32(num_heads_col)?,
            stored_step,
            stored_isl,
            row.u32(batch_size_col)?,
            row.f64(latency_col)?,
        );
    }
    if by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSA module rows loaded from {}",
            path.display()
        )));
    }
    Ok(DsaGrids { by_keys })
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

    const REPO_ROOT_HINT: &str = env!("CARGO_MANIFEST_DIR");

    fn b200_vllm_data_root() -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join("src/aiconfigurator/systems/data/b200_sxm/vllm/0.19.0")
    }

    const INDEX_TOPK: u32 = 2048;

    #[test]
    fn dsa_context_module_exact_hit() {
        // First row of dsa_context_module_perf.txt:
        // arch=DeepseekV32ForCausalLM mla=bfloat16 kv=bfloat16 gemm=bfloat16
        // n=128 b=1 isl=1 step=0 latency=1.0972. Exact (num_heads, isl, b) hit
        // — every dispatch branch collapses to the recorded leaf.
        let table = DsaTable::new(b200_vllm_data_root());
        let latency = table
            .query_context(
                1,
                1,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV32ForCausalLM",
                "flashmla_kv",
                0,
                INDEX_TOPK,
            )
            .expect("DSA context query must succeed");
        assert!(
            (latency - 1.0972).abs() < 1e-6,
            "expected recorded latency, got {latency}"
        );
    }

    #[test]
    fn dsa_unknown_architecture_errors() {
        let table = DsaTable::new(b200_vllm_data_root());
        let err = table
            .query_context(
                1,
                1024,
                128,
                KvCacheQuantMode::Bfloat16,
                FmhaQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "NotAnArchitecture",
                "flashmla_kv",
                0,
                INDEX_TOPK,
            )
            .unwrap_err();
        assert!(matches!(err, AicError::PerfDatabase(_)));
    }

    #[test]
    fn dsa_generation_uses_flash_bucket_when_trtllm_is_requested_but_absent() {
        // First generation row has kernel_source=default, which maps to the
        // flashmla_kv bucket: n=128, b=1, isl=1, step=2, latency=0.2126.
        // Python's selector falls back from requested trtllm to that sole
        // backend bucket; Rust must do the same.
        let table = DsaTable::new(b200_vllm_data_root());
        let latency = table
            .query_generation(
                1,
                3,
                128,
                KvCacheQuantMode::Bfloat16,
                GemmQuantMode::Bfloat16,
                "DeepseekV32ForCausalLM",
                "trtllm",
            )
            .expect("DSA generation backend fallback must succeed");
        assert!((latency - 0.2126).abs() < 1e-6, "got {latency}");
    }

    #[test]
    fn dsa_context_robust_batch_scaling() {
        // Synthetic single-batch prefix slice: only (isl=128, b=1) is recorded
        // at prefix=128. A query at b=4 has no exact/strict-3D path (singleton
        // isl and batch axes) and must fall to sampled-batch scaling:
        // leaf@(128,1) * 4/1. Mirrors the disagg DSA sglang/vllm probe shape.
        let mut slice: Grid3<f64> = BTreeMap::new();
        slice
            .entry(8)
            .or_default()
            .entry(128)
            .or_default()
            .insert(1, 2.6553);
        let value =
            dsv4_robust_3d_lookup(&slice, 8, 128, 4).expect("batch-scaling fallback must succeed");
        assert!(
            (value - 10.6212).abs() < 1e-6,
            "expected leaf*4 = 10.6212, got {value}"
        );

        // DSV3.2 analogue: leaf 1.4548 * 4 = 5.8192.
        let mut slice2: Grid3<f64> = BTreeMap::new();
        slice2
            .entry(16)
            .or_default()
            .entry(128)
            .or_default()
            .insert(1, 1.4548);
        let v2 = dsv4_robust_3d_lookup(&slice2, 16, 128, 4)
            .expect("batch-scaling fallback must succeed");
        assert!((v2 - 5.8192).abs() < 1e-6, "expected 5.8192, got {v2}");
    }

    #[test]
    fn dsa_context_piecewise_single_seq_returns_to_robust() {
        // A prefix slice with a single seq anchor makes the piecewise branch
        // return None (< 2 same-regime anchors), so `query_context` falls
        // through to the robust batch-scaling path. End-to-end check via the
        // public API using the synthetic exact-leaf helper is covered above;
        // here we assert the piecewise primitive itself returns None.
        // `interp_context_topk_piecewise` is imported at module scope.
        let mut curve: BTreeMap<u32, f64> = BTreeMap::new();
        curve.insert(128, 2.6553);
        assert_eq!(
            interp_context_topk_piecewise(&curve, 128, 1920),
            Some(2.6553)
        );
        // Non-exact query with a single anchor -> None.
        assert_eq!(interp_context_topk_piecewise(&curve, 200, 1920), None);
    }

    #[test]
    fn dsa_context_keys_keep_architecture_fmha_and_backend_buckets() {
        let base = DsaKey::context(
            "DeepseekV32ForCausalLM",
            "bfloat16",
            "fp8",
            "fp8_block",
            "flashmla_kv",
        );
        let other_arch = DsaKey::context(
            "GlmMoeDsaForCausalLM",
            "bfloat16",
            "fp8",
            "fp8_block",
            "flashmla_kv",
        );
        let other_fmha = DsaKey::context(
            "DeepseekV32ForCausalLM",
            "fp8",
            "fp8",
            "fp8_block",
            "flashmla_kv",
        );
        let other_backend = base.with_backend("trtllm");

        let buckets = BTreeMap::from([
            (base.clone(), 1.0),
            (other_arch, 2.0),
            (other_fmha, 3.0),
            (other_backend.clone(), 4.0),
        ]);
        assert_eq!(buckets.len(), 4);
        assert_eq!(select_dsa_backend(&buckets, &base), Some(&1.0));
        assert_eq!(select_dsa_backend(&buckets, &other_backend), Some(&4.0));

        let flash_only = BTreeMap::from([(base.clone(), 7.0)]);
        assert_eq!(select_dsa_backend(&flash_only, &other_backend), Some(&7.0));
    }

    #[test]
    fn dsa_generation_ignores_fmha_and_duplicate_physical_rows_are_last_wins() {
        // Generation has no FMHA key constructor by design: rows that differ
        // only by the CSV's provenance-only mla_dtype land on the same key.
        let key_from_bf16_row =
            DsaKey::generation("DeepseekV32ForCausalLM", "fp8", "fp8_block", "flashmla_kv");
        let key_from_fp8_row =
            DsaKey::generation("DeepseekV32ForCausalLM", "fp8", "fp8_block", "flashmla_kv");
        assert_eq!(key_from_bf16_row, key_from_fp8_row);

        let mut rows = BTreeMap::new();
        let (first_step, first_isl) = storage_coordinates(DsaPhase::Generation, 0, 1024).unwrap();
        insert_dsa_measurement(
            &mut rows,
            key_from_bf16_row,
            128,
            first_step,
            first_isl,
            1,
            11.0,
        );
        // A different raw `(step, isl)` pair with the same physical sequence
        // length collides, and the later file row wins just as in Python.
        let (second_step, second_isl) =
            storage_coordinates(DsaPhase::Generation, 512, 512).unwrap();
        insert_dsa_measurement(
            &mut rows,
            key_from_fp8_row.clone(),
            128,
            second_step,
            second_isl,
            1,
            19.0,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[&key_from_fp8_row][&128][&0][&1024][&1], 19.0);
    }

    #[test]
    fn dsa_generation_backends_do_not_overwrite_each_other() {
        let flash = DsaKey::generation("DeepseekV32ForCausalLM", "fp8", "fp8_block", "flashmla_kv");
        let trtllm = flash.with_backend("trtllm");
        let mut rows = BTreeMap::new();
        insert_dsa_measurement(&mut rows, flash.clone(), 128, 0, 1024, 1, 11.0);
        insert_dsa_measurement(&mut rows, trtllm.clone(), 128, 0, 1024, 1, 19.0);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[&flash][&128][&0][&1024][&1], 11.0);
        assert_eq!(rows[&trtllm][&128][&0][&1024][&1], 19.0);
    }

    #[test]
    fn dsa_backend_classifier_matches_python_and_collector() {
        assert_eq!(dsa_backend_from_kernel_source("trtllm_dsa"), "trtllm");
        assert_eq!(dsa_backend_from_kernel_source("TRTLLM_MLA"), "trtllm");
        assert_eq!(dsa_backend_from_kernel_source("dsa_nsa"), "flashmla_kv");
        assert_eq!(dsa_backend_from_kernel_source(""), "flashmla_kv");
    }
}
