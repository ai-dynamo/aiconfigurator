// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DSA (DeepSeek-V3.2 Dynamic Sparse Attention) module perf tables.
//!
//! Two parquet files: `dsa_context_module_perf.parquet` and
//! `dsa_generation_module_perf.parquet`. Both share columns: model,
//! architecture, mla_dtype, kv_cache_dtype, gemm_type, num_heads,
//! batch_size, isl, tp_size, step, latency.
//!
//! Data is nested by (architecture, mla_dtype, kv_cache_dtype, gemm_type)
//! → num_heads → step → isl → batch_size → latency. The `step` axis is the
//! "prefix value" — some architectures (e.g. GlmMoeDsaForCausalLM) need it
//! exposed; others use a single step=0 slice. The query API returns the
//! raw nested slice at a given (key, prefix) so the operator layer can run
//! the topk-piecewise dispatch.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::common::error::AicError;
use crate::config::{PerfDbSources, PerfSource};
use super::{kernel_source_ok, resolve_op_sources};
use super::interpolation::{
    interp_1d, interp_2d_1d_grid_extrapolate_inner, interp_2d_1d_grid_strict,
    interp_context_topk_piecewise, nearest_neighbors, Grid3,
};
use crate::perf_database::parquet_loader::PerfReader;

pub struct DsaTable {
    data_root: PathBuf,
    /// Ordered, priority-sorted sources for the context/generation DSA perf
    /// files (shared-layer aware; see [`PerfSource`]). Single-primary,
    /// no-filter by default (`DsaTable::new`).
    context_sources: Vec<PerfSource>,
    generation_sources: Vec<PerfSource>,
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

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DsaKey {
    pub architecture: String,
    pub fmha_quant: String,
    pub kv_quant: String,
    pub gemm_quant: String,
}

impl DsaTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf) -> Self {
        Self::with_sources(data_root, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each DSA file falls back to its
    /// primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(data_root: PathBuf, perf_db_sources: &PerfDbSources) -> Self {
        let context_sources =
            resolve_op_sources(perf_db_sources, "dsa_context_module_perf.parquet", &data_root);
        let generation_sources = resolve_op_sources(
            perf_db_sources,
            "dsa_generation_module_perf.parquet",
            &data_root,
        );
        Self {
            data_root,
            context_sources,
            generation_sources,
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
        prefix: u32,
        index_topk: u32,
    ) -> Result<f64, AicError> {
        let cache = self.load_context_prefix_grids()?;
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let by_prefix = cache.by_keys.get(&key).ok_or_else(|| {
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
        fmha_quant: FmhaQuantMode,
        gemm_quant: GemmQuantMode,
        architecture: &str,
    ) -> Result<f64, AicError> {
        let cache = self.load_generation_grids()?;
        let key = DsaKey {
            architecture: architecture.to_string(),
            fmha_quant: fmha_quant.name().to_string(),
            kv_quant: kv_quant.name().to_string(),
            gemm_quant: gemm_quant.name().to_string(),
        };
        let grid = cache
            .by_keys
            .get(&key)
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
        let cell = self
            .context
            .get_or_init(|| load_dsa_parquet(&self.context_sources));
        cell.as_ref().map_err(clone_err)
    }

    fn load_generation(&self) -> Result<&DsaGrids, AicError> {
        let cell = self
            .generation
            .get_or_init(|| load_dsa_parquet(&self.generation_sources));
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
                        slice.entry(n).or_default().entry(isl).or_default().insert(bb, lat);
                    }
                }
            }
        }
    }
    ContextPrefixCache { by_keys }
}

/// Materialise the per-`DsaKey` `Grid3` shape that `query_generation` needs
/// (axes: num_heads, seq=isl+step, batch). Iterates the same already-loaded
/// nested map in the same BTreeMap-sorted order, so the last-write-wins
/// semantics on `seq` ties (largest step wins) are identical to the pre-fix
/// per-call rebuild.
fn build_generation_grid_cache(grids: &DsaGrids) -> GenerationGridCache {
    let mut by_keys: BTreeMap<DsaKey, Grid3<f64>> = BTreeMap::new();
    for (key, by_heads) in &grids.by_keys {
        let grid = by_keys.entry(key.clone()).or_default();
        for (&n, by_step) in by_heads {
            for (&step, by_isl) in by_step {
                for (&isl, by_batch) in by_isl {
                    let seq = isl + step;
                    for (&bb, &lat) in by_batch {
                        grid.entry(n).or_default().entry(seq).or_default().insert(bb, lat);
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
fn build_exact_seq_curve(
    slice: &Grid3<f64>,
    num_heads: u32,
    b: u32,
) -> Option<BTreeMap<u32, f64>> {
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
    if let Ok(value) = interp_2d_1d_grid_strict(slice, num_heads, isl, b, "DSA context 3-D lookup") {
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

/// Load a DSA module table from an ordered, priority-sorted source list. Sources
/// are read in order; the first source containing a key wins (`or_insert`),
/// mirroring Python's `_read_filtered_rows` concatenation + `load_dsa_module_data`
/// skip-on-key-conflict. Missing files are skipped (a sibling declared in the
/// manifest need not exist for every system); an error is returned only when no
/// source yields rows. Shared by both the context and generation DSA files.
fn load_dsa_parquet(sources: &[PerfSource]) -> Result<DsaGrids, AicError> {
    let mut by_keys: BTreeMap<DsaKey, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, f64>>>>> =
        BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
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
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let key = DsaKey {
                architecture: row.str_owned(arch_col)?,
                fmha_quant: row.str_owned(mla_dtype_col)?,
                kv_quant: row.str_owned(kv_cache_dtype_col)?,
                gemm_quant: row.str_owned(gemm_type_col)?,
            };
            // First-wins parity with Python `load_dsa_module_data`, extended
            // across shared-layer sources (earlier source wins).
            by_keys
                .entry(key)
                .or_default()
                .entry(row.u32(num_heads_col)?)
                .or_default()
                .entry(row.u32(step_col)?)
                .or_default()
                .entry(row.u32(isl_col)?)
                .or_default()
                .entry(row.u32(batch_size_col)?)
                .or_insert(row.f64(latency_col)?);
        }
    }
    if !any_source || by_keys.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no DSA module rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources.first().map(|s| s.path().display().to_string()).unwrap_or_default()
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
                0,
                INDEX_TOPK,
            )
            .unwrap_err();
        assert!(matches!(err, AicError::PerfDatabase(_)));
    }

    #[test]
    fn dsa_context_robust_batch_scaling() {
        // Synthetic single-batch prefix slice: only (isl=128, b=1) is recorded
        // at prefix=128. A query at b=4 has no exact/strict-3D path (singleton
        // isl and batch axes) and must fall to sampled-batch scaling:
        // leaf@(128,1) * 4/1. Mirrors the disagg DSA sglang/vllm probe shape.
        let mut slice: Grid3<f64> = BTreeMap::new();
        slice.entry(8).or_default().entry(128).or_default().insert(1, 2.6553);
        let value = dsv4_robust_3d_lookup(&slice, 8, 128, 4)
            .expect("batch-scaling fallback must succeed");
        assert!(
            (value - 10.6212).abs() < 1e-6,
            "expected leaf*4 = 10.6212, got {value}"
        );

        // DSV3.2 analogue: leaf 1.4548 * 4 = 5.8192.
        let mut slice2: Grid3<f64> = BTreeMap::new();
        slice2.entry(16).or_default().entry(128).or_default().insert(1, 1.4548);
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
        assert_eq!(interp_context_topk_piecewise(&curve, 128, 1920), Some(2.6553));
        // Non-exact query with a single anchor -> None.
        assert_eq!(interp_context_topk_piecewise(&curve, 200, 1920), None);
    }
}
