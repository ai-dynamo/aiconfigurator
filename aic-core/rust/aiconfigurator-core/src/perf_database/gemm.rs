// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GEMM family perf tables: gemm, compute_scale, scale_matrix.
//!
//! Mirrors the SILICON-mode query algorithm of
//! `aiconfigurator.sdk.operations.gemm.GEMM._query_*_table`. SOL / EMPIRICAL
//! / HYBRID modes layer formulaic fallbacks on top of these queries; they
//! live with the operator code in `operators/gemm.rs`.
//!
//! Each table is lazy: the CSV is read on first query. The `gemm` table is
//! 3-D over `(m, n, k)`; the supporting `compute_scale` and `scale_matrix`
//! tables are 2-D over `(m, k)` and used only by the `fp8_static` quant
//! mode. The compute/scale CSVs are absent for backends that do not need
//! them (vLLM, SGLang); the loaders surface a clear error in that case.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use quick_cache::sync::{Cache, DefaultLifecycle};
use quick_cache::{DefaultHashBuilder, OptionsBuilder, UnitWeighter};

use super::interpolation::Grid3;
use super::perf_interp::{
    self, LeafValue, Node, OpInterpConfig, Resolver, SiteIndex, ValueTransform,
};
use super::{kernel_source_ok, resolve_op_sources};
use crate::common::enums::GemmQuantMode;
use crate::common::error::AicError;
use crate::common::system_spec::SystemSpec;
use crate::config::{PerfDbSources, PerfSource};
use crate::perf_database::parquet_loader::PerfReader;

const GEMM_QUERY_CACHE_CAPACITY: usize = 32_768;
// Keep construction and per-table memory independent of large host CPU counts.
const GEMM_QUERY_CACHE_SHARDS: usize = 16;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct GemmQueryKey {
    quant: GemmQuantMode,
    m: u32,
    n: u32,
    k: u32,
}

// Quick Cache is the intended pattern for hot, high-cardinality scalar-query
// memoization in the perf database. DSA's `sparse` map is deliberately
// different: it holds a small, unbounded set of lazily loaded table objects,
// rather than memoizing high-volume scalar results.
fn gemm_query_cache() -> Cache<GemmQueryKey, LeafValue> {
    let options = OptionsBuilder::new()
        .estimated_items_capacity(GEMM_QUERY_CACHE_CAPACITY)
        .weight_capacity(GEMM_QUERY_CACHE_CAPACITY as u64)
        .shards(GEMM_QUERY_CACHE_SHARDS)
        .build()
        .expect("valid static GEMM query cache options");
    Cache::with_options(
        options,
        UnitWeighter,
        DefaultHashBuilder::default(),
        DefaultLifecycle::default(),
    )
}

#[inline(always)]
fn resolve_gemm_query_cached<F>(
    cache: &Cache<GemmQueryKey, LeafValue>,
    key: GemmQueryKey,
    resolve: F,
) -> Result<LeafValue, AicError>
where
    F: FnOnce() -> Result<LeafValue, AicError>,
{
    if let Some(value) = cache.get(&key) {
        return Ok(value);
    }

    let value = resolve()?;
    // Preserve resolver behavior by returning non-finite results, but do
    // not make them sticky cache hits for later queries.
    if value.latency.is_finite() && value.power.is_finite() && value.energy.is_finite() {
        cache.insert(key, value);
    }
    Ok(value)
}

/// GEMM-family perf-data owner for one logical
/// `<system>/<backend>/<version>` selection.
///
/// Resolves the physical files under
/// `<system>/<family>/<backend>/<version>` and lazily loads the three parquet
/// tables. Construct via `GemmTable::new`; queries trigger the relevant
/// table's load on first use.
///
/// `system_spec` is kept for SOL clamping at load time, mirroring Python's
/// `GEMM._correct_sol`. The supporting `compute_scale` / `scale_matrix`
/// tables are NOT SOL-clamped — Python's `_correct_data` only touches the
/// main GEMM table.
pub struct GemmTable {
    data_root: PathBuf,
    system_spec: SystemSpec,
    /// Ordered, priority-sorted sources for each of the three GEMM-family perf
    /// files (shared-layer aware; see [`PerfSource`]). Single-primary,
    /// no-filter by default (`GemmTable::new`).
    gemm_sources: Vec<PerfSource>,
    compute_scale_sources: Vec<PerfSource>,
    scale_matrix_sources: Vec<PerfSource>,
    gemm: OnceLock<Result<GemmEngineGrids, AicError>>,
    compute_scale: OnceLock<Result<TwoDGrids, AicError>>,
    scale_matrix: OnceLock<Result<TwoDGrids, AicError>>,
    /// Successful finite GEMM query leaves (`{latency, power, energy}`),
    /// keyed by normalized quant and shape and bounded independently for
    /// each shard.
    query_cache: OnceLock<Cache<GemmQueryKey, LeafValue>>,
}

/// 3-D GEMM tables keyed by quant name -> m -> n -> k -> measured leaf
/// (`{latency, power, energy}`, mirroring Python `load_gemm_data`).
/// `quant_order` records first-seen (file row) order: the `BTreeMap` iterates
/// alphabetically, but the quant-transfer ladder's tie-breaks are pinned to
/// Python's dict-insertion (= file row) order.
struct GemmGrids {
    by_quant: BTreeMap<String, Grid3<LeafValue>>,
    quant_order: Vec<String>,
    /// Per-quant replica of Python's nested-dict KEY order (issue #1456):
    /// the site-transfer tie-break is parity surface, and Python enumerates
    /// (n, k) sites in the insertion order of its `data[m][n][k]` dict walk
    /// — an order the sorted `BTreeMap` grids above destroy. Recorded at
    /// row-accept time and replayed into the `SiteIndex` build.
    site_walk_by_quant: BTreeMap<String, SiteWalkOrder>,
}

/// First-seen key order of the `data[m][n][k]` nested dict, per level —
/// exactly what Python's insertion-ordered dicts remember. `site_order()`
/// replays the walk (`m` first-seen → `n` first-seen under that `m` → `k`
/// first-seen under that `(m, n)`) and yields each `(n, k)` site at its
/// first encounter, which is the enumeration order Python's site index
/// (`perf_interp/engine.py` `_site_index`) sees.
#[derive(Default)]
struct SiteWalkOrder {
    m_order: Vec<u32>,
    n_order: BTreeMap<u32, Vec<u32>>,
    k_order: BTreeMap<(u32, u32), Vec<u32>>,
}

impl SiteWalkOrder {
    fn site_order(&self) -> Vec<Vec<u32>> {
        let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        let mut out: Vec<Vec<u32>> = Vec::new();
        for &m in &self.m_order {
            let Some(ns) = self.n_order.get(&m) else {
                continue;
            };
            for &n in ns {
                let Some(ks) = self.k_order.get(&(m, n)) else {
                    continue;
                };
                for &k in ks {
                    if seen.insert((n, k)) {
                        out.push(vec![n, k]);
                    }
                }
            }
        }
        out
    }
}

/// Engine-ready GEMM tables: per quant, the nested table plus the scattered
/// (n, k)-site index, both built once at load (tables are immutable).
struct GemmEngineGrids {
    by_quant: BTreeMap<String, (Node, SiteIndex)>,
    quant_order: Vec<String>,
}

/// 2-D scale tables keyed by quant name -> m -> k -> measured leaf
/// (`{latency, power, energy}`, mirroring Python `load_compute_scale_data`
/// / `load_scale_matrix_data`).
struct TwoDGrids {
    by_quant: BTreeMap<String, Node>,
}

impl GemmTable {
    /// Construct an empty table for the given data directory. No I/O. Each
    /// perf file is sourced solely from `data_root/<basename>` with no
    /// `kernel_source` filter (pre-shared-layer behaviour).
    pub fn new(data_root: PathBuf, system_spec: SystemSpec) -> Self {
        Self::with_sources(data_root, system_spec, &PerfDbSources::default())
    }

    /// Construct with shared-layer (sibling/cross-version) sources resolved from
    /// `perf_db_sources` (Python-supplied). Each GEMM-family file falls back to
    /// its primary `data_root/<basename>` when absent from the map. No I/O.
    pub fn with_sources(
        data_root: PathBuf,
        system_spec: SystemSpec,
        perf_db_sources: &PerfDbSources,
    ) -> Self {
        let gemm_sources = resolve_op_sources(perf_db_sources, "gemm_perf.parquet", &data_root);
        let compute_scale_sources =
            resolve_op_sources(perf_db_sources, "computescale_perf.parquet", &data_root);
        let scale_matrix_sources =
            resolve_op_sources(perf_db_sources, "scale_matrix_perf.parquet", &data_root);
        Self {
            data_root,
            system_spec,
            gemm_sources,
            compute_scale_sources,
            scale_matrix_sources,
            gemm: OnceLock::new(),
            compute_scale: OnceLock::new(),
            scale_matrix: OnceLock::new(),
            query_cache: OnceLock::new(),
        }
    }

    /// Query the GEMM measured value (`{latency ms, power W, energy W·ms}`)
    /// for the given shape and quant mode.
    ///
    /// Mirrors the perf_interp v2 path of `GEMM._query_gemm_table`
    /// (`gemm_config`): (n, k) are scattered collected shapes, each owning
    /// an m-curve. Exact site -> its own curve (exact point / lerp /
    /// k_tail=3 util-hold beyond the sweep); unknown shape -> log2-IDW util
    /// transfer from <=4 covering neighbour sites within 2.0 octaves.
    pub fn query(
        &self,
        quant: GemmQuantMode,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<LeafValue, AicError> {
        // `fp8_static` is a behavioral mode that reuses `fp8` perf tables,
        // mirroring Python `GEMM._normalize_for_lookup`. The
        // compute_scale / scale_matrix tables apply the same
        // normalization in their respective query methods.
        let lookup_quant = normalize_fp8_static_quant(quant);
        // Resolve flops BEFORE any perf-data lookup: Python resolves at
        // `_query_gemm_table` entry in every mode, so a missing dtype entry
        // must classify as MissingSystemFlops on both engines — not as a
        // data miss when the quant's table also happens to be uncollected.
        let spec = &self.system_spec;
        let tc_flops = quant_tc_flops(spec, lookup_quant.mapping())?;
        // Keep the cache probe below FLOPS resolution, even on hits. Hoisting
        // it would change MissingSystemFlops-over-PerfDatabase error precedence.
        let key = GemmQueryKey {
            quant: lookup_quant,
            m,
            n,
            k,
        };
        let cache = self.query_cache.get_or_init(gemm_query_cache);
        resolve_gemm_query_cached(cache, key, || {
            let grids = self.load_gemm()?;
            let quant_name = lookup_quant.name();
            let (_, index) = grids.by_quant.get(quant_name).ok_or_else(|| {
                AicError::PerfDatabase(format!(
                    "GEMM perf data missing for quant '{quant_name}' at {}; available: {:?}",
                    self.data_root.display(),
                    grids.by_quant.keys().collect::<Vec<_>>(),
                ))
            })?;
            let sol = move |c: &[f64]| {
                gemm_sol_latency_ms_with_flops(spec, lookup_quant, tc_flops, c[0], c[1], c[2])
            };
            let cfg = gemm_engine_config(&sol);
            index.resolve_value(&cfg, &[m as f64, n as f64, k as f64])
        })
    }

    /// Query compute-scale measured value — used by `fp8_static` GEMM only.
    ///
    /// Like the main GEMM table, the compute_scale data is keyed by `fp8`
    /// (not `fp8_static`) in the perf-DB; normalize before lookup to mirror
    /// Python's `GEMM._normalize_for_lookup`.
    ///
    /// compute_scale stores a quantization-overhead DELTA: beyond the grid
    /// it is deliberately held FLAT at the clamped boundary (Python
    /// `_query_compute_scale_table` contract), energy included.
    pub fn query_compute_scale(
        &self,
        quant: GemmQuantMode,
        m: u32,
        k: u32,
    ) -> Result<LeafValue, AicError> {
        let grids = self.load_compute_scale()?;
        let lookup = normalize_fp8_static_quant(quant);
        let spec = &self.system_spec;
        // sol_mem = 2 m k / bw * 1000 (read + write of the activation)
        let sol = move |c: &[f64]| 2.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0;
        query_scale_table(
            &grids.by_quant,
            lookup.name(),
            m,
            k,
            &sol,
            false,
            &self.data_root,
        )
    }

    /// Query scale-matrix measured value — used by `fp8_static` GEMM only.
    /// Same `fp8_static -> fp8` normalization as the GEMM and
    /// compute_scale lookups.
    ///
    /// scale_matrix is a real memory kernel: outside the grid the boundary
    /// utilization is frozen and SOL(q)/SOL(boundary) carries the growth
    /// for BOTH latency and energy (Python `_query_scale_matrix_table`
    /// contract).
    pub fn query_scale_matrix(
        &self,
        quant: GemmQuantMode,
        m: u32,
        k: u32,
    ) -> Result<LeafValue, AicError> {
        let grids = self.load_scale_matrix()?;
        let lookup = normalize_fp8_static_quant(quant);
        let spec = &self.system_spec;
        let sol = move |c: &[f64]| 3.0 * c[0] * c[1] / spec.gpu.mem_bw * 1000.0;
        query_scale_table(
            &grids.by_quant,
            lookup.name(),
            m,
            k,
            &sol,
            true,
            &self.data_root,
        )
    }

    /// Collected `(m, n, k) -> latency` points for the quant's table, for the
    /// operator-layer util-calibration grid (Python's
    /// `require_data_slice(_gemm_data, tqm)` + `iter_grid(..., depth=3)`).
    /// Missing quant / empty table is a typed `PerfDatabase` miss.
    pub fn gemm_points(&self, quant: GemmQuantMode) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_gemm()?;
        let quant_name = normalize_fp8_static_quant(quant).name();
        let (node, _) = grids.by_quant.get(quant_name).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "GEMM perf data missing for quant '{quant_name}' at {}",
                self.data_root.display()
            ))
        })?;
        let points = crate::perf_database::perf_interp::node_points(node);
        if points.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "GEMM perf data empty for quant '{quant_name}' at {}",
                self.data_root.display()
            )));
        }
        Ok(points)
    }

    /// Collected `(m, k) -> delta` points of the compute_scale table (zeroes
    /// included — they are measured deltas). Typed miss when absent.
    pub fn compute_scale_points(
        &self,
        quant: GemmQuantMode,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_compute_scale()?;
        Self::two_d_points(
            grids,
            normalize_fp8_static_quant(quant),
            "compute_scale",
            &self.data_root,
        )
    }

    /// Collected `(m, k) -> latency` points of the scale_matrix table.
    /// Typed miss when absent.
    pub fn scale_matrix_points(
        &self,
        quant: GemmQuantMode,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let grids = self.load_scale_matrix()?;
        Self::two_d_points(
            grids,
            normalize_fp8_static_quant(quant),
            "scale_matrix",
            &self.data_root,
        )
    }

    fn two_d_points(
        grids: &TwoDGrids,
        quant: GemmQuantMode,
        table_name: &str,
        data_root: &Path,
    ) -> Result<Vec<(Vec<f64>, f64)>, AicError> {
        let quant_name = quant.name();
        let node = grids.by_quant.get(quant_name).ok_or_else(|| {
            AicError::PerfDatabase(format!(
                "{table_name} perf data missing for quant '{quant_name}' at {}",
                data_root.display()
            ))
        })?;
        let points = crate::perf_database::perf_interp::node_points(node);
        if points.is_empty() {
            return Err(AicError::PerfDatabase(format!(
                "{table_name} perf data empty for quant '{quant_name}' at {}",
                data_root.display()
            )));
        }
        Ok(points)
    }

    /// Distinct quant names of the loaded GEMM table, in first-seen (file
    /// row) order — the exact analogue of `MoeTable::available_quants` and of
    /// Python's dict-insertion iteration over the gemm data. The
    /// quant-transfer ladder's tie-breaks depend on this order; the
    /// alphabetical `BTreeMap` iteration must NOT be used for it.
    pub fn available_quants(&self) -> Result<&[String], AicError> {
        Ok(&self.load_gemm()?.quant_order)
    }

    fn load_gemm(&self) -> Result<&GemmEngineGrids, AicError> {
        let cell = self.gemm.get_or_init(|| {
            let mut grids = load_gemm_parquet(&self.gemm_sources)?;
            // Mirror Python `GEMM._correct_sol`: clamp every stored grid
            // entry to `>= SOL`. SOL is deterministic from the system spec
            // and (m, n, k, quant); on currently-aligned surfaces this is
            // a no-op (raw >= SOL already), so the clamp only affects
            // systems whose collected data drops below SOL (the prime
            // example is l40s at small bf16 shapes).
            //
            // Python interpolates over already-clamped grid points; we
            // mirror that ordering by mutating the grid before any
            // queries run. Off-grid interpolation/extrapolation therefore
            // sees the same monotone-bounded inputs as Python.
            clamp_gemm_grids_to_sol(&self.system_spec, &mut grids);
            // Build the engine table + (n, k)-site index once per quant. The
            // index enumerates sites in Python's dict-walk order (recorded at
            // load) so exact-distance tie-breaks at the nearest-k cutoff match
            // the frozen Python engine bit-for-bit (issue #1456).
            let quant_order = grids.quant_order;
            let site_walks = grids.site_walk_by_quant;
            let by_quant = grids
                .by_quant
                .into_iter()
                .map(|(quant_name, grid)| {
                    let node = grid3_to_node(&grid);
                    let site_order = site_walks
                        .get(&quant_name)
                        .map(|walk| walk.site_order())
                        .unwrap_or_default();
                    let index = SiteIndex::build_with_site_order(&[1, 2], 0, &node, &site_order);
                    (quant_name, (node, index))
                })
                .collect();
            Ok(GemmEngineGrids {
                by_quant,
                quant_order,
            })
        });
        cell.as_ref().map_err(|err| clone_err(err))
    }

    fn load_compute_scale(&self) -> Result<&TwoDGrids, AicError> {
        let cell = self
            .compute_scale
            .get_or_init(|| load_two_d_parquet(&self.compute_scale_sources));
        cell.as_ref().map_err(|err| clone_err(err))
    }

    fn load_scale_matrix(&self) -> Result<&TwoDGrids, AicError> {
        let cell = self
            .scale_matrix
            .get_or_init(|| load_two_d_parquet(&self.scale_matrix_sources));
        cell.as_ref().map_err(|err| clone_err(err))
    }
}

/// Speed-of-light GEMM latency in ms, from a pre-resolved `tc_flops`.
///
/// Mirrors Python's `GEMM._query_gemm_table::get_sol`:
/// - `sol_math = 2 * m * n * k / tc_flops * 1000`
/// - `sol_mem  = quant.memory * (m*n + m*k + n*k) / mem_bw * 1000`
/// - `sol      = max(sol_math, sol_mem)`
pub(crate) fn gemm_sol_latency_ms_with_flops(
    spec: &SystemSpec,
    quant: GemmQuantMode,
    tc_flops: f64,
    m: f64,
    n: f64,
    k: f64,
) -> f64 {
    let mapping = quant.mapping();
    let sol_math = 2.0 * m * n * k / tc_flops * 1000.0;
    let sol_mem = mapping.memory * (m * n + m * k + n * k) / spec.gpu.mem_bw * 1000.0;
    sol_math.max(sol_mem)
}

// Strict resolver lives in common/system_spec.rs (it depends only on
// common-layer types); re-exported here because every SOL caller in the
// perf_database/operators layers already imports it from this module.
pub(crate) use crate::common::system_spec::quant_tc_flops;

/// In-place SOL clamp for every entry in the GEMM grid set.
///
/// Silicon data can exist for a dtype whose `*_tc_flops` entry is missing
/// from the system YAML (e.g. b60 fp8). Mirror Python `GEMM._correct_sol`:
/// leave that quant's slice unclamped rather than failing the whole
/// database load; query-time SOL/HYBRID paths still reject the quant mode.
fn clamp_gemm_grids_to_sol(spec: &SystemSpec, grids: &mut GemmGrids) {
    for (quant_name, grid) in grids.by_quant.iter_mut() {
        let Some(quant) = gemm_quant_by_name(quant_name) else {
            continue;
        };
        let Ok(tc_flops) = quant_tc_flops(spec, quant.mapping()) else {
            continue;
        };
        for (&m, by_n) in grid.iter_mut() {
            for (&n, by_k) in by_n.iter_mut() {
                for (&k, leaf) in by_k.iter_mut() {
                    let sol = gemm_sol_latency_ms_with_flops(
                        spec, quant, tc_flops, m as f64, n as f64, k as f64,
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

pub(crate) fn gemm_quant_by_name(name: &str) -> Option<GemmQuantMode> {
    use GemmQuantMode::*;
    Some(match name {
        "bfloat16" => Bfloat16,
        "int8_wo" => Int8Wo,
        "int4_wo" => Int4Wo,
        "fp8" => Fp8,
        "fp8_static" => Fp8Static,
        "sq" => Sq,
        "fp8_block" => Fp8Block,
        "fp8_ootb" => Fp8Ootb,
        "nvfp4" => Nvfp4,
        _ => return None,
    })
}

/// Normalize the `fp8_static` quant mode to `fp8` for perf-table lookups.
/// Mirrors Python `GEMM._normalize_for_lookup`: the `fp8_static` mode is
/// behavioral (subtracts compute_scale + scale_matrix latency) but reuses
/// the fp8 perf tables — the perf-DB never stores rows under
/// `fp8_static`. Applied uniformly to the GEMM, compute_scale, and
/// scale_matrix table queries.
pub(crate) fn normalize_fp8_static_quant(quant: GemmQuantMode) -> GemmQuantMode {
    if quant == GemmQuantMode::Fp8Static {
        GemmQuantMode::Fp8
    } else {
        quant
    }
}

/// The GEMM engine record: (n, k) sites (axes 1, 2), m-curve (axis 0),
/// mirroring Python `perf_interp.gemm_config`.
fn gemm_engine_config<'a>(sol: &'a dyn Fn(&[f64]) -> f64) -> OpInterpConfig<'a> {
    OpInterpConfig {
        axes: &["m", "n", "k"],
        resolver: Resolver::ScatteredSites {
            site_axes: vec![1, 2],
            curve_axis: 0,
            nn_sites: 4,
            max_site_distance: Some(2.0),
            require_curve_coverage: true,
            k_tail: 3,
            own_curve_coverage_fallback: false,
        },
        sol_fn: sol,
        value_transform: ValueTransform::Raw,
        transform_axis: None,
    }
}

fn grid3_to_node(grid: &Grid3<LeafValue>) -> Node {
    let mut node = Node::branch();
    for (&m, by_n) in grid {
        for (&n, by_k) in by_n {
            for (&k, &leaf) in by_k {
                node.insert_value(&[m, n, k], leaf);
            }
        }
    }
    node
}

/// Scale-table (compute_scale / scale_matrix) query: clamp `(m, k)` into the
/// collected envelope FIRST (legacy contract), resolve the interior on the
/// engine (RAW 2-axis grid), then either hold FLAT at the boundary
/// (compute_scale: a quantization DELTA) or re-scale by SOL(q)/SOL(boundary)
/// (scale_matrix: a real memory kernel; latency AND energy scale by the
/// ratio, mirroring Python's `PerformanceResult(latency * ratio,
/// energy=interpolated.energy * ratio)`). Mirrors Python
/// `_query_compute_scale_table` / `_query_scale_matrix_table`.
fn query_scale_table(
    by_quant: &BTreeMap<String, Node>,
    quant_name: &str,
    m: u32,
    k: u32,
    sol: &dyn Fn(&[f64]) -> f64,
    sol_ratio_beyond_grid: bool,
    data_root: &Path,
) -> Result<LeafValue, AicError> {
    let node = by_quant.get(quant_name).ok_or_else(|| {
        AicError::PerfDatabase(format!(
            "perf data missing for quant '{quant_name}' at {}; available: {:?}",
            data_root.display(),
            by_quant.keys().collect::<Vec<_>>(),
        ))
    })?;
    let Node::Branch(rows) = node else {
        return Err(AicError::PerfDatabase("malformed scale table".to_string()));
    };
    if rows.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "empty scale table for quant '{quant_name}'"
        )));
    }
    let m_keys: Vec<u32> = rows.keys().copied().collect();
    let m_c = m.clamp(m_keys[0], m_keys[m_keys.len() - 1]);
    let mut k_min = u32::MAX;
    let mut k_max = 0u32;
    for row in rows.values() {
        if let Node::Branch(cols) = row {
            if let (Some(&lo), Some(&hi)) = (cols.keys().next(), cols.keys().next_back()) {
                k_min = k_min.min(lo);
                k_max = k_max.max(hi);
            }
        }
    }
    let k_c = k.clamp(k_min, k_max);

    let cfg = OpInterpConfig::grid(&["m", "k"], sol);
    let value = perf_interp::query_value(&cfg, node, &[m_c as f64, k_c as f64])?;
    if !sol_ratio_beyond_grid || (m_c == m && k_c == k) {
        return Ok(value);
    }
    // Outside the grid, freeze utilization at the clamped boundary:
    // L(q) = L(boundary) * SOL(q)/SOL(boundary); energy scales by the same
    // ratio (average power is unchanged), mirroring Python.
    let boundary_sol = sol(&[m_c as f64, k_c as f64]);
    let query_sol = sol(&[m as f64, k as f64]);
    if boundary_sol > 0.0 && query_sol > 0.0 {
        let ratio = query_sol / boundary_sol;
        Ok(LeafValue {
            latency: value.latency * ratio,
            power: value.power,
            energy: value.energy * ratio,
        })
    } else {
        Ok(value)
    }
}

/// Load the GEMM table from an ordered, priority-sorted source list. Sources are
/// read in order; the first source containing a shape wins (`or_insert`),
/// mirroring Python's `_read_filtered_rows` concatenation + `load_gemm_data`
/// skip-on-key-conflict. Missing files are skipped (a sibling declared in the
/// manifest need not exist for every system); an error is returned only when no
/// source yields rows. The OPTIONAL `power` column feeds
/// `LeafValue::with_power` (`energy = power * latency`, W·ms); absent column
/// -> power 0.0, mirroring Python's `row.get("power", 0.0)`.
fn load_gemm_parquet(sources: &[PerfSource]) -> Result<GemmGrids, AicError> {
    let mut by_quant: BTreeMap<String, Grid3<LeafValue>> = BTreeMap::new();
    let mut quant_order: Vec<String> = Vec::new();
    let mut site_walk_by_quant: BTreeMap<String, SiteWalkOrder> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let gemm_dtype_col = reader.col("gemm_dtype")?;
        let m_col = reader.col("m")?;
        let n_col = reader.col("n")?;
        let k_col = reader.col("k")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let dtype = row.str(gemm_dtype_col)?;
            // Skip quant modes AIC does not model in the perf path (matches the
            // legacy perf.rs behavior).
            if dtype == "awq" || dtype == "gptq" {
                continue;
            }
            let dtype = dtype.to_string();
            if !by_quant.contains_key(&dtype) {
                quant_order.push(dtype.clone());
            }
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            let (m, n, k) = (row.u32(m_col)?, row.u32(n_col)?, row.u32(k_col)?);
            // First-wins parity with Python's `load_gemm_data` try/except
            // KeyError, extended across shared-layer sources (earlier source
            // wins). Per-level newness is recorded BEFORE the inserts — it is
            // the Python dict insertion order the site-transfer tie-break
            // depends on (issue #1456).
            let grid = by_quant.entry(dtype.clone()).or_default();
            let m_new = !grid.contains_key(&m);
            let by_n = grid.entry(m).or_default();
            let n_new = !by_n.contains_key(&n);
            let by_k = by_n.entry(n).or_default();
            let k_new = !by_k.contains_key(&k);
            by_k.entry(k)
                .or_insert(LeafValue::with_power(latency, power));
            let walk = site_walk_by_quant.entry(dtype).or_default();
            if m_new {
                walk.m_order.push(m);
            }
            if n_new {
                walk.n_order.entry(m).or_default().push(n);
            }
            if k_new {
                walk.k_order.entry((m, n)).or_default().push(k);
            }
        }
    }
    if !any_source || by_quant.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no GEMM rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    Ok(GemmGrids {
        by_quant,
        quant_order,
        site_walk_by_quant,
    })
}

/// Load a 2-D (compute_scale / scale_matrix) table from an ordered source list.
/// Same first-wins-across-sources + missing-file-skip semantics as
/// [`load_gemm_parquet`], including the optional `power` column.
fn load_two_d_parquet(sources: &[PerfSource]) -> Result<TwoDGrids, AicError> {
    let mut raw: BTreeMap<String, BTreeMap<u32, BTreeMap<u32, LeafValue>>> = BTreeMap::new();
    let mut any_source = false;
    for source in sources {
        let path = source.path();
        if !path.exists() {
            continue;
        }
        any_source = true;
        let reader = PerfReader::open(path)?;
        let quant_dtype_col = reader.col("quant_dtype")?;
        let m_col = reader.col("m")?;
        let k_col = reader.col("k")?;
        let latency_col = reader.col("latency")?;
        let power_col = reader.col_optional("power");
        let ks_col = reader.col_optional("kernel_source");
        for row in reader.rows()? {
            let row = row?;
            if !kernel_source_ok(source.kernel_sources(), ks_col, &row)? {
                continue;
            }
            let latency = row.f64(latency_col)?;
            let power = row.f64_optional(power_col)?.unwrap_or(0.0);
            // First-wins parity (compute_scale / scale_matrix tables in Python),
            // extended across shared-layer sources.
            raw.entry(row.str_owned(quant_dtype_col)?)
                .or_default()
                .entry(row.u32(m_col)?)
                .or_default()
                .entry(row.u32(k_col)?)
                .or_insert(LeafValue::with_power(latency, power));
        }
    }
    if !any_source || raw.is_empty() {
        return Err(AicError::PerfDatabase(format!(
            "no rows loaded from {} source(s) (first: {})",
            sources.len(),
            sources
                .first()
                .map(|s| s.path().display().to_string())
                .unwrap_or_default()
        )));
    }
    let by_quant = raw
        .into_iter()
        .map(|(quant, rows)| {
            let mut node = Node::branch();
            for (m, cols) in rows {
                for (k, leaf) in cols {
                    node.insert_value(&[m, k], leaf);
                }
            }
            (quant, node)
        })
        .collect();
    Ok(TwoDGrids { by_quant })
}

/// Reconstruct an `AicError` from a borrowed cached error so we can hand a
/// fresh owned copy back to the caller (`OnceLock` returns `&Result`, but
/// the API surface returns `Result`).
fn clone_err(err: &AicError) -> AicError {
    AicError::PerfDatabase(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query_cache_len(table: &GemmTable) -> usize {
        table.query_cache.get().map_or(0, Cache::len)
    }

    /// Per-channel bit patterns of a query leaf, for exact-identity asserts
    /// (`assert_eq!` on `f64` would treat `-0.0 == 0.0` and miss NaN).
    fn leaf_bits(value: LeafValue) -> [u64; 3] {
        [
            value.latency.to_bits(),
            value.power.to_bits(),
            value.energy.to_bits(),
        ]
    }

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

    fn b200_gemm_parquet(backend: &str, version: &str) -> PathBuf {
        PathBuf::from(REPO_ROOT_HINT)
            .join("../..")
            .join(format!(
                "src/aiconfigurator_core/systems/data/b200_sxm/gemm/{backend}/{version}/gemm_perf.parquet"
            ))
    }

    fn gemm_shape_count(grids: &GemmGrids) -> usize {
        grids
            .by_quant
            .values()
            .flat_map(|by_m| by_m.values())
            .flat_map(|by_n| by_n.values())
            .map(|by_k| by_k.len())
            .sum()
    }

    /// Shared-layer sibling merge: sources are read in priority order, later
    /// sources only add shapes the earlier ones lack (first-wins), and a
    /// per-source `kernel_source` allowlist gates which sibling rows are
    /// admitted. Mirrors Python `_read_filtered_rows` + `load_gemm_data`.
    #[test]
    fn shared_layer_merges_siblings_with_kernel_source_filter_and_first_wins() {
        // trtllm 1.3.0rc10 primary + 1.2.0rc5 sibling — the real shape Python's
        // `_compute_perf_db_sources` emits for this backend.
        let primary = b200_gemm_parquet("trtllm", "1.3.0rc10");
        let sibling = b200_gemm_parquet("trtllm", "1.2.0rc5");

        let primary_only = load_gemm_parquet(&[PerfSource(primary.clone(), None)]).unwrap();

        // Sibling admitted unfiltered: never drops a primary shape, only adds.
        let merged = load_gemm_parquet(&[
            PerfSource(primary.clone(), None),
            PerfSource(sibling.clone(), None),
        ])
        .unwrap();
        assert!(
            gemm_shape_count(&merged) >= gemm_shape_count(&primary_only),
            "unfiltered sibling must not drop shapes"
        );

        // First-wins: every primary (quant,m,n,k) keeps the PRIMARY latency even
        // though the sibling also carries rows.
        for (q, by_m) in &primary_only.by_quant {
            for (m, by_n) in by_m {
                for (n, by_k) in by_n {
                    for (k, v) in by_k {
                        let got = merged
                            .by_quant
                            .get(q)
                            .and_then(|x| x.get(m))
                            .and_then(|x| x.get(n))
                            .and_then(|x| x.get(k))
                            .copied();
                        assert_eq!(got, Some(*v), "first source must win at ({q},{m},{n},{k})");
                    }
                }
            }
        }

        // A `kernel_source` allowlist that matches nothing drops every sibling
        // row, so the merged table equals primary-only.
        let blocked = load_gemm_parquet(&[
            PerfSource(primary.clone(), None),
            PerfSource(
                sibling.clone(),
                Some(vec!["__no_such_kernel_source__".to_string()]),
            ),
        ])
        .unwrap();
        assert_eq!(
            gemm_shape_count(&blocked),
            gemm_shape_count(&primary_only),
            "a non-matching kernel_source filter must exclude all sibling rows"
        );
    }

    #[test]
    fn gemm_exact_hit_returns_recorded_latency() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        // First row of b200_sxm/gemm/vllm/0.19.0/gemm_perf.parquet
        // (bfloat16 32768x65536x16384).
        let latency = table
            .query(GemmQuantMode::Bfloat16, 32768, 65536, 16384)
            .expect("query must succeed")
            .latency;
        assert!(
            (latency - 41.59673055013021).abs() < 1e-9,
            "expected recorded latency, got {latency}"
        );
    }

    #[test]
    fn gemm_query_returns_positive_latency_for_smoke_shape() {
        // Shape pulled from a MiniMax-M2.5 GEMM call: tp=8 ffn1 at hidden=6144.
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let latency = table
            .query(GemmQuantMode::Bfloat16, 1024, 6144, 6144)
            .expect("query must succeed")
            .latency;
        assert!(latency > 0.0, "interpolated latency must be positive");
        assert!(latency < 100.0, "shape this small shouldn't take 100ms");
    }

    #[test]
    fn gemm_lazy_loads_on_first_query_only() {
        // Same data root, two queries — second must not re-read the CSV.
        // We can't directly observe I/O count, but if the cache isn't being
        // hit the second query would still succeed, so verify both paths
        // return identical results (proxy for cache stability).
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let first = table
            .query(GemmQuantMode::Bfloat16, 32768, 65536, 16384)
            .unwrap();
        let second = table
            .query(GemmQuantMode::Bfloat16, 32768, 65536, 16384)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn gemm_query_cache_is_bit_identical_for_all_resolution_classes() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let cases = [
            (256, 32, 32),
            (259, 32, 32),
            (10_000_000, 32, 32),
            (256, 128, 96),
        ];

        for (m, n, k) in cases {
            let uncached = table.query(GemmQuantMode::Bfloat16, m, n, k).unwrap();
            let cached = table.query(GemmQuantMode::Bfloat16, m, n, k).unwrap();
            assert_eq!(
                leaf_bits(cached),
                leaf_bits(uncached),
                "cache changed ({m},{n},{k})"
            );
        }

        assert_eq!(query_cache_len(&table), cases.len());
    }

    #[test]
    fn gemm_query_cache_normalizes_quant_and_separates_shape_fields() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());

        let fp8 = table.query(GemmQuantMode::Fp8, 256, 32, 32).unwrap();
        let fp8_static = table.query(GemmQuantMode::Fp8Static, 256, 32, 32).unwrap();
        assert_eq!(leaf_bits(fp8), leaf_bits(fp8_static));
        assert_eq!(query_cache_len(&table), 1);

        for (quant, m, n, k) in [
            (GemmQuantMode::Bfloat16, 256, 32, 32),
            (GemmQuantMode::Bfloat16, 257, 32, 32),
            (GemmQuantMode::Bfloat16, 256, 64, 32),
            (GemmQuantMode::Bfloat16, 256, 32, 64),
        ] {
            table.query(quant, m, n, k).unwrap();
        }
        assert_eq!(query_cache_len(&table), 5);
    }

    #[test]
    fn gemm_query_errors_never_enter_the_cache() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        for _ in 0..2 {
            assert!(table
                .query(GemmQuantMode::Int4Wo, 1024, 4096, 4096)
                .is_err());
        }
        assert_eq!(query_cache_len(&table), 0);

        let missing = GemmTable::new(PathBuf::from("/nonexistent/aic/data/root"), b200_sxm_spec());
        for _ in 0..2 {
            assert!(missing.query(GemmQuantMode::Bfloat16, 1, 1, 1).is_err());
        }
        assert_eq!(query_cache_len(&missing), 0);
    }

    #[test]
    fn gemm_query_cache_enforces_production_per_shard_bound() {
        let cache = gemm_query_cache();
        let key = |m| GemmQueryKey {
            quant: GemmQuantMode::Bfloat16,
            m,
            n: 32,
            k: 32,
        };
        let per_shard_capacity = GEMM_QUERY_CACHE_CAPACITY / GEMM_QUERY_CACHE_SHARDS;

        assert_eq!(per_shard_capacity, 2_048);
        assert_eq!(cache.num_shards(), GEMM_QUERY_CACHE_SHARDS);
        assert_eq!(cache.shard_capacity(), per_shard_capacity as u64);
        assert_eq!(cache.capacity(), GEMM_QUERY_CACHE_CAPACITY as u64);

        let target_shard = cache.shard_index(&key(0));
        let keys: Vec<_> = (0..)
            .map(key)
            .filter(|key| cache.shard_index(key) == target_shard)
            .take(per_shard_capacity + 1)
            .collect();
        assert_eq!(keys.len(), 2_049);

        for (value, key) in keys.into_iter().enumerate() {
            cache.insert(key, LeafValue::latency_only(value as f64));
        }

        assert_eq!(cache.len(), per_shard_capacity);
        assert!(cache.len() < GEMM_QUERY_CACHE_CAPACITY);
    }

    #[test]
    fn gemm_query_cache_allows_concurrent_duplicate_misses() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};

        const THREADS: usize = 4;
        const EXPECTED: f64 = 1.25;

        let cache = Arc::new(gemm_query_cache());
        let barrier = Arc::new(Barrier::new(THREADS));
        let resolutions = Arc::new(AtomicUsize::new(0));
        let key = GemmQueryKey {
            quant: GemmQuantMode::Bfloat16,
            m: 259,
            n: 32,
            k: 32,
        };
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let barrier = Arc::clone(&barrier);
                let resolutions = Arc::clone(&resolutions);
                std::thread::spawn(move || {
                    resolve_gemm_query_cached(&cache, key, || {
                        resolutions.fetch_add(1, Ordering::Relaxed);
                        barrier.wait();
                        Ok(LeafValue::latency_only(EXPECTED))
                    })
                    .unwrap()
                })
            })
            .collect();
        let values: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        assert!(values
            .windows(2)
            .all(|pair| leaf_bits(pair[0]) == leaf_bits(pair[1])));
        assert_eq!(resolutions.load(Ordering::Relaxed), THREADS);
        assert_eq!(cache.len(), 1);

        let cached = resolve_gemm_query_cached(&cache, key, || -> Result<LeafValue, AicError> {
            panic!("cache hit unexpectedly invoked the resolver")
        })
        .unwrap();
        assert_eq!(
            leaf_bits(cached),
            leaf_bits(LeafValue::latency_only(EXPECTED))
        );
    }

    /// Values generated from the Python v2 engine on the same table
    /// (`db.query_gemm(..., SILICON)` on b200_sxm/vllm/0.19.0, bfloat16):
    /// exact hit, m-interp on a collected (n,k) site, m util-hold beyond the
    /// sweep, and an unknown (n,k) site via neighbour util transfer. The two
    /// engines must agree because they implement the same resolution chain.
    // NOTE(shared-layer merge): oracle generated pre-shared-layer; regenerate if this fails
    #[test]
    fn gemm_query_matches_python_v2_engine() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let q = GemmQuantMode::Bfloat16;
        let cases: &[(u32, u32, u32, f64)] = &[
            (256, 32, 32, 0.00186666660011),
            (259, 32, 32, 0.00184757819233),
            (10_000_000, 32, 32, 1.51111355145),
            (256, 128, 96, 0.00187964537818),
        ];
        for &(m, n, k, expected) in cases {
            let got = table.query(q, m, n, k).unwrap().latency;
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "({m},{n},{k}): rust {got} vs python {expected}"
            );
        }
    }

    #[test]
    fn gemm_missing_quant_mode_errors() {
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        // vLLM 0.19.0 b200 collects bfloat16/fp8/fp8_block/nvfp4 — int4_wo
        // is genuinely absent for this slice.
        match table.query(GemmQuantMode::Int4Wo, 1024, 4096, 4096) {
            Err(AicError::PerfDatabase(msg)) => {
                assert!(
                    msg.contains("int4_wo"),
                    "expected quant name in error: {msg}"
                );
            }
            other => panic!("expected PerfDatabase error, got {other:?}"),
        }
    }

    #[test]
    fn gemm_missing_data_root_errors_on_query() {
        let table = GemmTable::new(PathBuf::from("/nonexistent/aic/data/root"), b200_sxm_spec());
        let err = table.query(GemmQuantMode::Bfloat16, 1, 1, 1).unwrap_err();
        // The lazy loader should surface the missing file as the cause,
        // and the second access should see the cached error too.
        assert!(matches!(err, AicError::PerfDatabase(_)));
        let err2 = table.query(GemmQuantMode::Bfloat16, 1, 1, 1).unwrap_err();
        assert!(matches!(err2, AicError::PerfDatabase(_)));
    }

    #[test]
    fn compute_scale_absent_on_vllm_b200_errors_clearly() {
        // vLLM doesn't ship compute_scale data on b200; expect a clear IO error.
        let table = GemmTable::new(b200_vllm_data_root(), b200_sxm_spec());
        let err = table
            .query_compute_scale(GemmQuantMode::Fp8Static, 1024, 4096)
            .unwrap_err();
        match err {
            AicError::Io { .. } | AicError::PerfDatabase(_) => {}
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// Mirror of Python `test_gemm_op.py::TestStaticHelpers`: dtype-keyed
    /// resolution (sq -> int8, weight-only -> bf16), strict missing-key
    /// error, and the b300 fp4 entry beating the old 4x-bf16 extrapolation.
    #[test]
    fn quant_tc_flops_resolves_by_compute_dtype() {
        use crate::common::system_spec::{GpuSpec, MiscSpec, NodeSpec, SystemSpec};
        let mut spec = SystemSpec {
            data_dir: std::path::PathBuf::from("data/synthetic"),
            gpu: GpuSpec {
                mem_bw: 1.0,
                mem_bw_empirical_scaling_factor: 1.0,
                mem_empirical_constant_latency: 0.0,
                mem_capacity: None,
                bfloat16_tc_flops: Some(1000.0),
                int8_tc_flops: Some(30.0),
                fp8_tc_flops: Some(2000.0),
                fp4_tc_flops: None,
                power: None,
                sm_version: None,
            },
            node: NodeSpec {
                num_gpus_per_node: 8,
                intra_node_bw: 900.0,
                inter_node_bw: 100.0,
                pcie_bw: None,
                p2p_latency: 0.0,
                num_gpus_per_rack: None,
                inter_rack_bw: None,
            },
            misc: MiscSpec::default(),
        };

        use crate::common::enums::GemmQuantMode as Q;
        assert_eq!(
            quant_tc_flops(&spec, Q::Bfloat16.mapping()).unwrap(),
            1000.0
        );
        assert_eq!(quant_tc_flops(&spec, Q::Fp8.mapping()).unwrap(), 2000.0);
        // sq runs on the int8 pipeline, NOT fp8's (b300: int8 != fp8).
        assert_eq!(quant_tc_flops(&spec, Q::Sq.mapping()).unwrap(), 30.0);
        // weight-only modes dequantize to bf16 before the MMA.
        assert_eq!(quant_tc_flops(&spec, Q::Int8Wo.mapping()).unwrap(), 1000.0);

        // Missing fp4 entry: strict error, never bf16 * 4.
        match quant_tc_flops(&spec, Q::Nvfp4.mapping()) {
            Err(AicError::MissingSystemFlops(msg)) => assert!(msg.contains("fp4_tc_flops")),
            other => panic!("expected MissingSystemFlops, got {other:?}"),
        }

        // Memory-only modes have no compute pipeline.
        use crate::common::enums::KvCacheQuantMode;
        match quant_tc_flops(&spec, KvCacheQuantMode::Fp8.mapping()) {
            Err(AicError::MissingSystemFlops(msg)) => assert!(msg.contains("memory-only")),
            other => panic!("expected MissingSystemFlops, got {other:?}"),
        }

        // Non-finite entries (e.g. YAML `.inf`) are placeholders/typos: +inf
        // would zero sol_math and silently collapse SOL onto the memory roof.
        spec.gpu.fp4_tc_flops = Some(f64::INFINITY);
        assert!(matches!(
            quant_tc_flops(&spec, Q::Nvfp4.mapping()),
            Err(AicError::MissingSystemFlops(_))
        ));

        // b300 breaks the fixed 4x ratio: the YAML entry must win.
        spec.gpu.fp4_tc_flops = Some(1.4e16);
        assert_eq!(quant_tc_flops(&spec, Q::Nvfp4.mapping()).unwrap(), 1.4e16);
    }

    /// ENERGY oracle on a synthetic power-carrying fixture (the shipped
    /// tables have no `power` column). Python twin (pandas fixture with the
    /// SAME rows + the `testsys.yaml` spec of `energy_test_fixtures`):
    ///
    /// ```text
    /// db.query_gemm(192, 1024, 1024, GEMMQuantMode.bfloat16, SILICON)
    /// # -> latency=2.0, energy=300.0
    /// ```
    ///
    /// m=192 lerps the (n=1024, k=1024) site curve between (m=128, lat 1.0,
    /// power 100) and (m=256, lat 3.0, power 200): POWER blends to 150 and
    /// energy re-derives as 150 * 2.0 (a naive energy-lerp would give 350).
    #[test]
    fn gemm_energy_blend_matches_python_oracle() {
        use crate::perf_database::energy_test_fixtures::{energy_test_spec, write_parquet, Col};
        let tmp = tempfile::tempdir().expect("tmpdir");
        write_parquet(
            &tmp.path().join("gemm_perf.parquet"),
            &[
                Col::Str("gemm_dtype", vec!["bfloat16", "bfloat16"]),
                Col::I64("m", vec![128, 256]),
                Col::I64("n", vec![1024, 1024]),
                Col::I64("k", vec![1024, 1024]),
                Col::F64("latency", vec![1.0, 3.0]),
                Col::F64("power", vec![100.0, 200.0]),
            ],
        );
        let table = GemmTable::new(tmp.path().to_path_buf(), energy_test_spec());
        let v = table
            .query(GemmQuantMode::Bfloat16, 192, 1024, 1024)
            .unwrap();
        assert!((v.latency - 2.0).abs() < 1e-9, "latency {}", v.latency);
        assert!(
            (v.energy - 300.0).abs() < 1e-9 * 300.0,
            "energy {}",
            v.energy
        );
        assert!((v.power - 150.0).abs() < 1e-9 * 150.0, "power {}", v.power);
    }
}
