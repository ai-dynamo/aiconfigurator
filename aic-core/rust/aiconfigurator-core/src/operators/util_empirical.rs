// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data-calibrated empirical estimation via SOL-utilization.
//!
//! Mirrors `aiconfigurator.sdk.operations.util_empirical`: each op's
//! empirical estimate is `latency = SOL(query) / util` where
//! `util = SOL / measured > 0` is read best-effort from collected samples in
//! per-axis normalised log space. `util` is an effective calibration factor,
//! not a bounded physical efficiency (it may exceed 1); it is never clamped.
//! Every grid uses the same two-neighbour inverse-distance weighting
//! (`k=2`, `p=1`) without requiring a Cartesian product; queries outside the
//! measured range are clamped per axis before neighbour selection, so
//! extrapolation freezes boundary utilization.
//!
//! When *no* samples exist for the requested slice (no own-shape, no
//! cross-shape/sibling transfer reference), [`estimate`] returns
//! [`AicError::EmpiricalNotImplemented`] rather than a fabricated
//! `SOL / constant` — coverage gaps surface honestly, exactly like Python's
//! `EmpiricalNotImplementedError`. Genuinely table-less ops (mem / p2p /
//! element-wise) keep their analytic formulas and never call [`estimate`].
//!
//! Divergences from the Python module, by design:
//! - No provenance contextvar: provenance capture feeds the Python-side
//!   support matrix; the compiled engine only returns latencies. Reference
//!   grids still carry their provenance tag for cache keying.
//! - Caching is the caller's concern: Python keys grids by `id(node)` because
//!   database views share mutable table objects; Rust perf tables are
//!   immutable after load, so per-op wiring caches grids in a
//!   [`UtilGridCache`] keyed by the op's slice identity.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::common::error::AicError;

/// Empirical provenance tiers, mirroring Python's `PROVENANCE_ORDER`
/// (`sdk/operations/util_empirical.py`): ordered by DECREASING confidence,
/// so the max rank fired during a run is the run's effective data source
/// (Python `worst_provenance`). `Silicon` is the default when nothing fired;
/// operators never note it.
///
/// The accumulation cell lives on `PerfDatabase` (shared across mode views);
/// operators call `PerfDatabase::note_provenance` at the same sites Python
/// calls `note_provenance` — after a successful `estimate` with the tier the
/// call site knows (Python passes it as `estimate`'s `provenance` param).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ProvenanceTier {
    /// Pure silicon table data (default; never recorded by operators).
    Silicon = 0,
    /// Own-shape util (no transfer).
    Empirical = 1,
    /// Cross-shape, same quant.
    XShape = 2,
    /// Cross-quant, same profile.
    XQuant = 3,
    /// Cross-quant, cross profile.
    XProfile = 4,
    /// Cross-op (borrowed a different op's util).
    XOp = 5,
}

impl ProvenanceTier {
    /// The Python tag string (`PROVENANCE_ORDER` spelling) for this tier.
    pub fn as_str(self) -> &'static str {
        match self {
            ProvenanceTier::Silicon => "silicon",
            ProvenanceTier::Empirical => "empirical",
            ProvenanceTier::XShape => "xshape",
            ProvenanceTier::XQuant => "xquant",
            ProvenanceTier::XProfile => "xprofile",
            ProvenanceTier::XOp => "xop",
        }
    }

    /// Inverse of [`Self::as_str`] for call sites that carry the Python tag
    /// string (e.g. a reference grid's `reference_provenance`). Unknown tags
    /// yield `None` so callers choose their own default tier.
    pub fn from_tag(tag: &str) -> Option<ProvenanceTier> {
        match tag {
            "silicon" => Some(ProvenanceTier::Silicon),
            "empirical" => Some(ProvenanceTier::Empirical),
            "xshape" => Some(ProvenanceTier::XShape),
            "xquant" => Some(ProvenanceTier::XQuant),
            "xprofile" => Some(ProvenanceTier::XProfile),
            "xop" => Some(ProvenanceTier::XOp),
            _ => None,
        }
    }

    /// Inverse of `tier as u8` for reading the accumulation cell back.
    /// Out-of-range ranks clamp to the least-confident tier.
    pub fn from_rank(rank: u8) -> ProvenanceTier {
        match rank {
            0 => ProvenanceTier::Silicon,
            1 => ProvenanceTier::Empirical,
            2 => ProvenanceTier::XShape,
            3 => ProvenanceTier::XQuant,
            4 => ProvenanceTier::XProfile,
            _ => ProvenanceTier::XOp,
        }
    }
}

/// One collected calibration point: continuous-axis coordinates plus the
/// positive effective calibration factor `util = SOL / measured`.
#[derive(Clone, Debug, PartialEq)]
pub struct UtilSample {
    pub coords: Vec<f64>,
    pub util: f64,
}

impl UtilSample {
    pub fn new(coords: Vec<f64>, util: f64) -> Self {
        Self { coords, util }
    }
}

/// Build util samples from `(coords, latency_ms)` points and an analytic SOL.
///
/// Mirrors Python `build_samples`: a point is kept only when both the
/// measured latency and its SOL are strictly positive (NaN fails both
/// comparisons and is dropped, matching Python truthiness + `> 0`).
pub fn build_samples<I, F>(points: I, sol_fn: F) -> Vec<UtilSample>
where
    I: IntoIterator<Item = (Vec<f64>, f64)>,
    F: Fn(&[f64]) -> f64,
{
    let mut samples = Vec::new();
    for (coords, latency_ms) in points {
        if latency_ms > 0.0 {
            let sol = sol_fn(&coords);
            if sol > 0.0 {
                samples.push(UtilSample::new(coords, sol / latency_ms));
            }
        }
    }
    samples
}

/// Two-neighbour util lookup in per-axis normalised log space.
///
/// The query is clamped independently on every axis, then the two nearest
/// samples are combined with inverse-distance weights (`k=2`, `p=1`). Exact
/// hits return the collected utilization unchanged. Works for ragged grids
/// without operation-specific Cartesian bracketing; callers remain
/// responsible for slicing categorical/kernel-regime axes.
#[derive(Debug, Clone)]
pub struct UtilGrid {
    /// Normalised log-space coordinates, one row per sample.
    norm: Vec<Vec<f64>>,
    utils: Vec<f64>,
    mins: Vec<f64>,
    spans: Vec<f64>,
    /// Transfer tag of the reference slice this grid was built from
    /// (`xshape` / `xquant` / ...), when borrowed from a sibling.
    pub reference_provenance: Option<&'static str>,
}

fn log_floor(value: f64) -> f64 {
    value.max(1e-9).ln()
}

impl UtilGrid {
    pub fn new(samples: Vec<UtilSample>) -> Self {
        if samples.is_empty() {
            return Self {
                norm: Vec::new(),
                utils: Vec::new(),
                mins: Vec::new(),
                spans: Vec::new(),
                reference_provenance: None,
            };
        }
        let dims = samples[0].coords.len();
        let logc: Vec<Vec<f64>> = samples
            .iter()
            .map(|s| s.coords.iter().map(|&c| log_floor(c)).collect())
            .collect();
        let mut mins = vec![f64::INFINITY; dims];
        let mut maxs = vec![f64::NEG_INFINITY; dims];
        for row in &logc {
            for (a, &v) in row.iter().enumerate() {
                mins[a] = mins[a].min(v);
                maxs[a] = maxs[a].max(v);
            }
        }
        let spans: Vec<f64> = mins
            .iter()
            .zip(&maxs)
            .map(|(&lo, &hi)| if hi - lo > 0.0 { hi - lo } else { 1.0 })
            .collect();
        let norm: Vec<Vec<f64>> = logc
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(a, &v)| (v - mins[a]) / spans[a])
                    .collect()
            })
            .collect();
        let utils = samples.iter().map(|s| s.util).collect();
        Self {
            norm,
            utils,
            mins,
            spans,
            reference_provenance: None,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.utils.is_empty()
    }

    /// Interpolated utilization at `query`, or `None` for an empty grid.
    pub fn util(&self, query: &[f64]) -> Option<f64> {
        if self.utils.is_empty() {
            return None;
        }
        // Per-axis clamp to [0, 1] freezes boundary utilization for
        // out-of-range queries (mirrors `np.clip`).
        let q: Vec<f64> = query
            .iter()
            .enumerate()
            .map(|(a, &v)| ((log_floor(v) - self.mins[a]) / self.spans[a]).clamp(0.0, 1.0))
            .collect();
        let distances: Vec<f64> = self
            .norm
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&q)
                    .map(|(&x, &y)| (x - y) * (x - y))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        // Stable argsort: ties keep sample order, so duplicate /
        // log-floor-collapsed coordinates deterministically prefer the first
        // sample (mirrors `np.argsort(kind="stable")`).
        let mut order: Vec<usize> = (0..distances.len()).collect();
        order.sort_by(|&i, &j| distances[i].partial_cmp(&distances[j]).expect("finite distances"));

        if distances[order[0]] == 0.0 {
            return Some(self.utils[order[0]]);
        }

        let nearest = &order[..order.len().min(2)];
        let mut weighted = 0.0;
        let mut weight_sum = 0.0;
        for &i in nearest {
            let w = 1.0 / distances[i];
            weighted += self.utils[i] * w;
            weight_sum += w;
        }
        Some(weighted / weight_sum)
    }
}

/// Return `(latency_ms, util)` from the util grid, or the typed coverage
/// error.
///
/// Mirrors Python `estimate`: `None`, empty grids, and non-positive utils all
/// surface as [`AicError::EmpiricalNotImplemented`] — there is no own-shape,
/// cross-shape, or sibling data to calibrate from, so the gap surfaces
/// instead of inventing a `SOL / constant` placeholder.
///
/// `util_scale` is the cross-op level-alignment hook (1.0 = no change). When
/// a CROSS-OP transfer borrows a *different* op's util grid, the caller
/// passes a manual scale `k` so `latency = SOL / (util * k)`.
pub fn estimate(
    sol_query: f64,
    query: &[f64],
    grid: Option<&UtilGrid>,
    util_scale: f64,
) -> Result<(f64, f64), AicError> {
    if let Some(util) = grid.and_then(|g| g.util(query)) {
        if util > 0.0 {
            return Ok((sol_query / (util * util_scale), util));
        }
    }
    Err(AicError::EmpiricalNotImplemented(format!(
        "No empirical utilisation data to estimate this op at query={query:?}: \
         no own-shape, cross-shape, or sibling transfer reference available."
    )))
}

/// Nearest reference index by categorical shape features in per-dim
/// normalised log space (mirrors Python `_nearest_candidate`; ties keep the
/// first candidate, matching `np.argmin`). Returns `None` for an empty list.
pub fn nearest_candidate_index(query_features: &[f64], candidates: &[Vec<f64>]) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let dims = query_features.len();
    let feats: Vec<Vec<f64>> = candidates
        .iter()
        .map(|c| c.iter().map(|&v| log_floor(v)).collect())
        .collect();
    let mut mins = vec![f64::INFINITY; dims];
    let mut maxs = vec![f64::NEG_INFINITY; dims];
    for row in &feats {
        for (a, &v) in row.iter().enumerate() {
            mins[a] = mins[a].min(v);
            maxs[a] = maxs[a].max(v);
        }
    }
    let spans: Vec<f64> = mins
        .iter()
        .zip(&maxs)
        .map(|(&lo, &hi)| if hi - lo > 0.0 { hi - lo } else { 1.0 })
        .collect();
    // NOTE: the query is intentionally NOT clamped here (unlike UtilGrid) —
    // Python normalises the query into the candidates' span without clipping.
    let q: Vec<f64> = query_features
        .iter()
        .enumerate()
        .map(|(a, &v)| (log_floor(v) - mins[a]) / spans[a])
        .collect();
    let mut best = 0;
    let mut best_dist2 = f64::INFINITY;
    for (i, row) in feats.iter().enumerate() {
        let dist2: f64 = row
            .iter()
            .enumerate()
            .map(|(a, &v)| {
                let n = (v - mins[a]) / spans[a];
                (n - q[a]) * (n - q[a])
            })
            .sum();
        if dist2 < best_dist2 {
            best_dist2 = dist2;
            best = i;
        }
    }
    Some(best)
}

/// Process-lifetime memo of built util grids, keyed by the caller's slice
/// identity. Rust perf tables are immutable after load and each
/// `PerfDatabase` owns its cache, so a plain keyed map replaces Python's
/// `id(node)`-qualified module cache.
#[derive(Debug, Default)]
pub struct UtilGridCache {
    grids: Mutex<HashMap<String, Option<Arc<UtilGrid>>>>,
}

impl UtilGridCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetch or build the grid for `key`.
    ///
    /// `builder` mirrors Python's `grid_for` contract: `Ok(None)` when the
    /// slice has no usable calibration data (a typed coverage miss — memoised,
    /// the caller then raises via [`estimate`] with `grid=None`), `Err` for
    /// programming/schema errors (propagated, NOT memoised, never converted
    /// into a fallback).
    pub fn get_or_try_build<F>(&self, key: &str, builder: F) -> Result<Option<Arc<UtilGrid>>, AicError>
    where
        F: FnOnce() -> Result<Option<UtilGrid>, AicError>,
    {
        let mut grids = self.grids.lock().expect("util grid cache poisoned");
        if let Some(cached) = grids.get(key) {
            return Ok(cached.clone());
        }
        let built = builder()?.map(Arc::new);
        grids.insert(key.to_string(), built.clone());
        Ok(built)
    }
}

/// Nearest-point lookup over a non-negative latency *delta* table (the
/// `compute_scale` mechanism; Python `gemm._ZeroAwareDeltaLookup`).
///
/// `compute_scale` stores `max(dynamic_quant - static_quant, 0)`: zero is a
/// measured, meaningful delta, not a missing latency. A normal util grid
/// cannot represent it (`SOL / 0`), and dropping zeroes can make one positive
/// noise sample the reference for the whole table. Select the nearest point
/// on the complete 2-D grid first: a selected zero stays zero; a positive
/// point uses frozen utilization so extrapolation scales with the query's
/// amount of work.
#[derive(Debug)]
pub struct ZeroAwareDeltaLookup {
    coords: Vec<Vec<f64>>,
    latencies: Vec<f64>,
    mins: Vec<f64>,
    spans: Vec<f64>,
    norm: Vec<Vec<f64>>,
}

impl ZeroAwareDeltaLookup {
    /// Keep every point with `latency >= 0` (zero INCLUDED, unlike
    /// [`build_samples`]).
    pub fn new(points: Vec<(Vec<f64>, f64)>) -> Self {
        let kept: Vec<(Vec<f64>, f64)> = points.into_iter().filter(|(_, lat)| *lat >= 0.0).collect();
        if kept.is_empty() {
            return Self {
                coords: Vec::new(),
                latencies: Vec::new(),
                mins: Vec::new(),
                spans: Vec::new(),
                norm: Vec::new(),
            };
        }
        let dims = kept[0].0.len();
        let logc: Vec<Vec<f64>> = kept
            .iter()
            .map(|(c, _)| c.iter().map(|&v| log_floor(v)).collect())
            .collect();
        let mut mins = vec![f64::INFINITY; dims];
        let mut maxs = vec![f64::NEG_INFINITY; dims];
        for row in &logc {
            for (a, &v) in row.iter().enumerate() {
                mins[a] = mins[a].min(v);
                maxs[a] = maxs[a].max(v);
            }
        }
        let spans: Vec<f64> = mins
            .iter()
            .zip(&maxs)
            .map(|(&lo, &hi)| if hi - lo > 0.0 { hi - lo } else { 1.0 })
            .collect();
        let norm: Vec<Vec<f64>> = logc
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(a, &v)| (v - mins[a]) / spans[a])
                    .collect()
            })
            .collect();
        Self {
            latencies: kept.iter().map(|(_, lat)| *lat).collect(),
            coords: kept.into_iter().map(|(c, _)| c).collect(),
            mins,
            spans,
            norm,
        }
    }

    /// Nearest-point delta estimate (query is NOT clamped; the frozen-util
    /// rescale `query_sol / (ref_sol / ref_latency)` carries extrapolation).
    pub fn estimate<F>(&self, query: &[f64], sol_fn: F) -> Result<f64, AicError>
    where
        F: Fn(&[f64]) -> f64,
    {
        if self.latencies.is_empty() {
            return Err(AicError::EmpiricalNotImplemented(format!(
                "No empirical compute_scale delta data is available at query={query:?}."
            )));
        }
        let q: Vec<f64> = query
            .iter()
            .enumerate()
            .map(|(a, &v)| (log_floor(v) - self.mins[a]) / self.spans[a])
            .collect();
        let mut best = 0;
        let mut best_dist2 = f64::INFINITY;
        for (i, row) in self.norm.iter().enumerate() {
            let dist2: f64 = row.iter().zip(&q).map(|(&x, &y)| (x - y) * (x - y)).sum();
            if dist2 < best_dist2 {
                best_dist2 = dist2;
                best = i;
            }
        }
        let reference_latency = self.latencies[best];
        if reference_latency == 0.0 {
            return Ok(0.0);
        }
        let reference_sol = sol_fn(&self.coords[best]);
        let query_sol = sol_fn(query);
        if reference_sol <= 0.0 || query_sol <= 0.0 {
            return Err(AicError::EmpiricalNotImplemented(format!(
                "No positive SOL reference is available for compute_scale at query={query:?}."
            )));
        }
        Ok(query_sol / (reference_sol / reference_latency))
    }
}

/// Memo of built [`ZeroAwareDeltaLookup`]s (same keying/lifetime rationale as
/// [`UtilGridCache`]).
#[derive(Debug, Default)]
pub struct DeltaLookupCache {
    lookups: Mutex<HashMap<String, Arc<ZeroAwareDeltaLookup>>>,
}

impl DeltaLookupCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_try_build<F>(&self, key: &str, builder: F) -> Result<Arc<ZeroAwareDeltaLookup>, AicError>
    where
        F: FnOnce() -> Result<ZeroAwareDeltaLookup, AicError>,
    {
        let mut lookups = self.lookups.lock().expect("delta lookup cache poisoned");
        if let Some(cached) = lookups.get(key) {
            return Ok(cached.clone());
        }
        let built = Arc::new(builder()?);
        lookups.insert(key.to_string(), built.clone());
        Ok(built)
    }
}

#[cfg(test)]
mod tests {
    //! Mirrors the math anchors of `tests/unit/sdk/test_util_empirical.py`.
    //! The Python cache/`grid_for` contract tests are id()-specific and are
    //! covered by `UtilGridCache`'s own semantics instead.

    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
    }

    #[test]
    fn exact_singleton_duplicate_and_empty_grid_contracts() {
        let exact = UtilGrid::new(vec![
            UtilSample::new(vec![16.0], 0.8),
            UtilSample::new(vec![8.0], 0.2),
            UtilSample::new(vec![9.0], 0.4),
        ]);
        let duplicate = UtilGrid::new(vec![
            UtilSample::new(vec![4.0], 0.6),
            UtilSample::new(vec![4.0], 0.7),
        ]);

        approx(exact.util(&[9.0]).unwrap(), 0.4);
        approx(
            UtilGrid::new(vec![UtilSample::new(vec![0.0], 0.3)])
                .util(&[100.0])
                .unwrap(),
            0.3,
        );
        // Duplicate coordinates: first sample wins (stable ordering).
        approx(duplicate.util(&[4.0]).unwrap(), 0.6);
        assert!(UtilGrid::new(vec![]).util(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn one_dim_k2_idw_uses_nearest_samples_in_normalized_log_space() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![16.0], 0.8),
            UtilSample::new(vec![8.0], 0.2),
            UtilSample::new(vec![9.0], 0.4),
        ]);
        let distance_9 = 11.0_f64.ln() - 9.0_f64.ln();
        let distance_8 = 11.0_f64.ln() - 8.0_f64.ln();
        let expected = (0.4 / distance_9 + 0.2 / distance_8) / (1.0 / distance_9 + 1.0 / distance_8);

        approx(grid.util(&[11.0]).unwrap(), expected);
    }

    #[test]
    fn multidimensional_k2_idw_uses_nearest_samples() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![1.0, 1.0], 0.2),
            UtilSample::new(vec![100.0, 1.0], 0.6),
            UtilSample::new(vec![100.0, 100.0], 1.0),
        ]);

        // (10, 1) is equidistant from the first two normalized-log samples.
        approx(grid.util(&[10.0, 1.0]).unwrap(), 0.4);
    }

    #[test]
    fn one_dim_extrapolation_clamps_to_measured_bounds() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![8.0], 0.2),
            UtilSample::new(vec![16.0], 0.8),
        ]);

        approx(grid.util(&[1.0]).unwrap(), 0.2);
        approx(grid.util(&[128.0]).unwrap(), 0.8);
    }

    #[test]
    fn multidimensional_extrapolation_clamps_each_axis() {
        let grid = UtilGrid::new(vec![
            UtilSample::new(vec![1.0, 1.0], 0.2),
            UtilSample::new(vec![1.0, 10.0], 0.4),
            UtilSample::new(vec![10.0, 1.0], 0.8),
        ]);

        // Clamping (0.1, 100) produces the exact measured boundary (1, 10).
        approx(grid.util(&[0.1, 100.0]).unwrap(), 0.4);
    }

    #[test]
    fn build_samples_filters_non_positive_latency_and_sol() {
        let samples = build_samples(
            vec![
                (vec![2.0], 4.0),  // kept: util = sol/lat = 2/4
                (vec![3.0], 0.0),  // dropped: latency <= 0
                (vec![4.0], -1.0), // dropped: latency < 0
                (vec![0.5], 1.0),  // dropped: sol_fn returns 0 below 1.0
            ],
            |coords| if coords[0] >= 1.0 { coords[0] } else { 0.0 },
        );
        assert_eq!(samples.len(), 1);
        approx(samples[0].util, 0.5);
    }

    #[test]
    fn estimate_returns_latency_and_util_or_typed_miss() {
        let grid = UtilGrid::new(vec![UtilSample::new(vec![1.0], 0.5)]);
        let (latency, util) = estimate(1.0, &[1.0], Some(&grid), 1.0).unwrap();
        approx(latency, 2.0);
        approx(util, 0.5);

        // Cross-op level alignment: latency = SOL / (util * k).
        let (latency, _) = estimate(1.0, &[1.0], Some(&grid), 2.0).unwrap();
        approx(latency, 1.0);

        let missing = estimate(1.0, &[1.0], None, 1.0);
        assert!(matches!(missing, Err(AicError::EmpiricalNotImplemented(_))));
        let empty = UtilGrid::new(vec![]);
        let empty_res = estimate(1.0, &[1.0], Some(&empty), 1.0);
        assert!(matches!(empty_res, Err(AicError::EmpiricalNotImplemented(_))));
    }

    #[test]
    fn nearest_candidate_matches_python_normalised_log_selection() {
        // Query (90,) between features (1,) and (100,): log-nearer to 100.
        let idx = nearest_candidate_index(&[90.0], &[vec![1.0], vec![100.0]]);
        assert_eq!(idx, Some(1));
        // Single candidate always selected; empty list yields None.
        assert_eq!(nearest_candidate_index(&[5.0], &[vec![1.0]]), Some(0));
        assert_eq!(nearest_candidate_index(&[5.0], &[]), None);
    }

    #[test]
    fn util_grid_cache_memoises_hits_and_misses_but_not_errors() {
        let cache = UtilGridCache::new();
        let mut builds = 0;
        let grid = cache.get_or_try_build("k", || {
            builds += 1;
            Ok(Some(UtilGrid::new(vec![UtilSample::new(vec![1.0], 0.5)])))
        });
        assert!(grid.unwrap().is_some());
        let again = cache.get_or_try_build("k", || {
            builds += 1;
            Ok(None)
        });
        assert!(again.unwrap().is_some());
        assert_eq!(builds, 1);

        // Typed coverage miss (Ok(None)) is memoised, like Python's grid_for.
        let miss = cache.get_or_try_build("missing", || Ok(None));
        assert!(miss.unwrap().is_none());
        let miss_again =
            cache.get_or_try_build("missing", || panic!("memoised miss must not rebuild"));
        assert!(miss_again.unwrap().is_none());

        // Programming/schema errors propagate and are NOT memoised.
        let err = cache.get_or_try_build("broken", || {
            Err(AicError::PerfDatabase("schema".to_string()))
        });
        assert!(err.is_err());
        let recovered = cache.get_or_try_build("broken", || Ok(None));
        assert!(recovered.unwrap().is_none());
    }

    #[test]
    fn provenance_tier_mirrors_python_provenance_order() {
        // Rank order == PROVENANCE_ORDER index; tag strings match Python.
        let tiers = [
            (ProvenanceTier::Silicon, 0u8, "silicon"),
            (ProvenanceTier::Empirical, 1, "empirical"),
            (ProvenanceTier::XShape, 2, "xshape"),
            (ProvenanceTier::XQuant, 3, "xquant"),
            (ProvenanceTier::XProfile, 4, "xprofile"),
            (ProvenanceTier::XOp, 5, "xop"),
        ];
        for (tier, rank, tag) in tiers {
            assert_eq!(tier as u8, rank);
            assert_eq!(tier.as_str(), tag);
            assert_eq!(ProvenanceTier::from_rank(rank), tier);
            assert_eq!(ProvenanceTier::from_tag(tag), Some(tier));
        }
        assert_eq!(ProvenanceTier::from_tag("unknown"), None);
        // max-rank == worst_provenance semantics; overflow clamps to XOp.
        assert!(ProvenanceTier::XOp > ProvenanceTier::Empirical);
        assert_eq!(ProvenanceTier::from_rank(200), ProvenanceTier::XOp);
    }

    #[test]
    fn zero_aware_delta_keeps_zeroes_and_freezes_utilization() {
        let lookup = ZeroAwareDeltaLookup::new(vec![
            (vec![16.0, 16.0], 0.0),
            (vec![1024.0, 1024.0], 2.0),
        ]);
        let sol = |c: &[f64]| c[0] * c[1];

        // Nearest to the zero point: the measured zero delta stays zero.
        assert_eq!(lookup.estimate(&[8.0, 8.0], sol).unwrap(), 0.0);
        // Nearest to the positive point: frozen util scales with query SOL.
        let expected = (2048.0 * 2048.0) / ((1024.0 * 1024.0) / 2.0);
        assert!((lookup.estimate(&[2048.0, 2048.0], sol).unwrap() - expected).abs() < 1e-12);
        // Empty table -> typed empirical miss.
        let empty = ZeroAwareDeltaLookup::new(vec![]);
        assert!(matches!(
            empty.estimate(&[1.0, 1.0], sol),
            Err(AicError::EmpiricalNotImplemented(_))
        ));
    }
}
