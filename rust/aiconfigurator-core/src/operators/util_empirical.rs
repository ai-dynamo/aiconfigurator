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

    /// Fetch or build the grid for `key`. `builder` returns `None` when the
    /// slice has no usable calibration data (typed coverage miss) — that
    /// outcome is memoised too, mirroring Python's `grid_for` returning
    /// `None` (the caller then raises via [`estimate`] with `grid=None`).
    pub fn get_or_build<F>(&self, key: &str, builder: F) -> Option<Arc<UtilGrid>>
    where
        F: FnOnce() -> Option<UtilGrid>,
    {
        let mut grids = self.grids.lock().expect("util grid cache poisoned");
        if let Some(cached) = grids.get(key) {
            return cached.clone();
        }
        let built = builder().map(Arc::new);
        grids.insert(key.to_string(), built.clone());
        built
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
    fn util_grid_cache_memoises_including_misses() {
        let cache = UtilGridCache::new();
        let mut builds = 0;
        let grid = cache.get_or_build("k", || {
            builds += 1;
            Some(UtilGrid::new(vec![UtilSample::new(vec![1.0], 0.5)]))
        });
        assert!(grid.is_some());
        let again = cache.get_or_build("k", || {
            builds += 1;
            None
        });
        assert!(again.is_some());
        assert_eq!(builds, 1);

        let miss = cache.get_or_build("missing", || None);
        assert!(miss.is_none());
        let miss_again = cache.get_or_build("missing", || {
            panic!("memoised miss must not rebuild")
        });
        assert!(miss_again.is_none());
    }
}
