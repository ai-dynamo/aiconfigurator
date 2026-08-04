// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Workload-specific regression fallback for the forward-pass perf model.
//!
//! Used when native AIC support is unavailable: fits a workload-specific model
//! from bucketed `(feature vector, observed_ms)` samples once an inferred
//! workload kind has enough observations, and predicts from that fit.

use super::options::ForwardPassPerfOptions;
use super::samples::{AxisRange, BucketedSamples, StoreStats, WithOptions};

const HINGE_NEG_TOLERANCE: f64 = 1e-6;
const PREFILL_HINGE_MIN_OBSERVATIONS: usize = 6;
const RELATIVE_WEIGHT_FLOOR_MS: f64 = 25.0;
const HINGE_LOGSPACE_CANDIDATES: usize = 8;

#[derive(Clone, Debug)]
pub(crate) struct BucketedRegression {
    samples: BucketedSamples<f64>,
    ndim: usize,
    min_observations: usize,
    fit: Option<RegressionFit>,
}

impl WithOptions for BucketedRegression {
    fn with_options(options: &ForwardPassPerfOptions, axis_ranges: &[AxisRange]) -> Self {
        let ndim = axis_ranges.len();
        Self {
            samples: BucketedSamples::new_dynamic(options, ndim),
            ndim,
            min_observations: options.min_observations,
            fit: None,
        }
    }
}

impl StoreStats for BucketedRegression {
    fn observation_count(&self) -> usize {
        self.samples.total_observations
    }

    fn is_ready(&self) -> bool {
        self.fit.is_some()
    }
}

impl BucketedRegression {
    pub(crate) fn add_observation(&mut self, x: Vec<f64>, y: f64) {
        if self.samples.add(x, y) {
            let observations = self.samples.observations();
            self.fit = fit_regression(&observations, self.ndim, self.min_observations);
        }
    }

    pub(crate) fn predict(&self, x: &[f64]) -> Option<f64> {
        self.fit.as_ref().map(|fit| fit.predict(x).max(1e-6))
    }
}

#[derive(Clone, Debug)]
enum RegressionFit {
    Linear(LinearFit),
    WeightedHinge(WeightedHingeFit),
}

impl RegressionFit {
    fn predict(&self, x: &[f64]) -> f64 {
        match self {
            Self::Linear(fit) => fit.predict(x),
            Self::WeightedHinge(fit) => fit.predict(x),
        }
    }
}

#[derive(Clone, Debug)]
struct LinearFit {
    intercept: f64,
    coefficients: Vec<f64>,
}

impl LinearFit {
    fn predict(&self, x: &[f64]) -> f64 {
        self.intercept
            + self
                .coefficients
                .iter()
                .zip(x.iter())
                .map(|(coef, value)| coef * value)
                .sum::<f64>()
    }
}

#[derive(Clone, Debug)]
struct WeightedHingeFit {
    intercept: f64,
    left_slope: f64,
    right_slope_delta: f64,
    knot: f64,
}

impl WeightedHingeFit {
    fn predict(&self, x: &[f64]) -> f64 {
        let value = x.first().copied().unwrap_or_default();
        self.intercept
            + self.left_slope * value
            + self.right_slope_delta * (value - self.knot).max(0.0)
    }

    fn is_monotonic_nonnegative(&self) -> bool {
        self.intercept >= -HINGE_NEG_TOLERANCE
            && self.left_slope >= -HINGE_NEG_TOLERANCE
            && self.left_slope + self.right_slope_delta >= -HINGE_NEG_TOLERANCE
    }
}

fn fit_regression(
    observations: &[(Vec<f64>, f64)],
    ndim: usize,
    min_observations: usize,
) -> Option<RegressionFit> {
    // Prefill is the only one-dimensional workload kind today. Its latency is
    // overhead-heavy at low token counts and slope-heavy at high token counts,
    // so a weighted hinge avoids the negative intercepts a global line can fit.
    if ndim == 1 && observations.len() >= min_observations.max(PREFILL_HINGE_MIN_OBSERVATIONS) {
        if let Some(fit) = fit_weighted_hinge(observations) {
            return Some(RegressionFit::WeightedHinge(fit));
        }
    }

    fit_linear(observations, ndim, min_observations).map(RegressionFit::Linear)
}

fn fit_linear(
    observations: &[(Vec<f64>, f64)],
    ndim: usize,
    min_observations: usize,
) -> Option<LinearFit> {
    if observations.len() < min_observations {
        return None;
    }

    // The forward-pass regressions have at most two workload dimensions. Find
    // the non-negative least-squares solution by enumerating every active set:
    // each bit selects a slope that is free, while unselected slopes are fixed
    // at zero. The intercept remains unconstrained. Comparing the residuals of
    // all feasible faces gives the constrained optimum without relying on an
    // absolute coefficient tolerance across differently-scaled features.
    let active_set_count = 1usize.checked_shl(ndim.try_into().ok()?)?;
    let mut best: Option<(f64, LinearFit)> = None;
    for active_mask in 0..active_set_count {
        let active = (0..ndim)
            .filter(|idx| active_mask & (1usize << idx) != 0)
            .collect::<Vec<_>>();
        let Some(fit) = fit_linear_active_set(observations, ndim, &active) else {
            continue;
        };
        let squared_error = observations
            .iter()
            .map(|(x, y)| {
                let residual = fit.predict(x) - *y;
                residual * residual
            })
            .sum::<f64>();
        if !squared_error.is_finite() {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(best_error, _)| squared_error < *best_error)
        {
            best = Some((squared_error, fit));
        }
    }

    let fit = best.map(|(_, fit)| fit)?;
    let is_underdetermined = observations.len() <= ndim;
    let has_load_signal = fit.coefficients.iter().any(|coef| *coef > 0.0);

    // Preserve explicitly configured low-observation behavior while the
    // system is underdetermined. Once there are enough observations to
    // identify the slopes, an intercept-only boundary has no usable load
    // signal and must not make the model ready.
    (is_underdetermined || has_load_signal).then_some(fit)
}

fn fit_linear_active_set(
    observations: &[(Vec<f64>, f64)],
    ndim: usize,
    active: &[usize],
) -> Option<LinearFit> {
    let size = active.len() + 1;
    let mut lhs = vec![vec![0.0_f64; size]; size];
    let mut rhs = vec![0.0_f64; size];

    for (x, y) in observations {
        let mut row = Vec::with_capacity(size);
        row.push(1.0);
        for idx in active {
            row.push(*x.get(*idx)?);
        }
        for i in 0..size {
            rhs[i] += row[i] * *y;
            for j in 0..size {
                lhs[i][j] += row[i] * row[j];
            }
        }
    }

    let solution = solve_linear_system(lhs.clone(), rhs.clone())
        .or_else(|| solve_regularized_linear_system(lhs, rhs))?;
    if !solution.iter().all(|value| value.is_finite())
        || solution[1..].iter().any(|coef| *coef < 0.0)
    {
        return None;
    }
    let mut coefficients = vec![0.0; ndim];
    for (solution_idx, feature_idx) in active.iter().enumerate() {
        coefficients[*feature_idx] = solution[solution_idx + 1];
    }
    Some(LinearFit {
        intercept: solution[0],
        coefficients,
    })
}

fn fit_weighted_hinge(observations: &[(Vec<f64>, f64)]) -> Option<WeightedHingeFit> {
    hinge_candidates(observations)
        .into_iter()
        .filter_map(|knot| {
            let fit = fit_weighted_hinge_at(observations, knot)?;
            let score = weighted_hinge_rmse(observations, &fit);
            Some((score, fit))
        })
        .min_by(|(left_score, _), (right_score, _)| {
            left_score
                .partial_cmp(right_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, fit)| fit)
}

fn fit_weighted_hinge_at(observations: &[(Vec<f64>, f64)], knot: f64) -> Option<WeightedHingeFit> {
    let size = 3;
    let mut lhs = vec![vec![0.0_f64; size]; size];
    let mut rhs = vec![0.0_f64; size];

    for (x, y) in observations {
        let value = *x.first()?;
        let row = [1.0, value, (value - knot).max(0.0)];
        let weight = relative_error_weight(*y);
        for i in 0..size {
            let weighted_i = row[i] * weight;
            rhs[i] += weighted_i * *y * weight;
            for j in 0..size {
                lhs[i][j] += weighted_i * row[j] * weight;
            }
        }
    }

    let solution = solve_linear_system(lhs.clone(), rhs.clone())
        .or_else(|| solve_regularized_linear_system(lhs, rhs))?;
    let fit = WeightedHingeFit {
        intercept: solution[0],
        left_slope: solution[1],
        right_slope_delta: solution[2],
        knot,
    };
    fit.is_monotonic_nonnegative().then_some(fit)
}

fn weighted_hinge_rmse(observations: &[(Vec<f64>, f64)], fit: &WeightedHingeFit) -> f64 {
    let mse = observations
        .iter()
        .map(|(x, y)| {
            let residual = (fit.predict(x) - *y) * relative_error_weight(*y);
            residual * residual
        })
        .sum::<f64>()
        / observations.len().max(1) as f64;
    mse.sqrt()
}

fn hinge_candidates(observations: &[(Vec<f64>, f64)]) -> Vec<f64> {
    let mut xs = observations
        .iter()
        .filter_map(|(x, _)| x.first().copied())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() <= f64::EPSILON);
    if xs.len() < 3 {
        return Vec::new();
    }

    let min = xs[0];
    let max = xs[xs.len() - 1];
    let mut candidates = Vec::new();
    for numerator in [1usize, 2, 3, 4] {
        let idx = numerator * (xs.len() - 1) / 5;
        candidates.push(xs[idx]);
    }
    if min > 0.0 && max > min {
        let log_min = min.ln();
        let log_max = max.ln();
        for idx in 1..=HINGE_LOGSPACE_CANDIDATES {
            let t = idx as f64 / (HINGE_LOGSPACE_CANDIDATES + 1) as f64;
            candidates.push((log_min + t * (log_max - log_min)).exp());
        }
    }

    candidates.retain(|candidate| candidate.is_finite() && *candidate > min && *candidate < max);
    candidates.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|left, right| {
        let scale = left.abs().max(right.abs()).max(1.0);
        (*left - *right).abs() <= scale * 1e-9
    });
    candidates
}

fn relative_error_weight(observed_ms: f64) -> f64 {
    1.0 / observed_ms.max(RELATIVE_WEIGHT_FLOOR_MS)
}

fn solve_regularized_linear_system(mut lhs: Vec<Vec<f64>>, rhs: Vec<f64>) -> Option<Vec<f64>> {
    let scale = lhs
        .iter()
        .enumerate()
        .map(|(idx, row)| row[idx].abs())
        .sum::<f64>()
        .max(1.0);
    let ridge = scale * 1e-9;
    for (idx, row) in lhs.iter_mut().enumerate().skip(1) {
        row[idx] += ridge;
    }
    solve_linear_system(lhs, rhs)
}

fn solve_linear_system(mut lhs: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|a, b| {
            lhs[*a][col]
                .abs()
                .partial_cmp(&lhs[*b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if lhs[pivot][col].abs() < 1e-12 {
            return None;
        }
        lhs.swap(col, pivot);
        rhs.swap(col, pivot);

        let divisor = lhs[col][col];
        for j in col..n {
            lhs[col][j] /= divisor;
        }
        rhs[col] /= divisor;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = lhs[row][col];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                lhs[row][j] -= factor * lhs[col][j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    Some(rhs)
}
