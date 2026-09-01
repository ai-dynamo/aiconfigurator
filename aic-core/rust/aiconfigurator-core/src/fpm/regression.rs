// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-type-bound regression fallback for the forward-pass perf model.
//!
//! Regression always consumes two raw features: critical attention work and
//! global FFN/MoE work. Samples are retained using `log1p`-transformed bucket
//! coordinates, while fitting and prediction use standardized raw features.

use super::options::ForwardPassPerfOptions;
use super::samples::{BucketedSamples, StoreStats};

const FEATURE_DIMENSION: usize = 2;
const INACTIVE_SCALE_RELATIVE_TOLERANCE: f64 = 1e-12;
const MIN_POSITIVE_PREDICTION_MS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
struct RegressionObservation {
    raw_x: [f64; FEATURE_DIMENSION],
    observed_ms: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct BucketedRegression {
    samples: BucketedSamples<RegressionObservation>,
    min_observations: usize,
    fit: Option<LinearFit>,
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
    pub(crate) fn new(options: &ForwardPassPerfOptions) -> Self {
        Self {
            samples: BucketedSamples::new_dynamic(options, FEATURE_DIMENSION),
            min_observations: options.min_observations,
            fit: None,
        }
    }

    /// Retain an observation and refit from the currently retained raw data.
    ///
    /// Returns `false` without changing the model when a feature is negative
    /// or non-finite, or when the target is not finite and strictly positive.
    pub(crate) fn add_observation(
        &mut self,
        raw_x: [f64; FEATURE_DIMENSION],
        observed_ms: f64,
    ) -> bool {
        if !valid_features(&raw_x) || !observed_ms.is_finite() || observed_ms <= 0.0 {
            return false;
        }

        let bucket_x = raw_x.map(f64::ln_1p);
        let observation = RegressionObservation { raw_x, observed_ms };
        if !self.samples.add(bucket_x.to_vec(), observation) {
            return false;
        }

        let retained = self
            .samples
            .observations()
            .into_iter()
            .map(|(_, observation)| observation)
            .collect::<Vec<_>>();
        self.fit = fit_regression(&retained, self.min_observations);
        true
    }

    pub(crate) fn predict(&self, raw_x: &[f64; FEATURE_DIMENSION]) -> Option<f64> {
        if !valid_features(raw_x) {
            return None;
        }
        let prediction = self.fit.as_ref()?.predict(raw_x)?;
        Some(prediction.max(MIN_POSITIVE_PREDICTION_MS))
    }
}

fn valid_features(x: &[f64; FEATURE_DIMENSION]) -> bool {
    x.iter().all(|value| value.is_finite() && *value >= 0.0)
}

#[derive(Clone, Copy, Debug)]
struct Standardization {
    means: [f64; FEATURE_DIMENSION],
    scales: [f64; FEATURE_DIMENSION],
    active: [bool; FEATURE_DIMENSION],
}

impl Standardization {
    /// Compute population mean and standard deviation with Welford's method.
    fn from_observations(observations: &[RegressionObservation]) -> Option<Self> {
        if observations.is_empty() {
            return None;
        }

        let mut count = 0usize;
        let mut means = [0.0; FEATURE_DIMENSION];
        let mut squared_deviation_sums = [0.0; FEATURE_DIMENSION];
        for observation in observations {
            count += 1;
            let count_f64 = count as f64;
            for axis in 0..FEATURE_DIMENSION {
                let value = observation.raw_x[axis];
                let delta = value - means[axis];
                means[axis] += delta / count_f64;
                let delta_from_new_mean = value - means[axis];
                squared_deviation_sums[axis] += delta * delta_from_new_mean;
            }
        }

        if !means.iter().all(|value| value.is_finite())
            || !squared_deviation_sums.iter().all(|value| value.is_finite())
        {
            return None;
        }

        let mut scales = [0.0; FEATURE_DIMENSION];
        let mut active = [false; FEATURE_DIMENSION];
        for axis in 0..FEATURE_DIMENSION {
            // A tiny negative value can result from floating-point roundoff.
            let variance = (squared_deviation_sums[axis] / count as f64).max(0.0);
            let scale = variance.sqrt();
            let inactive_threshold = INACTIVE_SCALE_RELATIVE_TOLERANCE * means[axis].abs().max(1.0);
            scales[axis] = scale;
            active[axis] = scale.is_finite() && scale > inactive_threshold;
        }

        Some(Self {
            means,
            scales,
            active,
        })
    }

    fn transform(&self, raw_x: &[f64; FEATURE_DIMENSION]) -> Option<[f64; FEATURE_DIMENSION]> {
        let mut standardized = [0.0; FEATURE_DIMENSION];
        for axis in 0..FEATURE_DIMENSION {
            if self.active[axis] {
                standardized[axis] = (raw_x[axis] - self.means[axis]) / self.scales[axis];
                if !standardized[axis].is_finite() {
                    return None;
                }
            }
        }
        Some(standardized)
    }
}

#[derive(Clone, Debug)]
struct LinearFit {
    intercept: f64,
    coefficients: [f64; FEATURE_DIMENSION],
    standardization: Standardization,
}

impl LinearFit {
    fn predict(&self, raw_x: &[f64; FEATURE_DIMENSION]) -> Option<f64> {
        let standardized = self.standardization.transform(raw_x)?;
        let prediction = self.predict_standardized(&standardized);
        prediction.is_finite().then_some(prediction)
    }

    fn predict_standardized(&self, x: &[f64; FEATURE_DIMENSION]) -> f64 {
        self.intercept
            + self
                .coefficients
                .iter()
                .enumerate()
                .map(|(axis, coefficient)| coefficient * x[axis])
                .sum::<f64>()
    }
}

#[derive(Clone, Copy, Debug)]
struct StandardizedObservation {
    x: [f64; FEATURE_DIMENSION],
    observed_ms: f64,
}

fn fit_regression(
    observations: &[RegressionObservation],
    min_observations: usize,
) -> Option<LinearFit> {
    if observations.len() < min_observations {
        return None;
    }

    let standardization = Standardization::from_observations(observations)?;
    let varying_axes = (0..FEATURE_DIMENSION)
        .filter(|axis| standardization.active[*axis])
        .collect::<Vec<_>>();
    if varying_axes.is_empty() {
        return None;
    }

    let standardized = observations
        .iter()
        .map(|observation| {
            Some(StandardizedObservation {
                x: standardization.transform(&observation.raw_x)?,
                observed_ms: observation.observed_ms,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    // Find the non-negative least-squares solution by enumerating every face
    // of the two-dimensional slope constraint. The intercept is always free.
    let active_set_count = 1usize.checked_shl(varying_axes.len().try_into().ok()?)?;
    let mut best: Option<(f64, LinearFit)> = None;
    for active_mask in 0..active_set_count {
        let fitted_axes = varying_axes
            .iter()
            .enumerate()
            .filter_map(|(mask_axis, feature_axis)| {
                (active_mask & (1usize << mask_axis) != 0).then_some(*feature_axis)
            })
            .collect::<Vec<_>>();
        let Some(fit) = fit_linear_active_set(&standardized, standardization, &fitted_axes) else {
            continue;
        };
        let squared_error = standardized
            .iter()
            .map(|observation| {
                let residual = fit.predict_standardized(&observation.x) - observation.observed_ms;
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
    let effective_dimension = varying_axes.len();
    let is_underdetermined = observations.len() <= effective_dimension;
    let has_load_signal = fit
        .coefficients
        .iter()
        .any(|coefficient| *coefficient > 0.0);

    // Preserve explicitly configured low-observation behavior while the fit
    // is underdetermined. Once slopes are identifiable, an intercept-only
    // boundary does not provide a usable load signal and must remain unready.
    (is_underdetermined || has_load_signal).then_some(fit)
}

fn fit_linear_active_set(
    observations: &[StandardizedObservation],
    standardization: Standardization,
    fitted_axes: &[usize],
) -> Option<LinearFit> {
    let size = fitted_axes.len() + 1;
    let mut lhs = vec![vec![0.0_f64; size]; size];
    let mut rhs = vec![0.0_f64; size];

    for observation in observations {
        let mut row = Vec::with_capacity(size);
        row.push(1.0);
        for axis in fitted_axes {
            row.push(observation.x[*axis]);
        }
        for i in 0..size {
            rhs[i] += row[i] * observation.observed_ms;
            for j in 0..size {
                lhs[i][j] += row[i] * row[j];
            }
        }
    }

    let solution = solve_linear_system(lhs.clone(), rhs.clone())
        .or_else(|| solve_regularized_linear_system(lhs, rhs))?;
    if !solution.iter().all(|value| value.is_finite())
        || solution[1..].iter().any(|coefficient| *coefficient < 0.0)
    {
        return None;
    }

    let mut coefficients = [0.0; FEATURE_DIMENSION];
    for (solution_axis, feature_axis) in fitted_axes.iter().enumerate() {
        coefficients[*feature_axis] = solution[solution_axis + 1];
    }
    Some(LinearFit {
        intercept: solution[0],
        coefficients,
        standardization,
    })
}

/// Retry a singular normal equation with ridge regularization on slopes only.
fn solve_regularized_linear_system(mut lhs: Vec<Vec<f64>>, rhs: Vec<f64>) -> Option<Vec<f64>> {
    let scale = lhs
        .iter()
        .enumerate()
        .map(|(axis, row)| row[axis].abs())
        .sum::<f64>()
        .max(1.0);
    let ridge = scale * 1e-9;
    for (axis, row) in lhs.iter_mut().enumerate().skip(1) {
        row[axis] += ridge;
    }
    solve_linear_system(lhs, rhs)
}

fn solve_linear_system(mut lhs: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let dimension = rhs.len();
    for column in 0..dimension {
        let pivot = (column..dimension).max_by(|left, right| {
            lhs[*left][column]
                .abs()
                .partial_cmp(&lhs[*right][column].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if lhs[pivot][column].abs() < 1e-12 {
            return None;
        }
        lhs.swap(column, pivot);
        rhs.swap(column, pivot);

        let divisor = lhs[column][column];
        for value in &mut lhs[column][column..] {
            *value /= divisor;
        }
        rhs[column] /= divisor;

        for row in 0..dimension {
            if row == column {
                continue;
            }
            let factor = lhs[row][column];
            if factor == 0.0 {
                continue;
            }
            for inner_column in column..dimension {
                lhs[row][inner_column] -= factor * lhs[column][inner_column];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    Some(rhs)
}

#[cfg(test)]
mod tests {
    use super::{
        fit_linear_active_set, BucketedRegression, RegressionObservation, Standardization,
        StandardizedObservation, StoreStats, INACTIVE_SCALE_RELATIVE_TOLERANCE,
        MIN_POSITIVE_PREDICTION_MS,
    };
    use crate::fpm::options::ForwardPassPerfOptions;

    fn regression_options() -> ForwardPassPerfOptions {
        ForwardPassPerfOptions {
            min_observations: 5,
            max_observations: 64,
            ..ForwardPassPerfOptions::default()
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn buckets_on_log_coordinates_but_retains_raw_observation() {
        let mut regression = BucketedRegression::new(&regression_options());
        let raw_x = [99.0, 9_999.0];

        assert!(regression.add_observation(raw_x, 12.0));
        let retained = regression.samples.observations();
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].0, vec![100.0_f64.ln(), 10_000.0_f64.ln()]);
        assert_eq!(retained[0].1.raw_x, raw_x);
        assert_eq!(retained[0].1.observed_ms, 12.0);
    }

    #[test]
    fn expanding_log_bounds_rebuckets_existing_observations() {
        let options = ForwardPassPerfOptions {
            bucket_count: 4,
            ..regression_options()
        };
        let mut regression = BucketedRegression::new(&options);
        assert!(regression.add_observation([0.0, 0.0], 1.0));
        assert!(regression.add_observation([1.0, 1.0], 2.0));

        let key_for = |regression: &BucketedRegression, raw_x: [f64; 2]| {
            regression
                .samples
                .buckets
                .iter()
                .find_map(|(key, bucket)| {
                    bucket
                        .iter()
                        .any(|(_, observation)| observation.raw_x == raw_x)
                        .then(|| key.clone())
                })
                .unwrap()
        };
        assert_eq!(key_for(&regression, [1.0, 1.0]), vec![1, 1]);

        // log1p(1) occupies the upper cell while it is the maximum. Expanding
        // both bounds to log1p(15) moves that retained point into the lower
        // cell, proving that dynamic-bound expansion rebuilds existing keys.
        assert!(regression.add_observation([15.0, 15.0], 3.0));
        assert_eq!(key_for(&regression, [1.0, 1.0]), vec![0, 0]);
        assert_eq!(key_for(&regression, [15.0, 15.0]), vec![1, 1]);
    }

    #[test]
    fn rejects_invalid_raw_features_and_targets_without_mutation() {
        let mut regression = BucketedRegression::new(&regression_options());
        assert!(!regression.add_observation([-1.0, 1.0], 1.0));
        assert!(!regression.add_observation([f64::NAN, 1.0], 1.0));
        assert!(!regression.add_observation([1.0, f64::INFINITY], 1.0));
        assert!(!regression.add_observation([1.0, 1.0], 0.0));
        assert!(!regression.add_observation([1.0, 1.0], f64::NAN));
        assert_eq!(regression.observation_count(), 0);
        assert!(!regression.is_ready());
        assert_eq!(regression.predict(&[1.0, 1.0]), None);
    }

    #[test]
    fn fits_standardized_raw_features_instead_of_log_features() {
        let mut regression = BucketedRegression::new(&regression_options());
        for raw_x in [
            [1.0, 2.0],
            [2.0, 7.0],
            [4.0, 3.0],
            [8.0, 11.0],
            [16.0, 5.0],
            [32.0, 13.0],
        ] {
            let observed_ms = 7.0 + 2.5 * raw_x[0] + 4.0 * raw_x[1];
            assert!(regression.add_observation(raw_x, observed_ms));
        }

        assert!(regression.is_ready());
        assert_close(regression.predict(&[64.0, 17.0]).unwrap(), 235.0, 1e-8);
    }

    #[test]
    fn standardization_handles_large_feature_scale_disparity() {
        let mut regression = BucketedRegression::new(&regression_options());
        for raw_x in [
            [1.0e12, 1.0],
            [2.0e12, 8.0],
            [4.0e12, 3.0],
            [8.0e12, 10.0],
            [16.0e12, 5.0],
            [32.0e12, 13.0],
        ] {
            let observed_ms = 11.0 + 1.0e-9 * raw_x[0] + 3.0 * raw_x[1];
            assert!(regression.add_observation(raw_x, observed_ms));
        }

        let expected = 11.0 + 64_000.0 + 51.0;
        assert_close(
            regression.predict(&[64.0e12, 17.0]).unwrap(),
            expected,
            1e-6,
        );
    }

    #[test]
    fn constant_axis_is_inactive_and_predicts_with_fit_snapshot() {
        let mut regression = BucketedRegression::new(&regression_options());
        for attention in 1..=6 {
            let raw_x = [attention as f64, 7.0];
            assert!(regression.add_observation(raw_x, 2.0 + 4.0 * raw_x[0]));
        }

        let fit = regression.fit.as_ref().unwrap();
        assert_eq!(fit.standardization.active, [true, false]);
        assert_eq!(fit.standardization.scales[1], 0.0);
        assert_eq!(fit.coefficients[1], 0.0);
        // The inactive coordinate is ignored even when extrapolated.
        assert_close(regression.predict(&[10.0, 70.0]).unwrap(), 42.0, 1e-10);
    }

    #[test]
    fn refit_statistics_use_only_post_eviction_observations() {
        let options = ForwardPassPerfOptions {
            min_observations: 5,
            max_observations: 5,
            bucket_count: 1,
            ..ForwardPassPerfOptions::default()
        };
        let mut regression = BucketedRegression::new(&options);
        for value in 0..=5 {
            let raw_x = [value as f64, (value * value) as f64];
            assert!(regression.add_observation(raw_x, 1.0 + raw_x[0] + raw_x[1]));
        }

        assert_eq!(regression.observation_count(), 5);
        let fit = regression.fit.as_ref().unwrap();
        // With one bucket, inserting x=5 retires the oldest x=0 sample.
        assert_close(fit.standardization.means[0], 3.0, 1e-12);
        assert_close(fit.standardization.means[1], 11.0, 1e-12);
        assert_close(fit.standardization.scales[0], 2.0_f64.sqrt(), 1e-12);
    }

    #[test]
    fn relative_threshold_marks_nearly_constant_axis_inactive() {
        let mean = 1.0e6;
        let delta = INACTIVE_SCALE_RELATIVE_TOLERANCE * mean / 10.0;
        let observations = (0..6)
            .map(|index| RegressionObservation {
                raw_x: [
                    index as f64,
                    mean + if index % 2 == 0 { delta } else { -delta },
                ],
                observed_ms: index as f64 + 1.0,
            })
            .collect::<Vec<_>>();
        let standardization = Standardization::from_observations(&observations).unwrap();
        assert_eq!(standardization.active, [true, false]);
    }

    #[test]
    fn all_constant_features_never_make_regression_ready() {
        let mut regression = BucketedRegression::new(&regression_options());
        for observed_ms in 1..=6 {
            assert!(regression.add_observation([4.0, 9.0], observed_ms as f64));
        }
        assert_eq!(regression.observation_count(), 6);
        assert!(!regression.is_ready());
        assert_eq!(regression.predict(&[4.0, 9.0]), None);
    }

    #[test]
    fn identifiable_intercept_only_boundary_is_not_ready() {
        let mut regression = BucketedRegression::new(&regression_options());
        for attention in 1..=6 {
            let value = attention as f64;
            assert!(regression.add_observation([value, 0.0], 100.0 - value));
        }
        assert!(!regression.is_ready());
    }

    #[test]
    fn positive_slope_extrapolation_is_clamped_to_prediction_floor() {
        let mut regression = BucketedRegression::new(&regression_options());
        for attention in 6..=10 {
            let attention = attention as f64;
            assert!(regression.add_observation([attention, 0.0], attention - 5.0));
        }

        let fit = regression.fit.as_ref().unwrap();
        assert!(fit.coefficients[0] > 0.0);
        assert!(fit.predict(&[1.0, 0.0]).unwrap() <= 0.0);
        assert_eq!(
            regression.predict(&[1.0, 0.0]),
            Some(MIN_POSITIVE_PREDICTION_MS)
        );
    }

    #[test]
    fn collinear_active_axes_use_slope_regularization_with_free_intercept() {
        let observations = (1..=6)
            .map(|attention| RegressionObservation {
                raw_x: [attention as f64, 2.0 * attention as f64],
                observed_ms: 7.0 + 3.0 * attention as f64,
            })
            .collect::<Vec<_>>();
        let standardization = Standardization::from_observations(&observations).unwrap();
        assert_eq!(standardization.active, [true, true]);
        let standardized = observations
            .iter()
            .map(|observation| StandardizedObservation {
                x: standardization.transform(&observation.raw_x).unwrap(),
                observed_ms: observation.observed_ms,
            })
            .collect::<Vec<_>>();

        // The two standardized columns are identical, so the unregularized
        // normal equation is singular. The fallback regularizes only the two
        // slopes, leaving the free intercept at the population target mean.
        let regularized_fit =
            fit_linear_active_set(&standardized, standardization, &[0, 1]).unwrap();
        assert!(regularized_fit
            .coefficients
            .iter()
            .all(|slope| *slope > 0.0));
        assert_close(regularized_fit.intercept, 17.5, 1e-12);
        assert_close(regularized_fit.predict(&[7.0, 14.0]).unwrap(), 28.0, 1e-7);

        let mut regression = BucketedRegression::new(&regression_options());
        for observation in observations {
            assert!(regression.add_observation(observation.raw_x, observation.observed_ms));
        }
        assert!(regression.is_ready());
        assert_close(regression.predict(&[7.0, 14.0]).unwrap(), 28.0, 1e-7);
    }

    #[test]
    fn eviction_uses_fattest_cell_and_dynamic_extrema_do_not_shrink() {
        let options = ForwardPassPerfOptions {
            min_observations: 1,
            max_observations: 4,
            bucket_count: 4,
            ..ForwardPassPerfOptions::default()
        };
        let mut regression = BucketedRegression::new(&options);

        // The global oldest sample lives in the lower cell. The extreme is
        // instead the oldest sample in the upper cell, which is made uniquely
        // fattest before capacity is exceeded.
        assert!(regression.add_observation([0.0, 0.0], 1.0));
        assert!(regression.add_observation([1_000_000.0, 1_000_000.0], 4.0));
        assert!(regression.add_observation([2_000.0, 2_000.0], 2.0));
        assert!(regression.add_observation([3_000.0, 3_000.0], 3.0));
        assert!(regression.add_observation([100.0, 0.0], 1.5));

        let retained_raw = || {
            regression
                .samples
                .observations()
                .into_iter()
                .map(|(_, observation)| observation.raw_x)
                .collect::<Vec<_>>()
        };
        assert_eq!(regression.observation_count(), 4);
        assert!(retained_raw().contains(&[0.0, 0.0]));
        assert!(!retained_raw().contains(&[1_000_000.0, 1_000_000.0]));

        // If eviction had shrunk both upper bounds to 3,000, log1p(100)
        // would occupy cell [1, 1]. Retaining the evicted extrema keeps it in
        // cell [0, 0] when a subsequent sample is bucketed.
        assert!(regression.add_observation([100.0, 100.0], 1.75));
        let subsequent_key = regression
            .samples
            .buckets
            .iter()
            .find_map(|(key, bucket)| {
                bucket
                    .iter()
                    .any(|(_, observation)| observation.raw_x == [100.0, 100.0])
                    .then(|| key.clone())
            })
            .unwrap();
        assert_eq!(subsequent_key, vec![0, 0]);
    }

    #[test]
    fn readiness_uses_effective_active_dimension() {
        let options = ForwardPassPerfOptions {
            min_observations: 2,
            ..regression_options()
        };

        let mut two_active_axes = BucketedRegression::new(&options);
        assert!(two_active_axes.add_observation([0.0, 0.0], 2.0));
        assert!(two_active_axes.add_observation([1.0, 1.0], 1.0));
        assert_eq!(
            two_active_axes.fit.as_ref().unwrap().standardization.active,
            [true, true]
        );
        assert!(two_active_axes.is_ready());

        let mut one_active_axis = BucketedRegression::new(&options);
        assert!(one_active_axis.add_observation([0.0, 0.0], 2.0));
        assert!(one_active_axis.add_observation([1.0, 0.0], 1.0));
        assert!(!one_active_axis.is_ready());
    }

    #[test]
    fn prediction_rejects_invalid_features() {
        let mut regression = BucketedRegression::new(&regression_options());
        for raw_x in [[1.0, 2.0], [2.0, 7.0], [4.0, 3.0], [8.0, 11.0], [16.0, 5.0]] {
            assert!(regression.add_observation(raw_x, 5.0 + raw_x[0] + raw_x[1]));
        }
        assert_eq!(regression.predict(&[-1.0, 1.0]), None);
        assert_eq!(regression.predict(&[f64::NAN, 1.0]), None);
        assert_eq!(regression.predict(&[1.0, f64::INFINITY]), None);
    }
}
