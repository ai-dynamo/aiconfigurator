// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Workload-specific linear regression fallback for the forward-pass perf model.
//!
//! Used when native AIC support is unavailable: fits a linear model from
//! bucketed `(feature vector, observed_ms)` samples once an inferred workload
//! kind has enough observations, and predicts from that fit.

use super::options::ForwardPassPerfOptions;
use super::samples::{AxisRange, BucketedSamples, StoreStats, WithOptions};

const RELAXABLE_NEG_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Debug)]
pub(crate) struct BucketedRegression {
    samples: BucketedSamples<f64>,
    ndim: usize,
    min_observations: usize,
    relaxable: Vec<usize>,
    fit: Option<LinearFit>,
}

impl WithOptions for BucketedRegression {
    fn with_options(
        options: &ForwardPassPerfOptions,
        axis_ranges: &[AxisRange],
        relaxable: &[usize],
    ) -> Self {
        let ndim = axis_ranges.len();
        Self {
            samples: BucketedSamples::new_dynamic(options, ndim),
            ndim,
            min_observations: options.min_observations,
            relaxable: relaxable.to_vec(),
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
            self.fit = fit_linear(
                self.samples.observations(),
                self.ndim,
                self.min_observations,
                &self.relaxable,
            );
        }
    }

    pub(crate) fn predict(&self, x: &[f64]) -> Option<f64> {
        self.fit.as_ref().map(|fit| fit.predict(x).max(1e-6))
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

fn fit_linear(
    observations: Vec<(Vec<f64>, f64)>,
    ndim: usize,
    min_observations: usize,
    relaxable: &[usize],
) -> Option<LinearFit> {
    if observations.len() < min_observations {
        return None;
    }
    let size = ndim + 1;
    let mut lhs = vec![vec![0.0_f64; size]; size];
    let mut rhs = vec![0.0_f64; size];

    for (x, y) in observations {
        let mut row = Vec::with_capacity(size);
        row.push(1.0);
        row.extend(x.into_iter().take(ndim));
        for i in 0..size {
            rhs[i] += row[i] * y;
            for j in 0..size {
                lhs[i][j] += row[i] * row[j];
            }
        }
    }

    let solution = solve_linear_system(lhs.clone(), rhs.clone())
        .or_else(|| solve_regularized_linear_system(lhs, rhs))?;
    let mut coefficients = solution[1..].to_vec();
    let mut has_non_relaxable_negative = false;
    for (idx, coef) in coefficients.iter_mut().enumerate() {
        if *coef < 0.0 {
            if relaxable.contains(&idx) {
                if *coef < -RELAXABLE_NEG_TOLERANCE {
                    *coef = 0.0;
                }
            } else {
                has_non_relaxable_negative = true;
            }
        }
    }
    if has_non_relaxable_negative {
        return None;
    }
    Some(LinearFit {
        intercept: solution[0],
        coefficients,
    })
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
