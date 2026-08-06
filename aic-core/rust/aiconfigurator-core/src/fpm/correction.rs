// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online native-correction grid for the forward-pass perf model.
//!
//! Buckets observed `(workload feature, observed_ms / native_ms)` correction
//! samples into a fixed-bound grid and applies the local median ratio on top of
//! the native AIC estimate.

use super::options::ForwardPassPerfOptions;
use super::samples::{median_ratio, AxisRange, BucketedSamples, StoreStats, WithOptions};

#[derive(Clone, Debug)]
pub(crate) struct CorrectionBuckets {
    samples: BucketedSamples<CorrectionObservation>,
    min_observations: usize,
    max_correction_factor: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct CorrectionObservation {
    correction_factor: f64,
}

impl WithOptions for CorrectionBuckets {
    fn with_options(options: &ForwardPassPerfOptions, axis_ranges: &[AxisRange]) -> Self {
        Self {
            samples: BucketedSamples::new_fixed(options, axis_ranges),
            min_observations: options.min_observations,
            max_correction_factor: options.max_correction_factor,
        }
    }
}

impl StoreStats for CorrectionBuckets {
    fn observation_count(&self) -> usize {
        self.samples.total_observations
    }

    fn is_ready(&self) -> bool {
        // Match planner's regression readiness semantics: min_observations is
        // checked across the whole inferred workload kind, not per region.
        // Regions only decide which correction factor to apply once the
        // workload kind is ready.
        self.samples.total_observations >= self.min_observations
    }
}

impl CorrectionBuckets {
    pub(crate) fn add_observation(&mut self, x: Vec<f64>, observed_ms: f64, native_ms: f64) {
        if native_ms.is_finite() && native_ms > 0.0 && observed_ms.is_finite() && observed_ms > 0.0
        {
            // Corrections are absolute observed/native samples, not
            // incremental multipliers. Bound each sample before the median is
            // computed so repeated outliers cannot exceed the ceiling.
            let correction_factor = observed_ms / native_ms;
            let bounded_correction_factor = self
                .max_correction_factor
                .map_or(correction_factor, |max_factor| {
                    correction_factor.min(max_factor)
                });
            self.samples.add(
                x,
                CorrectionObservation {
                    correction_factor: bounded_correction_factor,
                },
            );
        }
    }

    pub(crate) fn correction_factor_for(&self, x: &[f64]) -> f64 {
        if !self.is_ready() {
            return 1.0;
        }
        // Every region has an implicit correction factor of 1.0. A populated
        // in-range region overrides that default with its local median
        // observed/native ratio after the workload-kind-wide readiness gate passes.
        let Some(key) = self.samples.bucket_key_if_in_bounds(x) else {
            return 1.0;
        };
        let Some(bucket) = self.samples.buckets.get(&key) else {
            return 1.0;
        };
        median_ratio(
            bucket
                .iter()
                .map(|(_, observation)| observation.correction_factor),
        )
        .unwrap_or(1.0)
    }

    pub(crate) fn ready_bucket_count(&self) -> usize {
        if self.is_ready() {
            self.samples.buckets.len()
        } else {
            0
        }
    }

    pub(crate) fn correction_factors(&self) -> Vec<f64> {
        if !self.is_ready() {
            return Vec::new();
        }
        self.samples
            .buckets
            .values()
            .filter_map(|bucket| {
                median_ratio(
                    bucket
                        .iter()
                        .map(|(_, observation)| observation.correction_factor),
                )
            })
            .collect()
    }
}
