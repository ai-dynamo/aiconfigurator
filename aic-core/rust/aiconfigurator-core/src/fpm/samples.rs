// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared bucketed-sample infrastructure for the forward-pass perf model.
//!
//! [`BucketedSamples`] retains a bounded set of `(feature vector, value)`
//! observations partitioned into per-axis buckets, used by both the native
//! correction grid ([`super::correction`]) and the regression fallback
//! ([`super::regression`]). [`AxisRange`] describes a fixed correction axis
//! bound, and the [`WithOptions`] / [`StoreStats`] traits let the workload
//! store dispatch generically over either backend.

use std::collections::HashMap;

use super::options::ForwardPassPerfOptions;

pub(crate) trait WithOptions {
    fn with_options(
        options: &ForwardPassPerfOptions,
        axis_ranges: &[AxisRange],
        relaxable: &[usize],
    ) -> Self;
}

pub(crate) trait StoreStats {
    fn observation_count(&self) -> usize;
    fn is_ready(&self) -> bool;
}

#[derive(Clone, Debug)]
pub(crate) struct BucketedSamples<T> {
    pub(crate) buckets: HashMap<Vec<usize>, Vec<(Vec<f64>, T)>>,
    pub(crate) total_observations: usize,
    axis_min: Vec<f64>,
    axis_max: Vec<f64>,
    fixed_bounds: bool,
    buckets_per_axis: usize,
    max_observations: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AxisRange {
    min: f64,
    max: f64,
}

impl AxisRange {
    pub(crate) fn from_zero_to(max: u32) -> Self {
        Self {
            min: 0.0,
            max: f64::from(max),
        }
    }
}

impl<T: Clone> BucketedSamples<T> {
    pub(crate) fn new_dynamic(options: &ForwardPassPerfOptions, ndim: usize) -> Self {
        let buckets_per_axis = if ndim == 1 {
            options.bucket_count
        } else {
            integer_sqrt(options.bucket_count)
        };
        Self {
            buckets: HashMap::new(),
            total_observations: 0,
            axis_min: vec![f64::INFINITY; ndim],
            axis_max: vec![f64::NEG_INFINITY; ndim],
            fixed_bounds: false,
            buckets_per_axis: buckets_per_axis.max(1),
            max_observations: options.max_observations,
        }
    }

    pub(crate) fn new_fixed(options: &ForwardPassPerfOptions, axis_ranges: &[AxisRange]) -> Self {
        let mut samples = Self::new_dynamic(options, axis_ranges.len());
        samples.axis_min = axis_ranges.iter().map(|range| range.min).collect();
        samples.axis_max = axis_ranges.iter().map(|range| range.max).collect();
        samples.fixed_bounds = true;
        samples
    }

    pub(crate) fn add(&mut self, x: Vec<f64>, y: T) -> bool {
        if x.len() != self.axis_min.len() || !x.iter().all(|value| value.is_finite()) {
            return false;
        }

        if self.fixed_bounds {
            if !self.is_in_bounds(&x) {
                return false;
            }
        } else {
            let bounds_changed = self.update_axis_bounds(&x);
            if bounds_changed && self.total_observations > 0 {
                self.rebuild_buckets();
            }
        }

        let key = self.bucket_key(&x);
        self.buckets.entry(key).or_default().push((x, y));
        self.total_observations += 1;

        if self.total_observations > self.max_observations {
            self.retire_from_fattest_bucket();
        }
        true
    }

    pub(crate) fn observations(&self) -> Vec<(Vec<f64>, T)> {
        self.buckets
            .values()
            .flat_map(|bucket| bucket.iter().cloned())
            .collect()
    }

    fn bucket_key(&self, x: &[f64]) -> Vec<usize> {
        x.iter()
            .enumerate()
            .map(|(i, value)| {
                let lo = self.axis_min[i];
                let hi = self.axis_max[i];
                if hi <= lo {
                    0
                } else {
                    let idx = ((*value - lo) / (hi - lo) * self.buckets_per_axis as f64) as isize;
                    idx.clamp(0, self.buckets_per_axis as isize - 1) as usize
                }
            })
            .collect()
    }

    pub(crate) fn bucket_key_if_in_bounds(&self, x: &[f64]) -> Option<Vec<usize>> {
        if self.total_observations == 0 || x.len() != self.axis_min.len() {
            return None;
        }

        // Estimation must not clamp outside configured correction bounds into
        // edge regions.
        let mut key = Vec::with_capacity(x.len());
        for (i, value) in x.iter().enumerate() {
            let lo = self.axis_min[i];
            let hi = self.axis_max[i];
            if !value.is_finite() || !lo.is_finite() || !hi.is_finite() {
                return None;
            }
            if hi <= lo {
                if *value != lo {
                    return None;
                }
                key.push(0);
                continue;
            }
            if *value < lo || *value > hi {
                return None;
            }

            let idx = ((*value - lo) / (hi - lo) * self.buckets_per_axis as f64) as isize;
            key.push(idx.clamp(0, self.buckets_per_axis as isize - 1) as usize);
        }
        Some(key)
    }

    fn is_in_bounds(&self, x: &[f64]) -> bool {
        if x.len() != self.axis_min.len() {
            return false;
        }
        x.iter().enumerate().all(|(i, value)| {
            value.is_finite()
                && self.axis_min[i].is_finite()
                && self.axis_max[i].is_finite()
                && *value >= self.axis_min[i]
                && *value <= self.axis_max[i]
        })
    }

    fn update_axis_bounds(&mut self, x: &[f64]) -> bool {
        let mut changed = false;
        for (i, value) in x.iter().enumerate() {
            if *value < self.axis_min[i] {
                self.axis_min[i] = *value;
                changed = true;
            }
            if *value > self.axis_max[i] {
                self.axis_max[i] = *value;
                changed = true;
            }
        }
        changed
    }

    fn rebuild_buckets(&mut self) {
        let observations = self.observations();
        self.buckets.clear();
        for (x, y) in observations {
            let key = self.bucket_key(&x);
            self.buckets.entry(key).or_default().push((x, y));
        }
    }

    fn retire_from_fattest_bucket(&mut self) {
        // Eviction removes samples only. Dynamic regression bounds remain
        // monotonic; fixed correction bounds are configured at model creation.
        let Some(key) = self
            .buckets
            .iter()
            .max_by_key(|(_, bucket)| bucket.len())
            .map(|(key, _)| key.clone())
        else {
            return;
        };

        if let Some(bucket) = self.buckets.get_mut(&key) {
            if !bucket.is_empty() {
                bucket.remove(0);
                self.total_observations -= 1;
            }
            if bucket.is_empty() {
                self.buckets.remove(&key);
            }
        }
    }
}

pub(crate) fn median_ratio(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values = values
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

pub(crate) fn integer_sqrt(value: usize) -> usize {
    (value as f64).sqrt() as usize
}
