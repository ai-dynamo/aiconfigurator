// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    create_engine_step_estimator, validate_forward_pass_metrics, AicError, EngineConfig,
    EngineStepEstimator, ForwardPassMetrics,
};

const DEFAULT_MAX_OBSERVATIONS: usize = 64;
const DEFAULT_MIN_OBSERVATIONS: usize = 5;
const DEFAULT_BUCKET_COUNT: usize = 16;
const RELAXABLE_NEG_TOLERANCE: f64 = 1e-6;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForwardPassPerfOptions {
    #[serde(default = "default_max_observations")]
    pub max_observations: usize,
    #[serde(default = "default_min_observations")]
    pub min_observations: usize,
    #[serde(default = "default_bucket_count")]
    pub bucket_count: usize,
}

impl Default for ForwardPassPerfOptions {
    fn default() -> Self {
        Self {
            max_observations: DEFAULT_MAX_OBSERVATIONS,
            min_observations: DEFAULT_MIN_OBSERVATIONS,
            bucket_count: DEFAULT_BUCKET_COUNT,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassPerfDiagnostics {
    pub source: ForwardPassPerfSource,
    pub readiness: ForwardPassPerfReadiness,
    pub retained_observations: usize,
    pub correction_ready_buckets: usize,
    pub last_warning: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassPerfSource {
    Aic,
    FallbackRegression,
    AicWithCorrection,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassPerfReadiness {
    Ready,
    InsufficientData,
    UnsupportedConfig,
    InvalidInput,
}

#[derive(Clone, Debug)]
pub struct ForwardPassPerfModel {
    mode: ForwardPassPerfMode,
    options: ForwardPassPerfOptions,
    last_warning: Option<String>,
}

#[derive(Clone, Debug)]
enum ForwardPassPerfMode {
    Native {
        estimator: EngineStepEstimator,
        corrections: ShapeStores<CorrectionBuckets>,
    },
    Regression {
        regressions: ShapeStores<BucketedRegression>,
    },
}

impl ForwardPassPerfModel {
    pub fn from_native(
        config: EngineConfig,
        options: ForwardPassPerfOptions,
    ) -> Result<Self, AicError> {
        validate_options(&options)?;
        let estimator = create_engine_step_estimator(config)?;
        Ok(Self {
            mode: ForwardPassPerfMode::Native {
                estimator,
                corrections: ShapeStores::with_options(&options),
            },
            options,
            last_warning: None,
        })
    }

    pub fn from_native_with_roots(
        config: EngineConfig,
        options: ForwardPassPerfOptions,
        systems_root: impl AsRef<Path>,
        model_configs_root: impl AsRef<Path>,
    ) -> Result<Self, AicError> {
        validate_options(&options)?;
        let estimator =
            EngineStepEstimator::from_config_with_roots(config, systems_root, model_configs_root)?;
        Ok(Self {
            mode: ForwardPassPerfMode::Native {
                estimator,
                corrections: ShapeStores::with_options(&options),
            },
            options,
            last_warning: None,
        })
    }

    pub fn from_regression(options: ForwardPassPerfOptions) -> Result<Self, AicError> {
        validate_options(&options)?;
        Ok(Self {
            mode: ForwardPassPerfMode::Regression {
                regressions: ShapeStores::with_options(&options),
            },
            options,
            last_warning: None,
        })
    }

    pub fn auto(config: EngineConfig, options: ForwardPassPerfOptions) -> Result<Self, AicError> {
        match Self::from_native(config, options.clone()) {
            Ok(model) => Ok(model),
            Err(err) => Self::regression_with_warning(options, err),
        }
    }

    pub fn auto_with_roots(
        config: EngineConfig,
        options: ForwardPassPerfOptions,
        systems_root: impl AsRef<Path>,
        model_configs_root: impl AsRef<Path>,
    ) -> Result<Self, AicError> {
        match Self::from_native_with_roots(
            config,
            options.clone(),
            systems_root,
            model_configs_root,
        ) {
            Ok(model) => Ok(model),
            Err(err) => Self::regression_with_warning(options, err),
        }
    }

    fn regression_with_warning(
        options: ForwardPassPerfOptions,
        err: AicError,
    ) -> Result<Self, AicError> {
        let mut model = Self::from_regression(options)?;
        model.last_warning = Some(format!(
            "native forward-pass estimator unavailable; using fallback regression: {err}"
        ));
        Ok(model)
    }

    pub fn estimate_forward_pass_time_ms(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Option<f64>, AicError> {
        let feature = IterationFeatures::from_metrics(metrics_by_rank)?;
        let Some(feature) = feature else {
            return Ok(Some(0.0));
        };

        match &self.mode {
            ForwardPassPerfMode::Native {
                estimator,
                corrections,
            } => {
                let native = estimator.forward_pass_time_ms(metrics_by_rank)?;
                let corrected = corrections
                    .store(feature.shape)
                    .correction_for(&feature.x)
                    .map(|factor| native * factor)
                    .unwrap_or(native);
                Ok(Some(corrected))
            }
            ForwardPassPerfMode::Regression { regressions } => {
                Ok(regressions.store(feature.shape).predict(&feature.x))
            }
        }
    }

    pub fn tune_with_fpms(
        &mut self,
        iterations: &[Vec<ForwardPassMetrics>],
    ) -> Result<(), AicError> {
        for metrics_by_rank in iterations {
            let observation = IterationObservation::from_metrics(metrics_by_rank)?;
            let Some(observation) = observation else {
                continue;
            };

            match &mut self.mode {
                ForwardPassPerfMode::Native {
                    estimator,
                    corrections,
                } => {
                    let native = estimator.forward_pass_time_ms(metrics_by_rank)?;
                    corrections
                        .store_mut(observation.feature.shape)
                        .add_observation(observation.feature.x, observation.wall_time_ms, native);
                }
                ForwardPassPerfMode::Regression { regressions } => {
                    regressions
                        .store_mut(observation.feature.shape)
                        .add_observation(observation.feature.x, observation.wall_time_ms);
                }
            }
        }
        Ok(())
    }

    pub fn diagnostics(&self) -> ForwardPassPerfDiagnostics {
        match &self.mode {
            ForwardPassPerfMode::Native { corrections, .. } => {
                let ready_buckets = corrections.ready_bucket_count();
                ForwardPassPerfDiagnostics {
                    source: if ready_buckets > 0 {
                        ForwardPassPerfSource::AicWithCorrection
                    } else {
                        ForwardPassPerfSource::Aic
                    },
                    readiness: ForwardPassPerfReadiness::Ready,
                    retained_observations: corrections.observation_count(),
                    correction_ready_buckets: ready_buckets,
                    last_warning: self.last_warning.clone(),
                }
            }
            ForwardPassPerfMode::Regression { regressions } => {
                let ready = regressions.any_ready();
                ForwardPassPerfDiagnostics {
                    source: ForwardPassPerfSource::FallbackRegression,
                    readiness: if ready {
                        ForwardPassPerfReadiness::Ready
                    } else if self.last_warning.is_some() {
                        ForwardPassPerfReadiness::UnsupportedConfig
                    } else {
                        ForwardPassPerfReadiness::InsufficientData
                    },
                    retained_observations: regressions.observation_count(),
                    correction_ready_buckets: 0,
                    last_warning: self.last_warning.clone(),
                }
            }
        }
    }

    pub fn min_correction_factor(&self) -> Option<f64> {
        self.correction_factors()
            .into_iter()
            .reduce(|a, b| a.min(b))
    }

    pub fn max_correction_factor(&self) -> Option<f64> {
        self.correction_factors()
            .into_iter()
            .reduce(|a, b| a.max(b))
    }

    pub fn avg_correction_factor(&self) -> Option<f64> {
        let factors = self.correction_factors();
        if factors.is_empty() {
            None
        } else {
            Some(factors.iter().sum::<f64>() / factors.len() as f64)
        }
    }

    pub fn options(&self) -> &ForwardPassPerfOptions {
        &self.options
    }

    fn correction_factors(&self) -> Vec<f64> {
        match &self.mode {
            ForwardPassPerfMode::Native { corrections, .. } => corrections.correction_factors(),
            ForwardPassPerfMode::Regression { .. } => Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FeatureShape {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Clone, Debug)]
struct IterationFeatures {
    shape: FeatureShape,
    x: Vec<f64>,
}

impl IterationFeatures {
    fn from_metrics(metrics_by_rank: &[ForwardPassMetrics]) -> Result<Option<Self>, AicError> {
        if metrics_by_rank.is_empty() {
            return Err(AicError::InvalidForwardPassMetrics(
                "at least one attention-DP rank metric is required".to_string(),
            ));
        }
        for metrics in metrics_by_rank {
            validate_forward_pass_metrics(metrics)?;
        }

        let mut max_prefill_tokens = 0_u32;
        let mut max_decode_requests = 0_u32;
        let mut max_decode_kv_tokens = 0_u32;
        for metrics in metrics_by_rank {
            let scheduled = &metrics.scheduled_requests;
            max_prefill_tokens = max_prefill_tokens.max(scheduled.sum_prefill_tokens);
            max_decode_requests = max_decode_requests.max(scheduled.num_decode_requests);
            max_decode_kv_tokens = max_decode_kv_tokens.max(scheduled.sum_decode_kv_tokens);
        }

        let has_prefill = max_prefill_tokens > 0;
        let has_decode = max_decode_requests > 0 || max_decode_kv_tokens > 0;
        let feature = match (has_prefill, has_decode) {
            (false, false) => return Ok(None),
            (true, false) => Self {
                shape: FeatureShape::Prefill,
                x: vec![f64::from(max_prefill_tokens)],
            },
            (false, true) => Self {
                shape: FeatureShape::Decode,
                x: vec![
                    f64::from(max_decode_requests),
                    f64::from(max_decode_kv_tokens),
                ],
            },
            (true, true) => Self {
                shape: FeatureShape::Mixed,
                x: vec![
                    f64::from(max_prefill_tokens),
                    f64::from(max_decode_kv_tokens),
                ],
            },
        };
        Ok(Some(feature))
    }
}

#[derive(Clone, Debug)]
struct IterationObservation {
    feature: IterationFeatures,
    wall_time_ms: f64,
}

impl IterationObservation {
    fn from_metrics(metrics_by_rank: &[ForwardPassMetrics]) -> Result<Option<Self>, AicError> {
        let Some(feature) = IterationFeatures::from_metrics(metrics_by_rank)? else {
            return Ok(None);
        };
        let wall_time = metrics_by_rank
            .iter()
            .map(|metrics| metrics.wall_time)
            .filter(|wall_time| wall_time.is_finite() && *wall_time > 0.0)
            .fold(0.0_f64, f64::max);
        if wall_time <= 0.0 {
            return Ok(None);
        }
        Ok(Some(Self {
            feature,
            wall_time_ms: wall_time * 1000.0,
        }))
    }
}

#[derive(Clone, Debug)]
struct ShapeStores<T> {
    prefill: T,
    decode: T,
    mixed: T,
}

impl<T: WithOptions> ShapeStores<T> {
    fn with_options(options: &ForwardPassPerfOptions) -> Self {
        Self {
            prefill: T::with_options(options, 1, &[]),
            decode: T::with_options(options, 2, &[0]),
            mixed: T::with_options(options, 2, &[]),
        }
    }
}

impl<T: StoreStats> ShapeStores<T> {
    fn observation_count(&self) -> usize {
        self.prefill.observation_count()
            + self.decode.observation_count()
            + self.mixed.observation_count()
    }

    fn any_ready(&self) -> bool {
        self.prefill.is_ready() || self.decode.is_ready() || self.mixed.is_ready()
    }
}

impl ShapeStores<CorrectionBuckets> {
    fn ready_bucket_count(&self) -> usize {
        self.prefill.ready_bucket_count()
            + self.decode.ready_bucket_count()
            + self.mixed.ready_bucket_count()
    }

    fn correction_factors(&self) -> Vec<f64> {
        let mut factors = self.prefill.correction_factors();
        factors.extend(self.decode.correction_factors());
        factors.extend(self.mixed.correction_factors());
        factors
    }
}

impl<T> ShapeStores<T> {
    fn store(&self, shape: FeatureShape) -> &T {
        match shape {
            FeatureShape::Prefill => &self.prefill,
            FeatureShape::Decode => &self.decode,
            FeatureShape::Mixed => &self.mixed,
        }
    }

    fn store_mut(&mut self, shape: FeatureShape) -> &mut T {
        match shape {
            FeatureShape::Prefill => &mut self.prefill,
            FeatureShape::Decode => &mut self.decode,
            FeatureShape::Mixed => &mut self.mixed,
        }
    }
}

trait WithOptions {
    fn with_options(options: &ForwardPassPerfOptions, ndim: usize, relaxable: &[usize]) -> Self;
}

trait StoreStats {
    fn observation_count(&self) -> usize;
    fn is_ready(&self) -> bool;
}

#[derive(Clone, Debug)]
struct BucketedRegression {
    samples: BucketedSamples<f64>,
    ndim: usize,
    min_observations: usize,
    relaxable: Vec<usize>,
    fit: Option<LinearFit>,
}

impl WithOptions for BucketedRegression {
    fn with_options(options: &ForwardPassPerfOptions, ndim: usize, relaxable: &[usize]) -> Self {
        Self {
            samples: BucketedSamples::new(options, ndim),
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
    fn add_observation(&mut self, x: Vec<f64>, y: f64) {
        self.samples.add(x, y);
        self.fit = fit_linear(
            self.samples.observations(),
            self.ndim,
            self.min_observations,
            &self.relaxable,
        );
    }

    fn predict(&self, x: &[f64]) -> Option<f64> {
        self.fit.as_ref().map(|fit| fit.predict(x).max(1e-6))
    }
}

#[derive(Clone, Debug)]
struct CorrectionBuckets {
    samples: BucketedSamples<CorrectionObservation>,
    min_observations: usize,
}

#[derive(Clone, Copy, Debug)]
struct CorrectionObservation {
    observed_ms: f64,
    native_ms: f64,
}

impl WithOptions for CorrectionBuckets {
    fn with_options(options: &ForwardPassPerfOptions, ndim: usize, _relaxable: &[usize]) -> Self {
        Self {
            samples: BucketedSamples::new(options, ndim),
            min_observations: options.min_observations,
        }
    }
}

impl StoreStats for CorrectionBuckets {
    fn observation_count(&self) -> usize {
        self.samples.total_observations
    }

    fn is_ready(&self) -> bool {
        self.ready_bucket_count() > 0
    }
}

impl CorrectionBuckets {
    fn add_observation(&mut self, x: Vec<f64>, observed_ms: f64, native_ms: f64) {
        if native_ms.is_finite() && native_ms > 0.0 && observed_ms.is_finite() && observed_ms > 0.0
        {
            self.samples.add(
                x,
                CorrectionObservation {
                    observed_ms,
                    native_ms,
                },
            );
        }
    }

    fn correction_for(&self, x: &[f64]) -> Option<f64> {
        let key = self.samples.bucket_key(x);
        let bucket = self.samples.buckets.get(&key)?;
        if bucket.len() < self.min_observations {
            return None;
        }
        median_ratio(
            bucket
                .iter()
                .map(|(_, obs)| obs.observed_ms / obs.native_ms),
        )
    }

    fn ready_bucket_count(&self) -> usize {
        self.samples
            .buckets
            .values()
            .filter(|bucket| bucket.len() >= self.min_observations)
            .count()
    }

    fn correction_factors(&self) -> Vec<f64> {
        self.samples
            .buckets
            .values()
            .filter(|bucket| bucket.len() >= self.min_observations)
            .filter_map(|bucket| {
                median_ratio(
                    bucket
                        .iter()
                        .map(|(_, obs)| obs.observed_ms / obs.native_ms),
                )
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct BucketedSamples<T> {
    buckets: HashMap<Vec<usize>, Vec<(Vec<f64>, T)>>,
    total_observations: usize,
    axis_min: Vec<f64>,
    axis_max: Vec<f64>,
    buckets_per_axis: usize,
    max_observations: usize,
}

impl<T: Clone> BucketedSamples<T> {
    fn new(options: &ForwardPassPerfOptions, ndim: usize) -> Self {
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
            buckets_per_axis: buckets_per_axis.max(1),
            max_observations: options.max_observations,
        }
    }

    fn add(&mut self, x: Vec<f64>, y: T) {
        let bounds_changed = self.update_axis_bounds(&x);
        if bounds_changed && self.total_observations > 0 {
            self.rebuild_buckets();
        }

        let key = self.bucket_key(&x);
        self.buckets.entry(key).or_default().push((x, y));
        self.total_observations += 1;

        if self.total_observations > self.max_observations {
            self.retire_from_fattest_bucket();
        }
    }

    fn observations(&self) -> Vec<(Vec<f64>, T)> {
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

    let solution = solve_linear_system(lhs, rhs)?;
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

fn median_ratio(values: impl Iterator<Item = f64>) -> Option<f64> {
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

fn validate_options(options: &ForwardPassPerfOptions) -> Result<(), AicError> {
    if options.max_observations == 0 {
        return Err(AicError::InvalidForwardPassPerfOptions(
            "max_observations must be >= 1".to_string(),
        ));
    }
    if options.min_observations == 0 {
        return Err(AicError::InvalidForwardPassPerfOptions(
            "min_observations must be >= 1".to_string(),
        ));
    }
    if options.bucket_count == 0 {
        return Err(AicError::InvalidForwardPassPerfOptions(
            "bucket_count must be >= 1".to_string(),
        ));
    }
    let sqrt = integer_sqrt(options.bucket_count);
    if sqrt * sqrt != options.bucket_count {
        return Err(AicError::InvalidForwardPassPerfOptions(
            "bucket_count must be a perfect square".to_string(),
        ));
    }
    Ok(())
}

fn integer_sqrt(value: usize) -> usize {
    (value as f64).sqrt() as usize
}

fn default_max_observations() -> usize {
    DEFAULT_MAX_OBSERVATIONS
}

fn default_min_observations() -> usize {
    DEFAULT_MIN_OBSERVATIONS
}

fn default_bucket_count() -> usize {
    DEFAULT_BUCKET_COUNT
}
