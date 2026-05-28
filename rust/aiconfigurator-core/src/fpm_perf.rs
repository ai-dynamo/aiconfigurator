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
const DEFAULT_MAX_NUM_TOKENS: u32 = 8192;
const DEFAULT_MAX_BATCH_SIZE: u32 = 512;
const DEFAULT_MAX_KV_TOKENS: u32 = 2_000_000;
const RELAXABLE_NEG_TOLERANCE: f64 = 1e-6;

/// In-memory tuning controls for `ForwardPassPerfModel`.
///
/// These defaults match the current planner regression behavior: retain a
/// bounded sliding sample set, wait for enough observations before predicting
/// from learned data, and bucket observations by workload kind.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForwardPassPerfOptions {
    /// Maximum retained observations across all buckets for each inferred workload kind.
    #[serde(default = "default_max_observations")]
    pub max_observations: usize,
    /// Minimum retained observations required before a regression fit or native
    /// correction is used for an inferred workload kind.
    #[serde(default = "default_min_observations")]
    pub min_observations: usize,
    /// Target bucket count for workload-specific sample retirement and correction lookup.
    #[serde(default = "default_bucket_count")]
    pub bucket_count: usize,
    /// Upper bound for the `sum_prefill_tokens` correction axis.
    ///
    /// Used by prefill and mixed/agg workload kinds. The lower bound is always `0`.
    #[serde(default = "default_max_num_tokens")]
    pub max_num_tokens: u32,
    /// Upper bound for the `num_decode_requests` correction axis.
    ///
    /// Used by the decode workload kind. The lower bound is always `0`.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: u32,
    /// Upper bound for the `sum_decode_kv_tokens` correction axis.
    ///
    /// Used by decode and mixed/agg workload kinds. The lower bound is always `0`.
    #[serde(default = "default_max_kv_tokens")]
    pub max_kv_tokens: u32,
}

impl Default for ForwardPassPerfOptions {
    fn default() -> Self {
        Self {
            max_observations: DEFAULT_MAX_OBSERVATIONS,
            min_observations: DEFAULT_MIN_OBSERVATIONS,
            bucket_count: DEFAULT_BUCKET_COUNT,
            max_num_tokens: DEFAULT_MAX_NUM_TOKENS,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            max_kv_tokens: DEFAULT_MAX_KV_TOKENS,
        }
    }
}

/// Current readiness and tuning state for a `ForwardPassPerfModel`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassPerfDiagnostics {
    /// Active prediction source. Native models become `aic_with_correction`
    /// after at least one inferred workload kind has enough correction samples.
    pub source: ForwardPassPerfSource,
    /// Whether the model can currently produce learned estimates for at least
    /// one workload kind, or why it cannot.
    pub readiness: ForwardPassPerfReadiness,
    /// Number of retained tuning observations across all inferred workload kinds.
    pub retained_observations: usize,
    /// Number of populated native-correction regions whose workload kind has at least
    /// `min_observations` total retained samples.
    pub correction_ready_buckets: usize,
    /// Fallback reason when `best_available` had to use regression instead of native AIC.
    pub last_warning: Option<String>,
}

/// Prediction backend currently used by `ForwardPassPerfModel`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassPerfSource {
    /// Strict native AIC estimator with no correction workload kind ready yet.
    Aic,
    /// Workload-specific regression fallback, used without native AIC support.
    FallbackRegression,
    /// Native AIC estimator with at least one learned correction workload kind.
    AicWithCorrection,
}

/// Readiness state reported by `ForwardPassPerfDiagnostics`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassPerfReadiness {
    /// The model has either native AIC support or enough learned data.
    Ready,
    /// Regression fallback exists, but does not yet have enough observations.
    InsufficientData,
    /// Native AIC was unavailable and `best_available` fell back to regression.
    UnsupportedConfig,
    /// Reserved for callers that surface rejected FPM input as diagnostics.
    InvalidInput,
}

/// Forward-pass-level performance model with optional online tuning.
///
/// This API intentionally stays at AIC's forward-pass abstraction. It does not
/// model TTFT, ITL, SLA, engine capacity, queueing policy, or Dynamo engine
/// limits. Callers pass FPMs for one engine iteration and receive one
/// forward-pass latency estimate in milliseconds.
///
/// The prefill/decode/mixed workload kind is inferred from each iteration's
/// `scheduled_requests` fields; it is not chosen at construction:
///
/// - prefill: scheduled prefill tokens and no scheduled decode work, using
///   `[sum_prefill_tokens]`
/// - decode: scheduled decode work and no scheduled prefill tokens, using
///   `[num_decode_requests, sum_decode_kv_tokens]`
/// - mixed/agg: both scheduled prefill and decode work, using
///   `[sum_prefill_tokens, sum_decode_kv_tokens]`
/// - empty: no scheduled prefill or decode work, estimates `0.0` and is not
///   used for tuning
///
/// Native correction grids use fixed constructor-time ranges from
/// `ForwardPassPerfOptions`: `max_num_tokens` bounds `sum_prefill_tokens`,
/// `max_batch_size` bounds `num_decode_requests`, and `max_kv_tokens` bounds
/// `sum_decode_kv_tokens`.
///
/// Queued request fields are accepted for FPM schema parity but ignored by this
/// forward-pass-level model. `estimate_forward_pass_time_ms` treats FPM as a
/// workload descriptor: it uses scheduled workload fields and ignores
/// `wall_time`. `tune_with_fpms` treats FPM as observed telemetry: it uses the
/// same scheduled workload fields as features and uses positive `wall_time` as
/// the observation target. For attention-DP configurations, the input for one
/// iteration is one FPM per attention-DP rank; tuning merges that list into one
/// observation by taking max-rank load features and max nonzero `wall_time`.
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
        corrections: WorkloadStores<CorrectionBuckets>,
    },
    Regression {
        regressions: WorkloadStores<BucketedRegression>,
    },
}

impl ForwardPassPerfModel {
    /// API:
    /// `ForwardPassPerfModel::from_native(config, options) -> Result<Self, AicError>`
    ///
    /// Description: create a strict native AIC forward-pass model.
    ///
    /// This constructor fails if `config` cannot be served by the native AIC
    /// estimator. Use `best_available` when unsupported native configs should
    /// fall back to the learned regression model.
    pub fn from_native(
        config: EngineConfig,
        options: ForwardPassPerfOptions,
    ) -> Result<Self, AicError> {
        validate_options(&options)?;
        let estimator = create_engine_step_estimator(config)?;
        Ok(Self {
            mode: ForwardPassPerfMode::Native {
                estimator,
                corrections: WorkloadStores::with_options(&options),
            },
            options,
            last_warning: None,
        })
    }

    /// API:
    /// `ForwardPassPerfModel::from_native_with_roots(config, options, systems_root, model_configs_root) -> Result<Self, AicError>`
    ///
    /// Description: create a strict native AIC forward-pass model with explicit
    /// data roots.
    ///
    /// This is the testable/root-overridable variant of `from_native` and has
    /// the same tuning and failure behavior.
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
                corrections: WorkloadStores::with_options(&options),
            },
            options,
            last_warning: None,
        })
    }

    /// API:
    /// `ForwardPassPerfModel::from_regression(options) -> Result<Self, AicError>`
    ///
    /// Description: create a regression-only forward-pass model.
    ///
    /// This mode is for native-AIC-unsupported models. It returns `None` from
    /// `estimate_forward_pass_time_ms` for non-empty iterations until the
    /// inferred workload kind has at least `options.min_observations` tuning samples.
    /// Correction factor getters always return `None` in this mode.
    pub fn from_regression(options: ForwardPassPerfOptions) -> Result<Self, AicError> {
        validate_options(&options)?;
        Ok(Self {
            mode: ForwardPassPerfMode::Regression {
                regressions: WorkloadStores::with_options(&options),
            },
            options,
            last_warning: None,
        })
    }

    /// API:
    /// `ForwardPassPerfModel::best_available(config, options) -> Result<Self, AicError>`
    ///
    /// Description: create a native model when possible, otherwise fall back to
    /// regression.
    ///
    /// Fallback reason is preserved in `diagnostics().last_warning`. The
    /// resulting model still uses the same FPM workload-kind inference and
    /// tuning input contract as `from_native` and `from_regression`.
    pub fn best_available(
        config: EngineConfig,
        options: ForwardPassPerfOptions,
    ) -> Result<Self, AicError> {
        match Self::from_native(config, options.clone()) {
            Ok(model) => Ok(model),
            Err(err) if can_fallback_to_regression(&err) => {
                Self::regression_with_warning(options, err)
            }
            Err(err) => Err(err),
        }
    }

    /// API:
    /// `ForwardPassPerfModel::best_available_with_roots(config, options, systems_root, model_configs_root) -> Result<Self, AicError>`
    ///
    /// Description: create a `best_available` model with explicit data roots.
    pub fn best_available_with_roots(
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
            Err(err) if can_fallback_to_regression(&err) => {
                Self::regression_with_warning(options, err)
            }
            Err(err) => Err(err),
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

    /// API:
    /// `model.estimate_forward_pass_time_ms(metrics_by_rank) -> Result<Option<f64>, AicError>`
    ///
    /// Description: estimate one forward-pass iteration in milliseconds.
    ///
    /// `metrics_by_rank` must contain the FPMs for a single engine iteration,
    /// one entry per attention-DP rank. Single-rank callers pass a one-element
    /// slice. The inferred workload kind uses only `scheduled_requests` as described on
    /// `ForwardPassPerfModel`; queued fields and `wall_time` are ignored for
    /// estimation.
    ///
    /// Native models return an AIC estimate immediately, multiplied by the
    /// correction factor for the matching workload region. Correction factors
    /// default to `1.0` for inferred workload kinds with fewer than
    /// `min_observations` total samples, empty regions, and queries outside the
    /// configured correction bounds in `ForwardPassPerfOptions`. Regression
    /// models return `Ok(None)` until the matching inferred workload kind has
    /// enough tuning samples. Empty scheduled work returns `Ok(Some(0.0))`.
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
                let corrected = native
                    * corrections
                        .store(feature.workload_kind)
                        .correction_factor_for(&feature.x);
                Ok(Some(corrected))
            }
            ForwardPassPerfMode::Regression { regressions } => {
                Ok(regressions.store(feature.workload_kind).predict(&feature.x))
            }
        }
    }

    /// API:
    /// `model.tune_with_fpms(iterations) -> Result<(), AicError>`
    ///
    /// Description: tune the model from observed FPM iterations.
    ///
    /// The outer slice is a list of observed iterations. Each inner slice is
    /// the per-attention-DP-rank FPM list for one iteration:
    /// `[[iter0_rank0, iter0_rank1], [iter1_rank0, iter1_rank1]]`.
    /// Single-rank callers still use one FPM per inner slice.
    ///
    /// For each non-empty iteration, this method infers the workload kind from
    /// scheduled request fields, takes max-rank load features, and uses the max
    /// finite positive `wall_time` across ranks as the observed latency target
    /// in milliseconds. Iterations with no scheduled work or no positive
    /// `wall_time` are ignored. Native models update the matching region's
    /// median `observed_ms / native_ms` correction factor. Regions are used only
    /// after their inferred workload kind has `min_observations` total samples;
    /// empty regions keep the default factor `1.0`. Observations outside the
    /// configured correction bounds are ignored by native correction models.
    /// Regression models learn a workload-specific linear fit.
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
                        .store_mut(observation.feature.workload_kind)
                        .add_observation(observation.feature.x, observation.wall_time_ms, native);
                }
                ForwardPassPerfMode::Regression { regressions } => {
                    regressions
                        .store_mut(observation.feature.workload_kind)
                        .add_observation(observation.feature.x, observation.wall_time_ms);
                }
            }
        }
        Ok(())
    }

    /// API:
    /// `model.diagnostics() -> ForwardPassPerfDiagnostics`
    ///
    /// Description: return the current backend, readiness, retained sample
    /// count, and fallback warning.
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

    /// API:
    /// `model.min_correction_factor() -> Option<f64>`
    ///
    /// Description: return the smallest ready native correction factor across
    /// all workload kinds.
    ///
    /// Returns `None` before any native correction workload kind has enough samples.
    /// Regression-only models also return `None`.
    pub fn min_correction_factor(&self) -> Option<f64> {
        self.correction_factors()
            .into_iter()
            .reduce(|a, b| a.min(b))
    }

    /// API:
    /// `model.max_correction_factor() -> Option<f64>`
    ///
    /// Description: return the largest ready native correction factor across
    /// all workload kinds.
    ///
    /// Returns `None` before any native correction workload kind has enough samples.
    /// Regression-only models also return `None`.
    pub fn max_correction_factor(&self) -> Option<f64> {
        self.correction_factors()
            .into_iter()
            .reduce(|a, b| a.max(b))
    }

    /// API:
    /// `model.avg_correction_factor() -> Option<f64>`
    ///
    /// Description: return the arithmetic mean of ready native correction
    /// factors across all workload kinds.
    ///
    /// Returns `None` before any native correction workload kind has enough samples.
    /// Regression-only models also return `None`.
    pub fn avg_correction_factor(&self) -> Option<f64> {
        let factors = self.correction_factors();
        if factors.is_empty() {
            None
        } else {
            Some(factors.iter().sum::<f64>() / factors.len() as f64)
        }
    }

    /// API:
    /// `model.options() -> &ForwardPassPerfOptions`
    ///
    /// Description: return the immutable tuning options used by this model.
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
enum WorkloadKind {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Clone, Debug)]
struct IterationFeatures {
    workload_kind: WorkloadKind,
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

        Ok(metrics_by_rank
            .iter()
            .filter_map(Self::from_single_rank)
            .max_by(|left, right| {
                left.load_score()
                    .partial_cmp(&right.load_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }))
    }

    fn from_single_rank(metrics: &ForwardPassMetrics) -> Option<Self> {
        let scheduled = &metrics.scheduled_requests;
        let has_prefill = scheduled.sum_prefill_tokens > 0;
        let has_decode = scheduled.num_decode_requests > 0 || scheduled.sum_decode_kv_tokens > 0;
        let feature = match (has_prefill, has_decode) {
            (false, false) => return None,
            (true, false) => Self {
                workload_kind: WorkloadKind::Prefill,
                x: vec![f64::from(scheduled.sum_prefill_tokens)],
            },
            (false, true) => Self {
                workload_kind: WorkloadKind::Decode,
                x: vec![
                    f64::from(scheduled.num_decode_requests),
                    f64::from(scheduled.sum_decode_kv_tokens),
                ],
            },
            (true, true) => Self {
                workload_kind: WorkloadKind::Mixed,
                x: vec![
                    f64::from(scheduled.sum_prefill_tokens),
                    f64::from(scheduled.sum_decode_kv_tokens),
                ],
            },
        };
        Some(feature)
    }

    fn load_score(&self) -> f64 {
        self.x.iter().sum()
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
struct WorkloadStores<T> {
    prefill: T,
    decode: T,
    mixed: T,
}

impl<T: WithOptions> WorkloadStores<T> {
    fn with_options(options: &ForwardPassPerfOptions) -> Self {
        Self {
            prefill: T::with_options(
                options,
                &[AxisRange::from_zero_to(options.max_num_tokens)],
                &[],
            ),
            decode: T::with_options(
                options,
                &[
                    AxisRange::from_zero_to(options.max_batch_size),
                    AxisRange::from_zero_to(options.max_kv_tokens),
                ],
                &[0],
            ),
            mixed: T::with_options(
                options,
                &[
                    AxisRange::from_zero_to(options.max_num_tokens),
                    AxisRange::from_zero_to(options.max_kv_tokens),
                ],
                &[],
            ),
        }
    }
}

impl<T: StoreStats> WorkloadStores<T> {
    fn observation_count(&self) -> usize {
        self.prefill.observation_count()
            + self.decode.observation_count()
            + self.mixed.observation_count()
    }

    fn any_ready(&self) -> bool {
        self.prefill.is_ready() || self.decode.is_ready() || self.mixed.is_ready()
    }
}

impl WorkloadStores<CorrectionBuckets> {
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

impl<T> WorkloadStores<T> {
    fn store(&self, workload_kind: WorkloadKind) -> &T {
        match workload_kind {
            WorkloadKind::Prefill => &self.prefill,
            WorkloadKind::Decode => &self.decode,
            WorkloadKind::Mixed => &self.mixed,
        }
    }

    fn store_mut(&mut self, workload_kind: WorkloadKind) -> &mut T {
        match workload_kind {
            WorkloadKind::Prefill => &mut self.prefill,
            WorkloadKind::Decode => &mut self.decode,
            WorkloadKind::Mixed => &mut self.mixed,
        }
    }
}

trait WithOptions {
    fn with_options(
        options: &ForwardPassPerfOptions,
        axis_ranges: &[AxisRange],
        relaxable: &[usize],
    ) -> Self;
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
    fn add_observation(&mut self, x: Vec<f64>, y: f64) {
        if self.samples.add(x, y) {
            self.fit = fit_linear(
                self.samples.observations(),
                self.ndim,
                self.min_observations,
                &self.relaxable,
            );
        }
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
    fn with_options(
        options: &ForwardPassPerfOptions,
        axis_ranges: &[AxisRange],
        _relaxable: &[usize],
    ) -> Self {
        Self {
            samples: BucketedSamples::new_fixed(options, axis_ranges),
            min_observations: options.min_observations,
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

    fn correction_factor_for(&self, x: &[f64]) -> f64 {
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
                .map(|(_, obs)| obs.observed_ms / obs.native_ms),
        )
        .unwrap_or(1.0)
    }

    fn ready_bucket_count(&self) -> usize {
        if self.is_ready() {
            self.samples.buckets.len()
        } else {
            0
        }
    }

    fn correction_factors(&self) -> Vec<f64> {
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
    fixed_bounds: bool,
    buckets_per_axis: usize,
    max_observations: usize,
}

#[derive(Clone, Copy, Debug)]
struct AxisRange {
    min: f64,
    max: f64,
}

impl AxisRange {
    fn from_zero_to(max: u32) -> Self {
        Self {
            min: 0.0,
            max: f64::from(max),
        }
    }
}

impl<T: Clone> BucketedSamples<T> {
    fn new_dynamic(options: &ForwardPassPerfOptions, ndim: usize) -> Self {
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

    fn new_fixed(options: &ForwardPassPerfOptions, axis_ranges: &[AxisRange]) -> Self {
        let mut samples = Self::new_dynamic(options, axis_ranges.len());
        samples.axis_min = axis_ranges.iter().map(|range| range.min).collect();
        samples.axis_max = axis_ranges.iter().map(|range| range.max).collect();
        samples.fixed_bounds = true;
        samples
    }

    fn add(&mut self, x: Vec<f64>, y: T) -> bool {
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

    fn bucket_key_if_in_bounds(&self, x: &[f64]) -> Option<Vec<usize>> {
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
        return Err(invalid_perf_options("max_observations must be >= 1"));
    }
    if options.min_observations == 0 {
        return Err(invalid_perf_options("min_observations must be >= 1"));
    }
    if options.bucket_count == 0 {
        return Err(invalid_perf_options("bucket_count must be >= 1"));
    }
    if options.max_num_tokens == 0 {
        return Err(invalid_perf_options("max_num_tokens must be >= 1"));
    }
    if options.max_batch_size == 0 {
        return Err(invalid_perf_options("max_batch_size must be >= 1"));
    }
    if options.max_kv_tokens == 0 {
        return Err(invalid_perf_options("max_kv_tokens must be >= 1"));
    }
    if options.min_observations > options.max_observations {
        return Err(invalid_perf_options(
            "min_observations must be <= max_observations",
        ));
    }
    let sqrt = integer_sqrt(options.bucket_count);
    if sqrt * sqrt != options.bucket_count {
        return Err(invalid_perf_options(
            "bucket_count must be a perfect square",
        ));
    }
    Ok(())
}

fn invalid_perf_options(message: &str) -> AicError {
    AicError::InvalidEngineConfig(format!("invalid forward pass perf options: {message}"))
}

fn can_fallback_to_regression(err: &AicError) -> bool {
    matches!(
        err,
        AicError::UnsupportedModel(_)
            | AicError::DataRoot(_)
            | AicError::ModelConfig(_)
            | AicError::PerfDatabase(_)
            | AicError::Io { .. }
            | AicError::Csv { .. }
    )
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

fn default_max_num_tokens() -> u32 {
    DEFAULT_MAX_NUM_TOKENS
}

fn default_max_batch_size() -> u32 {
    DEFAULT_MAX_BATCH_SIZE
}

fn default_max_kv_tokens() -> u32 {
    DEFAULT_MAX_KV_TOKENS
}
