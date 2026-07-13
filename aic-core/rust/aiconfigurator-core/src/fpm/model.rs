// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward-pass perf model: native AIC estimate with optional online correction
//! and regression fallback, plus readiness/diagnostics.
//!
//! The `Native` variant holds an `Arc<Engine>` and the native estimate routes
//! through [`crate::engine::Engine::forward_pass_time_ms`]. The online
//! correction / regression / diagnostics / readiness logic is engine-agnostic.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::{AicError, EngineConfig, ForwardPassMetrics};

use super::correction::CorrectionBuckets;
use super::metrics::validate_forward_pass_metrics;
use super::options::{validate_options, ForwardPassPerfOptions};
use super::regression::BucketedRegression;
use super::samples::{AxisRange, StoreStats, WithOptions};

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
        /// Compiled engine. `Arc` so `ForwardPassPerfModel` stays `Clone`
        /// (the `Engine` itself is not `Clone`); cheap clones share the loaded
        /// op lists + perf-database tree.
        engine: Arc<Engine>,
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
    /// Compiles `config` into an [`Engine`] by crossing into Python once
    /// (mirroring [`crate::build_aic_engine`]): `compile_engine` walks the model
    /// and returns bincoded spec bytes, then [`Engine::from_spec_bytes`] loads
    /// the matching perf database. This constructor fails if `config` cannot be
    /// compiled. Use `best_available` when unsupported native configs should
    /// fall back to the learned regression model.
    pub fn from_native(
        config: EngineConfig,
        options: ForwardPassPerfOptions,
    ) -> Result<Self, AicError> {
        validate_options(&options)?;
        let engine = build_engine_via_python(&config, None)?;
        Ok(Self::from_engine(Arc::new(engine), options))
    }

    /// API:
    /// `ForwardPassPerfModel::from_native_with_roots(config, options, systems_root) -> Result<Self, AicError>`
    ///
    /// Description: create a strict native AIC forward-pass model with an
    /// explicit `systems/` data root (forwarded to `compile_engine` and used to
    /// load the perf database). Same tuning and failure behavior as
    /// `from_native`.
    pub fn from_native_with_roots(
        config: EngineConfig,
        options: ForwardPassPerfOptions,
        systems_root: impl AsRef<Path>,
    ) -> Result<Self, AicError> {
        validate_options(&options)?;
        let engine = build_engine_via_python(&config, Some(systems_root.as_ref()))?;
        Ok(Self::from_engine(Arc::new(engine), options))
    }

    /// Internal: build a native model directly from an already-compiled
    /// [`Engine`]. Holds the actual native-mode logic; the public `from_native`
    /// constructors compile the `Engine` (crossing into Python) and call this.
    /// Used by the `#[cfg(test)]` suite to construct a native model from a
    /// hand-built fixture `Engine` without Python.
    pub(crate) fn from_engine(engine: Arc<Engine>, options: ForwardPassPerfOptions) -> Self {
        Self {
            mode: ForwardPassPerfMode::Native {
                engine,
                corrections: WorkloadStores::with_options(&options),
            },
            options,
            last_warning: None,
        }
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
    /// `ForwardPassPerfModel::best_available_with_roots(config, options, systems_root) -> Result<Self, AicError>`
    ///
    /// Description: create a `best_available` model with an explicit `systems/`
    /// data root.
    pub fn best_available_with_roots(
        config: EngineConfig,
        options: ForwardPassPerfOptions,
        systems_root: impl AsRef<Path>,
    ) -> Result<Self, AicError> {
        match Self::from_native_with_roots(config, options.clone(), systems_root) {
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
    ///
    /// Pure Rust over the `Engine` — no Python re-entry.
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
                engine,
                corrections,
            } => {
                let native = engine.forward_pass_time_ms(metrics_by_rank)?;
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
    ///
    /// Pure Rust over the `Engine` — no Python re-entry.
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
                    engine,
                    corrections,
                } => {
                    let native = engine.forward_pass_time_ms(metrics_by_rank)?;
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

/// Build a compiled [`Engine`] from an [`EngineConfig`] by crossing into Python
/// once to run `aiconfigurator.sdk.engine.compile_engine`, then loading the
/// matching perf database via [`Engine::from_spec_bytes`]. Mirrors
/// [`crate::build_aic_engine`] but takes an `EngineConfig` (mapping its
/// modular fields onto the flat `compile_engine` kwargs).
///
/// `systems_root` overrides the bundled `systems/` dir for BOTH the
/// `compile_engine` call (`systems_path` kwarg) and the Rust-side perf-DB load.
fn build_engine_via_python(
    config: &EngineConfig,
    systems_root: Option<&Path>,
) -> Result<Engine, AicError> {
    // `compile_engine`'s `systems_path` kwarg: explicit override -> config's
    // own `systems_path` -> None (Python resolves it).
    let systems_path: Option<PathBuf> = systems_root
        .map(PathBuf::from)
        .or_else(|| config.systems_path.clone());
    // A non-UTF-8 override path cannot be passed through the Python kwarg; fail
    // loudly rather than silently dropping the override.
    let systems_path_str = match systems_path.as_ref() {
        Some(p) => Some(p.to_str().ok_or_else(|| {
            AicError::InvalidEngineConfig(format!(
                "systems_path is not valid UTF-8: {}",
                p.display()
            ))
        })?),
        None => None,
    };

    crate::py::compile_engine_to_engine(config, systems_path_str)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WorkloadKind {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Clone, Debug)]
pub(crate) struct IterationFeatures {
    pub(crate) workload_kind: WorkloadKind,
    pub(crate) x: Vec<f64>,
}

impl IterationFeatures {
    pub(crate) fn from_metrics(
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Option<Self>, AicError> {
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
pub(crate) struct IterationObservation {
    pub(crate) feature: IterationFeatures,
    pub(crate) wall_time_ms: f64,
}

impl IterationObservation {
    pub(crate) fn from_metrics(
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Option<Self>, AicError> {
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
pub(crate) struct WorkloadStores<T> {
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

/// Decide whether `best_available` should fall back to regression instead of
/// propagating `err`. Covers the unsupported-model / data-availability errors
/// that mean "this model can't be served natively". A failed native build via
/// Python `compile_engine` surfaces as [`AicError::UnsupportedModel`] (see
/// `py::compile_engine_from_flat`), which is covered here.
///
/// [`AicError::InvalidEngineConfig`] is deliberately NOT fallback-safe: it is
/// used for hard caller/config errors (e.g. a non-UTF-8 `systems_path`, invalid
/// FPM options, a malformed spec). Those must surface rather than silently
/// degrade `best_available` to regression mode.
fn can_fallback_to_regression(err: &AicError) -> bool {
    matches!(
        err,
        AicError::UnsupportedModel(_)
            | AicError::DataRoot(_)
            | AicError::ModelConfig(_)
            | AicError::PerfDatabase(_)
            | AicError::Io { .. }
            | AicError::Parquet { .. }
    )
}
