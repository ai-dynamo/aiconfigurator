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
use super::options::{validate_options, validate_regression_options, ForwardPassPerfOptions};
use super::regression::BucketedRegression;
use super::samples::{AxisRange, StoreStats, WithOptions};

/// Current readiness and tuning state for a `ForwardPassPerfModel`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassPerfDiagnostics {
    /// Active prediction source. Native models become `aic_with_correction`
    /// after at least one inferred workload kind has enough correction samples.
    pub source: ForwardPassPerfSource,
    /// Whether the active model can currently produce estimates, or why it
    /// cannot. Native models are immediately ready; regression readiness is
    /// determined by its single role-bound store.
    pub readiness: ForwardPassPerfReadiness,
    /// Number of retained tuning observations. This is the total across the
    /// three inferred workload kinds for Native and the single store count for
    /// Regression.
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
    /// Worker-type-bound regression fallback, used without native AIC support.
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

/// Engine-level worker role used by the regression fallback.
///
/// Unlike the native model's inferred rank-local workload kind, this identity
/// is fixed when the regression model is constructed and applies to every DP
/// rank and iteration observed by that model.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForwardPassWorkerType {
    /// A dedicated Prefill engine.
    Prefill,
    /// A dedicated Decode engine.
    Decode,
    /// An engine that may execute Prefill, Decode, or both.
    Aggregated,
}

/// Forward-pass-level performance model with optional online tuning.
///
/// This API intentionally stays at AIC's forward-pass abstraction. It does not
/// model TTFT, ITL, SLA, engine capacity, queueing policy, or Dynamo engine
/// limits. Callers pass FPMs for one engine iteration and receive one
/// forward-pass latency estimate in milliseconds.
///
/// Native AIC preserves its existing prefill/decode/mixed workload inference
/// from each iteration's `scheduled_requests` fields:
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
/// Regression instead binds one immutable [`ForwardPassWorkerType`] at
/// construction and owns one two-dimensional store. Its common axis order is
/// `[critical attention, global FFN/MoE]`; Prefill and Decode enforce strict
/// role compatibility, while Aggregated accepts any phase composition.
/// Regression retention buckets use `log1p` coordinates, but fitting and
/// prediction use standardized raw feature values.
///
/// Native correction grids use fixed constructor-time ranges from
/// `ForwardPassPerfOptions`: `max_num_tokens` bounds `sum_prefill_tokens`,
/// `max_batch_size` bounds `num_decode_requests`, and `max_kv_tokens` bounds
/// `sum_decode_kv_tokens`. `min_faster_correction_factor` and
/// `max_slower_correction_factor` place independent absolute bounds on learned
/// native correction factors in each direction, defaulting to `0.5` and `2.0`.
/// Callers may explicitly disable either bound.
///
/// Queued request fields are accepted for FPM schema parity but ignored by this
/// forward-pass-level model. `estimate_forward_pass_time_ms` treats FPM as a
/// workload descriptor: it uses scheduled workload fields and ignores
/// `wall_time`. `tune_with_fpms` treats FPM as observed telemetry: it uses the
/// same scheduled workload fields as features and uses positive `wall_time` as
/// the observation target. For attention-DP configurations, the input for one
/// iteration is one FPM per attention-DP rank; tuning merges that list into one
/// observation using the active backend's cross-rank feature rule and the
/// maximum finite positive `wall_time`.
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
        worker_type: ForwardPassWorkerType,
        regression: BucketedRegression,
    },
}

impl ForwardPassPerfModel {
    /// API:
    /// `ForwardPassPerfModel::from_native(config, options) -> Result<Self, AicError>`
    ///
    /// Description: create a strict native AIC forward-pass model.
    ///
    /// Compiles `config` into an [`Engine`] by crossing into Python once
    /// (mirroring [`crate::AicEngineBuilder`]): `compile_engine` walks the model
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
    /// `ForwardPassPerfModel::from_regression(worker_type, options) -> Result<Self, AicError>`
    ///
    /// Description: create a regression-only forward-pass model.
    ///
    /// This mode is for native-AIC-unsupported models. It returns `None` from
    /// `estimate_forward_pass_time_ms` for non-empty iterations until the
    /// role-bound store has at least `options.min_observations` tuning samples.
    /// Correction factor getters always return `None` in this mode.
    pub fn from_regression(
        worker_type: ForwardPassWorkerType,
        options: ForwardPassPerfOptions,
    ) -> Result<Self, AicError> {
        validate_regression_options(&options)?;
        Ok(Self {
            mode: ForwardPassPerfMode::Regression {
                worker_type,
                regression: BucketedRegression::new(&options),
            },
            options,
            last_warning: None,
        })
    }

    /// API:
    /// `ForwardPassPerfModel::best_available(config, worker_type, options) -> Result<Self, AicError>`
    ///
    /// Description: create a native model when possible, otherwise fall back to
    /// regression.
    ///
    /// Fallback reason is preserved in `diagnostics().last_warning`. The
    /// A successful native construction preserves native workload inference
    /// and ignores `worker_type` and regression-only weights. A fallback is
    /// bound to `worker_type` and validates those weights when it is created.
    pub fn best_available(
        config: EngineConfig,
        worker_type: ForwardPassWorkerType,
        options: ForwardPassPerfOptions,
    ) -> Result<Self, AicError> {
        match Self::from_native(config, options.clone()) {
            Ok(model) => Ok(model),
            Err(err) if can_fallback_to_regression(&err) => {
                Self::regression_with_warning(worker_type, options, err)
            }
            Err(err) => Err(err),
        }
    }

    /// API:
    /// `ForwardPassPerfModel::best_available_with_roots(config, worker_type, options, systems_root) -> Result<Self, AicError>`
    ///
    /// Description: create a `best_available` model with an explicit `systems/`
    /// data root.
    pub fn best_available_with_roots(
        config: EngineConfig,
        worker_type: ForwardPassWorkerType,
        options: ForwardPassPerfOptions,
        systems_root: impl AsRef<Path>,
    ) -> Result<Self, AicError> {
        match Self::from_native_with_roots(config, options.clone(), systems_root) {
            Ok(model) => Ok(model),
            Err(err) if can_fallback_to_regression(&err) => {
                Self::regression_with_warning(worker_type, options, err)
            }
            Err(err) => Err(err),
        }
    }

    fn regression_with_warning(
        worker_type: ForwardPassWorkerType,
        options: ForwardPassPerfOptions,
        err: AicError,
    ) -> Result<Self, AicError> {
        let mut model = Self::from_regression(worker_type, options)?;
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
    /// slice. Native workload inference and role-bound regression extraction
    /// use only `scheduled_requests`; queued fields and `wall_time` are ignored
    /// for estimation.
    ///
    /// Native models return an AIC estimate immediately, multiplied by the
    /// correction factor for the matching workload region. Correction factors
    /// default to `1.0` for inferred workload kinds with fewer than
    /// `min_observations` total samples, empty regions, and queries outside the
    /// configured correction-grid workload ranges in
    /// `ForwardPassPerfOptions`. Regression models return `Ok(None)` until
    /// their single role-bound store has a ready fit. Empty scheduled work
    /// returns `Ok(Some(0.0))`.
    ///
    /// Pure Rust over the `Engine` — no Python re-entry.
    pub fn estimate_forward_pass_time_ms(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Option<f64>, AicError> {
        match &self.mode {
            ForwardPassPerfMode::Native {
                engine,
                corrections,
            } => {
                let Some(feature) = IterationFeatures::from_metrics(metrics_by_rank)? else {
                    return Ok(Some(0.0));
                };
                let native = engine.forward_pass_time_ms(metrics_by_rank)?;
                let corrected = native
                    * corrections
                        .store(feature.workload_kind)
                        .correction_factor_for(&feature.x);
                Ok(Some(corrected))
            }
            ForwardPassPerfMode::Regression {
                worker_type,
                regression,
            } => {
                let Some(feature) = RegressionIterationFeatures::from_metrics(
                    metrics_by_rank,
                    *worker_type,
                    &self.options,
                )?
                else {
                    return Ok(Some(0.0));
                };
                Ok(regression.predict(&feature.x))
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
    /// For each non-empty iteration, this method extracts features with the
    /// active backend and uses the maximum finite positive `wall_time` across
    /// ranks as the observed latency target in milliseconds. Iterations with
    /// no scheduled work or no positive `wall_time` are ignored. Native models
    /// infer the workload kind and update the matching region's
    /// median `observed_ms / native_ms` correction factor, with each ratio
    /// bounded by `min_faster_correction_factor` and
    /// `max_slower_correction_factor` when configured. Regions are used only
    /// after their inferred workload kind has `min_observations` total samples;
    /// empty regions keep the default factor `1.0`. Observations outside the
    /// configured correction-grid workload ranges are ignored by native
    /// correction models. Regression models validate compatibility with their
    /// fixed worker type and update one two-dimensional constrained linear fit.
    ///
    /// Pure Rust over the `Engine` — no Python re-entry.
    pub fn tune_with_fpms(
        &mut self,
        iterations: &[Vec<ForwardPassMetrics>],
    ) -> Result<(), AicError> {
        let Self { mode, options, .. } = self;
        for metrics_by_rank in iterations {
            match mode {
                ForwardPassPerfMode::Native {
                    engine,
                    corrections,
                } => {
                    let Some(observation) = IterationObservation::from_metrics(metrics_by_rank)?
                    else {
                        continue;
                    };
                    let native = engine.forward_pass_time_ms(metrics_by_rank)?;
                    corrections
                        .store_mut(observation.feature.workload_kind)
                        .add_observation(observation.feature.x, observation.wall_time_ms, native);
                }
                ForwardPassPerfMode::Regression {
                    worker_type,
                    regression,
                } => {
                    let Some(observation) = RegressionIterationObservation::from_metrics(
                        metrics_by_rank,
                        *worker_type,
                        options,
                    )?
                    else {
                        continue;
                    };
                    regression.add_observation(observation.feature.x, observation.wall_time_ms);
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
            ForwardPassPerfMode::Regression { regression, .. } => {
                let ready = regression.is_ready();
                ForwardPassPerfDiagnostics {
                    source: ForwardPassPerfSource::FallbackRegression,
                    readiness: if ready {
                        ForwardPassPerfReadiness::Ready
                    } else if self.last_warning.is_some() {
                        ForwardPassPerfReadiness::UnsupportedConfig
                    } else {
                        ForwardPassPerfReadiness::InsufficientData
                    },
                    retained_observations: regression.observation_count(),
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
/// matching perf database via [`Engine::from_spec_bytes`]. This is the internal
/// `EngineConfig` counterpart to [`crate::AicEngineBuilder`] and maps its
/// modular fields onto the flat `compile_engine` kwargs.
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
        let Some(wall_time_ms) = max_positive_wall_time_ms(metrics_by_rank) else {
            return Ok(None);
        };
        Ok(Some(Self {
            feature,
            wall_time_ms,
        }))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RegressionIterationFeatures {
    pub(crate) x: [f64; 2],
}

impl RegressionIterationFeatures {
    pub(crate) fn from_metrics(
        metrics_by_rank: &[ForwardPassMetrics],
        worker_type: ForwardPassWorkerType,
        options: &ForwardPassPerfOptions,
    ) -> Result<Option<Self>, AicError> {
        if metrics_by_rank.is_empty() {
            return Err(AicError::InvalidForwardPassMetrics(
                "at least one attention-DP rank metric is required".to_string(),
            ));
        }
        for metrics in metrics_by_rank {
            validate_forward_pass_metrics(metrics)?;
            let scheduled = &metrics.scheduled_requests;
            match worker_type {
                ForwardPassWorkerType::Prefill if scheduled.num_decode_requests > 0 => {
                    return Err(AicError::InvalidForwardPassMetrics(
                        "prefill regression worker received scheduled decode work".to_string(),
                    ));
                }
                ForwardPassWorkerType::Decode if scheduled.sum_prefill_tokens > 0 => {
                    return Err(AicError::InvalidForwardPassMetrics(
                        "decode regression worker received scheduled prefill work".to_string(),
                    ));
                }
                _ => {}
            }
        }

        let has_scheduled_work = metrics_by_rank.iter().any(|metrics| {
            let scheduled = &metrics.scheduled_requests;
            scheduled.sum_prefill_tokens > 0 || scheduled.num_decode_requests > 0
        });
        if !has_scheduled_work {
            return Ok(None);
        }

        let alpha = options.regression_attention_kv_weight;
        let beta = options.regression_prefill_attention_pair_weight;
        let gamma = options.regression_ffn_token_weight;
        let mut critical_attention = 0.0_f64;
        let mut sum_prefill_tokens = 0.0_f64;
        let mut sum_decode_requests = 0.0_f64;

        for metrics in metrics_by_rank {
            let scheduled = &metrics.scheduled_requests;
            let num_prefill_requests = f64::from(scheduled.num_prefill_requests);
            let prefill_tokens = f64::from(scheduled.sum_prefill_tokens);
            let prefill_kv_tokens = f64::from(scheduled.sum_prefill_kv_tokens);
            let decode_requests = f64::from(scheduled.num_decode_requests);
            let decode_kv_tokens = f64::from(scheduled.sum_decode_kv_tokens);

            let (active_prefill_kv_tokens, prefill_attention_pairs) = if prefill_tokens > 0.0 {
                debug_assert!(num_prefill_requests > 0.0);
                (
                    prefill_kv_tokens,
                    prefill_kv_tokens * prefill_tokens / num_prefill_requests
                        + prefill_tokens * prefill_tokens / (2.0 * num_prefill_requests)
                        + prefill_tokens / 2.0,
                )
            } else {
                (0.0, 0.0)
            };

            let rank_attention = match worker_type {
                ForwardPassWorkerType::Prefill => {
                    alpha * active_prefill_kv_tokens + beta * prefill_attention_pairs
                }
                ForwardPassWorkerType::Decode => alpha * decode_kv_tokens,
                ForwardPassWorkerType::Aggregated => {
                    alpha * (active_prefill_kv_tokens + decode_kv_tokens)
                        + beta * prefill_attention_pairs
                }
            };
            critical_attention = critical_attention.max(rank_attention);
            sum_prefill_tokens += prefill_tokens;
            sum_decode_requests += decode_requests;
        }

        let global_ffn = gamma
            * match worker_type {
                ForwardPassWorkerType::Prefill => sum_prefill_tokens,
                ForwardPassWorkerType::Decode => sum_decode_requests,
                ForwardPassWorkerType::Aggregated => sum_prefill_tokens + sum_decode_requests,
            };
        let x = [critical_attention, global_ffn];
        if !x.iter().all(|value| value.is_finite() && *value >= 0.0) {
            return Err(AicError::InvalidForwardPassMetrics(
                "derived regression features must be finite and nonnegative".to_string(),
            ));
        }
        Ok(Some(Self { x }))
    }
}

#[derive(Clone, Debug)]
struct RegressionIterationObservation {
    feature: RegressionIterationFeatures,
    wall_time_ms: f64,
}

impl RegressionIterationObservation {
    fn from_metrics(
        metrics_by_rank: &[ForwardPassMetrics],
        worker_type: ForwardPassWorkerType,
        options: &ForwardPassPerfOptions,
    ) -> Result<Option<Self>, AicError> {
        let Some(feature) =
            RegressionIterationFeatures::from_metrics(metrics_by_rank, worker_type, options)?
        else {
            return Ok(None);
        };
        let Some(wall_time_ms) = max_positive_wall_time_ms(metrics_by_rank) else {
            return Ok(None);
        };
        Ok(Some(Self {
            feature,
            wall_time_ms,
        }))
    }
}

fn max_positive_wall_time_ms(metrics_by_rank: &[ForwardPassMetrics]) -> Option<f64> {
    let wall_time_seconds = metrics_by_rank
        .iter()
        .map(|metrics| metrics.wall_time)
        .filter(|wall_time| wall_time.is_finite() && *wall_time > 0.0)
        .fold(0.0_f64, f64::max);
    // Select a usable source value before converting units. The conversion may
    // overflow; correction and regression ingestion reject that target after
    // Native has had the opportunity to surface any engine estimation error.
    (wall_time_seconds > 0.0).then_some(wall_time_seconds * 1000.0)
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
            prefill: T::with_options(options, &[AxisRange::from_zero_to(options.max_num_tokens)]),
            decode: T::with_options(
                options,
                &[
                    AxisRange::from_zero_to(options.max_batch_size),
                    AxisRange::from_zero_to(options.max_kv_tokens),
                ],
            ),
            mixed: T::with_options(
                options,
                &[
                    AxisRange::from_zero_to(options.max_num_tokens),
                    AxisRange::from_zero_to(options.max_kv_tokens),
                ],
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
            | AicError::Yaml { .. }
            | AicError::Parquet { .. }
    )
}
