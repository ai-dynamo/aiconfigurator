// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::{
    create_engine_step_estimator, EngineConfig, EngineStepEstimator, ForwardPassMetrics,
    ForwardPassPerfModel, ForwardPassPerfOptions,
};

pub struct AicEngineStepEstimatorHandle {
    estimator: EngineStepEstimator,
}

pub struct AicForwardPassPerfModelHandle {
    model: ForwardPassPerfModel,
}

/// Create an engine-step estimator from JSON-serialized `EngineConfig`.
///
/// Returns null on success. On failure, returns a heap-allocated error string
/// that must be released with `aic_engine_step_string_free`.
#[no_mangle]
pub extern "C" fn aic_engine_step_estimator_new(
    config_json: *const c_char,
    out_estimator: *mut *mut AicEngineStepEstimatorHandle,
) -> *mut c_char {
    ffi_result(|| {
        if out_estimator.is_null() {
            return Err("out_estimator must not be null".to_string());
        }

        let config: EngineConfig = parse_json(config_json, "EngineConfig")?;
        let estimator = create_engine_step_estimator(config).map_err(|err| err.to_string())?;
        let handle = Box::new(AicEngineStepEstimatorHandle { estimator });
        unsafe {
            *out_estimator = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// Estimate one forward-pass iteration from JSON-serialized per-rank
/// `ForwardPassMetrics`.
///
/// Returns null on success. On failure, returns a heap-allocated error string
/// that must be released with `aic_engine_step_string_free`.
#[no_mangle]
pub extern "C" fn aic_engine_step_forward_pass_time_ms(
    estimator: *mut AicEngineStepEstimatorHandle,
    metrics_json: *const c_char,
    out_ms: *mut f64,
) -> *mut c_char {
    ffi_result(|| {
        if estimator.is_null() {
            return Err("estimator handle must not be null".to_string());
        }
        if out_ms.is_null() {
            return Err("out_ms must not be null".to_string());
        }

        let metrics: ForwardPassMetricsInput = parse_json(metrics_json, "ForwardPassMetrics list")?;
        let metrics = metrics.into_vec();
        let latency_ms = unsafe { &*estimator }
            .estimator
            .forward_pass_time_ms(&metrics)
            .map_err(|err| err.to_string())?;
        unsafe {
            *out_ms = latency_ms;
        }
        Ok(())
    })
}

/// Reset runtime caches on an estimator handle. Used by parity and
/// benchmark harnesses that need a cold cache between iterations.
#[no_mangle]
pub extern "C" fn aic_engine_step_estimator_clear_runtime_caches(
    estimator: *mut AicEngineStepEstimatorHandle,
) -> *mut c_char {
    ffi_result(|| {
        if estimator.is_null() {
            return Err("estimator handle must not be null".to_string());
        }
        unsafe { &*estimator }.estimator.clear_runtime_caches();
        Ok(())
    })
}

/// API:
/// `aic_forward_pass_perf_model_from_native(config_json, options_json, out_model) -> error_string`
///
/// Description: create a strict native forward-pass perf model from
/// JSON-serialized `EngineConfig` and optional `ForwardPassPerfOptions`.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_from_native(
    config_json: *const c_char,
    options_json: *const c_char,
    out_model: *mut *mut AicForwardPassPerfModelHandle,
) -> *mut c_char {
    ffi_result(|| {
        if out_model.is_null() {
            return Err("out_model must not be null".to_string());
        }

        let config: EngineConfig = parse_json(config_json, "EngineConfig")?;
        let options: ForwardPassPerfOptions =
            parse_optional_json(options_json, "ForwardPassPerfOptions")?;
        let model =
            ForwardPassPerfModel::from_native(config, options).map_err(|err| err.to_string())?;
        let handle = Box::new(AicForwardPassPerfModelHandle { model });
        unsafe {
            *out_model = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// API:
/// `aic_forward_pass_perf_model_best_available(config_json, options_json, out_model) -> error_string`
///
/// Description: create a native forward-pass perf model when supported,
/// otherwise create a fallback regression model with diagnostics.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_best_available(
    config_json: *const c_char,
    options_json: *const c_char,
    out_model: *mut *mut AicForwardPassPerfModelHandle,
) -> *mut c_char {
    ffi_result(|| {
        if out_model.is_null() {
            return Err("out_model must not be null".to_string());
        }

        let config: EngineConfig = parse_json(config_json, "EngineConfig")?;
        let options: ForwardPassPerfOptions =
            parse_optional_json(options_json, "ForwardPassPerfOptions")?;
        let model =
            ForwardPassPerfModel::best_available(config, options).map_err(|err| err.to_string())?;
        let handle = Box::new(AicForwardPassPerfModelHandle { model });
        unsafe {
            *out_model = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// API:
/// `aic_forward_pass_perf_model_from_regression(options_json, out_model) -> error_string`
///
/// Description: create a regression-only forward-pass perf model from optional
/// JSON-serialized `ForwardPassPerfOptions`.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_from_regression(
    options_json: *const c_char,
    out_model: *mut *mut AicForwardPassPerfModelHandle,
) -> *mut c_char {
    ffi_result(|| {
        if out_model.is_null() {
            return Err("out_model must not be null".to_string());
        }

        let options: ForwardPassPerfOptions =
            parse_optional_json(options_json, "ForwardPassPerfOptions")?;
        let model =
            ForwardPassPerfModel::from_regression(options).map_err(|err| err.to_string())?;
        let handle = Box::new(AicForwardPassPerfModelHandle { model });
        unsafe {
            *out_model = Box::into_raw(handle);
        }
        Ok(())
    })
}

/// API:
/// `aic_forward_pass_perf_model_estimate_forward_pass_time_ms(model, metrics_json, out_ms, out_has_value) -> error_string`
///
/// Description: estimate one FPM iteration and encode Rust `Option<f64>` as
/// `out_ms` plus `out_has_value`. Estimation treats FPM as a workload
/// descriptor: scheduled request fields are used, while `wall_time` and queued
/// request fields are ignored.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_estimate_forward_pass_time_ms(
    model: *mut AicForwardPassPerfModelHandle,
    metrics_json: *const c_char,
    out_ms: *mut f64,
    out_has_value: *mut bool,
) -> *mut c_char {
    ffi_result(|| {
        if model.is_null() {
            return Err("model handle must not be null".to_string());
        }
        if out_ms.is_null() {
            return Err("out_ms must not be null".to_string());
        }
        if out_has_value.is_null() {
            return Err("out_has_value must not be null".to_string());
        }

        let metrics: ForwardPassMetricsInput = parse_json(metrics_json, "ForwardPassMetrics list")?;
        let metrics = metrics.into_vec();
        let latency_ms = unsafe { &*model }
            .model
            .estimate_forward_pass_time_ms(&metrics)
            .map_err(|err| err.to_string())?;
        unsafe {
            match latency_ms {
                Some(value) => {
                    *out_ms = value;
                    *out_has_value = true;
                }
                None => {
                    *out_ms = 0.0;
                    *out_has_value = false;
                }
            }
        }
        Ok(())
    })
}

/// API:
/// `aic_forward_pass_perf_model_tune_with_fpms(model, iterations_json) -> error_string`
///
/// Description: tune the model with JSON-serialized observed FPM iterations.
/// Tuning treats FPM as telemetry: scheduled request fields are the workload
/// features and positive `wall_time` is the observed latency target.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_tune_with_fpms(
    model: *mut AicForwardPassPerfModelHandle,
    iterations_json: *const c_char,
) -> *mut c_char {
    ffi_result(|| {
        if model.is_null() {
            return Err("model handle must not be null".to_string());
        }

        let iterations: ForwardPassIterationsInput =
            parse_json(iterations_json, "ForwardPassMetrics iterations")?;
        let iterations = iterations.into_iterations();
        unsafe { &mut *model }
            .model
            .tune_with_fpms(&iterations)
            .map_err(|err| err.to_string())?;
        Ok(())
    })
}

/// API:
/// `aic_forward_pass_perf_model_diagnostics_json(model, out_json) -> error_string`
///
/// Description: return JSON-serialized `ForwardPassPerfDiagnostics`; the caller
/// must free `out_json` with `aic_engine_step_string_free`.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_diagnostics_json(
    model: *mut AicForwardPassPerfModelHandle,
    out_json: *mut *mut c_char,
) -> *mut c_char {
    ffi_result(|| {
        if model.is_null() {
            return Err("model handle must not be null".to_string());
        }
        if out_json.is_null() {
            return Err("out_json must not be null".to_string());
        }

        let json = serde_json::to_string(&unsafe { &*model }.model.diagnostics())
            .map_err(|err| err.to_string())?;
        unsafe {
            *out_json = string_to_raw(json);
        }
        Ok(())
    })
}

/// API:
/// `aic_forward_pass_perf_model_min_correction_factor(model, out_value, out_has_value) -> error_string`
///
/// Description: return the smallest ready native correction factor, if any.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_min_correction_factor(
    model: *mut AicForwardPassPerfModelHandle,
    out_value: *mut f64,
    out_has_value: *mut bool,
) -> *mut c_char {
    correction_factor_result(model, out_value, out_has_value, |model| {
        model.min_correction_factor()
    })
}

/// API:
/// `aic_forward_pass_perf_model_max_correction_factor(model, out_value, out_has_value) -> error_string`
///
/// Description: return the largest ready native correction factor, if any.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_max_correction_factor(
    model: *mut AicForwardPassPerfModelHandle,
    out_value: *mut f64,
    out_has_value: *mut bool,
) -> *mut c_char {
    correction_factor_result(model, out_value, out_has_value, |model| {
        model.max_correction_factor()
    })
}

/// API:
/// `aic_forward_pass_perf_model_avg_correction_factor(model, out_value, out_has_value) -> error_string`
///
/// Description: return the average ready native correction factor, if any.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_avg_correction_factor(
    model: *mut AicForwardPassPerfModelHandle,
    out_value: *mut f64,
    out_has_value: *mut bool,
) -> *mut c_char {
    correction_factor_result(model, out_value, out_has_value, |model| {
        model.avg_correction_factor()
    })
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ForwardPassMetricsInput {
    PerRank(Vec<ForwardPassMetrics>),
    Single(ForwardPassMetrics),
}

impl ForwardPassMetricsInput {
    fn into_vec(self) -> Vec<ForwardPassMetrics> {
        match self {
            Self::PerRank(metrics) => metrics,
            Self::Single(metrics) => vec![metrics],
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ForwardPassIterationsInput {
    Iterations(Vec<Vec<ForwardPassMetrics>>),
    PerRank(Vec<ForwardPassMetrics>),
    Single(ForwardPassMetrics),
}

impl ForwardPassIterationsInput {
    fn into_iterations(self) -> Vec<Vec<ForwardPassMetrics>> {
        match self {
            Self::Iterations(iterations) => iterations,
            Self::PerRank(metrics) => vec![metrics],
            Self::Single(metrics) => vec![vec![metrics]],
        }
    }
}

#[no_mangle]
pub extern "C" fn aic_engine_step_estimator_free(estimator: *mut AicEngineStepEstimatorHandle) {
    if estimator.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(estimator));
    }
}

/// API: `aic_forward_pass_perf_model_free(model) -> void`
///
/// Description: release a forward-pass perf model handle created by this C ABI.
#[no_mangle]
pub extern "C" fn aic_forward_pass_perf_model_free(model: *mut AicForwardPassPerfModelHandle) {
    if model.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(model));
    }
}

#[no_mangle]
pub extern "C" fn aic_engine_step_string_free(message: *mut c_char) {
    if message.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(message));
    }
}

fn ffi_result(f: impl FnOnce() -> Result<(), String>) -> *mut c_char {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => ptr::null_mut(),
        Ok(Err(message)) => string_to_raw(message),
        Err(_) => string_to_raw("Rust engine-step estimator panicked".to_string()),
    }
}

fn parse_json<T: DeserializeOwned>(input: *const c_char, kind: &str) -> Result<T, String> {
    if input.is_null() {
        return Err(format!("{kind} JSON pointer must not be null"));
    }
    let text = unsafe { CStr::from_ptr(input) }
        .to_str()
        .map_err(|err| format!("{kind} JSON must be valid UTF-8: {err}"))?;
    serde_json::from_str(text).map_err(|err| format!("failed to parse {kind} JSON: {err}"))
}

fn parse_optional_json<T: DeserializeOwned + Default>(
    input: *const c_char,
    kind: &str,
) -> Result<T, String> {
    if input.is_null() {
        return Ok(T::default());
    }
    parse_json(input, kind)
}

fn correction_factor_result(
    model: *mut AicForwardPassPerfModelHandle,
    out_value: *mut f64,
    out_has_value: *mut bool,
    get_value: impl FnOnce(&ForwardPassPerfModel) -> Option<f64>,
) -> *mut c_char {
    ffi_result(|| {
        if model.is_null() {
            return Err("model handle must not be null".to_string());
        }
        if out_value.is_null() {
            return Err("out_value must not be null".to_string());
        }
        if out_has_value.is_null() {
            return Err("out_has_value must not be null".to_string());
        }
        unsafe {
            match get_value(&(&*model).model) {
                Some(value) => {
                    *out_value = value;
                    *out_has_value = true;
                }
                None => {
                    *out_value = 0.0;
                    *out_has_value = false;
                }
            }
        }
        Ok(())
    })
}

fn string_to_raw(message: String) -> *mut c_char {
    match CString::new(message) {
        Ok(value) => value.into_raw(),
        Err(_) => CString::new("error message contained an interior NUL byte")
            .expect("static string has no NUL")
            .into_raw(),
    }
}
