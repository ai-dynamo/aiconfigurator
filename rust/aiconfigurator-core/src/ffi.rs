// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::{create_engine_step_estimator, EngineConfig, EngineStepEstimator, ForwardPassMetrics};

pub struct AicEngineStepEstimatorHandle {
    estimator: EngineStepEstimator,
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

#[no_mangle]
pub extern "C" fn aic_engine_step_estimator_free(estimator: *mut AicEngineStepEstimatorHandle) {
    if estimator.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(estimator));
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

fn string_to_raw(message: String) -> *mut c_char {
    match CString::new(message) {
        Ok(value) => value.into_raw(),
        Err(_) => CString::new("error message contained an interior NUL byte")
            .expect("static string has no NUL")
            .into_raw(),
    }
}
