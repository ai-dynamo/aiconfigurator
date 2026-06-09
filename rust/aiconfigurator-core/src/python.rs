// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 wrapper exposing the latency estimator and forward-pass perf model to Python.
//!
//! Mirrors the JSON-in / value-out contract of the raw C ABI in [`crate::ffi`] —
//! the Python wrapper in `aiconfigurator.sdk.rust_engine_step` already serializes
//! its `EngineConfig`, `ForwardPassPerfOptions`, and per-rank `ForwardPassMetrics`
//! to JSON, so the wire format is unchanged whether Python reaches the core
//! through `ctypes.CDLL` or `import`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::Deserialize;

use crate::{
    create_engine_step_estimator, EngineConfig, EngineStepEstimator, ForwardPassMetrics,
    ForwardPassPerfModel, ForwardPassPerfOptions,
};

#[pyclass]
struct PyEngineStepEstimator {
    inner: EngineStepEstimator,
}

#[pymethods]
impl PyEngineStepEstimator {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let config = parse_config(config_json)?;
        let inner = create_engine_step_estimator(config)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    fn forward_pass_time_ms(&self, metrics_json: &str) -> PyResult<f64> {
        let metrics = parse_metrics(metrics_json)?;
        self.inner
            .forward_pass_time_ms(&metrics)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    // TODO(remove-after-rust-migration): parity check/benchmark-only cache reset.
    fn clear_runtime_caches(&self) {
        self.inner.clear_runtime_caches();
    }
}

#[pyclass]
struct PyForwardPassPerfModel {
    inner: ForwardPassPerfModel,
}

#[pymethods]
impl PyForwardPassPerfModel {
    #[staticmethod]
    #[pyo3(signature = (config_json, options_json=None))]
    fn from_native(config_json: &str, options_json: Option<&str>) -> PyResult<Self> {
        let config = parse_config(config_json)?;
        let options = parse_options(options_json)?;
        let inner = ForwardPassPerfModel::from_native(config, options)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (config_json, options_json=None))]
    fn best_available(config_json: &str, options_json: Option<&str>) -> PyResult<Self> {
        let config = parse_config(config_json)?;
        let options = parse_options(options_json)?;
        let inner = ForwardPassPerfModel::best_available(config, options)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (options_json=None))]
    fn from_regression(options_json: Option<&str>) -> PyResult<Self> {
        let options = parse_options(options_json)?;
        let inner = ForwardPassPerfModel::from_regression(options)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    fn estimate_forward_pass_time_ms(&self, metrics_json: &str) -> PyResult<Option<f64>> {
        let metrics = parse_metrics(metrics_json)?;
        self.inner
            .estimate_forward_pass_time_ms(&metrics)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn tune_with_fpms(&mut self, iterations_json: &str) -> PyResult<()> {
        let iterations = parse_iterations(iterations_json)?;
        self.inner
            .tune_with_fpms(&iterations)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn diagnostics_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.diagnostics()).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to serialize diagnostics: {err}"))
        })
    }

    fn min_correction_factor(&self) -> Option<f64> {
        self.inner.min_correction_factor()
    }

    fn max_correction_factor(&self) -> Option<f64> {
        self.inner.max_correction_factor()
    }

    fn avg_correction_factor(&self) -> Option<f64> {
        self.inner.avg_correction_factor()
    }

    fn options_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.options())
            .map_err(|err| PyRuntimeError::new_err(format!("failed to serialize options: {err}")))
    }
}

// Untagged inputs mirror `crate::ffi` so the JSON contract is identical across the
// C ABI and the PyO3 bindings.
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

fn parse_config(config_json: &str) -> PyResult<EngineConfig> {
    serde_json::from_str(config_json)
        .map_err(|err| PyValueError::new_err(format!("failed to parse EngineConfig JSON: {err}")))
}

fn parse_options(options_json: Option<&str>) -> PyResult<ForwardPassPerfOptions> {
    match options_json {
        Some(text) => serde_json::from_str(text).map_err(|err| {
            PyValueError::new_err(format!("failed to parse ForwardPassPerfOptions JSON: {err}"))
        }),
        None => Ok(ForwardPassPerfOptions::default()),
    }
}

fn parse_metrics(metrics_json: &str) -> PyResult<Vec<ForwardPassMetrics>> {
    let metrics: ForwardPassMetricsInput = serde_json::from_str(metrics_json).map_err(|err| {
        PyValueError::new_err(format!("failed to parse ForwardPassMetrics list JSON: {err}"))
    })?;
    Ok(metrics.into_vec())
}

fn parse_iterations(iterations_json: &str) -> PyResult<Vec<Vec<ForwardPassMetrics>>> {
    let iterations: ForwardPassIterationsInput =
        serde_json::from_str(iterations_json).map_err(|err| {
            PyValueError::new_err(format!(
                "failed to parse ForwardPassMetrics iterations JSON: {err}"
            ))
        })?;
    Ok(iterations.into_iterations())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngineStepEstimator>()?;
    m.add_class::<PyForwardPassPerfModel>()?;
    Ok(())
}
