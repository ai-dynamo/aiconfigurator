// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 wrapper exposing the latency estimator to Python.
//!
//! Mirrors the JSON-in / f64-out contract of the raw C ABI in [`crate::ffi`] —
//! the Python wrapper in `aiconfigurator.sdk.rust_engine_step` already
//! serializes its `EngineConfig` and per-rank `ForwardPassMetrics` to JSON, so
//! the wire format is unchanged when moving from `ctypes.CDLL` to `import`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::{create_engine_step_estimator, EngineConfig, EngineStepEstimator, ForwardPassMetrics};

#[pyclass]
struct PyEngineStepEstimator {
    inner: EngineStepEstimator,
}

#[pymethods]
impl PyEngineStepEstimator {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let config: EngineConfig = serde_json::from_str(config_json)
            .map_err(|err| PyValueError::new_err(format!("failed to parse EngineConfig JSON: {err}")))?;
        let inner = create_engine_step_estimator(config)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    fn forward_pass_time_ms(&self, metrics_json: &str) -> PyResult<f64> {
        let metrics: Vec<ForwardPassMetrics> = serde_json::from_str(metrics_json).map_err(|err| {
            PyValueError::new_err(format!("failed to parse ForwardPassMetrics list JSON: {err}"))
        })?;
        self.inner
            .forward_pass_time_ms(&metrics)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pymodule]
fn aiconfigurator_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngineStepEstimator>()?;
    Ok(())
}
