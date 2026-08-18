// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SPIKE (PR-7 design validation, not final code): export Rust op structs as
//! Python classes. One family (GEMM) end-to-end to validate the architecture:
//!
//! * `Operation` base pyclass holds the typed [`Op`] enum value; family
//!   classes are stateless `extends=` subclasses whose `#[new]` builds the
//!   right variant.
//! * Python-side shells subclass the family classes and keep only the
//!   class-level data-binding surface (`load_data` etc.).
//! * Pickle goes through `__getnewargs_ex__` (default object reduce), so the
//!   shell subclass identity survives `ProcessPoolExecutor` without a custom
//!   `__reduce__`.
//! * Post-construction mutation sites (`op._name = ...`, `op._seq_split = ...`)
//!   are served by data descriptors so writes reach the Rust struct instead of
//!   landing in the shell instance `__dict__`.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::common::enums::GemmQuantMode;
use crate::operators::{GemmOp, Op};

/// Parse a quant-mode argument the way the Python classes accepted it: the
/// `common.GEMMQuantMode` enum member (has `.name`) or its snake_case name.
fn extract_gemm_quant(obj: &Bound<'_, PyAny>) -> PyResult<GemmQuantMode> {
    let name: String = if let Ok(s) = obj.extract::<String>() {
        s
    } else {
        obj.getattr("name")
            .map_err(|_| {
                PyTypeError::new_err(
                    "quant_mode must be a common.GEMMQuantMode member or its snake_case name",
                )
            })?
            .extract()?
    };
    serde_json::from_value::<GemmQuantMode>(serde_json::Value::String(name.clone()))
        .map_err(|_| PyValueError::new_err(format!("unknown GEMM quant mode: {name:?}")))
}

fn gemm_quant_name(q: GemmQuantMode) -> String {
    match serde_json::to_value(q) {
        Ok(serde_json::Value::String(s)) => s,
        _ => unreachable!("GemmQuantMode serializes as a string"),
    }
}

/// Base class of every engine-backed op: owns the typed [`Op`] value.
#[pyclass(subclass, name = "Operation", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyOperation {
    pub(crate) inner: Op,
}

impl PyOperation {
    fn gemm(&self) -> PyResult<&GemmOp> {
        match &self.inner {
            Op::Gemm(g) => Ok(g),
            _ => Err(PyTypeError::new_err("not a GEMM op")),
        }
    }

    fn gemm_mut(&mut self) -> PyResult<&mut GemmOp> {
        match &mut self.inner {
            Op::Gemm(g) => Ok(g),
            _ => Err(PyTypeError::new_err("not a GEMM op")),
        }
    }
}

#[pymethods]
impl PyOperation {
    /// Constant weight bytes for this op (`Op::weight_bytes`, scale treatment
    /// included). The Rust value is cheap; no instance cache needed.
    fn get_weights(&self) -> f64 {
        self.inner.weight_bytes()
    }

    #[getter(_name)]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[setter(_name)]
    fn set_name(&mut self, value: String) -> PyResult<()> {
        match &mut self.inner {
            Op::Gemm(g) => g.name = value,
            _ => {
                return Err(PyTypeError::new_err(
                    "spike: _name setter only wired for GEMM",
                ))
            }
        }
        Ok(())
    }

    /// The op's engine wire form (opspec JSON). Diagnostic / FFI helper.
    fn _spec_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("<{} op {:?}>", self.inner.name(), std::mem::discriminant(&self.inner))
    }
}

/// GEMM: dense matmul `M=x, N=n, K=k`.
#[pyclass(extends = PyOperation, subclass, name = "GEMM", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyGemm;

#[pymethods]
impl PyGemm {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[new]
    #[pyo3(signature = (name, scale_factor, n, k, quant_mode, *, seq_split=1, scale_num_tokens=1, low_precision_input=false, below_grid_sol=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        n: u32,
        k: u32,
        quant_mode: &Bound<'_, PyAny>,
        seq_split: u32,
        scale_num_tokens: u32,
        low_precision_input: bool,
        below_grid_sol: bool,
    ) -> PyResult<(Self, PyOperation)> {
        let quant_mode = extract_gemm_quant(quant_mode)?;
        let inner = Op::Gemm(GemmOp {
            name,
            scale_factor,
            n,
            k,
            quant_mode,
            scale_num_tokens,
            low_precision_input,
            seq_split,
            below_grid_sol,
        });
        Ok((PyGemm, PyOperation { inner }))
    }

    /// Default-protocol pickle support: rebuild through the constructor so a
    /// Python shell subclass keeps its identity across processes.
    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let g = slf.as_super().gemm()?;
        let args = (
            g.name.clone(),
            g.scale_factor,
            g.n,
            g.k,
            gemm_quant_name(g.quant_mode),
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seq_split", g.seq_split)?;
        kwargs.set_item("scale_num_tokens", g.scale_num_tokens)?;
        kwargs.set_item("low_precision_input", g.low_precision_input)?;
        kwargs.set_item("below_grid_sol", g.below_grid_sol)?;
        Ok((args, kwargs))
    }

    #[getter(_n)]
    fn n(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gemm()?.n)
    }

    #[getter(_k)]
    fn k(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gemm()?.k)
    }

    #[getter(_quant_mode)]
    fn quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let name = gemm_quant_name(slf.as_super().gemm()?.quant_mode);
        // Return the canonical Python enum member for drop-in compatibility.
        py.import("aiconfigurator_core.sdk.common")?
            .getattr("GEMMQuantMode")?
            .get_item(name)
    }

    #[getter(_seq_split)]
    fn seq_split(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gemm()?.seq_split)
    }

    #[setter(_seq_split)]
    fn set_seq_split(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().gemm_mut()?.seq_split = value;
        Ok(())
    }
}
