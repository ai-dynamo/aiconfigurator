// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Rust op structs exported as Python classes (the deprecation-cleanup
//! PR's pyo3 op unification, #1357 ladder item 4).
//!
//! `Operation` is the base pyclass holding the typed [`Op`] enum value; each
//! family class is a stateless `extends = Operation` subclass whose `#[new]`
//! builds the right variant with the SAME calling shape the retired Python
//! `__init__` accepted (positional + keyword args, quant modes as the
//! `common.*QuantMode` enum members or their snake_case names). The Python
//! side subclasses these as thin shells that keep only the class-level
//! data-plane surface (`load_data` / `clear_cache` / `supported_quant_modes`
//! and the `_ENGINE_QUERY_SHAPE` kwarg-mapping declaration); every
//! construction-time field lives HERE, single-owner, ending the two-sided
//! `_to_opspec` schema-sync discipline.
//!
//! Conventions, applied uniformly:
//! * Getters/setters use the retired underscore attribute names (`_name`,
//!   `_seq_split`, ...): they are data descriptors on the type, so they win
//!   over a shell instance `__dict__` and post-construction mutation
//!   (`op._name = ...`, the hybrid models' CP rewiring) reaches the Rust
//!   struct. Attributes the wire never carried (e.g. the composites'
//!   `_seq_split`, `MoEDispatch._reduce_results`) deliberately have NO
//!   descriptor — writes land in the shell `__dict__` exactly as before,
//!   engine-invisible either way.
//! * Pickle rides `__getnewargs_ex__` (the default object reduce): rebuild
//!   goes through the subclass constructor, so a Python shell keeps its
//!   identity across `ProcessPoolExecutor` (fork and spawn).
//! * Constructor parameters that existed only for retired Python-side math
//!   (`empirical_bw_scaling_factor`, `enable_fp4_all2all`,
//!   `reduce_results`, ...) are accepted and dropped so the 500+ model
//!   construction sites keep their calling shape.
//! * The context-parallel audit gate survives: families that never opted in
//!   (`_CP_AWARE = False`) reject `seq_split > 1` at construction with the
//!   retired base-class message shape.

use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::common::enums::{
    CommQuantMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, MoeQuantMode,
};
use crate::operators::dsa::DsaProjectionQuants;
use crate::operators::{
    ContextAttentionOp, ContextMlaOp, CustomAllReduceOp, ElementwiseOp, EmbeddingOp,
    EncoderAttentionOp, GemmOp, GenerationAttentionOp, GenerationMlaOp, MhcModuleOp, MlaBmmOp,
    MlaModuleOp, NcclOp, Op, P2POp,
};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Serde token (snake_case wire name) of a quant enum value.
fn enum_token<T: serde::Serialize>(value: &T) -> String {
    match serde_json::to_value(value) {
        Ok(serde_json::Value::String(s)) => s,
        _ => unreachable!("quant enums serialize as strings"),
    }
}

/// The canonical Python enum member for a wire token, for drop-in getter
/// compatibility (`op._quant_mode is common.GEMMQuantMode.fp8_static`).
fn py_enum_member<'py>(
    py: Python<'py>,
    enum_name: &str,
    token: &str,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("aiconfigurator_core.sdk.common")?
        .getattr(enum_name)?
        .get_item(token)
}

macro_rules! quant_extractor {
    ($fn_name:ident, $ty:ty, $label:literal) => {
        /// Accept the Python enum member (has `.name`) or its snake_case name.
        fn $fn_name(obj: &Bound<'_, PyAny>) -> PyResult<$ty> {
            let name: String = if let Ok(s) = obj.extract::<String>() {
                s
            } else {
                obj.getattr("name")
                    .map_err(|_| {
                        PyTypeError::new_err(concat!(
                            $label,
                            " must be a common enum member or its snake_case name"
                        ))
                    })?
                    .extract()?
            };
            serde_json::from_value::<$ty>(serde_json::Value::String(name.clone())).map_err(|_| {
                PyValueError::new_err(format!(concat!("unknown ", $label, ": {:?}"), name))
            })
        }
    };
}

quant_extractor!(gemm_quant, GemmQuantMode, "GEMM quant mode");
quant_extractor!(kv_quant, KvCacheQuantMode, "KV-cache quant mode");
quant_extractor!(fmha_quant, FmhaQuantMode, "FMHA quant mode");
quant_extractor!(moe_quant, MoeQuantMode, "MoE quant mode");
quant_extractor!(comm_quant, CommQuantMode, "comm quant mode");

/// The retired base-class context-parallel audit gate: constructing with
/// `seq_split > 1` on a family that has NOT opted in raises.
fn cp_audit_gate(class_name: &str, cp_aware: bool, seq_split: u32) -> PyResult<()> {
    if seq_split > 1 && !cp_aware {
        return Err(PyNotImplementedError::new_err(format!(
            "{class_name} has not been audited for context parallelism (seq_split={seq_split}). \
             Opt the family in after verifying its token-count treatment (or handle CP at the \
             model construction site)."
        )));
    }
    Ok(())
}

/// Wrap a plain [`Op`] value in its exact Rust family class (base + subclass
/// chain). Used by the composite child getters and their pickle args; a
/// rebuilt child is the Rust class, not the Python shell — the shells add no
/// instance state, so behavior is identical.
pub(crate) fn wrap_op(py: Python<'_>, op: Op) -> PyResult<Py<PyAny>> {
    macro_rules! wrap {
        ($sub:ident) => {{
            let init = PyClassInitializer::from(PyOperation { inner: op }).add_subclass($sub);
            Ok(Py::new(py, init)?.into_any())
        }};
    }
    match &op {
        Op::Gemm(_) => wrap!(PyGemm),
        Op::Embedding(_) => wrap!(PyEmbedding),
        Op::Elementwise(_) => wrap!(PyElementWise),
        Op::ContextAttention(_) => wrap!(PyContextAttention),
        Op::GenerationAttention(_) => wrap!(PyGenerationAttention),
        Op::EncoderAttention(_) => wrap!(PyEncoderAttention),
        Op::ContextMla(_) => wrap!(PyContextMLA),
        Op::GenerationMla(_) => wrap!(PyGenerationMLA),
        Op::MlaModuleContext(_) | Op::MlaModuleGeneration(_) => wrap!(PyMLAModule),
        Op::MlaBmm(_) => wrap!(PyMLABmm),
        Op::CustomAllReduce(_) => wrap!(PyCustomAllReduce),
        Op::Nccl(_) => wrap!(PyNCCL),
        Op::P2P(_) => wrap!(PyP2P),
        Op::Mhc(_) => wrap!(PyDeepSeekV4MHCModule),
        Op::Moe(_) => wrap!(PyMoE),
        Op::MoeDispatch(_) => wrap!(PyMoEDispatch),
        Op::MoeAllToAll(_) => wrap!(PyMoEAllToAll),
        Op::EpMoe(_) => wrap!(PyModeledEPMoE),
        Op::Dsv4MegaMoe(_) => wrap!(PyDeepSeekV4MegaMoEModule),
        Op::DsaContext(_) => wrap!(PyContextDSAModule),
        Op::DsaGeneration(_) => wrap!(PyGenerationDSAModule),
        Op::MsaContext(_) => wrap!(PyContextMSAModule),
        Op::MsaGeneration(_) => wrap!(PyGenerationMSAModule),
        Op::Dsv4Context(_) => wrap!(PyContextDeepSeekV4AttentionModule),
        Op::Dsv4Generation(_) => wrap!(PyGenerationDeepSeekV4AttentionModule),
        Op::Mamba2(_) => wrap!(PyMamba2Kernel),
        Op::Gdn(_) => wrap!(PyGDNKernel),
        Op::Kda(_) => wrap!(PyKDAKernel),
        Op::WideEpContextMla(_) => wrap!(PyWideEPContextMLA),
        Op::WideEpGenerationMla(_) => wrap!(PyWideEPGenerationMLA),
        Op::Overlap(_) => wrap!(PyOverlapOp),
        Op::Fallback(_) => wrap!(PyFallbackOp),
        // FpmForward has no family class: FPMForwardOp stays a Python class
        // (callable slot + pinned signature) whose spec adapter converts to a
        // BASE-wrapped engine op for list assembly.
        Op::FpmForward(_) => Ok(Py::new(py, PyOperation { inner: op })?.into_any()),
        // Vision is never wrapped: compile decomposes it into child ops.
        other => Err(PyTypeError::new_err(format!(
            "no Python class wrapper for engine op variant {:?}",
            std::mem::discriminant(other)
        ))),
    }
}

// ---------------------------------------------------------------------------
// Base class
// ---------------------------------------------------------------------------

/// Base class of every engine-backed op: owns the typed [`Op`] value.
#[pyclass(subclass, name = "Operation", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyOperation {
    pub(crate) inner: Op,
}

impl PyOperation {
    pub(crate) fn op(&self) -> &Op {
        &self.inner
    }
}

macro_rules! inner_accessor {
    ($name:ident, $name_mut:ident, $variant:ident, $ty:ty, $label:literal) => {
        impl PyOperation {
            fn $name(&self) -> PyResult<&$ty> {
                match &self.inner {
                    Op::$variant(o) => Ok(o),
                    _ => Err(PyTypeError::new_err(concat!("not a ", $label, " op"))),
                }
            }
            fn $name_mut(&mut self) -> PyResult<&mut $ty> {
                match &mut self.inner {
                    Op::$variant(o) => Ok(o),
                    _ => Err(PyTypeError::new_err(concat!("not a ", $label, " op"))),
                }
            }
        }
    };
}

inner_accessor!(gemm, gemm_mut, Gemm, GemmOp, "GEMM");
inner_accessor!(embedding, embedding_mut, Embedding, EmbeddingOp, "Embedding");
inner_accessor!(elementwise, elementwise_mut, Elementwise, ElementwiseOp, "ElementWise");
inner_accessor!(
    context_attention,
    context_attention_mut,
    ContextAttention,
    ContextAttentionOp,
    "ContextAttention"
);
inner_accessor!(
    generation_attention,
    generation_attention_mut,
    GenerationAttention,
    GenerationAttentionOp,
    "GenerationAttention"
);
inner_accessor!(
    encoder_attention,
    encoder_attention_mut,
    EncoderAttention,
    EncoderAttentionOp,
    "EncoderAttention"
);
inner_accessor!(context_mla, context_mla_mut, ContextMla, ContextMlaOp, "ContextMLA");
inner_accessor!(
    generation_mla,
    generation_mla_mut,
    GenerationMla,
    GenerationMlaOp,
    "GenerationMLA"
);
inner_accessor!(mla_bmm, mla_bmm_mut, MlaBmm, MlaBmmOp, "MLABmm");
inner_accessor!(
    custom_all_reduce,
    custom_all_reduce_mut,
    CustomAllReduce,
    CustomAllReduceOp,
    "CustomAllReduce"
);
inner_accessor!(nccl, nccl_mut, Nccl, NcclOp, "NCCL");
inner_accessor!(p2p, p2p_mut, P2P, P2POp, "P2P");
inner_accessor!(mhc, mhc_mut, Mhc, MhcModuleOp, "DeepSeekV4MHCModule");

impl PyOperation {
    /// The MLA module struct regardless of phase variant.
    fn mla_module(&self) -> PyResult<&MlaModuleOp> {
        match &self.inner {
            Op::MlaModuleContext(o) | Op::MlaModuleGeneration(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not an MLAModule op")),
        }
    }
    fn mla_module_mut(&mut self) -> PyResult<&mut MlaModuleOp> {
        match &mut self.inner {
            Op::MlaModuleContext(o) | Op::MlaModuleGeneration(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not an MLAModule op")),
        }
    }
}

#[pymethods]
impl PyOperation {
    /// Shim call-shape label (`None` on the base; each family overrides).
    /// Read by `OpShellKit._engine_query` and the composite phase walker
    /// (`overlap._infer_phase`), which sees Rust-wrapped children.
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: Option<&'static str> = None;

    /// Constant weight bytes for this op, computed by the engine
    /// (`Op::weight_bytes`, scale treatment included per family). The Rust
    /// value is cheap to recompute; the retired per-instance cache is gone.
    #[pyo3(signature = (**_kwargs))]
    fn get_weights(&self, _kwargs: Option<&Bound<'_, PyDict>>) -> f64 {
        self.inner.weight_bytes()
    }

    #[getter(_name)]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[setter(_name)]
    fn set_name(&mut self, value: String) {
        self.inner.set_name(value);
    }

    #[getter(_scale_factor)]
    fn scale_factor(&self) -> PyResult<f64> {
        let json = serde_json::to_value(&self.inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        json.as_object()
            .and_then(|m| m.values().next())
            .and_then(|fields| fields.get("scale_factor"))
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| PyTypeError::new_err("op family carries no scale_factor"))
    }

    /// Default `_seq_split` read: the variant's CP shard factor where one
    /// exists, else 1 (audit-gated). CP-aware family classes override this
    /// getset pair with their own getter + setter.
    #[getter(_seq_split)]
    fn base_seq_split(&self) -> u32 {
        self.inner.seq_split()
    }

    /// The op's engine wire form (externally-tagged opspec JSON) — the same
    /// serde document `EngineHandle.evaluate_ops_json` consumes.
    fn _spec_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let class_name = slf.get_type().qualname()?;
        Ok(format!("<{} {:?}>", class_name, slf.borrow().inner.name()))
    }
}

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

/// GEMM: dense matmul `M=x, N=n, K=k`.
#[pyclass(extends = PyOperation, subclass, name = "GEMM", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyGemm;

#[pymethods]
impl PyGemm {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

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
        let inner = Op::Gemm(GemmOp {
            name,
            scale_factor,
            n,
            k,
            quant_mode: gemm_quant(quant_mode)?,
            scale_num_tokens,
            low_precision_input,
            seq_split,
            below_grid_sol,
        });
        Ok((PyGemm, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().gemm()?;
        let args = (o.name.clone(), o.scale_factor, o.n, o.k, enum_token(&o.quant_mode))
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seq_split", o.seq_split)?;
        kwargs.set_item("scale_num_tokens", o.scale_num_tokens)?;
        kwargs.set_item("low_precision_input", o.low_precision_input)?;
        kwargs.set_item("below_grid_sol", o.below_grid_sol)?;
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
        py_enum_member(py, "GEMMQuantMode", &enum_token(&slf.as_super().gemm()?.quant_mode))
    }

    #[getter(_scale_num_tokens)]
    fn scale_num_tokens(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gemm()?.scale_num_tokens)
    }

    #[getter(_low_precision_input)]
    fn low_precision_input(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().gemm()?.low_precision_input)
    }

    #[getter(_below_grid_sol)]
    fn below_grid_sol(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().gemm()?.below_grid_sol)
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

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Embedding lookup. `empirical_bw_scaling_factor` is accepted for calling-
/// shape compatibility and dropped (its math retired with the Python query
/// stack); the engine types the quant as bfloat16 (memory-only op).
#[pyclass(extends = PyOperation, subclass, name = "Embedding", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyEmbedding;

#[pymethods]
impl PyEmbedding {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, row_size, column_size, empirical_bw_scaling_factor=0.3, *, seq_split=1))]
    fn new(
        name: String,
        scale_factor: f64,
        row_size: u32,
        column_size: u32,
        empirical_bw_scaling_factor: f64,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let _ = empirical_bw_scaling_factor;
        let inner = Op::Embedding(EmbeddingOp {
            name,
            scale_factor,
            vocab_size: row_size,
            hidden_size: column_size,
            quant_mode: GemmQuantMode::Bfloat16,
            seq_split,
        });
        Ok((PyEmbedding, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().embedding()?;
        let args = (o.name.clone(), o.scale_factor, o.vocab_size, o.hidden_size).into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seq_split", o.seq_split)?;
        Ok((args, kwargs))
    }

    #[getter(_row_size)]
    fn row_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().embedding()?.vocab_size)
    }

    #[getter(_column_size)]
    fn column_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().embedding()?.hidden_size)
    }

    #[getter(_seq_split)]
    fn seq_split(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().embedding()?.seq_split)
    }

    #[setter(_seq_split)]
    fn set_seq_split(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().embedding_mut()?.seq_split = value;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ElementWise
// ---------------------------------------------------------------------------

/// Element-wise memory op. The wire carries the derived
/// `bytes_per_token = 2 * (dim_in + dim_out)` (bf16 in + out), exactly the
/// retired `_to_opspec` derivation; `empirical_bw_scaling_factor` is
/// accepted and dropped.
#[pyclass(extends = PyOperation, subclass, name = "ElementWise", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyElementWise;

#[pymethods]
impl PyElementWise {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, dim_in, dim_out, empirical_bw_scaling_factor=0.8, *, seq_split=1, scale_num_tokens=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        dim_in: u64,
        dim_out: u64,
        empirical_bw_scaling_factor: f64,
        seq_split: u32,
        scale_num_tokens: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let _ = empirical_bw_scaling_factor;
        let inner = Op::Elementwise(ElementwiseOp {
            name,
            scale_factor,
            bytes_per_token: (dim_in * 2 + dim_out * 2) as f64,
            // Python: `op._scale_num_tokens if op._scale_num_tokens else 1`.
            scale_num_tokens: if scale_num_tokens == 0 { 1 } else { scale_num_tokens },
            seq_split,
        });
        Ok((PyElementWise, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().elementwise()?;
        // dim_in/dim_out are not stored (only their byte sum crosses the
        // wire); rebuild with the equivalent (sum, 0) split — bytes_per_token
        // and every engine-visible value round-trip exactly.
        let dim_sum = (o.bytes_per_token / 2.0) as u64;
        let args = (o.name.clone(), o.scale_factor, dim_sum, 0u64).into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seq_split", o.seq_split)?;
        kwargs.set_item("scale_num_tokens", o.scale_num_tokens)?;
        Ok((args, kwargs))
    }

    #[getter(_scale_num_tokens)]
    fn scale_num_tokens(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().elementwise()?.scale_num_tokens)
    }

    #[getter(_seq_split)]
    fn seq_split(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().elementwise()?.seq_split)
    }

    #[setter(_seq_split)]
    fn set_seq_split(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().elementwise_mut()?.seq_split = value;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CustomAllReduce / NCCL / P2P
// ---------------------------------------------------------------------------

/// TP custom all-reduce (`quant` pinned to half, the Python parity value).
#[pyclass(extends = PyOperation, subclass, name = "CustomAllReduce", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyCustomAllReduce;

#[pymethods]
impl PyCustomAllReduce {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, h, tp_size, *, seq_split=1))]
    fn new(
        name: String,
        scale_factor: f64,
        h: u32,
        tp_size: u32,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let inner = Op::CustomAllReduce(CustomAllReduceOp {
            name,
            scale_factor,
            hidden_size: h,
            tp_size,
            quant: CommQuantMode::Half,
            seq_split,
        });
        Ok((PyCustomAllReduce, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().custom_all_reduce()?;
        let args = (o.name.clone(), o.scale_factor, o.hidden_size, o.tp_size).into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seq_split", o.seq_split)?;
        Ok((args, kwargs))
    }

    #[getter(_h)]
    fn h(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().custom_all_reduce()?.hidden_size)
    }

    #[getter(_tp_size)]
    fn tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().custom_all_reduce()?.tp_size)
    }

    #[getter(_seq_split)]
    fn seq_split(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().custom_all_reduce()?.seq_split)
    }

    #[setter(_seq_split)]
    fn set_seq_split(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().custom_all_reduce_mut()?.seq_split = value;
        Ok(())
    }
}

/// NCCL collective (`nccl_op` = all_gather / all_reduce / ...;
/// `num_elements_per_token` may be fractional — KV bytes / comm bytes).
#[pyclass(extends = PyOperation, subclass, name = "NCCL", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyNCCL;

#[pymethods]
impl PyNCCL {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, nccl_op, num_elements_per_token, num_gpus, comm_quant_mode, *, seq_split=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        nccl_op: String,
        num_elements_per_token: f64,
        num_gpus: u32,
        comm_quant_mode: &Bound<'_, PyAny>,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let inner = Op::Nccl(NcclOp {
            name,
            scale_factor,
            hidden_size: num_elements_per_token,
            num_gpus,
            dtype: comm_quant(comm_quant_mode)?,
            operation: nccl_op,
            seq_split,
        });
        Ok((PyNCCL, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().nccl()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.operation.clone(),
            o.hidden_size,
            o.num_gpus,
            enum_token(&o.dtype),
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seq_split", o.seq_split)?;
        Ok((args, kwargs))
    }

    #[getter(_nccl_op)]
    fn nccl_op(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().nccl()?.operation.clone())
    }

    #[getter(_num_elements_per_token)]
    fn num_elements_per_token(slf: PyRef<'_, Self>) -> PyResult<f64> {
        Ok(slf.as_super().nccl()?.hidden_size)
    }

    #[getter(_num_gpus)]
    fn num_gpus(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().nccl()?.num_gpus)
    }

    #[getter(_comm_quant_mode)]
    fn comm_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "CommQuantMode", &enum_token(&slf.as_super().nccl()?.dtype))
    }

    #[getter(_seq_split)]
    fn seq_split(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().nccl()?.seq_split)
    }

    #[setter(_seq_split)]
    fn set_seq_split(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().nccl_mut()?.seq_split = value;
        Ok(())
    }
}

/// Pipeline-parallel P2P transfer.
#[pyclass(extends = PyOperation, subclass, name = "P2P", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyP2P;

#[pymethods]
impl PyP2P {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, h, pp_size, *, seq_split=1))]
    fn new(
        name: String,
        scale_factor: f64,
        h: u32,
        pp_size: u32,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let inner = Op::P2P(P2POp {
            name,
            scale_factor,
            pp_size,
            hidden_size: h,
            seq_split,
        });
        Ok((PyP2P, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().p2p()?;
        let args = (o.name.clone(), o.scale_factor, o.hidden_size, o.pp_size).into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seq_split", o.seq_split)?;
        Ok((args, kwargs))
    }

    #[getter(_h)]
    fn h(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().p2p()?.hidden_size)
    }

    #[getter(_pp_size)]
    fn pp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().p2p()?.pp_size)
    }

    #[getter(_seq_split)]
    fn seq_split(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().p2p()?.seq_split)
    }

    #[setter(_seq_split)]
    fn set_seq_split(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().p2p_mut()?.seq_split = value;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DeepSeek-V4 mHC module
// ---------------------------------------------------------------------------

/// DeepSeek-V4 mHC (multi-head compression) module. `architecture` is a new
/// REQUIRED keyword: the retired serializer injected it from
/// `model.architecture` at compile time; construction owns it now.
#[pyclass(extends = PyOperation, subclass, name = "DeepSeekV4MHCModule", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyDeepSeekV4MHCModule;

#[pymethods]
impl PyDeepSeekV4MHCModule {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, op, hidden_size, hc_mult, sinkhorn_iters, quant_mode, *, architecture, seq_split=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        op: String,
        hidden_size: u32,
        hc_mult: u32,
        sinkhorn_iters: u32,
        quant_mode: &Bound<'_, PyAny>,
        architecture: String,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        if !matches!(op.as_str(), "pre" | "post" | "both") {
            return Err(PyValueError::new_err(format!(
                "Unsupported DeepSeek-V4 mHC op: {op}"
            )));
        }
        let inner = Op::Mhc(MhcModuleOp {
            name,
            scale_factor,
            op,
            hc_mult,
            hidden_size,
            architecture,
            sinkhorn_iters,
            quant_mode: gemm_quant(quant_mode)?,
            seq_split,
        });
        Ok((PyDeepSeekV4MHCModule, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().mhc()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.op.clone(),
            o.hidden_size,
            o.hc_mult,
            o.sinkhorn_iters,
            enum_token(&o.quant_mode),
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("architecture", o.architecture.clone())?;
        kwargs.set_item("seq_split", o.seq_split)?;
        Ok((args, kwargs))
    }

    #[getter(_op)]
    fn op(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().mhc()?.op.clone())
    }

    #[getter(_hidden_size)]
    fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mhc()?.hidden_size)
    }

    #[getter(_hc_mult)]
    fn hc_mult(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mhc()?.hc_mult)
    }

    #[getter(_sinkhorn_iters)]
    fn sinkhorn_iters(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mhc()?.sinkhorn_iters)
    }

    #[getter(_quant_mode)]
    fn quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "GEMMQuantMode", &enum_token(&slf.as_super().mhc()?.quant_mode))
    }

    #[getter(_architecture)]
    fn architecture(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().mhc()?.architecture.clone())
    }

    #[getter(_seq_split)]
    fn seq_split(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mhc()?.seq_split)
    }

    #[setter(_seq_split)]
    fn set_seq_split(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().mhc_mut()?.seq_split = value;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Attention family
// ---------------------------------------------------------------------------

/// Prefill GQA/MHA attention (FMHA).
#[pyclass(extends = PyOperation, subclass, name = "ContextAttention", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyContextAttention;

#[pymethods]
impl PyContextAttention {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "context";

    #[new]
    #[pyo3(signature = (name, scale_factor, n, n_kv, kvcache_quant_mode, fmha_quant_mode, window_size=0, head_size=128, use_qk_norm=false, cp_size=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        n: u32,
        n_kv: u32,
        kvcache_quant_mode: &Bound<'_, PyAny>,
        fmha_quant_mode: &Bound<'_, PyAny>,
        window_size: u32,
        head_size: u32,
        use_qk_norm: bool,
        cp_size: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let inner = Op::ContextAttention(ContextAttentionOp {
            name,
            scale_factor,
            n,
            n_kv,
            head_size,
            window_size,
            kv_cache_dtype: kv_quant(kvcache_quant_mode)?,
            fmha_quant_mode: fmha_quant(fmha_quant_mode)?,
            use_qk_norm,
            cp_size,
        });
        Ok((PyContextAttention, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().context_attention()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.n,
            o.n_kv,
            enum_token(&o.kv_cache_dtype),
            enum_token(&o.fmha_quant_mode),
            o.window_size,
            o.head_size,
            o.use_qk_norm,
            o.cp_size,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_n)]
    fn n(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().context_attention()?.n)
    }

    #[getter(_n_kv)]
    fn n_kv(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().context_attention()?.n_kv)
    }

    #[getter(_head_size)]
    fn head_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().context_attention()?.head_size)
    }

    #[getter(_window_size)]
    fn window_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().context_attention()?.window_size)
    }

    #[getter(_kvcache_quant_mode)]
    fn kvcache_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "KVCacheQuantMode",
            &enum_token(&slf.as_super().context_attention()?.kv_cache_dtype),
        )
    }

    #[getter(_fmha_quant_mode)]
    fn fmha_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "FMHAQuantMode",
            &enum_token(&slf.as_super().context_attention()?.fmha_quant_mode),
        )
    }

    #[getter(_use_qk_norm)]
    fn use_qk_norm(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().context_attention()?.use_qk_norm)
    }

    #[getter(_cp_size)]
    fn cp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().context_attention()?.cp_size)
    }

    #[setter(_cp_size)]
    fn set_cp_size(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().context_attention_mut()?.cp_size = value;
        Ok(())
    }
}

/// Decode GQA/MHA attention.
#[pyclass(extends = PyOperation, subclass, name = "GenerationAttention", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyGenerationAttention;

#[pymethods]
impl PyGenerationAttention {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "generation";

    #[new]
    #[pyo3(signature = (name, scale_factor, n, n_kv, kv_cache_dtype, window_size=0, head_size=128, use_qk_norm=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        n: u32,
        n_kv: u32,
        kv_cache_dtype: &Bound<'_, PyAny>,
        window_size: u32,
        head_size: u32,
        use_qk_norm: bool,
    ) -> PyResult<(Self, PyOperation)> {
        // use_qk_norm is accepted for calling-shape compatibility; the decode
        // table never keyed on it (the retired serializer dropped it too).
        let _ = use_qk_norm;
        let inner = Op::GenerationAttention(GenerationAttentionOp {
            name,
            scale_factor,
            n,
            n_kv,
            head_size,
            window_size,
            kv_cache_dtype: kv_quant(kv_cache_dtype)?,
        });
        Ok((PyGenerationAttention, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().generation_attention()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.n,
            o.n_kv,
            enum_token(&o.kv_cache_dtype),
            o.window_size,
            o.head_size,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_n)]
    fn n(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().generation_attention()?.n)
    }

    #[getter(_n_kv)]
    fn n_kv(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().generation_attention()?.n_kv)
    }

    #[getter(_head_size)]
    fn head_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().generation_attention()?.head_size)
    }

    #[getter(_window_size)]
    fn window_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().generation_attention()?.window_size)
    }

    #[getter(_kv_cache_dtype)]
    fn kv_cache_dtype<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "KVCacheQuantMode",
            &enum_token(&slf.as_super().generation_attention()?.kv_cache_dtype),
        )
    }
}

/// Vision-encoder bidirectional attention.
#[pyclass(extends = PyOperation, subclass, name = "EncoderAttention", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyEncoderAttention;

#[pymethods]
impl PyEncoderAttention {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "context";

    #[new]
    #[pyo3(signature = (name, scale_factor, num_heads, head_size, fmha_quant_mode=None, partial_rotary_factor=0.0))]
    fn new(
        name: String,
        scale_factor: f64,
        num_heads: u32,
        head_size: u32,
        fmha_quant_mode: Option<&Bound<'_, PyAny>>,
        partial_rotary_factor: f64,
    ) -> PyResult<(Self, PyOperation)> {
        let fmha = match fmha_quant_mode {
            Some(obj) => fmha_quant(obj)?,
            None => FmhaQuantMode::Bfloat16,
        };
        if fmha != FmhaQuantMode::Bfloat16 {
            return Err(PyValueError::new_err(format!(
                "EncoderAttention only supports FMHAQuantMode.bfloat16, got {}",
                enum_token(&fmha)
            )));
        }
        if !(0.0..=1.0).contains(&partial_rotary_factor) {
            return Err(PyValueError::new_err(format!(
                "partial_rotary_factor must be in [0.0, 1.0], got {partial_rotary_factor}"
            )));
        }
        let inner = Op::EncoderAttention(EncoderAttentionOp {
            name,
            scale_factor,
            n: num_heads,
            head_size,
            fmha_quant_mode: fmha,
            partial_rotary_factor,
        });
        Ok((PyEncoderAttention, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().encoder_attention()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.n,
            o.head_size,
            enum_token(&o.fmha_quant_mode),
            o.partial_rotary_factor,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_n)]
    fn n(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().encoder_attention()?.n)
    }

    #[getter(_head_size)]
    fn head_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().encoder_attention()?.head_size)
    }

    #[getter(_fmha_quant_mode)]
    fn fmha_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "FMHAQuantMode",
            &enum_token(&slf.as_super().encoder_attention()?.fmha_quant_mode),
        )
    }

    #[getter(_partial_rotary_factor)]
    fn partial_rotary_factor(slf: PyRef<'_, Self>) -> PyResult<f64> {
        Ok(slf.as_super().encoder_attention()?.partial_rotary_factor)
    }
}

// ---------------------------------------------------------------------------
// MLA family
// ---------------------------------------------------------------------------

/// Prefill MLA (DeepSeek-style latent attention).
#[pyclass(extends = PyOperation, subclass, name = "ContextMLA", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyContextMLA;

#[pymethods]
impl PyContextMLA {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "context";

    #[new]
    #[pyo3(signature = (name, scale_factor, num_heads, kvcache_quant_mode, fmha_quant_mode, cp_size=1))]
    fn new(
        name: String,
        scale_factor: f64,
        num_heads: u32,
        kvcache_quant_mode: &Bound<'_, PyAny>,
        fmha_quant_mode: &Bound<'_, PyAny>,
        cp_size: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let inner = Op::ContextMla(ContextMlaOp {
            name,
            scale_factor,
            num_heads,
            kv_cache_dtype: kv_quant(kvcache_quant_mode)?,
            fmha_quant_mode: fmha_quant(fmha_quant_mode)?,
            cp_size,
        });
        Ok((PyContextMLA, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().context_mla()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.num_heads,
            enum_token(&o.kv_cache_dtype),
            enum_token(&o.fmha_quant_mode),
            o.cp_size,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_num_heads)]
    fn num_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().context_mla()?.num_heads)
    }

    #[getter(_kvcache_quant_mode)]
    fn kvcache_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "KVCacheQuantMode",
            &enum_token(&slf.as_super().context_mla()?.kv_cache_dtype),
        )
    }

    #[getter(_fmha_quant_mode)]
    fn fmha_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "FMHAQuantMode",
            &enum_token(&slf.as_super().context_mla()?.fmha_quant_mode),
        )
    }

    #[getter(_cp_size)]
    fn cp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().context_mla()?.cp_size)
    }

    #[setter(_cp_size)]
    fn set_cp_size(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().context_mla_mut()?.cp_size = value;
        Ok(())
    }
}

/// Decode MLA.
#[pyclass(extends = PyOperation, subclass, name = "GenerationMLA", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyGenerationMLA;

#[pymethods]
impl PyGenerationMLA {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "generation";

    #[new]
    #[pyo3(signature = (name, scale_factor, num_heads, kv_cache_dtype))]
    fn new(
        name: String,
        scale_factor: f64,
        num_heads: u32,
        kv_cache_dtype: &Bound<'_, PyAny>,
    ) -> PyResult<(Self, PyOperation)> {
        let inner = Op::GenerationMla(GenerationMlaOp {
            name,
            scale_factor,
            num_heads,
            kv_cache_dtype: kv_quant(kv_cache_dtype)?,
        });
        Ok((PyGenerationMLA, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().generation_mla()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.num_heads,
            enum_token(&o.kv_cache_dtype),
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_num_heads)]
    fn num_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().generation_mla()?.num_heads)
    }

    #[getter(_kv_cache_dtype)]
    fn kv_cache_dtype<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "KVCacheQuantMode",
            &enum_token(&slf.as_super().generation_mla()?.kv_cache_dtype),
        )
    }
}

/// Fused MLA module (one class, two engine variants by phase). Setting
/// `_is_context` swaps the variant — the retired Python class stored the
/// phase as an instance flag and the serializer picked the wire tag.
#[pyclass(extends = PyOperation, subclass, name = "MLAModule", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyMLAModule;

#[pymethods]
impl PyMLAModule {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "module";

    #[new]
    #[pyo3(signature = (name, scale_factor, is_context, num_heads, kvcache_quant_mode, fmha_quant_mode, gemm_quant_mode, native_num_heads=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        is_context: bool,
        num_heads: u32,
        kvcache_quant_mode: &Bound<'_, PyAny>,
        fmha_quant_mode: &Bound<'_, PyAny>,
        gemm_quant_mode: &Bound<'_, PyAny>,
        native_num_heads: Option<u32>,
    ) -> PyResult<(Self, PyOperation)> {
        let module = MlaModuleOp {
            name,
            scale_factor,
            num_heads,
            kv_cache_dtype: kv_quant(kvcache_quant_mode)?,
            fmha_quant_mode: fmha_quant(fmha_quant_mode)?,
            gemm_quant_mode: gemm_quant(gemm_quant_mode)?,
            native_num_heads,
        };
        let inner = if is_context {
            Op::MlaModuleContext(module)
        } else {
            Op::MlaModuleGeneration(module)
        };
        Ok((PyMLAModule, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let is_context = matches!(slf.as_super().op(), Op::MlaModuleContext(_));
        let o = slf.as_super().mla_module()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            is_context,
            o.num_heads,
            enum_token(&o.kv_cache_dtype),
            enum_token(&o.fmha_quant_mode),
            enum_token(&o.gemm_quant_mode),
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("native_num_heads", o.native_num_heads)?;
        Ok((args, kwargs))
    }

    #[getter(_is_context)]
    fn is_context(slf: PyRef<'_, Self>) -> bool {
        matches!(slf.as_super().op(), Op::MlaModuleContext(_))
    }

    #[setter(_is_context)]
    fn set_is_context(mut slf: PyRefMut<'_, Self>, value: bool) -> PyResult<()> {
        let base = slf.as_super();
        let module = base.mla_module()?.clone();
        base.inner = if value {
            Op::MlaModuleContext(module)
        } else {
            Op::MlaModuleGeneration(module)
        };
        Ok(())
    }

    #[getter(_num_heads)]
    fn num_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mla_module()?.num_heads)
    }

    #[getter(_native_num_heads)]
    fn native_num_heads(slf: PyRef<'_, Self>) -> PyResult<Option<u32>> {
        Ok(slf.as_super().mla_module()?.native_num_heads)
    }

    #[getter(_kvcache_quant_mode)]
    fn kvcache_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "KVCacheQuantMode",
            &enum_token(&slf.as_super().mla_module()?.kv_cache_dtype),
        )
    }

    #[getter(_fmha_quant_mode)]
    fn fmha_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "FMHAQuantMode",
            &enum_token(&slf.as_super().mla_module()?.fmha_quant_mode),
        )
    }

    #[getter(_gemm_quant_mode)]
    fn gemm_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "GEMMQuantMode",
            &enum_token(&slf.as_super().mla_module()?.gemm_quant_mode),
        )
    }
}

/// MLA pre/post BMM.
#[pyclass(extends = PyOperation, subclass, name = "MLABmm", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyMLABmm;

#[pymethods]
impl PyMLABmm {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "generation";

    #[new]
    #[pyo3(signature = (name, scale_factor, num_heads, quant_mode, if_pre=true))]
    fn new(
        name: String,
        scale_factor: f64,
        num_heads: u32,
        quant_mode: &Bound<'_, PyAny>,
        if_pre: bool,
    ) -> PyResult<(Self, PyOperation)> {
        let inner = Op::MlaBmm(MlaBmmOp {
            name,
            scale_factor,
            num_heads,
            quant_mode: gemm_quant(quant_mode)?,
            is_pre: if_pre,
        });
        Ok((PyMLABmm, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().mla_bmm()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.num_heads,
            enum_token(&o.quant_mode),
            o.is_pre,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_num_heads)]
    fn num_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mla_bmm()?.num_heads)
    }

    #[getter(_quant_mode)]
    fn quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "GEMMQuantMode", &enum_token(&slf.as_super().mla_bmm()?.quant_mode))
    }

    #[getter(_if_pre)]
    fn if_pre(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().mla_bmm()?.is_pre)
    }
}

// ---------------------------------------------------------------------------
// MoE family
// ---------------------------------------------------------------------------

/// Fused MoE FFN.
#[pyclass(extends = PyOperation, subclass, name = "MoE", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyMoE;

#[pymethods]
impl PyMoE {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size, quant_mode, workload_distribution, attention_dp_size, is_context=true, is_gated=true, *, moe_backend=None, enable_eplb=false, seq_split=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: &Bound<'_, PyAny>,
        workload_distribution: String,
        attention_dp_size: u32,
        is_context: bool,
        is_gated: bool,
        moe_backend: Option<String>,
        enable_eplb: bool,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        // The retired class was CP-aware but the MoE wire carries no
        // seq_split: models divide the token count at the construction site
        // (attention_dp globalization), so a non-default value here would be
        // silently ignored — reject it instead.
        if seq_split > 1 {
            return Err(PyNotImplementedError::new_err(
                "MoE carries no seq_split on the wire; apply CP token division at the model \
                 construction site (attention_dp/token globalization), not on the op.",
            ));
        }
        let inner = Op::Moe(crate::operators::MoeOp {
            name,
            scale_factor,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            attention_dp_size,
            quant_mode: moe_quant(quant_mode)?,
            workload_distribution,
            is_gated,
            moe_backend,
            enable_eplb,
            is_context,
        });
        Ok((PyMoE, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().moe()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.hidden_size,
            o.inter_size,
            o.topk,
            o.num_experts,
            o.moe_tp_size,
            o.moe_ep_size,
            enum_token(&o.quant_mode),
            o.workload_distribution.clone(),
            o.attention_dp_size,
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("is_context", o.is_context)?;
        kwargs.set_item("is_gated", o.is_gated)?;
        kwargs.set_item("moe_backend", o.moe_backend.clone())?;
        kwargs.set_item("enable_eplb", o.enable_eplb)?;
        Ok((args, kwargs))
    }

    #[getter(_hidden_size)]
    fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe()?.hidden_size)
    }

    #[getter(_inter_size)]
    fn inter_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe()?.inter_size)
    }

    #[getter(_topk)]
    fn topk(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe()?.topk)
    }

    #[getter(_num_experts)]
    fn num_experts(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe()?.num_experts)
    }

    #[getter(_moe_tp_size)]
    fn moe_tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe()?.moe_tp_size)
    }

    #[getter(_moe_ep_size)]
    fn moe_ep_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe()?.moe_ep_size)
    }

    #[getter(_attention_dp_size)]
    fn attention_dp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe()?.attention_dp_size)
    }

    #[getter(_quant_mode)]
    fn quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "MoEQuantMode", &enum_token(&slf.as_super().moe()?.quant_mode))
    }

    #[getter(_workload_distribution)]
    fn workload_distribution(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().moe()?.workload_distribution.clone())
    }

    #[getter(_is_context)]
    fn is_context(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().moe()?.is_context)
    }

    #[getter(_is_gated)]
    fn is_gated(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().moe()?.is_gated)
    }

    #[getter(_moe_backend)]
    fn moe_backend(slf: PyRef<'_, Self>) -> PyResult<Option<String>> {
        Ok(slf.as_super().moe()?.moe_backend.clone())
    }

    #[getter(_enable_eplb)]
    fn enable_eplb(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().moe()?.enable_eplb)
    }
}

/// MoE dispatch/combine communication. `backend` is a new REQUIRED keyword:
/// the retired serializer injected the framework name at compile time to
/// derive the dispatch flavor; construction owns it now.
/// `moe_backend="deepep_moe"` maps to the RetiredDeepEp tombstone flavor —
/// construction stays legal (Python builders still emit it), spec assembly
/// and evaluation refuse it, mirroring the retired conversion error.
#[pyclass(extends = PyOperation, subclass, name = "MoEDispatch", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyMoEDispatch;

impl PyOperation {
    fn moe_dispatch(&self) -> PyResult<&crate::operators::MoEDispatchOp> {
        match &self.inner {
            Op::MoeDispatch(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a MoEDispatch op")),
        }
    }
    fn moe_dispatch_mut(&mut self) -> PyResult<&mut crate::operators::MoEDispatchOp> {
        match &mut self.inner {
            Op::MoeDispatch(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a MoEDispatch op")),
        }
    }
}

#[pymethods]
impl PyMoEDispatch {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, hidden_size, topk, num_experts, moe_tp_size, moe_ep_size, attention_dp_size, pre_dispatch, enable_fp4_all2all=true, *, backend, sms=12, moe_backend=None, is_context=true, scale_num_tokens=1, quant_mode=None, reduce_results=true, attn_cp_size=1, attn_ar_modeled=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        hidden_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        attention_dp_size: u32,
        pre_dispatch: bool,
        enable_fp4_all2all: bool,
        backend: String,
        sms: u32,
        moe_backend: Option<String>,
        is_context: bool,
        scale_num_tokens: u32,
        quant_mode: Option<&Bound<'_, PyAny>>,
        reduce_results: bool,
        attn_cp_size: u32,
        attn_ar_modeled: bool,
    ) -> PyResult<(Self, PyOperation)> {
        use crate::common::enums::BackendKind;
        use crate::operators::DispatchFlavor;

        let _ = (enable_fp4_all2all, reduce_results);

        let backend_kind = match backend.as_str() {
            "trtllm" => BackendKind::Trtllm,
            "sglang" => BackendKind::Sglang,
            "vllm" => BackendKind::Vllm,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown backend for MoEDispatch: {other:?} (expected trtllm/sglang/vllm)"
                )))
            }
        };
        // The retired `_dispatch_flavor` derivation, minus the compile-time
        // raise: deepep_moe becomes the tombstone flavor and fails at spec
        // assembly / evaluation instead of at construction.
        let flavor = if moe_backend.as_deref() == Some("deepep_moe") {
            DispatchFlavor::RetiredDeepEp
        } else if backend_kind == BackendKind::Trtllm {
            DispatchFlavor::TrtllmAlltoall
        } else {
            DispatchFlavor::CustomAllReduce
        };
        let moe_quant_mode = match quant_mode {
            Some(obj) if !obj.is_none() => moe_quant(obj)?,
            _ => MoeQuantMode::Bfloat16,
        };
        let inner = Op::MoeDispatch(crate::operators::MoEDispatchOp {
            name,
            scale_factor,
            hidden_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            attention_dp_size,
            pre_dispatch,
            attn_ar_modeled,
            backend: backend_kind,
            flavor,
            comm_quant: CommQuantMode::Half,
            moe_quant: moe_quant_mode,
            attn_cp_size,
            is_context,
            sms,
            scale_num_tokens,
        });
        Ok((PyMoEDispatch, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        use crate::operators::DispatchFlavor;

        let o = slf.as_super().moe_dispatch()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.hidden_size,
            o.topk,
            o.num_experts,
            o.moe_tp_size,
            o.moe_ep_size,
            o.attention_dp_size,
            o.pre_dispatch,
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("backend", o.backend.as_str())?;
        kwargs.set_item("sms", o.sms)?;
        if o.flavor == DispatchFlavor::RetiredDeepEp {
            kwargs.set_item("moe_backend", "deepep_moe")?;
        }
        kwargs.set_item("is_context", o.is_context)?;
        kwargs.set_item("scale_num_tokens", o.scale_num_tokens)?;
        kwargs.set_item("quant_mode", enum_token(&o.moe_quant))?;
        kwargs.set_item("attn_cp_size", o.attn_cp_size)?;
        kwargs.set_item("attn_ar_modeled", o.attn_ar_modeled)?;
        Ok((args, kwargs))
    }

    #[getter(_hidden_size)]
    fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.hidden_size)
    }

    #[getter(_topk)]
    fn topk(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.topk)
    }

    #[getter(_num_experts)]
    fn num_experts(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.num_experts)
    }

    #[getter(_moe_tp_size)]
    fn moe_tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.moe_tp_size)
    }

    #[getter(_moe_ep_size)]
    fn moe_ep_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.moe_ep_size)
    }

    #[getter(_attention_dp_size)]
    fn attention_dp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.attention_dp_size)
    }

    #[getter(_attention_tp_size)]
    fn attention_tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        let o = slf.as_super().moe_dispatch()?;
        Ok((o.moe_tp_size * o.moe_ep_size) / o.attention_dp_size.max(1))
    }

    #[getter(num_gpus)]
    fn num_gpus(slf: PyRef<'_, Self>) -> PyResult<u32> {
        let o = slf.as_super().moe_dispatch()?;
        Ok(o.moe_ep_size * o.moe_tp_size)
    }

    #[getter(_pre_dispatch)]
    fn pre_dispatch(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().moe_dispatch()?.pre_dispatch)
    }

    #[getter(_sms)]
    fn sms(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.sms)
    }

    #[getter(_moe_backend)]
    fn moe_backend(slf: PyRef<'_, Self>) -> PyResult<Option<&'static str>> {
        use crate::operators::DispatchFlavor;
        Ok(match slf.as_super().moe_dispatch()?.flavor {
            DispatchFlavor::RetiredDeepEp => Some("deepep_moe"),
            _ => None,
        })
    }

    #[getter(_quant_mode)]
    fn quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "MoEQuantMode",
            &enum_token(&slf.as_super().moe_dispatch()?.moe_quant),
        )
    }

    #[getter(_scale_num_tokens)]
    fn scale_num_tokens(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.scale_num_tokens)
    }

    #[getter(_attn_ar_modeled)]
    fn attn_ar_modeled(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().moe_dispatch()?.attn_ar_modeled)
    }

    #[getter(_is_context)]
    fn is_context(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().moe_dispatch()?.is_context)
    }

    #[setter(_is_context)]
    fn set_is_context(mut slf: PyRefMut<'_, Self>, value: bool) -> PyResult<()> {
        slf.as_super().moe_dispatch_mut()?.is_context = value;
        Ok(())
    }

    #[getter(_attn_cp_size)]
    fn attn_cp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_dispatch()?.attn_cp_size)
    }

    #[setter(_attn_cp_size)]
    fn set_attn_cp_size(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().moe_dispatch_mut()?.attn_cp_size = value;
        Ok(())
    }
}

/// Unified large-EP all-to-all comm phase. Backend/phase feasibility
/// validation stays Python-side (the shell's `__init__` consults the
/// `MOE_A2A_BACKENDS` registry — single source in `operations/moe_comm.py`).
#[pyclass(extends = PyOperation, subclass, name = "MoEAllToAll", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyMoEAllToAll;

impl PyOperation {
    fn moe_a2a(&self) -> PyResult<&crate::operators::MoeAllToAllOp> {
        match &self.inner {
            Op::MoeAllToAll(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a MoEAllToAll op")),
        }
    }
}

#[pymethods]
impl PyMoEAllToAll {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, *, phase, comm_backend, hidden_size, topk, num_experts, moe_ep_size, node_num, comm_dtype="default", sms=0, attention_tp_size=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        phase: String,
        comm_backend: String,
        hidden_size: u32,
        topk: u32,
        num_experts: u32,
        moe_ep_size: u32,
        node_num: u32,
        comm_dtype: &str,
        sms: u32,
        attention_tp_size: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let inner = Op::MoeAllToAll(crate::operators::MoeAllToAllOp {
            name,
            scale_factor,
            phase,
            comm_backend,
            comm_dtype: comm_dtype.to_string(),
            hidden_size,
            topk,
            num_experts,
            moe_ep_size,
            node_num,
            sms,
            attention_tp_size,
        });
        Ok((PyMoEAllToAll, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().moe_a2a()?;
        let args = (o.name.clone(), o.scale_factor).into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("phase", o.phase.clone())?;
        kwargs.set_item("comm_backend", o.comm_backend.clone())?;
        kwargs.set_item("hidden_size", o.hidden_size)?;
        kwargs.set_item("topk", o.topk)?;
        kwargs.set_item("num_experts", o.num_experts)?;
        kwargs.set_item("moe_ep_size", o.moe_ep_size)?;
        kwargs.set_item("node_num", o.node_num)?;
        kwargs.set_item("comm_dtype", o.comm_dtype.clone())?;
        kwargs.set_item("sms", o.sms)?;
        kwargs.set_item("attention_tp_size", o.attention_tp_size)?;
        Ok((args, kwargs))
    }

    #[getter(_phase)]
    fn phase(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().moe_a2a()?.phase.clone())
    }

    #[getter(_comm_backend)]
    fn comm_backend(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().moe_a2a()?.comm_backend.clone())
    }

    #[getter(_comm_dtype)]
    fn comm_dtype(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().moe_a2a()?.comm_dtype.clone())
    }

    #[getter(_hidden_size)]
    fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_a2a()?.hidden_size)
    }

    #[getter(_topk)]
    fn topk(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_a2a()?.topk)
    }

    #[getter(_num_experts)]
    fn num_experts(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_a2a()?.num_experts)
    }

    #[getter(_moe_ep_size)]
    fn moe_ep_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_a2a()?.moe_ep_size)
    }

    #[getter(_node_num)]
    fn node_num(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_a2a()?.node_num)
    }

    #[getter(_sms)]
    fn sms(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_a2a()?.sms)
    }

    #[getter(_attention_tp_size)]
    fn attention_tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().moe_a2a()?.attention_tp_size)
    }
}

/// Large-EP local expert compute modeled from the stock MoE table.
#[pyclass(extends = PyOperation, subclass, name = "ModeledEPMoE", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyModeledEPMoE;

impl PyOperation {
    fn modeled_ep_moe(&self) -> PyResult<&crate::operators::ModeledEpMoeOp> {
        match &self.inner {
            Op::EpMoe(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a ModeledEPMoE op")),
        }
    }
}

#[pymethods]
impl PyModeledEPMoE {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, *, hidden_size, inter_size, topk, num_experts, moe_ep_size, quant_mode, attention_dp_size, inference_phase, is_gated=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_ep_size: u32,
        quant_mode: &Bound<'_, PyAny>,
        attention_dp_size: u32,
        inference_phase: String,
        is_gated: bool,
    ) -> PyResult<(Self, PyOperation)> {
        if moe_ep_size <= 1 {
            return Err(PyValueError::new_err(format!(
                "ModeledEPMoE requires moe_ep_size > 1, got {moe_ep_size}"
            )));
        }
        if num_experts % moe_ep_size != 0 {
            return Err(PyValueError::new_err(format!(
                "num_experts ({num_experts}) must be divisible by moe_ep_size ({moe_ep_size})"
            )));
        }
        if !matches!(inference_phase.as_str(), "context" | "generation") {
            return Err(PyValueError::new_err(format!(
                "invalid inference_phase {inference_phase:?}; expected context or generation"
            )));
        }
        let inner = Op::EpMoe(crate::operators::ModeledEpMoeOp {
            name,
            scale_factor,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_ep_size,
            quant_mode: moe_quant(quant_mode)?,
            workload_distribution: "balanced".to_string(),
            attention_dp_size,
            inference_phase,
            num_slots: None,
            kernel_source: None,
            is_gated,
            enable_eplb: false,
        });
        Ok((PyModeledEPMoE, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().modeled_ep_moe()?;
        let args = (o.name.clone(), o.scale_factor).into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("hidden_size", o.hidden_size)?;
        kwargs.set_item("inter_size", o.inter_size)?;
        kwargs.set_item("topk", o.topk)?;
        kwargs.set_item("num_experts", o.num_experts)?;
        kwargs.set_item("moe_ep_size", o.moe_ep_size)?;
        kwargs.set_item("quant_mode", enum_token(&o.quant_mode))?;
        kwargs.set_item("attention_dp_size", o.attention_dp_size)?;
        kwargs.set_item("inference_phase", o.inference_phase.clone())?;
        kwargs.set_item("is_gated", o.is_gated)?;
        Ok((args, kwargs))
    }

    #[getter(_hidden_size)]
    fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().modeled_ep_moe()?.hidden_size)
    }

    #[getter(_inter_size)]
    fn inter_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().modeled_ep_moe()?.inter_size)
    }

    #[getter(_topk)]
    fn topk(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().modeled_ep_moe()?.topk)
    }

    #[getter(_num_experts)]
    fn num_experts(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().modeled_ep_moe()?.num_experts)
    }

    #[getter(_moe_ep_size)]
    fn moe_ep_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().modeled_ep_moe()?.moe_ep_size)
    }

    #[getter(_quant_mode)]
    fn quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "MoEQuantMode",
            &enum_token(&slf.as_super().modeled_ep_moe()?.quant_mode),
        )
    }

    #[getter(_attention_dp_size)]
    fn attention_dp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().modeled_ep_moe()?.attention_dp_size)
    }

    #[getter(_inference_phase)]
    fn inference_phase(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().modeled_ep_moe()?.inference_phase.clone())
    }

    #[getter(_is_gated)]
    fn is_gated(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().modeled_ep_moe()?.is_gated)
    }
}

/// SGLang DeepSeek-V4 MegaMoE routed module (one class, both phases via
/// `is_context`). `workload_distribution` normalizes `uniform -> balanced`
/// at construction, the retired ctor's rule.
#[pyclass(extends = PyOperation, subclass, name = "DeepSeekV4MegaMoEModule", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyDeepSeekV4MegaMoEModule;

impl PyOperation {
    fn megamoe(&self) -> PyResult<&crate::operators::Dsv4MegaMoeOp> {
        match &self.inner {
            Op::Dsv4MegaMoe(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a DeepSeekV4MegaMoEModule op")),
        }
    }
}

#[pymethods]
impl PyDeepSeekV4MegaMoEModule {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "tokens";

    #[new]
    #[pyo3(signature = (name, scale_factor, hidden_size, inter_size, topk, num_experts, moe_tp_size, moe_ep_size, quant_mode, workload_distribution, is_context=true, source_policy="random", pre_dispatch="sglang_jit", num_fused_shared_experts=0, kernel_source="deepgemm_megamoe", kernel_dtype="fp8_fp4"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        hidden_size: u32,
        inter_size: u32,
        topk: u32,
        num_experts: u32,
        moe_tp_size: u32,
        moe_ep_size: u32,
        quant_mode: &Bound<'_, PyAny>,
        workload_distribution: String,
        is_context: bool,
        source_policy: &str,
        pre_dispatch: &str,
        num_fused_shared_experts: u32,
        kernel_source: &str,
        kernel_dtype: &str,
    ) -> PyResult<(Self, PyOperation)> {
        let workload_distribution = if workload_distribution == "uniform" {
            "balanced".to_string()
        } else {
            workload_distribution
        };
        let inner = Op::Dsv4MegaMoe(crate::operators::Dsv4MegaMoeOp {
            name,
            scale_factor,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant_mode: moe_quant(quant_mode)?,
            workload_distribution,
            is_context,
            source_policy: source_policy.to_string(),
            pre_dispatch: pre_dispatch.to_string(),
            num_fused_shared_experts,
            kernel_source: kernel_source.to_string(),
            kernel_dtype: kernel_dtype.to_string(),
        });
        Ok((PyDeepSeekV4MegaMoEModule, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().megamoe()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.hidden_size,
            o.inter_size,
            o.topk,
            o.num_experts,
            o.moe_tp_size,
            o.moe_ep_size,
            enum_token(&o.quant_mode),
            o.workload_distribution.clone(),
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("is_context", o.is_context)?;
        kwargs.set_item("source_policy", o.source_policy.clone())?;
        kwargs.set_item("pre_dispatch", o.pre_dispatch.clone())?;
        kwargs.set_item("num_fused_shared_experts", o.num_fused_shared_experts)?;
        kwargs.set_item("kernel_source", o.kernel_source.clone())?;
        kwargs.set_item("kernel_dtype", o.kernel_dtype.clone())?;
        Ok((args, kwargs))
    }

    #[getter(_hidden_size)]
    fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().megamoe()?.hidden_size)
    }

    #[getter(_inter_size)]
    fn inter_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().megamoe()?.inter_size)
    }

    #[getter(_topk)]
    fn topk(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().megamoe()?.topk)
    }

    #[getter(_num_experts)]
    fn num_experts(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().megamoe()?.num_experts)
    }

    #[getter(_moe_tp_size)]
    fn moe_tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().megamoe()?.moe_tp_size)
    }

    #[getter(_moe_ep_size)]
    fn moe_ep_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().megamoe()?.moe_ep_size)
    }

    #[getter(_quant_mode)]
    fn quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "MoEQuantMode", &enum_token(&slf.as_super().megamoe()?.quant_mode))
    }

    #[getter(_workload_distribution)]
    fn workload_distribution(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().megamoe()?.workload_distribution.clone())
    }

    #[getter(_is_context)]
    fn is_context(slf: PyRef<'_, Self>) -> PyResult<bool> {
        Ok(slf.as_super().megamoe()?.is_context)
    }

    #[getter(_source_policy)]
    fn source_policy(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().megamoe()?.source_policy.clone())
    }

    #[getter(_pre_dispatch)]
    fn pre_dispatch(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().megamoe()?.pre_dispatch.clone())
    }

    #[getter(_num_fused_shared_experts)]
    fn num_fused_shared_experts(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().megamoe()?.num_fused_shared_experts)
    }

    #[getter(_kernel_source)]
    fn kernel_source(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().megamoe()?.kernel_source.clone())
    }

    #[getter(_kernel_dtype)]
    fn kernel_dtype(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().megamoe()?.kernel_dtype.clone())
    }
}

inner_accessor!(moe, moe_mut, Moe, crate::operators::MoeOp, "MoE");

// ---------------------------------------------------------------------------
// State-space family (Mamba2 / GDN / KDA kernels)
// ---------------------------------------------------------------------------

inner_accessor!(mamba2, mamba2_mut, Mamba2, crate::operators::Mamba2Op, "Mamba2Kernel");
inner_accessor!(gdn, gdn_mut, Gdn, crate::operators::GdnOp, "GDNKernel");
inner_accessor!(kda, kda_mut, Kda, crate::operators::KdaOp, "KDAKernel");

/// Single Mamba2 kernel (conv1d or SSM). `seq_split` is accepted for
/// calling-shape compatibility but gated: the family never opted into CP.
#[pyclass(extends = PyOperation, subclass, name = "Mamba2Kernel", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyMamba2Kernel;

#[pymethods]
impl PyMamba2Kernel {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "module";

    #[new]
    #[pyo3(signature = (name, scale_factor, kernel_source, phase, hidden_size, nheads, head_dim, d_state, d_conv, n_groups, chunk_size, seq_split=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        kernel_source: String,
        phase: String,
        hidden_size: u32,
        nheads: u32,
        head_dim: u32,
        d_state: u32,
        d_conv: u32,
        n_groups: u32,
        chunk_size: u32,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        cp_audit_gate("Mamba2Kernel", false, seq_split)?;
        let inner = Op::Mamba2(crate::operators::Mamba2Op {
            name,
            scale_factor,
            kernel_source,
            phase,
            d_model: hidden_size,
            d_state,
            d_conv,
            nheads,
            head_dim,
            n_groups,
            chunk_size,
        });
        Ok((PyMamba2Kernel, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().mamba2()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.kernel_source.clone(),
            o.phase.clone(),
            o.d_model,
            o.nheads,
            o.head_dim,
            o.d_state,
            o.d_conv,
            o.n_groups,
            o.chunk_size,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_kernel_source)]
    fn kernel_source(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().mamba2()?.kernel_source.clone())
    }

    #[getter(_phase)]
    fn phase(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().mamba2()?.phase.clone())
    }

    #[setter(_phase)]
    fn set_phase(mut slf: PyRefMut<'_, Self>, value: String) -> PyResult<()> {
        slf.as_super().mamba2_mut()?.phase = value;
        Ok(())
    }

    #[getter(_hidden_size)]
    fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mamba2()?.d_model)
    }

    #[getter(_nheads)]
    fn nheads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mamba2()?.nheads)
    }

    #[getter(_head_dim)]
    fn head_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mamba2()?.head_dim)
    }

    #[getter(_d_state)]
    fn d_state(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mamba2()?.d_state)
    }

    #[getter(_d_conv)]
    fn d_conv(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mamba2()?.d_conv)
    }

    #[getter(_n_groups)]
    fn n_groups(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mamba2()?.n_groups)
    }

    #[getter(_chunk_size)]
    fn chunk_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().mamba2()?.chunk_size)
    }
}

/// Single Gated DeltaNet kernel (Qwen3.5 linear attention).
#[pyclass(extends = PyOperation, subclass, name = "GDNKernel", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyGDNKernel;

#[pymethods]
impl PyGDNKernel {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "module";

    #[new]
    #[pyo3(signature = (name, scale_factor, kernel_source, phase, d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv, seq_split=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        kernel_source: String,
        phase: String,
        d_model: u32,
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
        d_conv: u32,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        cp_audit_gate("GDNKernel", false, seq_split)?;
        let inner = Op::Gdn(crate::operators::GdnOp {
            name,
            scale_factor,
            kernel_source,
            phase,
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
        });
        Ok((PyGDNKernel, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().gdn()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.kernel_source.clone(),
            o.phase.clone(),
            o.d_model,
            o.num_k_heads,
            o.head_k_dim,
            o.num_v_heads,
            o.head_v_dim,
            o.d_conv,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_kernel_source)]
    fn kernel_source(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().gdn()?.kernel_source.clone())
    }

    #[getter(_phase)]
    fn phase(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().gdn()?.phase.clone())
    }

    #[setter(_phase)]
    fn set_phase(mut slf: PyRefMut<'_, Self>, value: String) -> PyResult<()> {
        slf.as_super().gdn_mut()?.phase = value;
        Ok(())
    }

    #[getter(_d_model)]
    fn d_model(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gdn()?.d_model)
    }

    #[getter(_num_k_heads)]
    fn num_k_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gdn()?.num_k_heads)
    }

    #[getter(_head_k_dim)]
    fn head_k_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gdn()?.head_k_dim)
    }

    #[getter(_num_v_heads)]
    fn num_v_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gdn()?.num_v_heads)
    }

    #[getter(_head_v_dim)]
    fn head_v_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gdn()?.head_v_dim)
    }

    #[getter(_d_conv)]
    fn d_conv(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().gdn()?.d_conv)
    }
}

/// Single KDA (Kimi Delta Attention) kernel — GDN plus the verify phase's
/// `draft_tokens`. A distinct engine variant, NOT a Python subclass of
/// GDNKernel any more (the classes are construction handles; the kernels'
/// tables are separate).
#[pyclass(extends = PyOperation, subclass, name = "KDAKernel", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyKDAKernel;

#[pymethods]
impl PyKDAKernel {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "module";

    #[new]
    #[pyo3(signature = (name, scale_factor, kernel_source, phase, d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv, seq_split=1, draft_tokens=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        kernel_source: String,
        phase: String,
        d_model: u32,
        num_k_heads: u32,
        head_k_dim: u32,
        num_v_heads: u32,
        head_v_dim: u32,
        d_conv: u32,
        seq_split: u32,
        draft_tokens: i64,
    ) -> PyResult<(Self, PyOperation)> {
        cp_audit_gate("KDAKernel", false, seq_split)?;
        let inner = Op::Kda(crate::operators::KdaOp {
            name,
            scale_factor,
            kernel_source,
            phase,
            d_model,
            d_conv,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            draft_tokens,
        });
        Ok((PyKDAKernel, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().kda()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.kernel_source.clone(),
            o.phase.clone(),
            o.d_model,
            o.num_k_heads,
            o.head_k_dim,
            o.num_v_heads,
            o.head_v_dim,
            o.d_conv,
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("draft_tokens", o.draft_tokens)?;
        Ok((args, kwargs))
    }

    #[getter(_kernel_source)]
    fn kernel_source(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().kda()?.kernel_source.clone())
    }

    #[getter(_phase)]
    fn phase(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().kda()?.phase.clone())
    }

    #[setter(_phase)]
    fn set_phase(mut slf: PyRefMut<'_, Self>, value: String) -> PyResult<()> {
        slf.as_super().kda_mut()?.phase = value;
        Ok(())
    }

    #[getter(_d_model)]
    fn d_model(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().kda()?.d_model)
    }

    #[getter(_draft_tokens)]
    fn draft_tokens(slf: PyRef<'_, Self>) -> PyResult<i64> {
        Ok(slf.as_super().kda()?.draft_tokens)
    }

    #[getter(_num_k_heads)]
    fn num_k_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().kda()?.num_k_heads)
    }

    #[getter(_head_k_dim)]
    fn head_k_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().kda()?.head_k_dim)
    }

    #[getter(_num_v_heads)]
    fn num_v_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().kda()?.num_v_heads)
    }

    #[getter(_head_v_dim)]
    fn head_v_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().kda()?.head_v_dim)
    }

    #[getter(_d_conv)]
    fn d_conv(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().kda()?.d_conv)
    }
}

// ---------------------------------------------------------------------------
// WideEP MLA family
// ---------------------------------------------------------------------------

inner_accessor!(
    wideep_context_mla,
    wideep_context_mla_mut,
    WideEpContextMla,
    crate::operators::WideEpContextMlaOp,
    "WideEPContextMLA"
);
inner_accessor!(
    wideep_generation_mla,
    wideep_generation_mla_mut,
    WideEpGenerationMla,
    crate::operators::WideEpGenerationMlaOp,
    "WideEPGenerationMLA"
);

/// WideEP prefill MLA. The op takes `tp_size`; the engine table axis is the
/// per-rank head count `128 // tp_size` (DeepSeek's 128 total heads), the
/// retired serializer's derivation, now applied at construction.
#[pyclass(extends = PyOperation, subclass, name = "WideEPContextMLA", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyWideEPContextMLA;

#[pymethods]
impl PyWideEPContextMLA {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "context";

    #[new]
    #[pyo3(signature = (name, scale_factor, tp_size, kvcache_quant_mode, fmha_quant_mode, attn_backend="flashinfer", cp_size=1))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        tp_size: u32,
        kvcache_quant_mode: &Bound<'_, PyAny>,
        fmha_quant_mode: &Bound<'_, PyAny>,
        attn_backend: &str,
        cp_size: u32,
    ) -> PyResult<(Self, PyOperation)> {
        if tp_size == 0 || 128 % tp_size != 0 {
            return Err(PyValueError::new_err(format!(
                "WideEPContextMLA tp_size must divide 128, got {tp_size}"
            )));
        }
        let inner = Op::WideEpContextMla(crate::operators::WideEpContextMlaOp {
            name,
            scale_factor,
            num_heads: 128 / tp_size,
            kv_cache_dtype: kv_quant(kvcache_quant_mode)?,
            fmha_quant_mode: fmha_quant(fmha_quant_mode)?,
            attn_backend: attn_backend.to_string(),
            cp_size,
        });
        Ok((PyWideEPContextMLA, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().wideep_context_mla()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            128 / o.num_heads.max(1),
            enum_token(&o.kv_cache_dtype),
            enum_token(&o.fmha_quant_mode),
            o.attn_backend.clone(),
            o.cp_size,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_tp_size)]
    fn tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(128 / slf.as_super().wideep_context_mla()?.num_heads.max(1))
    }

    #[getter(_kvcache_quant_mode)]
    fn kvcache_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "KVCacheQuantMode",
            &enum_token(&slf.as_super().wideep_context_mla()?.kv_cache_dtype),
        )
    }

    #[getter(_fmha_quant_mode)]
    fn fmha_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "FMHAQuantMode",
            &enum_token(&slf.as_super().wideep_context_mla()?.fmha_quant_mode),
        )
    }

    #[getter(_attn_backend)]
    fn attn_backend(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().wideep_context_mla()?.attn_backend.clone())
    }

    #[getter(_cp_size)]
    fn cp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().wideep_context_mla()?.cp_size)
    }

    #[setter(_cp_size)]
    fn set_cp_size(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().wideep_context_mla_mut()?.cp_size = value;
        Ok(())
    }
}

/// WideEP decode MLA.
#[pyclass(extends = PyOperation, subclass, name = "WideEPGenerationMLA", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyWideEPGenerationMLA;

#[pymethods]
impl PyWideEPGenerationMLA {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "generation";

    #[new]
    #[pyo3(signature = (name, scale_factor, tp_size, kvcache_quant_mode, fmha_quant_mode, attn_backend="flashinfer"))]
    fn new(
        name: String,
        scale_factor: f64,
        tp_size: u32,
        kvcache_quant_mode: &Bound<'_, PyAny>,
        fmha_quant_mode: &Bound<'_, PyAny>,
        attn_backend: &str,
    ) -> PyResult<(Self, PyOperation)> {
        if tp_size == 0 || 128 % tp_size != 0 {
            return Err(PyValueError::new_err(format!(
                "WideEPGenerationMLA tp_size must divide 128, got {tp_size}"
            )));
        }
        let inner = Op::WideEpGenerationMla(crate::operators::WideEpGenerationMlaOp {
            name,
            scale_factor,
            num_heads: 128 / tp_size,
            kv_cache_dtype: kv_quant(kvcache_quant_mode)?,
            fmha_quant_mode: fmha_quant(fmha_quant_mode)?,
            attn_backend: attn_backend.to_string(),
        });
        Ok((PyWideEPGenerationMLA, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().wideep_generation_mla()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            128 / o.num_heads.max(1),
            enum_token(&o.kv_cache_dtype),
            enum_token(&o.fmha_quant_mode),
            o.attn_backend.clone(),
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_tp_size)]
    fn tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(128 / slf.as_super().wideep_generation_mla()?.num_heads.max(1))
    }

    #[getter(_kvcache_quant_mode)]
    fn kvcache_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "KVCacheQuantMode",
            &enum_token(&slf.as_super().wideep_generation_mla()?.kv_cache_dtype),
        )
    }

    #[getter(_fmha_quant_mode)]
    fn fmha_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(
            py,
            "FMHAQuantMode",
            &enum_token(&slf.as_super().wideep_generation_mla()?.fmha_quant_mode),
        )
    }

    #[getter(_attn_backend)]
    fn attn_backend(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().wideep_generation_mla()?.attn_backend.clone())
    }
}

// ---------------------------------------------------------------------------
// MSA modules (context / generation share the payload struct)
// ---------------------------------------------------------------------------

impl PyOperation {
    fn msa(&self) -> PyResult<&crate::operators::MsaModuleOp> {
        match &self.inner {
            Op::MsaContext(o) | Op::MsaGeneration(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not an MSA module op")),
        }
    }
}

macro_rules! msa_class {
    ($cls:ident, $py_name:literal, $variant:ident, $is_context:literal) => {
        #[doc = concat!("MiniMax MSA ", $py_name, " module.")]
        #[pyclass(extends = PyOperation, subclass, name = $py_name, module = "aiconfigurator_core._aiconfigurator_core")]
        pub struct $cls;

        #[pymethods]
        impl $cls {
            #[classattr]
            #[allow(non_upper_case_globals)]
            const _CP_AWARE: bool = false;

            #[classattr]
            #[allow(non_upper_case_globals)]
            const _ENGINE_QUERY_SHAPE: &'static str =
                if $is_context { "context" } else { "generation" };

            #[new]
            #[pyo3(signature = (name, scale_factor, num_heads, num_kv_heads, hidden_size, head_dim, v_head_dim, index_n_heads, index_head_dim, index_topk, block_size, kvcache_quant_mode, fmha_quant_mode, gemm_quant_mode, dsa_architecture="GlmMoeDsaForCausalLM", dsa_scale_k=1.0))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                name: String,
                scale_factor: f64,
                num_heads: u32,
                num_kv_heads: u32,
                hidden_size: u32,
                head_dim: u32,
                v_head_dim: u32,
                index_n_heads: u32,
                index_head_dim: u32,
                index_topk: u32,
                block_size: u32,
                kvcache_quant_mode: &Bound<'_, PyAny>,
                fmha_quant_mode: &Bound<'_, PyAny>,
                gemm_quant_mode: &Bound<'_, PyAny>,
                dsa_architecture: &str,
                dsa_scale_k: f64,
            ) -> PyResult<(Self, PyOperation)> {
                let module = crate::operators::MsaModuleOp {
                    name,
                    scale_factor,
                    num_heads,
                    num_kv_heads,
                    hidden_size,
                    head_dim,
                    v_head_dim,
                    index_n_heads,
                    index_head_dim,
                    index_topk,
                    block_size,
                    kv_cache_dtype: kv_quant(kvcache_quant_mode)?,
                    fmha_quant_mode: fmha_quant(fmha_quant_mode)?,
                    gemm_quant_mode: gemm_quant(gemm_quant_mode)?,
                    dsa_architecture: dsa_architecture.to_string(),
                    dsa_scale_k,
                };
                Ok(($cls, PyOperation { inner: Op::$variant(module) }))
            }

            fn __getnewargs_ex__<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
                let o = slf.as_super().msa()?;
                let args = (
                    o.name.clone(),
                    o.scale_factor,
                    o.num_heads,
                    o.num_kv_heads,
                    o.hidden_size,
                    o.head_dim,
                    o.v_head_dim,
                    o.index_n_heads,
                    o.index_head_dim,
                    o.index_topk,
                    o.block_size,
                )
                    .into_pyobject(py)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("kvcache_quant_mode", enum_token(&o.kv_cache_dtype))?;
                kwargs.set_item("fmha_quant_mode", enum_token(&o.fmha_quant_mode))?;
                kwargs.set_item("gemm_quant_mode", enum_token(&o.gemm_quant_mode))?;
                kwargs.set_item("dsa_architecture", o.dsa_architecture.clone())?;
                kwargs.set_item("dsa_scale_k", o.dsa_scale_k)?;
                Ok((args, kwargs))
            }

            #[getter(_is_context)]
            fn is_context(_slf: PyRef<'_, Self>) -> bool {
                $is_context
            }

            #[getter(_num_heads)]
            fn num_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.num_heads)
            }

            #[getter(_hidden_size)]
            fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.hidden_size)
            }

            #[getter(_index_topk)]
            fn index_topk(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.index_topk)
            }

            #[getter(_gemm_quant_mode)]
            fn gemm_quant_mode<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                py_enum_member(
                    py,
                    "GEMMQuantMode",
                    &enum_token(&slf.as_super().msa()?.gemm_quant_mode),
                )
            }

            #[getter(_num_kv_heads)]
            fn num_kv_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.num_kv_heads)
            }

            #[getter(_head_dim)]
            fn head_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.head_dim)
            }

            #[getter(_v_head_dim)]
            fn v_head_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.v_head_dim)
            }

            #[getter(_index_n_heads)]
            fn index_n_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.index_n_heads)
            }

            #[getter(_index_head_dim)]
            fn index_head_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.index_head_dim)
            }

            #[getter(_block_size)]
            fn block_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().msa()?.block_size)
            }

            #[getter(_kvcache_quant_mode)]
            fn kvcache_quant_mode<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                py_enum_member(py, "KVCacheQuantMode", &enum_token(&slf.as_super().msa()?.kv_cache_dtype))
            }

            #[getter(_fmha_quant_mode)]
            fn fmha_quant_mode<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                py_enum_member(py, "FMHAQuantMode", &enum_token(&slf.as_super().msa()?.fmha_quant_mode))
            }

            #[getter(_dsa_architecture)]
            fn dsa_architecture(slf: PyRef<'_, Self>) -> PyResult<String> {
                Ok(slf.as_super().msa()?.dsa_architecture.clone())
            }

            #[getter(_dsa_scale_k)]
            fn dsa_scale_k(slf: PyRef<'_, Self>) -> PyResult<f64> {
                Ok(slf.as_super().msa()?.dsa_scale_k)
            }
        }
    };
}

msa_class!(PyContextMSAModule, "ContextMSAModule", MsaContext, true);
msa_class!(PyGenerationMSAModule, "GenerationMSAModule", MsaGeneration, false);

// ---------------------------------------------------------------------------
// DSA modules
// ---------------------------------------------------------------------------

impl PyOperation {
    fn dsa(&self) -> PyResult<&crate::operators::DsaModuleOp> {
        match &self.inner {
            Op::DsaContext(o) | Op::DsaGeneration(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a DSA module op")),
        }
    }
    fn dsa_mut(&mut self) -> PyResult<&mut crate::operators::DsaModuleOp> {
        match &mut self.inner {
            Op::DsaContext(o) | Op::DsaGeneration(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a DSA module op")),
        }
    }
}

/// `index_topk` for a DSA architecture — the retired serializer's
/// `DSA_MODEL_DIMS[arch]["index_topk"]` lookup (unknown arch falls back to
/// the default architecture's boundary, as `dict.get(arch, DIMS[DEFAULT])`
/// did). Both shipped architectures use 2048; a NEW architecture with a
/// different boundary must extend this fn AND the Python `DSA_MODEL_DIMS`
/// table (checkpoint facts, `operations/dsa.py`).
fn dsa_index_topk(_architecture: &str) -> u32 {
    2048
}

/// The four DSA projection groups' quant map: missing groups fill from
/// `gemm_quant_mode`, unknown group names fail loudly (the retired
/// `_normalize_projection_quant_modes`).
fn dsa_projection_quants(
    overrides: Option<&Bound<'_, PyDict>>,
    gemm: GemmQuantMode,
) -> PyResult<DsaProjectionQuants> {
    let mut quants = DsaProjectionQuants {
        q: gemm,
        kv: gemm,
        o: gemm,
        indexer: gemm,
    };
    if let Some(map) = overrides {
        let mut unknown: Vec<String> = Vec::new();
        for (key, value) in map.iter() {
            let group: String = key.extract()?;
            let mode = gemm_quant(&value)?;
            match group.as_str() {
                "q" => quants.q = mode,
                "kv" => quants.kv = mode,
                "o" => quants.o = mode,
                "indexer" => quants.indexer = mode,
                _ => unknown.push(group),
            }
        }
        if !unknown.is_empty() {
            unknown.sort();
            return Err(PyValueError::new_err(format!(
                "unknown DSA projection group(s) {unknown:?}; expected a subset of (\"q\", \"kv\", \"o\", \"indexer\")"
            )));
        }
    }
    Ok(quants)
}

fn dsa_projection_dict<'py>(
    py: Python<'py>,
    quants: &DsaProjectionQuants,
) -> PyResult<Bound<'py, PyDict>> {
    let map = PyDict::new(py);
    map.set_item("q", py_enum_member(py, "GEMMQuantMode", &enum_token(&quants.q))?)?;
    map.set_item("kv", py_enum_member(py, "GEMMQuantMode", &enum_token(&quants.kv))?)?;
    map.set_item("o", py_enum_member(py, "GEMMQuantMode", &enum_token(&quants.o))?)?;
    map.set_item(
        "indexer",
        py_enum_member(py, "GEMMQuantMode", &enum_token(&quants.indexer))?,
    )?;
    Ok(map)
}

/// GLM-5 / DeepSeek-V3.2 sparse-attention context module.
#[pyclass(extends = PyOperation, subclass, name = "ContextDSAModule", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyContextDSAModule;

#[pymethods]
impl PyContextDSAModule {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "context";

    #[new]
    #[pyo3(signature = (name, scale_factor, num_heads, kvcache_quant_mode, fmha_quant_mode, gemm_quant_mode, architecture="DeepseekV32ForCausalLM", cp_size=1, index_topk_freq=1, dsa_full_layer_fraction=None, attn_projection_quant_modes=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        num_heads: u32,
        kvcache_quant_mode: &Bound<'_, PyAny>,
        fmha_quant_mode: &Bound<'_, PyAny>,
        gemm_quant_mode: &Bound<'_, PyAny>,
        architecture: &str,
        cp_size: u32,
        index_topk_freq: i64,
        dsa_full_layer_fraction: Option<f64>,
        attn_projection_quant_modes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Self, PyOperation)> {
        let gemm = gemm_quant(gemm_quant_mode)?;
        let freq = index_topk_freq.max(1) as f64;
        let inner = Op::DsaContext(crate::operators::DsaModuleOp {
            name,
            scale_factor,
            num_heads,
            kv_cache_dtype: kv_quant(kvcache_quant_mode)?,
            fmha_quant_mode: fmha_quant(fmha_quant_mode)?,
            gemm_quant_mode: gemm,
            architecture: architecture.to_string(),
            index_topk: dsa_index_topk(architecture),
            cp_size,
            full_frac: dsa_full_layer_fraction.unwrap_or(1.0 / freq),
            attn_projection_quant_modes: Some(dsa_projection_quants(
                attn_projection_quant_modes,
                gemm,
            )?),
        });
        Ok((PyContextDSAModule, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().dsa()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.num_heads,
            enum_token(&o.kv_cache_dtype),
            enum_token(&o.fmha_quant_mode),
            enum_token(&o.gemm_quant_mode),
            o.architecture.clone(),
            o.cp_size,
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dsa_full_layer_fraction", o.full_frac)?;
        if let Some(quants) = &o.attn_projection_quant_modes {
            kwargs.set_item("attn_projection_quant_modes", dsa_projection_dict(py, quants)?)?;
        }
        Ok((args, kwargs))
    }

    #[getter(_num_heads)]
    fn num_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().dsa()?.num_heads)
    }

    #[getter(_architecture)]
    fn architecture(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().dsa()?.architecture.clone())
    }

    #[getter(_full_frac)]
    fn full_frac(slf: PyRef<'_, Self>) -> PyResult<f64> {
        Ok(slf.as_super().dsa()?.full_frac)
    }

    #[getter(_gemm_quant_mode)]
    fn gemm_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "GEMMQuantMode", &enum_token(&slf.as_super().dsa()?.gemm_quant_mode))
    }

    #[getter(_kvcache_quant_mode)]
    fn kvcache_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "KVCacheQuantMode", &enum_token(&slf.as_super().dsa()?.kv_cache_dtype))
    }

    #[getter(_fmha_quant_mode)]
    fn fmha_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "FMHAQuantMode", &enum_token(&slf.as_super().dsa()?.fmha_quant_mode))
    }

    #[getter(_attn_projection_quant_modes)]
    fn attn_projection_quant_modes<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &slf.as_super().dsa()?.attn_projection_quant_modes {
            Some(quants) => Ok(Some(dsa_projection_dict(py, quants)?)),
            None => Ok(None),
        }
    }

    #[getter(_cp_size)]
    fn cp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().dsa()?.cp_size)
    }

    #[setter(_cp_size)]
    fn set_cp_size(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
        slf.as_super().dsa_mut()?.cp_size = value;
        Ok(())
    }
}

/// GLM-5 / DeepSeek-V3.2 sparse-attention generation module. The retired
/// class had no separate FMHA mode; the wire carries bfloat16 for it.
#[pyclass(extends = PyOperation, subclass, name = "GenerationDSAModule", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyGenerationDSAModule;

#[pymethods]
impl PyGenerationDSAModule {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = false;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "generation";

    #[new]
    #[pyo3(signature = (name, scale_factor, num_heads, kv_cache_dtype, gemm_quant_mode, architecture="DeepseekV32ForCausalLM", index_topk_freq=1, dsa_full_layer_fraction=None, attn_projection_quant_modes=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scale_factor: f64,
        num_heads: u32,
        kv_cache_dtype: &Bound<'_, PyAny>,
        gemm_quant_mode: &Bound<'_, PyAny>,
        architecture: &str,
        index_topk_freq: i64,
        dsa_full_layer_fraction: Option<f64>,
        attn_projection_quant_modes: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Self, PyOperation)> {
        let gemm = gemm_quant(gemm_quant_mode)?;
        let freq = index_topk_freq.max(1) as f64;
        let inner = Op::DsaGeneration(crate::operators::DsaModuleOp {
            name,
            scale_factor,
            num_heads,
            kv_cache_dtype: kv_quant(kv_cache_dtype)?,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            gemm_quant_mode: gemm,
            architecture: architecture.to_string(),
            index_topk: dsa_index_topk(architecture),
            cp_size: 1,
            full_frac: dsa_full_layer_fraction.unwrap_or(1.0 / freq),
            attn_projection_quant_modes: Some(dsa_projection_quants(
                attn_projection_quant_modes,
                gemm,
            )?),
        });
        Ok((PyGenerationDSAModule, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().dsa()?;
        let args = (
            o.name.clone(),
            o.scale_factor,
            o.num_heads,
            enum_token(&o.kv_cache_dtype),
            enum_token(&o.gemm_quant_mode),
            o.architecture.clone(),
        )
            .into_pyobject(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("dsa_full_layer_fraction", o.full_frac)?;
        if let Some(quants) = &o.attn_projection_quant_modes {
            kwargs.set_item("attn_projection_quant_modes", dsa_projection_dict(py, quants)?)?;
        }
        Ok((args, kwargs))
    }

    #[getter(_num_heads)]
    fn num_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
        Ok(slf.as_super().dsa()?.num_heads)
    }

    #[getter(_architecture)]
    fn architecture(slf: PyRef<'_, Self>) -> PyResult<String> {
        Ok(slf.as_super().dsa()?.architecture.clone())
    }

    #[getter(_full_frac)]
    fn full_frac(slf: PyRef<'_, Self>) -> PyResult<f64> {
        Ok(slf.as_super().dsa()?.full_frac)
    }

    #[getter(_gemm_quant_mode)]
    fn gemm_quant_mode<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "GEMMQuantMode", &enum_token(&slf.as_super().dsa()?.gemm_quant_mode))
    }

    #[getter(_kv_cache_dtype)]
    fn kv_cache_dtype<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py_enum_member(py, "KVCacheQuantMode", &enum_token(&slf.as_super().dsa()?.kv_cache_dtype))
    }

    #[getter(_attn_projection_quant_modes)]
    fn attn_projection_quant_modes<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        match &slf.as_super().dsa()?.attn_projection_quant_modes {
            Some(quants) => Ok(Some(dsa_projection_dict(py, quants)?)),
            None => Ok(None),
        }
    }
}

// ---------------------------------------------------------------------------
// DSV4 attention modules (context / generation share the payload struct)
// ---------------------------------------------------------------------------

impl PyOperation {
    fn dsv4(&self) -> PyResult<&crate::operators::Dsv4ModuleOp> {
        match &self.inner {
            Op::Dsv4Context(o) | Op::Dsv4Generation(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a DSV4 attention module op")),
        }
    }
    fn dsv4_mut(&mut self) -> PyResult<&mut crate::operators::Dsv4ModuleOp> {
        match &mut self.inner {
            Op::Dsv4Context(o) | Op::Dsv4Generation(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a DSV4 attention module op")),
        }
    }
}

macro_rules! dsv4_class {
    ($cls:ident, $py_name:literal, $variant:ident, $is_context:literal) => {
        #[doc = concat!("DeepSeek-V4 ", $py_name, ". `attn_kind` derives from `compress_ratio` (4 -> Csa, else Hca), the retired serializer's rule; `architecture` is a new REQUIRED keyword (the serializer injected `model.architecture` at compile time).")]
        #[pyclass(extends = PyOperation, subclass, name = $py_name, module = "aiconfigurator_core._aiconfigurator_core")]
        pub struct $cls;

        #[pymethods]
        impl $cls {
            #[classattr]
            #[allow(non_upper_case_globals)]
            const _CP_AWARE: bool = false;

            #[classattr]
            #[allow(non_upper_case_globals)]
            const _ENGINE_QUERY_SHAPE: &'static str =
                if $is_context { "context" } else { "generation" };

            #[new]
            #[pyo3(signature = (name, scale_factor, num_heads, native_heads, tp_size, hidden_size, q_lora_rank, o_lora_rank, head_dim, rope_head_dim, index_n_heads, index_head_dim, index_topk, window_size, compress_ratio, o_groups, kvcache_quant_mode, fmha_quant_mode, gemm_quant_mode, *, architecture, cp_size=1))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                name: String,
                scale_factor: f64,
                num_heads: u32,
                native_heads: u32,
                tp_size: u32,
                hidden_size: u32,
                q_lora_rank: u32,
                o_lora_rank: u32,
                head_dim: u32,
                rope_head_dim: u32,
                index_n_heads: u32,
                index_head_dim: u32,
                index_topk: u32,
                window_size: u32,
                compress_ratio: u32,
                o_groups: u32,
                kvcache_quant_mode: &Bound<'_, PyAny>,
                fmha_quant_mode: &Bound<'_, PyAny>,
                gemm_quant_mode: &Bound<'_, PyAny>,
                architecture: &str,
                cp_size: u32,
            ) -> PyResult<(Self, PyOperation)> {
                use crate::perf_database::AttnKind;

                let module = crate::operators::Dsv4ModuleOp {
                    name,
                    scale_factor,
                    attn_kind: if compress_ratio == 4 { AttnKind::Csa } else { AttnKind::Hca },
                    num_heads,
                    native_heads,
                    tp_size,
                    kv_cache_dtype: kv_quant(kvcache_quant_mode)?,
                    fmha_quant_mode: fmha_quant(fmha_quant_mode)?,
                    gemm_quant_mode: gemm_quant(gemm_quant_mode)?,
                    architecture: architecture.to_string(),
                    cp_size,
                    window_size: Some(window_size),
                    hidden_size,
                    q_lora_rank,
                    o_lora_rank,
                    head_dim,
                    rope_head_dim,
                    index_n_heads,
                    index_head_dim,
                    index_topk,
                    o_groups: Some(o_groups),
                };
                Ok(($cls, PyOperation { inner: Op::$variant(module) }))
            }

            fn __getnewargs_ex__<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
                use crate::perf_database::AttnKind;

                let o = slf.as_super().dsv4()?;
                let args = (
                    o.name.clone(),
                    o.scale_factor,
                    o.num_heads,
                    o.native_heads,
                    o.tp_size,
                    o.hidden_size,
                    o.q_lora_rank,
                    o.o_lora_rank,
                    o.head_dim,
                    o.rope_head_dim,
                    o.index_n_heads,
                )
                    .into_pyobject(py)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("index_head_dim", o.index_head_dim)?;
                kwargs.set_item("index_topk", o.index_topk)?;
                kwargs.set_item("window_size", o.window_size.unwrap_or(0))?;
                kwargs.set_item(
                    "compress_ratio",
                    if o.attn_kind == AttnKind::Csa { 4u32 } else { 128u32 },
                )?;
                kwargs.set_item("o_groups", o.o_groups.unwrap_or(1))?;
                kwargs.set_item("kvcache_quant_mode", enum_token(&o.kv_cache_dtype))?;
                kwargs.set_item("fmha_quant_mode", enum_token(&o.fmha_quant_mode))?;
                kwargs.set_item("gemm_quant_mode", enum_token(&o.gemm_quant_mode))?;
                kwargs.set_item("architecture", o.architecture.clone())?;
                kwargs.set_item("cp_size", o.cp_size)?;
                Ok((args, kwargs))
            }

            #[getter(_is_context)]
            fn is_context(_slf: PyRef<'_, Self>) -> bool {
                $is_context
            }

            #[getter(_num_heads)]
            fn num_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.num_heads)
            }

            #[getter(_native_heads)]
            fn native_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.native_heads)
            }

            #[getter(_tp_size)]
            fn tp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.tp_size)
            }

            #[getter(_hidden_size)]
            fn hidden_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.hidden_size)
            }

            #[getter(_compress_ratio)]
            fn compress_ratio(slf: PyRef<'_, Self>) -> PyResult<u32> {
                use crate::perf_database::AttnKind;
                Ok(if slf.as_super().dsv4()?.attn_kind == AttnKind::Csa { 4 } else { 128 })
            }

            #[getter(_window_size)]
            fn window_size(slf: PyRef<'_, Self>) -> PyResult<Option<u32>> {
                Ok(slf.as_super().dsv4()?.window_size)
            }

            #[getter(_o_groups)]
            fn o_groups(slf: PyRef<'_, Self>) -> PyResult<Option<u32>> {
                Ok(slf.as_super().dsv4()?.o_groups)
            }

            #[getter(_index_topk)]
            fn index_topk(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.index_topk)
            }

            #[getter(_architecture)]
            fn architecture(slf: PyRef<'_, Self>) -> PyResult<String> {
                Ok(slf.as_super().dsv4()?.architecture.clone())
            }

            #[getter(_gemm_quant_mode)]
            fn gemm_quant_mode<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                py_enum_member(
                    py,
                    "GEMMQuantMode",
                    &enum_token(&slf.as_super().dsv4()?.gemm_quant_mode),
                )
            }

            #[getter(_kvcache_quant_mode)]
            fn kvcache_quant_mode<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                py_enum_member(
                    py,
                    "KVCacheQuantMode",
                    &enum_token(&slf.as_super().dsv4()?.kv_cache_dtype),
                )
            }

            #[getter(_fmha_quant_mode)]
            fn fmha_quant_mode<'py>(
                slf: PyRef<'py, Self>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                py_enum_member(
                    py,
                    "FMHAQuantMode",
                    &enum_token(&slf.as_super().dsv4()?.fmha_quant_mode),
                )
            }

            #[getter(_cp_size)]
            fn cp_size(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.cp_size)
            }

            #[getter(_head_dim)]
            fn head_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.head_dim)
            }

            #[getter(_rope_head_dim)]
            fn rope_head_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.rope_head_dim)
            }

            #[getter(_q_lora_rank)]
            fn q_lora_rank(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.q_lora_rank)
            }

            #[getter(_o_lora_rank)]
            fn o_lora_rank(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.o_lora_rank)
            }

            #[getter(_index_n_heads)]
            fn index_n_heads(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.index_n_heads)
            }

            #[getter(_index_head_dim)]
            fn index_head_dim(slf: PyRef<'_, Self>) -> PyResult<u32> {
                Ok(slf.as_super().dsv4()?.index_head_dim)
            }

            #[setter(_cp_size)]
            fn set_cp_size(mut slf: PyRefMut<'_, Self>, value: u32) -> PyResult<()> {
                slf.as_super().dsv4_mut()?.cp_size = value;
                Ok(())
            }
        }
    };
}

dsv4_class!(
    PyContextDeepSeekV4AttentionModule,
    "ContextDeepSeekV4AttentionModule",
    Dsv4Context,
    true
);
dsv4_class!(
    PyGenerationDeepSeekV4AttentionModule,
    "GenerationDeepSeekV4AttentionModule",
    Dsv4Generation,
    false
);

// ---------------------------------------------------------------------------
// Composites
// ---------------------------------------------------------------------------

impl PyOperation {
    fn overlap(&self) -> PyResult<&crate::operators::OverlapOp> {
        match &self.inner {
            Op::Overlap(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not an OverlapOp")),
        }
    }
    fn fallback(&self) -> PyResult<&crate::operators::FallbackOp> {
        match &self.inner {
            Op::Fallback(o) => Ok(o),
            _ => Err(PyTypeError::new_err("not a FallbackOp")),
        }
    }
}

/// Extract the inner [`Op`] from any engine-backed op object (a Rust family
/// class or a Python shell subclassing one).
fn extract_child_op(obj: &Bound<'_, PyAny>) -> PyResult<Op> {
    let base: PyRef<'_, PyOperation> = obj.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "composite children must be engine-backed ops, got {}",
            obj.get_type()
        ))
    })?;
    Ok(base.inner.clone())
}

fn extract_child_ops(objs: &Bound<'_, PyAny>) -> PyResult<Vec<Op>> {
    let mut ops = Vec::new();
    for item in objs.try_iter()? {
        ops.push(extract_child_op(&item?)?);
    }
    Ok(ops)
}

fn wrap_ops<'py>(py: Python<'py>, ops: &[Op]) -> PyResult<Vec<Py<PyAny>>> {
    ops.iter().map(|op| wrap_op(py, op.clone())).collect()
}

/// Two op groups whose latency overlaps (max); weights sum.
#[pyclass(extends = PyOperation, subclass, name = "OverlapOp", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyOverlapOp;

#[pymethods]
impl PyOverlapOp {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "module";

    #[new]
    #[pyo3(signature = (name, group_a, group_b, *, seq_split=1))]
    fn new(
        name: String,
        group_a: &Bound<'_, PyAny>,
        group_b: &Bound<'_, PyAny>,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        // seq_split never crossed the wire for composites (children carry
        // their own); accepted for calling-shape compatibility.
        let _ = seq_split;
        let inner = Op::Overlap(crate::operators::OverlapOp {
            name,
            group_a: extract_child_ops(group_a)?,
            group_b: extract_child_ops(group_b)?,
        });
        Ok((PyOverlapOp, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().overlap()?;
        let args = (
            o.name.clone(),
            wrap_ops(py, &o.group_a)?,
            wrap_ops(py, &o.group_b)?,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_group_a)]
    fn group_a(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        wrap_ops(py, &slf.as_super().overlap()?.group_a)
    }

    #[getter(_group_b)]
    fn group_b(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        wrap_ops(py, &slf.as_super().overlap()?.group_b)
    }
}

/// Primary op with a fallback chain on perf-data misses.
#[pyclass(extends = PyOperation, subclass, name = "FallbackOp", module = "aiconfigurator_core._aiconfigurator_core")]
pub struct PyFallbackOp;

#[pymethods]
impl PyFallbackOp {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const _CP_AWARE: bool = true;

    #[classattr]
    #[allow(non_upper_case_globals)]
    const _ENGINE_QUERY_SHAPE: &'static str = "module";

    #[new]
    #[pyo3(signature = (name, primary, fallback, *, seq_split=1))]
    fn new(
        name: String,
        primary: &Bound<'_, PyAny>,
        fallback: &Bound<'_, PyAny>,
        seq_split: u32,
    ) -> PyResult<(Self, PyOperation)> {
        let _ = seq_split;
        let inner = Op::Fallback(crate::operators::FallbackOp {
            name,
            primary: Box::new(extract_child_op(primary)?),
            fallback: extract_child_ops(fallback)?,
        });
        Ok((PyFallbackOp, PyOperation { inner }))
    }

    fn __getnewargs_ex__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let o = slf.as_super().fallback()?;
        let args = (
            o.name.clone(),
            wrap_op(py, (*o.primary).clone())?,
            wrap_ops(py, &o.fallback)?,
        )
            .into_pyobject(py)?;
        Ok((args, PyDict::new(py)))
    }

    #[getter(_primary)]
    fn primary(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        wrap_op(py, (*slf.as_super().fallback()?.primary).clone())
    }

    #[getter(_fallback)]
    fn fallback(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        wrap_ops(py, &slf.as_super().fallback()?.fallback)
    }
}

/// Register every op class on the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOperation>()?;
    m.add_class::<PyGemm>()?;
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyElementWise>()?;
    m.add_class::<PyContextAttention>()?;
    m.add_class::<PyGenerationAttention>()?;
    m.add_class::<PyEncoderAttention>()?;
    m.add_class::<PyContextMLA>()?;
    m.add_class::<PyGenerationMLA>()?;
    m.add_class::<PyMLAModule>()?;
    m.add_class::<PyMLABmm>()?;
    m.add_class::<PyWideEPContextMLA>()?;
    m.add_class::<PyWideEPGenerationMLA>()?;
    m.add_class::<PyMoE>()?;
    m.add_class::<PyMoEDispatch>()?;
    m.add_class::<PyMoEAllToAll>()?;
    m.add_class::<PyModeledEPMoE>()?;
    m.add_class::<PyDeepSeekV4MegaMoEModule>()?;
    m.add_class::<PyDeepSeekV4MHCModule>()?;
    m.add_class::<PyContextDSAModule>()?;
    m.add_class::<PyGenerationDSAModule>()?;
    m.add_class::<PyContextMSAModule>()?;
    m.add_class::<PyGenerationMSAModule>()?;
    m.add_class::<PyContextDeepSeekV4AttentionModule>()?;
    m.add_class::<PyGenerationDeepSeekV4AttentionModule>()?;
    m.add_class::<PyMamba2Kernel>()?;
    m.add_class::<PyGDNKernel>()?;
    m.add_class::<PyKDAKernel>()?;
    m.add_class::<PyCustomAllReduce>()?;
    m.add_class::<PyNCCL>()?;
    m.add_class::<PyP2P>()?;
    m.add_class::<PyOverlapOp>()?;
    m.add_class::<PyFallbackOp>()?;
    Ok(())
}


/// Deserialize one externally-tagged opspec JSON document into an
/// engine-backed op object (used by the FPMForwardOp spec adapter).
pub(crate) fn op_from_spec_json(py: Python<'_>, spec_json: &str) -> PyResult<Py<PyAny>> {
    let op: Op = serde_json::from_str(spec_json)
        .map_err(|e| PyValueError::new_err(format!("invalid opspec json: {e}")))?;
    wrap_op(py, op)
}

/// Extract the inner ops from a Python sequence of engine-backed op objects.
pub(crate) fn ops_from_sequence(objs: &Bound<'_, PyAny>) -> PyResult<Vec<Op>> {
    extract_child_ops(objs)
}

/// Compile-time refusal of tombstone ops, recursively (the retired
/// `_to_opspec` raise): a graph carrying a RetiredDeepEp dispatch cannot be
/// expressed natively.
pub(crate) fn reject_retired_ops(ops: &[Op]) -> Result<(), String> {
    use crate::operators::DispatchFlavor;
    for op in ops {
        match op {
            Op::MoeDispatch(o) if o.flavor == DispatchFlavor::RetiredDeepEp => {
                return Err(format!(
                    "MoEDispatch(moe_backend='deepep_moe') has no native variant (retired with \
                     AIC-1601; large-EP comm is modeled by MoeAllToAll): op '{}'",
                    o.name
                ));
            }
            Op::Overlap(o) => {
                reject_retired_ops(&o.group_a)?;
                reject_retired_ops(&o.group_b)?;
            }
            Op::Fallback(o) => {
                reject_retired_ops(std::slice::from_ref(&o.primary))?;
                reject_retired_ops(&o.fallback)?;
            }
            Op::FpmForward(o) => reject_retired_ops(&o.sol_ops)?,
            _ => {}
        }
    }
    Ok(())
}
