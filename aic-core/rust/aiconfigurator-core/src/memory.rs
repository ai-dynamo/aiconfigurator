// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-cache memory estimation.
//!
//! [`estimate_kv_cache`] is a top-level crate function (NOT a method on
//! `AicEngine`): estimation runs once at startup, uses overlapping but not
//! identical inputs to `build_aic_engine`, and is a separate concern from
//! latency prediction. The Dynamo Mocker is the primary external consumer; it
//! calls this once and derives `num_gpu_blocks_per_rank` from
//! `total_kv_size_tokens`.
//!
//! ## Rust is a pure forwarder; the estimate is computed in Python
//!
//! This mirrors how `build_aic_engine` forwards to the Python `compile_engine`:
//! ALL of the work -- fraction + tolerance validation, HF-config parsing, the
//! AIC backend memory model, the OfFree/OfTotal budget math, the naive heuristic
//! fallback, AND the tolerance margin -- lives in
//! `aiconfigurator.sdk.memory.estimate_kv_cache`. The Rust side:
//!
//! 1. crosses into Python once (`with_gil → import → call estimate_kv_cache →
//!    extract dict`), forwarding `tolerance_fraction` through;
//! 2. rebuilds a [`KvCacheEstimate`] from that dict (including
//!    `tolerance_adjusted`), with no math of its own.
//!
//! The two budget formulas (TRT-LLM free-fraction vs vLLM/SGLang total-fraction),
//! the naive fallback, and the tolerance margin all live on the Python side; see
//! the docstring of `aiconfigurator.sdk.memory.estimate_kv_cache`. The
//! [`KvCacheMemoryFraction`] enum still encodes the backend↔fraction XOR so the
//! request shape is unambiguous; the variant is validated against
//! `engine.backend` in Python.

use serde::{Deserialize, Serialize};

use crate::{BackendKind, EngineConfig};

/// KV-cache memory request. `engine` reuses the modularised [`EngineConfig`];
/// the remaining fields describe the runtime sizing budget.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KvCacheEstimateRequest {
    pub engine: EngineConfig,
    pub max_num_tokens: u32,
    pub max_batch_size: u32,
    pub kv_cache_memory_fraction: KvCacheMemoryFraction,
    /// Override for unknown SKUs; when `Some`, it wins over the SystemSpec
    /// capacity reported by the native path.
    pub gpu_memory_capacity_bytes_override: Option<u64>,
    /// `None` = raw estimate only; `Some(0.05)` = 5% safety margin.
    pub tolerance_fraction: Option<f64>,
    pub options: KvCacheEstimateOptions,
}

/// Backend-tagged memory fraction. The variant encodes the XOR between
/// TRT-LLM's free-fraction and vLLM/SGLang's total-fraction semantics; the
/// Python `estimate_kv_cache` validates it against `engine.backend` and returns
/// an error (mapped to [`KvCacheEstimateError::IncompatibleMemoryFraction`]) if
/// mismatched.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum KvCacheMemoryFraction {
    /// Fraction of TOTAL GPU memory. Compatible with vLLM
    /// (`gpu_memory_utilization`) and SGLang (`mem_fraction_static`).
    OfTotal(f64),
    /// Fraction of FREE (post-non-KV) GPU memory. Compatible with TRT-LLM
    /// (`free_gpu_memory_fraction`).
    OfFree(f64),
}

impl KvCacheMemoryFraction {
    /// `(kind, value)` pair for the flat Python call. `kind` is the wire string
    /// the Python `estimate_kv_cache` expects (`"of_total"` / `"of_free"`).
    fn to_wire(self) -> (&'static str, f64) {
        match self {
            Self::OfTotal(f) => ("of_total", f),
            Self::OfFree(f) => ("of_free", f),
        }
    }
}

/// Default for [`KvCacheEstimateOptions::naive_kv_reservation`] so a JSON request
/// from an embedded caller (the Mocker) that omits this newer field still
/// deserializes (to the same 0.80 the Python side defaults to).
fn default_naive_kv_reservation() -> f64 {
    0.80
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct KvCacheEstimateOptions {
    pub allow_naive_fallback: bool,
    pub allow_hf_config_download: bool,
    /// Fraction of post-weight memory the naive fallback reserves for KV
    /// (default `0.80`). Ignored on the native path. Exposed so the Mocker can
    /// tune the crude fallback budget.
    #[serde(default = "default_naive_kv_reservation")]
    pub naive_kv_reservation: f64,
}

/// KV-cache memory estimate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct KvCacheEstimate {
    pub total_gpu_capacity_bytes: u64,
    pub total_kv_size_bytes: u64,
    pub kv_size_per_token_bytes: u64,
    pub total_kv_size_tokens: u64,
    pub source: EstimateSource,
    /// `Some` on the native path; `None` on the naive fallback.
    pub memory_breakdown: Option<MemoryBreakdown>,
    /// `Some` iff `tolerance_fraction` set; `None` for the raw estimate.
    pub tolerance_adjusted: Option<KvCacheEstimateAdjusted>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EstimateSource {
    /// AIC's full backend memory model used.
    Native,
    /// Post-weight reservation heuristic (`naive_kv_reservation`, default 80%).
    NaiveFallback,
}

/// Non-KV memory components, in bytes. Maps AIC's `_get_memory_usage` dict:
/// `weights → weights`, `activations → activations`, `others →
/// runtime_overhead`, `nccl → comm_overhead`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryBreakdown {
    pub weights_bytes: u64,
    pub activations_bytes: u64,
    pub runtime_overhead_bytes: u64,
    pub comm_overhead_bytes: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct KvCacheEstimateAdjusted {
    pub tolerance_fraction: f64,
    pub total_kv_size_bytes: u64,
    pub total_kv_size_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvCacheEstimateError {
    Unsupported {
        model: String,
        backend: BackendKind,
        gpu_sku: String,
        reason: String,
    },
    InsufficientModelMetadata {
        missing_fields: Vec<String>,
    },
    NoKvBudget {
        total_gpu_capacity_bytes: u64,
        non_kv_bytes: u64,
    },
    IncompatibleMemoryFraction {
        backend: BackendKind,
        variant_kind: &'static str,
    },
    BadConfig {
        field: String,
        reason: String,
    },
    HfConfigFetchFailed {
        hf_id: String,
        source: String,
    },
}

impl std::fmt::Display for KvCacheEstimateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported {
                model,
                backend,
                gpu_sku,
                reason,
            } => write!(
                f,
                "unsupported model/backend/GPU for KV-cache estimation: model={model}, \
                 backend={backend:?}, gpu_sku={gpu_sku}: {reason}"
            ),
            Self::InsufficientModelMetadata { missing_fields } => {
                write!(
                    f,
                    "insufficient model metadata; missing: {missing_fields:?}"
                )
            }
            Self::NoKvBudget {
                total_gpu_capacity_bytes,
                non_kv_bytes,
            } => write!(
                f,
                "no KV budget: non-KV memory ({non_kv_bytes} bytes) meets/exceeds the \
                 KV-cache memory limit (capacity={total_gpu_capacity_bytes} bytes)"
            ),
            Self::IncompatibleMemoryFraction {
                backend,
                variant_kind,
            } => write!(
                f,
                "incompatible memory fraction: backend {backend:?} does not accept \
                 KvCacheMemoryFraction::{variant_kind}"
            ),
            Self::BadConfig { field, reason } => {
                write!(f, "bad memory config field {field:?}: {reason}")
            }
            Self::HfConfigFetchFailed { hf_id, source } => {
                write!(f, "HF config fetch failed for {hf_id:?}: {source}")
            }
        }
    }
}

impl std::error::Error for KvCacheEstimateError {}

/// Estimate KV-cache memory (raw estimate + optional tolerance margin).
///
/// Pure forwarder: crosses into Python once to compute the COMPLETE estimate. The
/// Python `aiconfigurator.sdk.memory.estimate_kv_cache` does the backend↔fraction
/// and tolerance validation, the native AIC memory breakdown + budget math, the
/// naive heuristic fallback, AND the tolerance margin (`tolerance_adjusted`); the
/// Rust side rebuilds a [`KvCacheEstimate`] from the returned dict with no math
/// of its own.
///
/// Errors from the Python side (unsupported model/backend, incompatible memory
/// fraction, out-of-range tolerance, no KV budget, HF config fetch failure) cross
/// the PyO3 boundary as a `ValueError` whose message carries the failure detail
/// and are surfaced here as [`KvCacheEstimateError::Unsupported`] with that
/// message.
pub fn estimate_kv_cache(
    req: KvCacheEstimateRequest,
) -> Result<KvCacheEstimate, KvCacheEstimateError> {
    fetch_python_estimate(&req)
}

/// Cross into Python once to compute the complete estimate.
///
/// Mirrors the `build_aic_engine` → `compile_engine` forwarder shape: `with_gil
/// → import aiconfigurator.sdk.memory → call estimate_kv_cache(...) → extract
/// the returned dict`. `tolerance_fraction` is forwarded; the Python fn applies
/// the tolerance and returns `tolerance_adjusted` in the dict.
fn fetch_python_estimate(
    req: &KvCacheEstimateRequest,
) -> Result<KvCacheEstimate, KvCacheEstimateError> {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    let engine = &req.engine;
    let parallel = &engine.parallel;
    let quant = &engine.quantization;
    let nextn = engine
        .speculative
        .as_ref()
        .and_then(|s| s.nextn)
        .unwrap_or(0);
    let (fraction_kind, fraction_value) = req.kv_cache_memory_fraction.to_wire();

    Python::with_gil(|py| -> PyResult<KvCacheEstimate> {
        let engine_mod = py.import("aiconfigurator.sdk.memory")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("backend_version", engine.backend_version.as_deref())?;
        kwargs.set_item("max_num_tokens", req.max_num_tokens)?;
        kwargs.set_item("max_batch_size", req.max_batch_size)?;
        kwargs.set_item("memory_fraction_kind", fraction_kind)?;
        kwargs.set_item("memory_fraction_value", fraction_value)?;
        kwargs.set_item("tp_size", parallel.tp_size)?;
        kwargs.set_item("pp_size", parallel.pp_size)?;
        kwargs.set_item("attention_dp_size", parallel.attention_dp_size.unwrap_or(1))?;
        kwargs.set_item("moe_tp_size", parallel.moe_tp_size)?;
        kwargs.set_item("moe_ep_size", parallel.moe_ep_size)?;
        kwargs.set_item(
            "gemm_quant_mode",
            quant.weight_dtype.as_ref().map(dtype_str),
        )?;
        kwargs.set_item("moe_quant_mode", quant.moe_dtype.as_ref().map(dtype_str))?;
        kwargs.set_item(
            "kvcache_quant_mode",
            quant.kv_cache_dtype.as_ref().map(dtype_str),
        )?;
        kwargs.set_item(
            "fmha_quant_mode",
            quant.activation_dtype.as_ref().map(dtype_str),
        )?;
        // `comm_quant_mode` is intentionally NOT forwarded: the comm/NCCL
        // overhead comes from `system_spec` (`nccl_mem` / `other_mem`), not the
        // comm quant mode, so it does not affect the non-KV breakdown.
        kwargs.set_item("nextn", nextn)?;
        kwargs.set_item(
            "systems_path",
            engine.systems_path.as_deref().and_then(|p| p.to_str()),
        )?;
        kwargs.set_item(
            "gpu_memory_capacity_bytes_override",
            req.gpu_memory_capacity_bytes_override,
        )?;
        kwargs.set_item("tolerance_fraction", req.tolerance_fraction)?;
        kwargs.set_item("naive_kv_reservation", req.options.naive_kv_reservation)?;
        kwargs.set_item("allow_naive_fallback", req.options.allow_naive_fallback)?;
        kwargs.set_item(
            "allow_hf_config_download",
            req.options.allow_hf_config_download,
        )?;

        let out = engine_mod.call_method(
            "estimate_kv_cache",
            (
                engine.model_name.as_str(),
                engine.system_name.as_str(),
                engine.backend.as_str(),
            ),
            Some(&kwargs),
        )?;

        estimate_from_dict(&out)
    })
    // PyErr → KvCacheEstimateError inline (keeps the error enum self-contained).
    // The Python side raises a ValueError whose message already carries the
    // specific failure detail; surface it as Unsupported so the (deferred)
    // Mocker fallback decision still keys on the same variant it did before.
    .map_err(|e| KvCacheEstimateError::Unsupported {
        model: engine.model_name.clone(),
        backend: engine.backend.clone(),
        gpu_sku: engine.system_name.clone(),
        reason: format!("estimate_kv_cache: {e}"),
    })
}

/// Rebuild a [`KvCacheEstimate`] from the Python `estimate_kv_cache` dict.
/// The dict carries every struct field (`total_*`, `kv_size_per_token_bytes`,
/// `source`, `memory_breakdown`, and `tolerance_adjusted`); the Python fn applies
/// the tolerance, so `tolerance_adjusted` is a nested dict iff
/// `tolerance_fraction` was set, `None` otherwise.
fn estimate_from_dict(
    out: &pyo3::Bound<'_, pyo3::types::PyAny>,
) -> pyo3::PyResult<KvCacheEstimate> {
    use pyo3::exceptions::PyValueError;
    use pyo3::types::PyAnyMethods;

    let u64_at = |k: &str| -> pyo3::PyResult<u64> { out.get_item(k)?.extract::<u64>() };

    let source = match out.get_item("source")?.extract::<String>()?.as_str() {
        "native" => EstimateSource::Native,
        "naive_fallback" => EstimateSource::NaiveFallback,
        other => {
            return Err(PyValueError::new_err(format!(
                "estimate_kv_cache returned unknown source {other:?}"
            )))
        }
    };

    let breakdown_item = out.get_item("memory_breakdown")?;
    let memory_breakdown = if breakdown_item.is_none() {
        None
    } else {
        let get = |k: &str| -> pyo3::PyResult<u64> { breakdown_item.get_item(k)?.extract::<u64>() };
        Some(MemoryBreakdown {
            weights_bytes: get("weights_bytes")?,
            activations_bytes: get("activations_bytes")?,
            runtime_overhead_bytes: get("runtime_overhead_bytes")?,
            comm_overhead_bytes: get("comm_overhead_bytes")?,
        })
    };

    // The Python fn applies the tolerance, so `tolerance_adjusted` is a nested
    // dict iff `tolerance_fraction` was set (keys mirror the Python emit side:
    // `tolerance_fraction` / `total_kv_size_bytes` / `total_kv_size_tokens`).
    let adjusted_item = out.get_item("tolerance_adjusted")?;
    let tolerance_adjusted = if adjusted_item.is_none() {
        None
    } else {
        Some(KvCacheEstimateAdjusted {
            tolerance_fraction: adjusted_item.get_item("tolerance_fraction")?.extract::<f64>()?,
            total_kv_size_bytes: adjusted_item.get_item("total_kv_size_bytes")?.extract::<u64>()?,
            total_kv_size_tokens: adjusted_item.get_item("total_kv_size_tokens")?.extract::<u64>()?,
        })
    };

    Ok(KvCacheEstimate {
        total_gpu_capacity_bytes: u64_at("total_gpu_capacity_bytes")?,
        total_kv_size_bytes: u64_at("total_kv_size_bytes")?,
        kv_size_per_token_bytes: u64_at("kv_size_per_token_bytes")?,
        total_kv_size_tokens: u64_at("total_kv_size_tokens")?,
        source,
        memory_breakdown,
        tolerance_adjusted,
    })
}

/// Map a [`crate::DataType`] to the snake_case quant-mode string the Python
/// `_build_model_config` accepts (the serde `rename` already produces these).
fn dtype_str(dt: &crate::DataType) -> &'static str {
    use crate::DataType::*;
    match dt {
        Bfloat16 => "bfloat16",
        Float16 => "float16",
        Fp8 => "fp8",
        Fp8Static => "fp8_static",
        Fp8Block => "fp8_block",
        Nvfp4 => "nvfp4",
        Int8 => "int8",
        Int4 => "int4",
        W4afp8 => "w4afp8",
        W4a16Mxfp4 => "w4a16_mxfp4",
        W4a8Mxfp4Mxfp8 => "w4a8_mxfp4_mxfp8",
        W4a8Mxfp4Mxfp8Trtllm => "w4a8_mxfp4_mxfp8_trtllm",
        W4a16Mxfp4Cutlass => "w4a16_mxfp4_cutlass",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance validation + application and the native/naive budget math now
    // live entirely in Python (`aiconfigurator.sdk.memory.estimate_kv_cache`),
    // exercised by `tests/unit/sdk/test_memory_estimation.py` and the integration
    // parity test. The Rust side is a pure forwarder; the only pure-Rust unit
    // left here is the memory-fraction wire mapping. The dict round-trip
    // (`fetch_python_estimate` → `estimate_from_dict`, including the
    // `tolerance_adjusted` parse) is covered end-to-end by the Mocker consumer.

    /// The memory fraction must cross to Python as the wire `(kind, value)` pair
    /// the Python `estimate_kv_cache` expects.
    #[test]
    fn memory_fraction_to_wire() {
        assert_eq!(
            KvCacheMemoryFraction::OfFree(0.9).to_wire(),
            ("of_free", 0.9)
        );
        assert_eq!(
            KvCacheMemoryFraction::OfTotal(0.85).to_wire(),
            ("of_total", 0.85)
        );
    }
}
