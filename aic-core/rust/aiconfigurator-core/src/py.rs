// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for the compiled-engine core.
//!
//! Exposes two call directions over a single PyO3 dependency:
//!
//! * **Python → Rust (hot path).** [`AicEngine`] is a `#[pyclass]` wrapping the
//!   [`Engine`]. Its `#[pymethods]` (`from_spec`, `run_static`,
//!   `predict_prefill_latency`, `predict_decode_latency`, `mixed_step_latency`,
//!   `decode_step_latency`) are the surface the Python sweep / Mocker bridge
//!   calls per point. The agg sweep is orchestrated in Python; there is no
//!   Rust `run_agg`. Each method
//!   releases the GIL around the Rust compute via [`Python::allow_threads`],
//!   so the Rust compute runs without holding the GIL.
//! * **Rust → Python → Rust (embedded path).** [`AicEngineBuilder`] is the
//!   preferred Rust entry point. The flat [`build_aic_engine`] function remains
//!   as a source-compatibility adapter for callers such as the Dynamo Mocker.
//!   Both cross into Python once to run
//!   `aiconfigurator_core.sdk.engine.compile_engine`, then build an [`Engine`]
//!   from the returned bincode bytes. After that the `predict_*` hot path is
//!   pure Rust with no GIL.
//!
//! Two error conversions live inline here (NOT in `common/error.rs`, which must
//! stay free of the pyo3 dependency):
//! * `AicError → PyErr` (`aic_to_py`) for the `#[pymethods]` boundary.
//! * `PyErr → AicError` (inline in [`build_aic_engine`]) for the embedded path.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::common::error::AicError;
use crate::engine::runtime::{
    Engine, RuntimeConfig, StaticMode, StaticResult, DEFAULT_STATIC_STRIDE,
};
use crate::{BackendKind, DataType, EngineConfig, ENGINE_CONFIG_SCHEMA_VERSION};

/// Trivial smoke export: returns the engine-config schema version so callers
/// can confirm the extension built and imported correctly.
#[pyfunction]
fn _build_smoke() -> u32 {
    ENGINE_CONFIG_SCHEMA_VERSION
}

/// Convert a crate error into a Python exception at the `#[pymethods]` boundary.
/// Inline (not a `From` impl in `error.rs`) so the error module stays pyo3-free.
fn aic_to_py(e: AicError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Map the `mode` string (Python's `_run_static_breakdown` convention) to the
/// Rust [`StaticMode`]. `"static" → Both`, `"static_ctx" → Context`,
/// `"static_gen" → Generation`; anything else is a `ValueError`.
fn parse_mode(mode: &str) -> PyResult<StaticMode> {
    match mode {
        "static" => Ok(StaticMode::Both),
        "static_ctx" => Ok(StaticMode::Context),
        "static_gen" => Ok(StaticMode::Generation),
        other => Err(PyValueError::new_err(format!(
            "invalid mode {other:?}; expected one of \"static\", \"static_ctx\", \"static_gen\""
        ))),
    }
}

/// Resolve the bundled `systems/` directory for [`AicEngine::from_spec`].
///
/// Mirrors the systems-root half of `DataRoots::discover` but does NOT require
/// a `model_configs` root: `compile_engine` (the only thing that needs model
/// configs) runs in Python, so the Rust side only loads the perf database.
/// Precedence: explicit `systems_path` arg → `AICONFIGURATOR_SYSTEMS_PATH` env
/// → the installed core wheel's SDK resource path → repo-relative
/// `src/aiconfigurator_core/systems`.
fn resolve_systems_root(systems_path: Option<&str>) -> PyResult<PathBuf> {
    if let Some(p) = systems_path {
        return Ok(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("AICONFIGURATOR_SYSTEMS_PATH") {
        return Ok(PathBuf::from(p));
    }
    let installed_root = Python::with_gil(|py| -> PyResult<Option<PathBuf>> {
        let Ok(perf_database) = py.import("aiconfigurator_core.sdk.perf_database") else {
            return Ok(None);
        };
        let paths: Vec<String> = perf_database.call_method0("get_systems_paths")?.extract()?;
        Ok(paths.into_iter().next().map(PathBuf::from))
    })?;
    if let Some(p) = installed_root {
        return Ok(p);
    }
    crate::repo_relative("src/aiconfigurator_core/systems").ok_or_else(|| {
        PyValueError::new_err(
            "could not resolve systems path: pass systems_path, set \
             AICONFIGURATOR_SYSTEMS_PATH, install aiconfigurator-core, or run \
             from an AIC checkout",
        )
    })
}

/// PyO3 wrapper around the [`Engine`]: a compiled engine the Python sweep /
/// Mocker bridge drives per point.
///
/// Holds `Arc<Engine>` (the `Engine` already owns its `Arc<PerfDatabase>`), so
/// the handle is cheap to clone and the GIL can be released around every
/// compute call. The full `RuntimeConfig` is reconstructed from positional
/// args on each call rather than stored — it varies per point.
#[pyclass(name = "AicEngine")]
pub struct AicEngine {
    inner: Arc<Engine>,
}

impl AicEngine {
    /// Internal constructor shared by [`AicEngine::from_spec`] and
    /// [`build_aic_engine`].
    fn new(engine: Engine) -> Self {
        AicEngine {
            inner: Arc::new(engine),
        }
    }

    /// Pure-Rust prefill-step latency (ms). No PyO3 `py` token: this is the
    /// GIL-free Mocker hot path for Rust callers in OTHER crates (the Dynamo
    /// Mocker, `tests/embedded_round_trip.rs`), which cannot reach the private
    /// `inner` [`Engine`] directly. Delegates to
    /// [`Engine::predict_prefill_latency`]. The `#[pymethods]`
    /// `predict_prefill_latency` wraps this in `allow_threads`; this inherent
    /// form is the same compute with no GIL ever acquired.
    pub fn prefill_latency_ms(&self, bs: u32, isl: u32, prefix: u32) -> Result<f64, AicError> {
        self.inner.predict_prefill_latency(bs, isl, prefix)
    }

    /// Pure-Rust decode-step latency (ms). No PyO3 `py` token. Delegates to
    /// [`Engine::predict_decode_latency`].
    pub fn decode_latency_ms(&self, bs: u32, isl: u32, osl: u32) -> Result<f64, AicError> {
        self.inner.predict_decode_latency(bs, isl, osl)
    }
}

#[pymethods]
impl AicEngine {
    /// Build an `AicEngine` from bincoded [`EngineSpec`] bytes (the output of
    /// Python's `compile_engine`). `systems_path` overrides the bundled
    /// `systems/` directory; `None` resolves it via env / repo-relative
    /// fallback (see [`resolve_systems_root`]).
    ///
    /// Named constructor → `#[staticmethod]`, so Python calls
    /// `AicEngine.from_spec(bytes, systems_path)`.
    #[staticmethod]
    #[pyo3(signature = (bytes, systems_path=None))]
    fn from_spec(bytes: &[u8], systems_path: Option<&str>) -> PyResult<AicEngine> {
        let systems_root = resolve_systems_root(systems_path)?;
        // Engine::from_spec_bytes does from_bincode + PerfDatabase::load +
        // Engine::build. No GIL is held inside the Rust core; releasing it
        // here lets concurrent Python threads proceed during the DB load.
        let engine = Python::with_gil(|py| {
            py.allow_threads(|| Engine::from_spec_bytes(bytes, &systems_root))
        })
        .map_err(aic_to_py)?;
        Ok(AicEngine::new(engine))
    }

    /// Python `run_static` / `run_static_latency_only` restricted to the
    /// latency breakdown. Returns `(context_ms, generation_ms, total_ms)`.
    ///
    /// Positional args mirror the Python `BaseBackend.run_static` runtime shape,
    /// in this exact order: `batch_size, beam_width, isl, osl, prefix,
    /// seq_imbalance_correction_scale, gen_seq_imbalance_correction_scale`
    /// (all seven required), then `mode` (`"static"|"static_ctx"|"static_gen"`,
    /// default `"static"`) and `stride` (default `DEFAULT_STATIC_STRIDE`).
    #[pyo3(signature = (
        batch_size,
        beam_width,
        isl,
        osl,
        prefix,
        seq_imbalance_correction_scale,
        gen_seq_imbalance_correction_scale,
        mode="static",
        stride=DEFAULT_STATIC_STRIDE,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn run_static(
        &self,
        py: Python<'_>,
        batch_size: u32,
        beam_width: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
        seq_imbalance_correction_scale: f64,
        gen_seq_imbalance_correction_scale: f64,
        mode: &str,
        stride: u32,
    ) -> PyResult<(f64, f64, f64)> {
        // Arg extraction / mode parsing happen BEFORE allow_threads.
        let rt = RuntimeConfig {
            batch_size,
            beam_width,
            isl,
            osl,
            prefix,
            seq_imbalance_correction_scale,
            gen_seq_imbalance_correction_scale,
        };
        let mode = parse_mode(mode)?;
        // Rust compute runs with the GIL released.
        let result: StaticResult = py
            .allow_threads(|| self.inner.run_static(&rt, mode, stride))
            .map_err(aic_to_py)?;
        Ok((result.context_ms, result.generation_ms, result.total_ms))
    }

    /// Mocker H1: prefill-step latency in ms. Thin shim over `run_static` with
    /// `mode=Context` (osl is irrelevant for the context phase, so it is fixed
    /// at 1). Returns the total ms (== context_ms in this mode).
    #[pyo3(signature = (bs, isl, prefix=0))]
    fn predict_prefill_latency(
        &self,
        py: Python<'_>,
        bs: u32,
        isl: u32,
        prefix: u32,
    ) -> PyResult<f64> {
        py.allow_threads(|| self.inner.predict_prefill_latency(bs, isl, prefix))
            .map_err(aic_to_py)
    }

    /// Mocker H2: decode-step latency in ms. Thin shim over `run_static` with
    /// `mode=Generation`. Mocker passes `osl=2` (one decode step at
    /// `s = isl + 1`). Returns the total ms (== generation_ms in this mode).
    #[pyo3(signature = (bs, isl, osl=2))]
    fn predict_decode_latency(&self, py: Python<'_>, bs: u32, isl: u32, osl: u32) -> PyResult<f64> {
        py.allow_threads(|| self.inner.predict_decode_latency(bs, isl, osl))
            .map_err(aic_to_py)
    }

    /// One mixed (chunked-prefill + decode) engine-step latency in ms. Binds
    /// [`Engine::mixed_step_latency`]; the Python agg orchestration
    /// (`base_backend._get_mix_step_latency`) calls this per mix step. Mirrors
    /// the live FPM bridge `estimate_mixed_step_latency_with_rust`.
    #[pyo3(signature = (ctx_tokens, gen_tokens, isl, osl, prefix=0))]
    fn mixed_step_latency(
        &self,
        py: Python<'_>,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> PyResult<f64> {
        py.allow_threads(|| {
            self.inner
                .mixed_step_latency(ctx_tokens, gen_tokens, isl, osl, prefix)
        })
        .map_err(aic_to_py)
    }

    /// Component latencies for one mixed engine step. Returns
    /// ``(total, shared_non_attention, context_attention, decode_attention)``.
    #[pyo3(signature = (ctx_tokens, gen_tokens, isl, osl, prefix=0))]
    fn mixed_step_breakdown(
        &self,
        py: Python<'_>,
        ctx_tokens: u32,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
        prefix: u32,
    ) -> PyResult<(f64, f64, f64, f64)> {
        py.allow_threads(|| {
            self.inner
                .mixed_step_breakdown(ctx_tokens, gen_tokens, isl, osl, prefix)
        })
        .map(|parts| (parts[0], parts[1], parts[2], parts[3]))
        .map_err(aic_to_py)
    }

    /// One generation-only engine-step latency in ms. Binds
    /// [`Engine::decode_step_latency`]; the Python agg orchestration
    /// (`base_backend._get_genonly_step_latency`) calls this per genonly step.
    /// Mirrors the live FPM bridge `estimate_decode_step_latency_with_rust`.
    #[pyo3(signature = (gen_tokens, isl, osl))]
    fn decode_step_latency(
        &self,
        py: Python<'_>,
        gen_tokens: u32,
        isl: u32,
        osl: u32,
    ) -> PyResult<f64> {
        py.allow_threads(|| self.inner.decode_step_latency(gen_tokens, isl, osl))
            .map_err(aic_to_py)
    }
}

/// Convert a JSON-encoded [`EngineSpec`] into bincode bytes (Python → Rust
/// op-transfer). Python's `compile_engine` builds the `EngineSpec` as a JSON
/// string (externally-tagged `Op` variants + `EngineConfig`) — JSON is the
/// debuggable wire and the only format Python can produce — and calls this to
/// get the bincode bytes that `AicEngine.from_spec` / `build_aic_engine`
/// consume. `serde_json` round-trips `EngineConfig`'s `#[serde(flatten)]`
/// cleanly (only bincode rejected it; that is exactly why `to_bincode`
/// re-encodes `engine` as JSON inside the bincode payload).
#[pyfunction]
fn engine_spec_bincode_from_json(spec_json: &str) -> PyResult<Vec<u8>> {
    let spec: crate::engine::spec::EngineSpec = serde_json::from_str(spec_json)
        .map_err(|e| PyValueError::new_err(format!("engine spec JSON decode: {e}")))?;
    spec.to_bincode().map_err(aic_to_py)
}

/// Internal request shared by every Rust -> Python -> Rust construction path.
///
/// The builder and the flat compatibility function both normalize into this
/// representation before the one-time Python compile. Keeping this type private
/// lets the public builder evolve without creating a second public config API.
#[derive(Clone, Debug)]
struct EngineBuildRequest {
    model_path: String,
    system: String,
    backend: String,
    backend_version: Option<String>,
    tp_size: u32,
    pp_size: u32,
    attention_dp_size: u32,
    moe_tp_size: Option<u32>,
    moe_ep_size: Option<u32>,
    gemm_quant_mode: Option<String>,
    moe_quant_mode: Option<String>,
    kvcache_quant_mode: Option<String>,
    fmha_quant_mode: Option<String>,
    comm_quant_mode: Option<String>,
    nextn: u32,
    kv_block_size: Option<u32>,
    systems_path: Option<String>,
}

/// Ergonomic builder for the Rust -> Python -> Rust compiled-engine entry point.
///
/// Only the model, system, and backend are required. Parallelism defaults to
/// one, speculative decoding defaults to disabled, and all other options defer
/// to Python's `compile_engine` defaults. New callers should use this builder.
/// [`build_aic_engine`] remains available as a source-compatibility adapter for
/// existing callers through 0.10 and is planned for removal in version 0.11.0.
#[derive(Clone, Debug)]
pub struct AicEngineBuilder {
    request: EngineBuildRequest,
}

impl AicEngineBuilder {
    /// Start an engine build for a model, target system, and backend.
    pub fn new(
        model_path: impl Into<String>,
        system: impl Into<String>,
        backend: BackendKind,
    ) -> Self {
        Self {
            request: EngineBuildRequest {
                model_path: model_path.into(),
                system: system.into(),
                backend: backend.as_str().to_owned(),
                backend_version: None,
                tp_size: 1,
                pp_size: 1,
                attention_dp_size: 1,
                moe_tp_size: None,
                moe_ep_size: None,
                gemm_quant_mode: None,
                moe_quant_mode: None,
                kvcache_quant_mode: None,
                fmha_quant_mode: None,
                comm_quant_mode: None,
                nextn: 0,
                kv_block_size: None,
                systems_path: None,
            },
        }
    }

    /// Select a specific backend version.
    pub fn backend_version(mut self, value: impl Into<String>) -> Self {
        self.request.backend_version = Some(value.into());
        self
    }

    /// Set tensor parallelism.
    pub fn tp_size(mut self, value: u32) -> Self {
        self.request.tp_size = value;
        self
    }

    /// Set pipeline parallelism.
    pub fn pp_size(mut self, value: u32) -> Self {
        self.request.pp_size = value;
        self
    }

    /// Set attention data parallelism.
    pub fn attention_dp_size(mut self, value: u32) -> Self {
        self.request.attention_dp_size = value;
        self
    }

    /// Set optional MoE tensor and expert parallelism.
    pub fn moe_parallelism(mut self, tp_size: Option<u32>, ep_size: Option<u32>) -> Self {
        self.request.moe_tp_size = tp_size;
        self.request.moe_ep_size = ep_size;
        self
    }

    /// Override the GEMM quantization mode.
    pub fn gemm_quant_mode(mut self, value: impl Into<String>) -> Self {
        self.request.gemm_quant_mode = Some(value.into());
        self
    }

    /// Override the MoE quantization mode.
    pub fn moe_quant_mode(mut self, value: impl Into<String>) -> Self {
        self.request.moe_quant_mode = Some(value.into());
        self
    }

    /// Override the KV-cache quantization mode.
    pub fn kvcache_quant_mode(mut self, value: impl Into<String>) -> Self {
        self.request.kvcache_quant_mode = Some(value.into());
        self
    }

    /// Override the FMHA quantization mode.
    pub fn fmha_quant_mode(mut self, value: impl Into<String>) -> Self {
        self.request.fmha_quant_mode = Some(value.into());
        self
    }

    /// Override the communication quantization mode.
    pub fn comm_quant_mode(mut self, value: impl Into<String>) -> Self {
        self.request.comm_quant_mode = Some(value.into());
        self
    }

    /// Configure speculative decoding.
    pub fn speculative_decoding(mut self, nextn: u32) -> Self {
        self.request.nextn = nextn;
        self
    }

    /// Override the KV-cache block size.
    pub fn kv_block_size(mut self, value: u32) -> Self {
        self.request.kv_block_size = Some(value);
        self
    }

    /// Override the bundled systems-data root.
    pub fn systems_path(mut self, value: impl Into<String>) -> Self {
        self.request.systems_path = Some(value.into());
        self
    }

    /// Compile the Python engine specification once and return a Rust handle.
    pub fn build(self) -> Result<AicEngine, AicError> {
        build_aic_engine_impl(self.request)
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn builder_defaults_match_compile_engine_defaults() {
        let builder = AicEngineBuilder::new("model", "system", BackendKind::Vllm);
        assert_eq!(builder.request.tp_size, 1);
        assert_eq!(builder.request.pp_size, 1);
        assert_eq!(builder.request.attention_dp_size, 1);
        assert_eq!(builder.request.nextn, 0);
        assert!(builder.request.backend_version.is_none());
        assert!(builder.request.moe_tp_size.is_none());
        assert!(builder.request.moe_ep_size.is_none());
        assert!(builder.request.kv_block_size.is_none());
    }

    #[test]
    fn builder_retains_explicit_options() {
        let builder = AicEngineBuilder::new("model", "system", BackendKind::Sglang)
            .backend_version("0.5.9")
            .tp_size(8)
            .pp_size(2)
            .attention_dp_size(4)
            .moe_parallelism(Some(1), Some(8))
            .speculative_decoding(2)
            .kv_block_size(16)
            .systems_path("/tmp/systems");
        assert_eq!(builder.request.backend, "sglang");
        assert_eq!(builder.request.backend_version.as_deref(), Some("0.5.9"));
        assert_eq!((builder.request.tp_size, builder.request.pp_size), (8, 2));
        assert_eq!(builder.request.attention_dp_size, 4);
        assert_eq!(
            (builder.request.moe_tp_size, builder.request.moe_ep_size),
            (Some(1), Some(8))
        );
        assert_eq!(builder.request.nextn, 2);
        assert_eq!(builder.request.kv_block_size, Some(16));
        assert_eq!(
            builder.request.systems_path.as_deref(),
            Some("/tmp/systems")
        );
    }
}

/// Compatibility Rust → Python → Rust embedded build entry point.
///
/// A plain `pub` Rust fn (NOT a `#[pyfunction]`): Rust callers (e.g. the Dynamo
/// Mocker) call it with flat scalars. It crosses into Python exactly once to
/// run `aiconfigurator_core.sdk.engine.compile_engine`, which walks the model's
/// op lists and returns bincoded [`crate::engine::spec::EngineSpec`] bytes. It
/// then builds the [`Engine`] from those bytes (via
/// [`Engine::from_spec_bytes`], which does `from_bincode` +
/// `PerfDatabase::load` + `Engine::build`). The call shape is
/// `with_gil → import → call_method1("compile_engine", ...) →
/// extract::<Vec<u8>>() → build`.
///
/// The flat arg list matches the `compile_engine` signature. `systems_path` is
/// the Rust-side perf-DB root (it is also forwarded to `compile_engine` so the
/// two stay aligned).
///
/// The full Rust → Python → Rust round-trip is validated end-to-end by
/// `tests/embedded_round_trip.rs`.
///
/// # Compatibility
///
/// This flat function remains source-compatible through the 0.10 release for
/// existing consumers. New code should use [`AicEngineBuilder`]. The flat
/// function is planned for removal in version 0.11.0.
// `pub` and re-exported from `lib.rs` for embedded callers (the Dynamo Mocker,
// `tests/embedded_round_trip.rs`).
#[allow(clippy::too_many_arguments)]
pub fn build_aic_engine(
    model_path: &str,
    system: &str,
    backend: &str,
    backend_version: Option<&str>,
    tp_size: u32,
    pp_size: u32,
    attention_dp_size: u32,
    moe_tp_size: Option<u32>,
    moe_ep_size: Option<u32>,
    gemm_quant_mode: Option<&str>,
    moe_quant_mode: Option<&str>,
    kvcache_quant_mode: Option<&str>,
    fmha_quant_mode: Option<&str>,
    comm_quant_mode: Option<&str>,
    nextn: u32,
    kv_block_size: Option<u32>,
    systems_path: Option<&str>,
) -> Result<AicEngine, AicError> {
    build_aic_engine_impl(EngineBuildRequest {
        model_path: model_path.to_owned(),
        system: system.to_owned(),
        backend: backend.to_owned(),
        backend_version: backend_version.map(str::to_owned),
        tp_size,
        pp_size,
        attention_dp_size,
        moe_tp_size,
        moe_ep_size,
        gemm_quant_mode: gemm_quant_mode.map(str::to_owned),
        moe_quant_mode: moe_quant_mode.map(str::to_owned),
        kvcache_quant_mode: kvcache_quant_mode.map(str::to_owned),
        fmha_quant_mode: fmha_quant_mode.map(str::to_owned),
        comm_quant_mode: comm_quant_mode.map(str::to_owned),
        nextn,
        kv_block_size,
        systems_path: systems_path.map(str::to_owned),
    })
}

/// Construct the public handle from the one canonical build request.
fn build_aic_engine_impl(request: EngineBuildRequest) -> Result<AicEngine, AicError> {
    let engine = compile_engine_from_request(request)?;
    Ok(AicEngine::new(engine))
}

/// Shared `Rust → Python → Rust` compile body: cross into Python once to run
/// `compile_engine` from one normalized request, then build an [`Engine`] from
/// the returned bincode bytes (`from_bincode` + `PerfDatabase::load` +
/// `Engine::build`).
/// The builder, compatibility wrapper, and [`compile_engine_to_engine`] all use
/// this function, so Python argument names and defaults cannot drift.
fn compile_engine_from_request(request: EngineBuildRequest) -> Result<Engine, AicError> {
    let systems_root = resolve_systems_root(request.systems_path.as_deref())
        .map_err(|e| AicError::DataRoot(format!("resolve systems path: {e}")))?;
    let systems_root_str = systems_root.to_str().ok_or_else(|| {
        AicError::DataRoot(format!(
            "systems path is not valid UTF-8: {}",
            systems_root.display()
        ))
    })?;
    let spec_bytes: Vec<u8> = Python::with_gil(|py| -> PyResult<Vec<u8>> {
        let engine_mod = py.import("aiconfigurator_core.sdk.engine")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("backend_version", request.backend_version.as_deref())?;
        kwargs.set_item("tp_size", request.tp_size)?;
        kwargs.set_item("pp_size", request.pp_size)?;
        kwargs.set_item("attention_dp_size", request.attention_dp_size)?;
        kwargs.set_item("moe_tp_size", request.moe_tp_size)?;
        kwargs.set_item("moe_ep_size", request.moe_ep_size)?;
        kwargs.set_item("gemm_quant_mode", request.gemm_quant_mode.as_deref())?;
        kwargs.set_item("moe_quant_mode", request.moe_quant_mode.as_deref())?;
        kwargs.set_item("kvcache_quant_mode", request.kvcache_quant_mode.as_deref())?;
        kwargs.set_item("fmha_quant_mode", request.fmha_quant_mode.as_deref())?;
        kwargs.set_item("comm_quant_mode", request.comm_quant_mode.as_deref())?;
        kwargs.set_item("nextn", request.nextn)?;
        kwargs.set_item("kv_block_size", request.kv_block_size)?;
        kwargs.set_item("systems_path", systems_root_str)?;
        engine_mod
            .call_method(
                "compile_engine",
                (
                    request.model_path.as_str(),
                    request.system.as_str(),
                    request.backend.as_str(),
                ),
                Some(&kwargs),
            )?
            .extract::<Vec<u8>>()
    })
    // PyErr → AicError inline (keeps error.rs pyo3-free). A `compile_engine`
    // failure means the model cannot be built natively, so it maps to
    // `UnsupportedModel` — the variant `best_available` treats as
    // fallback-safe. Hard caller/config errors use `InvalidEngineConfig` (which
    // is NOT fallback-safe) so they surface instead of silently degrading.
    .map_err(|e| AicError::UnsupportedModel(format!("compile_engine: {e}")))?;

    Engine::from_spec_bytes(&spec_bytes, systems_root.as_path() as &Path)
}

/// Build a compiled [`Engine`] from a modular [`EngineConfig`] (the
/// `fpm::ForwardPassPerfModel::from_native` entry point). Maps the
/// config's nested fields onto the shared [`EngineBuildRequest`].
///
/// Quantization: each `DataType` is mapped to the matching backend quant-mode
/// enum NAME per target field (see [`gemm_quant_name`] etc.). DataTypes with no
/// valid name in a given target enum (e.g. `float16` for GEMM) map to `None`,
/// so `compile_engine` auto-infers quant from the model's HF config — which is
/// the correct behavior for quantized HF model IDs (quant is override-when-
/// present in `_apply_model_quant_defaults`). `pub(crate)` and called from the
/// private `fpm` module via `crate::py::compile_engine_to_engine`.
pub(crate) fn compile_engine_to_engine(
    config: &EngineConfig,
    systems_path: Option<&str>,
) -> Result<Engine, AicError> {
    let nextn = config
        .speculative
        .as_ref()
        .and_then(|s| s.nextn)
        .unwrap_or(0);
    compile_engine_from_request(EngineBuildRequest {
        model_path: config.model_name.clone(),
        system: config.system_name.clone(),
        backend: config.backend.as_str().to_owned(),
        backend_version: config.backend_version.clone(),
        tp_size: config.parallel.tp_size,
        pp_size: config.parallel.pp_size,
        attention_dp_size: config.parallel.attention_dp_size.unwrap_or(1),
        moe_tp_size: config.parallel.moe_tp_size,
        moe_ep_size: config.parallel.moe_ep_size,
        gemm_quant_mode: gemm_quant_name(config.quantization.weight_dtype.as_ref())
            .map(str::to_owned),
        moe_quant_mode: moe_quant_name(config.quantization.moe_dtype.as_ref()).map(str::to_owned),
        kvcache_quant_mode: kvcache_quant_name(config.quantization.kv_cache_dtype.as_ref())
            .map(str::to_owned),
        fmha_quant_mode: fmha_quant_name(config.quantization.activation_dtype.as_ref())
            .map(str::to_owned),
        // Comm quant is not carried on EngineConfig; let Python default it.
        comm_quant_mode: None,
        nextn,
        kv_block_size: config.kv_block_size,
        systems_path: systems_path.map(str::to_owned),
    })
}

/// `DataType` → `GEMMQuantMode` enum name. `None` (auto-infer) for DataTypes
/// with no GEMM equivalent. Identity for bf16/fp8/fp8_static/fp8_block/nvfp4;
/// `int8`→`int8_wo`, `int4`→`int4_wo`.
fn gemm_quant_name(dtype: Option<&DataType>) -> Option<&'static str> {
    match dtype? {
        DataType::Bfloat16 => Some("bfloat16"),
        DataType::Fp8 => Some("fp8"),
        DataType::Fp8Static => Some("fp8_static"),
        DataType::Fp8Block => Some("fp8_block"),
        DataType::Nvfp4 => Some("nvfp4"),
        DataType::Int8 => Some("int8_wo"),
        DataType::Int4 => Some("int4_wo"),
        // float16, w4afp8, w4a16_mxfp4, w4a8_mxfp4_mxfp8 have no GEMM enum name.
        _ => None,
    }
}

/// `DataType` → `MoEQuantMode` enum name. `None` for unmapped.
fn moe_quant_name(dtype: Option<&DataType>) -> Option<&'static str> {
    match dtype? {
        DataType::Bfloat16 => Some("bfloat16"),
        DataType::Fp8 => Some("fp8"),
        DataType::Fp8Block => Some("fp8_block"),
        DataType::Nvfp4 => Some("nvfp4"),
        DataType::Int4 => Some("int4_wo"),
        DataType::W4afp8 => Some("w4afp8"),
        DataType::W4a16Mxfp4 => Some("w4a16_mxfp4"),
        DataType::W4a8Mxfp4Mxfp8 => Some("w4a8_mxfp4_mxfp8"),
        _ => None,
    }
}

/// `DataType` → `FMHAQuantMode` enum name. Only bf16/fp8/fp8_block exist.
fn fmha_quant_name(dtype: Option<&DataType>) -> Option<&'static str> {
    match dtype? {
        DataType::Bfloat16 => Some("bfloat16"),
        DataType::Fp8 => Some("fp8"),
        DataType::Fp8Block => Some("fp8_block"),
        _ => None,
    }
}

/// `DataType` → `KVCacheQuantMode` enum name. Only bf16/int8/fp8 exist.
fn kvcache_quant_name(dtype: Option<&DataType>) -> Option<&'static str> {
    match dtype? {
        DataType::Bfloat16 => Some("bfloat16"),
        DataType::Int8 => Some("int8"),
        DataType::Fp8 => Some("fp8"),
        _ => None,
    }
}

/// PyO3 wrapper around [`crate::ForwardPassPerfModel`] (PR #1152), the
/// forward-pass latency model with online correction + regression fallback.
///
/// The hot path (`estimate_forward_pass_time_ms` / `tune_with_fpms`) is pure
/// Rust over the embedded [`Engine`] with NO Python re-entry — the GIL is
/// released via [`Python::allow_threads`] around each compute. Only the
/// constructors (`from_native` / `best_available`) cross into Python once to
/// compile the model (`compile_engine`); that crossing re-acquires the GIL via
/// `with_gil`, which is re-entrant, so calling it from inside a `#[pymethod]`
/// staticmethod is fine.
///
/// Constructors take the engine config + options as JSON strings (the same
/// marshalling the Python `RustForwardPassPerfModel` wrapper used to pass over
/// ctypes), so the Python wrapper's public surface is unchanged.
#[pyclass(name = "RustForwardPassPerfModel")]
pub struct PyForwardPassPerfModel {
    inner: crate::ForwardPassPerfModel,
}

/// Parse the optional options JSON into [`ForwardPassPerfOptions`], defaulting
/// when `None`/empty. Serde fills missing fields from the per-field defaults.
fn parse_fpm_options(options_json: Option<&str>) -> PyResult<crate::ForwardPassPerfOptions> {
    match options_json {
        None => Ok(crate::ForwardPassPerfOptions::default()),
        Some(s) if s.trim().is_empty() => Ok(crate::ForwardPassPerfOptions::default()),
        Some(s) => serde_json::from_str(s)
            .map_err(|e| PyValueError::new_err(format!("invalid options JSON: {e}"))),
    }
}

/// Parse an FPM payload JSON into one iteration's per-rank list. Accepts either
/// a single FPM object or a JSON array of FPM objects (the wrapper's
/// single-rank convenience form). Matches the old ctypes marshalling.
fn parse_fpm_iteration(fpm_json: &str) -> PyResult<Vec<crate::ForwardPassMetrics>> {
    let value: serde_json::Value = serde_json::from_str(fpm_json)
        .map_err(|e| PyValueError::new_err(format!("invalid FPM JSON: {e}")))?;
    let metrics: Vec<crate::ForwardPassMetrics> = if value.is_array() {
        serde_json::from_value(value)
    } else {
        serde_json::from_value(value).map(|m| vec![m])
    }
    .map_err(|e| PyValueError::new_err(format!("invalid FPM payload: {e}")))?;
    Ok(metrics)
}

#[pymethods]
impl PyForwardPassPerfModel {
    /// `RustForwardPassPerfModel.from_native(config_json, options_json=None)`:
    /// strict native AIC model. Compiles the engine via Python `compile_engine`;
    /// raises if the config cannot be compiled.
    #[staticmethod]
    #[pyo3(signature = (config_json, options_json=None))]
    fn from_native(config_json: &str, options_json: Option<&str>) -> PyResult<Self> {
        let config: EngineConfig = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid engine config JSON: {e}")))?;
        let options = parse_fpm_options(options_json)?;
        let inner = crate::ForwardPassPerfModel::from_native(config, options).map_err(aic_to_py)?;
        Ok(Self { inner })
    }

    /// `RustForwardPassPerfModel.best_available(config_json, options_json=None)`:
    /// native when possible, else regression fallback (reason in
    /// `diagnostics()["last_warning"]`).
    #[staticmethod]
    #[pyo3(signature = (config_json, options_json=None))]
    fn best_available(config_json: &str, options_json: Option<&str>) -> PyResult<Self> {
        let config: EngineConfig = serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("invalid engine config JSON: {e}")))?;
        let options = parse_fpm_options(options_json)?;
        let inner =
            crate::ForwardPassPerfModel::best_available(config, options).map_err(aic_to_py)?;
        Ok(Self { inner })
    }

    /// `RustForwardPassPerfModel.from_regression(options_json=None)`:
    /// regression-only model (no native engine, no Python compile).
    #[staticmethod]
    #[pyo3(signature = (options_json=None))]
    fn from_regression(options_json: Option<&str>) -> PyResult<Self> {
        let options = parse_fpm_options(options_json)?;
        let inner = crate::ForwardPassPerfModel::from_regression(options).map_err(aic_to_py)?;
        Ok(Self { inner })
    }

    /// Estimate one forward-pass iteration in ms. `fpm_json` is one iteration as
    /// a single FPM object or a per-attention-DP-rank array. Returns `None` for
    /// regression models without enough data yet. Pure-Rust compute (GIL freed).
    fn estimate_forward_pass_time_ms(
        &self,
        py: Python<'_>,
        fpm_json: &str,
    ) -> PyResult<Option<f64>> {
        let metrics = parse_fpm_iteration(fpm_json)?;
        py.allow_threads(|| self.inner.estimate_forward_pass_time_ms(&metrics))
            .map_err(aic_to_py)
    }

    /// Tune from observed FPM iterations. `iterations_json` is the nested list
    /// `[[iter0_rank0, ...], [iter1_rank0, ...]]` (the wrapper normalizes the
    /// convenience forms before calling). Pure-Rust compute (GIL freed).
    fn tune_with_fpms(&mut self, py: Python<'_>, iterations_json: &str) -> PyResult<()> {
        let iterations: Vec<Vec<crate::ForwardPassMetrics>> = serde_json::from_str(iterations_json)
            .map_err(|e| PyValueError::new_err(format!("invalid tuning iterations JSON: {e}")))?;
        py.allow_threads(|| self.inner.tune_with_fpms(&iterations))
            .map_err(aic_to_py)
    }

    /// Diagnostics (source / readiness / retained count / warning) as JSON.
    fn diagnostics(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.diagnostics())
            .map_err(|e| PyValueError::new_err(format!("diagnostics serialize: {e}")))
    }

    /// Smallest ready native correction factor; `None` until a bucket is ready.
    fn min_correction_factor(&self) -> Option<f64> {
        self.inner.min_correction_factor()
    }

    /// Largest ready native correction factor; `None` until a bucket is ready.
    fn max_correction_factor(&self) -> Option<f64> {
        self.inner.max_correction_factor()
    }

    /// Mean ready native correction factor; `None` until a bucket is ready.
    fn avg_correction_factor(&self) -> Option<f64> {
        self.inner.avg_correction_factor()
    }
}

/// The compiled extension module `aiconfigurator_core._aiconfigurator_core`.
///
/// The `#[pymodule]` function name is the last component of
/// `[tool.maturin] module-name` in `pyproject.toml` (`_aiconfigurator_core`),
/// because PyO3 emits the init symbol as `PyInit_<function name>`. The
/// user-facing top-level `aiconfigurator_core` package
/// (`src/aiconfigurator/aiconfigurator_core/__init__.py`) re-exports the public
/// names from this inner module. This is distinct from `[lib] name` in
/// `Cargo.toml`, which stays `aiconfigurator_core` and drives the ctypes dylib
/// filename.
///
/// Note `build_aic_engine` is intentionally NOT added here: it is a Rust-only
/// entry point for embedded callers, not part of the Python surface.
#[pymodule]
fn _aiconfigurator_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_build_smoke, m)?)?;
    m.add_function(wrap_pyfunction!(engine_spec_bincode_from_json, m)?)?;
    m.add_class::<AicEngine>()?;
    m.add_class::<PyForwardPassPerfModel>()?;
    Ok(())
}

#[cfg(all(test, feature = "embed-python"))]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
    use crate::engine::spec::EngineSpec;
    use crate::operators::op::Op;
    use crate::operators::{ContextAttentionOp, ElementwiseOp, GemmOp, GenerationAttentionOp};
    use crate::{BackendKind, EngineConfig, ParallelMapping, QuantizationConfig};

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems")
    }

    const TEST_MODEL: &str = "MiniMaxAI/MiniMax-M2.5";

    /// Hand-built context op list against the b200_sxm/vllm/0.19.0 perf tables.
    /// `Elementwise` is DB-free (pure mem-bandwidth SOL); `Gemm` and
    /// `ContextAttention` hit `gemm_perf` / `context_attention_perf`, both of
    /// which exist for this fixture. Mirrors a MiniMax-shaped context graph
    /// closely enough to exercise the binding/runtime orchestration without
    /// rebuilding the (deleted) model layer.
    fn context_ops() -> Vec<Op> {
        vec![
            Op::Elementwise(ElementwiseOp {
                name: "rmsnorm".into(),
                scale_factor: 1.0,
                bytes_per_token: 8192.0,
                seq_split: 1,
            }),
            Op::Gemm(GemmOp {
                name: "qkv_gemm".into(),
                scale_factor: 1.0,
                n: 4096,
                k: 4096,
                quant_mode: GemmQuantMode::Fp8Block,
                scale_num_tokens: 0,
                low_precision_input: false,
                seq_split: 1,
            }),
            Op::ContextAttention(ContextAttentionOp {
                name: "context_attention".into(),
                scale_factor: 1.0,
                n: 32,
                n_kv: 8,
                head_size: 128,
                window_size: 0,
                kv_cache_dtype: KvCacheQuantMode::Fp8,
                fmha_quant_mode: FmhaQuantMode::Bfloat16,
                use_qk_norm: false,
                cp_size: 1,
            }),
        ]
    }

    /// Hand-built generation op list. `GenerationAttention` hits
    /// `generation_attention_perf`, which exists for the fixture.
    fn generation_ops() -> Vec<Op> {
        vec![
            Op::Elementwise(ElementwiseOp {
                name: "rmsnorm".into(),
                scale_factor: 1.0,
                bytes_per_token: 8192.0,
                seq_split: 1,
            }),
            Op::GenerationAttention(GenerationAttentionOp {
                name: "generation_attention".into(),
                scale_factor: 1.0,
                n: 32,
                n_kv: 8,
                head_size: 128,
                window_size: 0,
                kv_cache_dtype: KvCacheQuantMode::Fp8,
            }),
        ]
    }

    fn fixture_engine_config() -> EngineConfig {
        EngineConfig {
            schema_version: crate::ENGINE_CONFIG_SCHEMA_VERSION,
            model_name: TEST_MODEL.to_string(),
            system_name: "b200_sxm".to_string(),
            systems_path: None,
            backend: BackendKind::Vllm,
            backend_version: Some("0.19.0".to_string()),
            kv_block_size: None,
            parallel: ParallelMapping {
                tp_size: 8,
                pp_size: 1,
                attention_dp_size: Some(1),
                moe_tp_size: Some(1),
                moe_ep_size: Some(8),
                cp_size: None,
            },
            quantization: QuantizationConfig {
                weight_dtype: None,
                moe_dtype: None,
                activation_dtype: None,
                kv_cache_dtype: None,
            },
            speculative: None,
            perf_db_sources: Default::default(),
            extra: BTreeMap::new(),
        }
    }

    /// Build bincoded `EngineSpec` bytes from hand-built op lists. The lists
    /// query the real b200_sxm/vllm/0.19.0 perf tables so the binding
    /// pass-through numbers are real, not synthetic.
    fn fixture_spec_bytes() -> Vec<u8> {
        let spec = EngineSpec::new(fixture_engine_config(), context_ops(), generation_ops());
        spec.to_bincode().unwrap()
    }

    /// The binding layer must be a faithful pass-through: an `AicEngine` built
    /// from spec bytes via `from_spec` must produce the SAME numbers as a raw
    /// `Engine` built from the same bytes via `from_spec_bytes`.
    #[test]
    fn aic_engine_matches_raw_engine() {
        let bytes = fixture_spec_bytes();
        let root = systems_root();

        let raw = Engine::from_spec_bytes(&bytes, &root).unwrap();
        let aic = AicEngine::from_spec(&bytes, root.to_str()).unwrap();

        // run_static (Both): tuple from the binding == StaticResult from raw.
        let rt = RuntimeConfig {
            batch_size: 1,
            isl: 1024,
            osl: 8,
            ..Default::default()
        };
        let raw_static = raw
            .run_static(&rt, StaticMode::Both, DEFAULT_STATIC_STRIDE)
            .unwrap();
        // Positional order: (bs, beam, isl, osl, prefix, seq_corr, gen_seq_corr, mode, stride).
        let (ctx, gen, total) = Python::with_gil(|py| {
            aic.run_static(
                py,
                1,
                1,
                1024,
                8,
                0,
                1.0,
                1.0,
                "static",
                DEFAULT_STATIC_STRIDE,
            )
        })
        .unwrap();
        assert!((ctx - raw_static.context_ms).abs() < 1e-12);
        assert!((gen - raw_static.generation_ms).abs() < 1e-12);
        assert!((total - raw_static.total_ms).abs() < 1e-12);

        // predict_prefill_latency == raw Context-mode total.
        let raw_prefill = raw
            .run_static(
                &RuntimeConfig {
                    batch_size: 2,
                    isl: 1024,
                    osl: 1,
                    prefix: 0,
                    ..Default::default()
                },
                StaticMode::Context,
                DEFAULT_STATIC_STRIDE,
            )
            .unwrap()
            .total_ms;
        let prefill = Python::with_gil(|py| aic.predict_prefill_latency(py, 2, 1024, 0)).unwrap();
        assert!((prefill - raw_prefill).abs() < 1e-12);

        // predict_decode_latency (osl=2) == raw Generation-mode total.
        let raw_decode = raw
            .run_static(
                &RuntimeConfig {
                    batch_size: 4,
                    isl: 1024,
                    osl: 2,
                    ..Default::default()
                },
                StaticMode::Generation,
                DEFAULT_STATIC_STRIDE,
            )
            .unwrap()
            .total_ms;
        let decode = Python::with_gil(|py| aic.predict_decode_latency(py, 4, 1024, 2)).unwrap();
        assert!((decode - raw_decode).abs() < 1e-12);
    }

    /// `mode` string mapping must match the Rust `StaticMode` semantics, and an
    /// unknown mode must raise (not silently default).
    #[test]
    fn mode_strings_map_correctly() {
        let bytes = fixture_spec_bytes();
        let root = systems_root();
        let aic = AicEngine::from_spec(&bytes, root.to_str()).unwrap();

        // Positional order: (bs, beam, isl, osl, prefix, seq_corr, gen_seq_corr, mode, stride).
        Python::with_gil(|py| {
            let ctx_only = aic
                .run_static(py, 1, 1, 1024, 8, 0, 1.0, 1.0, "static_ctx", 32)
                .unwrap();
            assert!(ctx_only.0 > 0.0 && ctx_only.1 == 0.0);

            let gen_only = aic
                .run_static(py, 1, 1, 1024, 8, 0, 1.0, 1.0, "static_gen", 32)
                .unwrap();
            assert!(gen_only.0 == 0.0 && gen_only.1 > 0.0);

            assert!(aic
                .run_static(py, 1, 1, 1024, 8, 0, 1.0, 1.0, "bogus", 32)
                .is_err());
        });
    }

    /// `mixed_step_latency` / `decode_step_latency` bindings must pass through
    /// the raw `Engine` numbers unchanged.
    #[test]
    fn per_step_bindings_match_raw_engine() {
        let bytes = fixture_spec_bytes();
        let root = systems_root();
        let raw = Engine::from_spec_bytes(&bytes, &root).unwrap();
        let aic = AicEngine::from_spec(&bytes, root.to_str()).unwrap();

        let raw_mixed = raw.mixed_step_latency(1024, 2, 1024, 8, 0).unwrap();
        let mixed = Python::with_gil(|py| aic.mixed_step_latency(py, 1024, 2, 1024, 8, 0)).unwrap();
        assert!((mixed - raw_mixed).abs() < 1e-12);
        let raw_breakdown = raw.mixed_step_breakdown(1024, 2, 1024, 8, 0).unwrap();
        let breakdown =
            Python::with_gil(|py| aic.mixed_step_breakdown(py, 1024, 2, 1024, 8, 0)).unwrap();
        assert_eq!(
            breakdown,
            (
                raw_breakdown[0],
                raw_breakdown[1],
                raw_breakdown[2],
                raw_breakdown[3],
            )
        );

        let raw_decode = raw.decode_step_latency(4, 1024, 8).unwrap();
        let decode = Python::with_gil(|py| aic.decode_step_latency(py, 4, 1024, 8)).unwrap();
        assert!((decode - raw_decode).abs() < 1e-12);
    }

    /// `engine_spec_bincode_from_json` round-trips: JSON → bincode → decoded
    /// `EngineSpec` equals the original spec.
    #[test]
    fn engine_spec_json_to_bincode_round_trips() {
        let bytes = fixture_spec_bytes();
        let original = EngineSpec::from_bincode(&bytes).unwrap();
        let json = serde_json::to_string(&original).unwrap();
        let out = engine_spec_bincode_from_json(&json).unwrap();
        let decoded = EngineSpec::from_bincode(&out).unwrap();
        assert_eq!(original, decoded);
    }

    /// Pure-Rust inherent `prefill_latency_ms` / `decode_latency_ms` (no `py`
    /// token) must match the raw `Engine` predict methods — this is the
    /// GIL-free Mocker hot path surface that `tests/embedded_round_trip.rs`
    /// exercises end-to-end.
    #[test]
    fn inherent_predict_matches_raw_engine() {
        let bytes = fixture_spec_bytes();
        let root = systems_root();
        let raw = Engine::from_spec_bytes(&bytes, &root).unwrap();
        let aic = AicEngine::from_spec(&bytes, root.to_str()).unwrap();

        let raw_prefill = raw.predict_prefill_latency(2, 1024, 0).unwrap();
        let aic_prefill = aic.prefill_latency_ms(2, 1024, 0).unwrap();
        assert!((aic_prefill - raw_prefill).abs() < 1e-12);
        assert!(aic_prefill > 0.0 && aic_prefill.is_finite());

        let raw_decode = raw.predict_decode_latency(4, 1024, 2).unwrap();
        let aic_decode = aic.decode_latency_ms(4, 1024, 2).unwrap();
        assert!((aic_decode - raw_decode).abs() < 1e-12);
        assert!(aic_decode > 0.0 && aic_decode.is_finite());
    }
}
