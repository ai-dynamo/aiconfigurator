// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 bindings for the Phase 1.5 compiled-engine core (commit E4).
//!
//! Exposes two call directions over a single PyO3 dependency:
//!
//! * **Python → Rust (hot path).** [`AicEngine`] is a `#[pyclass]` wrapping the
//!   E3 [`Engine`]. Its `#[pymethods]` (`from_spec`, `run_static`,
//!   `predict_prefill_latency`, `predict_decode_latency`, `mixed_step_latency`,
//!   `decode_step_latency`) are the surface the Python sweep / Mocker bridge
//!   calls per point. The agg sweep is orchestrated in Python; there is no
//!   Rust `run_agg`. Each method
//!   releases the GIL around the Rust compute via [`Python::allow_threads`],
//!   so the migration's whole point — Rust computing without holding the GIL —
//!   holds.
//! * **Rust → Python → Rust (embedded path).** [`build_aic_engine`] is a plain
//!   `pub` Rust fn (NOT a `#[pyfunction]`): Rust callers such as the Dynamo
//!   Mocker call it with flat scalars, it crosses into Python once to run
//!   `aiconfigurator.sdk.engine.compile_engine`, then builds an [`Engine`] from
//!   the returned bincode bytes. After that the `predict_*` hot path is pure
//!   Rust with no GIL.
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
use crate::engine::runtime::{Engine, RuntimeConfig, StaticMode, StaticResult, DEFAULT_STATIC_STRIDE};
use crate::ENGINE_CONFIG_SCHEMA_VERSION;

/// Trivial smoke export retained from E1: returns the engine-config schema
/// version so callers can confirm the extension built and imported correctly.
#[pyfunction]
fn _phase15_smoke() -> u32 {
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
/// → repo-relative `src/aiconfigurator/systems`.
fn resolve_systems_root(systems_path: Option<&str>) -> PyResult<PathBuf> {
    if let Some(p) = systems_path {
        return Ok(PathBuf::from(p));
    }
    if let Some(p) = std::env::var_os("AICONFIGURATOR_SYSTEMS_PATH") {
        return Ok(PathBuf::from(p));
    }
    crate::repo_relative("src/aiconfigurator/systems").ok_or_else(|| {
        PyValueError::new_err(
            "could not resolve systems path: pass systems_path, set \
             AICONFIGURATOR_SYSTEMS_PATH, or run from an AIC checkout",
        )
    })
}

/// PyO3 wrapper around the E3 [`Engine`]: a compiled engine the Python sweep /
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
    /// Positional args mirror Phase 1's `BaseBackend.run_static` runtime shape,
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
    fn predict_prefill_latency(&self, py: Python<'_>, bs: u32, isl: u32, prefix: u32) -> PyResult<f64> {
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
        py.allow_threads(|| self.inner.mixed_step_latency(ctx_tokens, gen_tokens, isl, osl, prefix))
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

/// Rust → Python → Rust embedded build entry point.
///
/// A plain `pub` Rust fn (NOT a `#[pyfunction]`): Rust callers (e.g. the Dynamo
/// Mocker) call it with flat scalars. It crosses into Python exactly once to
/// run `aiconfigurator.sdk.engine.compile_engine`, which walks the model's
/// op lists and returns bincoded [`EngineSpec`] bytes; it then builds the
/// [`Engine`] from those bytes (via [`Engine::from_spec_bytes`], which does
/// `from_bincode` + `PerfDatabase::load` + `Engine::build`). Follows the
/// plan's "Rust → Python call shape" skeleton (`with_gil → import →
/// call_method1("compile_engine", ...) → extract::<Vec<u8>>() → build`).
///
/// The flat arg list matches the E0 `compile_engine` signature decision
/// (`docs/phase-1.5-opspec-audit.md`). `systems_path` is the Rust-side perf-DB
/// root (it is also forwarded to `compile_engine` so the two stay aligned).
///
/// **NOT end-to-end testable in E4:** Python's `compile_engine`
/// (`sdk/engine.py`) does not exist until E5, so any test that calls this is
/// gated `#[ignore]` ("un-ignore at E5"). The full round-trip validation lands
/// in E7's `tests/embedded_round_trip.rs`.
// `pub` and re-exported from `lib.rs` for E5/E7 embedded callers (Mocker,
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
    nextn_accept_rates: Option<Vec<f64>>,
    kv_block_size: Option<u32>,
    systems_path: Option<&str>,
) -> Result<AicEngine, AicError> {
    let spec_bytes: Vec<u8> = Python::with_gil(|py| -> PyResult<Vec<u8>> {
        let engine_mod = py.import("aiconfigurator.sdk.engine")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("backend_version", backend_version)?;
        kwargs.set_item("tp_size", tp_size)?;
        kwargs.set_item("pp_size", pp_size)?;
        kwargs.set_item("attention_dp_size", attention_dp_size)?;
        kwargs.set_item("moe_tp_size", moe_tp_size)?;
        kwargs.set_item("moe_ep_size", moe_ep_size)?;
        kwargs.set_item("gemm_quant_mode", gemm_quant_mode)?;
        kwargs.set_item("moe_quant_mode", moe_quant_mode)?;
        kwargs.set_item("kvcache_quant_mode", kvcache_quant_mode)?;
        kwargs.set_item("fmha_quant_mode", fmha_quant_mode)?;
        kwargs.set_item("comm_quant_mode", comm_quant_mode)?;
        kwargs.set_item("nextn", nextn)?;
        kwargs.set_item("nextn_accept_rates", nextn_accept_rates)?;
        kwargs.set_item("kv_block_size", kv_block_size)?;
        kwargs.set_item("systems_path", systems_path)?;
        engine_mod
            .call_method("compile_engine", (model_path, system, backend), Some(&kwargs))?
            .extract::<Vec<u8>>()
    })
    // PyErr → AicError inline (keeps error.rs pyo3-free).
    .map_err(|e| AicError::InvalidEngineConfig(format!("compile_engine: {e}")))?;

    let systems_root: PathBuf = match systems_path {
        Some(p) => PathBuf::from(p),
        None => std::env::var_os("AICONFIGURATOR_SYSTEMS_PATH")
            .map(PathBuf::from)
            .or_else(|| crate::repo_relative("src/aiconfigurator/systems"))
            .ok_or_else(|| {
                AicError::DataRoot(
                    "set AICONFIGURATOR_SYSTEMS_PATH or pass systems_path".to_string(),
                )
            })?,
    };
    let engine = Engine::from_spec_bytes(&spec_bytes, systems_root.as_path() as &Path)?;
    Ok(AicEngine::new(engine))
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
    m.add_function(wrap_pyfunction!(_phase15_smoke, m)?)?;
    m.add_function(wrap_pyfunction!(engine_spec_bincode_from_json, m)?)?;
    m.add_class::<AicEngine>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
    use crate::engine::spec::EngineSpec;
    use crate::operators::op::Op;
    use crate::operators::{
        ContextAttentionOp, ElementwiseOp, GemmOp, GenerationAttentionOp,
    };
    use crate::{BackendKind, EngineConfig, ParallelMapping, QuantizationConfig};

    fn systems_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator/systems")
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
            }),
            Op::Gemm(GemmOp {
                name: "qkv_gemm".into(),
                scale_factor: 1.0,
                n: 4096,
                k: 4096,
                quant_mode: GemmQuantMode::Fp8Block,
                scale_num_tokens: 0,
                low_precision_input: false,
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
            },
            quantization: QuantizationConfig {
                weight_dtype: None,
                moe_dtype: None,
                activation_dtype: None,
                kv_cache_dtype: None,
            },
            speculative: None,
            extra: BTreeMap::new(),
        }
    }

    /// Build bincoded `EngineSpec` bytes from hand-built op lists (the model
    /// layer that previously sourced them was deleted in E7). The lists query
    /// the real b200_sxm/vllm/0.19.0 perf tables so the binding pass-through
    /// numbers are real, not synthetic.
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
            aic.run_static(py, 1, 1, 1024, 8, 0, 1.0, 1.0, "static", DEFAULT_STATIC_STRIDE)
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
            let ctx_only =
                aic.run_static(py, 1, 1, 1024, 8, 0, 1.0, 1.0, "static_ctx", 32).unwrap();
            assert!(ctx_only.0 > 0.0 && ctx_only.1 == 0.0);

            let gen_only =
                aic.run_static(py, 1, 1, 1024, 8, 0, 1.0, 1.0, "static_gen", 32).unwrap();
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
