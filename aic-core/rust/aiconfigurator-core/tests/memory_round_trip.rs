// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end KV-cache capacity round-trip.
//!
//! Drives the full Rust → Python → Rust capacity path: calls the top-level
//! [`aiconfigurator_core::estimate_kv_cache`] (the **pure forwarder** the Dynamo
//! Mocker uses) for a real fixture model. That crosses into Python once to run
//! `aiconfigurator.sdk.memory.estimate_kv_cache` (which owns the budget math AND
//! the tolerance margin) and rebuilds a [`aiconfigurator_core::KvCacheEstimate`]
//! from the returned dict.
//!
//! ## Why this test exists
//!
//! Removing the `aiconfigurator_core.estimate_kv_cache` `#[pyfunction]` removed
//! the only in-repo path that exercised the Rust `fetch_python_estimate`
//! (forwarding `tolerance_fraction`) and `estimate_from_dict` (parsing
//! `tolerance_adjusted`). This test restores that coverage: a typo in a forwarded
//! kwarg name or a parsed dict key fails here instead of first surfacing in the
//! deferred downstream Mocker PR. It mirrors `embedded_round_trip.rs`, the sibling
//! `build_aic_engine` round-trip.
//!
//! ## Run requirements
//!
//! Same as `embedded_round_trip.rs`: an embedded Python interpreter that can
//! import `aiconfigurator.sdk.memory` (which imports the maturin-built
//! `aiconfigurator_core`), plus the perf DB (LFS) for the native SystemSpec
//! capacity. Run after
//! `uv run maturin develop -m aic-core/rust/aiconfigurator-core/Cargo.toml --release`:
//! ```text
//! AIC_REQUIRE_EMBEDDED_ROUND_TRIP=1 \
//!   PYTHONPATH="$PWD/aic-core/src:$PWD/.venv/lib/python3.12/site-packages:$PWD/src" \
//!   cargo test -p aiconfigurator-core --test memory_round_trip -- --nocapture
//! ```
//!
//! ## Honest skip vs. enforced run
//!
//! On a bare `cargo test` the Python imports are usually unavailable, so the
//! test prints a visible `SKIP` and returns (it does NOT read as coverage).
//! `AIC_REQUIRE_EMBEDDED_ROUND_TRIP=1` makes an import failure a HARD failure so
//! the test cannot false-pass. The estimate itself additionally needs the perf
//! DB; when that is absent the call errors and the test skips with a message
//! (matching the Python integration test in `tests/integration/test_memory_estimation.py`).

#![cfg(feature = "embed-python")]

use std::collections::BTreeMap;

use aiconfigurator_core::{
    estimate_kv_cache, BackendKind, EngineConfig, EstimateSource, KvCacheEstimateOptions,
    KvCacheEstimateRequest, KvCacheMemoryFraction, ParallelMapping, QuantizationConfig,
    ENGINE_CONFIG_SCHEMA_VERSION,
};
use pyo3::prelude::*;

const TEST_MODEL: &str = "Qwen/Qwen3-32B";
const TOLERANCE: f64 = 0.05;

/// Soft-skip guard: true only when the embedded interpreter can import
/// `aiconfigurator.sdk.memory` (which transitively imports the maturin-built
/// `aiconfigurator_core`).
fn python_memory_importable() -> bool {
    Python::with_gil(|py| match py.import("aiconfigurator.sdk.memory") {
        Ok(_) => true,
        Err(e) => {
            let exe: String = py
                .import("sys")
                .and_then(|s| s.getattr("executable"))
                .and_then(|x| x.extract())
                .unwrap_or_default();
            eprintln!("memory_round_trip: import failed (sys.executable={exe}): {e}");
            false
        }
    })
}

/// TRT-LLM native request for the fixture model. `tolerance_fraction` is the
/// only field that varies between the two calls under test.
fn request(tolerance_fraction: Option<f64>) -> KvCacheEstimateRequest {
    KvCacheEstimateRequest {
        engine: EngineConfig {
            schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
            model_name: TEST_MODEL.to_string(),
            system_name: "h200_sxm".to_string(),
            systems_path: None,
            backend: BackendKind::Trtllm,
            backend_version: Some("1.3.0rc10".to_string()),
            kv_block_size: None,
            parallel: ParallelMapping {
                tp_size: 1,
                pp_size: 1,
                attention_dp_size: Some(1),
                moe_tp_size: None,
                moe_ep_size: None,
                cp_size: None,
            },
            quantization: QuantizationConfig {
                weight_dtype: None,
                moe_dtype: None,
                activation_dtype: None,
                kv_cache_dtype: None,
            },
            speculative: None,
            perf_db_sources: BTreeMap::new(),
            database_mode: Default::default(),
            transfer_policy: None,
            extra: BTreeMap::new(),
        },
        max_num_tokens: 8192,
        max_batch_size: 256,
        kv_cache_memory_fraction: KvCacheMemoryFraction::OfFree(0.9),
        gpu_memory_capacity_bytes_override: None,
        tolerance_fraction,
        options: KvCacheEstimateOptions {
            allow_naive_fallback: false,
            allow_hf_config_download: false,
            naive_kv_reservation: 0.80,
        },
    }
}

#[test]
fn memory_round_trip_forwards_tolerance_and_parses_adjusted() {
    let required = std::env::var_os("AIC_REQUIRE_EMBEDDED_ROUND_TRIP").is_some();
    if !python_memory_importable() {
        assert!(
            !required,
            "memory_round_trip: AIC_REQUIRE_EMBEDDED_ROUND_TRIP is set but \
             `aiconfigurator.sdk.memory` is not importable — run after \
             `maturin develop` with PYTHONPATH including aic-core/src, the venv \
             site-packages, and src."
        );
        eprintln!(
            "memory_round_trip: SKIP — `aiconfigurator.sdk.memory` not importable. \
             Set AIC_REQUIRE_EMBEDDED_ROUND_TRIP=1 + PYTHONPATH to enforce."
        );
        return;
    }

    // Raw estimate (no tolerance forwarded). The native path needs the perf DB;
    // when it is absent the Python side raises the "unsupported model/backend/GPU"
    // ValueError (native build could not run), which we tolerate as a skip — same
    // as the Python integration test. ANY other error (a forwarder kwarg typo, a
    // dict-parse regression, budget math) must fail the test rather than hide.
    let raw = match estimate_kv_cache(request(None)) {
        Ok(est) => est,
        Err(e) if e.to_string().contains("unsupported model/backend/GPU for KV-cache estimation") => {
            eprintln!("memory_round_trip: SKIP — native estimate unavailable (perf DB?): {e}");
            return;
        }
        Err(e) => panic!("memory_round_trip: native estimate failed unexpectedly (not a missing-fixture error): {e}"),
    };

    // `estimate_from_dict` parsed the scalar fields and the breakdown; with no
    // tolerance forwarded, `tolerance_adjusted` must be None.
    assert_eq!(raw.source, EstimateSource::Native);
    assert!(raw.total_kv_size_bytes > 0 && raw.kv_size_per_token_bytes > 0);
    assert!(raw.memory_breakdown.is_some());
    assert!(raw.tolerance_adjusted.is_none());

    // Tolerance forwarded: `fetch_python_estimate` must pass `tolerance_fraction`
    // through, and `estimate_from_dict` must parse the `tolerance_adjusted` dict.
    let adjusted = estimate_kv_cache(request(Some(TOLERANCE)))
        .expect("tolerance estimate must succeed once the raw one did");
    let adj = adjusted
        .tolerance_adjusted
        .expect("tolerance forwarded -> tolerance_adjusted must be Some");

    assert_eq!(adj.tolerance_fraction, TOLERANCE);
    // Same formula the deleted Rust `apply_tolerance` used, now computed Python-side.
    let expected_bytes = (raw.total_kv_size_bytes as f64 * (1.0 - TOLERANCE)) as u64;
    assert_eq!(adj.total_kv_size_bytes, expected_bytes);
    assert_eq!(
        adj.total_kv_size_tokens,
        expected_bytes / raw.kv_size_per_token_bytes
    );
    // The raw fields are unchanged when a tolerance is applied.
    assert_eq!(adjusted.total_kv_size_bytes, raw.total_kv_size_bytes);
}
