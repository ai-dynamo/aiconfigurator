// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end embedded round-trip.
//!
//! Drives the full Rust → Python → Rust embedded build path through both the
//! preferred [`aiconfigurator_core::AicEngineBuilder`] and the flat
//! [`aiconfigurator_core::build_aic_engine`] compatibility adapter. Each
//! crosses into Python once to run
//! `aiconfigurator_core.sdk.engine.compile_engine`, gets bincoded `EngineSpec`
//! bytes back, loads the matching perf database, and returns an
//! [`aiconfigurator_core::AicEngine`]. The test asserts both paths agree and
//! that the **pure-Rust hot path** produces finite, positive latencies.
//!
//! ## Why this proves the Mocker hot path is PyO3-free
//!
//! After construction, the test calls the **inherent** Rust methods
//! `AicEngine::prefill_latency_ms` / `decode_latency_ms` — these take NO PyO3
//! `py` token (unlike the `#[pymethods]` forms). They compile and run without
//! acquiring the GIL; any re-entry into Python would require a `py` token,
//! which these signatures cannot produce. The call succeeding is the
//! observable proof that the per-point predict path is pure Rust over the
//! compiled `Engine`, not a Python re-entry.
//!
//! ## Run requirements
//!
//! Engine construction embeds a Python interpreter (PyO3 `auto-initialize`)
//! and imports `aiconfigurator_core.sdk.engine`, which itself imports the
//! maturin-built `aiconfigurator_core` extension. The test therefore needs
//! `aiconfigurator_core` installed into the interpreter
//! (`uv run maturin develop -m aic-core/rust/aiconfigurator-core/Cargo.toml --release`).
//!
//! The embedded interpreter (the framework libpython the test binary links) is
//! NOT the uv venv, so it does not see the venv's installed core package or the
//! maturin-built `aiconfigurator_core` automatically. Point it at the core
//! source and venv site-packages via **absolute** `PYTHONPATH` entries
//! (relative paths do not resolve under cargo's test cwd):
//! ```text
//! AIC_REQUIRE_EMBEDDED_ROUND_TRIP=1 \
//!   PYTHONPATH="$PWD/aic-core/src:$PWD/.venv/lib/python3.12/site-packages" \
//!   cargo test -p aiconfigurator-core --test embedded_round_trip -- --nocapture
//! ```
//! (run after `uv run maturin develop -m aic-core/rust/aiconfigurator-core/Cargo.toml
//! --release`, from the repo root; adjust the venv python version if needed).
//!
//! ## Honest skip vs. enforced run
//!
//! On a bare `cargo test` the Python imports are usually unavailable, so the
//! test prints a visible `SKIP` line and returns — it does NOT silently read
//! as coverage. To make the skip impossible (CI / the documented invocation),
//! set `AIC_REQUIRE_EMBEDDED_ROUND_TRIP=1`: then an import failure is a HARD
//! failure (panic), so the test cannot false-pass. The documented run command
//! below sets it.

#![cfg(feature = "embed-python")]

use aiconfigurator_core::{build_aic_engine, AicEngineBuilder, BackendKind};
use pyo3::prelude::*;

const TEST_MODEL: &str = "MiniMaxAI/MiniMax-M2.5";

/// Soft-skip guard: true only when the embedded interpreter can import the
/// `aiconfigurator_core.sdk.engine` module (which transitively imports the
/// maturin-built `aiconfigurator_core`). Returns false otherwise so the test
/// passes without running the assertions on a bare `cargo test`.
fn python_engine_importable() -> bool {
    Python::with_gil(|py| match py.import("aiconfigurator_core.sdk.engine") {
        Ok(_) => true,
        Err(e) => {
            let exe: String = py
                .import("sys")
                .and_then(|s| s.getattr("executable"))
                .and_then(|x| x.extract())
                .unwrap_or_default();
            eprintln!("embedded_round_trip: import failed (sys.executable={exe}): {e}");
            false
        }
    })
}

#[test]
fn embedded_builder_and_compatibility_adapter_match() {
    let required = std::env::var_os("AIC_REQUIRE_EMBEDDED_ROUND_TRIP").is_some();
    if !python_engine_importable() {
        assert!(
            !required,
            "embedded_round_trip: AIC_REQUIRE_EMBEDDED_ROUND_TRIP is set but \
             `aiconfigurator_core.sdk.engine` is not importable — run after \
             `maturin develop` with PYTHONPATH including aic-core/src, the venv \
             site-packages."
        );
        eprintln!(
            "embedded_round_trip: SKIP — `aiconfigurator_core.sdk.engine` not importable. \
             Set AIC_REQUIRE_EMBEDDED_ROUND_TRIP=1 + PYTHONPATH to enforce."
        );
        return;
    }

    // Preferred Rust -> Python (compile_engine) -> Rust path.
    let engine = AicEngineBuilder::new(TEST_MODEL, "b200_sxm", BackendKind::Vllm)
        .backend_version("0.19.0")
        .tp_size(8)
        .moe_parallelism(Some(1), Some(8))
        .build()
        .expect("AicEngineBuilder must succeed end-to-end");

    // Flat compatibility adapter must remain behaviorally identical during the
    // announced migration window.
    let compatibility_engine = build_aic_engine(
        TEST_MODEL,
        "b200_sxm",
        "vllm",
        Some("0.19.0"),
        8,       // tp_size
        1,       // pp_size
        1,       // attention_dp_size
        Some(1), // moe_tp_size
        Some(8), // moe_ep_size
        None,    // gemm_quant_mode (inferred by compile_engine)
        None,    // moe_quant_mode
        None,    // kvcache_quant_mode
        None,    // fmha_quant_mode
        None,    // comm_quant_mode
        0,       // nextn
        None,    // kv_block_size
        None,    // resolve the installed core wheel's bundled systems data
    )
    .expect("build_aic_engine compatibility adapter must succeed end-to-end");

    // Pure-Rust hot path: these inherent methods take NO `py` token, so this
    // compute happens with the GIL never acquired. Finite, positive latencies
    // confirm the compiled Engine ran over real perf data.
    let prefill = engine
        .prefill_latency_ms(1, 1024, 0)
        .expect("prefill predict must succeed");
    assert!(
        prefill.is_finite() && prefill > 0.0,
        "prefill latency must be finite and > 0, got {prefill}"
    );
    let compatibility_prefill = compatibility_engine
        .prefill_latency_ms(1, 1024, 0)
        .expect("compatibility prefill predict must succeed");
    assert_eq!(prefill, compatibility_prefill);

    let decode = engine
        .decode_latency_ms(1, 1024, 2)
        .expect("decode predict must succeed");
    assert!(
        decode.is_finite() && decode > 0.0,
        "decode latency must be finite and > 0, got {decode}"
    );
    let compatibility_decode = compatibility_engine
        .decode_latency_ms(1, 1024, 2)
        .expect("compatibility decode predict must succeed");
    assert_eq!(decode, compatibility_decode);
}
