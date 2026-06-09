// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native core latency API for AIConfigurator.
//!
//! The compiled-engine path is the only supported entry point: Python's
//! `compile_engine` walks the model once and emits an [`engine::spec::EngineSpec`]
//! (op lists + [`EngineConfig`] identity); the Rust [`engine::Engine`] executes
//! it without re-entering Python. [`build_aic_engine`] is the Rust → Python →
//! Rust embedded build entry point for callers in other crates (the Dynamo
//! Mocker, `tests/embedded_round_trip.rs`); [`AicEngine`] is the PyO3 hot-path
//! pyclass.

use std::path::PathBuf;

mod py;

// Modular core. `common/` holds shared foundation types (enums, error,
// system_spec) with no AIC-domain knowledge. Top-level files (`config`,
// `session`) and directories (`operators`, `perf_database`) carry the domain
// logic the compiled `engine` executes.
mod common;
mod config;
pub mod engine;
mod fpm;
mod operators;
mod perf_database;
mod session;

pub use common::AicError;
// Forward-pass perf model (PR #1152): a forward-pass latency model with online
// correction, regression fallback, diagnostics, and readiness, built on the
// compiled [`engine::Engine`]. Re-exported so Rust embedders (the Dynamo
// planner / Mocker) can use it natively; also exposed to Python via the
// `RustForwardPassPerfModel` pyclass in `py.rs`.
pub use fpm::{
    ForwardPassPerfDiagnostics, ForwardPassPerfModel, ForwardPassPerfOptions,
    ForwardPassPerfReadiness, ForwardPassPerfSource,
};
// Forward-pass metrics telemetry types and schema version, plus the
// crate-internal validation helper. Re-exported at the crate root so existing
// `crate::ForwardPassMetrics` / `crate::FPM_VERSION` references (in `py.rs`,
// `engine/runtime.rs`) keep resolving after the types moved into `fpm`.
pub(crate) use fpm::validate_forward_pass_metrics;
pub use fpm::{ForwardPassMetrics, QueuedRequestMetrics, ScheduledRequestMetrics, FPM_VERSION};
// PyO3 bindings. `AicEngine` is the Python -> Rust hot-path pyclass;
// `build_aic_engine` is the Rust -> Python -> Rust embedded build entry point
// for callers in OTHER crates (the Dynamo Mocker,
// `tests/embedded_round_trip.rs`). They must be `pub`-re-exported here because
// the `py` module itself is private.
pub use py::{build_aic_engine, AicEngine};
// Public wire/identity config types live in `config`. Re-exported at the crate
// root so existing `crate::EngineConfig` / `crate::BackendKind` / ... paths
// resolve unchanged across the crate and for external consumers.
pub use config::{
    BackendKind, DataType, EngineConfig, ParallelMapping, QuantizationConfig, SpeculativeConfig,
    ENGINE_CONFIG_SCHEMA_VERSION, ENGINE_SPEC_SCHEMA_VERSION,
};

/// Resolve a repo-relative path by walking up from the crate manifest dir.
/// Used by [`py`] (e.g. `crate::repo_relative("src/aiconfigurator/systems")`)
/// to locate the bundled data roots when developing in-tree.
pub(crate) fn repo_relative(rel: &str) -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}
