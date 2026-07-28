// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Forward-pass-level performance model with optional online tuning (PR #1152).
//!
//! Built on the compiled [`crate::engine::Engine`]: the `Native` variant holds
//! an `Arc<Engine>` and the native estimate routes through
//! [`crate::engine::Engine::forward_pass_time_ms`]. The online correction /
//! regression / diagnostics / readiness logic is engine-agnostic.
//!
//! Constructing a native model crosses into Python exactly once to compile the
//! model into an [`crate::engine::spec::EngineSpec`] (mirroring
//! [`crate::build_aic_engine`]); after that the hot path
//! (`estimate_forward_pass_time_ms` / `tune_with_fpms`) is pure Rust over the
//! `Engine` with no Python re-entry.
//!
//! Submodules:
//! - [`metrics`]: the `ForwardPassMetrics` telemetry types and validation.
//! - [`model`]: the public [`ForwardPassPerfModel`] and its diagnostics.
//! - [`correction`]: the native online-correction grid.
//! - [`regression`]: the regression fallback.
//! - [`samples`]: shared bucketed-sample infrastructure.
//! - [`options`]: tuning controls.

mod correction;
mod metrics;
mod model;
mod options;
mod regression;
mod samples;

#[cfg(test)]
mod tests;

pub(crate) use metrics::validate_forward_pass_metrics;
pub use metrics::{ForwardPassMetrics, QueuedRequestMetrics, ScheduledRequestMetrics, FPM_VERSION};
pub use model::{
    ForwardPassPerfDiagnostics, ForwardPassPerfModel, ForwardPassPerfReadiness,
    ForwardPassPerfSource,
};
pub use options::ForwardPassPerfOptions;
