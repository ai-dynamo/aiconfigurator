// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Crate-wide error type.
//!
//! `lib.rs` re-exports `AicError` so the FFI surface and other callers can
//! use it without depending on this module path. The operator-layer
//! `PerformanceResult` companion type lives in `operators/base.rs`.

use std::path::PathBuf;

use thiserror::Error;

/// All errors surfaced by the Rust core.
#[derive(Debug, Error)]
pub enum AicError {
    #[error("unsupported schema version for {kind}: got {got}, expected {expected}")]
    UnsupportedSchemaVersion {
        kind: &'static str,
        got: u32,
        expected: u32,
    },
    #[error("invalid engine config: {0}")]
    InvalidEngineConfig(String),
    #[error("engine spec wire-format error: {0}")]
    EngineSpec(String),
    #[error("invalid forward pass metrics: {0}")]
    InvalidForwardPassMetrics(String),
    #[error("unsupported model for Rust core estimator: {0}")]
    UnsupportedModel(String),
    #[error("failed to find AIC data roots: {0}")]
    DataRoot(String),
    #[error("model config error: {0}")]
    ModelConfig(String),
    #[error("perf database error: {0}")]
    PerfDatabase(String),
    /// The HYBRID/EMPIRICAL path found no calibration data for the requested
    /// slice — no own-shape, cross-shape, or sibling transfer reference.
    /// Mirrors Python's `EmpiricalNotImplementedError`: a coverage gap, never
    /// converted into a fabricated `SOL / constant` value.
    #[error("empirical estimation not implemented: {0}")]
    EmpiricalNotImplemented(String),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("YAML error at {path}: {source}")]
    Yaml {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
    #[error("Parquet error at {path}: {source}")]
    Parquet {
        path: PathBuf,
        #[source]
        source: parquet::errors::ParquetError,
    },
}

impl AicError {
    /// The "missing silicon data" class that HYBRID converts into an
    /// empirical estimate. Mirrors Python's
    /// `_MISSING_SILICON_DATA_EXCEPTIONS` (`PerfDataNotAvailableError`,
    /// `InterpolationDataNotAvailableError`) — the same set `Op::Fallback`
    /// catches. `EmpiricalNotImplemented` is deliberately NOT in this set:
    /// it is the terminal miss raised after the empirical path itself found
    /// no calibration data.
    pub fn is_missing_perf_data(&self) -> bool {
        matches!(self, Self::PerfDatabase(_) | Self::Io { .. })
    }
}

