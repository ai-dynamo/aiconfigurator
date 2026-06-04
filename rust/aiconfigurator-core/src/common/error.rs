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
    #[error("invalid engine config: {0}")]
    InvalidEngineConfig(String),
    #[error("engine spec wire-format error: {0}")]
    EngineSpec(String),
    #[error("failed to find AIC data roots: {0}")]
    DataRoot(String),
    #[error("perf database error: {0}")]
    PerfDatabase(String),
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

