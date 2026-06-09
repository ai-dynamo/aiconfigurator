// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiled-engine layer.
//!
//! Holds the serializable wire format ([`spec::EngineSpec`]) that Python's
//! `compile_engine` emits and the Rust runner consumes ([`spec`]), plus the
//! runtime [`Engine`] that builds from an `EngineSpec` and executes the
//! static-inference composition. PyO3 bindings over `Engine` live in
//! [`crate::py`].

pub mod runtime;
pub mod spec;

pub use runtime::{Engine, RuntimeConfig, StaticMode, StaticResult, DEFAULT_STATIC_STRIDE};
