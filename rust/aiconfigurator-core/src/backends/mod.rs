// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-specific quirks: `base` defines the shared `BackendOptions`
//! struct; per-backend submodules provide construction defaults.
//!
//! The Rust port keeps the backend layer intentionally thin — model graphs
//! read defaults from `BackendOptions` at construction time rather than
//! dispatching through a trait at query time.

pub mod base;
pub mod vllm;

pub use base::BackendOptions;
