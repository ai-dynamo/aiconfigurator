// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared foundation types used by every other module in the crate.
//!
//! Contents have no AIC-domain knowledge of their own (no model graphs, no
//! perf-DB queries, no backends). Domain modules import from here; nothing
//! here imports from domain modules.

pub mod enums;
pub mod error;
pub mod system_spec;

pub use error::AicError;
