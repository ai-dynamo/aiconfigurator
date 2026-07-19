// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared grid type alias for the per-family loaders.
//!
//! All interpolation itself lives in [`super::perf_interp`] (the v2 resolver
//! engine, mirroring `sdk/perf_interp/engine.py`). The scalar interp family
//! that used to live here (`nearest_neighbors`, `interp_1d`, `bilinear`, the
//! `interp_2d_1d_grid*` ladder, `interp_context_topk_piecewise`) mirrored the
//! deleted Python `sdk/interpolation.py` and lost its last consumer when the
//! per-family queries moved onto the engine.

use std::collections::BTreeMap;

/// 3-level nested grid used by loaders while assembling tables:
/// `axis0 -> axis1 -> axis2 -> value`.
pub type Grid3<T> = BTreeMap<u32, BTreeMap<u32, BTreeMap<u32, T>>>;
