// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend abstraction.
//!
//! Mirrors `aiconfigurator.sdk.backends.base_backend.BaseBackend` at a minimal
//! level: backend-specific defaults for ops that don't have an explicit
//! value, plus dispatch hooks for the MoE workload-distribution decision
//! and memory accounting. Backend-specific behavior in this Rust port stays
//! intentionally thin — model graphs read backend hints from the
//! `BackendOptions` struct rather than dispatching through trait virtuals.

use crate::common::enums::{BackendKind, GemmQuantMode};

/// Backend-specific defaults read by model graphs at construction time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendOptions {
    pub kind: BackendKind,
    /// Whether the backend enables QK-norm in attention by default.
    pub default_use_qk_norm: bool,
    /// Whether the backend supports overlapping collective with compute
    /// (sets the `overlap_groups` used by the overlap operator).
    pub supports_overlap: bool,
    /// Default GEMM quant override applied to weight matrices when the
    /// model config doesn't specify one. Typically matches the backend's
    /// most common collected quant.
    pub default_weight_quant: Option<GemmQuantMode>,
}

impl BackendOptions {
    pub fn vllm() -> Self {
        crate::backends::vllm::vllm_defaults()
    }

    pub fn sglang() -> Self {
        Self {
            kind: BackendKind::Sglang,
            default_use_qk_norm: false,
            supports_overlap: true,
            default_weight_quant: None,
        }
    }

    pub fn trtllm() -> Self {
        Self {
            kind: BackendKind::Trtllm,
            default_use_qk_norm: false,
            supports_overlap: true,
            default_weight_quant: None,
        }
    }
}
