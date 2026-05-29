// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM backend defaults.
//!
//! Mirrors `aiconfigurator.sdk.backends.vllm_backend`. vLLM's default
//! behavior differs from TRT-LLM/SGLang in a few places that matter for
//! op-graph construction: it doesn't enable QK-norm by default and uses
//! flashinfer kernels which have different memory characteristics.

use crate::backends::base::BackendOptions;
use crate::common::enums::BackendKind;

pub fn vllm_defaults() -> BackendOptions {
    BackendOptions {
        kind: BackendKind::Vllm,
        default_use_qk_norm: false,
        supports_overlap: false,
        default_weight_quant: None,
    }
}
