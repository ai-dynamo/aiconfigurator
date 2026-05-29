// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-family model op-graph builders. `base` defines the `ModelSpec`
//! abstraction and KV-cache sizing; `registry` + `config_loader` map HF
//! `config.json` architectures to families; family modules
//! (`hybrid_moe`, `deepseek`, ...) build the operator sequence for one
//! model family on one backend. `ModelSpec` here is the source of truth
//! for the engine-step path.

pub mod base;
pub mod config_loader;
pub mod deepseek;
pub mod deepseek_v32;
pub mod deepseek_v4;
pub mod deepseek_wideep;
pub mod deepseek_wideep_trtllm;
pub mod factory;
pub mod gemma4_moe;
pub mod gpt;
pub mod hybrid_moe;
pub mod llama;
pub mod moe;
pub mod nemotron_h;
pub mod nemotron_nas;
pub mod qwen35;
pub mod qwen3vl;
pub mod registry;

pub use base::{
    DtypeConfig, Model, ModelConfig, ModelSpec, ParallelConfig, WorkloadDistribution,
};
pub use factory::build_model;
