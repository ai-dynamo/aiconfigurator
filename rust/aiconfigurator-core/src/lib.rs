// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native core latency API for AIConfigurator.
//!
//! The compiled-engine path is the only supported entry point: Python's
//! `compile_engine` walks the model once and emits an [`engine::spec::EngineSpec`]
//! (op lists + [`EngineConfig`] identity); the Rust [`engine::Engine`] executes
//! it without re-entering Python. [`build_aic_engine`] is the Rust → Python →
//! Rust embedded build entry point for callers in other crates (the Dynamo
//! Mocker, `tests/embedded_round_trip.rs`); [`AicEngine`] is the PyO3 hot-path
//! pyclass.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

mod py;

// Modular core. `common/` holds shared foundation types (enums, error,
// system_spec) with no AIC-domain knowledge. Top-level files (`config`,
// `session`) and directories (`operators`, `perf_database`) carry the domain
// logic the compiled `engine` executes.
mod common;
mod config;
pub mod engine;
mod interpolation;
mod operators;
mod perf_database;
mod session;

pub use common::AicError;
// PyO3 bindings (E4). `AicEngine` is the Python -> Rust hot-path pyclass;
// `build_aic_engine` is the Rust -> Python -> Rust embedded build entry point
// for callers in OTHER crates (the Dynamo Mocker, E7's
// `tests/embedded_round_trip.rs`). They must be `pub`-re-exported here because
// the `py` module itself is private.
pub use py::{build_aic_engine, AicEngine};

pub const ENGINE_CONFIG_SCHEMA_VERSION: u32 = 1;
pub const ENGINE_SPEC_SCHEMA_VERSION: u32 = 1;

/// Static engine identity and setup information carried by an
/// [`engine::spec::EngineSpec`].
///
/// Cohesive multi-field groupings (`parallel`, `quantization`,
/// `speculative`) are extracted into sub-structs but `#[serde(flatten)]`-ed
/// so the wire JSON stays flat. Python (`sdk/engine.py`) emits a flat object
/// with keys like `tp_size`, `weight_dtype`, `nextn`, which deserialize into
/// the regrouped struct unchanged.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EngineConfig {
    pub schema_version: u32,

    // Model
    pub model_name: String,

    // System
    pub system_name: String,
    /// Optional override for the bundled `systems/` directory. `None` (the
    /// default) uses the resolution path baked into the build/env.
    #[serde(default)]
    pub systems_path: Option<PathBuf>,

    // Backend
    pub backend: BackendKind,
    pub backend_version: Option<String>,

    // KV
    pub kv_block_size: Option<u32>,

    // Cohesive groupings (multi-field, semantically coupled).
    #[serde(flatten)]
    pub parallel: ParallelMapping,
    #[serde(flatten)]
    pub quantization: QuantizationConfig,
    #[serde(flatten)]
    pub speculative: Option<SpeculativeConfig>,

    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

/// Parallelism layout. Flattened into [`EngineConfig`] so the flat wire keys
/// (`tp_size`, `pp_size`, ...) parse unchanged.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ParallelMapping {
    pub tp_size: u32,
    pub pp_size: u32,
    #[serde(default)]
    pub attention_dp_size: Option<u32>,
    #[serde(default)]
    pub moe_tp_size: Option<u32>,
    #[serde(default)]
    pub moe_ep_size: Option<u32>,
}

/// Precision/quantization dtypes. Flattened into [`EngineConfig`]. Field
/// names and types are unchanged from the former flat struct so the flat
/// wire keys (`weight_dtype`, `moe_dtype`, ...) parse unchanged.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct QuantizationConfig {
    pub weight_dtype: Option<DataType>,
    #[serde(default)]
    pub moe_dtype: Option<DataType>,
    pub activation_dtype: Option<DataType>,
    pub kv_cache_dtype: Option<DataType>,
}

/// Multi-Token Prediction speculative-decoding parameters. Wrapped in
/// `Option<>` on [`EngineConfig`] so models without MTP don't carry the
/// noise, and `#[serde(flatten)]`-ed so the flat wire keys (`nextn`,
/// `nextn_accept_rates`) parse unchanged.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeculativeConfig {
    /// Multi-Token Prediction speculative decoding depth (Python's
    /// `task_config.nextn`). `None`/0 disables MTP scaling. Python sets this
    /// to 1 for DeepSeek-family + Qwen3.5 models (`sdk/task.py:448-449`);
    /// other families leave it at 0/None.
    #[serde(default)]
    pub nextn: Option<u32>,

    /// Per-step accept-rate prior used by MTP scaling. Mirrors Python's
    /// `task_config.nextn_accept_rates` (default `[0.85, 0.3, 0.0, 0.0,
    /// 0.0]`). Ignored when `nextn` is `None` or 0.
    #[serde(default)]
    pub nextn_accept_rates: Option<Vec<f64>>,
}

/// Backend performance database family.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Trtllm,
    Sglang,
    Vllm,
}

impl BackendKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Trtllm => "trtllm",
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
        }
    }
}

/// Precision/quantization dtypes carried on the engine-config wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    #[serde(rename = "bfloat16")]
    Bfloat16,
    #[serde(rename = "float16")]
    Float16,
    #[serde(rename = "fp8")]
    Fp8,
    #[serde(rename = "fp8_static")]
    Fp8Static,
    #[serde(rename = "fp8_block")]
    Fp8Block,
    #[serde(rename = "nvfp4")]
    Nvfp4,
    #[serde(rename = "int8")]
    Int8,
    #[serde(rename = "int4")]
    Int4,
    #[serde(rename = "w4afp8")]
    W4afp8,
    #[serde(rename = "w4a16_mxfp4")]
    W4a16Mxfp4,
    #[serde(rename = "w4a8_mxfp4_mxfp8")]
    W4a8Mxfp4Mxfp8,
}

/// Resolve a repo-relative path by walking up from the crate manifest dir.
/// Used by [`py`] (e.g. `crate::repo_relative("src/aiconfigurator/systems")`)
/// to locate the bundled data roots when developing in-tree.
pub(crate) fn repo_relative(rel: &str) -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(test)]
mod engine_config_wire_tests {
    use super::*;

    /// Python's `compile_engine` (`sdk/engine.py`) emits a flat JSON object.
    /// The regrouped `EngineConfig` uses `#[serde(flatten)]` to keep that wire
    /// contract. This guards that the flat shape - including explicit nulls and
    /// the now-dropped `model_arch` key - still deserializes into the nested
    /// struct.
    #[test]
    fn flat_python_payload_deserializes_into_regrouped_config() {
        let json = r#"{
            "schema_version": 1,
            "model_name": "Qwen/Qwen3-32B",
            "model_arch": "Qwen3ForCausalLM",
            "system_name": "h200_sxm",
            "backend": "trtllm",
            "backend_version": "1.0.0",
            "tp_size": 2,
            "pp_size": 1,
            "moe_tp_size": null,
            "moe_ep_size": null,
            "attention_dp_size": null,
            "weight_dtype": "bfloat16",
            "moe_dtype": null,
            "activation_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "kv_block_size": null,
            "nextn": null,
            "nextn_accept_rates": null,
            "extra": {}
        }"#;

        let config: EngineConfig = serde_json::from_str(json).expect("flat payload must parse");

        // Parallelism regrouping.
        assert_eq!(config.parallel.tp_size, 2);
        assert_eq!(config.parallel.pp_size, 1);
        assert_eq!(config.parallel.attention_dp_size, None);
        assert_eq!(config.parallel.moe_tp_size, None);
        assert_eq!(config.parallel.moe_ep_size, None);

        // Quantization regrouping.
        assert_eq!(config.quantization.weight_dtype, Some(DataType::Bfloat16));
        assert_eq!(config.quantization.moe_dtype, None);

        // Speculative: Python always emits the `nextn` key, so the flattened
        // option is `Some` with inner `None` (MTP disabled), not `None`.
        let nextn = config.speculative.as_ref().and_then(|s| s.nextn);
        assert_eq!(nextn, None);

        // `model_arch` was dropped; the stray key must be ignored, not rejected.
        assert!(!config.extra.contains_key("model_arch"));
        assert_eq!(config.model_name, "Qwen/Qwen3-32B");

        // `systems_path` is new and defaults to None when absent.
        assert_eq!(config.systems_path, None);
    }
}
