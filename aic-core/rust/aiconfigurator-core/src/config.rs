// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public wire/identity configuration types carried by an
//! [`crate::engine::spec::EngineSpec`]. [`EngineConfig`] and its cohesive
//! sub-structs ([`ParallelMapping`], [`QuantizationConfig`],
//! [`SpeculativeConfig`]) mirror the flat JSON object emitted by Python's
//! `compile_engine` (`sdk/engine.py`); [`BackendKind`] and [`DataType`] are
//! the wire enums those structs reference. These are re-exported at the crate
//! root, so `crate::EngineConfig`, `crate::BackendKind`, ... resolve unchanged.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const ENGINE_CONFIG_SCHEMA_VERSION: u32 = 1;
// bincode op payloads are positional, so a producer/consumer skew is only
// distinguishable by this version — `EngineSpec::from_bincode` reads and
// checks it before decoding the op lists. Bump whenever an `OpSpec` field
// changes; keep in lockstep with `sdk/engine.py::ENGINE_SPEC_SCHEMA_VERSION`.
// History:
// - 2 (v0.10.0): op-payload layout change — the context-parallelism +
//   perf-DB refactor added serialized fields such as `seq_split` /
//   `cp_size` to `OpSpec`.
// - 3 (PR #1405): MTP acceptance moved above aic-core —
//   `nextn_accept_rates` removed from the spec payload.
// - 4 (PR #1355): `Msa{Context,Generation}` variants inserted (bincode enum
//   indices after `DsaGeneration` shifted). The MSA insertion and #1405
//   each claimed version 3 on their own branch, so their merge needed a
//   fresh number.
pub const ENGINE_SPEC_SCHEMA_VERSION: u32 = 4;

/// Static engine identity and setup information carried by an
/// [`crate::engine::spec::EngineSpec`].
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

    /// Shared-layer (sibling/cross-version) perf-data sources per op-file
    /// basename (e.g. `gemm_perf.parquet`), resolved in Python
    /// (`sdk/engine.py::_compute_perf_db_sources`) so the Rust core inherits the
    /// SAME rows Python does under SILICON/HYBRID. Each entry is
    /// `(abs_path, Option<kernel_source_allowlist>)` in priority order — the
    /// first source containing a shape wins (mirrors Python
    /// `_read_filtered_rows` + skip-on-key-conflict). Absent/empty = fall back
    /// to the single primary `data_root` (back-compat with pre-shared-layer
    /// specs).
    #[serde(default)]
    pub perf_db_sources: PerfDbSources,

    /// Perf-database lookup mode (Python's `database._default_database_mode`).
    /// SILICON queries collected tables only; HYBRID falls back to the
    /// util-space empirical layer on a typed silicon miss; EMPIRICAL always
    /// answers `SOL/util`. Absent on old specs -> Silicon (back-compat).
    #[serde(default)]
    pub database_mode: crate::common::enums::DatabaseMode,

    /// Enabled empirical transfer kinds as explicit tokens (`xshape` /
    /// `xquant` / `xprofile` / `xop`). Python resolves preset names before
    /// serialising, so no preset vocabulary exists on the wire. `None` =
    /// the default ALL-transfers policy (mirrors `common.ALL_TRANSFERS`).
    #[serde(default)]
    pub transfer_policy: Option<Vec<String>>,

    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

/// Per-op-file ordered source list, keyed by op-file basename. See
/// [`EngineConfig::perf_db_sources`].
pub type PerfDbSources = BTreeMap<String, Vec<PerfSource>>;

/// One perf-data source: an absolute file path plus an optional
/// `kernel_source` allowlist. `None` admits every row (the primary source);
/// `Some(set)` keeps only rows whose `kernel_source` is in the set (sibling
/// inheritance). Wire form is a 2-element JSON array `[path, [ks...] | null]`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PerfSource(pub PathBuf, pub Option<Vec<String>>);

impl PerfSource {
    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
    pub fn kernel_sources(&self) -> Option<&[String]> {
        self.1.as_deref()
    }
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
    /// Context-parallel size. Part of the engine identity so cp variants get
    /// distinct compiled handles. `None`/1 means no CP. The per-op CP math is
    /// carried on the ops themselves (seq_split / cp_size / attn_cp_size), not
    /// re-derived from this field.
    #[serde(default)]
    pub cp_size: Option<u32>,
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
/// noise, and `#[serde(flatten)]`-ed so the flat wire key (`nextn`) parses
/// unchanged. Accepted-token progress is modeled above `aic-core`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeculativeConfig {
    /// Multi-Token Prediction speculative decoding depth / draft length
    /// (Python's `task_config.nextn`). `None`/0 disables MTP scaling. MTP is
    /// never auto-enabled; the user opts in explicitly.
    #[serde(default)]
    pub nextn: Option<u32>,
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
    #[serde(rename = "w4a8_mxfp4_mxfp8_trtllm")]
    W4a8Mxfp4Mxfp8Trtllm,
    #[serde(rename = "w4a16_mxfp4_cutlass")]
    W4a16Mxfp4Cutlass,
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
