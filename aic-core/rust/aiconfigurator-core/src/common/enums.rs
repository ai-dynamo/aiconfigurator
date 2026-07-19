// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend, database-mode, quant-mode, model-family, and perf-data-filename
//! enums plus the `QuantMapping` payload shared by the quant-mode enums.
//! Mirrors the enum surface of `src/aiconfigurator/sdk/common.py`.
//!
//! HF-architecture-to-family routing lives in `models/registry.rs`, not
//! here, because it carries AIC-specific lookup tables that depend on
//! `ModelFamily`.
//!
//! The FFI surface in `lib.rs` exposes a smaller `BackendKind` / `DataType`
//! pair shaped for JSON wire compatibility; the richer enums defined here
//! are used internally by the session pipeline.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Inference backend.
///
/// Mirrors `common.BackendName`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Trtllm,
    Sglang,
    Vllm,
}

impl BackendKind {
    /// String identifier used as a directory key under `systems/data/<system>/<backend>/`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Trtllm => "trtllm",
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Performance database lookup mode.
///
/// Mirrors `common.DatabaseMode`. SILICON is the only mode currently
/// active in the engine-step path; the other variants exist so the schema
/// does not regress when they are wired in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DatabaseMode {
    Silicon,
    Hybrid,
    Empirical,
    Sol,
    SolFull,
}

impl Default for DatabaseMode {
    fn default() -> Self {
        Self::Silicon
    }
}

/// Per-variant payload mirroring Python's
/// `QuantMapping = namedtuple("QuantMapping", ["memory", "compute", "name"])`.
///
/// `memory` is the per-element byte cost relative to bf16 (1.0 = same, 0.5 =
/// half). `compute` is the TC-FLOPS multiplier relative to bf16. `name` is the
/// stable string identifier used as a perf-DB column key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantMapping {
    pub memory: f64,
    pub compute: f64,
    pub name: &'static str,
}

/// GEMM quantization mode. Mirrors `common.GEMMQuantMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmQuantMode {
    Bfloat16,
    Int8Wo,
    Int4Wo,
    Fp8,
    Fp8Static,
    Sq,
    Fp8Block,
    Fp8Ootb,
    Nvfp4,
}

impl GemmQuantMode {
    pub fn mapping(self) -> QuantMapping {
        match self {
            Self::Bfloat16 => QuantMapping { memory: 2.0, compute: 1.0, name: "bfloat16" },
            Self::Int8Wo => QuantMapping { memory: 1.0, compute: 1.0, name: "int8_wo" },
            Self::Int4Wo => QuantMapping { memory: 0.5, compute: 1.0, name: "int4_wo" },
            Self::Fp8 => QuantMapping { memory: 1.0, compute: 2.0, name: "fp8" },
            Self::Fp8Static => QuantMapping { memory: 1.0, compute: 2.0, name: "fp8_static" },
            Self::Sq => QuantMapping { memory: 1.0, compute: 2.0, name: "sq" },
            Self::Fp8Block => QuantMapping { memory: 1.0, compute: 2.0, name: "fp8_block" },
            Self::Fp8Ootb => QuantMapping { memory: 1.0, compute: 2.0, name: "fp8_ootb" },
            Self::Nvfp4 => QuantMapping { memory: 9.0 / 16.0, compute: 4.0, name: "nvfp4" },
        }
    }

    pub fn name(self) -> &'static str {
        self.mapping().name
    }
}

/// MoE quantization mode. Mirrors `common.MoEQuantMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeQuantMode {
    Bfloat16,
    Fp8,
    Int4Wo,
    Fp8Block,
    W4afp8,
    Nvfp4,
    W4a16Mxfp4,
    W4a8Mxfp4Mxfp8,
}

impl MoeQuantMode {
    pub fn mapping(self) -> QuantMapping {
        match self {
            Self::Bfloat16 => QuantMapping { memory: 2.0, compute: 1.0, name: "bfloat16" },
            Self::Fp8 => QuantMapping { memory: 1.0, compute: 2.0, name: "fp8" },
            Self::Int4Wo => QuantMapping { memory: 0.5, compute: 1.0, name: "int4_wo" },
            Self::Fp8Block => QuantMapping { memory: 1.0, compute: 2.0, name: "fp8_block" },
            Self::W4afp8 => QuantMapping { memory: 0.5, compute: 2.0, name: "w4afp8" },
            Self::Nvfp4 => QuantMapping { memory: 9.0 / 16.0, compute: 4.0, name: "nvfp4" },
            Self::W4a16Mxfp4 => QuantMapping { memory: 0.5, compute: 1.0, name: "w4a16_mxfp4" },
            Self::W4a8Mxfp4Mxfp8 => {
                QuantMapping { memory: 0.5, compute: 2.0, name: "w4a8_mxfp4_mxfp8" }
            }
        }
    }

    pub fn name(self) -> &'static str {
        self.mapping().name
    }
}

/// FMHA (fused multi-head attention) quantization mode. Mirrors `common.FMHAQuantMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FmhaQuantMode {
    Bfloat16,
    Fp8,
    Fp8Block,
}

impl FmhaQuantMode {
    pub fn mapping(self) -> QuantMapping {
        match self {
            Self::Bfloat16 => QuantMapping { memory: 2.0, compute: 1.0, name: "bfloat16" },
            Self::Fp8 => QuantMapping { memory: 1.0, compute: 2.0, name: "fp8" },
            Self::Fp8Block => QuantMapping { memory: 1.0, compute: 2.0, name: "fp8_block" },
        }
    }

    pub fn name(self) -> &'static str {
        self.mapping().name
    }
}

/// KV cache quantization mode. Mirrors `common.KVCacheQuantMode`. Compute
/// factor is always 0 because KV cache is a memory-only consideration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheQuantMode {
    Bfloat16,
    Int8,
    Fp8,
}

impl KvCacheQuantMode {
    pub fn mapping(self) -> QuantMapping {
        match self {
            Self::Bfloat16 => QuantMapping { memory: 2.0, compute: 0.0, name: "bfloat16" },
            Self::Int8 => QuantMapping { memory: 1.0, compute: 0.0, name: "int8" },
            Self::Fp8 => QuantMapping { memory: 1.0, compute: 0.0, name: "fp8" },
        }
    }

    pub fn name(self) -> &'static str {
        self.mapping().name
    }
}

/// Collective communication quantization mode. Mirrors `common.CommQuantMode`.
/// Only the `half` variant is present in Python today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommQuantMode {
    Half,
}

impl CommQuantMode {
    pub fn mapping(self) -> QuantMapping {
        match self {
            Self::Half => QuantMapping { memory: 2.0, compute: 0.0, name: "half" },
        }
    }

    pub fn name(self) -> &'static str {
        self.mapping().name
    }
}

/// AIC model family. Mirrors the values in `common.ModelFamily`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelFamily {
    Gpt,
    Llama,
    Moe,
    DeepSeek,
    DeepSeekV32,
    DeepSeekV4,
    KimiK25,
    NemotronNas,
    NemotronH,
    HybridMoe,
    Qwen35,
    Gemma4Moe,
    Qwen3Vl,
    Qwen3VlMoe,
}

impl ModelFamily {
    /// String identifier matching Python's family name (used as a routing key
    /// and as a perf-DB metadata field).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gpt => "GPT",
            Self::Llama => "LLAMA",
            Self::Moe => "MOE",
            Self::DeepSeek => "DEEPSEEK",
            Self::DeepSeekV32 => "DEEPSEEKV32",
            Self::DeepSeekV4 => "DEEPSEEKV4",
            Self::KimiK25 => "KIMIK25",
            Self::NemotronNas => "NEMOTRONNAS",
            Self::NemotronH => "NEMOTRONH",
            Self::HybridMoe => "HYBRIDMOE",
            Self::Qwen35 => "QWEN35",
            Self::Gemma4Moe => "GEMMA4MOE",
            Self::Qwen3Vl => "QWEN3VL",
            Self::Qwen3VlMoe => "QWEN3VL_MOE",
        }
    }
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Performance data CSV filenames. Mirrors `common.PerfDataFilename`.
///
/// Used by `perf_database/` modules to locate per-table CSV files under
/// `systems/data/<system>/<backend>/<version>/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerfDataFilename {
    Gemm,
    Nccl,
    Oneccl,
    GenerationAttention,
    ContextAttention,
    EncoderAttention,
    ContextMla,
    GenerationMla,
    MlaBmm,
    Moe,
    CustomAllreduce,
    WideepContextMla,
    WideepGenerationMla,
    WideepContextMoe,
    WideepGenerationMoe,
    WideepDeepepNormal,
    WideepDeepepLl,
    WideepMoeCompute,
    TrtllmAlltoall,
    ComputeScale,
    ScaleMatrix,
    Mamba2,
    Gdn,
    MlaContextModule,
    MlaGenerationModule,
    DsaContextModule,
    DsaGenerationModule,
    MhcModule,
    Dsv4CsaContextModule,
    Dsv4HcaContextModule,
    Dsv4CsaGenerationModule,
    Dsv4HcaGenerationModule,
    Dsv4PagedMqaLogitsModule,
    Dsv4HcaAttnModule,
}

impl PerfDataFilename {
    /// Stable CSV basename used by the perf DB loaders. Matches Python's
    /// `PerfDataFilename` enum values.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gemm => "gemm_perf.txt",
            Self::Nccl => "nccl_perf.txt",
            Self::Oneccl => "oneccl_perf.txt",
            Self::GenerationAttention => "generation_attention_perf.txt",
            Self::ContextAttention => "context_attention_perf.txt",
            Self::EncoderAttention => "encoder_attention_perf.txt",
            Self::ContextMla => "context_mla_perf.txt",
            Self::GenerationMla => "generation_mla_perf.txt",
            Self::MlaBmm => "mla_bmm_perf.txt",
            Self::Moe => "moe_perf.txt",
            Self::CustomAllreduce => "custom_allreduce_perf.txt",
            Self::WideepContextMla => "wideep_context_mla_perf.txt",
            Self::WideepGenerationMla => "wideep_generation_mla_perf.txt",
            Self::WideepContextMoe => "wideep_context_moe_perf.txt",
            Self::WideepGenerationMoe => "wideep_generation_moe_perf.txt",
            Self::WideepDeepepNormal => "wideep_deepep_normal_perf.txt",
            Self::WideepDeepepLl => "wideep_deepep_ll_perf.txt",
            Self::WideepMoeCompute => "wideep_moe_perf.txt",
            Self::TrtllmAlltoall => "trtllm_alltoall_perf.txt",
            Self::ComputeScale => "computescale_perf.txt",
            Self::ScaleMatrix => "scale_matrix_perf.txt",
            Self::Mamba2 => "mamba2_perf.txt",
            Self::Gdn => "gdn_perf.txt",
            Self::MlaContextModule => "mla_context_module_perf.txt",
            Self::MlaGenerationModule => "mla_generation_module_perf.txt",
            Self::DsaContextModule => "dsa_context_module_perf.txt",
            Self::DsaGenerationModule => "dsa_generation_module_perf.txt",
            Self::MhcModule => "mhc_module_perf.txt",
            Self::Dsv4CsaContextModule => "dsv4_csa_context_module_perf.txt",
            Self::Dsv4HcaContextModule => "dsv4_hca_context_module_perf.txt",
            Self::Dsv4CsaGenerationModule => "dsv4_csa_generation_module_perf.txt",
            Self::Dsv4HcaGenerationModule => "dsv4_hca_generation_module_perf.txt",
            Self::Dsv4PagedMqaLogitsModule => "dsv4_paged_mqa_logits_module_perf.txt",
            Self::Dsv4HcaAttnModule => "dsv4_hca_attn_module_perf.txt",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_string_keys_match_python() {
        assert_eq!(BackendKind::Trtllm.as_str(), "trtllm");
        assert_eq!(BackendKind::Sglang.as_str(), "sglang");
        assert_eq!(BackendKind::Vllm.as_str(), "vllm");
    }

    #[test]
    fn database_mode_default_is_silicon() {
        assert_eq!(DatabaseMode::default(), DatabaseMode::Silicon);
    }

    #[test]
    fn gemm_quant_payloads_match_python_quant_mapping() {
        // Sampled from common.GEMMQuantMode.
        assert_eq!(
            GemmQuantMode::Bfloat16.mapping(),
            QuantMapping { memory: 2.0, compute: 1.0, name: "bfloat16" }
        );
        assert_eq!(
            GemmQuantMode::Fp8.mapping(),
            QuantMapping { memory: 1.0, compute: 2.0, name: "fp8" }
        );
        assert_eq!(
            GemmQuantMode::Nvfp4.mapping(),
            QuantMapping { memory: 9.0 / 16.0, compute: 4.0, name: "nvfp4" }
        );
    }

    #[test]
    fn moe_quant_payloads_match_python_quant_mapping() {
        assert_eq!(
            MoeQuantMode::W4afp8.mapping(),
            QuantMapping { memory: 0.5, compute: 2.0, name: "w4afp8" }
        );
        assert_eq!(
            MoeQuantMode::W4a8Mxfp4Mxfp8.mapping(),
            QuantMapping { memory: 0.5, compute: 2.0, name: "w4a8_mxfp4_mxfp8" }
        );
    }

    #[test]
    fn kvcache_compute_is_zero() {
        for mode in [KvCacheQuantMode::Bfloat16, KvCacheQuantMode::Int8, KvCacheQuantMode::Fp8] {
            assert_eq!(mode.mapping().compute, 0.0);
        }
    }

    #[test]
    fn model_family_string_round_trip() {
        // Spot-check Python-faithful family names.
        assert_eq!(ModelFamily::Qwen3VlMoe.as_str(), "QWEN3VL_MOE");
        assert_eq!(ModelFamily::DeepSeekV32.as_str(), "DEEPSEEKV32");
        assert_eq!(ModelFamily::KimiK25.as_str(), "KIMIK25");
    }

    #[test]
    fn perf_data_filenames_match_python_enum() {
        // Sampled across categories from common.PerfDataFilename.
        assert_eq!(PerfDataFilename::Gemm.as_str(), "gemm_perf.txt");
        assert_eq!(
            PerfDataFilename::ContextAttention.as_str(),
            "context_attention_perf.txt"
        );
        assert_eq!(PerfDataFilename::CustomAllreduce.as_str(), "custom_allreduce_perf.txt");
        assert_eq!(PerfDataFilename::WideepDeepepLl.as_str(), "wideep_deepep_ll_perf.txt");
        assert_eq!(PerfDataFilename::TrtllmAlltoall.as_str(), "trtllm_alltoall_perf.txt");
        assert_eq!(
            PerfDataFilename::Dsv4HcaGenerationModule.as_str(),
            "dsv4_hca_generation_module_perf.txt"
        );
    }
}
