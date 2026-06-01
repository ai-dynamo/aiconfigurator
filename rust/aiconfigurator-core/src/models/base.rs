// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `ModelSpec`, `ParallelConfig`, `DtypeConfig`, and the central `Model`
//! struct that holds typed op lists for context / generation / encoder
//! phases.
//!
//! Mirrors Python's `aiconfigurator.sdk.models.base.BaseModel` shape: a
//! model is a config + three pre-built op lists. The session driver
//! iterates the lists; the model itself has no orchestration logic.

use crate::common::enums::{
    BackendKind, CommQuantMode, FmhaQuantMode, GemmQuantMode, KvCacheQuantMode, ModelFamily,
    MoeQuantMode,
};
use crate::common::system_spec::SystemSpec;
use crate::models::config_loader::{
    BlockConfig, DeepSeekV4Config, Gemma4MoEConfig, HfModelConfig, HybridMoEConfig,
    NemotronHConfig, Qwen35Config, VisionEncoderConfig,
};
use crate::operators::Op;

/// HF-config-derived dimensions for a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelSpec {
    pub architecture: String,
    pub family: ModelFamily,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub top_k: u32,
    pub num_experts: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
    pub use_qk_norm: bool,
    /// NemotronNas per-layer block configs (empty for all other families).
    pub block_configs: Vec<BlockConfig>,
    /// Qwen3.5 hybrid (GDN + full-attention) extras. `None` for non-Qwen3.5.
    pub qwen35: Option<Qwen35Config>,
    /// NemotronH hybrid (Mamba + attention + MLP/MoE) extras. `None`
    /// for non-NemotronH families.
    pub nemotron_h: Option<NemotronHConfig>,
    /// DeepSeek-V4 compressed-attention + mHC extras. `None` for non-DSv4
    /// families.
    pub deepseek_v4: Option<DeepSeekV4Config>,
    /// HybridMoE per-layer pattern + SWA dims. `None` for non-HybridMoE
    /// families.
    pub hybrid_moe: Option<HybridMoEConfig>,
    /// Gemma-4 hybrid SWA/global + shared-MLP-plus-MoE extras. `None`
    /// for non-Gemma4 families.
    pub gemma4_moe: Option<Gemma4MoEConfig>,
    /// Vision encoder config for multimodal VL models. `None` for
    /// text-only families.
    pub vision_encoder: Option<VisionEncoderConfig>,
}

impl From<HfModelConfig> for ModelSpec {
    fn from(cfg: HfModelConfig) -> Self {
        Self {
            architecture: cfg.architecture,
            family: cfg.family,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            context_length: cfg.context_length,
            top_k: cfg.top_k,
            num_experts: cfg.num_experts,
            moe_intermediate_size: cfg.moe_intermediate_size,
            shared_expert_intermediate_size: cfg.shared_expert_intermediate_size,
            use_qk_norm: cfg.use_qk_norm,
            block_configs: cfg.block_configs,
            qwen35: cfg.qwen35,
            nemotron_h: cfg.nemotron_h,
            deepseek_v4: cfg.deepseek_v4,
            hybrid_moe: cfg.hybrid_moe,
            gemma4_moe: cfg.gemma4_moe,
            vision_encoder: cfg.vision_encoder,
        }
    }
}

impl ModelSpec {
    pub fn kv_heads_per_gpu(&self, tp_size: u32) -> u32 {
        let tp = tp_size.max(1);
        (self.num_key_value_heads + tp - 1) / tp
    }

    pub fn uses_moe(&self) -> bool {
        matches!(
            self.family,
            ModelFamily::Moe
                | ModelFamily::DeepSeek
                | ModelFamily::DeepSeekV32
                | ModelFamily::DeepSeekV4
                | ModelFamily::KimiK25
                | ModelFamily::HybridMoe
                | ModelFamily::Gemma4Moe
                | ModelFamily::Qwen3VlMoe
        ) || (matches!(self.family, ModelFamily::NemotronH | ModelFamily::Qwen35)
            && self.num_experts > 0
            && self.top_k > 0)
    }

    pub fn uses_mla_attention(&self) -> bool {
        matches!(
            self.family,
            ModelFamily::DeepSeek | ModelFamily::DeepSeekV32 | ModelFamily::KimiK25
        )
    }
}

/// Parallelism configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelConfig {
    pub tp_size: u32,
    pub pp_size: u32,
    pub attention_dp_size: u32,
    pub moe_tp_size: u32,
    pub moe_ep_size: u32,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            attention_dp_size: 1,
            moe_tp_size: 1,
            moe_ep_size: 1,
        }
    }
}

/// Dtype configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DtypeConfig {
    pub gemm_quant: GemmQuantMode,
    pub moe_quant: MoeQuantMode,
    pub fmha_quant: FmhaQuantMode,
    pub kv_cache_quant: KvCacheQuantMode,
    pub comm_quant: CommQuantMode,
}

impl Default for DtypeConfig {
    fn default() -> Self {
        Self {
            gemm_quant: GemmQuantMode::Bfloat16,
            moe_quant: MoeQuantMode::Bfloat16,
            fmha_quant: FmhaQuantMode::Bfloat16,
            kv_cache_quant: KvCacheQuantMode::Bfloat16,
            comm_quant: CommQuantMode::Half,
        }
    }
}

/// Workload distribution name passed to MoE queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadDistribution {
    Uniform,
    PowerLaw,
}

impl WorkloadDistribution {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            // Python's MOE model appends `_{power_law_alpha}` (default 1.2)
            // to the configured distribution string before querying.
            Self::PowerLaw => "power_law_1.2",
        }
    }
}

/// Wide-EP routing mode. Mirrors Python's two distinct DeepSeek
/// variant classes:
/// - `Off` — standard `build_deepseek_model` (vLLM, default sglang).
/// - `SglangDeepEp` — `WideEPDeepSeekModel`: SGLang + DeepEP MoE backend
///   (`moe_backend="deepep_moe"`). Uses `WideEpContextMlaOp` /
///   `WideEpGenerationMlaOp` + standard MoE/MoEDispatch.
/// - `Trtllm` — `TrtllmWideEPDeepSeekModel`: TRT-LLM + `enable_wideep`.
///   Uses standard `ContextMlaOp` / `GenerationMlaOp` + `WideEpMoeOp` +
///   TrtllmAlltoall dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WideEpMode {
    #[default]
    Off,
    SglangDeepEp,
    Trtllm,
}

/// Multi-Token Prediction speculative decoding parameters.
///
/// Mirrors `aiconfigurator.sdk.models.base.BaseModel._nextn` /
/// `_nextn_accept_rates`. Python's `_mtp_scale_factor` scales every
/// generation-phase op by `1/(1+E[accept]) * (nextn+layers)/layers`,
/// reflecting that MTP adds `nextn` extra transformer layers but accepts
/// `E[accept]` extra tokens per step. The default (disabled) keeps the
/// scale at 1.0.
///
/// Only DeepSeek-family and Qwen3.5 builders consume this; Python's
/// TaskConfig only sets `nextn=1` for those families
/// (`sdk/task.py:448-449`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MtpConfig {
    pub nextn: u32,
    pub accept_rates: Vec<f64>,
}

impl MtpConfig {
    /// Mirror of Python `helpers.calc_expectation` (`models/helpers.py:219`).
    /// Recursive: `E[accept] = prod(rates[0..nextn-1]) + E[nextn-1]`.
    fn calc_expectation(&self) -> f64 {
        Self::calc_recursive(self.nextn, &self.accept_rates)
    }

    fn calc_recursive(nextn: u32, rates: &[f64]) -> f64 {
        if nextn == 0 {
            return 0.0;
        }
        let upper = (nextn as usize).min(rates.len());
        let mut prob = 1.0_f64;
        for i in 0..upper {
            prob *= rates[i];
        }
        // Python's recursion assumes `rates[i]` exists for all `i < nextn`;
        // if the caller provides a shorter list we treat missing entries as
        // implicit zeros (the trailing `prob *= 0` factor wipes the term).
        // This matches Python's index-out-of-range behavior at runtime —
        // an out-of-range nextn is a configuration bug, not a normal path.
        if nextn > 1 {
            prob + Self::calc_recursive(nextn - 1, rates)
        } else {
            prob
        }
    }

    /// Mirror of `BaseModel._mtp_scale_factor` (`models/base.py:105-110`):
    /// `1/(1 + E[accept]) * (nextn + num_layers) / num_layers`.
    /// Returns 1.0 when MTP is disabled (`nextn == 0`).
    pub fn scale_factor(&self, num_layers: u32) -> f64 {
        if self.nextn == 0 || num_layers == 0 {
            return 1.0;
        }
        let exp = self.calc_expectation();
        let layers = num_layers as f64;
        let nextn = self.nextn as f64;
        (1.0 / (1.0 + exp)) * (nextn + layers) / layers
    }
}

/// Composite of (HF spec, parallelism, dtypes, system) — what every model
/// constructor consumes to populate its op lists.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub spec: ModelSpec,
    pub parallel: ParallelConfig,
    pub dtypes: DtypeConfig,
    pub system_spec: SystemSpec,
    pub backend: BackendKind,
    /// WideEP routing decision. Default `Off` keeps every existing model
    /// builder on the standard path.
    pub wideep_mode: WideEpMode,
    /// When true, the TRT-LLM WideEP variant suffixes the workload
    /// distribution with `_eplb` (selects the EPLB perf table rows).
    /// SGLang's sglang DeepEP variant uses this for the prefill power-law
    /// alpha selection. Defaults to false.
    pub enable_eplb: bool,
    /// Multi-Token Prediction speculative decoding parameters. Default
    /// (`nextn=0`) produces `mtp_scale_factor=1.0` — i.e. no scaling. Only
    /// DeepSeek-family + Qwen3.5 builders multiply per-layer generation-op
    /// scale factors by this; other model builders ignore it.
    pub mtp: MtpConfig,
}

impl ModelConfig {
    pub fn heads_per_gpu(&self) -> u32 {
        let tp = self.parallel.tp_size.max(1);
        (self.spec.num_attention_heads + tp - 1) / tp
    }

    pub fn kv_heads_per_gpu(&self) -> u32 {
        self.spec.kv_heads_per_gpu(self.parallel.tp_size)
    }

    pub fn intermediate_size_per_gpu(&self) -> u32 {
        let tp = self.parallel.tp_size.max(1);
        (self.spec.intermediate_size + tp - 1) / tp
    }

    pub fn moe_intermediate_size_per_gpu(&self) -> u32 {
        let mt = self.parallel.moe_tp_size.max(1);
        (self.spec.moe_intermediate_size + mt - 1) / mt
    }

    /// Multi-Token Prediction scale factor for this config's model layers.
    /// Returns 1.0 when MTP is disabled (`mtp.nextn == 0`). DeepSeek-family
    /// + Qwen3.5 builders multiply per-layer generation-op scale_factors by
    /// this value; non-MTP families ignore it (default `MtpConfig::default()`
    /// → 1.0).
    pub fn mtp_scale_factor(&self) -> f64 {
        self.mtp.scale_factor(self.spec.num_hidden_layers)
    }
}

/// A built model with typed op lists per phase.
///
/// Constructed by family-specific builders in `models/{llama, moe,
/// deepseek}.rs`. The session driver iterates the lists exactly the way
/// Python's `_run_context_phase` / `_run_generation_phase` iterate
/// `model.context_ops` / `model.generation_ops`.
#[derive(Debug)]
pub struct Model {
    pub config: ModelConfig,
    pub context_ops: Vec<Op>,
    pub generation_ops: Vec<Op>,
    pub encoder_ops: Vec<Op>,
}

impl Model {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            context_ops: Vec::new(),
            generation_ops: Vec::new(),
            encoder_ops: Vec::new(),
        }
    }
}
