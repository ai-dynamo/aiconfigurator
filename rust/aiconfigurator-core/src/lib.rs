// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native core latency API for AIConfigurator.
//!
//! This crate is the Phase 1 sidecar implementation for Dynamo Mocker and
//! other Rust callers. It intentionally does not change the existing Python
//! SDK behavior. The v1 hot-path input is shaped like Dynamo's
//! ForwardPassMetrics so Dynamo can use its existing scheduler telemetry
//! contract first, then evolve toward richer request/block-aware inputs later.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

mod ffi;
mod fpm_perf;
mod model;
mod perf;

#[cfg(feature = "extension-module")]
mod python;

use perf::PerfDatabase;

pub use fpm_perf::{
    ForwardPassPerfDiagnostics, ForwardPassPerfModel, ForwardPassPerfOptions,
    ForwardPassPerfReadiness, ForwardPassPerfSource,
};
pub use model::{ModelFamily, ModelSpec};

pub const ENGINE_CONFIG_SCHEMA_VERSION: u32 = 1;
pub const FPM_VERSION: u32 = 1;

/// Static engine identity and setup information used to initialize an
/// [`EngineStepEstimator`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EngineConfig {
    pub schema_version: u32,

    pub model_name: String,
    pub model_arch: Option<String>,

    pub system_name: String,

    pub backend: BackendKind,
    pub backend_version: Option<String>,

    pub tp_size: u32,
    pub pp_size: u32,
    pub moe_tp_size: Option<u32>,
    pub moe_ep_size: Option<u32>,
    pub attention_dp_size: Option<u32>,

    pub weight_dtype: Option<DataType>,
    #[serde(default)]
    pub moe_dtype: Option<DataType>,
    pub activation_dtype: Option<DataType>,
    pub kv_cache_dtype: Option<DataType>,

    pub kv_block_size: Option<u32>,

    #[serde(default)]
    pub extra: BTreeMap<String, String>,
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
    fn as_str(&self) -> &'static str {
        match self {
            Self::Trtllm => "trtllm",
            Self::Sglang => "sglang",
            Self::Vllm => "vllm",
        }
    }
}

/// Precision/quantization dtypes exposed by the v1 Rust core API.
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

impl DataType {
    fn gemm_quant_name(&self) -> &'static str {
        match self {
            Self::Bfloat16 | Self::Float16 => "bfloat16",
            Self::Fp8 => "fp8",
            Self::Fp8Static => "fp8_static",
            Self::Fp8Block => "fp8_block",
            Self::Nvfp4 => "nvfp4",
            Self::Int8 => "int8_wo",
            Self::Int4 | Self::W4afp8 | Self::W4a16Mxfp4 | Self::W4a8Mxfp4Mxfp8 => "int4_wo",
        }
    }

    fn fmha_quant_name(&self) -> &'static str {
        match self {
            Self::Fp8 | Self::Fp8Static | Self::Fp8Block | Self::Nvfp4 => "fp8",
            _ => "bfloat16",
        }
    }

    fn kv_cache_quant_name(&self) -> &'static str {
        match self {
            Self::Fp8 | Self::Fp8Static | Self::Fp8Block | Self::Nvfp4 => "fp8",
            Self::Int8 => "int8",
            _ => "bfloat16",
        }
    }

    fn moe_quant_name(&self) -> &'static str {
        match self {
            Self::Bfloat16 | Self::Float16 => "bfloat16",
            Self::Fp8 | Self::Fp8Static => "fp8",
            Self::Fp8Block => "fp8_block",
            Self::Nvfp4 => "nvfp4",
            Self::Int8 => "int8_wo",
            Self::Int4 => "int4_wo",
            Self::W4afp8 => "w4afp8",
            Self::W4a16Mxfp4 => "w4a16_mxfp4",
            Self::W4a8Mxfp4Mxfp8 => "w4a8_mxfp4_mxfp8",
        }
    }
}

/// Metrics for requests scheduled in one forward-pass iteration.
///
/// This mirrors Dynamo ForwardPassMetrics v1 scheduled request telemetry.
/// AIC owns this Rust copy so AIC does not depend on Dynamo crates.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ScheduledRequestMetrics {
    /// Number of prefill requests, including new requests and chunked-prefill
    /// continuations.
    #[serde(default)]
    pub num_prefill_requests: u32,
    /// Total tokens freshly computed for prefill requests in this iteration.
    #[serde(default)]
    pub sum_prefill_tokens: u32,
    /// Population variance of total prompt lengths across prefill requests.
    #[serde(default)]
    pub var_prefill_length: f64,
    /// Total KV tokens read for prefill requests, including prefix cache hits
    /// and previously computed chunks.
    #[serde(default)]
    pub sum_prefill_kv_tokens: u32,
    /// Number of decode requests.
    #[serde(default)]
    pub num_decode_requests: u32,
    /// Total KV context length across decode requests.
    #[serde(default)]
    pub sum_decode_kv_tokens: u32,
    /// Population variance of KV context lengths across decode requests.
    #[serde(default)]
    pub var_decode_kv_tokens: f64,
}

/// Metrics for requests queued but not scheduled in one iteration.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct QueuedRequestMetrics {
    #[serde(default)]
    pub num_prefill_requests: u32,
    #[serde(default)]
    pub sum_prefill_tokens: u32,
    #[serde(default)]
    pub var_prefill_length: f64,
    #[serde(default)]
    pub num_decode_requests: u32,
    #[serde(default)]
    pub sum_decode_kv_tokens: u32,
    #[serde(default)]
    pub var_decode_kv_tokens: f64,
}

/// Per-iteration forward-pass metrics.
///
/// In Dynamo this struct is telemetry emitted after an engine iteration. In
/// AIC Phase 1, the scheduled portion is also the estimator input. The strict
/// native `EngineStepEstimator` ignores `wall_time` and `queued_requests`. The
/// tuned `ForwardPassPerfModel` uses `wall_time` only when observations are
/// passed to `tune_with_fpms`; queued fields remain schema-only at the
/// forward-pass layer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ForwardPassMetrics {
    #[serde(default = "default_fpm_version")]
    pub version: u32,
    #[serde(default)]
    pub worker_id: String,
    #[serde(default)]
    pub dp_rank: u32,
    #[serde(default)]
    pub counter_id: u64,
    #[serde(default)]
    pub wall_time: f64,
    #[serde(default)]
    pub scheduled_requests: ScheduledRequestMetrics,
    #[serde(default)]
    pub queued_requests: QueuedRequestMetrics,
}

impl Default for ForwardPassMetrics {
    fn default() -> Self {
        Self {
            version: FPM_VERSION,
            worker_id: String::new(),
            dp_rank: 0,
            counter_id: 0,
            wall_time: 0.0,
            scheduled_requests: ScheduledRequestMetrics::default(),
            queued_requests: QueuedRequestMetrics::default(),
        }
    }
}

/// Reusable estimator that owns loaded model metadata and perf-file data.
#[derive(Clone, Debug)]
pub struct EngineStepEstimator {
    config: EngineConfig,
    model: ModelSpec,
    perf: PerfDatabase,
}

/// Create a reusable estimator using the default AIC data roots.
///
/// The default roots are discovered from:
/// - `AICONFIGURATOR_SYSTEMS_PATH`, if set
/// - `AICONFIGURATOR_MODEL_CONFIGS_PATH`, if set
/// - repository-relative `src/aiconfigurator/systems` and
///   `src/aiconfigurator/model_configs` during Phase 1 development
pub fn create_engine_step_estimator(config: EngineConfig) -> Result<EngineStepEstimator, AicError> {
    let roots = DataRoots::discover()?;
    EngineStepEstimator::from_config_with_roots(
        config,
        &roots.systems_root,
        &roots.model_configs_root,
    )
}

impl EngineStepEstimator {
    /// Test/dev constructor that avoids putting data-root fields into
    /// [`EngineConfig`].
    pub fn from_config_with_roots(
        config: EngineConfig,
        systems_root: impl AsRef<Path>,
        model_configs_root: impl AsRef<Path>,
    ) -> Result<Self, AicError> {
        validate_engine_config(&config)?;
        let model = ModelSpec::load(&config.model_name, model_configs_root.as_ref())?;
        let perf = PerfDatabase::load(
            systems_root.as_ref(),
            &config.system_name,
            config.backend.as_str(),
            config.backend_version.as_deref(),
        )?;

        Ok(Self {
            config,
            model,
            perf,
        })
    }

    /// Estimate one forward-pass iteration from per-attention-DP-rank metrics.
    ///
    /// The primary Rust API returns [`Duration`] to keep units explicit.
    pub fn forward_pass_time(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Duration, AicError> {
        let ms = self.forward_pass_time_ms(metrics_by_rank)?;
        Ok(Duration::from_secs_f64(ms / 1000.0))
    }

    /// Numeric millisecond convenience API for callers that track virtual time
    /// as floating-point milliseconds.
    pub fn forward_pass_time_ms(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<f64, AicError> {
        if metrics_by_rank.is_empty() {
            return Err(AicError::InvalidForwardPassMetrics(
                "at least one attention-DP rank metric is required".to_string(),
            ));
        }
        let expected_ranks = self.config.attention_dp_size.unwrap_or(1).max(1) as usize;
        if metrics_by_rank.len() != expected_ranks {
            return Err(AicError::InvalidForwardPassMetrics(format!(
                "expected {expected_ranks} attention-DP rank metric(s), got {}",
                metrics_by_rank.len()
            )));
        }
        for metrics in metrics_by_rank {
            validate_forward_pass_metrics(metrics)?;
        }

        let workloads = metrics_by_rank
            .iter()
            .map(RankWorkload::from_metrics)
            .collect::<Vec<_>>();
        let dense_tokens = workloads
            .iter()
            .map(RankWorkload::non_attention_tokens)
            .max()
            .unwrap_or(0);
        let moe_tokens = workloads.iter().fold(0_u32, |tokens, workload| {
            tokens.saturating_add(workload.non_attention_tokens())
        });
        let logits_batch = workloads
            .iter()
            .map(RankWorkload::logits_batch)
            .max()
            .unwrap_or(0);

        let non_attention_ms =
            self.model_non_attention_latency_ms(dense_tokens, moe_tokens, logits_batch)?;
        let attention_ms = self.prefill_attention_step_time_ms(&workloads)?
            + self.decode_attention_step_time_ms(&workloads)?;

        Ok((non_attention_ms + attention_ms).max(0.0))
    }

    /// Engine-step naming alias over the FPM-shaped v1 input.
    pub fn engine_step_time(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<Duration, AicError> {
        self.forward_pass_time(metrics_by_rank)
    }

    /// Engine-step naming alias over the FPM-shaped v1 input.
    pub fn engine_step_time_ms(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<f64, AicError> {
        self.forward_pass_time_ms(metrics_by_rank)
    }

    // TODO(remove-after-rust-migration): parity check/benchmark-only cache reset.
    pub fn clear_runtime_caches(&self) {
        self.perf.clear_query_cache();
    }

    fn prefill_attention_step_time_ms(&self, workloads: &[RankWorkload]) -> Result<f64, AicError> {
        let mut latency_ms = 0.0_f64;
        for workload in workloads {
            if let Some((batch_size, effective_isl, prefix)) = workload.prefill_shape() {
                latency_ms = latency_ms.max(self.model_prefill_attention_latency_ms(
                    batch_size,
                    effective_isl,
                    prefix,
                )?);
            }
        }
        Ok(latency_ms)
    }

    fn decode_attention_step_time_ms(&self, workloads: &[RankWorkload]) -> Result<f64, AicError> {
        let mut latency_ms = 0.0_f64;
        for workload in workloads {
            if let Some((batch_size, context_length)) = workload.decode_shape() {
                latency_ms = latency_ms
                    .max(self.model_decode_attention_latency_ms(batch_size, context_length)?);
            }
        }
        Ok(latency_ms)
    }

    fn model_non_attention_latency_ms(
        &self,
        dense_tokens: u32,
        moe_tokens: u32,
        logits_batch: u32,
    ) -> Result<f64, AicError> {
        if dense_tokens == 0 {
            return Ok(0.0);
        }
        let tp = self.config.tp_size.max(1);
        let model = &self.model;
        let layers = model.num_hidden_layers as f64;
        let quant = self.gemm_quant_name();

        let qkv_out = model.num_attention_heads * model.head_dim / tp
            + model.head_dim * model.kv_heads_per_gpu(tp) * 2;
        let proj_k = model.num_attention_heads * model.head_dim / tp;
        let ffn1_out = 2 * model.intermediate_size / tp;
        let ffn2_k = model.intermediate_size / tp;
        let hidden = model.hidden_size;
        let dense_tokens_u64 = u64::from(dense_tokens);

        if model.uses_moe() {
            return self.model_moe_non_attention_latency_ms(
                dense_tokens,
                moe_tokens,
                logits_batch,
                qkv_out,
                proj_k,
            );
        }

        let dense_ffn = self
            .perf
            .query_gemm(quant, dense_tokens, ffn1_out, model.hidden_size)?
            + self
                .perf
                .query_gemm(quant, dense_tokens, model.hidden_size, ffn2_k)?;

        let per_layer = self
            .perf
            .query_gemm(quant, dense_tokens, qkv_out, model.hidden_size)?
            + self
                .perf
                .query_gemm(quant, dense_tokens, model.hidden_size, proj_k)?
            + dense_ffn;

        let logits = if logits_batch > 0 {
            self.perf
                .query_gemm(
                    self.logits_gemm_quant_name(),
                    logits_batch,
                    model.vocab_size / tp,
                    model.hidden_size,
                )
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let non_gemm = self.memory_op_latency_ms(dense_tokens_u64, hidden)
            + layers
                * (self.elementwise_latency_ms(dense_tokens_u64, 2 * hidden, 2 * hidden)
                    + self.elementwise_latency_ms(dense_tokens_u64, 2 * hidden, 2 * hidden)
                    + self.elementwise_latency_ms(dense_tokens_u64, ffn1_out, ffn2_k))
            + self.custom_allreduce_latency_ms(1.0, dense_tokens_u64, hidden)
            + self.custom_allreduce_latency_ms(layers, dense_tokens_u64, hidden)
            + self.custom_allreduce_latency_ms(layers, dense_tokens_u64, hidden)
            + self.p2p_latency_ms(layers, dense_tokens_u64, hidden);

        Ok(per_layer * layers + logits + non_gemm)
    }

    fn model_moe_non_attention_latency_ms(
        &self,
        dense_tokens: u32,
        moe_tokens: u32,
        logits_batch: u32,
        qkv_out: u32,
        proj_k: u32,
    ) -> Result<f64, AicError> {
        let tp = self.config.tp_size.max(1);
        let model = &self.model;
        let layers = model.num_hidden_layers as f64;
        let quant = self.gemm_quant_name();
        let hidden = model.hidden_size;
        let dense_tokens_u64 = u64::from(dense_tokens);
        let moe_tp = self.config.moe_tp_size.unwrap_or(tp).max(1);
        let moe_ep = self.config.moe_ep_size.unwrap_or(1).max(1);
        let attention_dp = self.config.attention_dp_size.unwrap_or(1);

        let dense_ffn =
            self.perf.query_gemm(
                quant,
                dense_tokens,
                2 * model.intermediate_size / tp,
                hidden,
            )? + self
                .perf
                .query_gemm(quant, dense_tokens, hidden, model.intermediate_size / tp)?;
        let router = self.perf.query_gemm(
            DataType::Bfloat16.gemm_quant_name(),
            dense_tokens,
            model.num_experts.max(1),
            hidden,
        )?;
        let moe = self
            .perf
            .query_moe(
                self.moe_quant_name(),
                moe_tokens.max(1),
                hidden,
                model.moe_intermediate_size.max(1),
                model.top_k.max(1),
                model.num_experts.max(1),
                moe_tp,
                moe_ep,
                &self.moe_workload_distribution(),
            )
            .unwrap_or(dense_ffn);
        let dispatch_pre = self.moe_dispatch_latency_ms(
            dense_tokens_u64,
            hidden,
            moe_tp,
            moe_ep,
            attention_dp,
            true,
        )?;
        let dispatch_post = self.moe_dispatch_latency_ms(
            dense_tokens_u64,
            hidden,
            moe_tp,
            moe_ep,
            attention_dp,
            false,
        )?;
        let shared_expert =
            self.shared_expert_latency_ms(dense_tokens, dense_tokens_u64, hidden)?;

        let per_layer = self.perf.query_gemm(quant, dense_tokens, qkv_out, hidden)?
            + self.perf.query_gemm(quant, dense_tokens, hidden, proj_k)?
            + router
            + dispatch_pre
            + moe
            + dispatch_post
            + shared_expert;

        let logits = if logits_batch > 0 {
            self.perf
                .query_gemm(
                    self.logits_gemm_quant_name(),
                    logits_batch,
                    model.vocab_size / tp,
                    hidden,
                )
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let non_gemm = self.memory_op_latency_ms(dense_tokens_u64, hidden)
            + layers
                * (self.elementwise_latency_ms(dense_tokens_u64, 2 * hidden, 2 * hidden)
                    + self.elementwise_latency_ms(dense_tokens_u64, 2 * hidden, 2 * hidden))
            + self.custom_allreduce_latency_ms(1.0, dense_tokens_u64, hidden)
            + self.p2p_latency_ms(layers, dense_tokens_u64, hidden);

        Ok(per_layer * layers + logits + non_gemm)
    }

    fn model_prefill_attention_latency_ms(
        &self,
        batch_size: u32,
        effective_isl: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let tp = self.config.tp_size.max(1);
        let model = &self.model;
        let layers = model.num_hidden_layers as f64;
        let attention = if model.uses_mla_attention() || model.uses_module_attention() {
            self.perf
                .query_context_mla(
                    self.fmha_quant_name(),
                    self.kv_cache_quant_name(),
                    batch_size,
                    effective_isl.saturating_add(prefix),
                    model.num_attention_heads / tp,
                )
                .unwrap_or_else(|| {
                    self.perf
                        .query_context_attention(
                            self.fmha_quant_name(),
                            self.kv_cache_quant_name(),
                            batch_size,
                            effective_isl,
                            prefix,
                            model.num_attention_heads / tp,
                            model.kv_heads_per_gpu(tp),
                            model.head_dim,
                        )
                        .unwrap_or(0.0)
                })
        } else {
            self.perf.query_context_attention(
                self.fmha_quant_name(),
                self.kv_cache_quant_name(),
                batch_size,
                effective_isl,
                prefix,
                model.num_attention_heads / tp,
                model.kv_heads_per_gpu(tp),
                model.head_dim,
            )?
        };
        Ok(attention * layers)
    }

    fn model_decode_attention_latency_ms(
        &self,
        batch_size: u32,
        context_length: u32,
    ) -> Result<f64, AicError> {
        let tp = self.config.tp_size.max(1);
        let model = &self.model;
        let layers = model.num_hidden_layers as f64;
        let sequence_tokens = context_length.saturating_add(1);
        let attention = if model.uses_mla_attention() || model.uses_module_attention() {
            self.perf
                .query_generation_mla(
                    self.kv_cache_quant_name(),
                    batch_size,
                    sequence_tokens,
                    model.num_attention_heads / tp,
                )
                .unwrap_or_else(|| {
                    self.perf
                        .query_generation_attention(
                            self.kv_cache_quant_name(),
                            batch_size,
                            sequence_tokens,
                            model.num_attention_heads / tp,
                            model.kv_heads_per_gpu(tp),
                            model.head_dim,
                        )
                        .unwrap_or(0.0)
                })
        } else {
            self.perf.query_generation_attention(
                self.kv_cache_quant_name(),
                batch_size,
                sequence_tokens,
                model.num_attention_heads / tp,
                model.kv_heads_per_gpu(tp),
                model.head_dim,
            )?
        };
        Ok(attention * layers)
    }

    fn gemm_quant_name(&self) -> &'static str {
        match self.config.weight_dtype.as_ref() {
            Some(dtype) => dtype.gemm_quant_name(),
            None => DataType::Bfloat16.gemm_quant_name(),
        }
    }

    fn fmha_quant_name(&self) -> &'static str {
        match self
            .config
            .activation_dtype
            .as_ref()
            .or(self.config.weight_dtype.as_ref())
        {
            Some(dtype) => dtype.fmha_quant_name(),
            None => DataType::Bfloat16.fmha_quant_name(),
        }
    }

    fn kv_cache_quant_name(&self) -> &'static str {
        match self.config.kv_cache_dtype.as_ref() {
            Some(dtype) => dtype.kv_cache_quant_name(),
            None => DataType::Bfloat16.kv_cache_quant_name(),
        }
    }

    fn moe_quant_name(&self) -> &'static str {
        match self
            .config
            .moe_dtype
            .as_ref()
            .or(self.config.weight_dtype.as_ref())
        {
            Some(dtype) => dtype.moe_quant_name(),
            None => DataType::Bfloat16.moe_quant_name(),
        }
    }

    fn logits_gemm_quant_name(&self) -> &'static str {
        // Match the Python op graph: logits/LM-head GEMM stays BF16 even when
        // transformer layer GEMMs use quantized kernels.
        DataType::Bfloat16.gemm_quant_name()
    }

    fn memory_op_latency_ms(&self, tokens: u64, dim: u32) -> f64 {
        self.perf
            .query_mem_op(tokens.saturating_mul(u64::from(dim)).saturating_mul(2))
    }

    fn elementwise_latency_ms(&self, tokens: u64, dim_in: u32, dim_out: u32) -> f64 {
        let bytes = tokens
            .saturating_mul(u64::from(dim_in).saturating_add(u64::from(dim_out)))
            .saturating_mul(2);
        self.perf.query_mem_op(bytes)
    }

    fn custom_allreduce_latency_ms(&self, scale_factor: f64, tokens: u64, hidden: u32) -> f64 {
        if self.config.tp_size <= 1 {
            return 0.0;
        }
        let size = tokens.saturating_mul(u64::from(hidden));
        self.perf.query_custom_allreduce(self.config.tp_size, size) * scale_factor
    }

    fn moe_dispatch_latency_ms(
        &self,
        tokens: u64,
        hidden: u32,
        moe_tp: u32,
        moe_ep: u32,
        attention_dp: u32,
        pre_dispatch: bool,
    ) -> Result<f64, AicError> {
        let num_gpus = moe_tp.saturating_mul(moe_ep).max(1);
        if attention_dp == 0 || attention_dp > num_gpus || num_gpus % attention_dp != 0 {
            return Err(AicError::InvalidEngineConfig(format!(
                "invalid MoE dispatch topology: moe_tp ({moe_tp}) * moe_ep ({moe_ep}) must be divisible by attention_dp_size ({attention_dp})"
            )));
        }
        let attention_tp = num_gpus / attention_dp;
        if matches!(self.config.backend, BackendKind::Sglang)
            && attention_tp > 1
            && attention_dp > 1
        {
            return Err(AicError::InvalidEngineConfig(
                "SGLang non-DeepEP MoE dispatch does not support attention TP > 1 and attention DP > 1 together"
                    .to_string(),
            ));
        }
        let volume = tokens.saturating_mul(u64::from(hidden));
        if attention_tp > 1 {
            Ok(self.perf.query_custom_allreduce(num_gpus, volume))
        } else if attention_dp > 1 {
            let operation = if pre_dispatch {
                "all_gather"
            } else {
                "reduce_scatter"
            };
            Ok(self.perf.query_nccl(
                "half",
                num_gpus,
                operation,
                volume.saturating_mul(u64::from(attention_dp)),
            ))
        } else {
            Ok(0.0)
        }
    }

    fn shared_expert_latency_ms(
        &self,
        dense_tokens: u32,
        dense_tokens_u64: u64,
        hidden: u32,
    ) -> Result<f64, AicError> {
        let shared_intermediate = self.model.shared_expert_intermediate_size;
        if shared_intermediate == 0 {
            return Ok(0.0);
        }
        let tp = self.config.tp_size.max(1);
        let shared_per_tp = shared_intermediate / tp;
        Ok(self
            .perf
            .query_gemm(self.gemm_quant_name(), dense_tokens, shared_per_tp, hidden)?
            + self.elementwise_latency_ms(dense_tokens_u64, shared_per_tp, shared_per_tp)
            + self
                .perf
                .query_gemm(self.gemm_quant_name(), dense_tokens, hidden, shared_per_tp)?)
    }

    fn moe_workload_distribution(&self) -> String {
        let distribution = self
            .config
            .extra
            .get("moe_workload_distribution")
            .or_else(|| self.config.extra.get("workload_distribution"))
            .map(String::as_str)
            .unwrap_or("power_law");
        if distribution != "power_law" {
            return distribution.to_string();
        }
        let alpha = match self.model.family {
            ModelFamily::Moe | ModelFamily::Qwen35 => "1.2",
            _ => "1.01",
        };
        format!("power_law_{alpha}")
    }

    fn p2p_latency_ms(&self, scale_factor: f64, tokens: u64, hidden: u32) -> f64 {
        if self.config.pp_size <= 1 || scale_factor == 0.0 {
            return 0.0;
        }
        let bytes = tokens.saturating_mul(u64::from(hidden)).saturating_mul(2);
        self.perf.query_p2p(bytes) * scale_factor
    }
}

#[derive(Clone, Copy, Debug)]
struct RankWorkload {
    prefill_requests: u32,
    prefill_tokens: u32,
    prefill_kv_tokens: u32,
    decode_requests: u32,
    decode_kv_tokens: u32,
}

impl RankWorkload {
    fn from_metrics(metrics: &ForwardPassMetrics) -> Self {
        let scheduled = &metrics.scheduled_requests;
        Self {
            prefill_requests: scheduled.num_prefill_requests,
            prefill_tokens: scheduled.sum_prefill_tokens,
            prefill_kv_tokens: scheduled.sum_prefill_kv_tokens,
            decode_requests: scheduled.num_decode_requests,
            decode_kv_tokens: scheduled.sum_decode_kv_tokens,
        }
    }

    fn prefill_shape(&self) -> Option<(u32, u32, u32)> {
        if self.prefill_requests == 0 || self.prefill_tokens == 0 {
            return None;
        }
        Some((
            self.prefill_requests,
            ceil_div_u32(self.prefill_tokens, self.prefill_requests),
            ceil_div_u32(self.prefill_kv_tokens, self.prefill_requests),
        ))
    }

    fn decode_shape(&self) -> Option<(u32, u32)> {
        if self.decode_requests == 0 {
            return None;
        }
        Some((
            self.decode_requests,
            ceil_div_u32(self.decode_kv_tokens, self.decode_requests),
        ))
    }

    fn non_attention_tokens(&self) -> u32 {
        self.prefill_tokens.saturating_add(self.decode_requests)
    }

    fn logits_batch(&self) -> u32 {
        if self.non_attention_tokens() == 0 {
            0
        } else {
            self.prefill_requests
                .saturating_add(self.decode_requests)
                .max(1)
        }
    }
}

#[derive(Debug, Error)]
pub enum AicError {
    #[error("unsupported schema version for {kind}: got {got}, expected {expected}")]
    UnsupportedSchemaVersion {
        kind: &'static str,
        got: u32,
        expected: u32,
    },
    #[error("invalid engine config: {0}")]
    InvalidEngineConfig(String),
    #[error("invalid forward pass metrics: {0}")]
    InvalidForwardPassMetrics(String),
    #[error("unsupported model for Rust core estimator: {0}")]
    UnsupportedModel(String),
    #[error("failed to find AIC data roots: {0}")]
    DataRoot(String),
    #[error("model config error: {0}")]
    ModelConfig(String),
    #[error("perf database error: {0}")]
    PerfDatabase(String),
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("CSV error at {path}: {source}")]
    Csv {
        path: PathBuf,
        #[source]
        source: csv::Error,
    },
    #[error("JSON error at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

struct DataRoots {
    systems_root: PathBuf,
    model_configs_root: PathBuf,
}

impl DataRoots {
    fn discover() -> Result<Self, AicError> {
        let systems_root = std::env::var_os("AICONFIGURATOR_SYSTEMS_PATH")
            .map(PathBuf::from)
            .or_else(|| repo_relative("src/aiconfigurator/systems"));
        let model_configs_root = std::env::var_os("AICONFIGURATOR_MODEL_CONFIGS_PATH")
            .map(PathBuf::from)
            .or_else(|| repo_relative("src/aiconfigurator/model_configs"));

        let Some(systems_root) = systems_root else {
            return Err(AicError::DataRoot(
                "set AICONFIGURATOR_SYSTEMS_PATH or run from an AIC checkout".to_string(),
            ));
        };
        let Some(model_configs_root) = model_configs_root else {
            return Err(AicError::DataRoot(
                "set AICONFIGURATOR_MODEL_CONFIGS_PATH or run from an AIC checkout".to_string(),
            ));
        };

        Ok(Self {
            systems_root,
            model_configs_root,
        })
    }
}

fn repo_relative(rel: &str) -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn validate_engine_config(config: &EngineConfig) -> Result<(), AicError> {
    if config.schema_version != ENGINE_CONFIG_SCHEMA_VERSION {
        return Err(AicError::UnsupportedSchemaVersion {
            kind: "EngineConfig",
            got: config.schema_version,
            expected: ENGINE_CONFIG_SCHEMA_VERSION,
        });
    }
    if config.model_name.trim().is_empty() {
        return Err(AicError::InvalidEngineConfig(
            "model_name must be non-empty".to_string(),
        ));
    }
    if config.system_name.trim().is_empty() {
        return Err(AicError::InvalidEngineConfig(
            "system_name must be non-empty".to_string(),
        ));
    }
    if config.tp_size == 0 {
        return Err(AicError::InvalidEngineConfig(
            "tp_size must be >= 1".to_string(),
        ));
    }
    if config.pp_size == 0 {
        return Err(AicError::InvalidEngineConfig(
            "pp_size must be >= 1".to_string(),
        ));
    }
    if config.pp_size != 1 {
        return Err(AicError::UnsupportedModel(
            "Phase 1 Rust estimator supports pp_size=1; pipeline-parallel P2P composition will be added later"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_forward_pass_metrics(metrics: &ForwardPassMetrics) -> Result<(), AicError> {
    if metrics.version != FPM_VERSION {
        return Err(AicError::UnsupportedSchemaVersion {
            kind: "ForwardPassMetrics",
            got: metrics.version,
            expected: FPM_VERSION,
        });
    }
    let scheduled = &metrics.scheduled_requests;
    if scheduled.num_prefill_requests == 0
        && (scheduled.sum_prefill_tokens > 0 || scheduled.sum_prefill_kv_tokens > 0)
    {
        return Err(AicError::InvalidForwardPassMetrics(
            "prefill token sums require num_prefill_requests > 0".to_string(),
        ));
    }
    if scheduled.num_decode_requests == 0 && scheduled.sum_decode_kv_tokens > 0 {
        return Err(AicError::InvalidForwardPassMetrics(
            "decode KV token sum requires num_decode_requests > 0".to_string(),
        ));
    }
    Ok(())
}

fn ceil_div_u32(sum: u32, count: u32) -> u32 {
    if count == 0 {
        0
    } else {
        sum.div_ceil(count)
    }
}

fn default_fpm_version() -> u32 {
    FPM_VERSION
}
