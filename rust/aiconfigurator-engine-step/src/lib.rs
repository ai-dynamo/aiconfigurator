// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native engine-step latency API for AIConfigurator.
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

mod model;
mod perf;

use model::DenseModelConfig;
use perf::PerfDatabase;

pub const ENGINE_CONFIG_SCHEMA_VERSION: u32 = 1;
pub const FPM_VERSION: u32 = 1;

/// Static engine identity and setup information used to initialize an
/// [`EngineStepEstimator`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EngineConfig {
    pub schema_version: u32,

    pub model_name: String,
    pub model_arch: Option<String>,
    pub max_sequence_length: Option<u32>,

    pub system_name: String,

    pub backend: BackendKind,
    pub backend_version: Option<String>,

    pub tp_size: u32,
    pub pp_size: u32,
    pub dp_size: u32,
    pub moe_tp_size: Option<u32>,
    pub moe_ep_size: Option<u32>,
    pub attention_dp_size: Option<u32>,

    pub weight_dtype: Option<DataType>,
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

/// Precision/quantization dtypes exposed by the v1 engine-step API.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    Bfloat16,
    Float16,
    Fp8,
    Fp8Static,
    Fp8Block,
    Nvfp4,
    Int8,
    Int4,
}

impl DataType {
    fn gemm_quant_name(&self) -> &'static str {
        match self {
            Self::Bfloat16 => "bfloat16",
            Self::Float16 => "bfloat16",
            Self::Fp8 => "fp8",
            Self::Fp8Static => "fp8_static",
            Self::Fp8Block => "fp8_block",
            Self::Nvfp4 => "nvfp4",
            Self::Int8 => "int8_wo",
            Self::Int4 => "int4_wo",
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
/// AIC Phase 1, the scheduled portion is also the estimator input. `wall_time`
/// and `queued_requests` are accepted for schema parity but ignored by the
/// latency estimator.
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
    model: DenseModelConfig,
    perf: PerfDatabase,
}

/// Create a reusable engine-step estimator using the default AIC data roots.
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
        let model = DenseModelConfig::load(&config.model_name, model_configs_root.as_ref())?;
        model.validate_supported()?;
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

    /// Estimate one forward-pass iteration. The primary Rust API returns
    /// [`Duration`] to keep units explicit.
    pub fn forward_pass_time(&self, metrics: &ForwardPassMetrics) -> Result<Duration, AicError> {
        let ms = self.forward_pass_time_ms(metrics)?;
        Ok(Duration::from_secs_f64(ms / 1000.0))
    }

    /// Numeric millisecond convenience API for callers that track virtual time
    /// as floating-point milliseconds.
    pub fn forward_pass_time_ms(&self, metrics: &ForwardPassMetrics) -> Result<f64, AicError> {
        validate_forward_pass_metrics(metrics)?;

        let scheduled = &metrics.scheduled_requests;
        let mut total_ms = 0.0;

        if scheduled.num_prefill_requests > 0 && scheduled.sum_prefill_tokens > 0 {
            total_ms += self.prefill_step_time_ms(
                scheduled.num_prefill_requests,
                scheduled.sum_prefill_tokens,
                scheduled.sum_prefill_kv_tokens,
            )?;
        }
        if scheduled.num_decode_requests > 0 {
            total_ms += self.decode_step_time_ms(
                scheduled.num_decode_requests,
                scheduled.sum_decode_kv_tokens,
            )?;
        }
        Ok(total_ms.max(0.0))
    }

    /// Engine-step naming alias over the FPM-shaped v1 input.
    pub fn engine_step_time(&self, metrics: &ForwardPassMetrics) -> Result<Duration, AicError> {
        self.forward_pass_time(metrics)
    }

    /// Engine-step naming alias over the FPM-shaped v1 input.
    pub fn engine_step_time_ms(&self, metrics: &ForwardPassMetrics) -> Result<f64, AicError> {
        self.forward_pass_time_ms(metrics)
    }

    fn prefill_step_time_ms(
        &self,
        batch_size: u32,
        sum_prefill_tokens: u32,
        sum_prefill_kv_tokens: u32,
    ) -> Result<f64, AicError> {
        let effective_isl = ceil_div_u32(sum_prefill_tokens, batch_size);
        let prefix = ceil_div_u32(sum_prefill_kv_tokens, batch_size);
        if batch_size == 0 || effective_isl == 0 {
            return Ok(0.0);
        }

        self.dense_prefill_latency_ms(batch_size, effective_isl, prefix)
    }

    fn decode_step_time_ms(
        &self,
        batch_size: u32,
        sum_decode_kv_tokens: u32,
    ) -> Result<f64, AicError> {
        let context_length = ceil_div_u32(sum_decode_kv_tokens, batch_size);
        if batch_size == 0 {
            return Ok(0.0);
        }
        self.dense_decode_latency_ms(batch_size, context_length, 2)
    }

    fn dense_prefill_latency_ms(
        &self,
        batch_size: u32,
        effective_isl: u32,
        prefix: u32,
    ) -> Result<f64, AicError> {
        let tp = self.config.tp_size.max(1);
        let model = &self.model;
        let layers = model.num_hidden_layers as f64;
        let m = batch_size * effective_isl;
        let quant = self.gemm_quant_name();

        let qkv_out = model.num_attention_heads * model.head_dim / tp
            + model.head_dim * model.kv_heads_per_gpu(tp) * 2;
        let proj_k = model.num_attention_heads * model.head_dim / tp;
        let ffn1_out = 2 * model.intermediate_size / tp;
        let ffn2_k = model.intermediate_size / tp;

        let per_layer = self.perf.query_gemm(quant, m, qkv_out, model.hidden_size)?
            + self.perf.query_context_attention(
                self.fmha_quant_name(),
                self.kv_cache_quant_name(),
                batch_size,
                effective_isl,
                prefix,
                model.num_attention_heads / tp,
                model.kv_heads_per_gpu(tp),
                model.head_dim,
            )?
            + self.perf.query_gemm(quant, m, model.hidden_size, proj_k)?
            + self
                .perf
                .query_gemm(quant, m, ffn1_out, model.hidden_size)?
            + self.perf.query_gemm(quant, m, model.hidden_size, ffn2_k)?;

        let logits = self
            .perf
            .query_gemm(
                "bfloat16",
                batch_size,
                model.vocab_size / tp,
                model.hidden_size,
            )
            .unwrap_or(0.0);

        Ok(per_layer * layers + logits)
    }

    fn dense_decode_latency_ms(
        &self,
        batch_size: u32,
        context_length: u32,
        osl: u32,
    ) -> Result<f64, AicError> {
        if osl <= 1 {
            return Ok(0.0);
        }

        let tp = self.config.tp_size.max(1);
        let model = &self.model;
        let layers = model.num_hidden_layers as f64;
        let quant = self.gemm_quant_name();
        let m = batch_size;

        let qkv_out = model.num_attention_heads * model.head_dim / tp
            + model.head_dim * model.kv_heads_per_gpu(tp) * 2;
        let proj_k = model.num_attention_heads * model.head_dim / tp;
        let ffn1_out = 2 * model.intermediate_size / tp;
        let ffn2_k = model.intermediate_size / tp;

        let mut total = 0.0;
        for step in 0..(osl - 1) {
            let s = context_length + step + 1;
            let per_layer = self.perf.query_gemm(quant, m, qkv_out, model.hidden_size)?
                + self.perf.query_generation_attention(
                    self.kv_cache_quant_name(),
                    batch_size,
                    s,
                    model.num_attention_heads / tp,
                    model.kv_heads_per_gpu(tp),
                    model.head_dim,
                )?
                + self.perf.query_gemm(quant, m, model.hidden_size, proj_k)?
                + self
                    .perf
                    .query_gemm(quant, m, ffn1_out, model.hidden_size)?
                + self.perf.query_gemm(quant, m, model.hidden_size, ffn2_k)?;
            let logits = self
                .perf
                .query_gemm(
                    "bfloat16",
                    batch_size,
                    model.vocab_size / tp,
                    model.hidden_size,
                )
                .unwrap_or(0.0);
            total += per_layer * layers + logits;
        }

        Ok(total)
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
    #[error("unsupported model for Rust engine-step estimator: {0}")]
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
    if config.dp_size == 0 {
        return Err(AicError::InvalidEngineConfig(
            "dp_size must be >= 1".to_string(),
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
        ((u64::from(sum) + u64::from(count) - 1) / u64::from(count)) as u32
    }
}

fn default_fpm_version() -> u32 {
    FPM_VERSION
}
