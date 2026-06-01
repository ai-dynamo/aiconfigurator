// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native core latency API for AIConfigurator.
//!
//! The v1 hot-path input is shaped like Dynamo's `ForwardPassMetrics` so
//! Dynamo Mocker and other Rust callers can plug in their existing scheduler
//! telemetry contract. Richer request/block-aware inputs are a future
//! extension.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

mod ffi;
mod fpm_perf;

// Modular core. `common/` holds shared foundation types (enums, error,
// system_spec) with no AIC-domain knowledge. Top-level files (`config`,
// `session`) and directories (`models`, `operators`, `backends`,
// `perf_database`) carry the domain logic. `EngineStepEstimator` routes
// exclusively through `SessionEstimator`.
mod backends;
mod common;
mod config;
mod interpolation;
mod models;
mod operators;
mod perf_database;
mod session;

pub use common::AicError;
pub use fpm_perf::{
    ForwardPassPerfDiagnostics, ForwardPassPerfModel, ForwardPassPerfOptions,
    ForwardPassPerfReadiness, ForwardPassPerfSource,
};
pub use models::config_loader::load_path as load_model_config_path;

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

    /// Multi-Token Prediction speculative decoding depth (Python's
    /// `task_config.nextn`). `None` (default) disables MTP scaling. Python
    /// sets this to 1 for DeepSeek-family + Qwen3.5 models
    /// (`sdk/task.py:448-449`); other families leave it at 0/None. Rust
    /// model builders that don't consume MTP (everything except DeepSeek-
    /// family + Qwen3.5) ignore the value.
    #[serde(default)]
    pub nextn: Option<u32>,

    /// Per-step accept-rate prior used by MTP scaling. Mirrors Python's
    /// `task_config.nextn_accept_rates` (default `[0.85, 0.3, 0.0, 0.0,
    /// 0.0]`). Ignored when `nextn` is `None` or 0.
    #[serde(default)]
    pub nextn_accept_rates: Option<Vec<f64>>,

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
/// AIC the scheduled portion is also the estimator input. `wall_time` and
/// `queued_requests` are accepted for schema parity but ignored by the
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

/// Reusable estimator that owns the modular session pipeline.
///
/// Wraps `SessionEstimator` in `Arc` so cheap clones share the loaded
/// model + perf-database tree; required by callers like
/// `fpm_perf::ForwardPassPerfModel` which embed an estimator and need
/// to be `Clone`. The estimator's runtime state is already
/// `OnceLock`-cached internally, so sharing is sound.
#[derive(Clone, Debug)]
pub struct EngineStepEstimator {
    config: EngineConfig,
    session: Arc<session::SessionEstimator>,
}

/// Create a reusable estimator using the default AIC data roots.
///
/// The default roots are discovered from:
/// - `AICONFIGURATOR_SYSTEMS_PATH`, if set
/// - `AICONFIGURATOR_MODEL_CONFIGS_PATH`, if set
/// - repository-relative `src/aiconfigurator/systems` and
///   `src/aiconfigurator/model_configs` when developing in-tree
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
        let hf = crate::models::config_loader::load(
            &config.model_name,
            model_configs_root.as_ref(),
        )?;
        let system_yaml = systems_root
            .as_ref()
            .join(format!("{}.yaml", config.system_name));
        let system_spec = crate::common::system_spec::SystemSpec::load(&system_yaml)?;
        let version = config
            .backend_version
            .as_deref()
            .ok_or_else(|| AicError::InvalidEngineConfig(
                "backend_version is required to load the perf database".to_string(),
            ))?;
        let new_db = crate::perf_database::PerfDatabase::load(
            systems_root.as_ref(),
            &config.system_name,
            config.backend.as_str(),
            version,
        )?;
        let backend = session::backend_from_legacy(&config.backend);
        let parallel = crate::models::ParallelConfig {
            tp_size: config.tp_size.max(1),
            pp_size: config.pp_size.max(1),
            attention_dp_size: config.attention_dp_size.unwrap_or(1).max(1),
            moe_tp_size: config.moe_tp_size.unwrap_or(1).max(1),
            moe_ep_size: config.moe_ep_size.unwrap_or(1).max(1),
        };
        let dtypes = session::dtypes_from_legacy(
            config.weight_dtype.as_ref(),
            config.moe_dtype.as_ref(),
            config.activation_dtype.as_ref(),
            config.kv_cache_dtype.as_ref(),
        );
        let mtp = match (config.nextn, config.nextn_accept_rates.as_ref()) {
            (Some(n), _) if n == 0 => crate::models::base::MtpConfig::default(),
            (Some(n), Some(rates)) => crate::models::base::MtpConfig {
                nextn: n,
                accept_rates: rates.clone(),
            },
            (Some(n), None) => crate::models::base::MtpConfig {
                nextn: n,
                accept_rates: Vec::new(),
            },
            (None, _) => crate::models::base::MtpConfig::default(),
        };
        let estimator = session::SessionEstimator::build_with_options(
            hf,
            system_spec,
            new_db,
            backend,
            parallel,
            dtypes,
            Default::default(),
            false,
            mtp,
        )?;

        Ok(Self { config, session: Arc::new(estimator) })
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
    ///
    /// Routes through `session::SessionEstimator`.
    pub fn forward_pass_time_ms(
        &self,
        metrics_by_rank: &[ForwardPassMetrics],
    ) -> Result<f64, AicError> {
        if metrics_by_rank.is_empty() {
            return Err(AicError::InvalidForwardPassMetrics(
                "at least one attention-DP rank metric is required".to_string(),
            ));
        }
        for metrics in metrics_by_rank {
            validate_forward_pass_metrics(metrics)?;
        }
        self.session.forward_pass_time_ms(metrics_by_rank)
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

    /// Reset runtime caches (perf-DB query memoization and interpolation
    /// state). Exposed for parity-suite and benchmark runs that need a cold
    /// cache between iterations; production callers normally don't touch
    /// this.
    pub fn clear_runtime_caches(&self) {
        self.session.clear_runtime_caches();
    }

    /// Per-op context-phase latency breakdown (ms) for parity drilldown.
    /// Returns `Vec<(op_name, latency_ms)>` for the model's context graph at
    /// the given batch/ISL/prefix point. Used by integration tests and the
    /// parity scan runner; production callers normally use `forward_pass_*`.
    pub fn debug_context_breakdown(
        &self,
        batch_size: u32,
        effective_isl: u32,
        prefix: u32,
    ) -> Result<Vec<(String, f64)>, AicError> {
        self.session
            .debug_context_breakdown(batch_size, effective_isl, prefix)
    }

    /// Per-op generation-phase latency breakdown (ms) for parity drilldown.
    /// Returns `Vec<(op_name, latency_ms)>` for the model's generation graph
    /// at the given batch / KV-seq-tokens point.
    pub fn debug_generation_breakdown(
        &self,
        batch_size: u32,
        kv_seq_tokens: u32,
    ) -> Result<Vec<(String, f64)>, AicError> {
        self.session
            .debug_generation_breakdown(batch_size, kv_seq_tokens)
    }
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
            "Rust estimator supports pp_size=1; pipeline-parallel P2P composition will be added later"
                .to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_forward_pass_metrics(metrics: &ForwardPassMetrics) -> Result<(), AicError> {
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


fn default_fpm_version() -> u32 {
    FPM_VERSION
}
