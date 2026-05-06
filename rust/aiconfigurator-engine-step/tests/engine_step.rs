// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use aiconfigurator_engine_step::{
    BackendKind, DataType, EngineConfig, EngineStepEstimator, ForwardPassMetrics,
    ScheduledRequestMetrics, ENGINE_CONFIG_SCHEMA_VERSION, FPM_VERSION,
};
use tempfile::TempDir;

#[test]
fn prefill_estimate_uses_perf_files() {
    let fixture = Fixture::new();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 2,
            sum_prefill_tokens: 20,
            sum_prefill_kv_tokens: 0,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&metrics).unwrap();

    assert_close(latency, 30.5);
}

#[test]
fn decode_estimate_uses_perf_files() {
    let fixture = Fixture::new();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests: 2,
            sum_decode_kv_tokens: 32,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&metrics).unwrap();

    assert_close(latency, 3.9);
}

#[test]
fn mixed_estimate_sums_phase_estimates() {
    let fixture = Fixture::new();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 2,
            sum_prefill_tokens: 20,
            sum_prefill_kv_tokens: 0,
            num_decode_requests: 2,
            sum_decode_kv_tokens: 32,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.engine_step_time_ms(&metrics).unwrap();

    assert_close(latency, 34.4);
}

#[test]
fn empty_step_returns_zero() {
    let fixture = Fixture::new();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics::default();

    let latency = estimator.forward_pass_time_ms(&metrics).unwrap();

    assert_eq!(latency, 0.0);
}

#[test]
fn invalid_schema_rejected() {
    let fixture = Fixture::new();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        version: 999,
        ..Default::default()
    };

    let err = estimator
        .forward_pass_time_ms(&metrics)
        .unwrap_err()
        .to_string();

    assert!(err.contains("unsupported schema version"));
}

#[test]
fn default_fpm_version_matches_constant() {
    assert_eq!(ForwardPassMetrics::default().version, FPM_VERSION);
}

#[test]
fn git_lfs_pointer_is_reported() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n",
    )
    .unwrap();

    let err = EngineStepEstimator::from_config_with_roots(
        engine_config(),
        fixture.systems_root(),
        fixture.model_configs_root(),
    )
    .unwrap_err()
    .to_string();

    assert!(err.contains("Git LFS pointer"));
}

struct Fixture {
    _temp: TempDir,
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let temp = TempDir::new().unwrap();
        let root = temp.path().to_path_buf();
        let systems_root = root.join("systems");
        let data_root = systems_root.join("data/test_sxm/vllm/1.0.0");
        let model_configs_root = root.join("model_configs");

        fs::create_dir_all(&data_root).unwrap();
        fs::create_dir_all(&model_configs_root).unwrap();
        fs::write(
            systems_root.join("test_sxm.yaml"),
            "data_dir: data/test_sxm\n",
        )
        .unwrap();
        fs::write(
            model_configs_root.join("Test--Dense_config.json"),
            r#"{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 8,
  "hidden_size": 32,
  "intermediate_size": 64,
  "vocab_size": 160
}
"#,
        )
        .unwrap();
        fs::write(
            data_root.join("gemm_perf.txt"),
            "gemm_dtype,m,n,k,latency\n\
bfloat16,20,64,32,1.0\n\
bfloat16,20,32,32,2.0\n\
bfloat16,20,128,32,3.0\n\
bfloat16,20,32,64,4.0\n\
bfloat16,2,64,32,0.1\n\
bfloat16,2,32,32,0.2\n\
bfloat16,2,128,32,0.3\n\
bfloat16,2,32,64,0.4\n\
bfloat16,2,160,32,0.5\n",
        )
        .unwrap();
        fs::write(
            data_root.join("context_attention_perf.txt"),
            "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,2,10,4,2,8,5.0\n",
        )
        .unwrap();
        fs::write(
            data_root.join("generation_attention_perf.txt"),
            "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,step,latency\n\
bfloat16,bfloat16,2,16,4,2,8,1,0.7\n",
        )
        .unwrap();

        Self { _temp: temp, root }
    }

    fn systems_root(&self) -> PathBuf {
        self.root.join("systems")
    }

    fn model_configs_root(&self) -> PathBuf {
        self.root.join("model_configs")
    }

    fn perf_dir(&self) -> PathBuf {
        self.root.join("systems/data/test_sxm/vllm/1.0.0")
    }

    fn estimator(&self) -> EngineStepEstimator {
        EngineStepEstimator::from_config_with_roots(
            engine_config(),
            self.systems_root(),
            self.model_configs_root(),
        )
        .unwrap()
    }
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
        model_name: "Test/Dense".to_string(),
        model_arch: None,
        max_sequence_length: None,
        system_name: "test_sxm".to_string(),
        backend: BackendKind::Vllm,
        backend_version: Some("1.0.0".to_string()),
        tp_size: 1,
        pp_size: 1,
        dp_size: 1,
        moe_tp_size: None,
        moe_ep_size: None,
        attention_dp_size: None,
        weight_dtype: Some(DataType::Bfloat16),
        activation_dtype: Some(DataType::Bfloat16),
        kv_cache_dtype: Some(DataType::Bfloat16),
        kv_block_size: None,
        extra: BTreeMap::new(),
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "actual={actual}, expected={expected}"
    );
}
