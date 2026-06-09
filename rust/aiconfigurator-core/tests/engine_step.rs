// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use aiconfigurator_core::{
    BackendKind, DataType, EngineConfig, EngineStepEstimator, ForwardPassMetrics,
    ForwardPassPerfModel, ForwardPassPerfOptions, ForwardPassPerfReadiness, ForwardPassPerfSource,
    ModelSpec, ScheduledRequestMetrics, ENGINE_CONFIG_SCHEMA_VERSION, FPM_VERSION,
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

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

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

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 3.9);
}

#[test]
fn mixed_estimate_combines_non_attention_tokens() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,22,64,32,1.0\n\
bfloat16,22,32,32,2.0\n\
bfloat16,22,128,32,3.0\n\
bfloat16,22,32,64,4.0\n\
bfloat16,4,160,32,0.5\n",
    )
    .unwrap();
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

    let latency = estimator.engine_step_time_ms(&[metrics]).unwrap();

    assert_close(latency, 31.9);
}

#[test]
fn gemm_queries_extrapolate_token_dimension_for_matching_shape() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,20,64,32,1.0\n\
bfloat16,20,32,32,2.0\n\
bfloat16,20,128,32,3.0\n\
bfloat16,20,32,64,4.0\n\
bfloat16,40,64,32,2.0\n\
bfloat16,40,32,32,4.0\n\
bfloat16,40,128,32,6.0\n\
bfloat16,40,32,64,8.0\n\
bfloat16,1,160,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,1,60,4,2,8,0.0\n",
    )
    .unwrap();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 60.0);
}

#[test]
fn moe_queries_extrapolate_token_dimension_for_matching_shape() {
    let fixture = Fixture::new_moe();
    let estimator = fixture.estimator_with_config(moe_engine_config());
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 56.0);
}

#[test]
fn moe_defaults_to_power_law_distribution_when_available() {
    let fixture = Fixture::new_moe();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,60,64,32,0.0\n\
bfloat16,60,32,32,0.0\n\
bfloat16,60,128,32,0.0\n\
bfloat16,60,32,64,0.0\n\
bfloat16,60,4,32,0.0\n\
bfloat16,1,160,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("moe_perf.txt"),
        "moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n\
bfloat16,60,32,64,2,4,1,1,uniform,1.0\n\
bfloat16,60,32,64,2,4,1,1,power_law_1.2,7.0\n",
    )
    .unwrap();
    let estimator = fixture.estimator_with_config(moe_engine_config());
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 14.0);
}

#[test]
fn moe_dtype_selects_moe_specific_perf_rows() {
    let fixture = Fixture::new_moe();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,60,64,32,0.0\n\
bfloat16,60,32,32,0.0\n\
bfloat16,60,128,32,0.0\n\
bfloat16,60,32,64,0.0\n\
bfloat16,60,4,32,0.0\n\
bfloat16,1,160,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("moe_perf.txt"),
        "moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n\
bfloat16,60,32,64,2,4,1,1,power_law_1.2,1.0\n\
w4a16_mxfp4,60,32,64,2,4,1,1,power_law_1.2,7.0\n",
    )
    .unwrap();
    let mut config = moe_engine_config();
    config.moe_dtype = Some(DataType::W4a16Mxfp4);
    let estimator = fixture.estimator_with_config(config);
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 14.0);
}

#[test]
fn moe_non_attention_includes_router_and_dispatch_costs() {
    let fixture = Fixture::new_moe();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,60,64,32,1.0\n\
bfloat16,60,32,32,2.0\n\
bfloat16,60,128,32,0.0\n\
bfloat16,60,32,64,0.0\n\
bfloat16,60,4,32,3.0\n\
bfloat16,1,160,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("moe_perf.txt"),
        "moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n\
bfloat16,120,32,64,2,4,1,2,power_law_1.2,5.0\n",
    )
    .unwrap();
    let nccl_root = fixture.systems_root().join("data/test_sxm/nccl/1.0.0");
    fs::create_dir_all(&nccl_root).unwrap();
    fs::write(
        nccl_root.join("nccl_perf.txt"),
        "op_name,nccl_dtype,num_gpus,message_size,latency\n\
all_gather,half,2,3840,7.0\n\
reduce_scatter,half,2,3840,11.0\n",
    )
    .unwrap();
    let mut config = moe_engine_config();
    config.moe_ep_size = Some(2);
    config.attention_dp_size = Some(2);
    let estimator = fixture.estimator_with_config(config);
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator
        .forward_pass_time_ms(&[metrics.clone(), metrics])
        .unwrap();

    assert_close(latency, 58.0);
}

#[test]
fn moe_dispatch_rejects_invalid_attention_dp_topology() {
    let fixture = Fixture::new_moe();
    let mut config = moe_engine_config();
    config.moe_ep_size = Some(1);
    config.attention_dp_size = Some(2);
    let estimator = fixture.estimator_with_config(config);
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let err = estimator
        .forward_pass_time_ms(&[metrics.clone(), metrics])
        .unwrap_err();

    assert!(err.to_string().contains("invalid MoE dispatch topology"));
}

#[test]
fn attention_dp_rank_count_must_match_config() {
    let fixture = Fixture::new();
    let mut config = engine_config();
    config.attention_dp_size = Some(2);
    let estimator = fixture.estimator_with_config(config);
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let err = estimator.forward_pass_time_ms(&[metrics]).unwrap_err();

    assert!(err.to_string().contains("expected 2 attention-DP rank"));
}

#[test]
fn moe_non_attention_includes_shared_expert_costs() {
    let fixture = Fixture::new_moe();
    fs::write(
        fixture.model_configs_root().join("Test--Moe_config.json"),
        r#"{
  "architectures": ["Qwen2MoeForCausalLM"],
  "model_type": "qwen2_moe",
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 8,
  "hidden_size": 32,
  "intermediate_size": 64,
  "vocab_size": 160,
  "num_experts_per_tok": 2,
  "num_experts": 4,
  "moe_intermediate_size": 64,
  "shared_expert_intermediate_size": 96
}
"#,
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,60,64,32,0.0\n\
bfloat16,60,32,32,0.0\n\
bfloat16,60,128,32,0.0\n\
bfloat16,60,32,64,0.0\n\
bfloat16,60,4,32,0.0\n\
bfloat16,60,96,32,4.0\n\
bfloat16,60,32,96,6.0\n\
bfloat16,1,160,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("moe_perf.txt"),
        "moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n\
bfloat16,60,32,64,2,4,1,1,power_law_1.2,0.0\n",
    )
    .unwrap();
    let estimator = fixture.estimator_with_config(moe_engine_config());
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 20.0);
}

#[test]
fn context_attention_queries_extrapolate_batch_for_matching_shape() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,30,64,32,0.0\n\
bfloat16,30,32,32,0.0\n\
bfloat16,30,128,32,0.0\n\
bfloat16,30,32,64,0.0\n\
bfloat16,3,160,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,1,10,4,2,8,5.0\n\
bfloat16,bfloat16,2,10,4,2,8,10.0\n",
    )
    .unwrap();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 3,
            sum_prefill_tokens: 30,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 30.0);
}

#[test]
fn generation_attention_queries_extrapolate_batch_for_matching_shape() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,3,64,32,0.0\n\
bfloat16,3,32,32,0.0\n\
bfloat16,3,128,32,0.0\n\
bfloat16,3,32,64,0.0\n\
bfloat16,3,160,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("generation_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,step,latency\n\
bfloat16,bfloat16,1,15,4,2,8,1,2.0\n\
bfloat16,bfloat16,2,15,4,2,8,1,4.0\n",
    )
    .unwrap();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests: 3,
            sum_decode_kv_tokens: 45,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 12.0);
}

#[test]
fn gemm_queries_interpolate_matrix_shape_for_matching_tokens() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,60,64,32,0.0\n\
bfloat16,60,32,32,0.0\n\
bfloat16,60,32,64,0.0\n\
bfloat16,1,160,32,0.0\n\
bfloat16,60,64,16,4.0\n\
bfloat16,60,64,48,6.0\n\
bfloat16,60,192,16,8.0\n\
bfloat16,60,192,48,10.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,1,60,4,2,8,0.0\n",
    )
    .unwrap();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 60,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 9.0);
}

#[test]
fn non_attention_includes_custom_allreduce_ops() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,10,32,32,0.0\n\
bfloat16,10,32,16,0.0\n\
bfloat16,10,64,32,0.0\n\
bfloat16,1,80,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,1,10,2,1,8,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("custom_allreduce_perf.txt"),
        "allreduce_dtype,num_gpus,message_size,latency,kernel_source,backend\n\
bfloat16,2,320,1.0,vLLM_custom_graph,vllm_graph\n",
    )
    .unwrap();
    let estimator = fixture.estimator_with_config(EngineConfig {
        tp_size: 2,
        ..engine_config()
    });
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 10,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 5.0);
}

#[test]
fn custom_allreduce_scales_node_local_perf_for_multi_node_tp() {
    let fixture = Fixture::new();
    fs::write(
        fixture.systems_root().join("test_sxm.yaml"),
        "data_dir: data/test_sxm\n\
gpu:\n\
  mem_bw: 1000000000000000000000000000000\n\
  mem_bw_empirical_scaling_factor: 1.0\n\
  mem_empirical_constant_latency: 0.0\n\
node:\n\
  num_gpus_per_node: 2\n\
  inter_node_bw: 50\n\
  intra_node_bw: 100\n\
  p2p_latency: 0.0\n\
misc:\n\
  nccl_version: '1.0.0'\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,10,24,32,0.0\n\
bfloat16,10,32,8,0.0\n\
bfloat16,10,32,32,0.0\n\
bfloat16,10,32,16,0.0\n\
bfloat16,1,40,32,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,1,10,1,1,8,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("custom_allreduce_perf.txt"),
        "allreduce_dtype,num_gpus,message_size,latency,kernel_source,backend\n\
bfloat16,2,320,1.0,vLLM_custom_graph,vllm_graph\n",
    )
    .unwrap();
    let estimator = fixture.estimator_with_config(EngineConfig {
        tp_size: 4,
        ..engine_config()
    });
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 10,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 15.0);
}

#[test]
fn attention_dp_rank_metrics_use_max_rank_attention_workload() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\n\
bfloat16,40,64,32,1.0\n\
bfloat16,40,32,32,2.0\n\
bfloat16,40,128,32,3.0\n\
bfloat16,40,32,64,4.0\n\
bfloat16,2,160,32,0.5\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,1,20,4,2,8,3.0\n\
bfloat16,bfloat16,2,20,4,2,8,7.0\n",
    )
    .unwrap();
    let estimator = fixture.estimator_with_config(EngineConfig {
        attention_dp_size: Some(2),
        ..engine_config()
    });
    let rank0 = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 20,
            ..Default::default()
        },
        ..Default::default()
    };
    let rank1 = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 2,
            sum_prefill_tokens: 40,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[rank0, rank1]).unwrap();

    assert_close(latency, 34.5);
}

#[test]
fn empty_step_returns_zero() {
    let fixture = Fixture::new();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics::default();

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

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
        .forward_pass_time_ms(&[metrics])
        .unwrap_err()
        .to_string();

    assert!(err.contains("unsupported schema version"));
}

#[test]
fn default_fpm_version_matches_constant() {
    assert_eq!(ForwardPassMetrics::default().version, FPM_VERSION);
}

#[test]
fn all_checked_in_model_configs_are_classified_or_best_available_fallback() {
    let root = repo_model_configs_root();
    if !root.is_dir() {
        return;
    }

    let mut checked = 0;
    for entry in fs::read_dir(&root).unwrap() {
        let path = entry.unwrap().path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !file_name.ends_with("_config.json") {
            continue;
        }
        match ModelSpec::load_path(&path) {
            Ok(_) => {}
            Err(_) => {
                let model_name = file_name
                    .strip_suffix("_config.json")
                    .unwrap()
                    .replace("--", "/");
                let model = ForwardPassPerfModel::best_available(
                    EngineConfig {
                        model_name,
                        ..engine_config()
                    },
                    ForwardPassPerfOptions::default(),
                )
                .unwrap();
                assert_eq!(
                    model.diagnostics().source,
                    ForwardPassPerfSource::FallbackRegression
                );
            }
        }
        checked += 1;
    }

    assert!(checked >= 40, "expected checked-in AIC model configs");
}

#[test]
fn forward_pass_perf_options_reject_min_observations_above_max() {
    let err = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        max_observations: 2,
        min_observations: 3,
        bucket_count: 16,
        ..Default::default()
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("min_observations must be <= max_observations"));
}

#[test]
fn forward_pass_perf_options_reject_invalid_correction_bounds() {
    let err = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        max_num_tokens: 0,
        ..Default::default()
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("max_num_tokens must be >= 1"));

    let err = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        max_batch_size: 0,
        ..Default::default()
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("max_batch_size must be >= 1"));

    let err = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        max_kv_tokens: 0,
        ..Default::default()
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("max_kv_tokens must be >= 1"));
}

#[test]
fn forward_pass_perf_best_available_does_not_fallback_on_invalid_schema() {
    let fixture = Fixture::new();
    let err = ForwardPassPerfModel::best_available_with_roots(
        EngineConfig {
            schema_version: ENGINE_CONFIG_SCHEMA_VERSION + 1,
            ..engine_config()
        },
        ForwardPassPerfOptions::default(),
        fixture.systems_root(),
        fixture.model_configs_root(),
    )
    .unwrap_err()
    .to_string();

    assert!(err.contains("unsupported schema version for EngineConfig"));
}

#[test]
fn forward_pass_perf_best_available_falls_back_when_native_is_unsupported() {
    let fixture = Fixture::new();
    fs::write(
        fixture
            .model_configs_root()
            .join("Test--Unsupported_config.json"),
        r#"{
  "architectures": ["UnsupportedForCausalLM"],
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "hidden_size": 32,
  "vocab_size": 160
}
"#,
    )
    .unwrap();

    let model = ForwardPassPerfModel::best_available_with_roots(
        EngineConfig {
            model_name: "Test/Unsupported".to_string(),
            ..engine_config()
        },
        ForwardPassPerfOptions::default(),
        fixture.systems_root(),
        fixture.model_configs_root(),
    )
    .unwrap();

    let diagnostics = model.diagnostics();
    assert_eq!(
        diagnostics.source,
        ForwardPassPerfSource::FallbackRegression
    );
    assert_eq!(
        diagnostics.readiness,
        ForwardPassPerfReadiness::UnsupportedConfig
    );
    assert!(diagnostics
        .last_warning
        .unwrap()
        .contains("fallback regression"));
}

#[test]
fn fallback_regression_returns_none_until_sufficient_data() {
    let model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 3,
        ..Default::default()
    })
    .unwrap();
    let metrics = prefill_fpm(10, 0.01);

    assert_eq!(
        model.estimate_forward_pass_time_ms(&[metrics]).unwrap(),
        None
    );
    assert_eq!(
        model
            .estimate_forward_pass_time_ms(&[ForwardPassMetrics::default()])
            .unwrap(),
        Some(0.0)
    );
}

#[test]
fn fallback_regression_predicts_prefill_decode_and_mixed_workload_kinds() {
    let mut model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 3,
        ..Default::default()
    })
    .unwrap();

    model
        .tune_with_fpms(&[
            vec![prefill_fpm(10, 0.010)],
            vec![prefill_fpm(20, 0.020)],
            vec![prefill_fpm(30, 0.030)],
            vec![decode_fpm(1, 10, 0.007)],
            vec![decode_fpm(2, 10, 0.009)],
            vec![decode_fpm(1, 20, 0.012)],
            vec![mixed_fpm(10, 10, 0.015)],
            vec![mixed_fpm(20, 10, 0.025)],
            vec![mixed_fpm(10, 20, 0.020)],
        ])
        .unwrap();

    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(40, 0.0)])
            .unwrap()
            .unwrap(),
        40.0,
    );
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[decode_fpm(2, 20, 0.0)])
            .unwrap()
            .unwrap(),
        14.0,
    );
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[mixed_fpm(20, 20, 0.0)])
            .unwrap()
            .unwrap(),
        30.0,
    );
}

#[test]
fn fallback_regression_predicts_with_rank_deficient_samples() {
    let mut model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 3,
        ..Default::default()
    })
    .unwrap();

    model
        .tune_with_fpms(&[
            vec![prefill_fpm(10, 0.010)],
            vec![prefill_fpm(10, 0.012)],
            vec![prefill_fpm(10, 0.014)],
        ])
        .unwrap();

    let prediction = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(10, 0.0)])
        .unwrap()
        .unwrap();
    assert!((prediction - 12.0).abs() < 1e-6);
}

#[test]
fn tune_with_fpms_uses_one_rank_feature_vector() {
    let mut model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 1,
        ..Default::default()
    })
    .unwrap();

    model
        .tune_with_fpms(&[vec![prefill_fpm(10, 0.010), decode_fpm(1, 100_000, 0.020)]])
        .unwrap();

    assert!(
        model
            .estimate_forward_pass_time_ms(&[decode_fpm(1, 100_000, 0.0)])
            .unwrap()
            .is_some(),
        "max-rank decode feature should be tuned"
    );
    assert_eq!(
        model
            .estimate_forward_pass_time_ms(&[mixed_fpm(10, 100_000, 0.0)])
            .unwrap(),
        None,
        "rank merge should not synthesize a mixed feature from separate ranks"
    );
}

#[test]
fn tune_with_fpms_handles_multiple_iterations_and_attention_dp_ranks() {
    let mut model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 2,
        ..Default::default()
    })
    .unwrap();

    model
        .tune_with_fpms(&[
            vec![prefill_fpm(10, 0.010), prefill_fpm(20, 0.020)],
            vec![prefill_fpm(30, 0.030), prefill_fpm(40, 0.040)],
        ])
        .unwrap();

    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(60, 0.0)])
            .unwrap()
            .unwrap(),
        60.0,
    );
}

#[test]
fn tuning_ignores_idle_wall_time_and_queued_only_work() {
    let mut model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 2,
        ..Default::default()
    })
    .unwrap();
    let mut queued_only = ForwardPassMetrics::default();
    queued_only.queued_requests.sum_prefill_tokens = 10_000;
    queued_only.wall_time = 1.0;

    model
        .tune_with_fpms(&[
            vec![prefill_fpm(10, 0.0)],
            vec![queued_only],
            vec![prefill_fpm(10, 0.010)],
            vec![prefill_fpm(20, 0.020)],
        ])
        .unwrap();

    assert_eq!(model.diagnostics().retained_observations, 2);
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(30, 0.0)])
            .unwrap()
            .unwrap(),
        30.0,
    );
}

#[test]
fn native_correction_applies_after_bucket_is_ready() {
    let fixture = Fixture::new();
    let mut model = ForwardPassPerfModel::from_native_with_roots(
        engine_config(),
        ForwardPassPerfOptions {
            min_observations: 2,
            ..Default::default()
        },
        fixture.systems_root(),
        fixture.model_configs_root(),
    )
    .unwrap();
    let native_metrics = prefill_fpm(20, 0.0);
    let native_ms = model
        .estimate_forward_pass_time_ms(&[native_metrics.clone()])
        .unwrap()
        .unwrap();
    let metrics = prefill_fpm(20, native_ms * 2.0 / 1000.0);

    assert_eq!(model.min_correction_factor(), None);
    model
        .tune_with_fpms(&[vec![metrics.clone()], vec![metrics.clone()]])
        .unwrap();

    assert_close(
        model
            .estimate_forward_pass_time_ms(&[metrics])
            .unwrap()
            .unwrap(),
        native_ms * 2.0,
    );
    assert_close(model.min_correction_factor().unwrap(), 2.0);
    assert_close(model.max_correction_factor().unwrap(), 2.0);
    assert_close(model.avg_correction_factor().unwrap(), 2.0);
    assert_eq!(
        model.diagnostics().source,
        ForwardPassPerfSource::AicWithCorrection
    );
}

#[test]
fn native_correction_min_observations_is_workload_kind_wide_and_empty_regions_default_to_one() {
    let fixture = Fixture::new();
    let mut model = ForwardPassPerfModel::from_native_with_roots(
        engine_config(),
        ForwardPassPerfOptions {
            min_observations: 2,
            bucket_count: 4,
            max_num_tokens: 100,
            ..Default::default()
        },
        fixture.systems_root(),
        fixture.model_configs_root(),
    )
    .unwrap();

    let native_10 = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(10, 0.0)])
        .unwrap()
        .unwrap();
    let native_30 = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(30, 0.0)])
        .unwrap()
        .unwrap();
    let native_50 = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(50, 0.0)])
        .unwrap()
        .unwrap();
    let native_100 = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(100, 0.0)])
        .unwrap()
        .unwrap();

    model
        .tune_with_fpms(&[
            vec![prefill_fpm(10, native_10 * 2.0 / 1000.0)],
            vec![prefill_fpm(10, native_10 * 2.0 / 1000.0)],
            vec![prefill_fpm(50, native_50 * 3.0 / 1000.0)],
        ])
        .unwrap();

    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(10, 0.0)])
            .unwrap()
            .unwrap(),
        native_10 * 2.0,
    );
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(30, 0.0)])
            .unwrap()
            .unwrap(),
        native_30,
    );
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(50, 0.0)])
            .unwrap()
            .unwrap(),
        native_50 * 3.0,
    );
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(100, 0.0)])
            .unwrap()
            .unwrap(),
        native_100,
    );
    assert_close(model.min_correction_factor().unwrap(), 2.0);
    assert_close(model.max_correction_factor().unwrap(), 3.0);
    assert_close(model.avg_correction_factor().unwrap(), 2.5);
    assert_eq!(model.diagnostics().correction_ready_buckets, 2);
}

#[test]
fn native_correction_uses_configured_bounds_and_ignores_out_of_range_observations() {
    let fixture = Fixture::new();
    let mut model = ForwardPassPerfModel::from_native_with_roots(
        engine_config(),
        ForwardPassPerfOptions {
            min_observations: 2,
            bucket_count: 4,
            max_num_tokens: 40,
            ..Default::default()
        },
        fixture.systems_root(),
        fixture.model_configs_root(),
    )
    .unwrap();

    let native_50 = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(50, 0.0)])
        .unwrap()
        .unwrap();

    model
        .tune_with_fpms(&[
            vec![prefill_fpm(50, native_50 * 2.0 / 1000.0)],
            vec![prefill_fpm(50, native_50 * 2.0 / 1000.0)],
        ])
        .unwrap();

    assert_eq!(model.diagnostics().retained_observations, 0);
    assert_eq!(model.min_correction_factor(), None);
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(50, 0.0)])
            .unwrap()
            .unwrap(),
        native_50,
    );
}

#[test]
fn fallback_regression_has_no_correction_factors() {
    let model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions::default()).unwrap();

    assert_eq!(model.min_correction_factor(), None);
    assert_eq!(model.max_correction_factor(), None);
    assert_eq!(model.avg_correction_factor(), None);
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

#[test]
fn missing_required_system_perf_field_is_reported() {
    let fixture = Fixture::new();
    fs::write(
        fixture.systems_root().join("test_sxm.yaml"),
        "data_dir: data/test_sxm\n\
gpu:\n\
  mem_bw_empirical_scaling_factor: 1.0\n\
  mem_empirical_constant_latency: 0.0\n\
node:\n\
  num_gpus_per_node: 8\n\
  inter_node_bw: 1000000000000\n\
  intra_node_bw: 1000000000000\n\
  p2p_latency: 0.0\n\
misc:\n\
  nccl_version: '1.0.0'\n",
    )
    .unwrap();

    let err = EngineStepEstimator::from_config_with_roots(
        engine_config(),
        fixture.systems_root(),
        fixture.model_configs_root(),
    )
    .unwrap_err()
    .to_string();

    assert!(err.contains("missing gpu.mem_bw"));
    assert!(err.contains("test_sxm.yaml"));
}

#[test]
fn long_prefill_prefix_rescale_avoids_u32_overflow() {
    let fixture = Fixture::new();
    fs::write(
        fixture.perf_dir().join("gemm_perf.txt"),
        "gemm_dtype,m,n,k,latency\nbfloat16,1,1,1,0.0\n",
    )
    .unwrap();
    fs::write(
        fixture.perf_dir().join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,1,131072,4,2,8,4.0\n",
    )
    .unwrap();
    let estimator = fixture.estimator();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 65_536,
            sum_prefill_kv_tokens: 65_536,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 6.0);
}

#[test]
fn omitted_backend_version_uses_numerically_latest_directory() {
    let fixture = Fixture::new();
    let stale_dir = fixture.version_dir("0.5.9");
    let latest_dir = fixture.version_dir("0.5.10");
    copy_perf_files(&fixture.perf_dir(), &stale_dir);
    copy_perf_files(&fixture.perf_dir(), &latest_dir);
    fs::write(fixture.perf_dir().join("INCOMPLETE.txt"), "").unwrap();
    fs::write(
        stale_dir.join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,2,10,4,2,8,100.0\n",
    )
    .unwrap();
    fs::write(
        latest_dir.join("context_attention_perf.txt"),
        "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,2,10,4,2,8,5.0\n",
    )
    .unwrap();
    let mut config = engine_config();
    config.backend_version = None;
    let estimator = EngineStepEstimator::from_config_with_roots(
        config,
        fixture.systems_root(),
        fixture.model_configs_root(),
    )
    .unwrap();
    let metrics = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 2,
            sum_prefill_tokens: 20,
            ..Default::default()
        },
        ..Default::default()
    };

    let latency = estimator.forward_pass_time_ms(&[metrics]).unwrap();

    assert_close(latency, 30.5);
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
            "data_dir: data/test_sxm\n\
gpu:\n\
  mem_bw: 1000000000000000000000000000000\n\
  mem_bw_empirical_scaling_factor: 1.0\n\
  mem_empirical_constant_latency: 0.0\n\
node:\n\
  num_gpus_per_node: 8\n\
  inter_node_bw: 1000000000000000000000000000000\n\
  intra_node_bw: 1000000000000000000000000000000\n\
  p2p_latency: 0.0\n\
misc:\n\
  nccl_version: '1.0.0'\n",
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

    fn new_moe() -> Self {
        let fixture = Self::new();
        fs::write(
            fixture.model_configs_root().join("Test--Moe_config.json"),
            r#"{
  "architectures": ["Qwen2MoeForCausalLM"],
  "model_type": "qwen2_moe",
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "head_dim": 8,
  "hidden_size": 32,
  "intermediate_size": 64,
  "vocab_size": 160,
  "num_experts_per_tok": 2,
  "num_experts": 4,
  "moe_intermediate_size": 64
}
"#,
        )
        .unwrap();
        fs::write(
            fixture.perf_dir().join("gemm_perf.txt"),
            "gemm_dtype,m,n,k,latency\n\
bfloat16,60,64,32,1.0\n\
bfloat16,60,32,32,2.0\n\
bfloat16,60,128,32,100.0\n\
bfloat16,60,32,64,100.0\n\
bfloat16,60,4,32,0.0\n\
bfloat16,1,160,32,0.0\n",
        )
        .unwrap();
        fs::write(
            fixture.perf_dir().join("context_attention_perf.txt"),
            "attn_dtype,kv_cache_dtype,batch_size,isl,num_heads,num_key_value_heads,head_dim,latency\n\
bfloat16,bfloat16,1,60,4,2,8,0.0\n",
        )
        .unwrap();
        fs::write(
            fixture.perf_dir().join("moe_perf.txt"),
            "moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n\
bfloat16,20,32,64,2,4,1,1,uniform,5.0\n\
bfloat16,40,32,64,2,4,1,1,uniform,15.0\n",
        )
        .unwrap();
        fixture
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

    fn version_dir(&self, version: &str) -> PathBuf {
        self.root.join("systems/data/test_sxm/vllm").join(version)
    }

    fn estimator(&self) -> EngineStepEstimator {
        self.estimator_with_config(engine_config())
    }

    fn estimator_with_config(&self, config: EngineConfig) -> EngineStepEstimator {
        EngineStepEstimator::from_config_with_roots(
            config,
            self.systems_root(),
            self.model_configs_root(),
        )
        .unwrap()
    }
}

fn copy_perf_files(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for file_name in [
        "gemm_perf.txt",
        "context_attention_perf.txt",
        "generation_attention_perf.txt",
    ] {
        fs::copy(source.join(file_name), destination.join(file_name)).unwrap();
    }
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
        model_name: "Test/Dense".to_string(),
        model_arch: None,
        system_name: "test_sxm".to_string(),
        backend: BackendKind::Vllm,
        backend_version: Some("1.0.0".to_string()),
        tp_size: 1,
        pp_size: 1,
        moe_tp_size: None,
        moe_ep_size: None,
        attention_dp_size: None,
        weight_dtype: Some(DataType::Bfloat16),
        moe_dtype: None,
        activation_dtype: Some(DataType::Bfloat16),
        kv_cache_dtype: Some(DataType::Bfloat16),
        kv_block_size: None,
        extra: BTreeMap::new(),
    }
}

fn moe_engine_config() -> EngineConfig {
    EngineConfig {
        model_name: "Test/Moe".to_string(),
        moe_tp_size: Some(1),
        moe_ep_size: Some(1),
        ..engine_config()
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "actual={actual}, expected={expected}"
    );
}

fn prefill_fpm(sum_prefill_tokens: u32, wall_time: f64) -> ForwardPassMetrics {
    ForwardPassMetrics {
        wall_time,
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn decode_fpm(
    num_decode_requests: u32,
    sum_decode_kv_tokens: u32,
    wall_time: f64,
) -> ForwardPassMetrics {
    ForwardPassMetrics {
        wall_time,
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests,
            sum_decode_kv_tokens,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn mixed_fpm(
    sum_prefill_tokens: u32,
    sum_decode_kv_tokens: u32,
    wall_time: f64,
) -> ForwardPassMetrics {
    ForwardPassMetrics {
        wall_time,
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens,
            num_decode_requests: 1,
            sum_decode_kv_tokens,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn repo_model_configs_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in manifest_dir.ancestors() {
        let candidate = ancestor.join("src/aiconfigurator/model_configs");
        if candidate.is_dir() {
            return candidate;
        }
    }
    PathBuf::new()
}
