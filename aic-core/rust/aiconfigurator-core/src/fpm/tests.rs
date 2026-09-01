// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::model::{
    IterationFeatures, IterationObservation, RegressionIterationFeatures, WorkloadKind,
};
use super::options::{validate_options, validate_regression_options};
use super::{
    ForwardPassMetrics, ForwardPassPerfModel, ForwardPassPerfOptions, ForwardPassPerfReadiness,
    ForwardPassPerfSource, ForwardPassWorkerType,
};
use crate::common::enums::{FmhaQuantMode, GemmQuantMode, KvCacheQuantMode};
use crate::engine::spec::EngineSpec;
use crate::engine::Engine;
use crate::operators::op::Op;
use crate::operators::{ContextAttentionOp, ElementwiseOp, GemmOp, GenerationAttentionOp};
use crate::perf_database::PerfDatabase;
use crate::{
    AicError, BackendKind, EngineConfig, ParallelMapping, QuantizationConfig,
    ScheduledRequestMetrics,
};

fn systems_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../src/aiconfigurator_core/systems")
}

const TEST_MODEL: &str = "MiniMaxAI/MiniMax-M2.5";

/// Hand-built context op list against the b200_sxm/vllm/0.24.0 perf tables
/// (same fixture pattern as `engine/runtime.rs` and `py.rs` tests).
fn context_ops() -> Vec<Op> {
    vec![
        Op::Elementwise(ElementwiseOp {
            name: "rmsnorm".into(),
            scale_factor: 1.0,
            bytes_per_token: 8192.0,
            scale_num_tokens: 1,
            seq_split: 1,
        }),
        Op::Gemm(GemmOp {
            name: "qkv_gemm".into(),
            scale_factor: 1.0,
            n: 4096,
            k: 4096,
            quant_mode: GemmQuantMode::Fp8Block,
            scale_num_tokens: 0,
            low_precision_input: false,
            seq_split: 1,
            below_grid_sol: false,
        }),
        Op::ContextAttention(ContextAttentionOp {
            name: "context_attention".into(),
            scale_factor: 1.0,
            n: 32,
            n_kv: 8,
            head_size: 128,
            window_size: 0,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            fmha_quant_mode: FmhaQuantMode::Bfloat16,
            use_qk_norm: false,
            cp_size: 1,
            lane_order: crate::operators::attention::b200_vllm_context_lane_order(),
        }),
    ]
}

fn generation_ops() -> Vec<Op> {
    vec![
        Op::Elementwise(ElementwiseOp {
            name: "rmsnorm".into(),
            scale_factor: 1.0,
            bytes_per_token: 8192.0,
            scale_num_tokens: 1,
            seq_split: 1,
        }),
        Op::GenerationAttention(GenerationAttentionOp {
            name: "generation_attention".into(),
            scale_factor: 1.0,
            n: 32,
            n_kv: 8,
            head_size: 128,
            window_size: 0,
            kv_cache_dtype: KvCacheQuantMode::Fp8,
            lane_order: crate::operators::attention::b200_vllm_generation_lane_order(),
        }),
    ]
}

fn fixture_engine_config() -> EngineConfig {
    EngineConfig {
        schema_version: crate::ENGINE_CONFIG_SCHEMA_VERSION,
        model_name: TEST_MODEL.to_string(),
        system_name: "b200_sxm".to_string(),
        systems_path: None,
        backend: BackendKind::Vllm,
        backend_version: Some("0.24.0".to_string()),
        forward_model: None,
        kv_block_size: None,
        parallel: ParallelMapping {
            tp_size: 8,
            pp_size: 1,
            attention_dp_size: Some(1),
            moe_tp_size: Some(1),
            moe_ep_size: Some(8),
            cp_size: None,
        },
        quantization: QuantizationConfig {
            weight_dtype: None,
            moe_dtype: None,
            activation_dtype: None,
            kv_cache_dtype: None,
        },
        speculative: None,
        enable_shared_layer: None,
        strict_provenance: false,
        database_mode: Default::default(),
        tolerate_dirless_version: false,
        transfer_policy: None,
        extra: BTreeMap::new(),
    }
}

/// A native model built from a hand-built fixture `Engine` (NO Python). The
/// public `from_native` constructors compile via Python; `from_engine` lets
/// the pure-Rust tests build the native variant directly.
fn native_model(options: ForwardPassPerfOptions) -> ForwardPassPerfModel {
    let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.24.0").unwrap();
    let spec = EngineSpec::new(fixture_engine_config(), context_ops(), generation_ops());
    let engine = Engine::build(spec, Arc::new(db)).unwrap();
    ForwardPassPerfModel::from_engine(Arc::new(engine), options)
}

fn regression_model(
    worker_type: ForwardPassWorkerType,
    options: ForwardPassPerfOptions,
) -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::from_regression(worker_type, options)
}

fn fixture_engine() -> Arc<Engine> {
    let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.24.0").unwrap();
    let spec = EngineSpec::new(fixture_engine_config(), context_ops(), generation_ops());
    Arc::new(Engine::build(spec, Arc::new(db)).unwrap())
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

fn regression_fpm(
    num_prefill_requests: u32,
    sum_prefill_tokens: u32,
    sum_prefill_kv_tokens: u32,
    num_decode_requests: u32,
    sum_decode_kv_tokens: u32,
    wall_time: f64,
) -> ForwardPassMetrics {
    ForwardPassMetrics {
        wall_time,
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests,
            sum_prefill_tokens,
            sum_prefill_kv_tokens,
            num_decode_requests,
            sum_decode_kv_tokens,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn regression_features(
    worker_type: ForwardPassWorkerType,
    metrics_by_rank: &[ForwardPassMetrics],
    options: &ForwardPassPerfOptions,
) -> Result<Option<[f64; 2]>, AicError> {
    Ok(
        RegressionIterationFeatures::from_metrics(metrics_by_rank, worker_type, options)?
            .map(|feature| feature.x),
    )
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected {expected}, got {actual}"
    );
}

// ---- native feature-selection parity ----

#[test]
fn native_phase_separated_ranks_keep_representative_rank_selection() {
    let prefill_dominant = regression_fpm(1, 2_000, 50_000, 0, 0, 0.0);
    let decode = regression_fpm(0, 0, 0, 4, 1_000, 0.0);

    let ranks = vec![prefill_dominant.clone(), decode.clone()];
    let feature = IterationFeatures::from_metrics(&ranks).unwrap().unwrap();
    assert_eq!(feature.workload_kind, WorkloadKind::Prefill);
    assert_eq!(feature.x, vec![2_000.0]);

    let mut reversed = ranks;
    reversed.reverse();
    let reversed_feature = IterationFeatures::from_metrics(&reversed).unwrap().unwrap();
    assert_eq!(reversed_feature.workload_kind, WorkloadKind::Prefill);
    assert_eq!(reversed_feature.x, vec![2_000.0]);

    let decode_dominant = vec![regression_fpm(1, 900, 75_000, 0, 0, 0.0), decode];
    let feature = IterationFeatures::from_metrics(&decode_dominant)
        .unwrap()
        .unwrap();
    assert_eq!(feature.workload_kind, WorkloadKind::Decode);
    assert_eq!(feature.x, vec![4.0, 1_000.0]);

    let mut reversed = decode_dominant;
    reversed.reverse();
    let reversed_feature = IterationFeatures::from_metrics(&reversed).unwrap().unwrap();
    assert_eq!(reversed_feature.workload_kind, WorkloadKind::Decode);
    assert_eq!(reversed_feature.x, vec![4.0, 1_000.0]);
}

#[test]
fn native_iteration_observation_uses_max_finite_positive_wall_time() {
    let ranks = vec![
        prefill_fpm(10, f64::NAN),
        prefill_fpm(20, f64::INFINITY),
        prefill_fpm(30, -1.0),
        prefill_fpm(40, 0.0),
        prefill_fpm(50, 0.007),
        prefill_fpm(100, 0.001),
    ];
    let observation = IterationObservation::from_metrics(&ranks).unwrap().unwrap();
    assert_eq!(observation.feature.workload_kind, WorkloadKind::Prefill);
    assert_eq!(observation.feature.x, vec![100.0]);
    assert_eq!(observation.wall_time_ms, 7.0);

    let mut reversed = ranks;
    reversed.reverse();
    let reversed_observation = IterationObservation::from_metrics(&reversed)
        .unwrap()
        .unwrap();
    assert_eq!(
        reversed_observation.feature.workload_kind,
        WorkloadKind::Prefill
    );
    assert_eq!(reversed_observation.feature.x, vec![100.0]);
    assert_eq!(reversed_observation.wall_time_ms, 7.0);

    let overflowing_observation =
        IterationObservation::from_metrics(&[prefill_fpm(50, 0.007), prefill_fpm(100, f64::MAX)])
            .unwrap()
            .unwrap();
    assert!(overflowing_observation.wall_time_ms.is_infinite());
}

#[test]
fn native_tuning_preserves_engine_errors_when_wall_time_conversion_overflows() {
    let finite_metrics = regression_fpm(u32::MAX, 1, 0, 0, 0, 0.001);
    let mut overflowing_metrics = finite_metrics.clone();
    overflowing_metrics.wall_time = f64::MAX;

    let mut finite_model = native_model(ForwardPassPerfOptions::default());
    let finite_error = finite_model
        .tune_with_fpms(&[vec![finite_metrics]])
        .unwrap_err();
    assert_eq!(finite_model.diagnostics().retained_observations, 0);

    let mut overflowing_model = native_model(ForwardPassPerfOptions::default());
    let overflowing_error = overflowing_model
        .tune_with_fpms(&[vec![overflowing_metrics]])
        .unwrap_err();
    assert_eq!(overflowing_model.diagnostics().retained_observations, 0);

    assert_eq!(overflowing_error.to_string(), finite_error.to_string());
}

#[test]
fn regression_tuning_does_not_retain_overflowing_wall_time_conversion() {
    let mut model = regression_model(
        ForwardPassWorkerType::Prefill,
        ForwardPassPerfOptions::default(),
    )
    .unwrap();
    let metrics = regression_fpm(u32::MAX, 1, 0, 0, 0, f64::MAX);

    model.tune_with_fpms(&[vec![metrics]]).unwrap();

    assert_eq!(model.diagnostics().retained_observations, 0);
}

// ---- Engine::forward_pass_time_ms dispatch parity ----

/// A prefill-only FPM through `forward_pass_time_ms` must equal the shared
/// `run_context_ops` free fn at the same (batch, isl, prefix). Proves the
/// dispatch port is faithful for the prefill branch.
#[test]
fn forward_pass_prefill_matches_run_context_ops() {
    let engine = fixture_engine();
    let fpm = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 4,
            sum_prefill_tokens: 4 * 1024,
            sum_prefill_kv_tokens: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    let via_fpm = engine.forward_pass_time_ms(&[fpm]).unwrap();
    let direct = crate::session::run_context_ops(
        engine.context_ops_for_test(),
        engine.database(),
        4,
        1024,
        0,
        1.0,
        crate::session::ContextOpFilter::All,
    )
    .unwrap();
    assert_close(via_fpm, direct);
    assert!(via_fpm > 0.0);
}

/// A decode-only FPM through `forward_pass_time_ms` must equal the shared
/// `run_generation_ops_step` free fn at the same (batch, kv_seq).
#[test]
fn forward_pass_decode_matches_run_generation_ops_step() {
    let engine = fixture_engine();
    let fpm = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_decode_requests: 8,
            sum_decode_kv_tokens: 8 * 2048,
            ..Default::default()
        },
        ..Default::default()
    };
    let via_fpm = engine.forward_pass_time_ms(&[fpm]).unwrap();
    let direct = crate::session::run_generation_ops_step(
        engine.generation_ops_for_test(),
        engine.database(),
        8,
        2048,
        1.0,
        false,
    )
    .unwrap();
    assert_close(via_fpm, direct);
    assert!(via_fpm > 0.0);
}

/// Empty FPM list is rejected; an empty-workload FPM yields 0.0 via the
/// model (`estimate_forward_pass_time_ms`).
#[test]
fn forward_pass_empty_inputs() {
    let engine = fixture_engine();
    assert!(engine.forward_pass_time_ms(&[]).is_err());
    let model = native_model(ForwardPassPerfOptions::default());
    assert_eq!(
        model
            .estimate_forward_pass_time_ms(&[ForwardPassMetrics::default()])
            .unwrap(),
        Some(0.0)
    );
}

/// Max across attention-DP ranks: the slowest rank gates the iteration.
#[test]
fn forward_pass_takes_max_across_ranks() {
    let engine = fixture_engine();
    let light = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 128,
            ..Default::default()
        },
        ..Default::default()
    };
    let heavy = ForwardPassMetrics {
        scheduled_requests: ScheduledRequestMetrics {
            num_prefill_requests: 1,
            sum_prefill_tokens: 4096,
            ..Default::default()
        },
        ..Default::default()
    };
    let max_pair = engine
        .forward_pass_time_ms(&[light.clone(), heavy.clone()])
        .unwrap();
    let heavy_only = engine.forward_pass_time_ms(&[heavy]).unwrap();
    assert_close(max_pair, heavy_only);
}

/// Invalid schema version is rejected by the model's estimate path.
#[test]
fn invalid_schema_rejected() {
    let model = native_model(ForwardPassPerfOptions::default());
    let mut bad = prefill_fpm(10, 0.0);
    bad.version = 999;
    assert!(model.estimate_forward_pass_time_ms(&[bad]).is_err());
}

// ---- options validation ----

#[test]
fn options_reject_min_observations_above_max() {
    let err = regression_model(
        ForwardPassWorkerType::Aggregated,
        ForwardPassPerfOptions {
            min_observations: 10,
            max_observations: 5,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, AicError::InvalidEngineConfig(_)));
}

#[test]
fn options_reject_non_square_bucket_count() {
    let err = regression_model(
        ForwardPassWorkerType::Aggregated,
        ForwardPassPerfOptions {
            bucket_count: 7,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, AicError::InvalidEngineConfig(_)));
}

#[test]
fn options_reject_zero_bounds() {
    let err = regression_model(
        ForwardPassWorkerType::Aggregated,
        ForwardPassPerfOptions {
            max_num_tokens: 0,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, AicError::InvalidEngineConfig(_)));
}

#[test]
fn options_default_directional_correction_factors() {
    let defaults = ForwardPassPerfOptions::default();
    assert_eq!(defaults.min_faster_correction_factor, Some(0.5));
    assert_eq!(defaults.max_slower_correction_factor, Some(2.0));

    let omitted: ForwardPassPerfOptions = serde_json::from_str("{}").unwrap();
    assert_eq!(omitted.min_faster_correction_factor, Some(0.5));
    assert_eq!(omitted.max_slower_correction_factor, Some(2.0));

    let unbounded: ForwardPassPerfOptions = serde_json::from_str(
        r#"{
            "min_faster_correction_factor": null,
            "max_slower_correction_factor": null
        }"#,
    )
    .unwrap();
    assert_eq!(unbounded.min_faster_correction_factor, None);
    assert_eq!(unbounded.max_slower_correction_factor, None);

    let no_floor: ForwardPassPerfOptions =
        serde_json::from_str(r#"{"min_faster_correction_factor": null}"#).unwrap();
    assert_eq!(no_floor.min_faster_correction_factor, None);
    assert_eq!(no_floor.max_slower_correction_factor, Some(2.0));

    let no_ceiling: ForwardPassPerfOptions =
        serde_json::from_str(r#"{"max_slower_correction_factor": null}"#).unwrap();
    assert_eq!(no_ceiling.min_faster_correction_factor, Some(0.5));
    assert_eq!(no_ceiling.max_slower_correction_factor, None);
}

#[test]
fn options_validate_directional_correction_factors() {
    let model = regression_model(
        ForwardPassWorkerType::Aggregated,
        ForwardPassPerfOptions {
            min_faster_correction_factor: Some(0.5),
            max_slower_correction_factor: Some(2.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.options().min_faster_correction_factor, Some(0.5));
    assert_eq!(model.options().max_slower_correction_factor, Some(2.0));

    for valid_factor in [f64::MIN_POSITIVE, 1.0] {
        regression_model(
            ForwardPassWorkerType::Aggregated,
            ForwardPassPerfOptions {
                min_faster_correction_factor: Some(valid_factor),
                ..Default::default()
            },
        )
        .unwrap();
    }
    regression_model(
        ForwardPassWorkerType::Aggregated,
        ForwardPassPerfOptions {
            max_slower_correction_factor: Some(1.0),
            ..Default::default()
        },
    )
    .unwrap();

    for invalid_factor in [f64::NEG_INFINITY, -1.0, 0.0, 1.001, f64::INFINITY, f64::NAN] {
        let err = regression_model(
            ForwardPassWorkerType::Aggregated,
            ForwardPassPerfOptions {
                min_faster_correction_factor: Some(invalid_factor),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, AicError::InvalidEngineConfig(_)));
    }

    for invalid_factor in [f64::NEG_INFINITY, -1.0, 0.0, 0.999, f64::INFINITY, f64::NAN] {
        let err = regression_model(
            ForwardPassWorkerType::Aggregated,
            ForwardPassPerfOptions {
                max_slower_correction_factor: Some(invalid_factor),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, AicError::InvalidEngineConfig(_)));
    }
}

// ---- regression-only mode (engine-agnostic) ----

#[test]
fn fallback_regression_returns_none_until_sufficient_data() {
    let model = regression_model(
        ForwardPassWorkerType::Prefill,
        ForwardPassPerfOptions {
            min_observations: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(10, 0.0)])
            .unwrap(),
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
fn regression_worker_type_serialization_uses_exact_public_names() {
    for (worker_type, encoded) in [
        (ForwardPassWorkerType::Prefill, r#""prefill""#),
        (ForwardPassWorkerType::Decode, r#""decode""#),
        (ForwardPassWorkerType::Aggregated, r#""aggregated""#),
    ] {
        assert_eq!(serde_json::to_string(&worker_type).unwrap(), encoded);
        assert_eq!(
            serde_json::from_str::<ForwardPassWorkerType>(encoded).unwrap(),
            worker_type
        );
    }
    assert!(serde_json::from_str::<ForwardPassWorkerType>(r#""agg""#).is_err());
}

#[test]
fn prefill_features_use_critical_attention_then_global_ffn() {
    let ranks = vec![
        regression_fpm(1, 10, 90, 0, 0, 0.0),
        regression_fpm(2, 20, 40, 0, 0, 0.0),
    ];
    let default_options = ForwardPassPerfOptions::default();
    let x = regression_features(ForwardPassWorkerType::Prefill, &ranks, &default_options)
        .unwrap()
        .unwrap();

    // Rank 0: Q = 90*10 + 10^2/2 + 10/2 = 955, so A = 1045.
    // Rank 1: Q = 40*20/2 + 20^2/(2*2) + 20/2 = 510, so A = 550.
    assert_eq!(x, [1045.0, 30.0]);

    let mut reversed = ranks.clone();
    reversed.reverse();
    assert_eq!(
        regression_features(ForwardPassWorkerType::Prefill, &reversed, &default_options,)
            .unwrap()
            .unwrap(),
        x,
        "cross-rank maxima and sums must be permutation invariant"
    );

    let weighted = ForwardPassPerfOptions {
        regression_attention_kv_weight: 2.0,
        regression_prefill_attention_pair_weight: 0.5,
        regression_ffn_token_weight: 3.0,
        ..Default::default()
    };
    assert_eq!(
        regression_features(ForwardPassWorkerType::Prefill, &ranks, &weighted)
            .unwrap()
            .unwrap(),
        [657.5, 90.0]
    );
}

#[test]
fn decode_features_use_critical_attention_then_global_ffn() {
    let ranks = vec![
        regression_fpm(0, 0, 0, 2, 100, 0.0),
        regression_fpm(0, 0, 0, 3, 80, 0.0),
    ];
    let options = ForwardPassPerfOptions::default();
    let x = regression_features(ForwardPassWorkerType::Decode, &ranks, &options)
        .unwrap()
        .unwrap();
    assert_eq!(x, [100.0, 5.0]);
    let mut reversed = ranks.clone();
    reversed.reverse();
    assert_eq!(
        regression_features(ForwardPassWorkerType::Decode, &reversed, &options)
            .unwrap()
            .unwrap(),
        x
    );

    let weighted = ForwardPassPerfOptions {
        regression_attention_kv_weight: 2.0,
        regression_ffn_token_weight: 3.0,
        ..Default::default()
    };
    assert_eq!(
        regression_features(ForwardPassWorkerType::Decode, &ranks, &weighted)
            .unwrap()
            .unwrap(),
        [200.0, 15.0]
    );
}

#[test]
fn aggregated_features_compose_attention_on_the_same_rank() {
    let ranks = vec![
        regression_fpm(1, 10, 90, 2, 20, 0.0),
        regression_fpm(1, 0, 500, 3, 200, 0.0),
    ];
    let options = ForwardPassPerfOptions::default();
    let x = regression_features(ForwardPassWorkerType::Aggregated, &ranks, &options)
        .unwrap()
        .unwrap();

    // Rank 0 contributes H + K + Q = 90 + 20 + 955 = 1065.
    // Rank 1 has P=0, so its H is gated out and it contributes only K=200.
    // The implementation must not combine H/Q from rank 0 with K from rank 1.
    assert_eq!(x, [1065.0, 15.0]);

    let mut reversed = ranks.clone();
    reversed.reverse();
    assert_eq!(
        regression_features(ForwardPassWorkerType::Aggregated, &reversed, &options)
            .unwrap()
            .unwrap(),
        x
    );

    let weighted = ForwardPassPerfOptions {
        regression_attention_kv_weight: 2.0,
        regression_prefill_attention_pair_weight: 0.5,
        regression_ffn_token_weight: 3.0,
        ..Default::default()
    };
    assert_eq!(
        regression_features(ForwardPassWorkerType::Aggregated, &ranks, &weighted)
            .unwrap()
            .unwrap(),
        [697.5, 45.0]
    );
}

#[test]
fn regression_feature_arithmetic_is_finite_at_max_u32_counters() {
    let max = u32::MAX;
    let metrics = regression_fpm(max, max, max, max, max, 0.0);
    let x = regression_features(
        ForwardPassWorkerType::Aggregated,
        &[metrics],
        &ForwardPassPerfOptions::default(),
    )
    .unwrap()
    .unwrap();
    assert!(x.iter().all(|value| value.is_finite() && *value >= 0.0));
    assert!(x[0] > f64::from(max));
    assert_eq!(x[1], 2.0 * f64::from(max));
}

#[test]
fn prefill_feature_arithmetic_converts_before_max_u32_products_and_sums() {
    let max = u32::MAX;
    let ranks = [
        regression_fpm(1, max, max, 0, 0, 0.0),
        regression_fpm(1, max, 0, 0, 0, 0.0),
    ];
    let x = regression_features(
        ForwardPassWorkerType::Prefill,
        &ranks,
        &ForwardPassPerfOptions::default(),
    )
    .unwrap()
    .unwrap();

    let max = f64::from(max);
    let q = max * max + max * max / 2.0 + max / 2.0;
    assert_eq!(x, [max + q, 2.0 * max]);
    assert!(x.iter().all(|value| value.is_finite() && *value >= 0.0));
}

#[test]
fn finite_huge_weights_reject_nonfinite_derived_features_without_retention() {
    let options = ForwardPassPerfOptions {
        regression_attention_kv_weight: f64::MAX,
        regression_prefill_attention_pair_weight: f64::MAX,
        regression_ffn_token_weight: f64::MAX,
        ..Default::default()
    };
    let mut model = regression_model(ForwardPassWorkerType::Prefill, options).unwrap();
    let metrics = regression_fpm(1, 2, 2, 0, 0, 0.010);

    assert!(matches!(
        model.estimate_forward_pass_time_ms(&[metrics.clone()]),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert_eq!(model.diagnostics().retained_observations, 0);

    assert!(matches!(
        model.tune_with_fpms(&[vec![metrics]]),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert_eq!(model.diagnostics().retained_observations, 0);
}

#[test]
fn cached_prefill_metadata_without_tokens_is_idle_or_decode_only() {
    let cached_metadata = regression_fpm(1, 0, 4096, 0, 0, 1.0);
    for worker_type in [
        ForwardPassWorkerType::Prefill,
        ForwardPassWorkerType::Decode,
        ForwardPassWorkerType::Aggregated,
    ] {
        assert_eq!(
            regression_features(
                worker_type,
                &[cached_metadata.clone()],
                &ForwardPassPerfOptions::default(),
            )
            .unwrap(),
            None,
            "H without P or B must not create scheduled work"
        );
    }

    let cached_decode = regression_fpm(1, 0, 4096, 2, 100, 0.0);
    assert_eq!(
        regression_features(
            ForwardPassWorkerType::Decode,
            &[cached_decode.clone()],
            &ForwardPassPerfOptions::default(),
        )
        .unwrap()
        .unwrap(),
        [100.0, 2.0]
    );
    assert_eq!(
        regression_features(
            ForwardPassWorkerType::Aggregated,
            &[cached_decode],
            &ForwardPassPerfOptions::default(),
        )
        .unwrap()
        .unwrap(),
        [100.0, 2.0]
    );
}

#[test]
fn regression_worker_roles_enforce_compatibility() {
    let decode = decode_fpm(1, 100, 0.0);
    let prefill = regression_fpm(1, 10, 20, 0, 0, 0.0);

    assert!(matches!(
        regression_features(
            ForwardPassWorkerType::Prefill,
            &[decode.clone()],
            &ForwardPassPerfOptions::default(),
        ),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert!(matches!(
        regression_features(
            ForwardPassWorkerType::Decode,
            &[prefill.clone()],
            &ForwardPassPerfOptions::default(),
        ),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));

    let aggregated = regression_features(
        ForwardPassWorkerType::Aggregated,
        &[prefill, decode],
        &ForwardPassPerfOptions::default(),
    )
    .unwrap()
    .unwrap();
    assert!(aggregated[0] > 0.0);
    assert_eq!(aggregated[1], 11.0);
}

#[test]
fn regression_public_model_dispatch_enforces_shape_and_role() {
    let mut prefill = regression_model(
        ForwardPassWorkerType::Prefill,
        ForwardPassPerfOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        prefill.estimate_forward_pass_time_ms(&[]),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert!(matches!(
        prefill.tune_with_fpms(&[Vec::new()]),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert!(matches!(
        prefill.estimate_forward_pass_time_ms(&[decode_fpm(1, 100, 0.0)]),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert!(matches!(
        prefill.tune_with_fpms(&[vec![decode_fpm(1, 100, 0.010)]]),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert_eq!(prefill.diagnostics().retained_observations, 0);

    let mut decode = regression_model(
        ForwardPassWorkerType::Decode,
        ForwardPassPerfOptions::default(),
    )
    .unwrap();
    assert!(matches!(
        decode.estimate_forward_pass_time_ms(&[prefill_fpm(10, 0.0)]),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert!(matches!(
        decode.tune_with_fpms(&[vec![prefill_fpm(10, 0.010)]]),
        Err(AicError::InvalidForwardPassMetrics(_))
    ));
    assert_eq!(decode.diagnostics().retained_observations, 0);
}

#[test]
fn aggregated_role_trains_pure_and_mixed_iterations_in_one_store() {
    let options = ForwardPassPerfOptions {
        min_observations: 5,
        ..Default::default()
    };
    let mut model = regression_model(ForwardPassWorkerType::Aggregated, options.clone()).unwrap();
    let mut iterations = vec![
        vec![regression_fpm(1, 8, 16, 0, 0, 0.0)],
        vec![regression_fpm(0, 0, 0, 2, 20, 0.0)],
        vec![regression_fpm(1, 12, 24, 1, 10, 0.0)],
        vec![
            regression_fpm(2, 20, 10, 0, 0, 0.0),
            regression_fpm(0, 0, 0, 3, 50, 0.0),
        ],
        vec![regression_fpm(3, 30, 60, 4, 80, 0.0)],
    ];
    for ranks in &mut iterations {
        let x = regression_features(ForwardPassWorkerType::Aggregated, ranks, &options)
            .unwrap()
            .unwrap();
        let observed_ms = 3.0 + 0.002 * x[0] + 0.5 * x[1];
        let rank_count = ranks.len();
        for (index, metrics) in ranks.iter_mut().enumerate() {
            metrics.wall_time = if index + 1 == rank_count {
                observed_ms / 1000.0
            } else {
                observed_ms / 2000.0
            };
        }
    }

    model.tune_with_fpms(&iterations).unwrap();
    let diagnostics = model.diagnostics();
    assert_eq!(diagnostics.retained_observations, 5);
    assert_eq!(diagnostics.readiness, ForwardPassPerfReadiness::Ready);

    let query = [regression_fpm(2, 18, 35, 2, 40, 0.0)];
    let x = regression_features(ForwardPassWorkerType::Aggregated, &query, &options)
        .unwrap()
        .unwrap();
    let expected = 3.0 + 0.002 * x[0] + 0.5 * x[1];
    let actual = model
        .estimate_forward_pass_time_ms(&query)
        .unwrap()
        .unwrap();
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn prefill_regression_fits_raw_linear_features_not_log_bucket_coordinates() {
    let options = ForwardPassPerfOptions {
        min_observations: 5,
        ..Default::default()
    };
    let mut model = regression_model(ForwardPassWorkerType::Prefill, options.clone()).unwrap();
    let mut iterations = vec![
        vec![regression_fpm(1, 10, 0, 0, 0, 0.0)],
        vec![regression_fpm(1, 20, 0, 0, 0, 0.0)],
        vec![regression_fpm(2, 20, 0, 0, 0, 0.0)],
        vec![regression_fpm(1, 10, 100, 0, 0, 0.0)],
        vec![regression_fpm(2, 30, 50, 0, 0, 0.0)],
        vec![regression_fpm(3, 45, 20, 0, 0, 0.0)],
    ];
    for ranks in &mut iterations {
        let x = regression_features(ForwardPassWorkerType::Prefill, ranks, &options)
            .unwrap()
            .unwrap();
        ranks[0].wall_time = (7.0 + 0.02 * x[0] + 0.4 * x[1]) / 1000.0;
    }
    model.tune_with_fpms(&iterations).unwrap();

    let query = [regression_fpm(2, 24, 30, 0, 0, 0.0)];
    let x = regression_features(ForwardPassWorkerType::Prefill, &query, &options)
        .unwrap()
        .unwrap();
    let expected = 7.0 + 0.02 * x[0] + 0.4 * x[1];
    let actual = model
        .estimate_forward_pass_time_ms(&query)
        .unwrap()
        .unwrap();
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn fallback_regression_keeps_decode_fit_ready_when_ols_kv_slope_is_negative() {
    let mut model = regression_model(
        ForwardPassWorkerType::Decode,
        ForwardPassPerfOptions {
            min_observations: 6,
            ..Default::default()
        },
    )
    .unwrap();

    // Highly correlated decode batch and KV-token features can make
    // unconstrained OLS assign a negative slope to KV tokens even while the
    // combined observed latency trend is positive. A monotonic constrained fit
    // should put that slope on the zero boundary instead of staying unready.
    let observations = [
        (8, 16_000),
        (16, 33_000),
        (24, 47_000),
        (32, 66_000),
        (40, 79_000),
        (48, 98_000),
    ];
    model
        .tune_with_fpms(
            &observations
                .into_iter()
                .map(|(requests, kv_tokens)| {
                    let wall_time_ms =
                        5.0 + 0.07 * f64::from(requests) - 0.00003 * f64::from(kv_tokens);
                    vec![decode_fpm(requests, kv_tokens, wall_time_ms / 1_000.0)]
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();

    assert_eq!(
        model.diagnostics().readiness,
        ForwardPassPerfReadiness::Ready
    );
    let lower_kv = model
        .estimate_forward_pass_time_ms(&[decode_fpm(32, 60_000, 0.0)])
        .unwrap()
        .unwrap();
    let higher_kv = model
        .estimate_forward_pass_time_ms(&[decode_fpm(32, 80_000, 0.0)])
        .unwrap()
        .unwrap();
    assert!(
        higher_kv >= lower_kv,
        "decode estimate must not decrease with KV load: lower={lower_kv}, higher={higher_kv}"
    );

    let fewer_requests = model
        .estimate_forward_pass_time_ms(&[decode_fpm(24, 70_000, 0.0)])
        .unwrap()
        .unwrap();
    let more_requests = model
        .estimate_forward_pass_time_ms(&[decode_fpm(40, 70_000, 0.0)])
        .unwrap()
        .unwrap();
    assert!(
        more_requests > fewer_requests,
        "constrained fit must retain a positive decode load signal: fewer={fewer_requests}, more={more_requests}"
    );
}

#[test]
fn fallback_regression_rejects_intercept_only_fit_when_slopes_are_identifiable() {
    let mut model = regression_model(
        ForwardPassWorkerType::Decode,
        ForwardPassPerfOptions {
            min_observations: 4,
            ..Default::default()
        },
    )
    .unwrap();

    model
        .tune_with_fpms(&[
            vec![decode_fpm(8, 10_000, 0.020)],
            vec![decode_fpm(16, 10_000, 0.018)],
            vec![decode_fpm(8, 20_000, 0.017)],
            vec![decode_fpm(16, 20_000, 0.015)],
        ])
        .unwrap();

    assert_eq!(
        model.diagnostics().readiness,
        ForwardPassPerfReadiness::InsufficientData
    );
    assert_eq!(
        model
            .estimate_forward_pass_time_ms(&[decode_fpm(12, 15_000, 0.0)])
            .unwrap(),
        None
    );
}

#[test]
fn regression_weights_default_and_validate_only_for_regression() {
    let defaults = ForwardPassPerfOptions::default();
    let omitted: ForwardPassPerfOptions = serde_json::from_str("{}").unwrap();
    for options in [&defaults, &omitted] {
        assert_eq!(options.regression_attention_kv_weight, 1.0);
        assert_eq!(options.regression_prefill_attention_pair_weight, 1.0);
        assert_eq!(options.regression_ffn_token_weight, 1.0);
    }

    let invalid_options = [
        ForwardPassPerfOptions {
            regression_attention_kv_weight: 0.0,
            ..Default::default()
        },
        ForwardPassPerfOptions {
            regression_prefill_attention_pair_weight: -1.0,
            ..Default::default()
        },
        ForwardPassPerfOptions {
            regression_ffn_token_weight: f64::NAN,
            ..Default::default()
        },
        ForwardPassPerfOptions {
            regression_attention_kv_weight: f64::INFINITY,
            ..Default::default()
        },
    ];
    for options in invalid_options {
        assert!(matches!(
            regression_model(ForwardPassWorkerType::Aggregated, options),
            Err(AicError::InvalidEngineConfig(_))
        ));
    }

    let invalid_regression_weights = ForwardPassPerfOptions {
        regression_attention_kv_weight: f64::NAN,
        regression_prefill_attention_pair_weight: -1.0,
        regression_ffn_token_weight: f64::INFINITY,
        ..Default::default()
    };
    assert!(
        validate_options(&invalid_regression_weights).is_ok(),
        "native option validation must ignore regression-only weights"
    );
    assert!(matches!(
        validate_regression_options(&invalid_regression_weights),
        Err(AicError::InvalidEngineConfig(_))
    ));

    let metrics = mixed_fpm(32, 4096, 0.0);
    let baseline = native_model(ForwardPassPerfOptions::default())
        .estimate_forward_pass_time_ms(&[metrics.clone()])
        .unwrap()
        .unwrap();
    let native_with_invalid_regression_weights = native_model(ForwardPassPerfOptions {
        regression_attention_kv_weight: f64::NAN,
        regression_prefill_attention_pair_weight: -1.0,
        regression_ffn_token_weight: f64::INFINITY,
        ..Default::default()
    });
    assert_close(
        native_with_invalid_regression_weights
            .estimate_forward_pass_time_ms(&[metrics])
            .unwrap()
            .unwrap(),
        baseline,
    );
}

#[test]
fn native_tuning_and_correction_ignore_regression_weights() {
    let options = ForwardPassPerfOptions {
        min_observations: 2,
        ..Default::default()
    };
    let mut baseline = native_model(options.clone());
    let mut weighted = native_model(ForwardPassPerfOptions {
        regression_attention_kv_weight: 2.0,
        regression_prefill_attention_pair_weight: 3.0,
        regression_ffn_token_weight: 4.0,
        ..options
    });
    let queries = [
        prefill_fpm(20, 0.0),
        decode_fpm(4, 4096, 0.0),
        mixed_fpm(16, 2048, 0.0),
    ];
    let mut iterations = Vec::new();
    for query in &queries {
        let native_ms = baseline
            .estimate_forward_pass_time_ms(&[query.clone()])
            .unwrap()
            .unwrap();
        let mut observation = query.clone();
        observation.wall_time = native_ms * 1.5 / 1000.0;
        iterations.push(vec![observation.clone()]);
        iterations.push(vec![observation]);
    }

    baseline.tune_with_fpms(&iterations).unwrap();
    weighted.tune_with_fpms(&iterations).unwrap();
    assert_eq!(baseline.diagnostics(), weighted.diagnostics());
    for query in queries {
        assert_close(
            baseline
                .estimate_forward_pass_time_ms(&[query.clone()])
                .unwrap()
                .unwrap(),
            weighted
                .estimate_forward_pass_time_ms(&[query])
                .unwrap()
                .unwrap(),
        );
    }
}

#[test]
fn tuning_ignores_idle_wall_time_and_queued_only_work() {
    let mut model = regression_model(
        ForwardPassWorkerType::Prefill,
        ForwardPassPerfOptions {
            min_observations: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let mut queued_only = ForwardPassMetrics::default();
    queued_only.queued_requests.sum_prefill_tokens = 10_000;
    queued_only.wall_time = 1.0;
    let cached_only = regression_fpm(1, 0, 10_000, 0, 0, 1.0);

    model
        .tune_with_fpms(&[
            vec![prefill_fpm(10, 0.0)],
            vec![prefill_fpm(30, f64::MAX)],
            vec![queued_only],
            vec![cached_only],
            vec![prefill_fpm(10, 0.010)],
            vec![prefill_fpm(20, 0.020)],
        ])
        .unwrap();

    assert_eq!(model.diagnostics().retained_observations, 2);
}

#[test]
fn fallback_regression_has_no_correction_factors() {
    let model = regression_model(
        ForwardPassWorkerType::Aggregated,
        ForwardPassPerfOptions {
            min_faster_correction_factor: Some(0.5),
            max_slower_correction_factor: Some(2.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(model.min_correction_factor(), None);
    assert_eq!(model.max_correction_factor(), None);
    assert_eq!(model.avg_correction_factor(), None);
    assert_eq!(
        model.diagnostics().source,
        ForwardPassPerfSource::FallbackRegression
    );
}

// ---- native correction (Engine-backed; uses the fixture Engine) ----

/// After a correction bucket is ready, the native estimate is multiplied by
/// the learned median observed/native ratio. Drives the ratio off the
/// model's own native estimate so the factor is exactly 2.0.
#[test]
fn native_correction_applies_after_bucket_is_ready() {
    let mut model = native_model(ForwardPassPerfOptions {
        min_observations: 2,
        ..Default::default()
    });
    let native_metrics = prefill_fpm(20, 0.0);
    let native_ms = model
        .estimate_forward_pass_time_ms(&[native_metrics.clone()])
        .unwrap()
        .unwrap();
    // wall_time is in seconds; observed_ms = wall_time * 1000 = native_ms*2.
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

/// The configured ceiling is absolute relative to the native estimate. It does
/// not compound as matching outliers are added to an already-corrected bucket.
#[test]
fn native_slower_correction_ceiling_is_absolute_across_repeated_outliers() {
    let mut model = native_model(ForwardPassPerfOptions {
        min_observations: 2,
        max_slower_correction_factor: Some(2.0),
        ..Default::default()
    });
    let metrics = prefill_fpm(20, 0.0);
    let native_ms = model
        .estimate_forward_pass_time_ms(&[metrics])
        .unwrap()
        .unwrap();
    let outlier = prefill_fpm(20, native_ms * 90.0 / 1000.0);

    model
        .tune_with_fpms(&[vec![outlier.clone()], vec![outlier.clone()]])
        .unwrap();
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(20, 0.0)])
            .unwrap()
            .unwrap(),
        native_ms * 2.0,
    );

    model
        .tune_with_fpms(&[
            vec![outlier.clone()],
            vec![outlier.clone()],
            vec![outlier.clone()],
            vec![outlier],
        ])
        .unwrap();
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(20, 0.0)])
            .unwrap()
            .unwrap(),
        native_ms * 2.0,
    );
    assert_close(model.min_correction_factor().unwrap(), 2.0);
    assert_close(model.max_correction_factor().unwrap(), 2.0);
    assert_close(model.avg_correction_factor().unwrap(), 2.0);
}

/// Saturating the slower ceiling does not make a correction monotonic. Lower
/// observations move the retained-sample median down as capped samples age out.
#[test]
fn native_correction_recovers_from_saturated_slower_ceiling() {
    let mut model = native_model(ForwardPassPerfOptions {
        min_observations: 2,
        max_observations: 4,
        max_slower_correction_factor: Some(2.0),
        ..Default::default()
    });
    let metrics = prefill_fpm(20, 0.0);
    let native_ms = model
        .estimate_forward_pass_time_ms(&[metrics])
        .unwrap()
        .unwrap();
    let outlier = prefill_fpm(20, native_ms * 90.0 / 1000.0);
    let recovered = prefill_fpm(20, native_ms / 1000.0);

    model
        .tune_with_fpms(&[
            vec![outlier.clone()],
            vec![outlier.clone()],
            vec![outlier.clone()],
            vec![outlier],
        ])
        .unwrap();
    assert_close(model.max_correction_factor().unwrap(), 2.0);

    model
        .tune_with_fpms(&[vec![recovered.clone()], vec![recovered.clone()]])
        .unwrap();
    assert_close(model.max_correction_factor().unwrap(), 1.5);

    model
        .tune_with_fpms(&[vec![recovered.clone()], vec![recovered]])
        .unwrap();
    assert_close(model.max_correction_factor().unwrap(), 1.0);
}

/// Default directional limits are applied to each correction sample before
/// taking the median, retaining observations inside either bound.
#[test]
fn native_default_directional_correction_bounds_are_applied_at_observation_ingestion() {
    let mut model = native_model(ForwardPassPerfOptions {
        min_observations: 2,
        ..Default::default()
    });
    let metrics = prefill_fpm(20, 0.0);
    let native_ms = model
        .estimate_forward_pass_time_ms(&[metrics])
        .unwrap()
        .unwrap();

    model
        .tune_with_fpms(&[
            vec![prefill_fpm(20, native_ms * 0.1 / 1000.0)],
            vec![prefill_fpm(20, native_ms * 100.0 / 1000.0)],
        ])
        .unwrap();

    // The stored samples are [0.5, 2.0], whose median is 1.25. Applying bounds
    // only after taking the raw [0.1, 100.0] median would produce 2.0.
    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(20, 0.0)])
            .unwrap()
            .unwrap(),
        native_ms * 1.25,
    );
    assert_close(model.min_correction_factor().unwrap(), 1.25);
    assert_close(model.max_correction_factor().unwrap(), 1.25);
    assert_close(model.avg_correction_factor().unwrap(), 1.25);
}

/// An upper ceiling does not clamp genuine observations below the native
/// estimate.
#[test]
fn native_slower_correction_ceiling_preserves_faster_corrections() {
    let mut model = native_model(ForwardPassPerfOptions {
        min_observations: 2,
        min_faster_correction_factor: None,
        max_slower_correction_factor: Some(2.0),
        ..Default::default()
    });
    let metrics = prefill_fpm(20, 0.0);
    let native_ms = model
        .estimate_forward_pass_time_ms(&[metrics])
        .unwrap()
        .unwrap();
    let faster = prefill_fpm(20, native_ms * 0.5 / 1000.0);

    model
        .tune_with_fpms(&[vec![faster.clone()], vec![faster]])
        .unwrap();

    assert_close(
        model
            .estimate_forward_pass_time_ms(&[prefill_fpm(20, 0.0)])
            .unwrap()
            .unwrap(),
        native_ms * 0.5,
    );
    assert_close(model.max_correction_factor().unwrap(), 0.5);
}

/// A faster-correction floor bounds only ratios below one and leaves slower
/// corrections unbounded when no slower ceiling is configured.
#[test]
fn native_faster_correction_floor_is_independent() {
    let options = ForwardPassPerfOptions {
        min_observations: 2,
        min_faster_correction_factor: Some(0.5),
        max_slower_correction_factor: None,
        ..Default::default()
    };

    let mut faster_model = native_model(options.clone());
    let faster_metrics = prefill_fpm(20, 0.0);
    let faster_native_ms = faster_model
        .estimate_forward_pass_time_ms(&[faster_metrics])
        .unwrap()
        .unwrap();
    let faster = prefill_fpm(20, faster_native_ms * 0.1 / 1000.0);
    faster_model
        .tune_with_fpms(&[vec![faster.clone()], vec![faster]])
        .unwrap();
    assert_close(faster_model.min_correction_factor().unwrap(), 0.5);

    let mut slower_model = native_model(options);
    let slower_metrics = prefill_fpm(20, 0.0);
    let slower_native_ms = slower_model
        .estimate_forward_pass_time_ms(&[slower_metrics])
        .unwrap()
        .unwrap();
    let slower = prefill_fpm(20, slower_native_ms * 8.0 / 1000.0);
    slower_model
        .tune_with_fpms(&[vec![slower.clone()], vec![slower]])
        .unwrap();
    assert_close(slower_model.max_correction_factor().unwrap(), 8.0);
}

/// min_observations is workload-kind-wide; empty in-range regions keep the
/// default factor 1.0. Two distinct prefill buckets get distinct factors.
#[test]
fn native_correction_min_observations_is_workload_kind_wide_and_empty_regions_default_to_one() {
    let mut model = native_model(ForwardPassPerfOptions {
        min_observations: 2,
        bucket_count: 4,
        max_num_tokens: 100,
        max_slower_correction_factor: None,
        ..Default::default()
    });

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
    // 30 lives in an empty in-range region: factor 1.0.
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
    assert_close(model.min_correction_factor().unwrap(), 2.0);
    assert_close(model.max_correction_factor().unwrap(), 3.0);
    assert_close(model.avg_correction_factor().unwrap(), 2.5);
    assert_eq!(model.diagnostics().correction_ready_buckets, 2);
}

/// Observations outside the configured correction-grid workload ranges are ignored.
#[test]
fn native_correction_uses_configured_bounds_and_ignores_out_of_range_observations() {
    let mut model = native_model(ForwardPassPerfOptions {
        min_observations: 2,
        bucket_count: 4,
        max_num_tokens: 40,
        ..Default::default()
    });

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

/// A fresh native model reports source = Aic (no correction yet) and is
/// Ready.
#[test]
fn native_model_starts_ready_with_aic_source() {
    let model = native_model(ForwardPassPerfOptions::default());
    let diag = model.diagnostics();
    assert_eq!(diag.source, ForwardPassPerfSource::Aic);
    assert_eq!(diag.readiness, ForwardPassPerfReadiness::Ready);
    assert_eq!(diag.retained_observations, 0);
}
