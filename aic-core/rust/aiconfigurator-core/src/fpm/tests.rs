// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::{
    ForwardPassMetrics, ForwardPassPerfModel, ForwardPassPerfOptions, ForwardPassPerfReadiness,
    ForwardPassPerfSource,
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

/// Hand-built context op list against the b200_sxm/vllm/0.19.0 perf tables
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
        backend_version: Some("0.19.0".to_string()),
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
        perf_db_sources: Default::default(),
        database_mode: Default::default(),
        transfer_policy: None,
        extra: BTreeMap::new(),
    }
}

/// A native model built from a hand-built fixture `Engine` (NO Python). The
/// public `from_native` constructors compile via Python; `from_engine` lets
/// the pure-Rust tests build the native variant directly.
fn native_model(options: ForwardPassPerfOptions) -> ForwardPassPerfModel {
    let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0").unwrap();
    let spec = EngineSpec::new(fixture_engine_config(), context_ops(), generation_ops());
    let engine = Engine::build(spec, Arc::new(db)).unwrap();
    ForwardPassPerfModel::from_engine(Arc::new(engine), options)
}

fn fixture_engine() -> Arc<Engine> {
    let db = PerfDatabase::load(&systems_root(), "b200_sxm", "vllm", "0.19.0").unwrap();
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

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "expected {expected}, got {actual}"
    );
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
    let err = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 10,
        max_observations: 5,
        ..Default::default()
    })
    .unwrap_err();
    assert!(matches!(err, AicError::InvalidEngineConfig(_)));
}

#[test]
fn options_reject_non_square_bucket_count() {
    let err = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        bucket_count: 7,
        ..Default::default()
    })
    .unwrap_err();
    assert!(matches!(err, AicError::InvalidEngineConfig(_)));
}

#[test]
fn options_reject_zero_bounds() {
    let err = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        max_num_tokens: 0,
        ..Default::default()
    })
    .unwrap_err();
    assert!(matches!(err, AicError::InvalidEngineConfig(_)));
}

// ---- regression-only mode (engine-agnostic) ----

#[test]
fn fallback_regression_returns_none_until_sufficient_data() {
    let model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 3,
        ..Default::default()
    })
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
fn fallback_regression_prefill_weighted_hinge_avoids_small_token_collapse() {
    let mut model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions {
        min_observations: 6,
        max_observations: 64,
        ..Default::default()
    })
    .unwrap();

    // Production-shaped prefill observations: low-token passes are dominated by
    // fixed overhead, while larger passes steepen. A single global line can
    // drive the intercept negative and collapse small-token predictions to the
    // 1e-6 ms floor.
    model
        .tune_with_fpms(&[
            vec![prefill_fpm(59, 0.0317)],
            vec![prefill_fpm(120, 0.0450)],
            vec![prefill_fpm(1_500, 0.0400)],
            vec![prefill_fpm(5_000, 0.0540)],
            vec![prefill_fpm(8_000, 0.1000)],
            vec![prefill_fpm(12_000, 0.1300)],
            vec![prefill_fpm(20_000, 0.2260)],
            vec![prefill_fpm(50_000, 0.7290)],
            vec![prefill_fpm(100_000, 1.5680)],
        ])
        .unwrap();

    let small = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(1_000, 0.0)])
        .unwrap()
        .unwrap();
    let mid = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(12_000, 0.0)])
        .unwrap()
        .unwrap();
    let large = model
        .estimate_forward_pass_time_ms(&[prefill_fpm(100_000, 0.0)])
        .unwrap()
        .unwrap();

    assert!(
        small > 1.0,
        "small prefill should not collapse to the 1e-6 ms floor, got {small}"
    );
    assert!(
        small < mid && mid < large,
        "prefill estimates should remain monotonic: small={small}, mid={mid}, large={large}"
    );
    assert!(
        large > 1_000.0,
        "large prefill should stay in the observed seconds-scale regime, got {large} ms"
    );
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
fn fallback_regression_has_no_correction_factors() {
    let model = ForwardPassPerfModel::from_regression(ForwardPassPerfOptions::default()).unwrap();
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

/// min_observations is workload-kind-wide; empty in-range regions keep the
/// default factor 1.0. Two distinct prefill buckets get distinct factors.
#[test]
fn native_correction_min_observations_is_workload_kind_wide_and_empty_regions_default_to_one() {
    let mut model = native_model(ForwardPassPerfOptions {
        min_observations: 2,
        bucket_count: 4,
        max_num_tokens: 100,
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

/// Observations outside the configured correction bounds are ignored.
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
