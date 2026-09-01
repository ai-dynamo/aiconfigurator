// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time contract tests from an external crate's point of view.

use std::path::Path;

use aiconfigurator_core::{
    AicEngine, AicEngineBuilder, AicError, BackendKind, EngineConfig, ForwardPassPerfModel,
    ForwardPassPerfOptions, ForwardPassWorkerType, KvCacheEstimateRequest,
};

/// Compile the ergonomic engine builder without starting embedded Python.
pub fn configured_builder() -> AicEngineBuilder {
    AicEngineBuilder::new("Qwen/Qwen3-32B", "h200_sxm", BackendKind::Vllm)
        .backend_version("0.10.2")
        .tp_size(2)
        .pp_size(1)
        .attention_dp_size(1)
        .moe_parallelism(None, None)
        .gemm_quant_mode("bfloat16")
        .moe_quant_mode("bfloat16")
        .kvcache_quant_mode("bfloat16")
        .fmha_quant_mode("bfloat16")
        .comm_quant_mode("bfloat16")
        .speculative_decoding(0)
        .kv_block_size(16)
        .systems_path("/tmp/systems")
}

/// Compile the builder's terminal operation as an external consumer would.
/// The function is intentionally not called by the tests because it embeds
/// Python and needs installed model/system data.
pub fn build_engine(builder: AicEngineBuilder) -> Result<AicEngine, AicError> {
    builder.build()
}

/// Compile the forward-pass model's public constructor and telemetry type.
pub fn regression_model() -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::from_regression(ForwardPassWorkerType::Aggregated, regression_options())
}

/// Construct and expose every public regression-weight option from an external crate.
pub fn regression_options() -> ForwardPassPerfOptions {
    ForwardPassPerfOptions {
        regression_attention_kv_weight: 2.0,
        regression_prefill_attention_pair_weight: 3.0,
        regression_ffn_token_weight: 4.0,
        ..ForwardPassPerfOptions::default()
    }
}

/// Compile the fallback-capable constructor with its required worker type.
/// This is not called because native construction embeds Python.
pub fn best_available_model(config: EngineConfig) -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::best_available(
        config,
        ForwardPassWorkerType::Decode,
        ForwardPassPerfOptions::default(),
    )
}

/// Compile the explicit-systems-root constructor from an external crate.
pub fn best_available_model_with_roots(
    config: EngineConfig,
    systems_root: impl AsRef<Path>,
) -> Result<ForwardPassPerfModel, AicError> {
    ForwardPassPerfModel::best_available_with_roots(
        config,
        ForwardPassWorkerType::Prefill,
        ForwardPassPerfOptions::default(),
        systems_root,
    )
}

/// Keep the KV request type in the external-consumer contract without
/// constructing an environment-dependent estimate.
pub fn accept_kv_request(request: KvCacheEstimateRequest) -> KvCacheEstimateRequest {
    request
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiconfigurator_core::engine::{Engine, PerOpValue, RuntimeConfig, StaticMode};
    use aiconfigurator_core::{
        ForwardPassMetrics, ENGINE_CONFIG_SCHEMA_VERSION, ENGINE_SPEC_SCHEMA_VERSION, FPM_VERSION,
    };

    type PerOpResult = Result<Vec<PerOpValue>, AicError>;
    type StaticPerOpResult = Result<(Vec<PerOpValue>, Vec<PerOpValue>), AicError>;
    type MixedPerOpResult = Result<(Vec<PerOpValue>, Vec<PerOpValue>, Vec<PerOpValue>), AicError>;

    #[test]
    fn schema_constants_and_metric_defaults_are_public() {
        assert_eq!(ENGINE_CONFIG_SCHEMA_VERSION, 1);
        // v5: MlaModuleOp gained native_num_heads (#1458).
        // v6: Kda op variant appended (Kimi-K3; renumbered at the merge).
        // v7: MoEDispatchOp gained attn_ar_modeled.
        // v8: GemmOp gained below_grid_sol.
        // v9: FpmForward whole-model variant appended (renumbered at each
        // merge from concurrent claims of v5/v7/v8).
        // v10: MhcModuleOp gained seq_split (issue #1498; renumbered at the
        // rebase from a concurrent claim of v7).
        // v11: wideEP MoE variants removed, MoeAllToAll/MoeExpertCompute
        // appended after FpmForward; MoeExpertComputeOp gained enable_eplb
        // (AIC-1601).
        // v12: DsaModuleOp gained attn_projection_quant_modes (PR-6 weight
        // physics) — a positional bincode op-layout change.
        // v13: the engine owns shared-layer source resolution — EngineConfig
        // dropped the Python-resolved perf_db_sources map (a bincode
        // config-layout change) for enable_shared_layer + strict_provenance
        // (deprecation-cleanup PR).
        assert_eq!(ENGINE_SPEC_SCHEMA_VERSION, 13);
        assert_eq!(FPM_VERSION, 1);
        assert_eq!(ForwardPassMetrics::default().version, FPM_VERSION);
    }

    #[test]
    fn ergonomic_builder_is_available_to_external_crates() {
        let _builder = configured_builder();
    }

    #[test]
    fn regression_constructor_is_environment_independent() {
        let _model = regression_model().expect("construct regression model");
        let _roles = [
            ForwardPassWorkerType::Prefill,
            ForwardPassWorkerType::Decode,
            ForwardPassWorkerType::Aggregated,
        ];
    }

    #[test]
    fn regression_weight_fields_are_public() {
        let options = regression_options();
        assert_eq!(options.regression_attention_kv_weight, 2.0);
        assert_eq!(options.regression_prefill_attention_pair_weight, 3.0);
        assert_eq!(options.regression_ffn_token_weight, 4.0);
    }

    #[test]
    fn per_op_engine_interface_remains_four_field_tuples() {
        let value: PerOpValue = ("op".to_string(), 1.0, 0.0, "silicon");
        let (_name, _latency_ms, _energy_wms, _source): (String, f64, f64, &'static str) = value;

        let _: fn(&Engine, &RuntimeConfig, StaticMode, u32) -> StaticPerOpResult =
            Engine::run_static_per_op;
        let _: fn(&Engine, u32, u32, u32, u32, u32, f64, f64) -> MixedPerOpResult =
            Engine::mixed_step_breakdown_per_op;
        let _: fn(&Engine, u32, u32, u32, f64) -> PerOpResult = Engine::decode_step_per_op;
        let _: fn(&Engine, &[usize], u32, u32, u32, f64, Option<u32>) -> PerOpResult =
            Engine::evaluate_context_ops;
        let _: fn(&Engine, &[usize], u32, u32, f64, u32, Option<u32>) -> PerOpResult =
            Engine::evaluate_generation_ops;
        let _: fn(&Engine, &str, bool, u32, u32, u32, f64, Option<u32>) -> PerOpResult =
            Engine::evaluate_ops_json;
    }
}
