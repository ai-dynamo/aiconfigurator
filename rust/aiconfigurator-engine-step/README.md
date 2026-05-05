# AIConfigurator Forward Pass Estimator

Phase 1 Rust sidecar for estimating forward-pass latency from AIC model metadata and perf files.

This crate intentionally does not change the existing Python SDK. It gives Rust callers, especially Dynamo Mocker, a reusable estimator that loads metadata once and then serves per-iteration estimates without Python/GIL overhead.

The v1 input is AIC-owned Rust `ForwardPassMetrics`, aligned with Dynamo FPM v1. That keeps this first iteration close to existing Dynamo telemetry while avoiding a direct AIC dependency on Dynamo crates.

```rust
use aiconfigurator_engine_step::{
    create_engine_step_estimator, BackendKind, DataType, EngineConfig, ForwardPassMetrics,
    ScheduledRequestMetrics, ENGINE_CONFIG_SCHEMA_VERSION,
};

let estimator = create_engine_step_estimator(EngineConfig {
    schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
    model_name: "Qwen/Qwen3-8B".to_string(),
    model_arch: None,
    max_sequence_length: None,
    system_name: "h100_sxm".to_string(),
    backend: BackendKind::Vllm,
    backend_version: Some("0.19.0".to_string()),
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
    extra: Default::default(),
})?;

let latency = estimator.forward_pass_time(&ForwardPassMetrics {
    scheduled_requests: ScheduledRequestMetrics {
        num_decode_requests: 1,
        sum_decode_kv_tokens: 2048,
        ..Default::default()
    },
    ..Default::default()
})?;
```

Current scope:

- Dense transformer prefill, decode, and mixed prefill plus decode steps.
- FPM v1 aggregate scheduled request fields as the estimator input.
- AIC-style Hugging Face model config JSON files.
- AIC `gemm_perf.txt`, `context_attention_perf.txt`, and `generation_attention_perf.txt` CSV files.
- Explicit Git LFS pointer detection for perf data that has not been pulled locally.

Known Phase 1 limits:

- Request-level FPM v2 fields, KV block modeling, MoE, MLA, P2P pipeline composition, WideEP accuracy work, and Python bindings are intentionally left for later PRs.
- Tests use fixture perf files so they can run without the large AIC perf databases.
