# AIConfigurator Rust Core

Rust core for estimating forward-pass latency from AIC model metadata and perf files. Distributed two ways:

- **As the `aiconfigurator-core` Python wheel** — built by [maturin](https://www.maturin.rs/) from this directory's `pyproject.toml`, exposing a PyO3 extension module as the package `aiconfigurator_core` (compiled submodule `aiconfigurator_core._core`). Consumed by the main `aiconfigurator` package via the optional `[rust]` extra (`pip install aiconfigurator[rust]`). Bare `pip install aiconfigurator` keeps working without it; the SDK falls back to its Python latency path. PyO3's `abi3-py310` stable-ABI feature means a single compiled artifact works on every CPython ≥ 3.10.
- **As a regular Rust crate** — the `rlib` artifact for Rust consumers like Dynamo Mocker. The raw C ABI in `src/ffi.rs` is retained for non-Python consumers and is not exposed through the Python wrapper.

End users of `aiconfigurator-core` don't need a Rust toolchain — they install the precompiled wheel. Running `cargo build` directly is for crate development only.

The v1 input is a per-attention-DP-rank list of AIC-owned Rust `ForwardPassMetrics`, aligned with Dynamo FPM v1. That keeps this first iteration close to existing Dynamo telemetry while avoiding a direct AIC dependency on Dynamo crates.

```rust
use aiconfigurator_core::{
    create_engine_step_estimator, BackendKind, DataType, EngineConfig, ForwardPassMetrics,
    ScheduledRequestMetrics, ENGINE_CONFIG_SCHEMA_VERSION,
};

let estimator = create_engine_step_estimator(EngineConfig {
    schema_version: ENGINE_CONFIG_SCHEMA_VERSION,
    model_name: "Qwen/Qwen3-8B".to_string(),
    model_arch: None,
    system_name: "h100_sxm".to_string(),
    backend: BackendKind::Vllm,
    backend_version: Some("0.19.0".to_string()),
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
    extra: Default::default(),
})?;

let latency = estimator.forward_pass_time(&[ForwardPassMetrics {
    scheduled_requests: ScheduledRequestMetrics {
        num_decode_requests: 1,
        sum_decode_kv_tokens: 2048,
        ..Default::default()
    },
    ..Default::default()
}])?;
```

Current scope:

- Prefill, decode, and mixed prefill plus decode steps for all model families represented in AIC's checked-in model configs.
- FPM v1 aggregate scheduled request fields per attention-DP rank as the estimator input.
- AIC-style Hugging Face model config JSON files.
- AIC `gemm_perf.txt`, `context_attention_perf.txt`, `generation_attention_perf.txt`, `moe_perf.txt`, `context_mla_perf.txt`, and `generation_mla_perf.txt` CSV files.
- Explicit Git LFS pointer detection for perf data that has not been pulled locally.
- A PyO3 extension module exposing the estimator to Python as `aiconfigurator_core` (compiled submodule `aiconfigurator_core._core`; shipped as the `aiconfigurator-core` wheel; consumed by `aiconfigurator` via the `[rust]` extra), plus a parallel C ABI in `src/ffi.rs` for non-Python consumers.

Known Phase 1 limits:

- Request-level FPM v2 fields, KV block modeling, exact Python parity for every specialized op, P2P pipeline composition, WideEP accuracy work, and production Python bindings are intentionally left for later PRs.
- Tests use fixture perf files so they can run without the large AIC perf databases.
