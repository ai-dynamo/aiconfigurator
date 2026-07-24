# AIConfigurator Rust Core

Rust-native core that estimates prefill, decode, and forward-pass latency from
AIC model metadata and perf files, without re-entering Python on the hot path.

This crate intentionally does not change the existing Python SDK. It gives Rust
callers — especially the Dynamo Mocker and planner — a reusable estimator that
loads metadata once and then serves per-iteration latency estimates with the
GIL never acquired.

## How it works

The **compiled-engine path is the only supported entry point**. Python's
`compile_engine` walks the model once and emits an `EngineSpec` (the op lists
plus an `EngineConfig` identity); the Rust `Engine` then executes that spec over
the loaded perf database natively.

> Note that "compile" here produces a serialized `EngineSpec` (an op-list plan),
not a native binary. Python builds the spec as JSON, it is re-encoded to bincode
bytes as the Python → Rust wire format, and the Rust `Engine` deserializes and
interprets it — closer to a compiled query plan than a compiled executable. The
one-time compile just resolves the model into a fixed, serializable op list so
the hot path never re-walks the model or re-enters Python.

- `AicEngineBuilder` is the preferred Rust → Python (`compile_engine`) → Rust
  entry point for callers in other crates. The flat `build_aic_engine(...)`
  function remains source-compatible through the 0.10 release for existing
  consumers and is planned for removal in version 0.11.0. Both normalize into
  one private build request and embed a Python interpreter only for the one-time
  compile step.
- The returned `AicEngine` exposes GIL-free inherent methods
  (`prefill_latency_ms`, `decode_latency_ms`) for the pure-Rust hot path, plus
  `#[pymethods]` wrappers and an FPM-aggregate `estimate_forward_pass_time_ms`
  for Python callers.
- `ForwardPassPerfModel` layers online correction, a regression fallback, and
  readiness/diagnostics on top of the compiled `Engine`, and is re-exported for
  native embedders and exposed to Python as `RustForwardPassPerfModel`.

Standalone binaries that call `AicEngineBuilder::build()` must enable the
crate's `embed-python` feature, which enables PyO3's `auto-initialize` support:

```toml
[dependencies]
aiconfigurator-core = { version = "0.10.0", features = ["embed-python"] }
```

Applications that embed Python in an existing host may initialize the
interpreter themselves instead. In either case, the matching
`aiconfigurator-core` Python wheel and its bundled data must be available to the
embedded interpreter.

```rust,no_run
use aiconfigurator_core::{AicEngineBuilder, BackendKind};

// Rust -> Python (compile_engine) -> Rust (Engine build + perf-DB load).
let engine = AicEngineBuilder::new(
    "Qwen/Qwen3-8B",
    "b200_sxm",
    BackendKind::Vllm,
)
.backend_version("0.19.0")
.tp_size(8)
.moe_parallelism(Some(1), Some(8))
.build()?;

// Pure-Rust hot path: no `py` token, so the GIL is never acquired here.
let prefill = engine.prefill_latency_ms(/* bs */ 1, /* isl */ 1024, /* prefix */ 0)?;
let decode = engine.decode_latency_ms(/* bs */ 1, /* isl */ 1024, /* osl */ 2)?;
```

The `EngineConfig` identity carried by the spec groups its fields into cohesive
sub-structs — `ParallelMapping` (`tp_size`, `pp_size`, `attention_dp_size`,
`moe_tp_size`, `moe_ep_size`, `cp_size`), `QuantizationConfig`, and an optional
`SpeculativeConfig` (`nextn`) — all `#[serde(flatten)]`-ed
so the wire JSON stays the flat object Python emits.

## Supported vs. not supported

**Supported today**

- **Silicon mode.** Latency is computed from the collected perf `.parquet`
  tables. Ops that are inherently analytical (memory-bound element-wise, some
  communication) compute their formula and tag their result `Empirical` / `Sol`,
  but that is per-op provenance, not a separate run mode — there is no global
  `db_mode` selector in the Rust core.
- **KV-cache memory estimation.** The `estimate_kv_cache` API (a native estimate
  with an optional naive fallback via `allow_naive_fallback`), run once at
  startup, separate from the latency hot path.
- **Shared-layer source inheritance.** Sibling / cross-version silicon rows are
  resolved by Python and carried on the spec (`perf_db_sources`), so Rust queries
  the same rows Python does.

**Not supported**

- **Hybrid mode.** Python's silicon-plus-empirical gap-filling — falling back to
  an analytical estimate when a specific silicon shape or table is missing — is
  not implemented. A missing table or shape **errors** rather than being
  estimated. This is why the parity gate uses error-symmetry: both sides erroring
  on the same input counts as a match (see [Parity](#parity)).
- **Pure empirical / SOL modes.** The Rust core does not run a whole model
  analytically; it interprets collected silicon data.
- Model- and feature-level gaps (DSA/dsv4 context parallelism, FPM v2, …) are
  listed under [Known limits](#known-limits).

## Current scope

- Prefill, decode, and mixed prefill-plus-decode steps for the model families
  represented in AIC's checked-in model configs (dense GQA and MoE).
- MLA attention (context + generation) and DeepSeek-family models, including MTP
  speculative decoding (`nextn`).
- Context parallelism: dense GQA/MoE (seq-split token-major ops + zigzag
  `ContextAttention`) and MLA prefill (zigzag `ContextMLA`).
- MoE attention-DP token scaling (SGLang all-gathers DP-sharded tokens before
  the MoE) and DSA generation boundary-utilization extrapolation, both matched
  to Python.
- Shared-layer perf-data sources: sibling / cross-version rows are resolved in
  Python (`sdk/engine.py::_compute_perf_db_sources`) and carried on the spec as
  `perf_db_sources`, so the Rust core inherits the same silicon rows Python
  resolves for the shared layer.
- FPM v1 aggregate scheduled-request fields per attention-DP rank as the
  estimator input, aligned with Dynamo FPM v1.
- AIC-style Hugging Face model config JSON files.
- AIC perf `.parquet` files (`gemm_perf`, `context_attention_perf`,
  `generation_attention_perf`, `moe_perf`, `mla_context_module_perf`,
  `mla_generation_module_perf`, `dsa_*_module_perf`, communication tables, …),
  with explicit Git LFS pointer detection for data that has not been pulled.
- A PyO3 hot-path pyclass (`AicEngine`) plus a minimal C ABI so Python tests can
  opt into the Rust estimator without changing the default Python SDK path.

## Parity

Rust output is held to the Python SDK by the engine-step parity gate
(`parity_tests/test_engine_step_parity.py`, run on PRs). The contract is a 1%
tolerance with error-symmetry: both sides erroring is a pass; one side erroring
is a fail. Rust loaders and operators deliberately mirror Python's exact
semantics (e.g. the dsv4 loader's last-source-wins overwrite), so parity-shaped
code should not be "cleaned up" in ways that diverge from the pinned reference.

## Known limits

- Full context-parallel modeling for DSA (DeepSeek-V3.2) and dsv4
  (DeepSeek-V4-Flash) is blocked on missing sparse mqa/topk/dsa perf tables; the
  dense and MLA CP paths are complete.
- Request-level FPM v2 fields and further WideEP accuracy work are left for later
  PRs.
- Unit and integration tests use fixture perf files so most can run without the
  large AIC perf databases. CI runs the workspace both without Python embedding
  and with all features; embedding-only tests such as `embedded_round_trip` and
  `memory_round_trip` are feature-gated and also exercised by the installed-wheel
  harness.

See [`aic-core/API.md`](../../API.md) for the complete wheel/crate public API and
compatibility contract.
