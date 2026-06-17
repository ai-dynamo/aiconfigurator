# Phase 1.5 Execution Plan — Python builds, Rust executes

**Status:** draft. Awaiting approval before implementation starts.
**Branch base:** `rust-migration/phase1` tip.
**Supersedes:** Phase 5 of `phase1/migration-execution-plan.md`.

## Motivation

Phase 1 shipped a faithful apple-to-apple port of Python's engine-step
pipeline into Rust. It hit parity (`phase1/phase-1-checkpoint.md`) but landed
two structural costs:

1. Model intelligence is **duplicated**. Python's `models/*.py` and
   Rust's `models/*.rs` independently encode the same architectures
   (~7.5 kLoC each); every Python model fix during D1–D8 had to be
   re-stated in Rust.
2. Hot-path performance is **FFI-tax-bound** on small graphs. The
   current ctypes path costs ~15–25 µs per call regardless of inner
   work; a sweep with N points pays N × that.

The PoC under `poc/` (and `design_doc.html`) validated an alternative
architecture: Python builds the op list, Rust executes it via a
compiled `Engine` artifact, with PyO3 used in both call directions.
14/14 parity tests passed bit-identical at 1e-12 tolerance.

Phase 1.5 is the production-grade adoption of that architecture.

## Goal

> Python owns model construction. Rust owns execution.
> Two caller patterns share one bridge crate: a Python-sweep path
> (via `#[pyfunction]` exports) and a Rust-embedded path (Mocker,
> with PyO3 hidden inside the build step). Per-call Python op-walking
> is what disappears.

After Phase 1.5:
- `rust/aiconfigurator-core/src/models/` is **deleted** (~6.7 kLoC).
  `factory.rs`, `registry.rs`, `config_loader.rs`, family builders —
  none of it survives.
- `operators/`, `perf_database/`, `interpolation.rs`, `common/`, the
  parts of `session.rs` that drive op-list execution: **kept and
  reused**.
- Python's `sdk/models/*.py` becomes the single source of truth for
  model topology. A new `sdk/engine.py::compile_engine(model,
  runtime_config) -> bytes` walks the existing op list and ships it
  across the boundary as a bincode-serialised `EngineSpec`.
- The Python-sweep path calls `engine.run_static(runtime, mode,
  stride)` and `engine.run_agg(runtime, batch_size,
  num_context_tokens)` — same signatures as Phase 1's
  `BaseBackend.run_static` / `run_agg`.
- The Rust-embedded path (Mocker) calls
  `aiconfigurator_core::build_aic_engine(...)` once at startup, then
  `engine.predict_prefill_latency` / `predict_decode_latency` per
  scheduling decision — pure Rust, no PyO3 / GIL on the hot path.

## Out of scope

- Re-running collectors or regenerating perf DB data.
- Changing the perf DB schema or the support-matrix CSV format.
- Porting Python's CLI, generator, Pareto analysis, or webapp.
- The R1 / R5 scan DRIFT clusters (`phase1/support-matrix-scan.md`)
  — scan-comparator artifacts, tracked separately.
- A `--no-default-features` / offline-bytes Rust build mode.
  `aiconfigurator-core` always links libpython via PyO3.

## Build-system transition

Phase 1 installs as pure Python: `pyproject.toml` declares
`build-backend = "setuptools.build_meta"` with no Rust hooks. The
Rust core is compiled **lazily** at first runtime use, gated by the
`AICONFIGURATOR_RUST_CORE_AUTOBUILD=1` env var (read in
`sdk/rust_engine_step.py:23`). End users who don't set the env var
must have run `cargo build --release` on the Rust crate manually.

Phase 1.5 swaps the build-backend to `maturin` so the Rust extension
becomes a real component of the Python wheel. The compile cost
shifts from first-runtime-call to install-time (for source installs)
or to zero (for users on a prebuilt wheel):

| Trigger | Phase 1 today | Phase 1.5 |
| --- | --- | --- |
| `pip install aiconfigurator` (wheel from PyPI) | No Rust touch | No Rust build — wheel ships prebuilt `.so` |
| `pip install -e .` (source checkout) | No Rust touch | Maturin compiles eagerly during install (~30–60 s cold) |
| First engine-step call with `AUTOBUILD=1` | Lazy `cargo build --release` (~30–60 s) | `import aiconfigurator_core` — µs `dlopen`, no compile |
| Per-call after first load | ctypes dispatch | PyO3 dispatch |

What this requires (the work lives inside E1):

- `pyproject.toml`: switch `build-backend` from
  `setuptools.build_meta` to `maturin`. Add `[tool.maturin]` with
  `module-name = "aiconfigurator_core._aiconfigurator_core"`, `abi3 = true`,
  `python-source = "src"`.
- `AICONFIGURATOR_RUST_CORE_AUTOBUILD` becomes a no-op with a
  deprecation warning, kept one release cycle so existing
  automation doesn't silently break.
- `sdk/rust_engine_step.py` becomes a thin facade over the new
  `sdk/engine.py`. Marked deprecated; removal belongs to a later
  phase.
- CI: `maturin build --release --strip` for wheel publishing;
  `maturin develop` for source-checkout CI.

## Architecture target

Two caller patterns share one bridge crate (`aiconfigurator-core`)
and one boundary library (PyO3, used in both directions across the
Python ↔ Rust line). See also `workflow.uml` for the call sequence.

**Python-sweep pattern** (CLI / Predictor / sweep loop):

```text
sdk/models/<family>.py
builds Operation objects
       │
       ▼
sdk/engine.py
compile_engine(model, runtime_config)
   walks model.context_ops + model.generation_ops
   serialises to EngineSpec (bincode bytes)
       │  PyO3 #[pyfunction]  (Python → Rust)
       ▼
aiconfigurator_core
   Engine::from_spec_bytes(bytes) + PerfDatabase::load(...)
       │
       ▼
sweep loop calls engine.run_static(runtime, mode, stride)
or engine.run_agg(runtime, batch_size, num_context_tokens)
   PyO3 entry releases GIL during execution
       │
       ▼
Vec<StepResult> back across PyO3
```

**Rust-embedded pattern** (Dynamo Mocker, future Rust simulators):

```text
Mocker (Rust)
       │
       ▼
aiconfigurator_core::build_aic_engine(model_path, runtime_config)
       │  pure-Rust signature; PyO3 hidden inside (Rust → Python)
       ├─► Python::with_gil { py.import("aiconfigurator.sdk.engine")
       │                         .call_method1("compile_engine", (...)) }
       ◄─── returns bincode bytes ───
       │
       ▼
AicEngine { Arc<Engine>, Arc<PerfDatabase> }  ← Mocker holds this
       │
       ▼
hot path (per scheduling decision):
   engine.predict_prefill_latency(bs, isl, prefix)   ← pure Rust, no PyO3
   engine.predict_decode_latency (bs, isl, osl)      ← pure Rust, no PyO3
```

(KV cache sizing is a separate one-shot call at startup —
`aiconfigurator_core::estimate_kv_cache(req)`, a top-level crate
function rather than a method on `AicEngine`. See the
"Capacity API (Issue #1159)" section below.)

| Pattern | Caller direction | Build entry point | Hot-path entry point | PyO3 on hot path? |
| --- | --- | --- | --- | --- |
| Python-sweep | Python → Rust | `compile_engine` (Python) + PyO3 `#[pyfunction]` exports | `engine.run_static` / `engine.run_agg` — PyO3 per call | One hop per call (Python op-walk eliminated inside) |
| Rust-embedded | Rust → Rust (PyO3 hidden inside build step) | `aiconfigurator_core::build_aic_engine` — pure-Rust signature | `engine.predict_*_latency` — pure Rust | None |

`aiconfigurator-core` always depends on PyO3 + libpython. No
`--no-default-features` build mode, no offline-bytes path; the
in-memory bincode handoff is the only build channel.

### Rust → Python call shape

`build_aic_engine` uses the standard pattern from
<https://pyo3.rs/v0.28.3/python-from-rust/function-calls.html>:

```rust
pub fn build_aic_engine(
    model_path: &str,
    system: &str,
    backend: &str,
    backend_version: &str,
    tp_size: usize,
    /* ... remaining RuntimeConfig fields ... */
) -> Result<AicEngine, AicError> {
    let spec_bytes: Vec<u8> = Python::with_gil(|py| -> PyResult<Vec<u8>> {
        let aic_core = py.import("aiconfigurator.sdk.engine")?;
        aic_core
            .call_method1(
                "compile_engine",
                (model_path, system, backend, backend_version, tp_size, /* ... */),
            )?
            .extract::<Vec<u8>>()
    })
    .map_err(AicError::from)?;

    let spec: EngineSpec = bincode::deserialize(&spec_bytes)?;
    let db = Arc::new(PerfDatabase::load_from_spec(&spec)?);
    let engine = Engine::build(spec, Arc::clone(&db))?;
    Ok(AicEngine { inner: Arc::new(engine), db })
}
```

The inverse direction (Python → Rust) uses `#[pyfunction]`-decorated
entry points exported via maturin into the
`aiconfigurator_core._aiconfigurator_core` extension submodule. One PyO3
dependency, two call directions, no extra libraries.

## Data classes

Three typed inputs cross the API:

- **`EngineConfig`** — engine identity, baked into the compiled
  `AicEngine` at build time.
- **`RuntimeConfig`** — per-call inputs; varies per `run_static` /
  `run_agg`. Field-for-field mirror of Phase 1's
  `sdk/config.RuntimeConfig`.
- **`OpSpec`** — plain-data wire form of the `Op` enum, walked out
  of Python's `Operation` classes by `compile_engine` and bincoded
  into the engine artifact. See "The crux: OpSpec wire format"
  below.

### `EngineConfig`

Modularised from today's flat `lib.rs:48-90` struct per the Issue #1159
discussion (lands as commit **E1.5**, before E2). Trivial
1-2-field groupings stay flat; only fields that form a cohesive
multi-field unit get a sub-struct:

```rust
pub struct EngineConfig {
    pub schema_version: u32,

    // Model
    pub model_name: String,                 // HF id or local config path
                                            // (architecture inferred from HF config)

    // System
    pub system_name:  String,               // e.g. "h200_sxm", "b200_sxm"
    pub systems_path: Option<PathBuf>,      // override the bundled systems/ dir

    // Backend
    pub backend:         BackendKind,       // TrtLlm | Vllm | Sglang
    pub backend_version: Option<String>,

    // KV
    pub kv_block_size: Option<u32>,         // scheduler block size; None = backend default

    // Cohesive groupings (multi-field, semantically coupled)
    pub parallel:     ParallelMapping,
    pub quantization: QuantizationConfig,
    pub speculative:  Option<SpeculativeConfig>,

    #[serde(default)] pub extra: BTreeMap<String, String>,
}

pub struct ParallelMapping {
    pub tp_size:           u32,
    pub pp_size:           u32,
    pub attention_dp_size: u32,
    pub moe_tp_size:       Option<u32>,
    pub moe_ep_size:       Option<u32>,
}

pub struct QuantizationConfig {
    pub gemm_quant_mode:    Option<GemmQuantMode>,
    pub moe_quant_mode:     Option<MoeQuantMode>,
    pub kvcache_quant_mode: Option<KvCacheQuantMode>,
    pub fmha_quant_mode:    Option<FmhaQuantMode>,
    pub comm_quant_mode:    Option<CommQuantMode>,
}

pub struct SpeculativeConfig {
    pub nextn:              u32,
    pub nextn_accept_rates: Option<Vec<f64>>,
}
```

Field migration from today's flat struct:

- `model_name`, `system_name`, `backend`, `backend_version`,
  `kv_block_size` → stay flat with the same names.
- `model_arch` is **dropped**. AIC infers architecture from the HF
  config's `architectures` field (keyed by `model_name`) via
  `ARCHITECTURE_TO_MODEL_FAMILY` in `sdk/common.py`. The override
  was effectively unused in practice; if a real caller ever needs
  it, the `extra` map is the escape hatch.
- `tp_size`, `pp_size`, `moe_tp_size`, `moe_ep_size`,
  `attention_dp_size` → `ParallelMapping`.
- `weight_dtype`, `moe_dtype`, `activation_dtype`, `kv_cache_dtype`
  → `QuantizationConfig` as the corresponding `*_quant_mode` fields.
- `nextn`, `nextn_accept_rates` → `SpeculativeConfig`, wrapped in
  `Option<>` so models without MTP don't carry the noise.
- New: `systems_path` — override the bundled `systems/` directory
  (used by both engine build and the capacity API).

Serde aliases on the sub-structs let today's flat-JSON inputs (from
the ctypes FFI) keep parsing through one release cycle.

### `RuntimeConfig`

```rust
pub struct RuntimeConfig {
    pub batch_size: u32,
    pub beam_width: u32,                            // default 1
    pub isl: u32,
    pub osl: u32,
    pub prefix: u32,                                // cached tokens
    pub seq_imbalance_correction_scale: f64,        // default 1.0
    pub gen_seq_imbalance_correction_scale: f64,    // default 1.0
}
```

Same field set as Phase 1's `sdk/config.RuntimeConfig`. Crosses PyO3
as a struct (Python → Rust) or as positional args of `run_static` /
`run_agg`.

### `AicEngine` public API

Two surfaces — sweep entries (for the Python-sweep pattern) and
scalar entries (for the Rust-embedded pattern):

```rust
impl AicEngine {
    // ── Sweep entry points; mirror Phase 1's BaseBackend signatures: ──
    pub fn run_static(
        &self,
        runtime: &RuntimeConfig,
        mode:    StaticMode,             // Context | Generation | Both
        stride:  u32,                    // default 32
    ) -> StaticResult;

    pub fn run_agg(
        &self,
        runtime:            &RuntimeConfig,
        batch_size:         u32,         // decode batch (sweep coord)
        num_context_tokens: u32,         // chunked prefill chunk (sweep coord)
    ) -> AggResult;

    // ── Mocker entry points; scalar in/out: ──
    pub fn predict_prefill_latency(&self, bs: u32, isl: u32, prefix: u32) -> f64;
    pub fn predict_decode_latency (&self, bs: u32, isl: u32, osl: u32)    -> f64;
}
```

The scalar entries are thin shims over `run_static` with the
appropriate `StaticMode`. Disagg sweeps call `run_static` twice (once
per worker role); no separate `run_disagg` method needed. The agg
sweep iterating over many `(batch_size, num_context_tokens)` pairs
lives in Python (mirrors Phase 1's
`find_best_agg_result_under_constraints`); each pair is one
`run_agg` call.

Internal helpers (private; called by the public methods above):

```rust
impl AicEngine {
    fn run_context_phase   (&self, runtime: &RuntimeConfig) -> PhaseLatency;
    fn run_generation_phase(&self, runtime: &RuntimeConfig, stride: u32) -> PhaseLatency;
    fn mix_step_latency    (&self, runtime: &RuntimeConfig, ctx_tokens: u32, gen_tokens: u32) -> f64;
    fn genonly_step_latency(&self, runtime: &RuntimeConfig, gen_tokens: u32) -> f64;
}
```

These mirror Phase 1's private helpers in `BaseBackend`. Not exposed
through PyO3.

## AIC ↔ Dynamo Mocker handshake

Dynamo Mocker is the primary external consumer of the AIC perf
model. The actual contract is the **Rust trait `AicCallback`**
in `dynamo/lib/mocker/src/common/perf_model.rs`, not the
current Python module shape — Mocker's scheduler holds
`Arc<dyn AicCallback>` and never sees the Python object directly.
The Python module name (`dynamo._internal.aic`), class name
(`AicSession`), constructor signature, and call conventions are
all **negotiable** on both sides; trait signatures and numerical
outputs are what we hold stable.

### Current handshake topology

```text
Mocker scheduler (Rust)             — agg passes call H1 + H2 sequentially
   │  Arc<dyn AicCallback>
   ▼
PyAicCallback (Rust adapter; lib/bindings/python/rust/llm/aic_callback.rs)
   │  Python::with_gil + call_method1
   ▼
AicSession (Python; lib/bindings/python/src/dynamo/_internal/aic.py)
   │  walks model.{context,generation}_ops in Python
   ▼
aiconfigurator Phase 1 SDK (Python)
```

### Handshake inventory

Two `AicCallback` trait methods plus one startup function:

| # | Surface | Kind | Hot? | Phase 1 implementation |
| --- | --- | --- | --- | --- |
| **H1** | `AicCallback::predict_prefill(batch_size, effective_isl, prefix) -> f64` ms | Trait method | Yes — per scheduler pass | Python op-walk over `model.context_ops` |
| **H2** | `AicCallback::predict_decode(batch_size, isl, osl) -> f64` ms (Mocker passes `osl=2`) | Trait method | Yes — per scheduler pass; **agg mode calls H1+H2 sequentially per pass** | Python op-walk over `model.generation_ops` with `DEFAULT_STATIC_STRIDE=32` quadrature |
| **H3** | `dynamo._internal.aic.estimate_num_gpu_blocks(...) -> usize` (called from Rust via PyO3 at Mocker startup; **not** a trait method) | Startup function | No — once at startup | Python `backend._get_memory_usage` + per-backend KV budget formula. **Phase 1.5 replaces the body** with a call to top-level Rust `aiconfigurator_core::estimate_kv_cache` that returns the richer `KvCacheEstimate`; the Python wrapper stays as a thin compatibility shim. See "Capacity API (Issue #1159)". |

### Frozen

- `AicCallback` trait signatures (`predict_prefill`, `predict_decode`)
  and `dynamo._internal.aic.estimate_num_gpu_blocks` signature stay
  exactly as today.
- Numerical outputs within Phase 1's parity tolerance, validated by
  the E6 gate + one Mocker integration test post-flip.

### Negotiable

- Python module path `dynamo._internal.aic`, class name `AicSession`,
  and constructor signature. Renameable in lock-step.

### Target shape

- Mocker swaps `Arc<PyAicCallback>` for `Arc<RustAicCallback>`.
  `RustAicCallback` wraps an `aiconfigurator_core::AicEngine` and
  routes the trait methods to `engine.predict_prefill_latency` /
  `predict_decode_latency` directly. No PyO3, no GIL on the hot path.
- Mocker calls `aiconfigurator_core::build_aic_engine(...)` once at
  startup. The PyO3 plumbing is owned by `aiconfigurator-core`;
  Mocker never types `Python::with_gil`.
- Startup KV sizing (H3) calls top-level
  `aiconfigurator_core::estimate_kv_cache(...)` — not part of the
  `AicCallback` trait.

### Dynamo-side cleanup after Phase 1.5 + K-series

The migration enables, but does not require, the following cleanup
on the Dynamo side (separate PR, downstream of this plan):

- `dynamo._internal/aic.py` — **stays as a compat shim** for any
  Python callers that import it. Its `predict_*` bodies route
  through `aiconfigurator_core` via PyO3; Mocker itself bypasses
  this file entirely.
- `lib/bindings/python/rust/llm/aic_callback.rs` — **slims**.
  `PyAicCallback` (Python-walking impl) deletes; `RustAicCallback`
  (wraps `Arc<AicEngine>`) takes its place. The PyO3 helpers
  `create_aic_callback`, `create_aic_prefill_load_estimator`, and
  `estimate_aic_num_gpu_blocks` delete — replaced by direct calls
  to `aiconfigurator_core::build_aic_engine` /
  `estimate_kv_cache`.
- The `AicCallback` trait itself (`lib/mocker/src/common/perf_model.rs`)
  is unchanged.

### Coordination items, mapped to the commit sequence
- **Before E6:** pin Mocker's CI to a known-good aiconfigurator
  version so the parity flip doesn't break Mocker's gate.
- **At E6:** run one Mocker integration test against the new
  Engine-backed path. Assert H1/H2 outputs within Phase 1's 1%
  tolerance.
- **At E7:** Rust `models/` + `backends/` deletion is invisible to
  H3. Phase 1 Python `BaseBackend._get_memory_usage` continues to
  power `estimate_kv_cache`'s native path; only the Rust shadow
  goes away.
- **Post-E8 (Dynamo PR):** swap `Arc<PyAicCallback>` for
  `Arc<RustAicCallback>`. Re-run the Mocker integration test;
  outputs stay within tolerance. Hot-path GIL acquisitions drop to
  zero.

## Capacity API (Issue #1159)

KV cache capacity estimation is a separate concern from latency
prediction. It runs once at startup (not on the hot path), uses
overlapping but not identical inputs to `build_aic_engine`, and
needs a naive fallback path for models AIC can't fully model. Per
[Issue #1159](https://github.com/ai-dynamo/aiconfigurator/issues/1159)
consensus, it lives as a top-level crate function, not a method on
`AicEngine`.

### Top-level surface

```rust
pub fn estimate_kv_cache(req: KvCacheEstimateRequest)
    -> Result<KvCacheEstimate, KvCacheEstimateError>;
```

### Request

```rust
pub struct KvCacheEstimateRequest {
    pub engine: EngineConfig,                       // reuses the modularised form
    pub max_num_tokens:                     u32,
    pub max_batch_size:                     u32,
    pub kv_cache_memory_fraction:           KvCacheMemoryFraction,
    pub gpu_memory_capacity_bytes_override: Option<u64>,   // for unknown SKUs
    pub tolerance_fraction:                 Option<f64>,   // None = raw only; Some(0.05) = 5% safety
    pub options:                            KvCacheEstimateOptions,
}

/// Backend-tagged memory fraction. The enum variant encodes the XOR
/// between TRT-LLM's free-fraction and vLLM/SGLang's total-fraction
/// semantics; `estimate_kv_cache` validates against `engine.backend`
/// and returns `IncompatibleMemoryFraction` if mismatched.
pub enum KvCacheMemoryFraction {
    /// Fraction of TOTAL GPU memory. Compatible with vLLM
    /// (`gpu_memory_utilization`) and SGLang (`mem_fraction_static`).
    OfTotal(f64),
    /// Fraction of FREE (post-non-KV) GPU memory. Compatible with
    /// TRT-LLM (`free_gpu_memory_fraction`).
    OfFree(f64),
}

pub struct KvCacheEstimateOptions {
    pub allow_naive_fallback:     bool,
    pub allow_hf_config_download: bool,
}
```

### Response

```rust
pub struct KvCacheEstimate {
    pub total_gpu_capacity_bytes: u64,
    pub total_kv_size_bytes:      u64,
    pub kv_size_per_token_bytes:  u64,
    pub total_kv_size_tokens:     u64,
    pub source:             EstimateSource,
    pub memory_breakdown:   Option<MemoryBreakdown>,         // Some on native; None on fallback
    pub tolerance_adjusted: Option<KvCacheEstimateAdjusted>, // Some iff tolerance_fraction set
}

pub enum EstimateSource {
    Native,         // AIC's full backend memory model used
    NaiveFallback,  // 80%-of-post-weight heuristic
}

pub struct MemoryBreakdown {
    pub weights_bytes:          u64,
    pub activations_bytes:      u64,
    pub runtime_overhead_bytes: u64,
    pub comm_overhead_bytes:    u64,
}

pub struct KvCacheEstimateAdjusted {
    pub tolerance_fraction:   f64,
    pub total_kv_size_bytes:  u64,
    pub total_kv_size_tokens: u64,
}

pub enum KvCacheEstimateError {
    Unsupported { model: String, backend: BackendKind, gpu_sku: String, reason: String },
    InsufficientModelMetadata { missing_fields: Vec<String> },
    NoKvBudget { total_gpu_capacity_bytes: u64, non_kv_bytes: u64 },
    IncompatibleMemoryFraction { backend: BackendKind, variant_kind: &'static str },
    BadConfig { field: String, reason: String },
    HfConfigFetchFailed { hf_id: String, source: String },
}
```

### Native path vs naive fallback

The native path uses Phase 1's existing backend memory model
(`_get_memory_usage`) via PyO3, applied to the requested
parallelisation:

```text
non_kv_bytes         = weights + activations + runtime_overhead + comm_overhead
memory_limit         = (backend-specific function of kv_cache_memory_fraction)
total_kv_size_bytes  = memory_limit - non_kv_bytes
total_kv_size_tokens = floor(total_kv_size_bytes / kv_size_per_token_bytes)
```

`memory_breakdown` populated; `source = Native`.

The naive fallback fires when the native path errors with
`Unsupported` and `allow_naive_fallback = true`:

```text
estimated_weight_bytes = parse from HF config, or DEFAULT_FALLBACK_WEIGHT_BYTES
total_kv_size_bytes    = max(0, total_gpu_capacity_bytes - estimated_weight_bytes) * 0.80
kv_size_per_token      = parse from HF config (MLA-aware), or DEFAULT_FALLBACK_KV_BYTES_PER_TOKEN
total_kv_size_tokens   = floor(total_kv_size_bytes / kv_size_per_token_bytes)
```

`memory_breakdown = None`; `source = NaiveFallback`. The fallback
handles MLA / compressed-latent attention correctly
(`mla_cache_width = kv_lora_rank + qk_rope_head_dim` rather than
2 × kv_heads × head_dim).

### Mocker integration

Mocker calls `estimate_kv_cache` once at startup (Rust-embedded
pattern; same crate as `build_aic_engine`, no Python on the hot
path) and computes:

```rust
let num_gpu_blocks_per_rank = estimate.total_kv_size_tokens / scheduler_block_size as u64;
```

`dynamo._internal.aic.estimate_num_gpu_blocks()` becomes a thin
Python wrapper that calls `estimate_kv_cache` and applies the same
conversion. **As of K3 that Dynamo-side rewrite is a DEFERRED
downstream PR in the `ai-dynamo/dynamo` repo** (it is NOT done in
`aiconfigurator`). K3 ships the in-repo surface it consumes: the Rust
`aiconfigurator_core::estimate_kv_cache(req)` forwarder (for the
embedded Mocker) plus the AIC-side reference
`aiconfigurator.sdk.memory.estimate_num_gpu_blocks` helper (the same
`floor(tokens / scheduler_block_size)` conversion). See
`phase-1.5-capacity-followup.md`.

> **Post-K3 design change.** The capacity API moved out of `sdk/engine.py` into
> a dedicated `sdk/memory.py`. The redundant Python-callable
> `aiconfigurator_core.estimate_kv_cache` `#[pyfunction]` was dropped, and the
> tolerance validation + `tolerance_adjusted` margin moved out of Rust into the
> Python `sdk.memory.estimate_kv_cache` (the single source of truth). The Rust
> `estimate_kv_cache(req)` is now a pure forwarder; `estimate_num_gpu_blocks`
> calls the Python estimate directly (no Rust hop). The naive fallback reuses the
> SDK's existing HF-config loaders + `_parse_hf_config_json` (with a raw-key read
> for unsupported architectures) and its reservation fraction is a caller arg
> (`naive_kv_reservation`, default 0.80). The illustrative Rust snippet above uses
> the raw token count; a tolerance-aware Mocker reads
> `tolerance_adjusted.total_kv_size_tokens` when a tolerance is set.

## The crux: OpSpec wire format

The whole plan hinges on whether Python's `Operation` objects carry
**enough static state at build time** to round-trip through OpSpec.
This is **work-item E0**. Do not skip it; do not assume it.

For every Rust `Op` variant in `operators/op.rs` (~338 LoC), trace
every field to a source:

1. **Direct mirror of a Python `Operation` instance field** → easy.
   OpSpec just serializes it.
2. **Computed at Python build time from `(ModelConfig, RuntimeConfig)`**
   → also easy. OpSpec stores the computed value.
3. **Computed inside Python's `query()` method at call time, from
   `ForwardPassMetrics` / `EngineConfig`** → needs decision:
   - Move the computation into the Python builder (preferred:
     pre-bake at compile time).
   - Or keep the computation in Rust as a runtime-resolved field on
     the `Op` enum.
4. **Derived in Rust today from `ModelConfig` / `factory.rs` /
   `config_loader.rs` only** → must move. Either bake in Python
   `compile_engine` (if Python already knows it) or pass through as
   new `RuntimeConfig` schema.

Suspected category-3/4 hot spots, flagged from the D-series audit
trail in `phase1/migration-execution-plan.md` — explicitly verify each:

- `use_qk_norm` for Qwen3 / Qwen3MoE / MiniMaxM2 (D1, Rust forces it on
  via architecture). Python's `utils.py` already does — confirm and
  bake into the Python `Operation.kwargs`.
- `MoEDispatch` backend selection (D4: vLLM / SGLang-non-deepep /
  TRT-LLM / SGLang-deepep). Python switches inside `MoEDispatch.query`
  on `moe_backend`. Decide whether the OpSpec carries the resolved
  flavor or the raw `moe_backend` enum + dispatch happens Rust-side.
- MLA fallback chain (D5: `FallbackOp(primary, fallback)`). Phase 1
  encodes this in Rust at build time. Easy port: Python emits a
  `FallbackOp` OpSpec that holds two child OpSpec lists.
- DSv3 `combined_prefix` threading (D6 Pass-1 mix-step).
- Distribution strings (`power_law_1.01` vs `power_law_1.2`, D2/D5).
- `_mtp_scale_factor` (C4): Python passes a `nextn` field through
  `runtime_config` today; ensure the path survives the rewire.

**Acceptance for E0:** every Rust `Op` enum field has a documented
Python source, with category 1/2/3/4 classification. Category 3 and 4
fields each have a chosen disposition (move to Python builder, or
keep Rust-side with a documented input). E0's output is a one-page
audit table that gates E1.

## Commit sequence

| # | Commit | What lands | Parity-gated? |
| --- | --- | --- | --- |
| **E0** | OpSpec audit | One-page audit doc enumerating every Rust `Op` field, Python source, and category. Lands in `docs/phase-1.5-opspec-audit.md`. No code. | n/a |
| **E1** | Build-system: maturin + PyO3 (default) | **Cargo:** add `pyo3 = { version = "...", features = ["abi3-py39", "auto-initialize"] }` as a **non-optional** dependency; `[lib] crate-type = ["rlib", "cdylib"]` unchanged. **pyproject.toml:** switch `build-backend` from `setuptools.build_meta` to `maturin`; add `[tool.maturin]` (module-name, features, abi3, python-source). **Compatibility:** deprecate `AICONFIGURATOR_RUST_CORE_AUTOBUILD` to a no-op with warning. **Verify both call directions:** `maturin develop` produces a Python-importable extension (Python → Rust); a Rust integration test calls `build_aic_engine(...)` to verify the Rust → Python → Rust round-trip works. No `--no-default-features` mode in this iteration. | n/a |
| **E1.5** | EngineConfig modularisation | Refactor today's flat `EngineConfig` in `lib.rs:48-90`: extract `ParallelMapping`, `QuantizationConfig`, and `SpeculativeConfig` as sub-structs (multi-field cohesive groupings); leave model/system/backend/KV-block fields flat. Add serde aliases so today's flat-JSON inputs (ctypes FFI) keep parsing through one release cycle. Pure rename + restructure; no behaviour change. Unblocks both E2 (OpSpec types consume the new `EngineConfig`) and K1 (`estimate_kv_cache` takes it as input). | n/a |
| **E2** | OpSpec types | Public `OpSpec` enum + `EngineSpec` struct in `src/engine/spec.rs`. Mirror the `Op` enum 1:1 with `serde::{Serialize, Deserialize}`. Bincode round-trip unit tests for every variant. | n/a |
| **E3** | `Engine::build` + `Engine::run_static_internal` | New `src/engine/mod.rs` constructs `Engine` from an `EngineSpec` and exposes `run_static_internal(&db, point)`. **Reuses** existing `operators::*` and `perf_database::*` underneath. Initial implementation may shim through `session.rs`. | unit |
| **E4** | PyO3 bindings | `src/py.rs`: `#[pyfunction]` exports (registered in `#[pymodule]`) for the op-transfer `engine_spec_bincode_from_json` plus `_build_smoke`; `#[pymethods]` on the `AicEngine` `#[pyclass]` (`from_spec`, `run_static`, `predict_prefill_latency`, `predict_decode_latency`, ...). `compile_engine` is Python-side (`sdk/engine.py`), and `build_aic_engine` is a Rust-only `pub fn` for embedded callers — NOT a `#[pyfunction]`. | unit |
| **E5** | Python builder | `src/aiconfigurator/sdk/engine.py`: `compile_engine(model, runtime_config) -> bytes`. Walks `model.context_ops` + `model.generation_ops`, converts each `Operation` to an OpSpec, bincodes. Python `EngineHandle` wraps the bytes and exposes `run_static` / `run_agg` / `predict_*_latency` that shell through the PyO3 surface from E4. **Reuses** existing Python `sdk/models/*.py` unmodified. | integration |
| **E6** | Parity flip | Switch `sdk/rust_engine_step.py` from the ctypes JSON path to `compile_engine` + `EngineHandle.run_static` / `run_agg`. Re-run the 164-surface smoke harness; all assertions must hold bit-identical (or within Phase 1's 1% tolerance). | **GATE** |
| **E7** | Delete Rust model layer | Remove `src/models/`, `src/backends/`, the `factory.rs` / `registry.rs` / `config_loader.rs` files. Remove the `EngineStepEstimator` ctypes path and the JSON FFI in `src/ffi.rs`. Net delta: ~7.5 kLoC removed, ~2 kLoC added. Round-trip verification (`build_aic_engine` → `predict_*_latency` → no re-entry into Python) lands as a `tests/embedded_round_trip.rs` integration test rather than a standalone binary. | parity re-run |
| **E8** | Perf gate | Benchmark `engine.run_static` / `engine.run_agg` against Phase 1's ctypes path on the smoke set. Exit criterion: **≥3× p50 speedup** on small-graph families that today regress below 1× (MiniMax-M2.5 gen, NemotronNas gen, DSv3.2 ctx) by eliminating the per-call Python op-walking. Big-graph families stay at or above Phase 1 speedup. Numbers land in `parity_tests/benchmarks.md`. | **GATE** |
| **K1** | `estimate_kv_cache` native path | Top-level Rust function in `src/capacity.rs`. **PyO3 → Python `BaseBackend._get_memory_usage`** for the breakdown, then Rust math. Returns `KvCacheEstimate` with `EstimateSource::Native` and a populated `MemoryBreakdown`. Validates `KvCacheMemoryFraction` against `engine.backend` (TRT-LLM ↔ `OfFree`; vLLM / SGLang ↔ `OfTotal`). | unit |
| **K2** | `estimate_kv_cache` naive fallback | **Pure Rust** — no PyO3 hop. HF-config parsing for KV bytes per token (MLA-aware via `kv_lora_rank + qk_rope_head_dim`), 80%-of-post-weight reservation, `allow_hf_config_download` honoured. Returns `EstimateSource::NaiveFallback` with `memory_breakdown: None`. | unit |
| **K3** | Tolerance + capacity surface | `tolerance_adjusted` field populated when `tolerance_fraction` is set (native AND naive; `BadConfig` if `t ∉ [0,1)`). Tolerance validation and margin handling live in Python `aiconfigurator.sdk.memory.estimate_kv_cache` (the single source of truth); Rust `aiconfigurator_core::estimate_kv_cache` is a pure forwarder with no math of its own (the earlier `#[pyfunction]` re-export was dropped as redundant — see the "Post-K3 design change" note above). Add the AIC-side reference `aiconfigurator.sdk.memory.estimate_num_gpu_blocks` helper (`floor(total_kv_size_tokens / scheduler_block_size)`). The Dynamo-side rewrite of `dynamo._internal.aic.estimate_num_gpu_blocks` into a thin wrapper over this surface is a **DEFERRED downstream PR** in `ai-dynamo/dynamo` (out of scope here); documented in `phase-1.5-capacity-followup.md`. | integration |

## Commit dependencies

```text
E0 ──► E1 ──► E1.5 ──► E2 ──► E3 ──► E4 ──┬─► E5 ──► E6 ──► E7 ──► E8
                       │                  │
                       │                  └── E5 can start once E2 is in (parallel with E3/E4)
                       │
                       └── K1 ──► K2 ──► K3  (capacity series, runs in parallel with E2-E8)
```

E0 strictly first (it can invalidate E2's shape). E1 and E1.5 are
sequential build-system + data-class refactors. E2-E4 are sequential
Rust changes that consume the modularised `EngineConfig`. E5 unblocks
once E2 is in. E6 is the parity gate; E7 must not land before E6
passes. E8 is the perf gate.

The K-series runs in parallel with E2-E8 once E1.5 lands. K1-K3 only
consume the modularised `EngineConfig` plus Phase 1 Python
`BaseBackend._get_memory_usage` (still present after E7 because
Python `sdk/backends/` keeps its memory accounting). K3's Dynamo-
wrapper replacement is independent of the latency parity flip — it
can ship before, during, or after E6.

## Acceptance criteria

1. **Parity (E6 gate):** the 164-surface smoke harness asserts
   bit-identical-or-within-tolerance against Phase 1 numbers. The
   full-matrix scan shows `STRICT_PASS >= 1906`, `DRIFT <= 16`
   (modulo the NCCL fix already landed), `REGRESSION == 0`.
2. **Rust-embedded round-trip:** `tests/embedded_round_trip.rs`
   passes — `build_aic_engine` returns a usable `AicEngine`,
   `predict_*_latency` calls succeed, hot-path calls do not
   re-enter Python (verified via `Python::with_gil` counter).
3. **Performance (E8 gate):** small-graph families that today
   regress below 1× warm-path reach **≥3× the Phase 1 ctypes path**.
   Big-graph families stay at or above Phase 1 speedup.
4. **LoC discipline:** target −5 000 LoC net on the Rust crate
   after E7. Python `sdk/models/` unchanged.
5. **No CLI / generator / Pareto changes.** Phase 1.5 is strictly
   internal to `sdk/`.

## Risks

| Risk | Mitigation |
| --- | --- |
| **OpSpec audit (E0) surfaces fields Python doesn't know.** | E0 must complete *before* E2 commits to a wire format. If category-3/4 fields are non-trivial, escalate before any Rust deletion. |
| **Build-backend swap (setuptools → maturin).** Moves the Rust compile from runtime to install-time (for source installs). | E1 verifies both `maturin develop` (local dev) and `maturin build --release --strip` (wheel CI). `AUTOBUILD` env var stays as a deprecated no-op for one release cycle. `abi3-py39` keeps a single wheel valid across Py 3.9–3.12. |
| **Mocker contract drift.** Phase 1.5 must keep `AicCallback::predict_prefill` / `predict_decode` signatures and numerical outputs stable. | Internal impl swap (`PyAicCallback` → `RustAicCallback`) is transparent at the trait level. E6 gate includes a Mocker-style single-point parity check. See the **AIC ↔ Dynamo Mocker handshake** section. |
| **rayon non-determinism inside `engine` internals.** If any sweep helpers use rayon (e.g., `run_agg`'s inner pair iteration), output order or rounding could vary. | Op-graph execution is per-call pure; no cross-call state. E5 integration tests run with `RAYON_NUM_THREADS=1` and `=8` and assert identical output. |

## What this plan does NOT promise

- A schema change to the perf DB.
- A change to which Python surfaces the CLI exposes.
- Removal of `sdk/rust_engine_step.py` — it becomes a thin facade
  over `sdk/engine.py`. Final removal belongs to a later phase.
- CI wheels for macOS / Linux / Windows × Py 3.9–3.12. Local
  `maturin develop` is the deliverable; wheel-publishing CI is a
  downstream packaging task.

## Pointers

- Architectural framing (the PoC results that motivate this):
  `design_doc.html`.
- What Phase 1 delivered: `phase1/phase-1-checkpoint.md`.
- Job definition (immutable contract): `phase1/migration-execution-plan.md`.
- Module map and current Rust shape: `phase1/migration-map.md`.
- FFI-tax breakdown that drives the batched-entry decision:
  `parity_tests/benchmarks.md` ("FFI overhead caveat" section).
