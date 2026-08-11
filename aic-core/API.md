# `aic-core` public API contract

`aic-core` is released as two artifacts at the same version:

- the `aiconfigurator-core` Python wheel, imported as `aiconfigurator_core`;
- the `aiconfigurator-core` Rust crate, imported as `aiconfigurator_core`.

The wheel owns the estimator SDK, model and system data, and the native PyO3
extension. It does not depend on the upper `aiconfigurator` distribution. The
crate owns the compiled engine, forward-pass model, KV-cache request/response
types, and the embedded Rust-to-Python construction path.

## Stable Python facade

New Python code should import from the small facade:

```python
from aiconfigurator_core.sdk import (
    EngineHandle,
    ModelConfig,
    RuntimeConfig,
    RustForwardPassPerfModel,
    compile_engine,
    estimate_kv_cache,
    estimate_num_gpu_blocks,
)
```

The explicit module paths remain supported:

```python
from aiconfigurator_core.sdk.engine import EngineHandle, compile_engine
from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel
from aiconfigurator_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
```

`aiconfigurator_core.sdk.__all__` is the supported high-level surface. The
facade resolves lazily, so importing it does not load the model registry,
performance database, or native engine until a name is used.

The top-level `aiconfigurator_core` module exposes the lower-level native
extension contract:

- `AicEngine`
- `RustForwardPassPerfModel` (the raw PyO3 class, distinct from the ergonomic
  SDK wrapper)
- `engine_spec_bincode_from_json`
- `_build_smoke`

The wheel includes `py.typed` and a stub for that native extension. The SDK
Python modules carry their own annotations.

## Choosing a forward-pass API

For adaptive forward-pass modeling, use
`RustForwardPassPerfModel.best_available(...)` from Python or
`ForwardPassPerfModel::best_available(...)` from Rust. This path uses the
native AIC estimate when the native estimator can be built, learns online
correction factors from FPM observations, and falls back to regression for
eligible native build or data-availability failures. These include unsupported
models and missing or unreadable model, system, or performance data. Check
`diagnostics()` to determine whether the active source is `aic`,
`aic_with_correction`, or `fallback_regression`, and to inspect any fallback
warning.

Native online corrections default to an absolute factor range of `[0.5, 2.0]`.
Pass `None` explicitly as `min_faster_correction_factor` or
`max_slower_correction_factor` in the options dictionary to remove the bound
in that direction. Regression fallback ignores both options.

Use `from_native(...)` instead when native AIC support is required and an
unsupported configuration or native data failure should surface rather than
fall back.

`AicEngineBuilder` serves a different purpose: it constructs the strict native
Rust engine for direct public prefill and decode latency calls. It does not
provide regression fallback or online correction, so it is not a replacement
for `best_available(...)`.

```python
from aiconfigurator_core.sdk import RustForwardPassPerfModel

# Engine-config and per-rank FPM dictionary setup is omitted here.
model = RustForwardPassPerfModel.best_available(config)
diagnostics = model.diagnostics()
print(diagnostics["source"])
if diagnostics["last_warning"] is not None:
    print(diagnostics["last_warning"])

estimate_ms = model.estimate_forward_pass_time_ms(metrics_by_rank)
if estimate_ms is None:
    # Regression fallback starts without observations for each workload kind.
    # Supply observed FPM iterations with positive wall_time until the configured
    # min_observations threshold is reached, then retry the estimate.
    model.tune_with_fpms(observed_iterations)  # Observed-iteration setup omitted.
    estimate_ms = model.estimate_forward_pass_time_ms(metrics_by_rank)
```

## Stable Rust facade

Embedded consumers construct engines with `AicEngineBuilder`. It normalizes
configuration into one private build request and enters Python once to compile
an engine specification. Calls on the returned `AicEngine` are pure Rust and do
not re-enter Python.

Standalone binaries must enable the crate's `embed-python` feature; applications
hosted by an initialized Python interpreter do not. In either case, the matching
`aiconfigurator_core` Python package must be importable. Standard deployments
should install the upper `aiconfigurator` distribution, which supplies that
package: bundled in 0.10 and through a pinned core-wheel dependency in
split-package releases. Core-only consumers may install `aiconfigurator-core`
directly. See the
[crate README](rust/aiconfigurator-core/README.md) for setup and usage examples.

The supported root-level Rust surface is grouped as follows:

- compiled engine: `AicEngineBuilder`, `AicEngine`, `AicError`;
- forward-pass estimation: `ForwardPassPerfModel`,
  `ForwardPassPerfOptions`, diagnostics/readiness/source types, and the
  `ForwardPassMetrics` telemetry types;
- KV-cache estimation: `estimate_kv_cache`, `KvCacheEstimateRequest`,
  `KvCacheEstimateOptions`, `KvCacheMemoryFraction`, and estimate/result/error
  types;
- wire identity: `EngineConfig`, `ParallelMapping`, `QuantizationConfig`,
  `SpeculativeConfig`, `BackendKind`, and `DataType`;
- schema gates: `ENGINE_CONFIG_SCHEMA_VERSION`,
  `ENGINE_SPEC_SCHEMA_VERSION`, and `FPM_VERSION`.

Advanced consumers may use `engine::{Engine, RuntimeConfig, StaticMode,
StaticResult, PerOpValue}` and `engine::spec::{EngineSpec, OpSpec}` to load and
execute a previously compiled specification directly. `PerOpValue` is the
per-op result tuple `(name, latency_ms, energy_wms, source)` returned by the
`*_per_op` / `evaluate_*` methods (the thin op-list evaluation FFI); per-op
energy is 0.0 wherever the perf tables carry no power columns.

## Compatibility rules

- The wheel and crate versions must match for every `aic-core` release.
- A breaking `EngineConfig`, `EngineSpec`, or `ForwardPassMetrics` wire change
  must bump its corresponding schema constant. Consumers reject unsupported
  schema versions before using the payload.
- A supported facade name is not removed or given a new required parameter
  without a documented deprecation path. The package is pre-1.0, so an
  unavoidable incompatible API change also requires a minor-version bump.
- The raw PyO3 class and ergonomic SDK wrapper intentionally share the name
  `RustForwardPassPerfModel`; callers should import from `aiconfigurator_core.sdk`
  unless they specifically need the JSON-oriented native binding.

## CI contract

Every change is checked from three consumer viewpoints:

1. Python source and isolated installed-wheel imports, including the public
   facade, native stub, bundled data, and upper/core ownership boundary;
2. Rust tests with embedding disabled and with all features enabled;
3. a separate workspace crate that depends on `aiconfigurator-core` and
   compiles only against its public exports.
