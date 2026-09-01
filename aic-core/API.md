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

The large-EP MoE communication family is exposed as an explicit module path
(it is not part of the facade):

```python
from aiconfigurator_core.sdk.operations.moe_comm import MOE_A2A_BACKENDS, MoEAllToAll, MoEExpertCompute
```

`MOE_A2A_BACKENDS` maps each supported MoE all-to-all comm backend to its
`MoECommBackendSpec` (framework and phase applicability plus feasibility
rules). `MoEAllToAll` and `MoEExpertCompute` are the operation classes over the
unified `moe_a2a_perf` comm and `moe_expert_compute_perf` expert-compute tables. `PerfDatabase`
(`aiconfigurator_core.sdk.perf_database`) forwards to them through
`query_moe_a2a(...)` and `query_moe_expert_compute(...)` and adds the read-only
coverage probes `moe_a2a_coverage(...)` and `moe_expert_compute_coverage(...)`. These
queries serve measured silicon data only: SOL and empirical database modes
raise `EmpiricalNotImplementedError`, with an estimation tier as a planned
follow-up.

The MoE-block builder that consumes those ops is likewise an explicit module
path:

```python
from aiconfigurator_core.sdk.models.blocks import (
    MoEBlockShape,
    build_moe_block_ops,
    register_moe_block,
)
from aiconfigurator_core.sdk.models.blocks.moe import LARGE_EP_READY_FAMILIES
```

`MoEBlockShape` is a frozen dataclass capturing the checkpoint-level geometry
of a model's MoE block: `hidden_size`, `moe_inter_size`, `topk`,
`num_experts`, `num_shared_experts` (0 when absent), `num_moe_layers`, and
`is_gated` (default `True`). `MoEBlockShape.from_model_info(model_info)`
derives it from a `_get_model_info` dict and raises `ValueError` for a
non-MoE checkpoint.

`build_moe_block_ops(prefix, shape, cfg, quant_mode, workload_distribution,
*, scale_factor, backend_name, inference_phase, model_family="*",
attn_cp_size=1, gpus_per_node=None, shared_gemm_quant_mode=None,
dispatch_quant_mode=<forward>)` is the one
place MoE blocks are wired. It returns the block's op list for one inference
phase — router GEMM, shared-expert GEMMs, and either the fused
dispatch/compute/combine pipeline or, when `cfg.moe_comm_backend` names a
comm backend for `inference_phase`, the large-EP `MoEAllToAll`/`MoEExpertCompute`
emission. An omitted `gpus_per_node` resolves from `cfg.num_gpus_per_node`
for large-EP configs and raises when that field is unset. `scale_factor` is
deliberately model-owned (legacy model classes
scale their MoE ops by their own layer count, not `shape.num_moe_layers`),
`shared_gemm_quant_mode` overrides `cfg.gemm_quant_mode` for the
shared-expert GEMMs only, and `dispatch_quant_mode` overrides the fused
path's `MoEDispatch` quant mode only.

`register_moe_block(family="*", framework="*", system="*")` is the builder's
specialization registry: family/framework/system-specific deviations register
a variant for a `(family, framework, system)` key instead of adding model
classes, `"*"` being a per-position wildcard. Lookup is most-specific-wins
with family > framework > system priority; a duplicate key registration
raises `ValueError`. The decorated function is called as `fn(default, **ctx)`
where `default` is a zero-argument continuation returning a fresh copy of the
generic pipeline's ops — variants compose with the default rather than
reimplementing it. Two variants ship registered at import: the DeepSeek and
DeepSeek-V3.2 families on sglang strip the router GEMM under deepep backends
for legacy graph fidelity.

`LARGE_EP_READY_FAMILIES` (a frozenset, currently `{"MOE", "DEEPSEEK",
"DEEPSEEKV32", "KIMIK25"}`) names the model families whose classes construct
a large-EP graph when the enumerator sets `ModelConfig.moe_comm_backend`. It
is the enumerator's assignment gate: a comm backend must never be assigned
outside this set — HYBRIDMOE and MINIMAXM3 raise on one by design, and
QWEN3VL_MOE stays excluded until its `create()` forwards `backend_name`.

### Large-EP enablement is coverage-driven

`ModelConfig.moe_comm_backend` (`dict[str, str] | None`) and
`ModelConfig.num_gpus_per_node` (`int | None`) are internal, enumerator-owned
fields — never user flags. The enumerator sets them together, per parallel
tuple: `moe_comm_backend` maps each inference phase (`"context"` /
`"generation"`) to the comm backend resolved for that phase, and
`num_gpus_per_node` carries the system's node width — a hardware fact the
large-EP ops need at construction to derive the comm node span. The fields
must stay consistent: model classes raise `ValueError` when
`moe_comm_backend` is set without `num_gpus_per_node`, because a defaulted
node width would silently mis-price the cross-node all-to-all.

Enumeration follows the coverage probes. A parallel tuple with `moe_tp == 1`
and `moe_ep > 1` participates in the large-EP regime when
`moe_expert_compute_coverage(...)` covers its requested EP size under the run's
MoE quant mode for every phase the worker runs, plus the context phase for
every role (a worker's weights are sized from its context ops), and compatible
all-to-all data is reported by `moe_a2a_coverage(...)`. The resolver prefers
the exact requested EP/node topology. For multi-node SGLang DeepEP
(`deepep_ht` or `deepep_ll`) only, the canonical shape-filtered EP8/node1 row
may be used unchanged as `estimated`; no topology scaling is synthesized.
Other frameworks, backends, and single-node requests remain exact-topology
only.

Each tuple is resolved individually. A tuple that resolves a backend for every
required phase builds the large-EP graph. An unresolved intra-node tuple keeps
the fused graph, while an unresolved cross-node tuple raises
`PerfDataNotAvailableError`. Collecting both tables for a model shape on a
system is what makes large EP explorable there — no flag, no code change.

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
`RustForwardPassPerfModel.best_available(config, worker_type, options=None)`
from Python or
`ForwardPassPerfModel::best_available(config, worker_type, options)` from Rust.
`worker_type` is an immutable property of the engine model, not an inferred
property of one FPM iteration. Python accepts exactly `"prefill"`, `"decode"`,
or `"aggregated"`; Rust uses
`ForwardPassWorkerType::{Prefill, Decode, Aggregated}`.

This path uses the native AIC estimate when the native estimator can be built,
learns online correction factors from FPM observations, and falls back to the
regression associated with `worker_type` for eligible native build or
data-availability failures. These include unsupported models and missing or
unreadable model, system, or performance data. A successful native build keeps
the existing workload-kind inference and correction behavior; `worker_type`
and regression-only weights do not alter native estimates. Check
`diagnostics()` to determine whether the active source is `aic`,
`aic_with_correction`, or `fallback_regression`, and to inspect any fallback
warning.

Native online corrections default to an absolute factor range of `[0.5, 2.0]`.
Pass `None` explicitly as `min_faster_correction_factor` or
`max_slower_correction_factor` in the options dictionary to remove the bound
in that direction. Regression fallback ignores both options.

Use `from_native(...)` instead when native AIC support is required and an
unsupported configuration or native data failure should surface rather than
fall back. This strict-native constructor does not take `worker_type`.

Use `RustForwardPassPerfModel.from_regression(worker_type, options=None)` or
`ForwardPassPerfModel::from_regression(worker_type, options)` for a
regression-only model. It owns one two-dimensional retained sample set and one
fit for the engine's fixed role. Its axes are consistently ordered as
`[critical attention, global FFN/MoE]`; bucket retention uses `log1p` of those
raw features, while fitting uses standardized raw features. The optional
regression weights below default to `1.0` and must be finite and strictly
positive:

- `regression_attention_kv_weight` (`alpha`);
- `regression_prefill_attention_pair_weight` (`beta`);
- `regression_ffn_token_weight` (`gamma`).

The ergonomic Python facade accepts ordinary Python floats. For these three
fields only, it marshals nonfinite values on a shallow copy of the options
dictionary as the exact valid-JSON string sentinels `"NaN"`, `"Infinity"`, and
`"-Infinity"`; finite values remain JSON numbers. Callers of the JSON-oriented
raw PyO3 API may use the same sentinels directly. The sentinels preserve values
for backend-dependent validation rather than making them valid regression
weights: `from_native` and a successful native `best_available` ignore all
three fields, while `from_regression` and a fallback `best_available` reject a
decoded nonfinite value with the corresponding field-specific error. Other
strings and value types remain invalid.

The formulas, role-compatibility rules, and fitting pipeline are specified in
the [FPM regression design](../docs/fpm/aic-fpm-regression-design.md).

`AicEngineBuilder` serves a different purpose: it constructs the strict native
Rust engine for direct public prefill and decode latency calls. It does not
provide regression fallback or online correction, so it is not a replacement
for `best_available(...)`.

```python
from aiconfigurator_core.sdk import RustForwardPassPerfModel

# Engine-config and per-rank FPM dictionary setup is omitted here.
model = RustForwardPassPerfModel.best_available(config, "decode")
diagnostics = model.diagnostics()
print(diagnostics["source"])
if diagnostics["last_warning"] is not None:
    print(diagnostics["last_warning"])

estimate_ms = model.estimate_forward_pass_time_ms(metrics_by_rank)
if estimate_ms is None:
    # Regression fallback starts without observations for this worker type.
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

The flat `build_aic_engine` adapter remains available in the 0.11.0 release and
is removed from `main` for the next minor release. Consumers upgrading past
0.11.0 must migrate to `AicEngineBuilder`.

The supported root-level Rust surface is grouped as follows:

- compiled engine: `AicEngineBuilder`, `AicEngine`, `AicError`;
- forward-pass estimation: `ForwardPassPerfModel`,
  `ForwardPassWorkerType`, `ForwardPassPerfOptions`,
  diagnostics/readiness/source types, and the `ForwardPassMetrics` telemetry
  types;
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
- Requiring `worker_type` in `from_regression` and `best_available` is a
  documented incompatible change for the next minor release. Adding the three
  public regression-weight fields to `ForwardPassPerfOptions` is also a Rust
  source break for downstream exhaustive struct literals. Rust callers should
  prefer update syntax such as
  `ForwardPassPerfOptions { bucket_count: 8, ..Default::default() }` so future
  option fields do not require source changes. This feature worktree does not
  itself change package versions; release coordination must publish matching
  crate and wheel minor versions.
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
