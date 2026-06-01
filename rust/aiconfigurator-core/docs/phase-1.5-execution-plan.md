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
compiled `Engine` artifact, with PyO3 binding and feature-gated
no-Python build for external Rust callers. 14/14 parity tests passed
bit-identical at 1e-12 tolerance.

Phase 1.5 is the production-grade adoption of that architecture.

## Goal

> Python owns model construction. Rust owns execution.
> One Rust core. Two callers (PyO3 + direct crate).
> The hot path supports a **batched** entry point so a sweep amortizes
> FFI tax across all points in one call.

After Phase 1.5:
- `rust/aiconfigurator-core/src/models/` is **deleted** (~6.7 kLoC).
  `config_loader.rs`, `factory.rs`, `registry.rs`, family builders,
  `base::WideEpMode`. None of it survives.
- `rust/aiconfigurator-core/src/operators/`,
  `perf_database/`, `interpolation.rs`, `common/`, the parts of
  `session.rs` that drive op-list execution: **kept and reused**.
- Python's `sdk/models/*.py` becomes the single source of truth for
  model topology. A new `compile_engine(model, runtime_config) →
  EngineHandle` Python function walks the existing op list and ships
  it across the boundary.
- A single batched FFI entry point —
  `Engine.run_batch(points: &[StepPoint]) -> Vec<StepResult>` —
  amortizes serialization cost across sweep points and enables rayon
  fan-out on the Rust side.
- External Rust callers (Mocker today, others later) link the crate
  directly and call the same `run_static_internal` underneath, with
  zero Python in the picture.

## Out of scope

- Re-running collectors or regenerating perf DB data.
- Changing the perf DB schema or the support-matrix CSV format.
- Porting Python's CLI, generator, Pareto analysis, or webapp.
- The R1 / R5 scan DRIFT clusters (`phase1/support-matrix-scan.md`).
  They are scan-comparator artifacts and tracked separately.
- Multi-process / distributed sweeping. Single-process rayon
  parallelism is the Phase 1.5 deliverable; cross-process Dynamo
  Mocker integration is a downstream consumer of the same crate.

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

What this buys:

- **End users on wheels:** no env var, no first-call stall. `pip
  install` works and the Rust path is live on the first call.
- **Developers:** same total compile cost, paid at install instead
  of at first use. `maturin develop` handles the iteration loop.
- **CI:** one explicit `maturin build` step replaces the env-var-
  driven lazy compile.
- **Source-installers without wheels:** pay the same compile cost,
  just predictably at install time rather than during first run.

What this requires (the work lives inside E1):

- `pyproject.toml`: switch `build-backend` from
  `setuptools.build_meta` to `maturin`. Add `[tool.maturin]` with
  `module-name = "aiconfigurator_core"`, `features = ["python"]`,
  `abi3 = true`, `python-source = "src"`.
- The `AICONFIGURATOR_RUST_CORE_AUTOBUILD` env var becomes a no-op
  emitting a deprecation warning, kept for one release cycle to
  avoid breaking automation that sets it. Drop in a follow-up.
- `sdk/rust_engine_step.py` becomes a thin facade that imports
  `aiconfigurator_core` and translates Phase 1's ctypes-shaped
  calls into the PyO3 API. Marked deprecated; removal belongs to a
  later phase.
- CI: replace `cargo build --release` jobs with
  `maturin build --release --strip` for wheel-publishing flows;
  source-checkout CI uses `maturin develop`. The external-Rust-
  caller binary (E8) builds with plain `cargo build
  --no-default-features` and never pulls libpython.

## Architecture target

```text
                Python                                         Rust
   ───────────────────────────────                ────────────────────────────────
   sdk/models/<family>.py
   builds Operation objects
   ↓
   compile_engine(model, runtime_config)
   ↓
   walk model.context_ops + model.generation_ops
   ↓
   produce list[OpSpec]  (plain data)

   ─────── PyO3: build_engine ─────►   Engine { ops, metadata }
                                                  │
                                                  ↓
   for batch of (bs, isl, osl, mode) points:
       engine.run_batch(points)  ────►   par_iter(points).map(|p|
                                              run_static_internal(&db, &engine, p))
                                                  │
                                                  ↓
                                          Vec<StepResult>
   ◄────── results across PyO3 ───────
```

Same `Engine` type, two ABI surfaces:

| Surface | Caller | Build mode | Hot path |
| --- | --- | --- | --- |
| PyO3 extension `aiconfigurator_core.cpython-*.so` | Python | `maturin develop`, default feature | `engine.run_batch(points)` releases GIL once, runs rayon, returns Python list |
| `librlib` linked into an external Rust binary | Future external Rust callers (replay drivers, pure-Rust evaluators, possibly a future Mocker migration) | `cargo build --no-default-features` | direct `engine.run_static_internal(&db, p)` per call, or `engine.run_batch(...)` |

The `python` feature gates only the PyO3 bindings. Crate compilation
without `--features python` produces zero libpython references — the
external-Rust-caller contract from the PoC. Note that today's Dynamo
Mocker integration is **not** on this path; it goes through Python
(see the next section).

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

Three callbacks today:

| # | Trait method (Rust contract) | Hot? | Phase 1 implementation |
| --- | --- | --- | --- |
| **H1** | `predict_prefill(batch_size, effective_isl, prefix) -> f64` ms | Yes — per scheduler pass | Python op-walk over `model.context_ops` |
| **H2** | `predict_decode(batch_size, isl, osl) -> f64` ms (Mocker passes `osl=2`) | Yes — per scheduler pass; **agg mode calls H1+H2 sequentially per pass** | Python op-walk over `model.generation_ops` with `DEFAULT_STATIC_STRIDE=32` quadrature |
| **H3** | `estimate_num_gpu_blocks(backend, system, model_path, tp, block_size, max_num_batched_tokens, gpu_memory_utilization, mem_fraction_static, ...) -> usize` | No — once at startup | Python `backend._get_memory_usage` + per-backend KV budget formula |

### What's frozen

- **H1–H3 trait signatures** (already in production in Mocker's
  scheduler) — return types and parameter order stay.
- **Numerical outputs** within Phase 1's parity tolerance against
  the current Python op-walking path. Validated by the E6 parity
  gate plus one Mocker integration test post-flip.

### What's negotiable (Phase 1.5 may change freely)

- **Python module path** (`dynamo._internal.aic`) and class name
  (`AicSession`). Both sides can rename in lock-step.
- **Constructor signature** — `create_session(...)` kwargs,
  position of `attention_dp_size`, default-version fall-through,
  which side caches the `EngineHandle`. All open.
- **Implementation tier.** Three viable shapes for the
  `Arc<dyn AicCallback>` Mocker holds:

  | Tier | Where the impl lives | Per-call cost | Mocker delta |
  | --- | --- | --- | --- |
  | **A: Thin Python wrapper** | `AicSession` (Python) keeps an `EngineHandle`; `predict_*` routes to `engine.run_batch([point])[0]` | 1 GIL hop + Rust query (~5 µs) | None — `PyAicCallback` unchanged |
  | **B: Hybrid build-then-call** | Python compiles the `EngineHandle` at startup, serialises to bincode, hands bytes to Rust; Rust holds the engine and the callback | 0 GIL hops on hot path; 1 Python call at startup | Mocker swaps `Arc<PyAicCallback>` for `Arc<RustAicCallback>` post-init |
  | **C: Pure Rust** | `aiconfigurator_core` builds the engine without Python | 0 GIL hops; 0 Python at startup | Mocker drops the Python import entirely |

  Tier C requires `aiconfigurator_core` to grow a Python-free
  model-build path, which contradicts Phase 1.5's "Python owns
  model construction" decision. **Recommendation: ship Tier A in
  Phase 1.5 to close the per-call op-walk gap; sequence Tier B as
  a follow-up if Mocker's scheduler thread count makes GIL
  contention measurable.** Tier C stays out of scope.

### What Phase 1.5 does NOT require Mocker to change

- The PyO3 import-and-call pattern under Tier A. Mocker's
  `Python::with_gil` plus `module.call_method1("create_session", ...)`
  keeps working; only `AicSession`'s internal body swaps.
- The single-point call pattern. Mocker calls per scheduling
  decision, not per sweep batch — `engine.run_batch([single_point])[0]`
  is the shape on the AIC side. The batched-rayon win is for the
  AIC sweep caller, not Mocker.
- Adoption of the direct-rlib path (`src/bin/mocker_demo.rs`,
  E8). Available, not required. Tier B is the natural future home
  for that path; not a Phase 1.5 obligation.

### Coordination items, mapped to the commit sequence

- **Before E6:** pin Mocker's CI to a known-good aiconfigurator
  version so the parity flip doesn't break Mocker's gate.
- **At E6:** run one Mocker integration test against the new
  Tier-A Engine-backed path. Assert H1/H2 outputs within Phase 1's
  1% tolerance.
- **At E7:** Rust `models/` + `backends/` deletion is invisible to
  H3 (Phase 1 Python backends own `estimate_num_gpu_blocks`).
  Rerun the Mocker test to confirm.
- **Tier B and E8** are post-Phase-1.5 work; flagged here so the
  trait stays general enough to host them later.

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
| **E1** | Build-system: maturin + PyO3 + feature gate | **Cargo:** add `pyo3 = { version = "...", features = ["abi3-py39"], optional = true }` and a `python` feature; `[lib] crate-type = ["rlib", "cdylib"]` unchanged. **pyproject.toml:** switch `build-backend` from `setuptools.build_meta` to `maturin`; add `[tool.maturin]` (module-name, features, abi3, python-source). **Compatibility:** deprecate `AICONFIGURATOR_RUST_CORE_AUTOBUILD` to a no-op with warning. **Verify both build paths:** `maturin develop` produces a Python-importable extension; `cargo build --release --no-default-features` produces a clean rlib + cdylib with zero libpython symbols (`nm -u` check in CI). | n/a |
| **E2** | OpSpec types | Public `OpSpec` enum + `EngineSpec` struct in `src/engine/spec.rs`. Mirror the `Op` enum 1:1 with `serde::{Serialize, Deserialize}`. Bincode round-trip unit tests for every variant. | n/a |
| **E3** | `Engine::build` + `Engine::run_static_internal` | New `src/engine/mod.rs` constructs `Engine` from an `EngineSpec` and exposes `run_static_internal(&db, point)`. **Reuses** existing `operators::*` and `perf_database::*` underneath. Initial implementation may shim through `session.rs`. | unit |
| **E4** | PyO3 bindings | `src/py.rs` (feature-gated): `PyEngine::run_batch(py, points: Vec<PyStepPoint>) -> Vec<PyStepResult>`. Releases GIL via `py.allow_threads`. Rayon fan-out internal. `PyEngine::from_spec(bytes)` accepts the bincode artifact. | unit |
| **E5** | Python builder | `src/aiconfigurator/sdk/engine.py`: `compile_engine(model, runtime_config) -> EngineHandle`. Walks `model.context_ops` + `model.generation_ops`, converts each `Operation` to an OpSpec dict, bincodes via `aiconfigurator_core.encode_spec`, returns a handle. `EngineHandle.run_batch(points)` shells to the PyO3 entry point. **Reuses** existing Python `sdk/models/*.py` unmodified. | integration |
| **E6** | Parity flip | Switch `sdk/rust_engine_step.py` to drive `compile_engine` + `EngineHandle.run_batch` instead of the ctypes JSON path. Re-run the 164-surface smoke harness; all assertions must hold bit-identical (or within the same 1% tolerance Phase 1 holds). | **GATE** |
| **E7** | Delete Rust model layer | Remove `src/models/`, `src/backends/`, the `factory.rs` / `registry.rs` / `config_loader.rs` files. Remove the `EngineStepEstimator` ctypes path and the JSON FFI in `src/ffi.rs` (the binary external-Rust-caller path in E8 replaces it). Net delta: ~7.5 kLoC removed, ~2 kLoC added. | parity re-run |
| **E8** | External-Rust-caller path | `src/bin/mocker_demo.rs` (or rename per Mocker integration ask): loads a bincode `EngineSpec` from disk and runs `run_batch` with no Python in the picture. Compiles with `--no-default-features`. CI step that builds this target verifies the no-libpython contract. | regression suite |
| **E9** | Batched perf gate | Benchmark the **sweep workload** — N=1 / 8 / 64 / 1024 points — Python-loop vs `EngineHandle.run_batch`. Exit criterion: **≥3× p50 speedup on `N ≥ 64` for the small-graph families that today regress below 1× per call** (MiniMax-M2.5 gen, NemotronNas gen, DSv3.2 ctx). Big-graph families should hit the same or higher multiple. Numbers land in `parity_tests/benchmarks.md`. | **GATE** |

## Commit dependencies

```text
E0 ──► E1 ──► E2 ──► E3 ──► E4 ──┬─► E5 ──► E6 ──► E7 ──► E8 ──► E9
                                 │
                                 └── E5 can start once E2 is in (parallel with E3/E4)
```

E0 strictly first (it can invalidate E2's shape). E1–E4 are sequential
Rust changes. E5 unblocks once E2 lands. E6 is the parity gate; E7
must not land before E6 passes. E8 and E9 are independent of each
other but both depend on E7 (E8 verifies the no-libpython contract on
the cleaned-up crate; E9 measures the production hot path).

## Acceptance criteria

1. **Parity (E6 gate):** the existing 164-surface smoke harness
   asserts bit-identical-or-within-tolerance against the Phase 1
   numbers. The full-matrix scan re-runs on the same SQLite schema
   and shows `STRICT_PASS >= 1906`, `DRIFT <= 16` (modulo the NCCL fix
   that already landed), `REGRESSION == 0`.
2. **External-Rust-caller (E8 gate):** `cargo build --release
   --no-default-features --bin mocker_demo` produces a binary with
   zero `_Py_*` / `_PyObject_*` symbols (`nm -u target/release/mocker_demo
   | grep -E '_Py' && exit 1 || true`).
3. **Performance (E9 gate):** for the small-graph families that
   currently regress below 1× warm-path, batched throughput at
   `N ≥ 64` reaches **≥3× the Python sweep loop**. For
   medium / big graphs, batched throughput is **at least as good as
   Phase 1 per-call**.
4. **LoC discipline:** net `+/-` is negative. The Rust crate shrinks
   meaningfully (target: −5 000 LoC after E7). Python `sdk/models/`
   stays the size it is today plus the new `engine.py` (~300–500
   LoC).
5. **No CLI / generator / Pareto changes.** Phase 1.5 is strictly
   internal to `sdk/`.

## Risks and open questions

| Risk / question | Mitigation |
| --- | --- |
| **OpSpec audit (E0) surfaces fields Python doesn't know.** | E0 must complete *before* E2 commits to a wire format. If category-4 fields are non-trivial, escalate before any Rust deletion. |
| **Build-backend swap (setuptools → maturin).** Phase 1's `pyproject.toml` uses `setuptools.build_meta` with no Rust hooks; the Rust crate compiles lazily at runtime when `AICONFIGURATOR_RUST_CORE_AUTOBUILD=1` is set. Switching to maturin moves the compile from runtime to install (for source installs) and ships a real Python extension inside the wheel. | E1 verifies both paths: `maturin develop` for local dev, `maturin build --release --strip` for wheel CI, and `cargo build --no-default-features` for the external-Rust-caller binary. The `AUTOBUILD` env var stays as a deprecated no-op for one release cycle so existing automation doesn't break silently. `abi3-py39` keeps a single wheel valid across Py 3.9–3.12. |
| **Mocker `AicSession` contract.** Dynamo Mocker imports `dynamo._internal.aic` and calls `AicSession.predict_prefill` / `predict_decode` / `estimate_num_gpu_blocks` via Dynamo's embedded CPython — **not** the ctypes FFI. Phase 1.5 must preserve the Python class's public signatures and numerical outputs. | Internal swap from Python op-walking to `engine.run_batch([point])` is transparent to Mocker. E6 gate includes a Mocker-style single-point parity check; one Mocker integration test runs against the new path before E7 lands. See the **AIC ↔ Dynamo Mocker handshake** section. |
| **GIL release alone won't fix small-graph speed.** Already verified empirically (`benchmarks.md` FFI-tax breakdown). | Phase 1.5's perf criterion is batched, not per-call. E9 is unambiguous: ≥3× **on N ≥ 64 sweeps**, not per-call. |
| **rayon fan-out non-determinism vs the parity harness.** Op-graph execution is per-point pure; no cross-point state. Determinism should hold. | E5 integration tests run rayon with `RAYON_NUM_THREADS=1` and `=8` and assert identical output. |
| **Pyarrow / parquet version skew.** PoC used `parquet = "55"`; we should pin against the same version the existing crate already uses. | Already on `parquet = "55"` (see `Cargo.toml`); no change. |
| **CI wheels for macOS / Linux / Windows × Py 3.9–3.12.** Out of scope for Phase 1.5; flagged for downstream packaging. | abi3 + maturin makes this tractable later. Phase 1.5 only needs local-dev `maturin develop` to work. |

## What this plan does NOT promise

- A speedup on N=1 single-point calls. The per-call FFI tax exists by
  design (the caller chose to amortize across a batch). Single-point
  callers will see roughly the same per-call latency as Phase 1.
- A schema change to the perf DB.
- A change to which Python surfaces the CLI exposes.
- Removal of `sdk/rust_engine_step.py`. It becomes a thin facade over
  `sdk/engine.py`. Deprecation/removal belongs to a later phase that
  is outside this plan.

## Open questions for review

Surface these before approval:

1. **Cache strategy on the Python side.** Where does
   `EngineHandle` live across sweep points — Task object, Predictor
   strategy, module-level cache? Recommendation: hold on the
   Predictor that runs the sweep, keyed by `(model_id,
   runtime_config)`. Defers to the existing Phase 1 SDK
   `Predictor` Protocol.
2. **OpSpec versioning.** Embed a `schema_version: u16` in
   `EngineSpec` from day one so future incompatible changes are
   detectable, even though we don't ship across process boundaries
   today.
3. **Whether to keep the JSON FFI shim in E7** for one release
   cycle. Default: yes, marked deprecated.
4. **Mocker handoff plan.** Pin Mocker's CI to a known-good
   aiconfigurator version across the E6 parity flip; validate one
   Mocker integration test post-flip; coordinate the merge once
   green. The direct-rlib path (E8) is offered for future callers,
   not required for Mocker. See the **AIC ↔ Dynamo Mocker
   handshake** section.
5. **DRIFT re-scan timing.** The NCCL/OneCCL fix already landed on
   `rust-migration/phase1`. Re-run the full-matrix scan once, fold
   results into the Phase 1 checkpoint, then start Phase 1.5; or
   start Phase 1.5 in parallel? Recommendation: parallel — the scan
   takes hours, Phase 1.5 is a multi-week effort.

## Pointers

- Architectural framing (the PoC results that motivate this):
  `design_doc.html`.
- What Phase 1 delivered: `phase1/phase-1-checkpoint.md`.
- Job definition (immutable contract): `phase1/migration-execution-plan.md`.
- Module map and current Rust shape: `phase1/migration-map.md`.
- FFI-tax breakdown that drives the batched-entry decision:
  `parity_tests/benchmarks.md` ("FFI overhead caveat" section).
