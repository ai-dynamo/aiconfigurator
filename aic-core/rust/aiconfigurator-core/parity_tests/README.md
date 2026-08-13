# Rust/Python Parity Tests

Temporary harness for the Rust `aiconfigurator-core` migration. (To be deprecated after the transition)

Serves 2 purposes:
- Rust parity check against the frozen Python reference: the engine-step
  latency diff should be < 1% (tighter regimes for HYBRID/EMPIRICAL and SOL)
- Rust-Python speed benchmark & comparison: quantitively evaluate the speed boost from Rust


## Pytest Parity Suite

Run the engine-step parity checks:

```bash
AICONFIGURATOR_RUST_CORE_AUTOBUILD=1 uv run pytest -q -rx aic-core/rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py
```

The suite compares the live Rust engine against **golden fixtures** (see the
golden workflow below) for:

- `static`: `static_ctx`, `static_gen`, and `static_total` (plus the
  context/generation energy sums and power averages on the `POWER_CASES`,
  which sit on the power-carrying database identities)
- `mixed_step`: Rust `estimate_mixed_step_latency_with_rust` vs the frozen
  Python `_get_mix_step_latency` for the same shape
- `cp_static_ctx`: the context-phase static sum through the cp-aware model
  builder (`cli_estimate` has no cp knob, so the plain "static" surface
  cannot express a cp>1 case; issue #1498 anchor)
- `agg`: public `cli_estimate(mode="agg")`
- `disagg`: public `cli_estimate(mode="disagg")`
- `afd`: public `cli_estimate(mode="afd")` (ttft/tpot; the AFD session's
  per-op values cross the op-list evaluate FFI)

The case matrix has grown far past the original 3-case/12-surface smoke set:
`SMOKE_CASES` (61) x 4 surfaces, `POWER_CASES` (5, energy/power coverage) x 4,
`CP_CASES` (3, mixed only), `DSV4_CP_CASES` (1, cp_static_ctx + mixed —
the issue #1498 adjudicated repro on the reuse-carrying 0.5.12),
`HYBRID_CASES` (17) x 4 at a 1e-4 rtol,
`SOL_CASES` (4, static+mixed) at 1e-4, the two #1456 site-transfer
tie-break anchors (`TIE_AGG_CASES`/`TIE_DISAGG_CASES`), and `AFD_CASES`
(1, afd only) — 348 golden-backed (case, surface) pairs, plus the
typed-error/provenance contract tests and the anti-vacuous golden guards. If a parity assertion fails, the message prints
the golden (Python) value, Rust value, absolute delta, percent delta,
tolerance, and status for each metric.

`test_compile_engine_parity.py` covers the `compile_engine` -> `EngineHandle`
path specifically: op-transfer bincode round-trip fidelity, integration
parity against the frozen Python `BaseBackend` references, and the per-op
FFI anchor (`run_static_per_op` folded by name vs the frozen Python summary
per-op latency/energy/source dicts). Both suites run in the
`rust-engine-step-parity` CI job (`build-test.yml`).

Build the `aiconfigurator_core` extension first (the CI job does this with
`maturin develop --release`; from a clean checkout run
`cd aic-core && ../.venv/bin/maturin develop --release`), then return to the
repository root and run:

```bash
uv run pytest -q aic-core/rust/aiconfigurator-core/parity_tests/test_compile_engine_parity.py
```

## Golden Fixtures (dedup-plan Gate 2)

The parity suites used to run the Python engine live on every comparison.
Phase 2 of `docs/python-dedup-plan.md` deletes the duplicated Python latency
path, which would destroy that live differential oracle — so Gate 2 froze the
Python reference into fixtures **while the Python path is still alive**:

- `goldens/engine_step.json` — every (case, surface) pair in
  `ENGINE_STEP_GOLDEN_MATRIX`, as `{"values": {...}}` or (error-symmetry
  cases) `{"error": ExceptionClassName}` records.
- `goldens/compile_engine.json` — the compile-engine subset references
  (static/mixed/decode per case + chunked-prefill, imbalance-scale, and
  WideEP references).
- `goldens/per_op.json` — the Python summary per-op dicts (latency + energy
  + source) for the 10-case subset (one member sits on a power-carrying
  identity, so its `energy_wms` values are nonzero and the per-op energy
  comparison actually executes); the per-op op-list FFI anchor.

The tests compare **live Rust vs golden**; only
`regenerate_goldens.py` ever runs the Python side. Regenerate with:

```bash
.venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/regenerate_goldens.py
```

Regenerate **deliberately** — when a case list or a compared metric changes,
or when a Python-reference change is intentional and reviewed — and commit
the diff with that change. Never regenerate to silence a parity failure: a
red Rust-vs-golden test means the Rust engine drifted from the frozen
reference. The capture is byte-reproducible (sorted keys, full-precision
floats, no timestamps; the header records the capture HEAD), so running it
twice and diffing proves a clean capture. `TestGoldenComparisonGuards`
doctors an in-memory golden to prove the comparison itself still bites.

## Perf Gate (CI)

`test_engine_step_perf.py` is the performance analog of the parity suite: it
asserts the compiled Rust engine-step stays at least a floor multiple as fast as
the pure-Python step, per case.

```bash
uv run pytest -q -rA aic-core/rust/aiconfigurator-core/parity_tests/test_engine_step_perf.py
```

Because Python and Rust are timed **back-to-back on the same host**, the
reported speedup *ratio* is far **more comparable across machines** than an
absolute wall-clock number — most of the machine-speed variance divides out
(the ratio can still shift somewhat across architectures; see the perf report's
ARM-vs-x86 note). That is why it is safe as a blocking gate on shared CI runners
where absolute wall-clock is noisy (it runs as a step in the
`rust-engine-step-parity` job in `build-test.yml`, reusing the same built
extension).

It exists to catch algorithmic regressions in the Rust hot path — e.g. a
per-query `SiteIndex::resolve` that sorts every collected GEMM site
(`O(n log n)`) instead of selecting the nearest handful (`O(n)`). That class of
bug once pushed the Rust step to 0.15–0.78x of Python.

Per-case floors live in `MIN_SPEEDUP` and are all **≥ 1.0** — the gate encodes
the goal that Rust must be at least as fast as Python on every guarded case.
`nemotron-nas` (large graph, wide stable margin ~1.9–2.3x) uses a 1.5x floor
that also catches partial regressions; the small ~20 us graphs (`deepseek-v3`,
`minimax-m25`) sit near the FFI-tax floor (~1.1–1.5x) and use 1.0x. On failure
the assertion prints a per-phase table of Python p50, Rust p50, speedup, floor,
and status.

## Engine-Step Benchmark

The latest full-family speedup numbers (dated + commit-stamped) live in
[`perf-speedup-report.md`](../docs/perf-speedup-report.md) — regenerate it from this
harness when the Rust hot path changes.

Run the hot-cache Python SDK vs Rust engine-step API benchmark:

```bash
python aic-core/rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --warmup 5 --iterations 50
```

When `--case` is omitted, the benchmark runs all predefined cases.
Before each case starts, the script clears Python database/op/model caches and
Rust estimator/library caches. Before each table row, it also clears that
engine's runtime query caches. The configured warmup iterations then repopulate
the hot-path caches before timed samples are collected.

Use `--warmup 0` to skip pre-timing warmup. In `hot` mode, only the first timed
sample is cold; later samples are hot again. Use `--cache-mode cold` when every
timed sample should clear runtime caches first.

Useful variants:

```bash
python aic-core/rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --case minimax-m25 --warmup 5 --iterations 50
python aic-core/rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --case kimi-k25 --warmup 10 --iterations 100
python aic-core/rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --case kimi-k25 --cache-mode cold --warmup 0 --iterations 50
python aic-core/rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --case minimax-m25 --json
```

The benchmark reports, per phase:

- local API-call latency p50/p90/p99 in microseconds
- Rust speedup versus the Python hot path

It also reports one-time Python session setup and Rust estimator setup. Rust
setup includes importing the maturin-built `aiconfigurator_core` extension,
loading Rust model metadata and Rust perf DB data, and constructing the
estimator, but excludes `cargo build` / `maturin develop`. These setup costs
are excluded from the step-latency table.

Use command-line overrides such as `--model-path`, `--system-name`,
`--backend-version`, `--batch-size`, `--isl`, `--osl`, `--prefix`, and
parallelism flags when adding or investigating a specific parity case.
