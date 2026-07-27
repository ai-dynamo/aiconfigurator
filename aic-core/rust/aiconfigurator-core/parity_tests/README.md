# Rust/Python Parity Tests

Temporary harness for the Rust `aiconfigurator-core` migration. (To be deprecated after the transition)

Serves 2 purposes:
- Rust-Python Parity check: the engine-step latency diff of the 2 should be < 1%
- Rust-Python speed benchmark & comparison: quantitively evaluate the speed boost from Rust


## Pytest Parity Suite

Run the smoke parity checks:

```bash
AICONFIGURATOR_RUST_CORE_AUTOBUILD=1 uv run pytest -q -rx aic-core/rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py
```

The suite compares Python SDK output with Rust-backed output for:

- `static`: `static_ctx`, `static_gen`, and `static_total`
- `mixed_step`: Python `_get_mix_step_latency` vs Rust
  `estimate_mixed_step_latency_with_rust` for the same shape
- `agg`: public `cli_estimate(mode="agg")`
- `disagg`: public `cli_estimate(mode="disagg")`

After Phase 3 C8-C10, all 12 smoke surfaces (3 cases x 4 modes) pass within
the 1% drift tolerance and the tests assert hard. If a parity assertion ever
fails again, the failure message prints the Python value, Rust value, absolute
delta, percent delta, tolerance, and status for each metric.

`test_compile_engine_parity.py` covers the `compile_engine` -> `EngineHandle`
path specifically: op-transfer bincode round-trip fidelity plus integration
parity against the Python `BaseBackend`. Both suites run in the
`rust-engine-step-parity` CI job (`build-test.yml`).

Build the `aiconfigurator_core` extension first (the CI job does this with
`maturin develop --release`; from a clean checkout run
`cd aic-core && ../.venv/bin/maturin develop --release`), then return to the
repository root and run:

```bash
uv run pytest -q aic-core/rust/aiconfigurator-core/parity_tests/test_compile_engine_parity.py
```

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
setup includes loading the shared library through `ctypes`, loading Rust model
metadata and Rust perf DB data, and constructing the estimator, but excludes
`cargo build`. These setup costs are excluded from the step-latency table.

Use command-line overrides such as `--model-path`, `--system-name`,
`--backend-version`, `--batch-size`, `--isl`, `--osl`, `--prefix`, and
parallelism flags when adding or investigating a specific parity case.
