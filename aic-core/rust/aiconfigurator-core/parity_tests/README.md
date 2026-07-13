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

## Engine-Step Benchmark

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
