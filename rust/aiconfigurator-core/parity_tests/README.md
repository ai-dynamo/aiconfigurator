# Rust/Python Parity Tests

Temporary harness for the Rust `aiconfigurator-core` migration. (To be deprecated after the transition)

Serves 2 purposes:
- Rust-Python Parity check: the engine-step latency diff of the 2 should be < 1%
- Rust-Python speed benchmark & comparison: quantitively evaluate the speed boost from Rust


## Pytest Parity Suite

The parity suite and benchmark consume the Rust core as the installed
`aiconfigurator-core` PyO3 wheel. Install it first with
`pip install -e ".[rust]"` or `maturin develop --release` (from
`rust/aiconfigurator-core/`).

Run the smoke parity checks:

```bash
uv run pytest -q -rx rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py
```

The suite compares Python SDK output with Rust-backed output for:

- `static`: `static_ctx`, `static_gen`, and `static_total`
- `mixed_step`: one explicit mixed prefill/decode forward-pass metrics step
- `agg`: public `cli_estimate(mode="agg")`
- `disagg`: public `cli_estimate(mode="disagg")`

Current mismatches are reported through `pytest.xfail`. The xfail reason prints
the Python value, Rust value, absolute delta, percent delta, tolerance, and
status for each metric.

## Engine-Step Benchmark

Run the hot-cache Python SDK vs Rust engine-step API benchmark:

```bash
python rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --warmup 5 --iterations 50
```

When `--case` is omitted, the benchmark runs all predefined cases.
Before each case starts, the script clears Python database/op/model caches and
the Rust estimator/extension-import caches. Before each table row, it also clears that
engine's runtime query caches. The configured warmup iterations then repopulate
the hot-path caches before timed samples are collected.

Use `--warmup 0` to skip pre-timing warmup. In `hot` mode, only the first timed
sample is cold; later samples are hot again. Use `--cache-mode cold` when every
timed sample should clear runtime caches first.

Useful variants:

```bash
python rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --case minimax-m25 --warmup 5 --iterations 50
python rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --case kimi-k25 --warmup 10 --iterations 100
python rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --case kimi-k25 --cache-mode cold --warmup 0 --iterations 50
python rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --case minimax-m25 --json
```

The benchmark reports, per phase:

- local API-call latency p50/p90/p99 in microseconds
- Rust speedup versus the Python hot path

It also reports one-time Python session setup and Rust estimator setup. Rust
setup includes importing the PyO3 extension, loading Rust model metadata and
Rust perf DB data, and constructing the estimator. These setup costs are
excluded from the step-latency table.

Use command-line overrides such as `--model-path`, `--system-name`,
`--backend-version`, `--batch-size`, `--isl`, `--osl`, `--prefix`, and
parallelism flags when adding or investigating a specific parity case.
