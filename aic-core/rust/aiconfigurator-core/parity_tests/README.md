# Rust Engine Regression Tests (frozen goldens)

The migration harness that once compared the live Rust engine against the
live Python engine step. The Python step path is gone (dedup-plan Gate 3);
the golden fixtures captured from it while it was alive are now the
**permanent regression oracle** for the compiled engine.

## Pytest Suites

Run the engine-step golden checks:

```bash
AICONFIGURATOR_RUST_CORE_AUTOBUILD=1 uv run pytest -q -rx aic-core/rust/aiconfigurator-core/parity_tests/test_engine_step_parity.py
```

The suite compares the live Rust engine against **golden fixtures** for:

- `static`: `static_ctx`, `static_gen`, and `static_total` (plus the
  context/generation energy sums and power averages on the `POWER_CASES`,
  which sit on the power-carrying database identities)
- `mixed_step`: `estimate_mixed_step_latency_with_rust` for the same shape
- `cp_static_ctx`: the context phase through the cp-aware model builder
  (`cli_estimate` has no cp knob)
- `agg`: public `cli_estimate(mode="agg")`
- `disagg`: public `cli_estimate(mode="disagg")`
- `afd`: public `cli_estimate(mode="afd")` (ttft/tpot; the AFD session's
  per-op values cross the op-list evaluate FFI)

The case matrix: `SMOKE_CASES` x 4 surfaces, `POWER_CASES` (energy/power
coverage) x 4, `CP_CASES` (mixed only), `DSV4_CP_CASES` (cp_static_ctx +
mixed), `HYBRID_CASES` x 4 at a 1e-4 rtol, `SOL_CASES` (static+mixed) at
1e-4, the two #1456 site-transfer tie-break anchors
(`TIE_AGG_CASES`/`TIE_DISAGG_CASES`), and `AFD_CASES` — plus the
typed-error/provenance contract tests and the anti-vacuous golden guards.
If an assertion fails, the message prints the golden value, Rust value,
absolute delta, percent delta, tolerance, and status for each metric.

`test_compile_engine_parity.py` covers the `compile_engine` -> `EngineHandle`
path specifically: op-transfer bincode round-trip fidelity, integration
checks against the frozen references, and the per-op FFI anchor
(`run_static_per_op` folded by name vs the frozen per-op
latency/energy/source dicts). Both suites run in the
`rust-engine-step-parity` CI job (`build-test.yml`).

Build the `aiconfigurator_core` extension first (the CI job does this with
`maturin develop --release`; from a clean checkout run
`cd aic-core && ../.venv/bin/maturin develop --release`), then return to the
repository root and run:

```bash
uv run pytest -q aic-core/rust/aiconfigurator-core/parity_tests/test_compile_engine_parity.py
```

## Golden Fixtures (captured at Gate 2, frozen at Gate 3)

- `goldens/engine_step.json` — every (case, surface) pair in
  `ENGINE_STEP_GOLDEN_MATRIX`, as `{"values": {...}}` or (error-symmetry
  cases) `{"error": ExceptionClassName}` records.
- `goldens/compile_engine.json` — the compile-engine subset references
  (static/mixed/decode per case + chunked-prefill, imbalance-scale, and
  WideEP references).
- `goldens/per_op.json` — the summary per-op dicts (latency + energy +
  source) for the compile-engine subset; the per-op op-list FFI anchor.

The FPM parity class (`TestRustEngineStepFpmParity`) follows the same
freeze-then-delete pattern with INLINE frozen references (`_FPM_*_FROZEN`):
its dataset is generated per-run from `_FPM_ROWS`, so it sits outside
`ENGINE_STEP_GOLDEN_MATRIX` and its Python-side values were frozen into the
test module at capture time.

The Python-era records were captured from the live Python engine step by the
retired `regenerate_goldens.py` (byte-reproducible: sorted keys, full float
repr, thread caps and capture HEAD in the header) and can never be
recaptured — the reference implementation is gone. **Never edit a frozen
value to silence a red test**: a Rust-vs-golden failure means the engine
drifted from the frozen reference; either the drift is a bug (fix it) or it
is a deliberate modeling change (pin the new values, below, and let the
golden diff carry the review).

### Post-freeze golden maintenance: `pin_goldens.py`

```bash
.venv/bin/python aic-core/rust/aiconfigurator-core/parity_tests/pin_goldens.py
```

- **Default (append-only)**: pins records for matrix entries the fixtures
  lack — i.e. parity cases added after the freeze. Values come from the live
  rust engine and are provenance-marked in the file's `post_freeze_pins` map,
  so python-era frozen values and rust-pinned values stay distinguishable.
- `--refresh KEY ...` / `--refresh-all`: recompute existing records after a
  **deliberate, reviewed** rust-side modeling change. The golden diff in the
  PR is the review artifact — reviewers see exactly which numbers moved and
  by how much.

The pin script keeps the retired capture script's guards: clean-tree
requirement, pinned thread caps, byte-reproducible output, and
all-payloads-before-any-write. `TestGoldenComparisonGuards` proves the
comparison itself still bites.

## Engine-Step Benchmark

Historical Python-vs-Rust speedup numbers (dated + commit-stamped) live in
[`perf-speedup-report.md`](../docs/perf-speedup-report.md); they cannot be
regenerated (the Python arm is gone). The benchmark now times the rust
engine-step alone:

```bash
python aic-core/rust/aiconfigurator-core/parity_tests/benchmark_engine_step.py --warmup 5 --iterations 50
```

When `--case` is omitted, the benchmark runs all predefined cases.
Before each case starts, the script clears Python database/op/model caches and
Rust estimator/library caches. Use `--cache-mode cold` when every timed sample
should clear runtime caches first, and `--json` for machine-readable output.

The relative Rust-vs-Python CI perf gate (`test_engine_step_perf.py`)
retired with the Python step: its floors encoded "Rust must not lose to
Python", which the migration completed. If an absolute perf tripwire is
wanted, pin per-case wall-clock budgets from this benchmark on a quiet host.
