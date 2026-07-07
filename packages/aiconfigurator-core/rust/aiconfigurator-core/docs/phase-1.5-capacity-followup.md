# Phase 1.5 Capacity API — deferred Dynamo wrapper (follow-up)

**Status:** the in-repo surface is complete (K1–K3). The Dynamo-side rewrite is
a separate, downstream PR in the `ai-dynamo/dynamo` repo and is intentionally
NOT done here.

## What landed in-repo (K1–K3)

The KV-cache capacity API (Issue
[#1159](https://github.com/ai-dynamo/aiconfigurator/issues/1159)) is fully
implemented inside `aiconfigurator`:

- **Python source of truth** `aiconfigurator_core.sdk.memory.estimate_kv_cache(...)`
  — `src/aiconfigurator/sdk/memory.py`. Owns the COMPLETE estimate: fraction +
  tolerance validation, the native `BaseBackend._get_memory_usage` breakdown
  path, the naive HF-config fallback, AND the `tolerance_adjusted` margin. Flat
  kwargs in, a `dict` out (`total_*`, `source`, `memory_breakdown`,
  `tolerance_adjusted`).
- **Rust top-level fn** `aiconfigurator_core::estimate_kv_cache(req)` —
  `rust/aiconfigurator-core/src/memory.rs`. A **pure** PyO3 forwarder for
  embedded callers (the Dynamo Mocker): it crosses into Python once to call the
  function above (forwarding `tolerance_fraction`) and rebuilds a
  `KvCacheEstimate` from the returned dict, with no math of its own. It is NOT
  exposed as a `#[pyfunction]` — there is no `aiconfigurator_core.estimate_kv_cache`
  Python surface; in-repo Python callers use the `sdk.memory` function directly.
- **AIC-Python convenience** `aiconfigurator_core.sdk.memory.estimate_num_gpu_blocks(
  ..., scheduler_block_size)` — calls `estimate_kv_cache` directly (in-process,
  no Rust hop) and returns `floor(total_kv_size_tokens / scheduler_block_size)`,
  using the tolerance-adjusted token count when `tolerance_fraction` is set, else
  the raw count. This is the in-repo reference for the conversion the Dynamo shim
  will mirror, and it is exercised end-to-end against a real perf DB by the
  integration test (`tests/integration/test_memory_estimation.py`).

### Coverage

- **Python estimate** (logic, incl. the tolerance math that moved out of Rust):
  `tests/unit/sdk/test_memory_estimation.py` (synthetic breakdowns + naive
  fallback) and the native path end-to-end in
  `tests/integration/test_memory_estimation.py` (real perf DB). Neither is a
  Python-vs-Rust parity test -- there is no independent Rust implementation, since
  the Rust `estimate_kv_cache` forwards to this same Python code.
- **Rust forwarder round-trip** (`fetch_python_estimate` forwarding
  `tolerance_fraction` -> `estimate_from_dict` parsing `tolerance_adjusted` back):
  `rust/aiconfigurator-core/tests/memory_round_trip.rs`, an enforced end-to-end
  test against a real perf DB. The Python emit side and the Rust parse side share
  the same dict keys (`tolerance_fraction` / `total_kv_size_bytes` /
  `total_kv_size_tokens`), so a drift on either end fails this test.

### Deferred: share the budget formula with the OOM check ([#1208])

`sdk/memory.py` exposes `kv_cache_budget_bytes(...)` as the single KV-cache budget
formula (used by `estimate_kv_cache`). `InferenceSummary._check_and_set_kv_cache_oom`
in the config sweep still carries its own copy of the same math. Wiring the OOM
check onto `kv_cache_budget_bytes` (and reconciling the `reserved`/tolerance
handling) is tracked in [#1208] — kept out of this PR so the sweep's OOM path is
not perturbed by the Mocker-facing capacity work.

[#1208]: https://github.com/ai-dynamo/aiconfigurator/issues/1208

## The deferred Dynamo wrapper (downstream PR, NOT in this repo)

Dynamo's `dynamo._internal.aic.estimate_num_gpu_blocks(...)` (today: Python
`backend._get_memory_usage` + a per-backend KV-budget formula) should be
rewritten as a thin wrapper that:

1. Calls the estimate — for the fully Rust-embedded Mocker path, the top-level
   Rust `aiconfigurator_core::estimate_kv_cache(req)` (one PyO3 hop into the
   Python source of truth); a pure-Python caller calls
   `aiconfigurator_core.sdk.memory.estimate_kv_cache(...)` directly. (There is no
   `aiconfigurator_core.estimate_kv_cache` `#[pyfunction]` — it was removed as a
   redundant surface.)
2. Applies `num_gpu_blocks_per_rank = floor(total_kv_size_tokens /
   scheduler_block_size)` locally — exactly what
   `aiconfigurator_core.sdk.memory.estimate_num_gpu_blocks` does in-repo.

This change lives in `ai-dynamo/dynamo`
(`lib/bindings/python/src/dynamo/_internal/aic.py` and/or
`lib/bindings/python/rust/llm/aic_callback.rs`) and is out of scope for the
`aiconfigurator` repo. The `estimate_num_gpu_blocks` signature on the Dynamo
side stays frozen (see the "AIC ↔ Dynamo Mocker handshake" section of
`phase-1.5-execution-plan.md`, surface **H3**); only its body is replaced.

**The in-repo surface is ready for it:** the Rust `estimate_kv_cache(req)`
forwarder (for the embedded Mocker) plus the AIC `estimate_num_gpu_blocks` helper
give the downstream PR both a callable entry point and a reference implementation
of the floor conversion to match.
