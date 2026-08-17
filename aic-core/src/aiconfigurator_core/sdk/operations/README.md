# Operations package

This package holds every op class the SDK uses to DESCRIBE a model's work
— construction, weight sizing, and raw perf-data loading. Since #1357 PR-5
it deliberately does NOT hold performance math: per-op performance VALUES
(latency, energy, SOL decomposition) are computed only by the compiled Rust
engine (`aic-core/rust/aiconfigurator-core`). That invariant is policy
(`.claude/rules/rust-core/parity.md`, Rule 2) and is enforced by an
executable contract (`tests/cross_package/test_single_oracle_contract.py`:
frozen surfaces, banned def-name shapes, a frozen per-file def inventory,
and whitelists).

## What an op module owns

1. **A class** (subclass of `Operation`) — the typed parameter bag model
   builders construct (`GEMM(name, scale, n, k, quant_mode)`), plus
   `get_weights()` (the memory model) and `supported_quant_modes()`.
2. **A loader** (`load_<name>_data`) that parses parquet into a nested-dict
   perf table, plus the class-level cache and `load_data` classmethod. The
   loaded tables are the RAW collected data plane — consumed by
   enumeration (notebook charts, `create_charts`), the support matrix, and
   tests. No load-time SOL clamp or grid pre-expansion: the engine clamps
   and interpolates its own load.
3. **Optionally, a shim shape** for the deprecation window (see below).

Performance evaluation happens through the engine surfaces:

- **op-list FFI** (the canonical per-op path):
  `EngineHandle.evaluate_ops_json` / `evaluate_ops_sol_json` over
  `build_ops_json([...])`, or `evaluate_context_ops` /
  `evaluate_generation_ops` over a compiled spec;
- **per-phase**: `run_static_per_op` and friends;
- **whole runs**: `run_static` / `InferenceSession`.

`PerfDatabase.query_*` and `Operation.query()` still exist as one-release
DEPRECATION SHIMS routed through `sdk/engine.py::_evaluate_single_op` —
this surface is FROZEN (new ops do not add to it) and is removed by the
deprecation-cleanup PR.

## Adding a new op — the single-oracle flow

1. **Model the op in Rust**: an operator in
   `aic-core/rust/aiconfigurator-core/src/operators/` (query, SOL, energy)
   and, if table-backed, a loader in
   `aic-core/rust/aiconfigurator-core/src/perf_database/`. Anchor it with a
   Rust `#[cfg(test)]` oracle test (hand-derived — there is no Python
   reference to generate against).
2. **Add the Python op class here**: constructor + fields (the wire
   parameters), `get_weights`, and — if table-backed — the parquet loader,
   class cache, and `load_data` following the conventions below. NO
   `query()` body, NO `_query_*_table`, NO `get_sol`/`get_empirical`
   closures: the contract test rejects those def shapes, and every def you
   add must be declared in its frozen per-file inventory
   (`OPERATIONS_DEF_INVENTORY`) — that edit is the deliberate, reviewable
   declaration of the new function.
3. **Wire the spec conversion**: a `_to_opspec` branch in
   `aic-core/src/aiconfigurator_core/sdk/engine.py` and an `Op` variant in
   `aic-core/rust/aiconfigurator-core/src/operators/op.rs` — **append at the enum tail**
   (bincode variant indices are positional; a mid-enum insertion requires
   an `ENGINE_SPEC_SCHEMA_VERSION` bump on BOTH sides) — plus the
   `aic-core/rust/aiconfigurator-core/src/engine/spec.rs` round-trip fixture.
   `tests/unit/sdk/test_opspec_coverage.py` fails until the op converts or
   carries a justified `EXEMPT` entry.
4. **Pin the behavior**: a parity case via
   `aic-core/rust/aiconfigurator-core/parity_tests/pin_goldens.py`
   (append-only) once a config class reaches the op; golden diffs carry the
   review for any later modeling change.
5. **Deprecation-window shims are NOT extended**: do not add a
   `PerfDatabase.query_*` method or expect `Operation.query()` to serve a
   new op. If the new op must be reachable through the *existing* window
   surface (rare), give the class an `_ENGINE_QUERY_SHAPE`
   (`"tokens"` / `"context"` / `"generation"` / `"module"`), and an
   override-aware `_engine_query_plan` only when legacy kwargs need
   mapping. `query()` bodies are limited to the orchestration whitelist
   (the AFD comm ops, the deprecated `Mamba2` composite) — bodies that
   compose ENGINE-evaluated twin ops, never math.
6. **Data collection** is unchanged: define the collector op, register the
   family in `collector/op_backend_catalog.yaml`, and ship parquet under
   `systems/data/<system>/<family>/<backend>/<version>/` (Rust reads
   parquet only).

## Loader conventions (unchanged)

### Idempotency

`load_data` must be safe to call repeatedly. The cache-key check at the
top is the contract: same `(systems_root, system, backend, version,
enable_shared_layer)` → same result, regardless of how many times it
fires.

### Instance attribute bind gating

Use `if "_my_op_data" not in database.__dict__:` rather than
`hasattr(database, "_my_op_data")` for the post-load bind. Tests
sometimes pre-set the attribute to inject custom data; `hasattr`
would silently overwrite the override. (Note: injected synthetic tables
are visible to ENUMERATION consumers only — the engine loads its own
tables from disk, and the deprecated query shims raise `TypeError` for
non-`PerfDatabase` databases for exactly that reason.)

### Cache atomicity for ops with multiple slots

If your op owns N caches (GEMM has 3 — gemm / compute_scale /
scale_matrix; MoE has 4), load all of them into local variables FIRST,
then commit them all to the class cache as the last step. A loader
exception mid-sequence would otherwise leave the class cache with the
first slot populated but the others empty, and the next `load_data`
cache-hit would skip the load and crash downstream.

### Loader return contract

Return `None` when the file is missing (the loader stack treats `None` as
"no silicon data for this op"); return an empty dict for an existing file
with no rows. Store per-shape leaves as
`{"latency": float, "power": float, "energy": float}`.

### Missing-file semantics

`resolve_op_data_path` returns the legacy-shaped path even when nothing
exists, so callers keep their missing-file semantics; family-first layout
is discovered structurally (any first-level dir under the system data root
that is not a known backend dir).

## Where the retired pieces went

| Retired (PR-5) | Its oracle now |
| --- | --- |
| `_query_*_table` classmethods, `get_sol`/`get_empirical` closures | `operators/*.rs` |
| `sdk/perf_interp/` | `perf_database/*.rs` + `operators/util_empirical.rs` |
| `util_empirical` estimation math | `operators/util_empirical.rs` (the provenance pipeline and `quant_profile` remain here) |
| load-time SOL clamps | the engine's own load (`perf_database/gemm.rs` etc.) |
| per-call facade math | engine-routed shims (window-only) over `_evaluate_single_op` |
