<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PROPOSAL: single-oracle update for `.claude/rules/rust-core/parity.md`

Rule files are human-owned policy (repo-guide rule 2), so this PR does not
edit `.claude/rules/rust-core/parity.md`; it proposes the update below.
Context: the per-call Python query stack is now DELETED (#1357 PR-5) — the
condition the previous proposal (`parity-rule-golden-diff-proposal.md`)
attached to the dual-implementation clause ("survives until the #1357
Phase-3 sequel deletes it family-by-family") has fired. The clause is now
wrong in the dangerous direction: it instructs contributors to ADD Python
query math next to Rust changes, which would re-grow the very path this
migration removed.

## What changed in the codebase

- `operations/*.py` `_query_*_table` bodies, their `get_sol`/`get_empirical`
  closures, `sdk/perf_interp/`, and the `util_empirical` grid/transfer math
  are gone. The compiled engine is the ONLY per-op estimator.
- `PerfDatabase.query_*` and `Operation.query` survive one release as thin
  engine-routed deprecation shims (removed in PR-6); the AFD comm ops and
  the deprecated `Mamba2` composite keep Python ORCHESTRATION bodies whose
  per-message values come from engine-evaluated twin ops.
- Enforcement now exists in CI:
  `tests/cross_package/test_single_oracle_contract.py` freezes the shim
  surface, bans `_query_*_table`/`get_sol`/`get_empirical` math in
  `operations/`, whitelists the orchestration `query` overrides, and asserts
  `perf_interp` stays gone. `.coderabbit.yaml` carries matching path
  instructions for machine review.

## Proposed rule edits (maintainers to apply)

1. Replace the dual-implementation clause ("MUST in the same PR mirror the
   change in Python/Rust") with a **single-oracle invariant**:

   > Per-op performance values (latency, energy, SOL decomposition) are
   > computed ONLY in `aic-core/rust/aiconfigurator-core`. A change to
   > estimator behavior lands in the Rust operator/table layer with its
   > golden-diff evidence (`pin_goldens.py`); there is no Python side to
   > mirror. Adding per-op query/interpolation/roofline math anywhere under
   > `aic-core/src/aiconfigurator_core/sdk/` is a violation — extend the
   > engine FFI instead. The deliberate-edit gate is
   > `tests/cross_package/test_single_oracle_contract.py`: a PR that must
   > grow the whitelists there needs an explicit justification in its
   > description.

2. Drop the stale "known intentional splits" entries that this migration
   resolved (AFD per-op values now cross the op-list FFI; SOL/SOL_FULL are
   engine-served via `evaluate_ops_sol_json`).

3. Refresh `paths:` to the real locations
   (`aic-core/rust/aiconfigurator-core/**`,
   `aic-core/src/aiconfigurator_core/sdk/**`) — same fix the golden-diff
   proposal requested.

4. Keep unchanged: the append-only `Op` enum rule, the
   `ENGINE_SPEC_SCHEMA_VERSION` two-sided bump rule, selection-rule parity
   guidance for FUTURE Rust-side table work, and the golden-diff workflow.
