<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Spica

Spica is the Replay-backed **smart sweeper** for Dynamo deployments (Profiler V2).
It searches engine / router / planner configuration with a black-box optimizer,
evaluates each candidate with Dynamo Replay, and returns a ranked candidate set
(or a Pareto front under a `pareto` goal).

Design proposal: `docs/proposals/dgdr-profiler-smart-search-plan.md` in
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo).

## Docs

- [overview.md](overview.md) — what Spica is + the end-to-end sweep flow
  (validate → filter policies → load-predictor sub-sweep → enumerate branches →
  per-branch Vizier study → merge by goal).
- [optimization-goal.md](optimization-goal.md) — the `OptimizationGoal` targets,
  the per-GPU metric, the SLA rule, and how **`pareto`** (multi-objective) works.
- [traffic.md](traffic.md) — the `Workload` load shapes (trace / request-rate /
  concurrency), `num_request_ratio`, and the pareto concurrency sweep.
- [search-space.md](search-space.md) — every knob (type, default, searched/pinned,
  choices), the composite presets, and how `parallel_configs` are derived.
- [sample.md](sample.md) — the flat *unrolled sample* and the three ways to
  pin/override what it emits.

Spica's source lives in `src/spica` at the repository root and ships in the
`aiconfigurator` wheel. The root `pyproject.toml` owns its optional dependencies
and `spica` console entry point; tests live in `tests/spica`, tools in
`tools/spica`, and examples below this directory. Spica uses AIConfigurator's
lower-layer forward-pass / memory provider and `ai-dynamo` for Replay-backed
evaluation when the real sweep runs.

## Status

- Input schema (`SmartSearchConfig`) — done. See [search-space.md](search-space.md)
  for the full knob reference (what you can pin/search, composite-knob presets vs.
  raw-dict pins, and `parallel_configs`).
- Planner load-predictor independent grid sweep (`sweep_load_predictor`) —
  done; reuses the real dynamo planner predictors + the planner's densify-fixed
  trace→window tool.
- `run_smart_search` (the main Vizier + Replay sweep) — done: enumerate → sample →
  deploy → replay → score → rank. Real replay needs the `aic-forward-pass` build below.

## Develop

```bash
uv sync --extra dev --extra spica
uv run --extra dev --extra spica pytest tests/spica
```

The root `spica` extra installs the CPU Vizier/JAX stack used by
`run_smart_search`. The load-predictor sweep and replay evaluator also reuse
Dynamo planner/runtime dependencies pinned to commit
`cb7dc1a3a74018b7824ab7ef9d0191b80946758b`. CI installs those dependencies
from a matching Dynamo checkout; see the `spica-thorough-e2e` job in
`.github/workflows/build-test.yml` for the reproducible setup.

### Real replay (`aic-forward-pass`)

The replay-backed evaluator (`spica.evaluator.ReplayEvaluator`, what the real
sweep uses) drives the Dynamo mocker's **AIC perf model**. Those bindings are gated on
the optional `aic-forward-pass` Cargo feature, which the prebuilt/default install
does **not** enable — without it the planner bridge raises *"AIC perf model
requires the `aic-forward-pass` feature"*. Rebuild the bindings from a dynamo
source checkout (matching the pinned commit) into this venv:

```bash
AIC_REPO=/path/to/aiconfigurator
cd <dynamo>/lib/bindings/python
VIRTUAL_ENV="$AIC_REPO/.venv" "$AIC_REPO/.venv/bin/maturin" develop --uv --release --features aic-forward-pass
```

Probe whether the running install has it: `RustEnginePerfModel` is importable
from `dynamo._core` only when the feature is compiled in. The real-replay
integration tests skip when it is absent.

Validate an example config:

```bash
.venv/bin/python -c "from spica import SmartSearchConfig; print(SmartSearchConfig.from_yaml('docs/spica/examples/smart_sweep.yaml'))"
```
