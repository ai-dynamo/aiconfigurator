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

- [overview.md](docs/overview.md) — what Spica is + the end-to-end sweep flow
  (validate → filter policies → load-predictor sub-sweep → enumerate branches →
  per-branch Vizier study → merge by goal).
- [optimization-goal.md](docs/optimization-goal.md) — the `OptimizationGoal` targets,
  the per-GPU metric, the SLA rule, and how **`pareto`** (multi-objective) works.
- [traffic.md](docs/traffic.md) — the `Workload` load shapes (trace / request-rate /
  concurrency), `num_request_ratio`, and the pareto concurrency sweep.
- [search-space.md](docs/search-space.md) — every knob (type, default, searched/pinned,
  choices), the composite presets, and how `parallel_configs` are derived.
- [sample.md](docs/sample.md) — the flat *unrolled sample* and the three ways to
  pin/override what it emits.

It is an **independent project** living inside the aiconfigurator repo. It
depends on `aiconfigurator` (lower-layer forward-pass / memory provider) and,
later, on `ai-dynamo` (Replay). The `aiconfigurator` package never imports
Spica — Spica is the upper layer.

## Status

- Input schema (`SmartSearchConfig`) — done. See [docs/search-space.md](docs/search-space.md)
  for the full knob reference (what you can pin/search, composite-knob presets vs.
  raw-dict pins, and `parallel_configs`).
- Planner load-predictor independent grid sweep (`sweep_load_predictor`) —
  done; reuses the real dynamo planner predictors + the planner's densify-fixed
  trace→window tool.
- `run_smart_search` (the main Vizier + Replay sweep) — done: enumerate → sample →
  deploy → replay → score → rank. Real replay needs the `aic-forward-pass` build below.

## Develop

```bash
cd spica
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest          # pure tests; dynamo integration tests skip without [dynamo]
```

The load-predictor sweep reuses the dynamo planner predictors (Rust runtime +
prophet/pmdarima/filterpy), pinned to a dynamo commit. Installing that extra
needs `GIT_LFS_SKIP_SMUDGE=1` (the dynamo repo carries LFS media irrelevant to
the build) and a Rust toolchain (`ai-dynamo-runtime` builds from source until
1.3.0 ships a wheel). The `[search]` extra adds the Vizier optimizer that drives
the main `run_smart_search` sweep:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -e ".[dev,dynamo,search]"
```

`[search]` installs **CPU** jax — it resolves on every platform, but the Vizier
multi-objective GP suggest is slow on CPU (and can stall on larger sweeps). On a
**linux-x86_64 box with an NVIDIA GPU**, swap `search` → `search-gpu` to run the
optimizer on CUDA (XLA), which removes that bottleneck:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -e ".[dev,dynamo,search-gpu]"
```

`search-gpu` pulls the jax CUDA wheels, which exist **only for linux-x86_64**
(not macOS / Windows / aarch64 — those must use `[search]`). With no GPU present
jax just warns and falls back to CPU, so there's no reason to use it without
one — stay on `[search]`.

### Real replay (`aic-forward-pass`)

The `[dynamo]` extra above is enough for the load-predictor sweep, but the
replay-backed evaluator (`spica.evaluator.ReplayEvaluator`, what the real sweep
uses) drives the dynamo mocker's **AIC perf model**. Those bindings are gated on
the optional `aic-forward-pass` Cargo feature, which the prebuilt/default install
does **not** enable — without it the planner bridge raises *"AIC perf model
requires the `aic-forward-pass` feature"*. Rebuild the bindings from a dynamo
source checkout (matching the pinned commit) into this venv:

```bash
cd <dynamo>/lib/bindings/python
VIRTUAL_ENV=<spica>/.venv <spica>/.venv/bin/maturin develop --uv --release --features aic-forward-pass
```

Probe whether the running install has it: `RustEnginePerfModel` is importable
from `dynamo._core` only when the feature is compiled in. The real-replay
integration tests skip when it is absent.

Validate an example config:

```bash
.venv/bin/python -c "from spica import SmartSearchConfig; print(SmartSearchConfig.from_yaml('examples/smart_sweep.yaml'))"
```
