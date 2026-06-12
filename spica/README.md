<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Spica

Spica is the Replay-backed **smart sweeper** for Dynamo deployments (Profiler V2).
It searches engine / router / planner configuration with a black-box optimizer,
evaluates each candidate with Dynamo Replay, and returns a ranked candidate set.

Design proposal: `docs/proposals/dgdr-profiler-smart-search-plan.md` in
[ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo).

It is an **independent project** living inside the aiconfigurator repo. It
depends on `aiconfigurator` (lower-layer forward-pass / memory provider) and,
later, on `ai-dynamo` (Replay). The `aiconfigurator` package never imports
Spica — Spica is the upper layer.

## Status

- Input schema (`SmartSearchConfig`) — done.
- Planner load-predictor independent grid sweep (`sweep_load_predictor`) —
  done; reuses the real dynamo planner predictors + the planner's densify-fixed
  trace→window tool.
- `run_smart_search` (the main Vizier + Replay sweep) — still a stub; lands next.

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
1.3.0 ships a wheel):

```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -e ".[dev,dynamo]"
```

Validate an example config:

```bash
.venv/bin/python -c "from spica import SmartSearchConfig; print(SmartSearchConfig.from_yaml('examples/smart_sweep.yaml'))"
```
