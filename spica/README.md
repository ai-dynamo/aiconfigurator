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

Milestone 1 — project skeleton + input schema only. `run_smart_search` is a
stub; the search loop and the injectable Replay evaluator land next.

## Develop

```bash
cd spica
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Validate an example config:

```bash
.venv/bin/python -c "from spica import SmartSearchConfig; print(SmartSearchConfig.from_yaml('examples/smart_sweep.yaml'))"
```
