<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## CLI E2E tests

These tests run the installed `aiconfigurator` CLI via `subprocess`.

### Run

```bash
python3 -m pytest tests/e2e/cli
```

### Useful subsets

```bash
# GitHub build workflow subset (fast/stable; also a good quick sanity subset)
python3 -m pytest tests/e2e/cli -m build

# Compatibility matrix (sweep)
python3 -m pytest tests/e2e/cli -m sweep
```

### Real Spica Thorough Sweep

`test_cli_default_thorough_sweep_real.py` is skipped by default because it requires
Spica's optional thorough-sweep dependencies and Dynamo replay bindings. It runs two
bounded real sweeps:

- A small `default --thorough-sweep` case derived from ordinary CLI inputs.
- A one-candidate native `--thorough-config` case that carries a throughput objective,
  KV router, load-enabled planner policy, and KVBM host offload through replay into the
  generated Dynamo artifacts. The test parses the emitted frontend and planner args
  with the pinned Dynamo runtime schemas.

Both cases exercise the real thorough sweeper, final report, and saved deployment
artifacts. The native fixture pins its parallel config and all searched dimensions so
CI evaluates exactly one candidate.

```bash
AIC_RUN_SPICA_THOROUGH_E2E=true \
PYTHONPATH=/path/to/dynamo/components/src:/path/to/dynamo/lib/bindings/python/src \
python3 -m pytest tests/e2e/cli/test_cli_default_thorough_sweep_real.py -q
```

Set `AIC_SPICA_THOROUGH_E2E_ARTIFACT_DIR` to retain the command/stdout/stderr logs
and complete generated `save_dir` and `native_save_dir` trees. The dedicated GitHub
Actions job sets this automatically and uploads the directory as the
`spica-thorough-e2e-results` artifact.
