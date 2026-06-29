<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Spica's optional thorough-sweep dependencies and Dynamo replay bindings. It runs a small static `default --thorough-sweep` case with
environment-limited Spica sweep settings so it finishes quickly while still
exercising the real thorough sweeper, final report, and saved deployment artifacts.

```bash
AIC_RUN_SPICA_THOROUGH_E2E=true \
PYTHONPATH=/path/to/dynamo/components/src:/path/to/dynamo/lib/bindings/python/src \
python3 -m pytest tests/e2e/cli/test_cli_default_thorough_sweep_real.py -q
```
