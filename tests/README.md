<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Test suite usage

This repo uses **pytest** and organizes tests into two primary suites:

- **Unit**: fast, deterministic, no external services (`tests/unit/`)
- **E2E**: end-to-end validations, including CLI subprocess tests (`tests/e2e/`)

The test runner is **just pytest** (no custom wrapper scripts).

### Setup

The E2E CLI tests execute the installed `aiconfigurator` console script, so make sure you have an editable install:

```bash
python3 -m pip install -e ".[dev]"
```

### Quick start

```bash
# Run everything (includes sweep unless you filter by markers)
python3 -m pytest
```

### Recommended suites

```bash
# PR / local fast checks (unit only)
python3 -m pytest -m unit

# GitHub build workflow subset: unit + a small stable E2E subset
python3 -m pytest -m "unit or build"

# Full validation: all E2E tests + unit tests
TEST_SUPPORT_MATRIX=true python3 -m pytest -m "unit or e2e"
```

### Key markers

Markers are defined in `pytest.ini`:

- **unit**: fast tests (includes lightweight integration-style tests)
- **e2e**: end-to-end tests
- **build**: E2E subset intended for GitHub build workflows (fast & stable)
- **sweep**: large compatibility matrices (typically slow)

Examples:

```bash
# Only the fast/stable CI subset (quick sanity)
python3 -m pytest -m build

# Only the large matrix tests (slow)
python3 -m pytest -m sweep

# E2E tests, excluding the sweep
python3 -m pytest -m "e2e and not sweep"

# For github workflow
python3 -m pytest -m "unit or build"
```

> Note: if encoutering error `NotImplementedError: Implement enable_gui in a subclass`, please add `MPLBACKEND=agg` to your command

### Rust Engine Step Opt-In

The Python SDK keeps using the existing Python latency path by default. To run Python tests through the experimental Rust engine-step path, build the Rust core shared library first or let pytest build it on demand:

```bash
# Build manually, then run any pytest target with the Rust path enabled
cargo build --manifest-path rust/aiconfigurator-core/Cargo.toml
python3 -m pytest tests/unit/sdk/test_rust_engine_step.py --aic-engine-step-backend=rust

# Or let the test harness build the shared library when needed
python3 -m pytest tests/unit/sdk/test_rust_engine_step.py \
  --aic-engine-step-backend=rust \
  --aic-rust-core-autobuild
```

The support-matrix PR regression now runs Python first and compares Rust engine-step output against it.
Rust core autobuild is enabled by the support-matrix runner for the Rust pass, so the usual command is:

```bash
TEST_SUPPORT_MATRIX=true python3 -m pytest tests/e2e/support_matrix
```

For the full support-matrix generation runner, enable the same parity check explicitly:

```bash
python3 tools/support_matrix/generate_support_matrix.py --compare-engine-step-backends
```

### Where tests live

- **Unit**
  - `tests/unit/cli/`: CLI parser/workflow unit tests
  - `tests/unit/sdk/`: SDK unit tests (database queries, task config, utilities, etc.)
- **E2E**
  - `tests/e2e/cli/`: CLI E2E tests (subprocess; runs `aiconfigurator cli ...`)
  - `tests/e2e/support_matrix/`: support-matrix validation (gated by `TEST_SUPPORT_MATRIX=true`)

### Parallel execution

If `pytest-xdist` is installed:

```bash
python3 -m pytest -n 4 -m "unit or build"
```
