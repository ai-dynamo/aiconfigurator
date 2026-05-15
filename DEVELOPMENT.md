<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Developer Guide

This guide will help you get started with developing `aiconfigurator`. We welcome contributions from the community!

## Initial Setup

### 1. Install Git LFS

Git LFS is required to handle large database files in the repository.

```bash
apt-get install git-lfs
```

### 2. Clone the Repository

```bash
git clone https://github.com/ai-dynamo/aiconfigurator
cd aiconfigurator

# Pull LFS files
git lfs pull
```

### 3. Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 4. Install Development Dependencies

```bash
# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

This also compiles the Rust core (`rust/aiconfigurator-core`) via `maturin` as a
PyO3 extension module and drops the resulting shared library into
`src/aiconfigurator/_native/aiconfigurator_core.abi3.so` (or `.pyd` on Windows),
which `aiconfigurator.sdk.rust_engine_step` imports directly. A Rust toolchain
(`cargo`/`rustc`) on `PATH` is required for this step — install via
[rustup](https://rustup.rs/) if you don't have one.

For fast Rust-side iteration without a full reinstall, use `maturin develop`:

```bash
pip install maturin
maturin develop --release --manifest-path rust/aiconfigurator-core/Cargo.toml
```

### 5. Install Pre-Commit Hooks

```bash
pre-commit install
```

This installs:
- The `aiconfigurator` package in editable mode
- All runtime dependencies
- Development tools: `ruff`, `pre-commit`, `pytest` and related plugins

### Optional: Install Ruff Extension

If you are using VS Code or one of its forks (e.g. Cursor), you can install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) which will highlight linting issues in your editor. You can also configure your editor to auto-apply formatting when saving files using the instructions [here](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff#:~:text=Taken%20together%2C%20you%20can%20configure%20Ruff%20to%20format%2C%20fix%2C%20and%20organize%20imports%20on%2Dsave%20via%20the%20following%20settings.json%3A).

## Development Workflow

### Code Style and Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

#### Run Linting

```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check --fix .
```

#### Run Formatting

```bash
# Check formatting
ruff format --check .

# Apply formatting
ruff format .
```

### Pre-commit Hooks

Pre-commit hooks automatically run checks before each commit.

#### Run Pre-commit Manually

```bash
pre-commit run --all-files
```

### Building the Rust Core

The Rust core is compiled into the wheel by [maturin](https://www.maturin.rs/)
as a PyO3 extension module whenever you run `pip install -e .`, `pip install .`,
or `maturin build --release`. The resulting shared library lives at
`src/aiconfigurator/_native/aiconfigurator_core.abi3.so` and is imported
directly by `aiconfigurator.sdk.rust_engine_step` — no `ctypes`, no
filename-glob loader.

Because the crate is built with PyO3's `abi3-py310` stable-ABI feature, one
compiled artifact works on every CPython ≥ 3.10. The wheel is correspondingly
tagged `cp310-abi3-manylinux_2_28_<arch>`.

For fast iteration on the Rust side without reinstalling, use:

```bash
maturin develop --release --manifest-path rust/aiconfigurator-core/Cargo.toml
```

Enable the Rust path per-CLI-run with:

```bash
aiconfigurator cli ... --engine-step-backend rust
```

#### SBOM (CycloneDX)

CI-built wheels (`docker build --target wheel-out`) ship a CycloneDX JSON SBOM
for the Rust crate's dependency graph at
`<wheel>.dist-info/sboms/aiconfigurator-core-py3.cyclonedx.json` per PEP 770.
The SBOM is emitted by maturin's built-in `cargo-cyclonedx` integration — no
custom build steps. Local dev installs (`pip install -e .`, `maturin develop`)
do not emit the SBOM by default; only `maturin build` does.

To produce an SBOM-bearing wheel locally:

```bash
pip install maturin
cargo install --locked cargo-cyclonedx --version 0.5.7   # required on PATH for maturin's SBOM path
maturin build --release --manifest-path rust/aiconfigurator-core/Cargo.toml
```

To inspect the SBOM in a built wheel:

```bash
unzip -p dist/aiconfigurator-*.whl 'aiconfigurator-*.dist-info/sboms/*.cyclonedx.json' | jq .
```

### Running Tests

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing.

```bash
# Run all tests
pytest tests

# Run tests for a specific component
pytest tests/unit
pytest tests/e2e

# GitHub PR / build subset (unit + a small stable E2E subset)
pytest -m "unit or build"
```

## Data Collection (Advanced)

Data collection is typically not required for development. The repository includes pre-collected performance databases for supported systems.

If you need to collect new data for a new GPU type or framework version, refer to the [Collector README](collector/README.md).

## Contributing

Before contributing, please read:
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines and rules
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community standards

## Common Development Tasks

### Adding a New Model

Refer to [How to Add a New Model](docs/add_a_new_model.md).

### Running Automation Scripts

Refer to the [Automation README](tools/automation/README.md).

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Open an issue on [GitHub](https://github.com/ai-dynamo/aiconfigurator/issues)
- **Examples**: Explore `tools/simple_sdk_demo/` for SDK usage examples

## License

This project is licensed under Apache 2.0. All contributions must include SPDX license headers and DCO sign-off.

