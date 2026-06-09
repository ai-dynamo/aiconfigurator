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
# Install the package in editable mode with dev dependencies.
# This installs only the pure-Python wheel — no Rust toolchain required.
pip install -e ".[dev]"
```

The bare `aiconfigurator` package is **pure Python**: no Docker, no Rust
compile, installs anywhere pip works. If you want the Rust-accelerated forward-pass estimator (opt-in via `--engine-step-backend rust`), also install
the companion `aiconfigurator-rust-core` extension:

```bash
# Install both the pure package and the precompiled Rust extension (if a
# precompiled wheel for your platform is available).
pip install -e ".[dev,rust]"
```

For Rust-side iteration without reinstalling, work directly in the
`rust/aiconfigurator-core/` subproject:

```bash
pip install maturin
cd rust/aiconfigurator-core
maturin develop --release   # builds aiconfigurator_rust_core from this checkout
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

### Building the two wheels

`aiconfigurator` ships as **two separate PyPI distributions**:

| Distribution | Wheel tag | Contents | Build tool |
|---|---|---|---|
| `aiconfigurator` | `py3-none-any` | Pure Python SDK + CLI + bundled data files (model_configs, systems, generator templates) | setuptools |
| `aiconfigurator-rust-core` | `cp310-abi3-<platform>` (matrix below) | PyO3 extension module `aiconfigurator_rust_core.aiconfigurator_core` | maturin |

The pure-Python `aiconfigurator` works without the rust extension. The
SDK's `--engine-step-backend rust` flag is gated by `is_rust_core_available()`
which falls back to the Python latency path on platforms where the
extension isn't installed. Version coordination: lock-step minor versions
(`aiconfigurator 0.9.x` requires `aiconfigurator-rust-core>=0.9.0,<0.10.0`).

#### Supported platforms

CI publishes a precompiled `aiconfigurator-rust-core` wheel for each of the
following platforms; `pip install aiconfigurator[rust]` picks the right one
automatically:

| Platform | Wheel tag |
|---|---|
| Linux x86_64 (glibc ≥ 2.28; RHEL 8+, Ubuntu 20.04+) | `cp310-abi3-manylinux_2_28_x86_64` |
| Linux aarch64 (glibc ≥ 2.28) | `cp310-abi3-manylinux_2_28_aarch64` |
| Linux x86_64 (glibc ≥ 2.17; RHEL 7-era, Amazon Linux 2) | `cp310-abi3-manylinux2014_x86_64` |
| Linux aarch64 (glibc ≥ 2.17) | `cp310-abi3-manylinux2014_aarch64` |
| Linux x86_64 (musl; Alpine / distroless) | `cp310-abi3-musllinux_1_2_x86_64` |
| Linux aarch64 (musl) | `cp310-abi3-musllinux_1_2_aarch64` |
| macOS arm64 (Apple Silicon) | `cp310-abi3-macosx_14_0_arm64` |
| macOS x86_64 (Intel) | `cp310-abi3-macosx_10_13_x86_64` |
| Windows x86_64 | `cp310-abi3-win_amd64` |

Hosts outside this matrix (e.g. Windows ARM64, FreeBSD, PyPy) fall through
to the source distribution and need a local Rust toolchain to build. Bare
`pip install aiconfigurator` always works without the extension.

#### Local build commands

Single multi-stage `docker/Dockerfile` exposes one buildx target per
artifact, plus a `combined-test` target that exercises both backends
end-to-end. Same commands work locally and in CI.

```bash
# Pure-Python wheel (fast, no Docker actually needed):
uv build --wheel
# OR via Dockerfile for reproducibility:
docker buildx build --target wheel-out-pure --output type=local,dest=./dist \
  -f docker/Dockerfile .

# Rust extension wheel (requires docker buildx + a manylinux/musllinux container):
docker buildx build \
  --platform linux/arm64 \
  --build-arg WHEEL_BUILD_BASE=quay.io/pypa/manylinux_2_28_aarch64 \
  --target wheel-out-rust --output type=local,dest=./dist \
  -f docker/Dockerfile .

# Swap WHEEL_BUILD_BASE to target other Linux variants without other changes:
#   quay.io/pypa/manylinux2014_aarch64   (older glibc)
#   quay.io/pypa/musllinux_1_2_aarch64   (Alpine / distroless)
#   quay.io/pypa/manylinux_2_28_x86_64   (x86_64, default if you omit the arg)

# Unified end-to-end smoke (installs both wheels, runs CLI with rust then
# python backends, fails if either path is broken):
docker buildx build \
  --platform linux/arm64 \
  --build-arg WHEEL_BUILD_BASE=quay.io/pypa/manylinux_2_28_aarch64 \
  --target combined-test \
  -f docker/Dockerfile .
```

Enable the Rust path per-CLI-run with:

```bash
aiconfigurator cli ... --engine-step-backend rust
```

#### SBOM (CycloneDX)

The `aiconfigurator-rust-core` wheel ships a CycloneDX JSON SBOM for the
Rust crate's dependency graph at
`<wheel>.dist-info/sboms/aiconfigurator-core.cyclonedx.json` per PEP 770.
The SBOM is emitted by maturin's built-in `cargo-cyclonedx` integration —
no custom build steps. The pure-Python `aiconfigurator` wheel does not
ship an SBOM (no native dependencies to inventory).

To produce an SBOM-bearing rust-core wheel locally outside Docker:

```bash
pip install maturin
cargo install --locked cargo-cyclonedx --version 0.5.7   # required on PATH for maturin's SBOM path
cd rust/aiconfigurator-core
maturin build --release
```

To inspect the SBOM in a built wheel:

```bash
unzip -p dist/aiconfigurator_rust_core-*.whl \
  'aiconfigurator_rust_core-*.dist-info/sboms/*.cyclonedx.json' | jq .
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

