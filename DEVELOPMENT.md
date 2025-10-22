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

### 5. Install Pre-Commit Hooks

```bash
pre-commit install
```

This installs:
- The `aiconfigurator` package in editable mode
- All runtime dependencies
- Development tools: `ruff`, `pre-commit`, `pytest` and related plugins

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

### Running Tests

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing.

```bash
# Run all tests
pytest tests

# Run tests for a specific component
pytest tests/sdk
pytest tests/cli
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

