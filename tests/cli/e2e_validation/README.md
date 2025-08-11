<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# E2E Validation for CI/CD Pipelines

This directory provides comprehensive end-to-end validation testing for the Dynamo aiconfigurator, designed specifically for CI/CD integration. The testing framework supports three distinct validation modes through a **unified interface**.

## 🚀 **Unified Test Runner** - `e2e_runner.py`

**✨ Simplified interface for all E2E testing:**
```bash
python3 tests/cli/e2e_validation/e2e_runner.py --mode <smoke|selective|full> [options]
```

## 🎯 Three Testing Modes

### 1. 🚀 **Smoke Test** - Quick Validation (30s - 8m)
**Purpose**: Fast validation for pull requests and development branches  
**Usage**: `python3 e2e_runner.py --mode smoke --level basic`

- **Basic** (30-60s): Single configuration validation
- **Model** (2-3m): Multiple model compatibility check  
- **System** (2-3m): System comparison validation
- **Comprehensive** (5-8m): Key variation coverage

### 2. 🎯 **Selective Test** - Targeted Validation (Variable)
**Purpose**: Component-specific testing after changes  
**Usage**: `python3 e2e_runner.py --mode selective --models QWEN3_32B --systems h200_sxm`

- Model-specific testing
- System upgrade validation
- GPU scaling analysis
- Workload pattern testing
- Custom combination testing

### 3. 🔄 **Full Sweep** - Complete Validation (3-6 hours)
**Purpose**: Comprehensive validation for releases  
**Usage**: `python3 e2e_runner.py --mode full --parallel 4 --continue-on-error`

- ALL model combinations (~15 models)
- ALL system configurations (h100_sxm, h200_sxm)
- ALL GPU configurations (8, 512 GPUs)  
- ALL workload patterns (3 ISL/OSL combinations)
- ALL performance targets (2 TPOT values)
- **Total**: ~360 test combinations

## 🔧 CI/CD Integration

### Quick Start Commands

**💡 The unified script can be run from any directory - it automatically detects and switches to the project root.**

```bash
# Pull Request Validation (Fast)
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level basic --quiet

# Pre-deployment Check (Moderate)  
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level comprehensive --quiet

# Component Change Validation (Targeted)
python3 tests/cli/e2e_validation/e2e_runner.py --mode selective --models QWEN3_32B LLAMA3_8B --quiet

# Release Validation (Comprehensive)
python3 tests/cli/e2e_validation/e2e_runner.py --mode full --parallel 4 --continue-on-error --quiet
```

### CI/CD Pipeline Examples

#### GitHub Actions
```yaml
name: E2E Validation

on: [pull_request, push]

jobs:
  smoke-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Smoke Test
        run: python3 tests/cli/e2e_validation/smoke_test.py --level basic --quiet
        
  selective-test:
    runs-on: ubuntu-latest
    if: contains(github.event.head_commit.message, '[test-models]')
    steps:
      - uses: actions/checkout@v3
      - name: Model-specific Test
        run: python3 tests/cli/e2e_validation/selective_test.py --preset model-focus --quiet
```

#### Jenkins Pipeline
```groovy
pipeline {
    agent any
    stages {
        stage('PR Validation') {
            when { changeRequest() }
            steps {
                sh 'python3 tests/cli/e2e_validation/smoke_test.py --level basic --quiet'
            }
        }
        stage('Release Validation') {
            when { branch 'main' }
            steps {
                sh 'python3 tests/cli/e2e_validation/full_sweep.py --force --quiet --max-failures 20'
            }
        }
    }
}
```

## 📊 Test Coverage & Scope

The testing framework systematically validates:

- **15+ Models**: QWEN, LLAMA, DEEPSEEK, MOE variants
- **2 Systems**: h100_sxm, h200_sxm
- **2 GPU Configs**: 8 GPUs, 512 GPUs
- **3 Workload Patterns**: (4000,1000), (1000,2), (32,1000) ISL/OSL
- **2 Performance Targets**: 10ms, 100ms TPOT
- **1 Framework Version**: 0.20.0 (extensible)

## 🚀 Getting Started

### 1. Infrastructure Check
```bash
# Verify test infrastructure is ready
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --check-infrastructure
```

### 2. Choose Your Testing Mode

```bash
# Fast validation (recommended for PR checks)
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level basic

# Targeted testing (after specific changes)  
python3 tests/cli/e2e_validation/e2e_runner.py --mode selective --models QWEN3_32B --systems h200_sxm

# Complete validation (for releases)
python3 tests/cli/e2e_validation/e2e_runner.py --mode full --estimate-only  # Check estimated time first
python3 tests/cli/e2e_validation/e2e_runner.py --mode full --parallel 4 --continue-on-error
```
  
  ## 🔧 Advanced Usage
  
  ### Smoke Test Options
```bash
# Different smoke test levels
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level basic          # Single config (30-60s)
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level model          # Multiple models (2-3m)
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level system         # System comparison (2-3m)
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level comprehensive  # Key variations (5-8m)

# Show warnings and parallel execution
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level basic --show-warnings
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --level comprehensive --parallel 2
```

### Selective Test Options
```bash
# Model-specific testing
python3 tests/cli/e2e_validation/e2e_runner.py --mode selective --models QWEN3_32B LLAMA3_8B DEEPSEEK_V3_8B

# System comparison testing
python3 tests/cli/e2e_validation/e2e_runner.py --mode selective --systems h100_sxm h200_sxm --models QWEN3_32B

# GPU scaling testing
python3 tests/cli/e2e_validation/e2e_runner.py --mode selective --gpu-configs 8 512 --models QWEN3_32B

# Workload pattern testing
python3 tests/cli/e2e_validation/e2e_runner.py --mode selective --isl-osl 4000,1000 1000,2 --models QWEN3_32B

# Complex combinations with parallel execution
python3 tests/cli/e2e_validation/e2e_runner.py --mode selective --models QWEN3_32B LLAMA3_8B --systems h200_sxm --parallel 2
```

### Full Sweep Options
```bash
# Quick estimation
python3 tests/cli/e2e_validation/e2e_runner.py --mode full --estimate-only

# Parallel execution (requires pytest-xdist)
python3 tests/cli/e2e_validation/e2e_runner.py --mode full --parallel 4

# Continue on errors with custom failure limit
python3 tests/cli/e2e_validation/e2e_runner.py --mode full --continue-on-error --maxfail 20

# Comprehensive validation with all options
python3 tests/cli/e2e_validation/e2e_runner.py --mode full --parallel 4 --continue-on-error --show-warnings
```

## 📁 File Structure

```
tests/cli/e2e_validation/
├── common_utils.py         # 🔧 Shared utilities and infrastructure
├── smoke_test.py          # ⚡ Quick validation (30s-8m)
├── selective_test.py      # 🎯 Targeted testing (variable)
├── full_sweep.py          # 🔄 Complete validation (3-6h)
├── test_e2e_sweep.py      # 🧪 Core test implementation
└── README.md              # 📚 This documentation
```

## 📁 File Structure

```
tests/cli/e2e_validation/
├── e2e_runner.py     # 🚀 Main CLI entry point (run this!)
├── test_e2e_sweep.py    # 🧪 Pytest test implementation 
├── common_utils.py      # 🔧 Shared utilities
├── README.md            # 📚 This documentation
└── __init__.py         # 📦 Python package
```

### 📋 File Roles
- **`e2e_runner.py`** - CLI wrapper that parses arguments and calls pytest
- **`test_e2e_sweep.py`** - Contains the actual 553 test combinations
- **`common_utils.py`** - Shared functions for infrastructure and utilities

## 🔍 Result Analysis

Test results are stored in temporary directories during execution and include:

- **Error logs**: Detailed failure analysis with stack traces
- **Test reports**: Human-readable markdown summaries
- **Individual results**: Per-combination test outcomes
- **Recommendations**: Automated suggestions for issue resolution

## 🛠️ Troubleshooting

### Common Issues
- **Infrastructure not ready**: Run `--check-infrastructure` first
- **Long execution times**: Use smoke or selective tests for faster feedback
- **Memory issues**: Reduce parallel workers or test scope
- **Database unavailable**: Ensure systems database is properly configured

### Debug Mode
```bash
# Enable verbose output and show warnings
python3 tests/cli/e2e_validation/e2e_runner.py --mode selective --models QWEN3_32B --show-warnings

# Check what tests would run without executing
python3 tests/cli/e2e_validation/e2e_runner.py --mode full --estimate-only

# Check infrastructure status
python3 tests/cli/e2e_validation/e2e_runner.py --mode smoke --check-infrastructure
```

