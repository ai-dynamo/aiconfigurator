# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI module for aiconfigurator.

Provides both command-line interface and programmatic Python API.

Python API usage:
    from aiconfigurator.cli import cli_default, cli_exp, cli_generate, cli_support

    # cli_default: Run agg vs disagg comparison
    result = cli_default(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
    )

    # cli_exp: Run experiments from YAML or dict config
    result = cli_exp(yaml_path="experiments.yaml")
    # or
    result = cli_exp(config={"exp1": {...}, "exp2": {...}})

    # cli_generate: Generate naive config without sweeping
    result = cli_generate(
        model_path="Qwen/Qwen3-32B",
        total_gpus=8,
        system="h200_sxm",
        backend="trtllm",
        output_dir="./output",
    )

    # cli_support: Check model/hardware support
    agg_ok, disagg_ok = cli_support(
        model_path="Qwen/Qwen3-32B",
        system="h200_sxm",
    )
"""

from aiconfigurator.cli.api import (
    CLIResult,
    cli_default,
    cli_exp,
    cli_generate,
    cli_support,
)

__all__ = [
    "CLIResult",
    "cli_default",
    "cli_exp",
    "cli_generate",
    "cli_support",
]
