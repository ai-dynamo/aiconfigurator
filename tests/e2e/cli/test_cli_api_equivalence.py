# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests to verify that CLI APIs (cli_default, cli_exp, cli_generate)
are equivalent to running CLI commands via command line.

These tests run real CLI commands via subprocess and compare results with API calls.
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

from aiconfigurator.cli import cli_default, cli_exp

pytestmark = pytest.mark.e2e


def _find_output_dir(save_dir: str) -> str:
    """Recursively find the directory containing experiment results (CSV files)."""
    for root, dirs, files in os.walk(save_dir):
        if "best_config_topn.csv" in files or "pareto.csv" in files:
            # This is an experiment directory (e.g., .../agg/),
            # we want its parent which contains all experiment folders
            return os.path.dirname(root)
        # Also check if it's the random-suffix directory itself containing exp folders
        if any(
            os.path.isdir(os.path.join(root, d))
            and any(
                f.endswith(".csv") for f in os.listdir(os.path.join(root, d)) if os.path.isdir(os.path.join(root, d))
            )
            for d in dirs
        ):
            return root

    raise FileNotFoundError(f"Could not find output directory in {save_dir}")


def _assert_dataframes_equal(api_df: pd.DataFrame, cli_df: pd.DataFrame, name: str) -> None:
    """Assert two DataFrames are equivalent (same data, allowing for floating point tolerance)."""
    if api_df.empty and cli_df.empty:
        return

    # Same number of rows
    assert len(api_df) == len(cli_df), f"{name}: Row count mismatch ({len(api_df)} vs {len(cli_df)})"

    # Same columns (order may differ)
    api_cols = set(api_df.columns)
    cli_cols = set(cli_df.columns)
    assert api_cols == cli_cols, (
        f"{name}: Column mismatch. API has {api_cols - cli_cols}, CLI has {cli_cols - api_cols}"
    )

    # Compare all columns with appropriate tolerance for numeric types
    for col in api_df.columns:
        api_series = api_df[col].reset_index(drop=True)
        cli_series = cli_df[col].reset_index(drop=True)

        if pd.api.types.is_numeric_dtype(api_series):
            # Numeric columns: use relative tolerance
            pd.testing.assert_series_equal(
                api_series,
                cli_series,
                check_names=False,
                check_dtype=False,
                rtol=1e-5,
                atol=1e-10,
            )
        else:
            # Non-numeric columns: exact match
            pd.testing.assert_series_equal(
                api_series,
                cli_series,
                check_names=False,
                check_dtype=False,
            )


class TestCLIDefaultEquivalence:
    """Tests that cli_default API produces same results as CLI command."""

    def test_cli_default_api_vs_command(self, tmp_path):
        """cli_default API should produce same results as running CLI command."""
        # Run via Python API
        api_result = cli_default(
            model_path="Qwen/Qwen3-32B",
            total_gpus=32,
            system="h200_sxm",
        )

        # Run via CLI command with save_dir to capture output
        save_dir = tmp_path / "cli_output"
        save_dir.mkdir()

        cmd = [
            sys.executable,
            "-m",
            "aiconfigurator.main",
            "cli",
            "default",
            "--model-path",
            "Qwen/Qwen3-32B",
            "--total-gpus",
            "32",
            "--system",
            "h200_sxm",
            "--save-dir",
            str(save_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Find the output directory (CLI creates save_dir/random_suffix_dir/exp_name/)
        output_dir = _find_output_dir(str(save_dir))

        # Verify same experiments exist
        api_exp_names = set(api_result.best_configs.keys())
        all_items = os.listdir(output_dir)
        cli_exp_dirs = {d for d in all_items if os.path.isdir(os.path.join(output_dir, d))}
        assert api_exp_names == cli_exp_dirs, (
            f"Experiment mismatch: API={api_exp_names}, CLI={cli_exp_dirs}\n"
            f"save_dir={save_dir}\n"
            f"output_dir={output_dir}\n"
            f"all items in output_dir: {all_items}"
        )

        # Compare results for each experiment
        for exp_name in api_result.best_configs:
            exp_dir = os.path.join(output_dir, exp_name)

            # Compare best_config_topn.csv (best_configs)
            best_config_path = os.path.join(exp_dir, "best_config_topn.csv")
            assert os.path.exists(best_config_path), f"Missing best_config_topn.csv for {exp_name}"
            cli_best_df = pd.read_csv(best_config_path)
            api_best_df = api_result.best_configs[exp_name]
            _assert_dataframes_equal(api_best_df, cli_best_df, f"{exp_name}/best_configs")

            # Compare pareto.csv (pareto_fronts)
            pareto_path = os.path.join(exp_dir, "pareto.csv")
            assert os.path.exists(pareto_path), f"Missing pareto.csv for {exp_name}"
            cli_pareto_df = pd.read_csv(pareto_path)
            api_pareto_df = api_result.pareto_fronts[exp_name]
            _assert_dataframes_equal(api_pareto_df, cli_pareto_df, f"{exp_name}/pareto_fronts")


class TestCLIExpEquivalence:
    """Tests that cli_exp API produces same results as CLI command."""

    # Path to a smaller example YAML file to speed up tests
    EXAMPLE_YAML_PATH = "src/aiconfigurator/cli/exps/qwen3_32b_disagg.yaml"

    def test_cli_exp_api_vs_command_with_example_yaml(self, tmp_path):
        """cli_exp API should produce same results as running CLI command."""
        if not os.path.exists(self.EXAMPLE_YAML_PATH):
            pytest.skip(f"Example YAML not found: {self.EXAMPLE_YAML_PATH}")

        # Run via Python API
        api_result = cli_exp(yaml_path=self.EXAMPLE_YAML_PATH)

        # Run via CLI command
        save_dir = tmp_path / "cli_output"
        save_dir.mkdir()

        cmd = [
            sys.executable,
            "-m",
            "aiconfigurator.main",
            "cli",
            "exp",
            "--yaml-path",
            self.EXAMPLE_YAML_PATH,
            "--save-dir",
            str(save_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        output_dir = _find_output_dir(str(save_dir))

        # Verify same experiments exist
        api_exp_names = set(api_result.best_configs.keys())
        all_items = os.listdir(output_dir)
        cli_exp_dirs = {d for d in all_items if os.path.isdir(os.path.join(output_dir, d))}
        assert api_exp_names == cli_exp_dirs, (
            f"Experiment mismatch: API={api_exp_names}, CLI={cli_exp_dirs}\n"
            f"save_dir={save_dir}\n"
            f"output_dir={output_dir}\n"
            f"all items in output_dir: {all_items}"
        )

        # Compare results for each experiment
        for exp_name in api_result.best_configs:
            exp_dir = os.path.join(output_dir, exp_name)

            # Compare best_config_topn.csv (best_configs)
            best_config_path = os.path.join(exp_dir, "best_config_topn.csv")
            assert os.path.exists(best_config_path), f"Missing best_config_topn.csv for {exp_name}"
            cli_best_df = pd.read_csv(best_config_path)
            api_best_df = api_result.best_configs[exp_name]
            _assert_dataframes_equal(api_best_df, cli_best_df, f"{exp_name}/best_configs")

            # Compare pareto.csv (pareto_fronts)
            pareto_path = os.path.join(exp_dir, "pareto.csv")
            assert os.path.exists(pareto_path), f"Missing pareto.csv for {exp_name}"
            cli_pareto_df = pd.read_csv(pareto_path)
            api_pareto_df = api_result.pareto_fronts[exp_name]
            _assert_dataframes_equal(api_pareto_df, cli_pareto_df, f"{exp_name}/pareto_fronts")
