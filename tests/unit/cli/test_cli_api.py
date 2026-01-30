# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI API functions.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aiconfigurator.cli import CLIResult, cli_exp, cli_generate

pytestmark = pytest.mark.unit


class TestCLIExpUnit:
    """Unit tests for cli_exp API (mocked)."""

    @patch("aiconfigurator.cli.api._execute_task_configs_internal")
    @patch("aiconfigurator.cli.api.build_experiment_task_configs")
    def test_cli_exp_dict_config_equivalent_to_example_yaml(self, mock_build, mock_execute):
        """cli_exp with dict config should work correctly (mocked).

        Equivalent to exp_agg_simplified from src/aiconfigurator/cli/example.yaml:
            exp_agg_simplified:
              mode: "patch"
              serving_mode: "agg"
              model_path: "deepseek-ai/DeepSeek-V3"
              total_gpus: 8
              system_name: "h200_sxm"
        """
        # Setup mocks
        mock_task_config = MagicMock(name="TaskConfig")
        mock_build.return_value = {"exp_agg_simplified": mock_task_config}
        mock_execute.return_value = (
            "exp_agg_simplified",
            {"exp_agg_simplified": pd.DataFrame()},
            {"exp_agg_simplified": pd.DataFrame()},
            {"exp_agg_simplified": 100.0},
            {"exp_agg_simplified": mock_task_config},
        )

        # Simplified version based on example.yaml exp_agg_simplified
        config = {
            "exp_agg_simplified": {
                "mode": "patch",
                "serving_mode": "agg",
                "model_path": "deepseek-ai/DeepSeek-V3",
                "total_gpus": 8,
                "system_name": "h200_sxm",
            }
        }

        result = cli_exp(config=config)

        # Verify build_experiment_task_configs was called with correct params
        mock_build.assert_called_once_with(
            yaml_path=None,
            config=config,
        )

        assert isinstance(result, CLIResult)
        assert "exp_agg_simplified" in result.task_configs
        assert "exp_agg_simplified" in result.best_throughputs


class TestCLIGenerateEquivalence:
    """Tests that cli_generate produces same output as CLI command."""

    def test_cli_generate_api_vs_command(self, tmp_path):
        """cli_generate API should produce same config as CLI command."""
        import os
        import subprocess
        import sys

        import yaml

        def _find_output_dir(save_dir: str) -> str:
            """Recursively find the directory containing experiment results."""
            for root, dirs, files in os.walk(save_dir):
                if "generator_params.yaml" in files or "generator_config.yaml" in files:
                    return root
            raise FileNotFoundError(f"Could not find output directory in {save_dir}")

        # Run via Python API
        api_result = cli_generate(
            model_path="Qwen/Qwen3-32B",
            total_gpus=8,
            system="h200_sxm",
            backend="trtllm",
        )

        # Run via CLI command
        save_dir = tmp_path / "cli_output"
        save_dir.mkdir()

        cmd = [
            sys.executable,
            "-m",
            "aiconfigurator.main",
            "cli",
            "generate",
            "--model_path",
            "Qwen/Qwen3-32B",
            "--total_gpus",
            "8",
            "--system",
            "h200_sxm",
            "--backend",
            "trtllm",
            "--save_dir",
            str(save_dir),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # CLI generate creates files in a subdirectory within save_dir
        output_dir = _find_output_dir(str(save_dir))
        assert os.path.exists(output_dir), f"Expected output directory {output_dir}"

        # Compare parallelism values
        # API returns these directly
        api_tp = api_result["parallelism"]["tp"]
        api_pp = api_result["parallelism"]["pp"]
        api_replicas = api_result["parallelism"]["replicas"]
        api_gpus_used = api_result["parallelism"]["gpus_used"]

        # CLI saves generator_config.yaml in the agg subdirectory
        agg_dir = os.path.join(output_dir, "agg")
        if os.path.exists(agg_dir):
            generator_config_path = os.path.join(agg_dir, "generator_config.yaml")
            if os.path.exists(generator_config_path):
                with open(generator_config_path) as f:
                    cli_config = yaml.safe_load(f)
                # Extract TP/PP from the saved config
                cli_tp = cli_config.get("tensor_parallel_size")
                cli_pp = cli_config.get("pipeline_parallel_size")

                if cli_tp is not None and cli_pp is not None:
                    assert api_tp == cli_tp, f"TP mismatch: API={api_tp}, CLI={cli_tp}"
                    assert api_pp == cli_pp, f"PP mismatch: API={api_pp}, CLI={cli_pp}"

        # Verify API result has expected structure
        assert api_tp > 0, "TP should be positive"
        assert api_pp > 0, "PP should be positive"
        assert api_replicas > 0, "Replicas should be positive"
        assert api_gpus_used > 0, "GPUs used should be positive"
        assert api_tp * api_pp * api_replicas == api_gpus_used, "TP * PP * replicas should equal GPUs used"


class TestCLISupportEquivalence:
    """Tests that cli_support API produces same results as CLI command."""

    def test_cli_support_api_vs_command(self):
        """cli_support API should return same support status as CLI command."""
        import subprocess
        import sys

        from aiconfigurator.cli import cli_support

        # Run via Python API
        api_result = cli_support("Qwen/Qwen3-32B", "h200_sxm")

        # Run via CLI command
        cmd = [
            sys.executable,
            "-m",
            "aiconfigurator.main",
            "cli",
            "support",
            "--model_path",
            "Qwen/Qwen3-32B",
            "--system",
            "h200_sxm",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Parse CLI output for support status
        cli_agg_supported = "Aggregated Support:    YES" in result.stdout
        cli_disagg_supported = "Disaggregated Support: YES" in result.stdout

        # Compare results
        assert api_result.agg_supported == cli_agg_supported, (
            f"Aggregated support mismatch: API={api_result.agg_supported}, CLI={cli_agg_supported}"
        )
        assert api_result.disagg_supported == cli_disagg_supported, (
            f"Disaggregated support mismatch: API={api_result.disagg_supported}, CLI={cli_disagg_supported}"
        )
