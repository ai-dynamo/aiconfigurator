# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI API functions (mocked).
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aiconfigurator.cli import CLIResult, cli_exp

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
