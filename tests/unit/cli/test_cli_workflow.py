# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit-level workflow tests for CLI wiring.

These tests validate the wiring between CLI entrypoints and internal builders/executors,
while keeping heavy computation mocked out.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from aiconfigurator.cli.main import configure_parser
from aiconfigurator.cli.main import main as cli_main

pytestmark = pytest.mark.unit


class TestCLIIntegration:
    """Workflow tests for the CLI orchestration layer (builders/executor/save)."""

    @patch("aiconfigurator.cli.main.run_default_mode")
    def test_cli_main_success_flow(self, mock_run_default, sample_cli_args_with_save_dir):
        """Test successful CLI main execution flow for default mode."""
        mock_task_config = MagicMock(name="TaskConfig")
        mock_results_df = MagicMock(name="ResultsDF")
        mock_best_configs = {"agg": MagicMock(name="BestConfigDF")}
        mock_best_throughputs = {"agg": 123.4}
        mock_effective_configs = {"agg": mock_task_config}

        # run_default_mode returns a tuple
        mock_run_default.return_value = (
            "agg",
            mock_best_configs,
            {"agg": mock_results_df},
            mock_best_throughputs,
            mock_effective_configs,
        )

        with patch("aiconfigurator.cli.main.save_results") as mock_save:
            cli_main(sample_cli_args_with_save_dir)

        mock_run_default.assert_called_once()

        mock_save.assert_called_once()
        save_kwargs = mock_save.call_args.kwargs
        assert save_kwargs["args"] == sample_cli_args_with_save_dir
        assert save_kwargs["best_configs"] == mock_best_configs
        assert save_kwargs["pareto_fronts"] == {"agg": mock_results_df}
        assert save_kwargs["task_configs"] == mock_effective_configs
        assert save_kwargs["save_dir"] == sample_cli_args_with_save_dir.save_dir

    @patch("aiconfigurator.cli.main.run_exp_mode")
    def test_cli_main_success_flow_exp_mode(
        self,
        mock_run_exp,
        cli_args_factory,
        mock_exp_yaml_path,
    ):
        """Test successful CLI main execution flow for exp mode."""
        mock_task_config = MagicMock(name="TaskConfig")
        mock_results_df = MagicMock(name="ResultsDF")
        mock_best_configs = {"my_exp": MagicMock(name="BestConfigDF")}
        mock_best_throughputs = {"my_exp": 123.4}
        mock_effective_configs = {"my_exp": mock_task_config}

        mock_run_exp.return_value = (
            "my_exp",
            mock_best_configs,
            {"my_exp": mock_results_df},
            mock_best_throughputs,
            mock_effective_configs,
        )

        args = cli_args_factory(
            mode="exp",
            extra_args=["--yaml_path", str(mock_exp_yaml_path)],
            save_dir=str(mock_exp_yaml_path.parent),
        )

        with patch("aiconfigurator.cli.main.save_results") as mock_save:
            cli_main(args)

        mock_run_exp.assert_called_once()

        mock_save.assert_called_once()
        save_kwargs = mock_save.call_args.kwargs
        assert save_kwargs["args"] == args
        assert save_kwargs["best_configs"] == mock_best_configs
        assert save_kwargs["pareto_fronts"] == {"my_exp": mock_results_df}
        assert save_kwargs["task_configs"] == mock_effective_configs
        assert save_kwargs["save_dir"] == str(mock_exp_yaml_path.parent)

    @pytest.mark.parametrize(
        "mode,run_func_patch",
        [
            ("default", "aiconfigurator.cli.main.run_default_mode"),
            ("exp", "aiconfigurator.cli.main.run_exp_mode"),
        ],
    )
    def test_cli_main_build_dispatch(self, mode, run_func_patch, cli_args_factory, mock_exp_yaml_path):
        """Main should dispatch to the correct run function based on CLI mode."""
        mock_task_config = MagicMock(name="TaskConfig")
        mock_result = ("agg", {}, {}, {}, {"agg": mock_task_config})

        with patch(run_func_patch) as mock_run_func:
            mock_run_func.return_value = mock_result
            if mode == "exp":
                args = cli_args_factory(mode=mode, extra_args=["--yaml_path", str(mock_exp_yaml_path)])
            else:
                args = cli_args_factory(mode=mode)
            cli_main(args)

        mock_run_func.assert_called_once()

    def test_cli_main_unsupported_mode_raises(self, cli_args_factory):
        """Unsupported mode should cause SystemExit through argparse validation."""
        parser = argparse.ArgumentParser()
        configure_parser(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["invalid"])

    @pytest.mark.parametrize(
        "mode,run_func_patch",
        [
            ("default", "aiconfigurator.cli.main.run_default_mode"),
            ("exp", "aiconfigurator.cli.main.run_exp_mode"),
        ],
    )
    def test_cli_main_runtime_failure(self, mode, run_func_patch, cli_args_factory, tmp_path):
        """Execution errors propagate for visibility."""
        with patch(run_func_patch) as mock_run_func:
            mock_run_func.side_effect = RuntimeError("failed")

            if mode == "default":
                args = cli_args_factory(mode="default")
            else:
                yaml_file = tmp_path / "exp.yaml"
                yaml_file.write_text("exps: []")
                args = cli_args_factory(mode="exp", extra_args=["--yaml_path", str(yaml_file)])

            with pytest.raises((RuntimeError, SystemExit)):
                cli_main(args)

        mock_run_func.assert_called_once()

    @pytest.mark.parametrize("database_mode", ["SILICON", "HYBRID", "EMPIRICAL"])
    def test_cli_default_mode_with_database_mode(self, cli_args_factory, database_mode):
        """Test that database_mode is correctly parsed and passed through in default mode."""
        args = cli_args_factory(
            mode="default",
            extra_args=["--database_mode", database_mode],
        )
        assert args.database_mode == database_mode

    @patch("aiconfigurator.cli.main.run_exp_mode")
    def test_cli_exp_mode_with_database_mode_in_yaml(self, mock_run_exp, tmp_path):
        """Test that database_mode from YAML is correctly parsed in exp mode."""
        yaml_content = """
exp_with_db_mode:
    serving_mode: "agg"
    model_path: "Qwen/Qwen3-32B"
    system_name: "h200_sxm"
    total_gpus: 8
    database_mode: "HYBRID"
"""
        yaml_file = tmp_path / "exp_db_mode.yaml"
        yaml_file.write_text(yaml_content)

        mock_task_config = MagicMock(name="TaskConfig")
        mock_run_exp.return_value = (
            "exp_with_db_mode",
            {},
            {},
            {},
            {"exp_with_db_mode": mock_task_config},
        )

        parser = argparse.ArgumentParser()
        configure_parser(parser)
        args = parser.parse_args(["exp", "--yaml_path", str(yaml_file)])

        cli_main(args)

        mock_run_exp.assert_called_once()
        # Verify yaml_path was passed correctly
        assert mock_run_exp.call_args.kwargs.get("yaml_path") == str(yaml_file)
