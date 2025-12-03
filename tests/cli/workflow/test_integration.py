# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI integration functionality.

Tests the full CLI workflow and integration between components.
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from aiconfigurator.cli.main import configure_parser
from aiconfigurator.cli.main import main as cli_main


class TestCLIIntegration:
    """Integration tests for the full CLI workflow."""

    @patch("aiconfigurator.cli.main._execute_task_configs")
    @patch("aiconfigurator.cli.main._build_default_task_configs")
    def test_cli_main_success_flow(self, mock_build_default, mock_execute, sample_cli_args_with_save_dir):
        """Test successful CLI main execution flow for default mode."""
        mock_task_config = MagicMock(name="TaskConfig")
        mock_build_default.return_value = {"agg": mock_task_config}

        mock_results_df = MagicMock(name="ResultsDF")
        mock_best_configs = {"agg": MagicMock(name="BestConfigDF")}
        mock_best_throughputs = {"agg": 123.4}
        mock_execute.return_value = (
            "agg",
            mock_best_configs,
            {"agg": mock_results_df},
            mock_best_throughputs,
        )

        with patch("aiconfigurator.cli.main.save_results") as mock_save:
            cli_main(sample_cli_args_with_save_dir)

        mock_build_default.assert_called_once()
        mock_execute.assert_called_once()
        builder_args, builder_mode = mock_execute.call_args.args
        assert builder_args == {"agg": mock_task_config}
        assert builder_mode == "default"

        mock_save.assert_called_once()
        save_kwargs = mock_save.call_args.kwargs
        assert save_kwargs["args"] == sample_cli_args_with_save_dir
        assert save_kwargs["best_configs"] == mock_best_configs
        assert save_kwargs["pareto_fronts"] == {"agg": mock_results_df}
        assert save_kwargs["task_configs"] == {"agg": mock_task_config}
        assert save_kwargs["save_dir"] == sample_cli_args_with_save_dir.save_dir

    @patch("aiconfigurator.cli.main.save_results")
    @patch("aiconfigurator.cli.main._execute_task_configs")
    @patch("aiconfigurator.cli.main._build_experiment_task_configs")
    def test_cli_main_success_flow_exp_mode(
        self,
        mock_build_exp,
        mock_execute,
        mock_save,
        cli_args_factory,
        mock_exp_yaml_path,
    ):
        """Test successful CLI main execution flow for exp mode."""
        mock_task_config = MagicMock(name="TaskConfig")
        mock_build_exp.return_value = {"my_exp": mock_task_config}
        mock_results_df = MagicMock(name="ResultsDF")
        mock_best_configs = {"my_exp": MagicMock(name="BestConfigDF")}
        mock_best_throughputs = {"my_exp": 123.4}
        mock_execute.return_value = (
            "my_exp",
            mock_best_configs,
            {"my_exp": mock_results_df},
            mock_best_throughputs,
        )

        args = cli_args_factory(
            mode="exp",
            extra_args=["--yaml_path", str(mock_exp_yaml_path)],
            save_dir=str(mock_exp_yaml_path.parent),
        )

        cli_main(args)

        mock_build_exp.assert_called_once()
        mock_execute.assert_called_once()
        builder_args, builder_mode = mock_execute.call_args.args
        assert builder_args == {"my_exp": mock_task_config}
        assert builder_mode == "exp"

        mock_save.assert_called_once()
        save_kwargs = mock_save.call_args.kwargs
        assert save_kwargs["args"] == args
        assert save_kwargs["best_configs"] == mock_best_configs
        assert save_kwargs["pareto_fronts"] == {"my_exp": mock_results_df}
        assert save_kwargs["task_configs"] == {"my_exp": mock_task_config}
        assert save_kwargs["save_dir"] == str(mock_exp_yaml_path.parent)

    @pytest.mark.parametrize(
        "mode,build_patch",
        [
            ("default", "aiconfigurator.cli.main._build_default_task_configs"),
            ("exp", "aiconfigurator.cli.main._build_experiment_task_configs"),
        ],
    )
    @patch("aiconfigurator.cli.main._execute_task_configs")
    def test_cli_main_build_dispatch(self, mock_execute, mode, build_patch, cli_args_factory):
        """Main should dispatch to the correct builder based on CLI mode."""
        mock_execute.return_value = ("agg", {}, {}, {})

        with patch(build_patch) as mock_builder:
            mock_builder.return_value = {}
            args = cli_args_factory(mode=mode)
            cli_main(args)

        mock_builder.assert_called_once()
        mock_execute.assert_called_once_with({}, mode)

    @pytest.mark.parametrize(
        "builder_patch",
        [
            "aiconfigurator.cli.main._build_default_task_configs",
            "aiconfigurator.cli.main._build_experiment_task_configs",
        ],
    )
    def test_cli_main_unsupported_mode_raises(self, builder_patch, cli_args_factory):
        """Unsupported mode should cause SystemExit through argparse validation."""
        with patch(builder_patch) as mock_builder:
            mock_builder.return_value = {}
            parser = argparse.ArgumentParser()
            configure_parser(parser)
            with pytest.raises(SystemExit):
                parser.parse_args(["invalid"])
            mock_builder.assert_not_called()

    @pytest.mark.parametrize(
        "builder_patch",
        [
            "aiconfigurator.cli.main._build_default_task_configs",
            "aiconfigurator.cli.main._build_experiment_task_configs",
        ],
    )
    @patch("aiconfigurator.cli.main._execute_task_configs")
    def test_cli_main_runtime_failure(self, mock_execute, builder_patch, cli_args_factory, tmp_path):
        """Execution errors propagate as RuntimeError for visibility."""
        mock_execute.side_effect = RuntimeError("failed")

        with patch(builder_patch) as mock_builder:
            mock_builder.return_value = {}

            if "default" in builder_patch:
                args = cli_args_factory(mode="default")
            else:
                yaml_file = tmp_path / "exp.yaml"
                yaml_file.write_text("exps: []")
                args = cli_args_factory(mode="exp", extra_args=["--yaml_path", str(yaml_file)])

            with pytest.raises(RuntimeError):
                cli_main(args)

        mock_builder.assert_called_once()
        mock_execute.assert_called_once()

    @pytest.mark.parametrize("database_mode", ["SILICON", "HYBRID", "EMPIRICAL"])
    def test_cli_default_mode_with_database_mode(self, cli_args_factory, database_mode):
        """Test that database_mode is correctly parsed and passed through in default mode."""
        args = cli_args_factory(
            mode="default",
            extra_args=["--database_mode", database_mode],
        )
        assert args.database_mode == database_mode

    @patch("aiconfigurator.cli.main._execute_task_configs")
    @patch("aiconfigurator.cli.main._build_experiment_task_configs")
    def test_cli_exp_mode_with_database_mode_in_yaml(self, mock_build_exp, mock_execute, tmp_path):
        """Test that database_mode from YAML is correctly parsed in exp mode."""
        yaml_content = """
exp_with_db_mode:
    serving_mode: "agg"
    model_name: "QWEN3_32B"
    system_name: "h200_sxm"
    total_gpus: 8
    database_mode: "HYBRID"
"""
        yaml_file = tmp_path / "exp_db_mode.yaml"
        yaml_file.write_text(yaml_content)

        mock_task_config = MagicMock(name="TaskConfig")
        mock_build_exp.return_value = {"exp_with_db_mode": mock_task_config}
        mock_execute.return_value = ("exp_with_db_mode", {}, {}, {})

        parser = argparse.ArgumentParser()
        configure_parser(parser)
        args = parser.parse_args(["exp", "--yaml_path", str(yaml_file)])

        cli_main(args)

        mock_build_exp.assert_called_once()
        mock_execute.assert_called_once()
