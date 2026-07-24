# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI argument parsing functionality.

Tests CLI argument validation, choices, and default values.
"""

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""

    def test_default_mode_core_args_are_required(self, cli_parser):
        """Default mode requires the model, GPU budget, and system."""
        subparsers = [action for action in cli_parser._actions if action.dest == "mode"]
        assert len(subparsers) == 1

        subparser_action = subparsers[0]
        default_parser = subparser_action.choices["default"]

        required_actions = [action for action in default_parser._actions if getattr(action, "required", False)]
        required_args = [action.dest for action in required_actions]

        assert "model_path" in required_args
        assert "total_gpus" in required_args
        assert "system" in required_args

    def test_exp_mode_required_args(self, cli_parser):
        """Test that exp mode requires the yaml_path argument."""
        subparsers = [action for action in cli_parser._actions if action.dest == "mode"]
        assert len(subparsers) == 1

        subparser_action = subparsers[0]
        exp_parser = subparser_action.choices["exp"]

        required_actions = [action for action in exp_parser._actions if getattr(action, "required", False)]
        required_args = [action.dest for action in required_actions]

        assert "yaml_path" in required_args

    def test_mode_choices(self, cli_parser):
        """Ensure supported CLI modes are exposed."""
        action = next(action for action in cli_parser._actions if action.dest == "mode")
        assert set(action.choices.keys()) == {"default", "exp", "generate", "support", "estimate"}

    def test_generate_mode_required_args(self, cli_parser):
        """Test that generate mode requires the correct arguments."""
        subparsers = [action for action in cli_parser._actions if action.dest == "mode"]
        assert len(subparsers) == 1

        subparser_action = subparsers[0]
        generate_parser = subparser_action.choices["generate"]

        required_actions = [action for action in generate_parser._actions if getattr(action, "required", False)]
        required_args = [action.dest for action in required_actions]

        assert "model_path" in required_args
        assert "total_gpus" in required_args
        assert "system" in required_args

    def test_generate_mode_defaults(self, cli_parser):
        """Test that generate mode has correct defaults."""
        args = cli_parser.parse_args(
            ["generate", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "8", "--system", "h200_sxm"]
        )
        assert args.mode == "generate"
        assert args.model_path == "Qwen/Qwen3-32B"
        assert args.backend == common.BackendName.trtllm.value

    def test_generate_mode_model_path(self, cli_parser):
        """Test that generate mode accepts model_path."""
        args = cli_parser.parse_args(
            ["generate", "--model-path", "Qwen/Qwen3-8B", "--total-gpus", "8", "--system", "h200_sxm"]
        )
        assert args.model_path == "Qwen/Qwen3-8B"

    def test_backend_choices_validation(self, cli_parser):
        """Test that backend argument validates against supported choices."""
        subparser_action = next(action for action in cli_parser._actions if action.dest == "mode")
        default_parser = subparser_action.choices["default"]
        action = next(action for action in default_parser._actions if action.dest == "backend")
        expected_choices = [backend.value for backend in common.BackendName] + ["auto"]
        assert sorted(action.choices) == sorted(expected_choices)

    @pytest.mark.parametrize("system_value", ["h200_sxm", "b200_sxm", "gb200"])
    def test_supported_systems_parse_successfully(self, cli_parser, system_value):
        """System flag should accept supported platforms including b200 and gb200."""
        args = cli_parser.parse_args(
            ["default", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "16", "--system", system_value]
        )

        assert args.system == system_value

    def test_default_values_are_set(self, cli_parser):
        """Test that default values are properly set for optional arguments."""
        args = cli_parser.parse_args(
            ["default", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "8", "--system", "h200_sxm"]
        )

        assert args.backend == common.BackendName.trtllm.value
        assert args.backend_version is None
        assert args.database_mode == common.DatabaseMode.SILICON.name
        assert args.log_level is None
        assert args.decode_system is None
        assert args.generated_config_version is None
        assert args.generator_dynamo_version is None
        assert args.isl == 4000
        assert args.osl == 1000
        assert args.save_dir is None
        assert args.ttft == 2000.0
        assert args.tpot == 30.0
        assert args.request_latency is None
        assert args.inclusive_tpot is False
        assert args.prefix == 0
        assert args.engine_step_backend is None

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("--thorough-sweep", None),
            ("--thorough-config", "/tmp/spica.yaml"),
            ("--trace-path", "/tmp/traffic.jsonl"),
            ("--trace-sweep-rounds", "5"),
            ("--trace-parallel-evals", "5"),
        ],
    )
    def test_removed_spica_flags_are_rejected(self, cli_parser, flag, value):
        """The AIC CLI no longer exposes Spica or its trace-sweep controls."""
        argv = [
            "default",
            "--model-path",
            "Qwen/Qwen3-32B",
            "--total-gpus",
            "8",
            "--system",
            "h200_sxm",
            flag,
        ]
        if value is not None:
            argv.append(value)

        with pytest.raises(SystemExit):
            cli_parser.parse_args(argv)

    def test_inclusive_tpot_default_false_in_exp_mode(self, cli_parser, mock_exp_yaml_path):
        """--inclusive-tpot defaults to False in exp mode."""
        args = cli_parser.parse_args(["exp", "--yaml-path", str(mock_exp_yaml_path)])
        assert args.inclusive_tpot is False

    def test_inclusive_tpot_enabled_in_exp_mode(self, cli_parser, mock_exp_yaml_path):
        """--inclusive-tpot can be set in exp mode."""
        args = cli_parser.parse_args(["exp", "--yaml-path", str(mock_exp_yaml_path), "--inclusive-tpot"])
        assert args.inclusive_tpot is True

    def test_log_level_flag(self, cli_parser):
        """Test that --log-level is parsed and normalized."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--log-level",
                "debug",
            ]
        )

        assert args.log_level == "DEBUG"

    def test_legacy_debug_flag(self, cli_parser):
        """Legacy --debug is still accepted for backward compatibility."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--debug",
            ]
        )

        assert args.debug is True

    def test_engine_step_backend_flag(self, cli_parser):
        """Test that the experimental engine step backend can be selected."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--engine-step-backend",
                "rust",
            ]
        )

        assert args.engine_step_backend == "rust"

    def test_save_directory_argument(self, cli_parser):
        """Test that save directory can be specified."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--save-dir",
                "/tmp/test",
            ]
        )

        assert args.save_dir == "/tmp/test"

    @pytest.mark.parametrize(
        "optional_param,value,expected_type",
        [
            ("isl", "8000", int),
            ("osl", "2048", int),
            ("ttft", "300.0", float),
            ("tpot", "10.0", float),
            ("request_latency", "1200.0", float),
            ("prefix", "128", int),
            ("nextn", "3", int),
        ],
    )
    def test_optional_parameters(self, cli_parser, optional_param, value, expected_type):
        """Test that optional parameters can be set and have correct types."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                f"--{optional_param.replace('_', '-')}",
                value,
            ]
        )

        param_value = getattr(args, optional_param)
        assert isinstance(param_value, expected_type)
        assert param_value == expected_type(value)

    def test_nextn_accepts_auto(self, cli_parser):
        """--nextn auto is a literal pass-through; resolution against the
        checkpoint happens later, once the model config is loaded."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "deepseek-ai/DeepSeek-V3",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--nextn",
                "auto",
                "--nextn-accepted",
                "0.7",
            ]
        )
        assert args.nextn == "auto"

    def test_nextn_requires_explicit_acceptance(self, cli_parser):
        from aiconfigurator.cli.main import _resolve_and_validate_nextn

        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--nextn",
                "2",
            ]
        )

        with pytest.raises(SystemExit, match="nextn_accepted"):
            _resolve_and_validate_nextn(args)

    def test_nextn_auto_requires_explicit_acceptance_when_resolved_positive(self, cli_parser, monkeypatch):
        import aiconfigurator.cli.main as cli_main

        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--nextn",
                "auto",
            ]
        )
        monkeypatch.setattr(cli_main, "resolve_nextn_auto", lambda _model_path: 2)

        with pytest.raises(SystemExit, match=r"resolved to nextn=2.*nextn_accepted"):
            cli_main._resolve_and_validate_nextn(args)

    @pytest.mark.parametrize("bad_value", ["-1", "1.5", "always", ""])
    def test_nextn_rejects_non_auto_junk(self, cli_parser, bad_value):
        """--nextn takes a non-negative integer or the literal 'auto'."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(
                [
                    "default",
                    "--model-path",
                    "Qwen/Qwen3-32B",
                    "--total-gpus",
                    "8",
                    "--system",
                    "h200_sxm",
                    "--nextn",
                    bad_value,
                ]
            )

    def test_decode_system_defaults_to_system(self, cli_parser):
        """Decode system defaults to system when omitted and can be overridden."""
        args = cli_parser.parse_args(
            ["default", "--model-path", "Qwen/Qwen3-32B", "--total-gpus", "8", "--system", "h200_sxm"]
        )
        assert args.decode_system is None

        args_with_decode = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--decode-system",
                "gb200",
            ]
        )
        assert args_with_decode.decode_system == "gb200"

    def test_model_path_accepts_huggingface_id(self, cli_parser):
        """Test that --model-path accepts a HuggingFace model ID."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-8B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
            ]
        )
        assert args.model_path == "Qwen/Qwen3-8B"

    @pytest.mark.parametrize(
        "database_mode_value",
        ["SILICON", "HYBRID", "EMPIRICAL", "SOL"],
    )
    def test_database_mode_values_parse_successfully(self, cli_parser, database_mode_value):
        """Database mode flag should accept all supported mode values."""
        args = cli_parser.parse_args(
            [
                "default",
                "--model-path",
                "Qwen/Qwen3-32B",
                "--total-gpus",
                "8",
                "--system",
                "h200_sxm",
                "--database-mode",
                database_mode_value,
            ]
        )
        assert args.database_mode == database_mode_value

    def test_database_mode_invalid_value_raises(self, cli_parser):
        """Test that invalid database_mode value raises an error."""
        with pytest.raises(SystemExit):
            cli_parser.parse_args(
                [
                    "default",
                    "--model-path",
                    "Qwen/Qwen3-32B",
                    "--total-gpus",
                    "8",
                    "--system",
                    "h200_sxm",
                    "--database-mode",
                    "INVALID_MODE",
                ]
            )

    def test_database_mode_choices_validation(self, cli_parser):
        """Test that database_mode argument validates against supported choices."""
        subparser_action = next(action for action in cli_parser._actions if action.dest == "mode")
        default_parser = subparser_action.choices["default"]
        action = next(action for action in default_parser._actions if action.dest == "database_mode")
        expected_choices = [mode.name for mode in common.DatabaseMode if mode != common.DatabaseMode.SOL_FULL]
        assert sorted(action.choices) == sorted(expected_choices)
