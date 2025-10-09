# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for CLI argument parsing functionality.

Tests CLI argument validation, choices, and default values.
"""

import pytest
from aiconfigurator.sdk import common


class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""

    def test_default_mode_required_args(self, cli_parser):
        """Test that configure_parser creates a valid argument parser."""
        subparsers = [action for action in cli_parser._actions if action.dest == 'mode']
        assert len(subparsers) == 1

        subparser_action = subparsers[0]
        default_parser = subparser_action.choices['default']

        required_actions = [action for action in default_parser._actions if getattr(action, "required", False)]
        required_args = [action.dest for action in required_actions]

        assert 'model' in required_args
        assert 'total_gpus' in required_args
        assert 'system' in required_args

    def test_exp_mode_required_args(self, cli_parser):
        """Test that exp mode requires the yaml_path argument."""
        subparsers = [action for action in cli_parser._actions if action.dest == 'mode']
        assert len(subparsers) == 1

        subparser_action = subparsers[0]
        exp_parser = subparser_action.choices['exp']

        required_actions = [action for action in exp_parser._actions if getattr(action, "required", False)]
        required_args = [action.dest for action in required_actions]

        assert 'yaml_path' in required_args

    def test_mode_choices(self, cli_parser):
        """Ensure supported CLI modes are exposed."""
        action = next(action for action in cli_parser._actions if action.dest == 'mode')
        assert set(action.choices.keys()) == {'default', 'exp'}

    @pytest.mark.parametrize("param_name,expected_choices", [
        ("model", list(common.SupportedModels.keys())),
        ("backend", [backend.value for backend in common.BackendName])
    ])
    def test_argument_choices_validation(self, cli_parser, param_name, expected_choices):
        """Test that arguments validate against supported choices."""
        subparser_action = next(action for action in cli_parser._actions if action.dest == 'mode')
        default_parser = subparser_action.choices['default']
        action = next(action for action in default_parser._actions if action.dest == param_name)
        assert sorted(action.choices) == sorted(expected_choices)

    @pytest.mark.parametrize("system_value", ['h200_sxm', 'b200_sxm', 'gb200_sxm'])
    def test_supported_systems_parse_successfully(self, cli_parser, system_value):
        """System flag should accept supported platforms including b200 and gb200."""
        args = cli_parser.parse_args([
            'default',
            '--model', 'QWEN3_32B',
            '--total_gpus', '16',
            '--system', system_value
        ])

        assert args.system == system_value

    def test_default_values_are_set(self, cli_parser):
        """Test that default values are properly set for optional arguments."""
        args = cli_parser.parse_args([
            'default',
            '--model', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm'
        ])

        assert args.backend == common.BackendName.trtllm.value
        assert args.backend_version is None
        assert args.debug is False
        assert args.decode_system is None
        assert args.generated_config_version is None
        assert args.isl == 4000
        assert args.osl == 1000
        assert args.save_dir is None
        assert args.ttft == 1000.0
        assert args.tpot == 20.0

    def test_debug_mode_flag(self, cli_parser):
        """Test that debug mode can be enabled."""
        args = cli_parser.parse_args([
            'default',
            '--model', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm',
            '--debug'
        ])

        assert args.debug is True

    def test_save_directory_argument(self, cli_parser):
        """Test that save directory can be specified."""
        args = cli_parser.parse_args([
            'default',
            '--model', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm',
            '--save_dir', '/tmp/test'
        ])

        assert args.save_dir == '/tmp/test'

    @pytest.mark.parametrize("optional_param,value,expected_type", [
        ("isl", "8000", int),
        ("osl", "2048", int),
        ("ttft", "300.0", float),
        ("tpot", "10.0", float),
    ])
    def test_optional_parameters(self, cli_parser, optional_param, value, expected_type):
        """Test that optional parameters can be set and have correct types."""
        args = cli_parser.parse_args([
            'default',
            '--model', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm',
            f'--{optional_param}', value
        ])

        param_value = getattr(args, optional_param)
        assert isinstance(param_value, expected_type)
        assert param_value == expected_type(value) 

    def test_decode_system_defaults_to_system(self, cli_parser):
        """Decode system defaults to system when omitted and can be overridden."""
        args = cli_parser.parse_args([
            'default',
            '--model', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm'
        ])
        assert args.decode_system is None

        args_with_decode = cli_parser.parse_args([
            'default',
            '--model', 'QWEN3_32B',
            '--total_gpus', '8',
            '--system', 'h200_sxm',
            '--decode_system', 'gb200_sxm'
        ])
        assert args_with_decode.decode_system == 'gb200_sxm'