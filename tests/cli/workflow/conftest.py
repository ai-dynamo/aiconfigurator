# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import tempfile
from pathlib import Path
from typing import Any

import pytest

from aiconfigurator.cli.main import configure_parser as configure_cli_parser


@pytest.fixture
def cli_parser():
    """Pre-configured CLI parser for testing."""
    parser = argparse.ArgumentParser()
    configure_cli_parser(parser)
    return parser


@pytest.fixture
def cli_args_factory():
    """Factory to build parsed CLI arguments for the default mode."""

    def _factory(*, mode: str = "default", extra_args: list[str] | None = None, **overrides: Any):
        parser = argparse.ArgumentParser()
        configure_cli_parser(parser)

        base_args: dict[str, Any] = {}
        temp_file: Path | None = None

        if mode == "default":
            base_args.update(
                {
                    "model": "QWEN3_32B",
                    "total_gpus": 8,
                    "system": "h200_sxm",
                }
            )
        elif mode == "exp":
            yaml_override = overrides.pop("yaml_path", None)
            if yaml_override is None and (not extra_args or "--yaml_path" not in extra_args):
                with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
                    tmp.write("exps: []\n")
                    tmp.flush()
                    temp_file = Path(tmp.name)
                    yaml_override = tmp.name
            if yaml_override is not None:
                base_args["yaml_path"] = yaml_override

        for key, value in overrides.items():
            if key in base_args or value is not None:
                base_args[key] = value

        arg_list: list[str] = [mode]

        for key, value in base_args.items():
            option = f"--{key}"
            if isinstance(value, bool):
                if value:
                    arg_list.append(option)
            elif value is None:
                continue
            else:
                arg_list.extend([option, str(value)])

        if extra_args:
            arg_list.extend(extra_args)

        namespace = parser.parse_args(arg_list)

        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)

        return namespace

    return _factory


@pytest.fixture
def sample_cli_args(cli_args_factory):
    """Sample CLI arguments for testing."""
    return cli_args_factory()


@pytest.fixture
def sample_cli_args_with_save_dir(cli_args_factory, tmp_path):
    """Sample CLI arguments that include a save directory."""
    return cli_args_factory(save_dir=str(tmp_path))


@pytest.fixture
def mock_exp_yaml_path(tmp_path):
    """Creates a dummy experiment YAML file and returns its path."""
    yaml_content = """
    my_exp:
        serving_mode: "agg"
        model_name: "QWEN3_32B"
        system_name: "h200_sxm"
        total_gpus: 8
    """
    exp_file = tmp_path / "exp.yaml"
    exp_file.write_text(yaml_content)
    return exp_file
