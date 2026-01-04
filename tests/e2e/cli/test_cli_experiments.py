# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources as pkg_resources
import subprocess as sp
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _get_exp_yaml_files():
    """Dynamically discover all YAML files in the exps directory."""
    exps_dir = pkg_resources.files("aiconfigurator") / "cli" / "exps"
    return sorted([str(yaml_file) for yaml_file in exps_dir.iterdir() if yaml_file.suffix == ".yaml"])


_ALL_EXP_YAMLS = _get_exp_yaml_files()

# Mark a small subset as suitable for CI/build workflows.
# _BUILD_EXP_FILENAMES = {
#    # Keep this small and stable; it should be representative but fast.
#    "qwen3_32b_request_latency.yaml",
# }

# use all exps for build test now
_BUILD_EXP_FILENAMES = [Path(exp_yaml).name for exp_yaml in _ALL_EXP_YAMLS]


def _parametrize_exp_yamls(yaml_paths: list[str]) -> list:
    params: list = []
    for yaml_path in yaml_paths:
        name = Path(yaml_path).name
        if name in _BUILD_EXP_FILENAMES:
            params.append(pytest.param(yaml_path, id=name, marks=[pytest.mark.build]))
        else:
            params.append(pytest.param(yaml_path, id=name))
    return params


EXP_YAMLS_TO_TEST = _parametrize_exp_yamls(_ALL_EXP_YAMLS)


class TestExps:
    """Test aiconfigurator CLI with various exps."""

    @pytest.mark.parametrize("exp_yaml", EXP_YAMLS_TO_TEST)
    def test_exps(
        self,
        exp_yaml,
    ):
        cmd = ["aiconfigurator", "cli", "exp", "--yaml_path", exp_yaml]
        sp.run(cmd, check=True)
