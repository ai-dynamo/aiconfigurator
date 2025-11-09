# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources as pkg_resources
import subprocess as sp

import pytest


def _get_exp_yaml_files():
    """Dynamically discover all YAML files in the exps directory."""
    exps_dir = pkg_resources.files("aiconfigurator") / "cli" / "exps"
    return sorted([str(yaml_file) for yaml_file in exps_dir.iterdir() if yaml_file.suffix == ".yaml"])


EXP_YAMLS_TO_TEST = _get_exp_yaml_files()


class TestModelSystemCombinations:
    """Test aiconfigurator CLI with various model/system combinations."""

    @pytest.mark.parametrize("exp_yaml", EXP_YAMLS_TO_TEST)
    def test_model_system_combination(
        self,
        exp_yaml,
    ):
        cmd = ["aiconfigurator", "cli", "exp", "--yaml_path", exp_yaml]
        sp.run(cmd, check=True)
