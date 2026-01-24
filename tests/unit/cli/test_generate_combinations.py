# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for cli generate combinations.
"""

import os

import pytest
import yaml

from aiconfigurator.cli.main import main as cli_main


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["trtllm", "sglang", "vllm"])
@pytest.mark.parametrize("system", ["h200_sxm", "gb200_sxm"])
def test_cli_generate_combinations(
    cli_args_factory,
    tmp_path,
    backend,
    system,
):
    """
    Test that cli generate works for various backend and system combinations.
    Runs the actual CLI without mocking.
    """
    args = cli_args_factory(
        mode="generate",
        model_path="Qwen/Qwen3-32B",
        total_gpus=16,
        system=system,
        backend=backend,
        save_dir=str(tmp_path),
    )

    cli_main(args)

    # Verify output directory was created
    output_dirs = [d for d in os.listdir(tmp_path) if os.path.isdir(tmp_path / d)]
    assert len(output_dirs) == 1, f"Expected 1 output directory, found {output_dirs}"

    output_dir = tmp_path / output_dirs[0]

    # Verify generator_config.yaml was created
    generator_config_path = output_dir / "generator_config.yaml"
    assert generator_config_path.exists(), "generator_config.yaml should be created"

    # Load and verify the generated config
    with open(generator_config_path) as f:
        config = yaml.safe_load(f)

    # Check TP/PP logic worked (gb200 should have TP=4, h200 should have TP=8)
    tp = config["params"]["agg"]["tensor_parallel_size"]
    pp = config["params"]["agg"]["pipeline_parallel_size"]

    if system == "gb200_sxm":
        assert tp == 4, f"gb200_sxm should have TP=4, got {tp}"
    else:
        assert tp == 8, f"h200_sxm should have TP=8, got {tp}"

    assert pp == 1, f"PP should be 1, got {pp}"
    assert config["backend"] == backend
