# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for cli generate combinations.
"""

import os

import pytest
import yaml

from aiconfigurator.cli.main import main as cli_main

# Test cases: (model_path, expected_min_tp, description)
# expected_min_tp: None means any valid TP, int means TP must be >= that value
MODEL_TEST_CASES = [
    # Small/medium model - TP can be 1 or more
    ("Qwen/Qwen3-32B", None, "32B model fits on single GPU"),
    # Large model - requires TP > 1
    ("Qwen/Qwen3-235B-A22B", 2, "235B MoE model requires multiple GPUs"),
    # Huge model - caps at max TP (model too large for single node)
    ("deepseek-ai/DeepSeek-V3", "max", "671B MoE model caps at gpus_per_node"),
]


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["trtllm", "sglang", "vllm"])
@pytest.mark.parametrize("system", ["h200_sxm", "gb200_sxm"])
@pytest.mark.parametrize(
    "model_path,expected_min_tp,description",
    MODEL_TEST_CASES,
    ids=[case[2] for case in MODEL_TEST_CASES],
)
def test_cli_generate_combinations(
    cli_args_factory,
    tmp_path,
    backend,
    system,
    model_path,
    expected_min_tp,
    description,
):
    """
    Test that cli generate works for various model, backend, and system combinations.

    Tests TP calculation based on model size:
    - Small models: TP >= 1
    - Large models (235B): TP > 1 (requires multiple GPUs)
    - Huge models (671B): TP == max_tp (caps at gpus_per_node)
    """
    args = cli_args_factory(
        mode="generate",
        model_path=model_path,
        total_gpus=32,
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

    tp = config["params"]["agg"]["tensor_parallel_size"]
    pp = config["params"]["agg"]["pipeline_parallel_size"]

    # TP should be a power of 2
    assert tp >= 1, f"TP should be at least 1, got {tp}"
    assert tp & (tp - 1) == 0, f"TP should be a power of 2, got {tp}"

    # TP should not exceed gpus_per_node for the system
    max_tp = 4 if system == "gb200_sxm" else 8
    assert tp <= max_tp, f"TP should be <= {max_tp} for {system}, got {tp}"

    # Check expected TP constraints based on model size
    if expected_min_tp == "max":
        # Huge model should cap at max TP
        assert tp == max_tp, f"{description}: expected TP={max_tp}, got {tp}"
    elif expected_min_tp is not None:
        # Large model should require TP >= expected_min_tp
        assert tp >= expected_min_tp, f"{description}: expected TP >= {expected_min_tp}, got {tp}"

    assert pp == 1, f"PP should be 1, got {pp}"
    assert config["backend"] == backend

    # Verify correct number of run scripts are generated (one per node)
    gpus_per_node = 4 if system == "gb200_sxm" else 8
    gpus_per_worker = tp * pp
    num_workers = 32 // gpus_per_worker  # total_gpus=32
    workers_per_node = gpus_per_node // gpus_per_worker
    expected_nodes = (num_workers + workers_per_node - 1) // workers_per_node  # ceil division

    run_scripts = [f for f in os.listdir(output_dir) if f.startswith("run_") and f.endswith(".sh")]
    assert len(run_scripts) == expected_nodes, (
        f"Expected {expected_nodes} run scripts for {num_workers} workers "
        f"({gpus_per_worker} GPUs each) on {gpus_per_node} GPUs/node, "
        f"but found {len(run_scripts)}: {run_scripts}"
    )
