# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for cli generate combinations.
"""

from unittest.mock import MagicMock, patch

import pytest

from aiconfigurator.cli.main import main as cli_main


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["trtllm", "sglang", "vllm"])
@pytest.mark.parametrize("system", ["h200_sxm", "gb200_sxm"])
@patch("aiconfigurator.cli.main.generate_backend_artifacts")
@patch("aiconfigurator.cli.main.get_latest_database_version")
@patch("aiconfigurator.cli.main.safe_mkdir")
@patch("builtins.open", new_callable=MagicMock)
def test_cli_generate_combinations(
    mock_open,
    mock_safe_mkdir,
    mock_get_version,
    mock_generate_artifacts,
    cli_args_factory,
    tmp_path,
    backend,
    system,
):
    """
    Test that cli generate works for various backend and system combinations.
    """
    mock_safe_mkdir.return_value = str(tmp_path)
    mock_get_version.return_value = "1.0.0"
    mock_generate_artifacts.return_value = {"run_0.sh": "#!/bin/bash\n"}

    # Run cli generate for the combination
    args = cli_args_factory(
        mode="generate",
        model="QWEN3_32B",
        total_gpus=16,
        system=system,
        backend=backend,
        save_dir=str(tmp_path),
    )

    # This should complete without error
    cli_main(args)

    # Verify that generator params were built and artifacts were requested
    mock_generate_artifacts.assert_called_once()
    call_args = mock_generate_artifacts.call_args
    assert call_args.kwargs["backend"] == backend

    # Check if TP/PP logic worked (e.g. gb200 should have TP=4, h200 should have TP=8)
    params = call_args.kwargs["params"]
    tp = params["params"]["agg"]["tensor_parallel_size"]
    if system == "gb200_sxm":
        assert tp == 4
    else:
        assert tp == 8


@pytest.mark.unit
@patch("aiconfigurator.cli.main.generate_backend_artifacts")
@patch("aiconfigurator.cli.main.get_latest_database_version")
@patch("aiconfigurator.cli.main.safe_mkdir")
@patch("builtins.open", new_callable=MagicMock)
@patch("aiconfigurator.cli.main.get_model_config_from_hf_id")
def test_cli_generate_hf_id(
    mock_get_model_config,
    mock_open,
    mock_safe_mkdir,
    mock_get_version,
    mock_generate_artifacts,
    cli_args_factory,
    tmp_path,
):
    """
    Test that cli generate works with hf_id.
    """
    mock_safe_mkdir.return_value = str(tmp_path)
    mock_get_version.return_value = "1.0.0"
    mock_generate_artifacts.return_value = {"run_0.sh": "#!/bin/bash\n"}
    mock_get_model_config.return_value = ["LLAMA", 32, 32, 8, 128, 4096, 11008, 32000, 4096, 0, 0, 0, {}]

    # Run cli generate with hf_id
    args = cli_args_factory(
        mode="generate",
        hf_id="Qwen/Qwen2.5-7B",
        total_gpus=8,
        system="h200_sxm",
        save_dir=str(tmp_path),
    )

    # This should complete without error
    cli_main(args)

    mock_generate_artifacts.assert_called_once()
    call_args = mock_generate_artifacts.call_args
    params = call_args.kwargs["params"]
    assert params["service"]["model_name"] == "Qwen/Qwen2.5-7B"
