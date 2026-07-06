# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Naive generation must keep the system identity in NodeConfig.

collect_generator_params rebuilds NodeConfig with only num_gpus_per_node, so the
requested system was dropped from NodeConfig.system_name. The vLLM run.sh reads
that field to pick the device env var (B60 needs ONEAPI_DEVICE_SELECTOR, not
CUDA_VISIBLE_DEVICES), so it must survive into the generated params.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aiconfigurator.generator.naive import build_naive_generator_params

_SYS_CFG = {"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3}


@pytest.fixture(autouse=True)
def _mock_naive_env():
    with (
        patch("aiconfigurator.generator.naive._estimate_model_weight_bytes", return_value=15 * 1024**3),
        patch("aiconfigurator.generator.naive._get_system_config", return_value=_SYS_CFG),
    ):
        yield


def test_node_config_carries_requested_system():
    result = build_naive_generator_params(
        model_name="Qwen/Qwen3-8B",
        total_gpus=1,
        system_name="b60",
        backend_name="vllm",
    )
    assert result["NodeConfig"]["system_name"] == "b60"
    # num_gpus_per_node still present (was the only field before the fix).
    assert "num_gpus_per_node" in result["NodeConfig"]


def test_node_config_override_still_propagates():
    result = build_naive_generator_params(
        model_name="Qwen/Qwen3-8B",
        total_gpus=1,
        system_name="b60",
        backend_name="vllm",
        generator_overrides={"NodeConfig": {"system_name": "h200_sxm"}},
    )
    assert result["NodeConfig"]["system_name"] == "h200_sxm"


def test_non_b60_system_is_preserved():
    result = build_naive_generator_params(
        model_name="Qwen/Qwen3-8B",
        total_gpus=1,
        system_name="h200_sxm",
        backend_name="vllm",
    )
    assert result["NodeConfig"]["system_name"] == "h200_sxm"
