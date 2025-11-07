# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess as sp

import pytest

MODELS_TO_TEST = [
    "LLAMA2_7B",
    "LLAMA2_13B",
    "LLAMA2_70B",
    "LLAMA3.1_8B",
    "LLAMA3.1_70B",
    "LLAMA3.1_405B",
    "MOE_Mixtral8x7B",
    "MOE_Mixtral8x22B",
    "DEEPSEEK_V3",
    "QWEN2.5_1.5B",
    "QWEN2.5_7B",
    "QWEN2.5_32B",
    "QWEN2.5_72B",
    "QWEN3_32B",
    "QWEN3_0.6B",
    "QWEN3_1.7B",
    "QWEN3_8B",
    "QWEN3_235B",
    "QWEN3_480B",
    "Nemotron_super_v1.1",
]

SYSTEMS_TO_TEST = [
    # "a100_sxm",
    # "h100_sxm",
    # "h200_sxm",
    # "b200_sxm",
    # "gb200_sxm",
    "l40s",
]

BACKENDS_TO_TEST = [
    # "vllm",
    "trtllm",
]


class TestModelSystemCombinations:
    """Test aiconfigurator CLI with various model/system combinations."""

    @pytest.mark.parametrize("model", MODELS_TO_TEST)
    @pytest.mark.parametrize("system", SYSTEMS_TO_TEST)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_model_system_combination(
        self,
        model,
        system,
        backend,
    ):
        cmd = [
            "aiconfigurator",
            "cli",
            "default",
            "--total_gpus",
            "32",
            "--model",
            model,
            "--system",
            system,
            "--backend",
            backend,
        ]
        sp.run(cmd, check=True)
