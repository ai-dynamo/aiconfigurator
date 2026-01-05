# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess as sp
from functools import cache

import pytest

from aiconfigurator.sdk.models import get_model_family
from aiconfigurator.sdk.perf_database import get_latest_database_version

pytestmark = [pytest.mark.e2e, pytest.mark.sweep]

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
    "Nemotron_super_v1.1",
]

SYSTEMS_TO_TEST = [
    "a100_sxm",
    "h100_sxm",
    "h200_sxm",
    "b200_sxm",
    "gb200_sxm",
    "l40s",
]

BACKENDS_TO_TEST = [
    "vllm",
    "trtllm",
    "sglang",
]


@cache
def _latest_db_version(system: str, backend: str) -> str | None:
    return get_latest_database_version(system=system, backend=backend)


class TestModelSystemCombinations:
    """Broad CLI compatibility matrix across model/system/backend."""

    @pytest.mark.parametrize("model", MODELS_TO_TEST)
    @pytest.mark.parametrize("system", SYSTEMS_TO_TEST)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_model_system_combination(
        self,
        model,
        system,
        backend,
    ):
        # Skip combinations that are known to be unsupported by the codebase.
        if backend == "vllm" and get_model_family(model) == "DEEPSEEK":
            pytest.skip("DEEPSEEK models are not supported on the vllm backend.")

        # Skip combinations that don't have a database available for "latest".
        version = _latest_db_version(system, backend)
        if not version:
            pytest.skip(f"No latest database version found for {system=}, {backend=}")

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
        completed = sp.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            combined = f"{completed.stdout}\n{completed.stderr}".strip()
            raise AssertionError(f"CLI failed for {model=}, {system=}, {backend=}, {version=}:\n{combined}")
