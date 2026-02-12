# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess as sp
from functools import cache

import pytest

from aiconfigurator.sdk.models import get_model_family
from aiconfigurator.sdk.perf_database import get_latest_database_version

pytestmark = [pytest.mark.e2e, pytest.mark.sweep]

MODELS_TO_TEST = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-235B-A22B",
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
]

SYSTEMS_TO_TEST = [
    # "a100_sxm",
    # "h100_sxm",
    # "h200_sxm",
    # "b200_sxm",
    "gb200",
    # "l40s",
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
            "--total-gpus",
            "32",
            "--model-path",
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
