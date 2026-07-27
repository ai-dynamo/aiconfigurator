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
    "a100_sxm",
    "h100_sxm",
    "h200_sxm",
    "b200_sxm",
    "gb200",
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
        # DeepSeek on vLLM runs on data now (mla module tables), but the latest
        # repo vllm DBs still limit some systems: a100_sxm (SM80, no FP8
        # hardware) and l40s (data predates the SM89 fp8_block floor) reject
        # DeepSeek's default fp8_block quant at DB validation; h100_sxm finds
        # no memory-feasible parallel config because the data carries no
        # moe_ep>8 rows. Shrink this list as newer vllm data lands.
        if backend == "vllm" and get_model_family(model) == "DEEPSEEK" and system in ("a100_sxm", "l40s", "h100_sxm"):
            pytest.skip(f"DeepSeek on vllm lacks usable data on {system} (see comment).")

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
