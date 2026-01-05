# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess as sp

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.build]

_DEFAULT_BUILD_CASES = [
    pytest.param(
        {
            "model": "QWEN3_32B",
            "system": "h200_sxm",
            "backend": "trtllm",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 10,
        },
    ),
    pytest.param(
        {
            "model": "DEEPSEEK_V3",
            "system": "h200_sxm",
            "backend": "trtllm",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 100,
        },
    ),
    pytest.param(
        {
            "model": "QWEN3_32B",
            "system": "h200_sxm",
            "backend": "sglang",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 10,
        },
    ),
    pytest.param(
        {
            "model": "DEEPSEEK_V3",
            "system": "h200_sxm",
            "backend": "sglang",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 100,
        },
    ),
    pytest.param(
        {
            "model": "QWEN3_32B",
            "system": "h200_sxm",
            "backend": "vllm",
            "total_gpus": 32,
            "isl": 4000,
            "osl": 1000,
            "prefix": 0,
            "ttft": 5000,
            "tpot": 10,
        },
    ),
]


def _build_default_cmd(
    *,
    model: str,
    system: str,
    backend: str,
    total_gpus: int,
    isl: int,
    osl: int,
    prefix: int,
    ttft: int,
    tpot: int,
):
    return [
        "aiconfigurator",
        "cli",
        "default",
        "--model",
        model,
        "--system",
        system,
        "--backend",
        backend,
        "--total_gpus",
        str(total_gpus),
        "--isl",
        str(isl),
        "--osl",
        str(osl),
        "--prefix",
        str(prefix),
        "--ttft",
        str(ttft),
        "--tpot",
        str(tpot),
    ]


@pytest.mark.parametrize("case", _DEFAULT_BUILD_CASES)
def test_cli_default_build_subset(case: dict):
    """
    Small, stable E2E subset for the GitHub PR workflow.

    This mirrors the previously hard-coded CI build selection (a small matrix of
    model/system/backend combinations) while using the reorganized CLI E2E tests.
    """

    cmd = _build_default_cmd(**case)

    completed = sp.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        combined = f"{completed.stdout}\n{completed.stderr}".strip()
        raise AssertionError(f"CLI default failed:\n{combined}")

    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert "Dynamo aiconfigurator Final Results" in combined_output
    assert f"Model: {case['model']}" in combined_output
    assert f"Total GPUs: {case['total_gpus']}" in combined_output
