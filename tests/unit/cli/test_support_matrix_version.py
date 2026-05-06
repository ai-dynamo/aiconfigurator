# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.cli.main import _latest_support_matrix_version

pytestmark = pytest.mark.unit


def _row(
    *,
    model: str = "model",
    architecture: str = "Arch",
    system: str = "b200_sxm",
    backend: str = "sglang",
    version: str,
) -> dict[str, str]:
    return {
        "HuggingFaceID": model,
        "Architecture": architecture,
        "System": system,
        "Backend": backend,
        "Version": version,
    }


@pytest.mark.parametrize(
    ("matrix", "kwargs", "expected"),
    [
        pytest.param(
            [_row(version="0.5.9"), _row(version="0.5.10")],
            {},
            "0.5.10",
            id="semantic-version-sort",
        ),
        pytest.param(
            [_row(version=""), _row(version="bad-version"), _row(version="0.5.10")],
            {},
            "0.5.10",
            id="ignore-invalid-versions",
        ),
        pytest.param(
            [_row(version=""), _row(version="bad-version")],
            {},
            None,
            id="no-valid-versions",
        ),
        pytest.param(
            [
                _row(
                    model="Qwen/Qwen3-32B",
                    architecture="Qwen3ForCausalLM",
                    system="b300_sxm",
                    version="0.5.9",
                ),
                _row(
                    model="zai-org/GLM-5",
                    architecture="GlmMoeDsaForCausalLM",
                    system="b300_sxm",
                    version="0.5.10",
                ),
            ],
            {
                "system": "b300_sxm",
                "model": "Qwen/Qwen3-32B",
                "architecture": "Qwen3ForCausalLM",
            },
            "0.5.9",
            id="prefer-exact-model",
        ),
        pytest.param(
            [
                _row(
                    model="deepseek-ai/DeepSeek-V3.2",
                    architecture="GlmMoeDsaForCausalLM",
                    version="0.5.9",
                ),
                _row(
                    model="zai-org/GLM-5-FP8",
                    architecture="GlmMoeDsaForCausalLM",
                    version="0.5.10",
                ),
            ],
            {"model": "local-glm5-variant", "architecture": "GlmMoeDsaForCausalLM"},
            "0.5.10",
            id="architecture-fallback",
        ),
    ],
)
def test_latest_support_matrix_version(matrix, kwargs, expected):
    options = {"system": "b200_sxm", "backend": "sglang", "model": "model"} | kwargs

    assert _latest_support_matrix_version(matrix, **options) == expected
