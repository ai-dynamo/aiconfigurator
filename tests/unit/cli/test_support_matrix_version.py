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
    ("versions", "expected_version"),
    [
        pytest.param(
            ["0.5.9", "0.5.10"],
            "0.5.10",
            id="semantic-version-sort",
        ),
        pytest.param(
            ["", "bad-version", "0.5.10"],
            "0.5.10",
            id="ignore-invalid-versions",
        ),
        pytest.param(
            ["", "bad-version"],
            None,
            id="no-valid-versions",
        ),
    ],
)
def test_latest_support_matrix_version_selects_latest_valid_version(versions, expected_version):
    matrix = [_row(version=version) for version in versions]

    assert _latest_support_matrix_version(matrix, "b200_sxm", "sglang", model="model") == expected_version


@pytest.mark.parametrize(
    ("matrix", "system", "model", "architecture", "expected_version"),
    [
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
            "b300_sxm",
            "Qwen/Qwen3-32B",
            "Qwen3ForCausalLM",
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
            "b200_sxm",
            "local-glm5-variant",
            "GlmMoeDsaForCausalLM",
            "0.5.10",
            id="architecture-fallback",
        ),
    ],
)
def test_latest_support_matrix_version_scopes_rows_by_model_or_architecture(
    matrix,
    system,
    model,
    architecture,
    expected_version,
):
    assert (
        _latest_support_matrix_version(
            matrix,
            system,
            "sglang",
            model=model,
            architecture=architecture,
        )
        == expected_version
    )
