# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.cli.main import _latest_support_matrix_version

pytestmark = pytest.mark.unit


def test_latest_support_matrix_version_uses_semantic_sorting():
    matrix = [
        {
            "HuggingFaceID": "model",
            "Architecture": "Arch",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "0.5.9",
        },
        {
            "HuggingFaceID": "model",
            "Architecture": "Arch",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "0.5.10",
        },
    ]

    assert _latest_support_matrix_version(matrix, "b200_sxm", "sglang", model="model") == "0.5.10"


def test_latest_support_matrix_version_ignores_invalid_versions():
    matrix = [
        {
            "HuggingFaceID": "model",
            "Architecture": "Arch",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "",
        },
        {
            "HuggingFaceID": "model",
            "Architecture": "Arch",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "bad-version",
        },
        {
            "HuggingFaceID": "model",
            "Architecture": "Arch",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "0.5.10",
        },
    ]

    assert _latest_support_matrix_version(matrix, "b200_sxm", "sglang", model="model") == "0.5.10"


def test_latest_support_matrix_version_returns_none_without_valid_versions():
    matrix = [
        {
            "HuggingFaceID": "model",
            "Architecture": "Arch",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "",
        },
        {
            "HuggingFaceID": "model",
            "Architecture": "Arch",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "bad-version",
        },
    ]

    assert _latest_support_matrix_version(matrix, "b200_sxm", "sglang", model="model") is None


def test_latest_support_matrix_version_prefers_exact_model_rows():
    matrix = [
        {
            "HuggingFaceID": "Qwen/Qwen3-32B",
            "Architecture": "Qwen3ForCausalLM",
            "System": "b300_sxm",
            "Backend": "sglang",
            "Version": "0.5.9",
        },
        {
            "HuggingFaceID": "zai-org/GLM-5",
            "Architecture": "GlmMoeDsaForCausalLM",
            "System": "b300_sxm",
            "Backend": "sglang",
            "Version": "0.5.10",
        },
    ]

    assert (
        _latest_support_matrix_version(
            matrix,
            "b300_sxm",
            "sglang",
            model="Qwen/Qwen3-32B",
            architecture="Qwen3ForCausalLM",
        )
        == "0.5.9"
    )


def test_latest_support_matrix_version_uses_architecture_when_exact_rows_are_missing():
    matrix = [
        {
            "HuggingFaceID": "deepseek-ai/DeepSeek-V3.2",
            "Architecture": "GlmMoeDsaForCausalLM",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "0.5.9",
        },
        {
            "HuggingFaceID": "zai-org/GLM-5-FP8",
            "Architecture": "GlmMoeDsaForCausalLM",
            "System": "b200_sxm",
            "Backend": "sglang",
            "Version": "0.5.10",
        },
    ]

    assert (
        _latest_support_matrix_version(
            matrix,
            "b200_sxm",
            "sglang",
            model="local-glm5-variant",
            architecture="GlmMoeDsaForCausalLM",
        )
        == "0.5.10"
    )
