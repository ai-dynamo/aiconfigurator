# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.cli.main import _latest_support_matrix_version, _run_support_mode

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


def test_latest_support_matrix_version_matches_system_and_backend_case_insensitively():
    matrix = [_row(version="0.5.10")]

    assert _latest_support_matrix_version(matrix, "B200_SXM", "SGLang", model="model") == "0.5.10"


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


def test_latest_support_matrix_version_does_not_fall_back_to_unrelated_rows():
    matrix = [
        _row(model="Qwen/Qwen3-32B", architecture="Qwen3ForCausalLM", version="0.5.9"),
        _row(model="zai-org/GLM-5-FP8", architecture="GlmMoeDsaForCausalLM", version="0.5.10"),
    ]

    assert (
        _latest_support_matrix_version(
            matrix,
            "b200_sxm",
            "sglang",
            model="local-unknown-model",
            architecture="UnknownArchitecture",
        )
        is None
    )


def test_run_support_mode_stops_when_auto_version_is_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        "aiconfigurator.cli.main.get_model_config_from_model_path",
        lambda _model: {"architecture": "UnknownArchitecture"},
    )
    monkeypatch.setattr(
        "aiconfigurator.cli.main.common.get_support_matrix",
        lambda: [_row(model="Qwen/Qwen3-32B", architecture="Qwen3ForCausalLM", version="0.5.10")],
    )

    def fail_check_support(*_args, **_kwargs):
        raise AssertionError("check_support should not run without an auto-selected version")

    monkeypatch.setattr("aiconfigurator.cli.main.common.check_support", fail_check_support)

    _run_support_mode(
        SimpleNamespace(
            backend="sglang",
            backend_version=None,
            model_path="local-unknown-model",
            system="b200_sxm",
        )
    )

    assert "No valid support-matrix backend version found" in capsys.readouterr().out
