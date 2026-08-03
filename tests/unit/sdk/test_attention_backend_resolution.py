# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk.attention_backend import resolve_wideep_mla_attention_backend

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("requested", "version", "sm_version", "expected"),
    [
        (None, "0.5.10", 90, "fa3"),
        ("auto", "0.5.10", 100, "trtllm_mla"),
        (None, "v0.5.12", 103, "trtllm_mla"),
        (None, "0.5.14", 90, "fa3"),
    ],
)
def test_sglang_auto_resolution_is_versioned_and_architecture_aware(requested, version, sm_version, expected):
    resolved = resolve_wideep_mla_attention_backend(
        requested,
        framework="sglang",
        framework_version=version,
        model_family="DEEPSEEK",
        sm_version=sm_version,
    )

    assert resolved.effective == expected
    assert resolved.source == "framework_default"


@pytest.mark.parametrize("requested", ["flashinfer", "fa3", "trtllm_mla"])
def test_explicit_backend_is_never_substituted_from_available_data(requested):
    resolved = resolve_wideep_mla_attention_backend(
        requested,
        framework="sglang",
        framework_version="unknown",
        model_family="DEEPSEEK",
        sm_version=100,
    )

    assert resolved.effective == requested
    assert resolved.source == "explicit"


def test_unknown_auto_policy_fails_closed():
    with pytest.raises(ValueError, match="No reviewed SGLang DeepSeek WideEP MLA default"):
        resolve_wideep_mla_attention_backend(
            None,
            framework="sglang",
            framework_version="9.9.9",
            model_family="DEEPSEEK",
            sm_version=100,
        )


def test_invalid_explicit_backend_fails_closed():
    with pytest.raises(ValueError, match="Unsupported WideEP MLA attention backend"):
        resolve_wideep_mla_attention_backend(
            "torch",
            framework="sglang",
            framework_version="0.5.10",
            model_family="DEEPSEEK",
            sm_version=100,
        )
