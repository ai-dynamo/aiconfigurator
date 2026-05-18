# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

from collector import common_test_cases

pytestmark = pytest.mark.unit


_FLASH = "deepseek-ai/DeepSeek-V4-Flash"
_PRO = "deepseek-ai/DeepSeek-V4-Pro"


def test_dsv4_attn_cases_cover_default_models(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    cases = common_test_cases.get_dsv4_csa_context_test_cases()

    assert {case[6] for case in cases} == {_FLASH, _PRO}
    assert {case[7] for case in cases} == {"csa"}


def test_dsv4_attn_cases_include_same_gemm_types_for_default_models(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    cases = common_test_cases.get_dsv4_csa_context_test_cases()

    assert cases
    assert {model_path: {case[5] for case in cases if case[6] == model_path} for model_path in (_FLASH, _PRO)} == {
        _FLASH: {"bfloat16", "fp8_block"},
        _PRO: {"bfloat16", "fp8_block"},
    }


def test_dsv4_attn_cases_honor_model_filter(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _PRO)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    cases = common_test_cases.get_dsv4_hca_generation_test_cases()

    assert cases
    assert {case[6] for case in cases} == {_PRO}
    assert {case[7] for case in cases} == {"hca"}


def test_dsv4_sparse_smoke_cases_cover_default_models(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--smoke"])

    cases = common_test_cases.get_dsv4_paged_mqa_logits_test_cases()

    assert cases == [
        [1, 1024, 8192, 1, "paged_mqa_logits", _FLASH],
        [1, 1024, 8192, 1, "paged_mqa_logits", _PRO],
    ]


def test_dsv4_cases_skip_unrelated_model_filter(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "MiniMaxAI/MiniMax-M2.5")
    monkeypatch.setattr(sys, "argv", ["pytest"])

    assert common_test_cases.get_dsv4_csa_context_test_cases() == []
    assert common_test_cases.get_dsv4_hca_attn_test_cases() == []
