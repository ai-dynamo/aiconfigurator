# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from collector import case_generator

pytestmark = pytest.mark.unit

LAGUNA_MODEL_PATH = "poolside/Laguna-S-2.1-FP8"


@pytest.fixture(autouse=True)
def _laguna_model_filter(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", LAGUNA_MODEL_PATH)
    case_generator._load_model_cases_data.cache_clear()
    yield
    case_generator._load_model_cases_data.cache_clear()


def test_laguna_moe_quantization_matches_official_artifact():
    assert case_generator.moe_model_allows_quantization("vllm", LAGUNA_MODEL_PATH, "fp8_block")
    assert not case_generator.moe_model_allows_quantization("vllm", LAGUNA_MODEL_PATH, "fp8")
    assert not case_generator.moe_model_allows_quantization("vllm", LAGUNA_MODEL_PATH, "bfloat16")
    cases = case_generator.get_common_moe_test_cases(backend="vllm")
    assert len(cases) == 36
    assert sum(len(case.num_tokens_list) for case in cases) == 972
    assert all(case.tp < 16 for case in cases)
    assert {
        (1, 1),
        (1, 2),
        (1, 4),
        (2, 1),
        (4, 1),
    }.issubset({(case.tp, case.ep) for case in cases})


@pytest.mark.parametrize(
    ("phase", "shape_sweep"),
    [
        pytest.param(
            "context",
            lambda: case_generator.get_attention_context_shape_sweeps("vllm")[0],
            id="context",
        ),
        pytest.param(
            "generation",
            lambda: case_generator.get_attention_generation_shape_sweeps("vllm")[0],
            id="generation",
        ),
    ],
)
def test_laguna_attention_profiles_expand_only_native_tp_shapes(phase, shape_sweep):
    configs = case_generator.get_attention_head_configs(shape_sweep(), phase=phase)
    shapes = {(config.num_heads, config.num_kv_heads, config.head_dim, config.window_size) for config in configs}

    assert shapes == {
        (48, 8, 128, 0),
        (24, 4, 128, 0),
        (12, 2, 128, 0),
        (6, 1, 128, 0),
        (72, 8, 128, 512),
        (36, 4, 128, 512),
        (18, 2, 128, 512),
        (9, 1, 128, 512),
    }


def test_laguna_attention_profiles_do_not_expand_sglang():
    shape_sweep = case_generator.get_attention_context_shape_sweeps("sglang")[0]

    assert (
        case_generator.get_attention_head_configs(shape_sweep, phase="context", backend="sglang", sm_version=90) == []
    )
