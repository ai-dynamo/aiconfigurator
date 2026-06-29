# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_moe.py"


def _load_functions(*names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded


def _gptoss_case(*, tp: int, ep: int):
    return SimpleNamespace(
        num_tokens_list=[128],
        hidden_size=2880,
        inter_size=2880,
        topk=4,
        num_experts=128,
        tp=tp,
        ep=ep,
        model_name="openai/gpt-oss-120b",
        token_expert_distribution="balanced",
        power_law_alpha=None,
        architecture="GptOssForCausalLM",
    )


def _populate_gptoss_cases(cases):
    loaded = _load_functions(
        "get_moe_test_cases",
        namespace={
            "itertools": itertools,
            "get_sm_version": lambda: 100,
            "get_common_moe_test_cases": lambda: cases,
            "moe_model_allows_quantization": (lambda _backend, _model, mode: mode == "w4a8_mxfp4_mxfp8"),
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {},
            "_SM120_NEMOTRON_NVFP4_MODELS": set(),
        },
    )
    return loaded["get_moe_test_cases"]()


@pytest.mark.parametrize(("tp", "ep"), [(4, 8), (32, 1), (32, 8)])
def test_gptoss_mxfp4_population_retains_tp_and_ep_buckets(tp, ep):
    populated = _populate_gptoss_cases([_gptoss_case(tp=tp, ep=ep)])

    assert len(populated) == 1
    assert populated[0][0] == "w4a8_mxfp4_mxfp8"
    assert populated[0][6:8] == [tp, ep]
    assert populated[0][-1] is None
