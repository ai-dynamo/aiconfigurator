# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]


def _function(source_path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _assigned_torch_call(function: ast.FunctionDef, variable: str) -> list[ast.Call]:
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == variable for target in node.targets):
            continue
        if isinstance(node.value, ast.Call):
            calls.append(node.value)
    return calls


def _dtype_expression(call: ast.Call) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "dtype":
            return ast.unparse(keyword.value)
    return None


@pytest.mark.parametrize(
    "function_name",
    ("run_gdn_context_benchmark", "run_gdn_generation_benchmark"),
)
def test_vllm_gdn_temporal_state_matches_qwen35_fp32_cache(function_name):
    """All Qwen3.5 checkpoints declare mamba_ssm_dtype=float32, and vLLM's
    Qwen3_5ForConditionalGenerationConfig.verify_and_update_config
    (vllm/model_executor/models/config.py:603-628 @ v0.24.0) adopts it when
    --mamba-ssm-cache-dtype is left at 'auto' — serving ssm state is fp32."""
    source_path = REPO_ROOT / "collector" / "vllm" / "collect_gdn.py"
    calls = _assigned_torch_call(_function(source_path, function_name), "gdn_state")

    assert len(calls) == 1
    assert _dtype_expression(calls[0]) == "torch.float32"


def test_sglang_gdn_generation_matches_bf16_model_dt_bias():
    """sglang's Qwen3_5GatedDeltaNet creates dt_bias without an explicit
    dtype (python/sglang/srt/models/qwen3_5.py:237-239 @ v0.5.14), so the
    parameter follows the model dtype (bf16), not fp32."""
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_gdn.py"
    function = _function(source_path, "run_gdn_generation_benchmark")
    calls = _assigned_torch_call(function, "dt_bias")

    assert len(calls) == 1
    assert _dtype_expression(calls[0]) == "dtype"
