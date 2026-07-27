# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    ("relative_path", "function_name", "expected_guards"),
    [
        ("collector/sglang/collect_attn.py", "run_attention_torch", 1),
        ("collector/sglang/collect_attn_encoder.py", "run_encoder_attention_torch", 1),
        ("collector/sglang/collect_computescale.py", "run_computescale", 2),
        ("collector/sglang/collect_gemm.py", "run_gemm", 1),
        ("collector/sglang/collect_mhc_module.py", "_log_result", 1),
        ("collector/sglang/collect_mla.py", "run_mla", 1),
        ("collector/sglang/collect_mla_bmm.py", "run_mla_gen_pre", 1),
        ("collector/sglang/collect_mla_bmm.py", "run_mla_gen_post", 1),
    ],
)
def test_log_perf_false_directly_raises(relative_path, function_name, expected_guards):
    source_path = REPO_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)

    log_perf_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "log_perf"
    ]
    fail_closed_guards = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Call)
        and isinstance(node.test.operand.func, ast.Name)
        and node.test.operand.func.id == "log_perf"
    ]

    assert len(log_perf_calls) == expected_guards
    assert len(fail_closed_guards) == len(log_perf_calls)
    assert all(any(isinstance(statement, ast.Raise) for statement in guard.body) for guard in fail_closed_guards)


@pytest.mark.parametrize(
    "relative_path",
    [
        "collector/sglang/collect_gdn.py",
        "collector/sglang/collect_moe.py",
    ],
)
def test_benchmarks_do_not_silently_fall_back_from_cuda_graph(relative_path):
    source_path = REPO_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id != "benchmark_with_power":
            continue
        allow_graph_fail = next((kw.value for kw in call.keywords if kw.arg == "allow_graph_fail"), None)
        assert not (isinstance(allow_graph_fail, ast.Constant) and allow_graph_fail.value is True)


@pytest.mark.parametrize(
    ("relative_path", "expected_sources"),
    [
        (
            "collector/sglang/collect_gemm.py",
            {
                "sglang_torch_linear",
                "sglang_sgl_kernel_fp8_scaled_mm",
                "sglang_deepgemm_gemm_nt_f8f8bf16",
            },
        ),
        (
            "collector/sglang/collect_mla_bmm.py",
            {"sglang_torch_bmm", "sglang_sgl_kernel_bmm_fp8"},
        ),
        (
            "collector/sglang/collect_mhc_module.py",
            {
                "sglang_tilelang_mhc_pre",
                "sglang_deepgemm_mhc_pre",
                "sglang_torch_mhc_pre",
                "sglang_tilelang_mhc_post",
                "sglang_torch_mhc_post",
            },
        ),
    ],
)
def test_kernel_source_names_the_invoked_sglang_path(relative_path, expected_sources):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert all(value in source for value in expected_sources)
