# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from collector.helper import create_test_case_id
from collector.model_cases import (
    CaseSelector,
    OpCasePlan,
    build_collection_case_plan,
    default_model_cases_path,
    filter_test_cases,
)


def test_model_case_plan_merges_base_model_and_framework_specific_ops():
    plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")

    assert "gemm" in plan.op_cases
    assert "attention_context" in plan.op_cases
    assert "moe" in plan.op_cases
    assert "mla_context" in plan.op_cases
    assert "wideep_mla_context" in plan.op_cases
    assert "wideep_moe" in plan.op_cases
    assert "trtllm_moe_wideep" not in plan.op_cases


def test_model_cases_path_can_infer_model_path():
    model_cases_path = default_model_cases_path("sgl-project/DeepSeek-V4-Flash-FP8")

    plan = build_collection_case_plan(backend="sglang", model_cases_path=str(model_cases_path))

    assert plan.model_path == "sgl-project/DeepSeek-V4-Flash-FP8"
    assert "dsv4_flash_csa_context_module" in plan.op_cases
    assert "mhc_module" in plan.op_cases


def test_full_mode_aggregates_all_model_case_files():
    plan = build_collection_case_plan(backend="sglang", full=True)

    assert plan.model_path is None
    assert len(plan.model_cases_paths) >= 2
    assert "wideep_mla_context" in plan.op_cases
    assert "dsv4_flash_csa_context_module" in plan.op_cases


def test_gpu_exceptions_can_drop_framework_specific_op(tmp_path: Path):
    exceptions = tmp_path / "b200_sxm_exceptions.yaml"
    exceptions.write_text(
        """
schema_version: 1
gpu_type: b200_sxm
framework_specific_op_exceptions:
  sglang:
    wideep_moe:
      drop: true
""",
        encoding="utf-8",
    )

    plan = build_collection_case_plan(
        backend="sglang",
        model_path="deepseek-ai/DeepSeek-V3",
        gpu_type="b200_sxm",
        gpu_exceptions_path=str(exceptions),
    )

    assert "wideep_moe" not in plan.op_cases
    assert "wideep_mla_context" in plan.op_cases


def test_filter_test_cases_supports_case_ids_contains_and_indices():
    cases = ["tp=1 ep=1", "tp=2 ep=1", "tp=4 ep=1", "tp=8 ep=1"]
    case_id = create_test_case_id(cases[1], "run_moe_torch", "sglang.moe")
    plan = OpCasePlan(
        include=CaseSelector(case_ids={case_id}, contains={"tp=4"}, indices={3}),
        exclude=CaseSelector(contains={"tp=8"}),
    )

    filtered = filter_test_cases(cases, plan=plan, full_module_name="sglang.moe", run_func_name="run_moe_torch")

    assert filtered == ["tp=2 ep=1", "tp=4 ep=1"]


def test_filter_test_cases_supports_index_ranges_and_limit():
    cases = ["case0", "case1", "case2", "case3"]
    plan = OpCasePlan(include=CaseSelector(index_ranges=[(1, 3)], limit=2))

    filtered = filter_test_cases(cases, plan=plan, full_module_name="vllm.gemm", run_func_name="run_gemm")

    assert filtered == ["case1", "case2"]
