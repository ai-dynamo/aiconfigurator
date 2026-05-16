# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

from collector.helper import create_test_case_id
from collector.model_cases import (
    CaseSelector,
    OpCasePlan,
    build_collection_case_plan,
    default_architecture_cases_path,
    filter_test_cases,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SUPPORT_MATRIX_ROOT = REPO_ROOT / "src" / "aiconfigurator" / "systems" / "support_matrix"


def test_model_case_plan_merges_base_model_and_framework_specific_ops():
    plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")

    assert plan.model_architecture == "DeepseekV3ForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("DeepseekV3ForCausalLM")]
    assert "gemm" in plan.op_cases
    assert "attention_context" in plan.op_cases
    assert "moe" in plan.op_cases
    assert "mla_context" in plan.op_cases
    assert "wideep_mla_context" in plan.op_cases
    assert "wideep_moe" in plan.op_cases
    assert "trtllm_moe_wideep" not in plan.op_cases


def test_base_gemm_cases_are_readable_shape_specs():
    plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")
    gemm_plan = plan.op_cases["gemm"]
    context_plan = plan.op_cases["attention_context"]
    generation_plan = plan.op_cases["attention_generation"]

    assert len(gemm_plan.include.case_specs) == 1
    spec = gemm_plan.include.case_specs[0]
    assert spec["id"] == "base_transformer_gemm_shape_sweep"
    assert spec["token_counts"][:5] == [1, 2, 3, 4, 5]
    assert spec["feature_sizes"][:3] == [32, 64, 128]

    assert context_plan.include.case_specs[0]["id"] == "base_attention_context_shape_sweep"
    assert context_plan.include.case_specs[0]["kv_head_options"] == ["self", 1, 2, 4, 8]
    assert generation_plan.include.case_specs[0]["id"] == "base_attention_generation_shape_sweep"
    assert generation_plan.include.case_specs[0]["xqa_query_head_counts"][-1] == 128

    filtered = filter_test_cases(
        ["case0", "case1"], plan=gemm_plan, full_module_name="sglang.gemm", run_func_name="run_gemm"
    )

    assert filtered == ["case0", "case1"]


def test_gemm_common_cases_expand_from_base_yaml_shape_specs():
    from collector.common_test_cases import (
        ComputeScaleCommonTestCase,
        GemmCommonTestCase,
        get_compute_scale_common_test_cases,
        get_gemm_common_test_cases,
    )

    cases = get_gemm_common_test_cases()

    assert len(cases) == 35742
    assert cases[0] == GemmCommonTestCase(x=32768, n=65536, k=51200)
    assert cases[-1] == GemmCommonTestCase(x=1, n=32, k=32)
    assert not any(case.n == 65536 and case.k == 65536 for case in cases)

    compute_scale_cases = get_compute_scale_common_test_cases()
    assert len(compute_scale_cases) == 1628
    assert compute_scale_cases[0] == ComputeScaleCommonTestCase(m=32768, k=51200)
    assert compute_scale_cases[-1] == ComputeScaleCommonTestCase(m=1, k=65536)


def test_model_cases_path_can_infer_model_path():
    model_cases_path = default_architecture_cases_path("DeepseekV4ForCausalLM")

    plan = build_collection_case_plan(backend="sglang", model_cases_path=str(model_cases_path))

    assert plan.model_path == "sgl-project/DeepSeek-V4-Flash-FP8"
    assert plan.model_architecture == "DeepseekV4ForCausalLM"
    assert "dsv4_flash_csa_context_module" in plan.op_cases
    assert "mhc_module" in plan.op_cases


def test_model_architecture_can_select_case_file():
    plan = build_collection_case_plan(backend="trtllm", model_architecture="Qwen3MoeForCausalLM")

    assert plan.model_path == "Qwen/Qwen3-235B-A22B"
    assert plan.model_architecture == "Qwen3MoeForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Qwen3MoeForCausalLM")]
    assert "moe" in plan.op_cases


def test_model_path_alias_resolves_architecture_case_file():
    plan = build_collection_case_plan(backend="trtllm", model_path="Qwen/Qwen3-235B-A22B-FP8")

    assert plan.model_path == "Qwen/Qwen3-235B-A22B-FP8"
    assert plan.model_architecture == "Qwen3MoeForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Qwen3MoeForCausalLM")]
    assert "moe" in plan.op_cases


def test_full_mode_aggregates_all_model_case_files():
    plan = build_collection_case_plan(backend="sglang", full=True)

    assert plan.model_path is None
    assert len(plan.model_cases_paths) >= 18
    assert "wideep_mla_context" in plan.op_cases
    assert "dsv4_flash_csa_context_module" in plan.op_cases
    assert "gdn" in plan.op_cases


def test_support_matrix_models_have_model_case_aliases():
    case_aliases = set()
    for path in (REPO_ROOT / "collector" / "cases" / "models").glob("*_cases.yaml"):
        data = path.read_text(encoding="utf-8")
        for line in data.splitlines():
            stripped = line.strip()
            if stripped.startswith("model_path: "):
                case_aliases.add(stripped.removeprefix("model_path: ").strip())
            elif stripped.startswith("- "):
                case_aliases.add(stripped.removeprefix("- ").strip())

    support_matrix_models = set()
    for path in SUPPORT_MATRIX_ROOT.glob("*.csv"):
        with path.open(encoding="utf-8") as f:
            support_matrix_models.update(row["HuggingFaceID"] for row in csv.DictReader(f))

    assert support_matrix_models <= case_aliases


def test_support_matrix_moe_alias_generates_targeted_cases(monkeypatch):
    from collector.common_test_cases import get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-235B-A22B-FP8")

    cases = get_common_moe_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {"Qwen/Qwen3-235B-A22B-FP8"}


def test_support_matrix_mamba_alias_generates_targeted_cases(monkeypatch):
    from collector.common_test_cases import get_common_mamba2_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")

    cases = get_common_mamba2_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"}


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
