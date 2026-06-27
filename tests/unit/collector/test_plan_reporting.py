# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from collector.model_cases import build_collection_case_plan
from collector.planner.coverage import CoverageMismatchError
from collector.planner.reporting import build_plan_only_population_reports


def test_plan_only_reports_exact_targeted_attention_without_framework_imports():
    plan = build_collection_case_plan(
        backend="sglang",
        model_path="Qwen/Qwen3-32B-FP8",
        sm_version=100,
    )

    reports = build_plan_only_population_reports(
        plan,
        ["attention_context", "attention_generation", "gemm"],
        framework_version="0.5.10",
    )

    assert reports["attention_context"]["status"] == "physical_row_exact"
    assert reports["attention_context"]["scheduled"] == 2448
    assert reports["attention_context"]["protected_coverage"] == {"status": "targeted_plan_not_enforced"}
    assert reports["attention_generation"]["scheduled"] == 2354
    assert reports["gemm"] == {"status": "legacy_passthrough"}


def test_plan_only_reports_require_explicit_sm_for_hardware_sensitive_schema():
    plan = build_collection_case_plan(
        backend="vllm",
        model_path="Qwen/Qwen3-32B-FP8",
    )

    reports = build_plan_only_population_reports(
        plan,
        ["attention_context"],
        framework_version="0.19.0",
    )

    assert reports == {"attention_context": {"status": "requires_gpu_or_sm"}}


def test_xpu_plan_uses_xpu_variant_and_never_claims_encoder_support():
    plan = build_collection_case_plan(
        backend="vllm_xpu",
        model_path="Qwen/Qwen3-32B-FP8",
        sm_version=0,
    )

    reports = build_plan_only_population_reports(
        plan,
        ["attention_context", "attention_generation", "encoder_attention"],
        framework_version="0.19.0",
    )

    assert reports["attention_context"]["scheduled"] == 1302
    assert reports["attention_generation"]["scheduled"] == 2354
    assert reports["encoder_attention"] == {"status": "unsupported_backend"}


def test_full_plan_hard_checks_exact_v1_physical_subset():
    plan = build_collection_case_plan(
        backend="sglang",
        gpu_type="b200_sxm",
        sm_version=100,
        full=True,
    )

    reports = build_plan_only_population_reports(
        plan,
        ["attention_context"],
        framework_version="0.5.10",
    )

    assert reports["attention_context"]["scheduled"] == 50901
    assert reports["attention_context"]["protected_coverage"] == {
        "status": "preserved",
        "source_git_ref": "a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a",
        "legacy_physical_keys": 33714,
        "retained_legacy_keys": 33714,
        "added_physical_keys": 17187,
        "removed_protected_keys": 0,
        "generated_physical_keys": 50901,
    }


def test_full_xpu_plan_uses_canonical_scope_for_protected_subset():
    plan = build_collection_case_plan(
        backend="vllm_xpu",
        gpu_type="xpu",
        sm_version=0,
        full=True,
    )

    reports = build_plan_only_population_reports(
        plan,
        ["attention_context"],
        framework_version="0.19.0",
    )

    assert reports["attention_context"]["protected_coverage"]["status"] == "preserved"
    assert reports["attention_context"]["protected_coverage"]["removed_protected_keys"] == 0


def test_encoder_manifest_applies_to_later_compatible_framework_version():
    plan = build_collection_case_plan(
        backend="trtllm",
        gpu_type="b200_sxm",
        sm_version=100,
        full=True,
    )

    reports = build_plan_only_population_reports(
        plan,
        ["encoder_attention"],
        framework_version="1.3.0rc10",
    )

    coverage = reports["encoder_attention"]["protected_coverage"]
    assert coverage["status"] == "preserved"
    assert coverage["legacy_physical_keys"] == 7008
    assert coverage["generated_physical_keys"] == 7679
    assert coverage["removed_protected_keys"] == 0


def test_encoder_manifest_excludes_framework_without_collector_support():
    plan = build_collection_case_plan(
        backend="sglang",
        gpu_type="b200_sxm",
        sm_version=100,
        full=True,
    )

    reports = build_plan_only_population_reports(
        plan,
        ["encoder_attention"],
        framework_version="0.5.10",
    )

    coverage = reports["encoder_attention"]["protected_coverage"]
    assert coverage["status"] == "out_of_scope"
    assert coverage["scope_mismatches"]["framework_version"] == {
        "actual": "0.5.10",
        "manifest": ">=0.5.11",
    }


def test_full_plan_checks_coverage_after_selectors():
    plan = build_collection_case_plan(
        backend="sglang",
        gpu_type="b200_sxm",
        sm_version=100,
        full=True,
    )
    plan.op_cases["attention_context"].include.limit = 1

    with pytest.raises(CoverageMismatchError, match=r"missing .* legacy physical key"):
        build_plan_only_population_reports(
            plan,
            ["attention_context"],
            framework_version="0.5.10",
        )
