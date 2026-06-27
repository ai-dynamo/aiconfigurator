# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free population reports used by ``collect.py --plan-only``."""

from __future__ import annotations

from typing import Any

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_encoder_shape_sweeps,
    get_attention_generation_shape_sweeps,
)
from collector.model_cases import select_test_case_indices_with_report
from collector.planner.compatibility import check_protected_coverage
from collector.planner.compiler import compile_population
from collector.planner.context import use_case_catalog
from collector.planner.models import PlanContext, PopulationResult, RuleSource, legacy_rule
from collector.planner.schemas.attention import (
    build_attention_context_cases,
    build_attention_generation_cases,
    build_encoder_attention_cases,
    register_attention_schemas,
)

_ATTENTION_OPS = frozenset({"attention_context", "attention_generation", "encoder_attention"})
_RUN_FUNC_NAMES = {
    "attention_context": "run_attention_torch",
    "attention_generation": "run_attention_torch",
    "encoder_attention": "run_encoder_attention_torch",
}
_PERF_FILES = {
    "attention_context": "context_attention_perf.txt",
    "attention_generation": "generation_attention_perf.txt",
    "encoder_attention": "encoder_attention_perf.txt",
}


def _attention_cases(op: str, backend: str, sm_version: int, framework_version: str) -> list[Any]:
    if op == "attention_context":
        return build_attention_context_cases(
            backend,
            get_attention_context_shape_sweeps(backend),
            sm_version=sm_version,
            framework_version=framework_version,
        )
    if op == "attention_generation":
        return build_attention_generation_cases(
            backend,
            get_attention_generation_shape_sweeps(backend),
            sm_version=sm_version,
            framework_version=framework_version,
        )
    return build_encoder_attention_cases(
        backend,
        get_attention_encoder_shape_sweeps(backend),
        sm_version=sm_version,
        framework_version=framework_version,
    )


def build_plan_only_population_reports(
    case_plan,
    ops: list[str],
    *,
    framework_version: str,
) -> dict[str, dict[str, Any]]:
    """Return exact reports for pure schemas and explicit passthrough markers."""

    reports: dict[str, dict[str, Any]] = {}
    register_attention_schemas()
    backend = case_plan.backend
    framework_backend = "vllm" if backend == "vllm_xpu" else backend
    for op in ops:
        if op not in _ATTENTION_OPS:
            reports[op] = {"status": "legacy_passthrough"}
            continue
        if op == "encoder_attention" and backend == "vllm_xpu":
            reports[op] = {"status": "unsupported_backend"}
            continue
        if case_plan.sm_version is None:
            reports[op] = {"status": "requires_gpu_or_sm"}
            continue

        op_plan = case_plan.op_cases[op]
        context = PlanContext(
            backend=backend,
            op=op,
            perf_file=_PERF_FILES[op],
            schema_version="planner-v1",
            model_path=case_plan.model_path,
            model_architecture=case_plan.model_architecture,
            gpu_type=case_plan.gpu_type,
            sm_version=case_plan.sm_version,
            framework_version=framework_version,
            attributes={"framework_version": framework_version, "plan_only": True},
        )
        with use_case_catalog(case_plan.catalog, model_path=case_plan.model_path):
            legacy_cases = _attention_cases(op, backend, case_plan.sm_version, framework_version)

        module_name = f"{framework_backend}.{op}"
        run_func_name = _RUN_FUNC_NAMES[op]
        rules = [
            legacy_rule(
                legacy_cases,
                source=RuleSource(
                    rule_id=f"legacy:pure_attention.{op}",
                    model_path=case_plan.model_path,
                    backend=backend,
                    sm_version=case_plan.sm_version,
                ),
            )
        ]
        rules.extend(op_plan.population_rules)

        population = compile_population(rules, context, record_decisions=False)
        selected_indices, skipped = select_test_case_indices_with_report(
            [planned.payload for planned in population.cases],
            plan=op_plan,
            full_module_name=module_name,
            run_func_name=run_func_name,
            runtime_version=framework_version,
        )
        selector_filtered = len(population.cases) - len(selected_indices) - len(skipped)
        selected_population = PopulationResult(
            cases=tuple(population.cases[index] for index in selected_indices),
            report=population.report,
        )
        report = population.report.to_dict()
        report.update(
            {
                "status": "physical_row_exact",
                "selected": len(selected_indices),
                "selector_skipped": len(skipped),
                "selector_filtered": selector_filtered,
                "protected_coverage": (
                    check_protected_coverage(selected_population, context)
                    if case_plan.full
                    else {"status": "targeted_plan_not_enforced"}
                ),
            }
        )
        reports[op] = report
    return reports


__all__ = ["build_plan_only_population_reports"]
