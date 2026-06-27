# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import pytest

from collector.planner import (
    InvocationKey,
    LogicalCandidate,
    PlanContext,
    PopulationConflictError,
    PopulationRule,
    RuleSource,
    canonical_json,
    clear_schemas,
    compile_population,
    get_active_case_catalog,
    get_active_model_path,
    get_schema,
    legacy_passthrough,
    legacy_rule,
    register_schema,
    use_case_catalog,
)


class Precision(str, Enum):
    BF16 = "bf16"


@dataclass(frozen=True)
class Shape:
    heads: int
    precision: Precision


def _context(**changes):
    values = {
        "backend": "sglang",
        "op": "attention_context",
        "perf_file": "context_attention_perf.txt",
        "schema_version": "planner-v1",
    }
    values.update(changes)
    return PlanContext(**values)


def test_canonical_json_normalizes_supported_container_and_value_types():
    left = {
        "path": Path("collector/cases"),
        "shape": Shape(heads=32, precision=Precision.BF16),
        "dimensions": (128, [0, 4096]),
        "tps": {8, 1, 4},
    }
    right = {
        "tps": {4, 8, 1},
        "dimensions": [128, (0, 4096)],
        "shape": {"precision": "bf16", "heads": 32},
        "path": "collector/cases",
    }

    assert canonical_json(left) == canonical_json(right)
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(float("nan"))


def test_invocation_id_is_stable_and_scoped_by_execution_contract():
    context = _context()
    first = InvocationKey.from_value(context, {"window": 0, "heads": 32})
    reordered = InvocationKey.from_value(context, {"heads": 32, "window": 0})

    assert first == reordered
    assert first.id == reordered.id
    for field_name, value in (
        ("backend", "vllm"),
        ("op", "attention_generation"),
        ("perf_file", "generation_attention_perf.txt"),
        ("schema_version", "planner-v2"),
        ("framework_version", "0.6.0"),
        ("gpu_type", "B200"),
        ("sm_version", 100),
        ("model_path", "model/B"),
        ("model_architecture", "ArchitectureB"),
        ("plan_fingerprint", "catalog-b"),
    ):
        changed = InvocationKey.from_value(replace(context, **{field_name: value}), {"heads": 32, "window": 0})
        assert changed.id != first.id


class ShapeSchema:
    def normalize(self, candidate, context):
        payload = dict(candidate.payload)
        payload["precision"] = payload["precision"].lower()
        payload["artifact"] = payload["artifact"].removesuffix("-fp8")
        return LogicalCandidate(payload=payload, source=candidate.source, metadata=candidate.metadata)

    def is_reachable(self, candidate, context):
        if candidate.payload["heads"] % candidate.payload["tp"]:
            return False, "heads must be divisible by TP"
        return True

    def is_supported(self, candidate, context):
        if candidate.payload["precision"] == "nvfp4" and context.sm_version != 100:
            return False, "NVFP4 requires SM100"
        return True

    def invocation_key(self, candidate, context):
        # Artifact aliases do not affect this shape-only microbenchmark.
        return {key: value for key, value in candidate.payload.items() if key != "artifact"}


def test_population_filters_lazily_and_merges_ordered_provenance_for_duplicates():
    expanded = []

    def first_rule(context):
        expanded.append("first")
        yield {"artifact": "qwen", "heads": 32, "tp": 8, "precision": "BF16"}
        yield {"artifact": "qwen", "heads": 30, "tp": 8, "precision": "BF16"}

    def second_rule(context):
        # Rules are called independently; the compiler cannot cross their
        # candidate dimensions into an accidental Cartesian product.
        expanded.append("second")
        yield {"artifact": "qwen-fp8", "heads": 32, "tp": 8, "precision": "bf16"}
        yield {"artifact": "qwen", "heads": 32, "tp": 8, "precision": "nvfp4"}

    rules = [
        PopulationRule(RuleSource("base"), first_rule),
        PopulationRule(RuleSource("model_delta"), second_rule),
    ]
    result = compile_population(rules, _context(sm_version=90), schema=ShapeSchema())

    assert expanded == ["first", "second"]
    assert result.report.counts == {
        "expanded": 4,
        "scheduled": 1,
        "duplicate_invocations": 1,
        "unreachable": 1,
        "unsupported": 1,
    }
    assert [source.rule_id for source in result.cases[0].provenance] == ["base", "model_delta"]
    assert [decision.outcome for decision in result.report.decisions] == [
        "scheduled",
        "unreachable",
        "duplicate_invocation",
        "unsupported",
    ]
    assert result.report.decisions[1].reason == "heads must be divisible by TP"
    assert result.report.decisions[3].reason == "NVFP4 requires SM100"


def test_legacy_raw_case_adapter_uses_passthrough_fallback_and_ordered_dedupe():
    context = _context(op="unmigrated_op", perf_file="legacy_perf.txt")
    result = compile_population([legacy_rule([(1, 2), (3, 4), [1, 2]])], context)

    assert get_schema(context.op, backend=context.backend) is legacy_passthrough
    assert [case.payload for case in result.cases] == [(1, 2), (3, 4)]
    assert result.report.expanded == 3
    assert result.report.scheduled == 2
    assert result.report.duplicate_invocations == 1


def test_duplicate_invocation_merges_metadata_and_rejects_conflicts():
    context = _context(op="unmigrated_op", perf_file="legacy_perf.txt")
    first = LogicalCandidate([1, 2], RuleSource("first"), metadata={"expected": "oom"})
    compatible = LogicalCandidate([1, 2], RuleSource("second"), metadata={"owner": "sm100"})

    result = compile_population([legacy_rule([first, compatible])], context)

    assert result.cases[0].candidate.metadata == {"expected": "oom", "owner": "sm100"}
    conflicting = LogicalCandidate([1, 2], RuleSource("third"), metadata={"expected": "illegal_access"})
    with pytest.raises(PopulationConflictError, match=r"conflicting metadata.*expected"):
        compile_population([legacy_rule([first, conflicting])], context)


def test_registry_prefers_backend_specific_schema():
    generic = ShapeSchema()
    backend_specific = ShapeSchema()
    clear_schemas()
    try:
        register_schema("attention", generic)
        register_schema("attention", backend_specific, backend="vllm")

        assert get_schema("attention", backend="sglang") is generic
        assert get_schema("attention", backend="vllm") is backend_specific
        with pytest.raises(ValueError, match="already registered"):
            register_schema("attention", generic)
    finally:
        clear_schemas()


def test_active_case_catalog_and_model_path_are_context_local_and_restored():
    @dataclass
    class Catalog:
        model_path: str
        op_cases: dict

    catalog = Catalog("Qwen/Qwen3-32B", {})
    assert get_active_case_catalog() is None
    assert get_active_model_path() is None

    with use_case_catalog(catalog):
        assert get_active_case_catalog() is catalog
        assert get_active_model_path() == "Qwen/Qwen3-32B"
        with use_case_catalog(catalog, model_path="Qwen/Qwen3-32B-FP8"):
            assert get_active_model_path() == "Qwen/Qwen3-32B-FP8"
        assert get_active_model_path() == "Qwen/Qwen3-32B"

    assert get_active_case_catalog() is None
    assert get_active_model_path() is None
