# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import pytest

from collector.targeted_collection import (
    TargetCollectionSpec,
    TargetContext,
    TargetSelector,
    load_target_collection_spec,
)

pytestmark = pytest.mark.unit


def _task_id(case):
    return f"sglang.gdn:run_gdn:{case}"


def test_filters_data_points_by_model_gpu_and_case_fields():
    spec = TargetCollectionSpec(
        data_points=(
            TargetSelector(
                backend="sglang",
                op="gdn",
                models=("Qwen/Qwen3.5-27B",),
                gpus=("b200_sxm",),
                match={"0": "context", "1": 5120, "9": "Qwen/Qwen3.5-27B"},
            ),
        )
    )
    cases = [
        ["context", 5120, 4, 16, 128, 48, 128, [1, 2], [128], "Qwen/Qwen3.5-27B"],
        ["generation", 5120, 4, 16, 128, 48, 128, [1, 2], None, "Qwen/Qwen3.5-27B"],
        ["context", 4096, 4, 16, 128, 32, 128, [1, 2], [128], "Qwen/Qwen3.5-9B"],
    ]

    result = spec.filter_cases(
        cases,
        TargetContext(
            backend="sglang",
            op="gdn",
            target_gpu="b200_sxm",
            target_models=("Qwen/Qwen3.5-27B",),
        ),
        _task_id,
    )

    assert result.cases == [cases[0]]
    assert result.original_count == 3
    assert result.selected_count == 1
    assert result.skipped_by_exception == 0


def test_gpu_exception_removes_matching_case_only_for_matching_gpu():
    spec = TargetCollectionSpec(
        data_points=(TargetSelector(backend="sglang", op="gdn", contains=("Qwen/Qwen3.5-27B",)),),
        gpu_exceptions=(
            TargetSelector(
                backend="sglang",
                op="gdn",
                gpus=("b200_sxm",),
                match={"0": "generation", "9": "Qwen/Qwen3.5-27B"},
            ),
        ),
    )
    cases = [
        ["context", 5120, 4, 16, 128, 48, 128, [1], [128], "Qwen/Qwen3.5-27B"],
        ["generation", 5120, 4, 16, 128, 48, 128, [1], None, "Qwen/Qwen3.5-27B"],
    ]

    b200_result = spec.filter_cases(
        cases,
        TargetContext(backend="sglang", op="gdn", target_gpu="b200_sxm"),
        _task_id,
    )
    h100_result = spec.filter_cases(
        cases,
        TargetContext(backend="sglang", op="gdn", target_gpu="h100_sxm"),
        _task_id,
    )
    no_gpu_result = spec.filter_cases(
        cases,
        TargetContext(backend="sglang", op="gdn"),
        _task_id,
    )

    assert b200_result.cases == [cases[0]]
    assert b200_result.skipped_by_exception == 1
    assert h100_result.cases == cases
    assert h100_result.skipped_by_exception == 0
    assert no_gpu_result.cases == cases
    assert no_gpu_result.skipped_by_exception == 0


def test_case_id_selector_matches_exact_task_id():
    cases = [["a"], ["b"]]
    spec = TargetCollectionSpec(data_points=(TargetSelector(case_ids=("id:1",)),))

    result = spec.filter_cases(
        cases,
        TargetContext(backend="sglang", op="gdn"),
        task_id_factory=lambda case: "id:1" if case == ["b"] else "id:0",
    )

    assert result.cases == [["b"]]


def test_named_fields_work_for_dataclass_cases():
    @dataclass
    class FakeCase:
        phase: str
        d_model: int
        model_name: str

    cases = [
        FakeCase("context", 5120, "Qwen/Qwen3.5-27B"),
        FakeCase("context", 4096, "Qwen/Qwen3.5-9B"),
    ]
    spec = TargetCollectionSpec(
        data_points=(
            TargetSelector(
                backend="sglang",
                op="gdn",
                models=("Qwen/Qwen3.5-27B",),
                match={"phase": "context", "d_model": 5120},
            ),
        )
    )

    result = spec.filter_cases(cases, TargetContext(backend="sglang", op="gdn"), _task_id)

    assert result.cases == [cases[0]]


def test_load_target_collection_spec_accepts_aliases(tmp_path):
    spec_path = tmp_path / "target_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "data_points": [
                    {
                        "framework": "sglang",
                        "operation": "gdn",
                        "model": "Qwen/Qwen3.5-27B",
                        "gpu": "b200_sxm",
                        "case_match": {"0": "context"},
                    }
                ],
                "exceptions": [
                    {
                        "framework": "sglang",
                        "operation": "gdn",
                        "gpu": "b200_sxm",
                        "id_contains": "generation",
                    }
                ],
            }
        )
    )

    spec = load_target_collection_spec(spec_path)

    assert spec.data_points[0].backend == "sglang"
    assert spec.data_points[0].op == "gdn"
    assert spec.data_points[0].models == ("Qwen/Qwen3.5-27B",)
    assert spec.data_points[0].gpus == ("b200_sxm",)
    assert spec.data_points[0].match == {"0": "context"}
    assert spec.gpu_exceptions[0].id_contains == ("generation",)
