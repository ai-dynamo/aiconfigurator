# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan and coverage certification for the DeepEP low-latency collector."""

import csv

import pytest

from collector.wideep.sglang import collect_deepep_ll
from collector.wideep.sglang.collect_deepep_ll import _validate_expert_partition, _verify_token_coverage

pytestmark = pytest.mark.unit

_FIELDNAMES = (
    "node_num",
    "hidden_size",
    "num_token",
    "num_topk",
    "num_experts",
    "dispatch_avg_t_us",
)
_CASES = [
    {"hidden_size": 2048, "num_experts": 128, "topk": 8},
    {"hidden_size": 7168, "num_experts": 256, "topk": 8},
]
_TOKENS = [1, 64]


def _rows(cases, tokens=_TOKENS):
    return [
        {
            "node_num": 1,
            "hidden_size": case["hidden_size"],
            "num_token": num_token,
            "num_topk": case["topk"],
            "num_experts": case["num_experts"],
            "dispatch_avg_t_us": 42.0,
        }
        for case in cases
        for num_token in tokens
    ]


def _write_perf_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)


def test_full_token_sweep_passes_and_counts_planned_shapes(tmp_path):
    path = tmp_path / "wideep_deepep_ll_perf.txt"
    _write_perf_csv(path, _rows(_CASES))

    assert _verify_token_coverage(path, _TOKENS, _CASES) == len(_CASES)


def test_missing_token_is_rejected(tmp_path):
    path = tmp_path / "wideep_deepep_ll_perf.txt"
    rows = _rows(_CASES)
    rows.pop(1)
    _write_perf_csv(path, rows)

    with pytest.raises(RuntimeError, match=r"hidden=2048.*missing tokens=\[64\]"):
        _verify_token_coverage(path, _TOKENS, _CASES)


def test_whole_missing_shape_blocks_completion(tmp_path):
    path = tmp_path / "wideep_deepep_ll_perf.txt"
    _write_perf_csv(path, _rows([_CASES[0]]))

    with pytest.raises(RuntimeError, match=r"hidden=7168.*missing all 2 tokens"):
        _verify_token_coverage(path, _TOKENS, _CASES)


def test_header_only_output_is_rejected(tmp_path):
    path = tmp_path / "wideep_deepep_ll_perf.txt"
    _write_perf_csv(path, [])

    with pytest.raises(RuntimeError, match="wrote no data rows"):
        _verify_token_coverage(path, _TOKENS, _CASES)


def test_public_plan_does_not_filter_framework_kernel_limits(monkeypatch):
    shapes = [
        (8192, 512, 22),  # above the known DeepEP 1.2.1 low-latency top-k cap
        (4100, 128, 8),  # outside the known hidden specializations/divisibility
        (2048, 128, 8),
        (2048, 128, 8),  # duplicate config must still be deduplicated
    ]
    monkeypatch.setattr(collect_deepep_ll, "_iter_moe_configs", lambda: iter(shapes))

    assert collect_deepep_ll.get_deepep_ll_test_cases() == [
        {"hidden_size": 2048, "num_experts": 128, "topk": 8},
        {"hidden_size": 4100, "num_experts": 128, "topk": 8},
        {"hidden_size": 8192, "num_experts": 512, "topk": 22},
    ]


def test_nondivisible_expert_partition_fails_instead_of_skipping():
    with pytest.raises(RuntimeError, match=r"experts=130.*ranks=8"):
        _validate_expert_partition(hidden=2048, num_experts=130, num_topk=8, num_ranks=8)
