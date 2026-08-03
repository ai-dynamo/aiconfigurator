# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage certification for the DeepEP normal-mode collector."""

import csv

import pytest

from collector.wideep.sglang.collect_deepep_normal import _verify_axis_coverage

pytestmark = pytest.mark.unit

_FIELDNAMES = (
    "node_num",
    "hidden_size",
    "num_token",
    "num_topk",
    "num_experts",
    "dispatch_sms",
    "dispatch_transmit_us",
)
_CASES = [
    {"hidden_size": 2048, "num_experts": 128, "topk": 8},
    {"hidden_size": 7168, "num_experts": 256, "topk": 8},
]
_SMS = [4, 20]
_TOKENS = [1, 64]


def _rows(cases, sms_list=_SMS, tokens=_TOKENS):
    return [
        {
            "node_num": 1,
            "hidden_size": case["hidden_size"],
            "num_token": num_token,
            "num_topk": case["topk"],
            "num_experts": case["num_experts"],
            "dispatch_sms": sms,
            "dispatch_transmit_us": 42.0,
        }
        for case in cases
        for sms in sms_list
        for num_token in tokens
    ]


def _write_perf_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)


def test_full_pair_grid_passes_and_counts_planned_shapes(tmp_path):
    path = tmp_path / "wideep_deepep_normal_perf.txt"
    _write_perf_csv(path, _rows(_CASES))

    assert _verify_axis_coverage(path, _SMS, _TOKENS, _CASES) == len(_CASES)


def test_missing_pair_is_rejected_even_when_both_marginal_sets_are_complete(tmp_path):
    path = tmp_path / "wideep_deepep_normal_perf.txt"
    rows = _rows(_CASES)
    rows.remove(
        {
            "node_num": 1,
            "hidden_size": 2048,
            "num_token": 64,
            "num_topk": 8,
            "num_experts": 128,
            "dispatch_sms": 4,
            "dispatch_transmit_us": 42.0,
        }
    )
    _write_perf_csv(path, rows)

    with pytest.raises(RuntimeError, match=r"missing pairs=\[\(4, 64\)\]"):
        _verify_axis_coverage(path, _SMS, _TOKENS, _CASES)


def test_whole_missing_shape_blocks_completion(tmp_path):
    path = tmp_path / "wideep_deepep_normal_perf.txt"
    _write_perf_csv(path, _rows([_CASES[0]]))

    with pytest.raises(RuntimeError, match=r"hidden=7168.*missing all 4 pairs"):
        _verify_axis_coverage(path, _SMS, _TOKENS, _CASES)


def test_header_only_output_is_rejected(tmp_path):
    path = tmp_path / "wideep_deepep_normal_perf.txt"
    _write_perf_csv(path, [])

    with pytest.raises(RuntimeError, match="wrote no data rows"):
        _verify_axis_coverage(path, _SMS, _TOKENS, _CASES)
