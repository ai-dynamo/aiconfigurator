# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

_MODULE = "collector.sglang.collect_dsv4_megamoe"
_saved_module = sys.modules.pop(_MODULE, None)
_saved_torch = sys.modules.get("torch")
_saved_torch_distributed = sys.modules.get("torch.distributed")
sys.modules["torch"] = MagicMock()
sys.modules["torch.distributed"] = MagicMock()
try:
    from collector.sglang.collect_dsv4_megamoe import (
        CaseRunResult,
        MegaMoECase,
        aggregate_case_run_results,
        group_cases_for_logging,
    )
finally:
    sys.modules.pop(_MODULE, None)
    if _saved_module is not None:
        sys.modules[_MODULE] = _saved_module
    if _saved_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = _saved_torch
    if _saved_torch_distributed is None:
        sys.modules.pop("torch.distributed", None)
    else:
        sys.modules["torch.distributed"] = _saved_torch_distributed


def test_group_cases_for_logging_groups_seed_variants():
    cases = [
        MegaMoECase("context", 1024, "power_law_sampled_1.9", 8, 0),
        MegaMoECase("context", 1024, "power_law_sampled_1.9", 8, 1),
        MegaMoECase("context", 2048, "power_law_sampled_1.9", 8, 0),
    ]

    groups = group_cases_for_logging(cases)

    assert groups == [cases[:2], cases[2:]]


def test_aggregate_case_run_results_averages_latency_and_power():
    results = [
        CaseRunResult({"latency": "1.000000", "distribution": "power_law_sampled_1.9"}, {"power": 100.0}),
        CaseRunResult({"latency": "3.000000", "distribution": "power_law_sampled_1.9"}, {"power": 300.0}),
    ]

    aggregated = aggregate_case_run_results(results)

    assert aggregated.row["latency"] == "2.000000"
    assert aggregated.row["distribution"] == "power_law_sampled_1.9"
    assert aggregated.power_stats["power"] == 200.0
