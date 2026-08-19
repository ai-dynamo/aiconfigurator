# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.cli.api import _apply_power_coverage_gate, apply_row_power_coverage_gate
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary

pytestmark = pytest.mark.unit


def _summary(*, covered_latency: float, uncovered_latency: float) -> InferenceSummary:
    summary = InferenceSummary(RuntimeConfig(batch_size=1, isl=128, osl=16))
    summary.set_context_latency_dict(
        {
            "covered": covered_latency,
            "uncovered": uncovered_latency,
        }
    )
    summary.set_context_energy_wms_dict({"covered": 100.0})
    return summary


def test_partial_power_coverage_only_invalidates_power() -> None:
    result = {
        "power_w": 450.0,
        "ttft": 12.0,
        "tokens/s": 100.0,
    }

    gated = _apply_power_coverage_gate(
        _summary(covered_latency=89.0, uncovered_latency=11.0),
        result,
    )

    assert gated["power_w"] is None
    assert gated["power_coverage"] == pytest.approx(0.89)
    assert gated["ttft"] == result["ttft"]
    assert gated["tokens/s"] == result["tokens/s"]


def test_sufficient_power_coverage_preserves_power() -> None:
    gated = _apply_power_coverage_gate(
        _summary(covered_latency=90.0, uncovered_latency=10.0),
        {"power_w": 450.0},
    )

    assert gated["power_w"] == 450.0
    assert gated["power_coverage"] == pytest.approx(0.9)


def test_row_gate_applies_same_rule_to_prestamped_rows() -> None:
    assert apply_row_power_coverage_gate({"power_w": 450.0, "power_coverage": 0.95})["power_w"] == 450.0
    # Fail-closed: no evidence (missing / NaN coverage) hides the power too.
    assert apply_row_power_coverage_gate({"power_w": 450.0})["power_w"] is None
    assert apply_row_power_coverage_gate({"power_w": 450.0, "power_coverage": float("nan")})["power_w"] is None
