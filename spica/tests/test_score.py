# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring / feasibility / ranking over a replay trace_report (pure)."""

from spica.config import OptimizationTarget
from spica.score import (
    is_feasible,
    make_candidate,
    objective_value,
    rank,
    score_report,
)

# A representative trace_report (the keys the merged replay emits).
REPORT = {
    "output_throughput_tok_s": 5000.0,
    "mean_ttft_ms": 800.0,
    "mean_tpot_ms": 20.0,
    "mean_e2e_latency_ms": 1200.0,
    "goodput_output_throughput_tok_s": 4000.0,
    "gpu_hours": 2.0,
}


def test_objective_per_target():
    assert objective_value(REPORT, OptimizationTarget.THROUGHPUT) == 5000.0
    assert objective_value(REPORT, OptimizationTarget.E2E_LATENCY) == 1200.0
    assert objective_value(REPORT, OptimizationTarget.GOODPUT) == 4000.0
    # goodput_per_gpu_hour = goodput / gpu_hours = 4000 / 2 = 2000
    assert objective_value(REPORT, OptimizationTarget.GOODPUT_PER_GPU_HOUR) == 2000.0


def test_goodput_per_gpu_hour_zero_when_no_gpu_hours():
    assert (
        objective_value(
            {"goodput_output_throughput_tok_s": 4000.0, "gpu_hours": 0.0}, OptimizationTarget.GOODPUT_PER_GPU_HOUR
        )
        == 0.0
    )


def test_score_sign():
    # maximized targets keep sign; e2e_latency is negated (higher score = lower latency)
    assert score_report(REPORT, OptimizationTarget.GOODPUT_PER_GPU_HOUR) == 2000.0
    assert score_report(REPORT, OptimizationTarget.E2E_LATENCY) == -1200.0


def test_is_feasible_budget_only():
    # SLA is no longer a feasibility gate (the goodput targets bake it into the
    # metric); feasibility is purely the GPU budget.
    assert is_feasible(used_gpus=16, gpu_budget=32)
    assert is_feasible(used_gpus=32, gpu_budget=32)  # at the budget
    assert not is_feasible(used_gpus=64, gpu_budget=32)  # over budget


def test_make_candidate_and_rank():
    cfg = {"used_gpus": 16, "deployment_mode": "agg"}
    c = make_candidate(cfg, REPORT, OptimizationTarget.GOODPUT_PER_GPU_HOUR)
    assert c.used_gpus == 16
    assert c.score == 2000.0
    assert c.metrics["goodput_output_throughput_tok_s"] == 4000.0 and c.metrics["gpu_hours"] == 2.0

    a = make_candidate({"used_gpus": 8}, {**REPORT, "gpu_hours": 4.0}, OptimizationTarget.GOODPUT_PER_GPU_HOUR)  # 1000
    b = make_candidate({"used_gpus": 16}, REPORT, OptimizationTarget.GOODPUT_PER_GPU_HOUR)  # 2000
    tie = make_candidate({"used_gpus": 8}, REPORT, OptimizationTarget.GOODPUT_PER_GPU_HOUR)  # 2000, fewer gpus
    ranked = rank([a, b, tie])
    assert ranked[0] is tie and ranked[1] is b and ranked[2] is a  # 2000(8gpu), 2000(16gpu), 1000
