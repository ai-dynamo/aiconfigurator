# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure (no-dynamo) tests for the load-predictor sweep: loss math + the
trigger paths that return before touching the real predictors."""

from spica import SmartSearchConfig, sweep_load_predictor, window_loss


def _cfg(**search_space):
    ss = {"model_name": "m", "hardware_sku": "h200_sxm", **search_space}
    return SmartSearchConfig(
        search_space=ss,
        workload={"isl": 4000, "osl": 1000, "request_rate": 25},  # static
    )


def test_window_loss_zero_on_exact_match():
    assert window_loss(3, 100, 10, 3, 100, 10) == 0.0


def test_window_loss_zero_on_empty_window_predicted_zero():
    assert window_loss(0, 0, 0, 0, 0, 0) == 0.0


def test_window_loss_penalizes_overprediction():
    # over-predicting num_req inflates N*I and N*O
    assert window_loss(6, 100, 10, 3, 100, 10) > 0.0
    # predicting traffic during an actual lull is penalized
    assert window_loss(5, 100, 10, 0, 0, 0) > 0.0


def test_sweep_skips_when_no_throughput_policy():
    r = sweep_load_predictor(_cfg(planner_scaling_policy=["disabled", "load_180_5"]))
    assert r.reason == "no_throughput_scaling_candidate"
    assert r.best_by_interval == {}


def test_sweep_static_workload_uses_constant_per_interval():
    r = sweep_load_predictor(_cfg(planner_scaling_policy=["throughput_180_5", "hybrid_600_5"]))
    assert r.reason == "static_workload_constant"
    assert r.best_by_interval == {180: "constant_last", 600: "constant_last"}
