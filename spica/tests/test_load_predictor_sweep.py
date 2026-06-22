# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure (no-dynamo) tests for the load-predictor sweep: loss math + the
trigger paths that return before touching the real predictors."""

import pytest

from spica import SmartSearchConfig, sweep_load_predictor, window_loss
from spica.load_predictor_sweep import _entry_label, _internal_preset, predictor_fields


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


# --- load_predictor_presets as raw dicts (unrolled names) ---


def test_internal_preset_from_id_or_dict():
    # preset id -> the internal preset dict verbatim
    assert _internal_preset("prophet_w20_log1p") == {
        "family": "prophet",
        "log1p": True,
        "prophet_window_size": 20,
    }
    # custom dict (unrolled names) -> internal names + family defaults filled
    internal = _internal_preset({"load_predictor": "kalman", "load_predictor_log1p": True, "kalman_q_level": 3.0})
    assert internal["family"] == "kalman" and internal["log1p"] is True
    assert internal["q_level"] == 3.0 and internal["min_points"] == 5  # default


def test_internal_preset_rejects_unknown_family():
    with pytest.raises(ValueError, match="load_predictor must be one of"):
        _internal_preset({"load_predictor": "bogus"})


def test_predictor_fields_dict_emits_family_knobs_only():
    fields = predictor_fields({"load_predictor": "prophet", "prophet_window_size": 30})
    assert fields == {"load_predictor": "prophet", "load_predictor_log1p": False, "prophet_window_size": 30}
    assert "kalman_q_level" not in fields  # only the chosen family's knobs


def test_entry_label_ids_dicts_by_index():
    assert _entry_label("prophet_w20_raw", 0) == "prophet_w20_raw"
    assert _entry_label({"load_predictor": "kalman"}, 3) == "custom_3"
