# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from pydantic import ValidationError

from spica import OptimizationTarget, SmartSearchConfig
from spica.config import OptimizationGoal, SLATarget

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "smart_sweep.yaml"


def test_example_yaml_loads():
    cfg = SmartSearchConfig.from_yaml(EXAMPLE)
    assert cfg.search_space.model_name == "deepseek-ai/DeepSeek-V3"
    assert cfg.search_space.deployment_mode == ["disagg", "agg"]
    assert "disabled" in cfg.search_space.planner_scaling_policy
    assert cfg.workload.is_trace_based
    assert cfg.goal.target is OptimizationTarget.GOODPUT_PER_GPU_HOUR
    assert cfg.sweep.max_rounds == 40


def test_defaults_fill_in():
    cfg = SmartSearchConfig(
        search_space={"model_name": "m", "hardware_sku": "h200_sxm"},
        workload={"isl": 4000, "osl": 1000, "request_rate": 25},
    )
    # search-space defaults
    assert cfg.search_space.gpu_budget == 32
    assert cfg.search_space.prefill_block_size == 64
    assert len(cfg.search_space.load_predictor_presets) == 11
    # goal/sweep default factories
    assert cfg.goal.target is OptimizationTarget.THROUGHPUT
    assert cfg.sweep.parallel_evals == 16


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space={"model_name": "m", "hardware_sku": "h", "bogus": 1},
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )


def test_goodput_requires_sla():
    with pytest.raises(ValidationError):
        OptimizationGoal(target=OptimizationTarget.GOODPUT)  # no SLA
    # ttft alone is not enough (needs ttft+itl, or e2e)
    with pytest.raises(ValidationError):
        OptimizationGoal(target=OptimizationTarget.GOODPUT, sla=SLATarget(ttft_ms=2000))
    # ttft + itl is fine
    OptimizationGoal(target=OptimizationTarget.GOODPUT, sla=SLATarget(ttft_ms=2000, itl_ms=30))


def test_target_direction():
    assert OptimizationTarget.THROUGHPUT.maximize
    assert OptimizationTarget.GOODPUT_PER_GPU_HOUR.maximize
    assert not OptimizationTarget.E2E_LATENCY.maximize


def _search_space(**overrides):
    return {"model_name": "m", "hardware_sku": "h200_sxm", **overrides}


def test_subset_of_choices_is_accepted():
    cfg = SmartSearchConfig(
        search_space=_search_space(router_mode=["kv_router"], backend=["vllm", "sglang"]),
        workload={"isl": 1, "osl": 1, "concurrency": 1},
    )
    assert cfg.search_space.router_mode == ["kv_router"]
    assert cfg.search_space.backend == ["vllm", "sglang"]


def test_value_outside_choices_rejected():
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(router_mode=["bogus"]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )
    # numeric dimension: 999 is not a listed prefill_max_num_seqs choice
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(prefill_max_num_seqs=[1, 999]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )
    # planner: only listed presets allowed
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(planner_scaling_policy=["disabled", "made_up"]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )


def test_empty_choice_list_rejected():
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(router_mode=[]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )


# --- composite knobs accept raw dicts (preset OR pinned-dict per entry) ---


def test_composite_dict_entries_accepted():
    cfg = SmartSearchConfig(
        search_space=_search_space(
            planner_scaling_policy=[
                "disabled",
                {"enable_throughput_scaling": True, "throughput_adjustment_interval_seconds": 240},
            ],
            planner_fpm_sampling=[{"max_num_fpm_samples": 96, "fpm_sample_bucket_size": 16}],
            load_predictor_presets=[{"load_predictor": "prophet", "prophet_window_size": 30}],
        ),
        workload={"isl": 1, "osl": 1, "concurrency": 1},
    )
    assert isinstance(cfg.search_space.planner_scaling_policy[1], dict)
    assert cfg.search_space.planner_fpm_sampling[0]["max_num_fpm_samples"] == 96


def test_composite_dict_unknown_key_rejected():
    with pytest.raises(ValidationError, match="unknown keys"):
        SmartSearchConfig(
            search_space=_search_space(planner_fpm_sampling=[{"max_num_fpm_samples": 64, "bogus": 1}]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )


def test_dict_rejected_on_non_composite_knob():
    # router_mode is list[str]; a dict entry is not allowed
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(router_mode=[{"foo": 1}]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )


# --- pinned parallel_configs (structural validation; legality is in enumerate_branches) ---


def test_parallel_configs_pin_requires_single_mode():
    with pytest.raises(ValidationError, match="exactly one mode"):
        SmartSearchConfig(
            search_space=_search_space(deployment_mode=["agg", "disagg"], parallel_configs=[{"tp": 4}]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )


def test_parallel_configs_shape_matches_mode():
    # agg entry needs tp
    with pytest.raises(ValidationError, match="'tp' field"):
        SmartSearchConfig(
            search_space=_search_space(deployment_mode=["agg"], parallel_configs=[{"replicas": 2}]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )
    # disagg entry needs prefill + decode
    with pytest.raises(ValidationError, match="prefill"):
        SmartSearchConfig(
            search_space=_search_space(deployment_mode=["disagg"], parallel_configs=[{"tp": 4}]),
            workload={"isl": 1, "osl": 1, "concurrency": 1},
        )
    # well-formed agg + disagg entries pass structural validation
    SmartSearchConfig(
        search_space=_search_space(deployment_mode=["agg"], parallel_configs=[{"tp": 4, "moe_ep": 4, "replicas": 2}]),
        workload={"isl": 1, "osl": 1, "concurrency": 1},
    )
    SmartSearchConfig(
        search_space=_search_space(
            deployment_mode=["disagg"],
            parallel_configs=[{"prefill": {"tp": 8, "moe_ep": 8}, "decode": {"tp": 1, "attention_dp": 8, "moe_ep": 8}}],
        ),
        workload={"isl": 1, "osl": 1, "concurrency": 1},
    )
