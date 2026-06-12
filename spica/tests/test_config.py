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
    assert cfg.goal.target is OptimizationTarget.GOODPUT_PER_GPU
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
    assert OptimizationTarget.GOODPUT_PER_GPU.maximize
    assert not OptimizationTarget.E2E_LATENCY.maximize
