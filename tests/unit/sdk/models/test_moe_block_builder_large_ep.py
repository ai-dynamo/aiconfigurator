# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large-EP graph wiring and stock-MoE local-compute mapping."""

from unittest.mock import Mock

import pytest

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common, config
from aiconfigurator_core.sdk.models.blocks import MoEBlockShape, build_moe_block_ops
from aiconfigurator_core.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit

SHAPE = MoEBlockShape(
    hidden_size=7168,
    moe_inter_size=2048,
    topk=8,
    num_experts=256,
    num_shared_experts=0,
    num_moe_layers=61,
)


def _cfg(*, ep: int = 16, adp: int = 8) -> config.ModelConfig:
    cfg = config.ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=ep,
        attention_dp_size=adp,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        enable_eplb=True,
    )
    cfg.moe_comm_backend = {"context": "deepep_ht", "generation": "deepep_ll"}
    return cfg


def _build(phase: str = "context"):
    return build_moe_block_ops(
        phase,
        SHAPE,
        _cfg(),
        common.MoEQuantMode.fp8_block,
        "power_law_1.2",
        scale_factor=61,
        backend_name="vllm",
        inference_phase=phase,
        model_family="DEEPSEEK",
        gpus_per_node=8,
    )


def test_large_ep_wires_measured_a2a_around_modeled_local_compute():
    built = _build()
    dispatch, combine = [op for op in built if isinstance(op, ops.MoEAllToAll)]
    assert sum(isinstance(op, ops.ModeledEPMoE) for op in built) == 1
    assert [dispatch._phase, combine._phase] == ["dispatch", "combine"]
    assert dispatch._comm_backend == combine._comm_backend == "deepep_ht"


def test_modeled_coordinates_use_balanced_ep_local_token_semantics():
    modeled = next(op for op in _build() if isinstance(op, ops.ModeledEPMoE))
    # global=17*ADP8=136; local tokens=ceil(136/EP16)=9.
    assert modeled.modeled_coordinates(17) == {
        "global_tokens": 136,
        "num_tokens": 9,
        "topk": 8,
        "num_experts": 256,
        "moe_tp_size": 1,
        "moe_ep_size": 16,
        "workload_distribution": "balanced",
    }


def test_modeled_query_uses_stock_moe_and_marks_approximation():
    modeled = next(op for op in _build() if isinstance(op, ops.ModeledEPMoE))
    database = Mock()
    database.query_moe.return_value = PerformanceResult(0.25, energy=0.5, source="silicon")

    result = modeled.query(database, x=17)

    database.query_moe.assert_called_once_with(
        num_tokens=9,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=16,
        quant_mode=common.MoEQuantMode.fp8_block,
        workload_distribution="balanced",
        is_context=True,
        moe_backend=None,
        is_gated=True,
        enable_eplb=False,
    )
    assert float(result) == pytest.approx(0.25 * 61)
    assert result.energy == pytest.approx(0.5 * 61)
    assert result.source == "estimated"


def test_modeled_local_compute_has_no_eplb_or_num_slots_state():
    modeled = next(op for op in _build() if isinstance(op, ops.ModeledEPMoE))
    assert not hasattr(modeled, "_enable_eplb")
    assert not hasattr(modeled, "_num_slots")
    assert not hasattr(modeled, "_workload_distribution")


def test_modeled_local_compute_requires_integral_ep_expert_geometry():
    with pytest.raises(ValueError, match="must be divisible"):
        ops.ModeledEPMoE(
            "bad",
            1,
            hidden_size=128,
            inter_size=64,
            topk=2,
            num_experts=10,
            moe_ep_size=4,
            quant_mode=common.MoEQuantMode.fp8,
            attention_dp_size=1,
            inference_phase="context",
        )
