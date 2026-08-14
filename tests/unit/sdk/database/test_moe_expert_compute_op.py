# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the modeled large-EP MoE expert-compute op."""

from collections import defaultdict

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations import ModeledEPMoE

pytestmark = pytest.mark.unit


def _leaf(latency, power=0.0):
    return {"latency": latency, "power": power, "energy": latency * power}


def _nested_dict():
    return defaultdict(_nested_dict)


@pytest.fixture
def stock_moe_db(stub_perf_db):
    """A stub PerfDatabase with only the stock MoE slice ModeledEPMoE queries."""
    stub_perf_db.backend = common.BackendName.trtllm.value
    data = _nested_dict()
    data[common.MoEQuantMode.bfloat16]["balanced"][2][8][2048][8192][1][2].update(
        {
            8: _leaf(0.80, power=10.0),
            16: _leaf(1.60, power=10.0),
        }
    )
    stub_perf_db._moe_data = data
    stub_perf_db._moe_low_latency_data = None
    return stub_perf_db


def _make_op(scale_factor=1.0, **overrides):
    kwargs = {
        "hidden_size": 2048,
        "inter_size": 8192,
        "topk": 2,
        "num_experts": 8,
        "moe_ep_size": 2,
        "quant_mode": common.MoEQuantMode.bfloat16,
        "attention_dp_size": 4,
        "inference_phase": "context",
    }
    kwargs.update(overrides)
    return ModeledEPMoE("modeled_ep_moe", scale_factor, **kwargs)


def test_modeled_coordinates_globalize_then_shard_tokens():
    op = _make_op(attention_dp_size=3, moe_ep_size=4, num_experts=8)

    assert op.modeled_coordinates(x=5) == {
        "global_tokens": 15,
        "num_tokens": 4,
        "topk": 2,
        "num_experts": 8,
        "moe_tp_size": 1,
        "moe_ep_size": 4,
        "workload_distribution": "balanced",
    }


def test_query_uses_modeled_stock_moe_coordinates(stock_moe_db):
    op = _make_op(scale_factor=2.0)

    # x=4 globalizes to 16 tokens across attention DP and then queries the
    # local EP shard at ceil(16 / 2) = 8 stock-MoE tokens.
    result = op.query(stock_moe_db, x=4)

    assert float(result) == pytest.approx(0.80 * 2.0, rel=1e-12)
    assert result.energy == pytest.approx(8.0 * 2.0, rel=1e-12)
    assert result.source == "estimated"


def test_get_weights_counts_only_local_experts():
    quant = common.MoEQuantMode.bfloat16
    gated = _make_op(scale_factor=2.0, is_gated=True)
    assert gated.get_weights() == 2048 * 8192 * (8 // 2) * quant.value.memory * 3 * 2.0

    non_gated = _make_op(is_gated=False)
    assert non_gated.get_weights() == 2048 * 8192 * (8 // 2) * quant.value.memory * 2


def test_constructor_rejects_invalid_modeling_geometry():
    with pytest.raises(ValueError, match="moe_ep_size > 1"):
        _make_op(moe_ep_size=1)

    with pytest.raises(ValueError, match="must be divisible"):
        _make_op(num_experts=10, moe_ep_size=4)

    with pytest.raises(ValueError, match="inference_phase"):
        _make_op(inference_phase="prefill")
