# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified ``MoEExpertCompute`` op.

from collections import defaultdict

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations import MoEExpertCompute

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


# ---------------------------------------------------------------------------
# Retired with #1357 PR-5 (single oracle = the compiled engine): the query
# semantics this section pinned on the injected store — adp token
# globalization + interpolation, exact-token leaf hits, num_slots defaulting
# at lookup, distribution fallback (phase-scoped, sole-available,
# first-available), typed misses / EMPIRICAL tiers, token underflow/overflow
# holds, the sglang EPLB 0.8 context correction, and per-backend
# kernel-source auto-resolution — live in
# aic-core/rust/.../operators/moe_expert_compute.rs, anchored by
# the frozen parity
# goldens (the shims answer from DISK, so the injected in-memory store is
# invisible to them). Python-side contracts stay below.
# ---------------------------------------------------------------------------


def test_ctor_rejects_unknown_inference_phase():
    with pytest.raises(ValueError, match="inference_phase"):
        _make_op(inference_phase="prefill")


def test_gated_and_non_gated_weights_formula():
    quant = common.MoEQuantMode.fp8_block
    gated = _make_op(scale_factor=2.0, is_gated=True)
    assert gated.get_weights() == (7168 * 2048 * 256 * quant.value.memory * 3 // 16) * 2.0
    non_gated = _make_op(is_gated=False)
    assert non_gated.get_weights() == 7168 * 2048 * 256 * quant.value.memory * 2 // 16
