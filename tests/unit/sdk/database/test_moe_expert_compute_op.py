# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified ``MoEExpertCompute`` op.

Query semantics against an injected expert-compute store (the
``__dict__``-gated bind in ``load_data`` honors pre-set attributes): ADP token
scaling + interpolation and scale_factor, the ``num_slots`` default, the
distribution fallback chain (requested -> "uniform" -> first-available in
table insertion order, phase-scoped like the per-phase legacy sglang tables;
typed miss only when no distribution carries the phase), missing
phase raising, the gated/non-gated weights formula, inference_phase
validation, the silicon-only tier contract (SOL/SOL_FULL/EMPIRICAL raise
``EmpiricalNotImplementedError``), kernel_source auto-resolution
(sglang/vllm -> "deepep_moe"; trtllm -> the replicated
``TrtLLMWideEPMoE._select_kernel`` logic over the unified table's keys), and
the ``enable_eplb`` legacy-fidelity context correction (``int(tokens * 0.8)``
on sglang-adapted kernel legs only, mirroring ``operations/moe.py``).
"""

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations import MoEExpertCompute

pytestmark = pytest.mark.unit


def _leaf(latency, power=0.0):
    return {"latency": latency, "power": power, "energy": power * latency}


def _store(entries):
    """Build a nested moe_ep store from ``(11-part key, {tokens: leaf})`` pairs."""
    data = {}
    for key, tokens in entries:
        node = data
        for part in key[:-1]:
            node = node.setdefault(part, {})
        node[key[-1]] = tokens
    return data


# Shared slice shape: topk=8, experts=256, slots=256, hidden=7168, inter=2048,
# tp=1, ep=16 (key order after phase: topk, experts, slots, hidden, inter, tp, ep).
_SLICE = (8, 256, 256, 7168, 2048, 1, 16)


def _build_injected_store():
    return _store(
        [
            # deepep_moe/fp8_block/uniform: two-point context curve + a
            # generation point (phase-scoped fallback target).
            (
                ("deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", *_SLICE),
                {32: _leaf(0.10, power=100.0), 64: _leaf(0.20, power=100.0)},
            ),
            (
                ("deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "generation", *_SLICE),
                {8: _leaf(0.05)},
            ),
            # A context-only distribution: a generation query for it must fall
            # back within the generation phase (to "uniform" above).
            (
                ("deepep_moe", common.MoEQuantMode.fp8_block, "ctx_only_dist", "context", *_SLICE),
                {32: _leaf(0.90)},
            ),
            # num_slots axis: same shape collected under 512 slots.
            (
                ("deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 512, 7168, 2048, 1, 16),
                {32: _leaf(0.60)},
            ),
            # bfloat16: sole collected distribution (fallback target), context only.
            (
                ("deepep_moe", common.MoEQuantMode.bfloat16, "power_law_1.2", "context", *_SLICE),
                {16: _leaf(0.30)},
            ),
            # fp8: two non-uniform distributions, no uniform -> the
            # first-available (insertion-order) fallback answers.
            (
                ("deepep_moe", common.MoEQuantMode.fp8, "power_law_0.6", "context", *_SLICE),
                {16: _leaf(0.40)},
            ),
            (
                ("deepep_moe", common.MoEQuantMode.fp8, "power_law_1.2", "context", *_SLICE),
                {16: _leaf(0.50)},
            ),
            # deepgemm kernel (trtllm preferred hit for Blackwell + fp8_block).
            (
                ("deepgemm", common.MoEQuantMode.fp8_block, "uniform", "generation", *_SLICE),
                {16: _leaf(0.70)},
            ),
            # Singleton token curve (ep=32 slice) for the underflow guard.
            (
                ("deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 32),
                {64: _leaf(0.42)},
            ),
            # eplb-correction slices (num_experts=128 keeps them disjoint from
            # the shared 256-expert shapes): context tokens include 80 so the
            # corrected int(100 * 0.8) = 80 lands on a measured point, plus
            # generation and deepgemm curves that must stay uncorrected. The
            # adjacent 160/161 points pin the truncation ORDER: globalize by
            # attention_dp first, truncate second (int(101*2*0.8) = 161, not
            # int(101*0.8)*2 = 160).
            (
                ("deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 128, 128, 7168, 2048, 1, 16),
                {64: _leaf(0.15), 80: _leaf(0.25), 100: _leaf(0.40), 160: _leaf(0.62), 161: _leaf(0.66)},
            ),
            (
                ("deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "generation", 8, 128, 128, 7168, 2048, 1, 16),
                {80: _leaf(0.33), 100: _leaf(0.55)},
            ),
            (
                ("deepgemm", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 128, 128, 7168, 2048, 1, 16),
                {80: _leaf(0.61), 100: _leaf(0.77)},
            ),
        ]
    )


@pytest.fixture
def ep_db(stub_perf_db):
    """A stub PerfDatabase with an injected unified moe_ep store.

    ``stub_perf_db`` warm-up already bound ``_moe_ep_data`` (None on its
    unsupported stub backend); the assignment below replaces it and the
    ``__dict__`` gate in ``MoEExpertCompute.load_data`` keeps the injected store.
    """
    stub_perf_db._moe_ep_data = _build_injected_store()
    return stub_perf_db


def _make_op(scale_factor=1.0, **overrides):
    kwargs = {
        "hidden_size": 7168,
        "inter_size": 2048,
        "topk": 8,
        "num_experts": 256,
        "moe_ep_size": 16,
        "quant_mode": common.MoEQuantMode.fp8_block,
        "workload_distribution": "uniform",
        "attention_dp_size": 1,
        "inference_phase": "context",
        "kernel_source": "deepep_moe",
    }
    kwargs.update(overrides)
    return MoEExpertCompute("test_ep_moe", scale_factor, **kwargs)


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
