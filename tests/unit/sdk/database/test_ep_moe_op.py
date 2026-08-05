# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified ``EPMoE`` op and ``query_moe_ep``.

Query semantics against an injected ``db._moe_ep_data`` store (the
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
from aiconfigurator_core.sdk.errors import EmpiricalNotImplementedError, PerfDataNotAvailableError
from aiconfigurator_core.sdk.operations import EPMoE

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
            # generation and deepgemm curves that must stay uncorrected.
            (
                ("deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 128, 128, 7168, 2048, 1, 16),
                {64: _leaf(0.15), 80: _leaf(0.25), 100: _leaf(0.40)},
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
    ``__dict__`` gate in ``EPMoE.load_data`` keeps the injected store.
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
    return EPMoE("test_ep_moe", scale_factor, **kwargs)


# ---------------------------------------------------------------------------
# Query semantics on the injected store
# ---------------------------------------------------------------------------


def test_adp_token_scaling_interpolation_scales_by_scale_factor(ep_db):
    # attention_dp_size globalizes tokens: x=12 * adp=4 -> 48, the midpoint
    # of the {32, 64} context curve.
    op = _make_op(scale_factor=2.0, attention_dp_size=4)
    result = op.query(ep_db, x=12)
    assert float(result) == pytest.approx(0.15 * 2.0, rel=1e-12)
    # power lerps flat at 100 W -> energy = 100 * 0.15, scaled with latency.
    assert result.energy == pytest.approx(100.0 * 0.15 * 2.0, rel=1e-12)
    assert result.source == "silicon"


def test_exact_token_hit_returns_leaf_value(ep_db):
    result = ep_db.query_moe_ep(
        "deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 16, 64
    )
    assert float(result) == pytest.approx(0.20, rel=1e-12)
    assert result.energy == pytest.approx(100.0 * 0.20, rel=1e-12)


def test_num_slots_defaults_to_num_experts(ep_db):
    # No num_slots -> num_experts=256 slot slice (0.10 at 32 tokens), not the
    # 512-slot slice collected for the same shape.
    assert float(_make_op().query(ep_db, x=32)) == pytest.approx(0.10, rel=1e-12)
    assert float(_make_op(num_slots=512).query(ep_db, x=32)) == pytest.approx(0.60, rel=1e-12)


def test_requested_distribution_falls_back_to_uniform(ep_db):
    op = _make_op(workload_distribution="power_law_9.9")
    assert float(op.query(ep_db, x=32)) == pytest.approx(0.10, rel=1e-12)


def test_distribution_fallback_is_phase_scoped(ep_db):
    # "ctx_only_dist" exists, but not for generation: the generation query
    # must fall back to uniform's generation curve (the legacy sglang tables
    # are separate per phase, so their fallback is inherently phase-scoped).
    op = _make_op(workload_distribution="ctx_only_dist", inference_phase="generation")
    assert float(op.query(ep_db, x=8)) == pytest.approx(0.05, rel=1e-12)


def test_distribution_fallback_to_sole_available(ep_db):
    # bfloat16 has only "power_law_1.2": requested "uniform" is absent, the
    # sole collected distribution answers (first-available degenerate case).
    op = _make_op(quant_mode=common.MoEQuantMode.bfloat16, workload_distribution="uniform")
    assert float(op.query(ep_db, x=16)) == pytest.approx(0.30, rel=1e-12)


def test_multi_distribution_without_uniform_falls_back_to_first_available(ep_db):
    # fp8 has {power_law_0.6, power_law_1.2}, no uniform: the first collected
    # distribution in table insertion order answers — the trtllm oracle's
    # ``available_distributions[0]`` behavior (its shipped gb200 table has no
    # uniform, so this path is production-reachable, e.g. via "power_law").
    result = ep_db.query_moe_ep(
        "deepep_moe", common.MoEQuantMode.fp8, "power_law", "context", 8, 256, 256, 7168, 2048, 1, 16, 16
    )
    assert float(result) == pytest.approx(0.40, rel=1e-12)


def test_missing_phase_raises(ep_db):
    # bfloat16 has context data only: a generation query has no candidate
    # distribution carrying the phase.
    with pytest.raises(PerfDataNotAvailableError):
        ep_db.query_moe_ep(
            "deepep_moe",
            common.MoEQuantMode.bfloat16,
            "power_law_1.2",
            "generation",
            8,
            256,
            256,
            7168,
            2048,
            1,
            16,
            16,
        )


def test_missing_slice_raises_named_miss(ep_db):
    with pytest.raises(PerfDataNotAvailableError, match="requested slice"):
        ep_db.query_moe_ep(
            "deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 999, 999, 7168, 2048, 1, 16, 32
        )


def test_hybrid_missing_slice_raises_empirical_not_implemented(ep_db):
    with pytest.raises(EmpiricalNotImplementedError, match="silicon data required"):
        ep_db.query_moe_ep(
            "deepep_moe",
            common.MoEQuantMode.fp8_block,
            "uniform",
            "context",
            8,
            999,
            999,
            7168,
            2048,
            1,
            16,
            32,
            database_mode=common.DatabaseMode.HYBRID,
        )


def test_singleton_token_underflow_raises_but_overflow_holds(ep_db):
    # ep=32 slice has a single 64-token point. Below it: typed miss (a
    # singleton cannot define the low-token launch floor — the sglang oracle
    # guard, adopted family-wide). Above it: boundary util-hold, unchanged.
    with pytest.raises(PerfDataNotAvailableError, match="singleton"):
        ep_db.query_moe_ep(
            "deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 32, 16
        )
    result = ep_db.query_moe_ep(
        "deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 32, 128
    )
    assert float(result) > 0.0


# ---------------------------------------------------------------------------
# Validation, weights, and tier contract
# ---------------------------------------------------------------------------


def test_ctor_rejects_unknown_inference_phase():
    with pytest.raises(ValueError, match="inference_phase"):
        _make_op(inference_phase="prefill")


def test_query_rejects_unknown_inference_phase(ep_db):
    with pytest.raises(ValueError, match="inference_phase"):
        ep_db.query_moe_ep(
            "deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "prefill", 8, 256, 256, 7168, 2048, 1, 16, 32
        )


def test_gated_and_non_gated_weights_formula():
    quant = common.MoEQuantMode.fp8_block
    gated = _make_op(scale_factor=2.0, is_gated=True)
    assert gated.get_weights() == (7168 * 2048 * 256 * quant.value.memory * 3 // 16) * 2.0
    non_gated = _make_op(is_gated=False)
    assert non_gated.get_weights() == 7168 * 2048 * 256 * quant.value.memory * 2 // 16


@pytest.mark.parametrize("mode", [common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL, common.DatabaseMode.EMPIRICAL])
def test_estimation_tiers_raise_empirical_not_implemented(ep_db, mode):
    with pytest.raises(EmpiricalNotImplementedError) as excinfo:
        ep_db.query_moe_ep(
            "deepep_moe",
            common.MoEQuantMode.fp8_block,
            "uniform",
            "context",
            8,
            256,
            256,
            7168,
            2048,
            1,
            16,
            32,
            database_mode=mode,
        )
    message = str(excinfo.value)
    assert "silicon data required (estimation tier is a planned follow-up)" in message
    # Full query context is part of the message.
    for fragment in ("deepep_moe", "context", "7168", "256"):
        assert fragment in message


# ---------------------------------------------------------------------------
# enable_eplb context correction (legacy fidelity: operations/moe.py applies
# int(num_tokens * 0.8) when enable_eplb and is_context on the sglang path)
# ---------------------------------------------------------------------------


def test_eplb_context_correction_on_sglang_leg(ep_db):
    # int(100 * 0.8) = 80: the eplb-on context query must equal the eplb-off
    # query at 80 tokens and land exactly on the measured 80-token point.
    eplb_on = _make_op(num_experts=128, enable_eplb=True)
    eplb_off = _make_op(num_experts=128)
    result = eplb_on.query(ep_db, x=100)
    assert float(result) == float(eplb_off.query(ep_db, x=80))
    assert float(result) == pytest.approx(0.25, rel=1e-12)


def test_eplb_default_off_is_noop(ep_db):
    # Without enable_eplb the 100-token context query stays uncorrected.
    assert float(_make_op(num_experts=128).query(ep_db, x=100)) == pytest.approx(0.40, rel=1e-12)


def test_eplb_generation_phase_unaffected(ep_db):
    # The legacy correction is context-only (prefill): generation queries at
    # 100 tokens must hit the 100-token point, not the corrected 80.
    op = _make_op(num_experts=128, inference_phase="generation", enable_eplb=True)
    assert float(op.query(ep_db, x=100)) == pytest.approx(0.55, rel=1e-12)


def test_eplb_trtllm_deepgemm_leg_unaffected(ep_db):
    # TrtLLMWideEPMoE never applied the 0.8 correction (its EPLB effect rides
    # the ``_eplb`` distribution suffix): the deepgemm leg stays uncorrected
    # even with enable_eplb on.
    op = _make_op(num_experts=128, kernel_source="deepgemm", enable_eplb=True)
    assert float(op.query(ep_db, x=100)) == pytest.approx(0.77, rel=1e-12)


# ---------------------------------------------------------------------------
# kernel_source auto-resolution (kernel_source=None)
# ---------------------------------------------------------------------------


def test_kernel_source_auto_resolves_to_deepep_moe_on_sglang(ep_db):
    ep_db.backend = "sglang"
    op = _make_op(kernel_source=None)
    assert float(op.query(ep_db, x=32)) == pytest.approx(0.10, rel=1e-12)


def test_kernel_source_auto_resolves_to_deepep_moe_on_vllm(ep_db):
    ep_db.backend = "vllm"
    op = _make_op(kernel_source=None)
    assert float(op.query(ep_db, x=32)) == pytest.approx(0.10, rel=1e-12)


def test_kernel_source_auto_resolution_trtllm_prefers_deepgemm_on_blackwell(ep_db):
    # SM >= 100 + fp8_block -> "deepgemm" (present in the injected store).
    ep_db.backend = "trtllm"
    ep_db.system_spec["gpu"]["sm_version"] = 100
    op = _make_op(kernel_source=None, inference_phase="generation")
    assert float(op.query(ep_db, x=16)) == pytest.approx(0.70, rel=1e-12)


def test_kernel_source_auto_resolution_trtllm_falls_back_to_available(stub_perf_db):
    # Preferred kernel absent (nvfp4 -> "moe_torch_flow"): fall back to the
    # collected kernel key — the shipped gb200 wideep table shape.
    stub_perf_db.backend = "trtllm"
    stub_perf_db.system_spec["gpu"]["sm_version"] = 100
    stub_perf_db._moe_ep_data = _store(
        [
            (
                (
                    "wideep_compute_cutlass",
                    common.MoEQuantMode.nvfp4,
                    "power_law_1.01",
                    "context",
                    8,
                    256,
                    288,
                    7168,
                    2048,
                    1,
                    16,
                ),
                {4: _leaf(0.80)},
            )
        ]
    )
    op = _make_op(
        kernel_source=None,
        quant_mode=common.MoEQuantMode.nvfp4,
        workload_distribution="power_law_1.01",
        num_slots=288,
    )
    assert float(op.query(stub_perf_db, x=4)) == pytest.approx(0.80, rel=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
