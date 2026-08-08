# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pipeline-parallel (PP) modeling.

Covers ``aiconfigurator_core.sdk.pipeline``, which is split in two:

``PipelineLayout`` -- where the work lives (partition, op placement, per-stage
times, per-hop link cost). Owns the rules, so a second derivation from them
cannot drift.

``PipelineSteadyState`` -- how the pipe runs (cycle time, ``balance_factor``,
``fill_factor``). AIC's mean-field closed form, valid only under the assumption
that one step shape describes every stage.

Also pins the ``pp_size == 1`` identity that keeps single-stage results
bit-identical to the pre-existing model.
"""

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk.pipeline import (
    PipelineLayout,
    PipelineSteadyState,
    Placement,
    classify,
    even_partition,
    warn_on_unclassified_ops,
)

pytestmark = pytest.mark.unit


# A dense-model-shaped per-op breakdown: per-layer work totalling 64.0 ms over
# 64 layers (1.0 ms/layer), plus an un-sharded 8.0 ms head on the last stage.
_PER_LAYER_MS = 64.0
_NUM_LAYERS = 64
_STEP = {
    "generation_qkv_gemm": 20.0,
    "generation_attention": 14.0,
    "generation_ffn2_gemm": 30.0,
    "generation_embedding": 0.5,
    "generation_logits_gemm": 8.0,
}


def _pipe(pp_size, *, num_microbatches=None, partition=None, p2p_overlap=False):
    return PipelineSteadyState(
        layout=PipelineLayout(pp_size=pp_size, partition=partition),
        num_microbatches=num_microbatches,
        p2p_overlap=p2p_overlap,
    )


# ---------------------------------------------------------------------------
# Placement classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("context_embedding", Placement.FIRST),
        ("generation_embedding", Placement.FIRST),
        ("generation_embedding_ar", Placement.FIRST),
        ("context_logits_gemm", Placement.LAST),
        ("generation_logits_gemm", Placement.LAST),
        ("context_p2p", Placement.LINK),
        ("generation_p2p", Placement.LINK),
        ("context_qkv_gemm", Placement.PER_LAYER),
        ("generation_moe", Placement.PER_LAYER),
        ("context_attention", Placement.PER_LAYER),
        ("context_ar_1", Placement.PER_LAYER),
    ],
)
def test_classify_placement(name, expected):
    assert classify(name) == expected


# ---------------------------------------------------------------------------
# PipelineLayout: partitioning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_layers,pp,expected",
    [
        (64, 1, (64,)),
        (64, 2, (32, 32)),
        (64, 8, (8,) * 8),
        # Remainder goes to the leading stages (vLLM get_pp_indices / TRT-LLM).
        (64, 6, (11, 11, 11, 11, 10, 10)),
        (64, 5, (13, 13, 13, 13, 12)),
        (61, 8, (8, 8, 8, 8, 8, 7, 7, 7)),
    ],
)
def test_even_partition(num_layers, pp, expected):
    assert even_partition(num_layers, pp) == expected
    assert sum(expected) == num_layers
    assert PipelineLayout(pp_size=pp).layer_partition(num_layers) == expected


def test_explicit_partition_must_match_pp_size():
    with pytest.raises(ValueError, match="partition has 3 entries"):
        PipelineLayout(pp_size=4, partition=(16, 16, 32))


def test_explicit_partition_is_honored():
    # Front-load layers to compensate for the head on the last stage.
    assert PipelineLayout(pp_size=2, partition=(36, 28)).layer_partition(_NUM_LAYERS) == (36, 28)


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_nonpositive_pp_size(bad):
    with pytest.raises(ValueError, match="pp_size must be >= 1"):
        PipelineLayout(pp_size=bad)


# ---------------------------------------------------------------------------
# PipelineLayout: stage times and the link
# ---------------------------------------------------------------------------


def test_stage_times_place_head_and_embedding():
    stages = PipelineLayout(pp_size=8).stage_times(_STEP, _NUM_LAYERS)
    per_layer_per_stage = _PER_LAYER_MS / 8  # 8.0 ms of layer work per stage
    assert stages[0] == pytest.approx(per_layer_per_stage + 0.5)  # + embedding
    assert stages[-1] == pytest.approx(per_layer_per_stage + 8.0)  # + lm_head
    for mid in stages[1:-1]:
        assert mid == pytest.approx(per_layer_per_stage)


def test_stage_times_sum_to_step_total_minus_link():
    """No compute is lost or duplicated by the fold."""
    step = dict(_STEP, generation_p2p=7.0)
    stages = PipelineLayout(pp_size=8).stage_times(step, _NUM_LAYERS)
    assert sum(stages) == pytest.approx(sum(_STEP.values()))


def test_link_latency_is_excluded_from_stage_times_and_amortized_per_hop():
    step = dict(_STEP, generation_p2p=7.0)  # total across pp-1 == 7 hops
    layout = PipelineLayout(pp_size=8)
    # P2P must not inflate any stage's compute time...
    assert layout.stage_times(step, _NUM_LAYERS) == layout.stage_times(_STEP, _NUM_LAYERS)
    # ...but one hop is charged to the cycle.
    assert layout.per_hop_latency(step) == pytest.approx(1.0)


def test_single_stage_has_no_hop():
    assert PipelineLayout(pp_size=1).per_hop_latency(dict(_STEP, generation_p2p=7.0)) == 0.0


# ---------------------------------------------------------------------------
# PipelineSteadyState: the max-not-average cycle
# ---------------------------------------------------------------------------


def test_cycle_is_set_by_fattest_stage_not_the_average():
    pipe = _pipe(8)
    step_total = sum(_STEP.values())  # 72.5 ms
    ideal_cycle = step_total / 8  # 9.0625 ms
    cycle = pipe.cycle_time(_STEP, _NUM_LAYERS)  # 8.0 + 8.0 = 16.0 ms
    assert cycle == pytest.approx(16.0)
    assert cycle > ideal_cycle
    assert pipe.balance_factor(_STEP, _NUM_LAYERS) == pytest.approx(ideal_cycle / cycle)


def test_balance_degrades_as_stages_multiply():
    factors = [_pipe(pp).balance_factor(_STEP, _NUM_LAYERS) for pp in (1, 2, 4, 8)]
    assert factors == sorted(factors, reverse=True)
    assert factors[0] == 1.0
    assert factors[-1] < 0.9


def test_uneven_layer_split_costs_throughput():
    """64 layers over 6 stages -> the 11-layer stages gate the cycle."""
    balanced_only = {"generation_qkv_gemm": _PER_LAYER_MS}  # no head, isolate the split
    assert _pipe(6).balance_factor(balanced_only, _NUM_LAYERS) == pytest.approx((64 / 6) / 11)


def test_perfectly_balanced_step_keeps_full_speedup():
    """A step with no first/last-stage ops and an even split loses nothing."""
    assert _pipe(8).balance_factor({"generation_qkv_gemm": _PER_LAYER_MS}, _NUM_LAYERS) == pytest.approx(1.0)


def test_cycle_charges_one_hop_unless_overlapped():
    step = dict(_STEP, generation_p2p=7.0)
    assert _pipe(8).cycle_time(step, _NUM_LAYERS) == pytest.approx(16.0 + 1.0)
    assert _pipe(8, p2p_overlap=True).cycle_time(step, _NUM_LAYERS) == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# PipelineSteadyState: pipe fill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m,expected", [(1, 0.125), (2, 0.25), (4, 0.5), (8, 1.0), (16, 1.0)])
def test_fill_factor_saturates_at_pp_size(m, expected):
    assert _pipe(8, num_microbatches=m).fill_factor() == pytest.approx(expected)


def test_default_microbatches_fills_the_pipe():
    """None reproduces the historical 'always exactly full' assumption."""
    assert _pipe(8).fill_factor() == 1.0


def test_efficiency_is_balance_times_fill():
    pipe = _pipe(8, num_microbatches=4)
    assert pipe.efficiency(_STEP, _NUM_LAYERS) == pytest.approx(pipe.balance_factor(_STEP, _NUM_LAYERS) * 0.5)


def test_rejects_nonpositive_microbatches():
    with pytest.raises(ValueError, match="num_microbatches must be >= 1"):
        _pipe(4, num_microbatches=0)


# ---------------------------------------------------------------------------
# The layout / steady-state boundary
# ---------------------------------------------------------------------------


def test_layout_carries_no_scheduling_policy():
    """The rules must not carry the steady-state collapse.

    ``PipelineLayout`` states where work lives; the factors are only valid
    under the mean-field assumption that one step shape describes every stage.
    Letting occupancy leak onto the layout would make the rules unusable by any
    derivation that does not share that assumption.
    """
    layout = PipelineLayout(pp_size=8)
    for occupancy_attr in ("fill_factor", "balance_factor", "efficiency", "cycle_time", "num_microbatches"):
        assert not hasattr(layout, occupancy_attr), f"{occupancy_attr} belongs to PipelineSteadyState"


def test_steady_state_reuses_the_layout_it_was_given():
    """A custom partition must reach the cycle, not be silently recomputed."""
    layout = PipelineLayout(pp_size=2, partition=(36, 28))
    pipe = PipelineSteadyState(layout=layout)
    assert pipe.layer_partition(_NUM_LAYERS) == (36, 28)
    assert pipe.layout is layout
    # 36 vs 28 layers of 1.0 ms/layer, head (8.0) on the last stage:
    # stage times are 36.5 (incl. embedding) and 36.0 -> the split compensates.
    assert pipe.cycle_time(_STEP, _NUM_LAYERS) == pytest.approx(36.5)


# ---------------------------------------------------------------------------
# pp_size == 1 identity
# ---------------------------------------------------------------------------


def test_single_stage_is_exactly_neutral():
    pipe = _pipe(1)
    assert pipe.balance_factor(_STEP, _NUM_LAYERS) == 1.0
    assert pipe.fill_factor() == 1.0
    assert pipe.efficiency(_STEP, _NUM_LAYERS) == 1.0


def test_empty_step_is_neutral():
    assert _pipe(8).balance_factor({}, _NUM_LAYERS) == 1.0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_describe_reports_the_critical_stage():
    d = _pipe(8).describe(_STEP, _NUM_LAYERS)
    assert d["critical_stage"] == 7  # the lm_head stage
    assert d["partition"] == [8] * 8
    assert d["cycle_time_ms"] > d["ideal_cycle_ms"]
    assert 0.0 < d["efficiency"] < 1.0
    assert d["efficiency"] == pytest.approx(d["balance_factor"] * d["fill_factor"])


def test_warns_when_a_small_scale_op_is_treated_as_per_layer(caplog):
    class _Op:
        def __init__(self, name, scale):
            self._name = name
            self._scale_factor = scale

    ops = [_Op("generation_qkv_gemm", _NUM_LAYERS), _Op("generation_mystery_head", 1)]
    with caplog.at_level("WARNING"):
        warn_on_unclassified_ops(ops, _NUM_LAYERS)
    assert "generation_mystery_head" in caplog.text
    assert "generation_qkv_gemm" not in caplog.text
