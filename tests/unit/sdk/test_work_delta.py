# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of the intra-batch prefill work-delta model.

The properties pinned here are the ones a plausible refactor would break
silently: which rows enter which column, what each regime can identify on its
own, and the all-or-nothing shape of applying a price.
"""

import pytest

from aiconfigurator_core.sdk.work_delta import (
    CellFit,
    CoefficientField,
    Measurement,
    classify,
    idx_work,
    mla_work,
    plan_cell,
    predict_delta,
    segments_for,
    solve_cell,
    work_columns,
)

TOPK = 128


# --------------------------------------------------------------- the basis


def test_scoring_work_is_not_gated_on_the_bound():
    """The hardware has no short-circuit, and this function models the
    hardware. Which rows are CHARGED is work_columns' decision, not this one."""
    assert idx_work(64, 0, TOPK) == 64 * 64 / 2
    assert idx_work(64, 0, 1 << 30) == idx_work(64, 0, TOPK)


def test_attention_work_is_capped_at_the_bound():
    assert mla_work(100, TOPK, TOPK) == 100 * TOPK
    assert mla_work(100, 4 * TOPK, TOPK) == 100 * TOPK


def test_only_rows_crossing_the_bound_enter_the_scoring_column():
    """x_idx prices crossing the bound, not running the indexer.

    Both rows here score candidates -- the hardware runs the indexer on each --
    but neither crosses topk, so the deviation they leave in x_idx is zero and
    their whole cost is carried by x_mla. Including them would make a pure
    unsaturated segment identify nothing, since within it the scan and the
    attention are the same trapezoid.
    """
    rows = [(96, 0), (32, 0)]
    x_idx, x_mla = work_columns(rows, 64, 0, TOPK)
    assert x_idx == 0.0
    assert x_mla != 0.0
    assert all(idx_work(s, p, TOPK) > 0 for s, p in rows)


def test_a_crossing_row_moves_the_scoring_column():
    rows = [(200, 0), (24, 0), (24, 0), (24, 0)]
    x_idx, _ = work_columns(rows, 68, 0, TOPK)
    assert x_idx > 0.0


def test_uniform_batch_deviates_in_neither_column():
    assert work_columns([(64, 0)] * 4, 64, 0, TOPK) == (0.0, 0.0)


def test_the_subtrahend_follows_the_average_point_not_the_batch():
    """A dense average point never touches the scoring column, so a batch that
    pushes rows across the bound shows the whole crossing as a gain rather than
    a difference against some baseline it never had."""
    x_idx, _ = work_columns([(200, 0), (24, 0), (24, 0), (24, 0)], 68, 0, TOPK)
    charged = idx_work(200, 0, TOPK)
    assert x_idx == pytest.approx(charged)


# --------------------------------------------------------------- regimes


def test_a_row_at_the_bound_is_dense():
    assert classify([(TOPK, 0)], TOPK) == "unsat"
    assert classify([(TOPK + 1, 0)], TOPK) == "sat"
    assert classify([(TOPK + 1, 0), (8, 0)], TOPK) == "mixed"


def test_available_segments_follow_from_the_conserved_total():
    assert segments_for(4, 8, 0, TOPK) == ("unsat",)
    assert segments_for(4, 64, 0, TOPK) == ("unsat", "mixed")
    assert segments_for(4, 256, 0, TOPK) == ("mixed", "sat")


# --------------------------------------------------------------- solving


def _measure(b, s_bar, p_bar, rows, c_idx, c_mla, topk=TOPK, noise=1e-6):
    x_idx, x_mla = work_columns(rows, s_bar, p_bar, topk)
    return Measurement(
        b,
        s_bar,
        p_bar,
        classify(rows, topk),
        s_bar + p_bar > topk,
        x_idx,
        x_mla,
        c_idx * x_idx + c_mla * x_mla,
        noise=noise,
    )


def test_a_pure_unsaturated_segment_identifies_the_attention_price_alone():
    """x_idx is zero throughout, so one rung fixes c_mla and says nothing
    about c_idx."""
    ms = [_measure(4, 64, 0, [(96, 0), (48, 0), (48, 0), (64, 0)], 3.0, 7.0)]
    fit = solve_cell(4, 64, 0, False, ms)
    assert fit.c_mla == pytest.approx(7.0)
    assert fit.c_idx is None


def test_a_pure_saturated_segment_identifies_the_crossing_price_alone():
    rows = [(400, 0), (200, 0), (200, 0), (224, 0)]
    assert classify(rows, TOPK) == "sat"
    fit = solve_cell(4, 256, 0, True, [_measure(4, 256, 0, rows, 3.0, 7.0)])
    assert fit.c_idx == pytest.approx(3.0)


def test_mixed_rungs_close_the_price_the_pure_segment_left_open():
    """One pure rung and one mixed rung determine both prices; the planner's
    second mixed rung is what leaves a residual to read."""
    pure = [(96, 0), (48, 0), (48, 0), (64, 0)]
    mixed_a = [(200, 0), (24, 0), (16, 0), (16, 0)]
    mixed_b = [(180, 0), (32, 0), (24, 0), (20, 0)]
    ms = [_measure(4, 64, 0, r, 3.0, 7.0) for r in (pure, mixed_a, mixed_b)]
    fit = solve_cell(4, 64, 0, False, ms)
    assert fit.c_mla == pytest.approx(7.0)
    assert fit.c_idx == pytest.approx(3.0)
    assert fit.accepted


def test_a_cell_whose_labels_are_all_noise_is_rejected():
    rows = [(96, 0), (48, 0), (48, 0), (64, 0)]
    x_idx, x_mla = work_columns(rows, 64, 0, TOPK)
    quiet = Measurement(4, 64, 0, classify(rows, TOPK), False, x_idx, x_mla, 0.0, noise=1.0)
    fit = solve_cell(4, 64, 0, False, [quiet])
    assert not fit.accepted
    assert fit.below_noise == 1


def test_residuals_are_reported_but_never_reject():
    """Collection affords one rung per pure segment, so an exactly determined
    cell has no residual to show. Gating on it would reject cells for having
    been over-sampled rather than for being wrong."""
    pure = [(96, 0), (48, 0), (48, 0), (64, 0)]
    fit = solve_cell(4, 64, 0, False, [_measure(4, 64, 0, pure, 3.0, 7.0)])
    assert fit.residuals == {}
    assert fit.accepted


# --------------------------------------------------------------- applying


def _prices(c_idx, c_mla):
    return CellFit(4, 64, 0, False, c_idx=c_idx, c_mla=c_mla)


def test_an_unpriced_column_that_moves_work_vetoes_the_correction():
    """Dropping it would leave the priced column to explain its work too."""
    assert predict_delta(1e9, 1e9, _prices(1e-6, None)) == 0.0
    assert predict_delta(1e9, 1e9, _prices(None, 1e-6)) == 0.0


def test_an_unpriced_column_carrying_no_work_is_harmless():
    """A purely unsaturated cell never prices the crossing, because none of its
    batches can cross -- and that same fact makes the missing price multiply
    zero. Refusing here would throw away a correction that is already whole."""
    assert predict_delta(0.0, 1e10, _prices(None, 3e-6)) == pytest.approx(3e4)


def test_a_negative_price_vetoes_the_correction():
    assert predict_delta(1e9, 1e9, _prices(-1e-6, 1e-6)) == 0.0
    assert predict_delta(1e9, 1e9, _prices(1e-6, -1e-6)) == 0.0


def test_a_deviation_below_the_gates_is_not_priced():
    assert predict_delta(1.0, 1.0, _prices(1e-6, 1e-6)) == 0.0


def test_a_correction_under_the_jitter_is_withheld():
    """Below the spread it would have to be checked against, a correction
    cannot be verified either way."""
    delta = predict_delta(1e10, 0.0, _prices(1e-6, 1e-6), noise=1e9)
    assert delta == 0.0


def test_a_priced_correction_is_the_sum_of_its_columns():
    assert predict_delta(1e10, 1e10, _prices(2e-6, 3e-6)) == pytest.approx(5e4)


# --------------------------------------------------------------- the field


def _fit(b, c_idx=1e-6, c_mla=2e-6):
    return CellFit(b, 256, 0, True, c_idx=c_idx, c_mla=c_mla)


def test_a_calibrated_cell_answers_itself():
    field = CoefficientField({(8, 256, 0): _fit(8)}, TOPK)
    got = field.at(8, 256, 0)
    assert got is not None and got.source == "calibrated"


def test_a_query_between_two_calibrated_sizes_interpolates():
    field = CoefficientField({(4, 256, 0): _fit(4, 1e-6, 2e-6), (16, 256, 0): _fit(16, 3e-6, 6e-6)}, TOPK)
    got = field.at(10, 256, 0)
    assert got is not None
    assert got.c_idx == pytest.approx(2e-6)
    assert got.c_mla == pytest.approx(4e-6)


def test_a_query_outside_the_bracket_declines():
    """Single-ended extrapolation is where every measured regression came
    from, on batches that were already fine."""
    field = CoefficientField({(4, 256, 0): _fit(4)}, TOPK)
    assert field.at(16, 256, 0) is None


def test_a_negative_price_at_either_end_voids_the_query():
    field = CoefficientField({(4, 256, 0): _fit(4, -1e-6, 2e-6), (16, 256, 0): _fit(16)}, TOPK)
    assert field.at(10, 256, 0) is None


def test_a_column_only_one_end_priced_voids_the_query():
    """A real price at one size with nothing to blend it against would have to
    be extrapolated from that end alone."""
    field = CoefficientField({(4, 256, 0): _fit(4, 1e-6, None), (16, 256, 0): _fit(16)}, TOPK)
    assert field.at(10, 256, 0) is None


def test_a_column_neither_end_priced_is_carried_through_unpriced():
    """Both ends failing to identify a column is what a purely unsaturated
    average point looks like; the decision then belongs to predict_delta,
    which knows whether the column actually moves any work."""
    field = CoefficientField({(4, 256, 0): _fit(4, None, 2e-6), (16, 256, 0): _fit(16, None, 6e-6)}, TOPK)
    got = field.at(10, 256, 0)
    assert got is not None and got.c_idx is None
    assert got.c_mla == pytest.approx(4e-6)
    assert predict_delta(0.0, 1e10, got) == pytest.approx(4e4)
    assert predict_delta(1e10, 1e10, got) == 0.0


def test_an_interpolated_pair_applies_through_the_same_gates():
    """The interpolated path and the exact-hit path share one application, so
    a carried price is held to what a calibrated one is."""
    field = CoefficientField({(4, 256, 0): _fit(4, -1e-6, 2e-6), (16, 256, 0): _fit(16)}, TOPK)
    assert field.at(10, 256, 0) is None
    ok = CoefficientField({(4, 256, 0): _fit(4), (16, 256, 0): _fit(16)}, TOPK)
    got = ok.at(10, 256, 0)
    assert predict_delta(1e10, 1e10, got) > 0.0


def test_bracketing_points_reports_what_the_field_can_answer():
    field = CoefficientField({(4, 256, 0): _fit(4), (16, 256, 0): _fit(16), (8, 512, 0): _fit(8)}, TOPK)
    assert field.bracketing_points() == [(256, 0)]


# --------------------------------------------------------------- planning


def test_a_plan_conserves_both_totals_exactly():
    plan = plan_cell(8, 4096, 1024, 2048, kv_block=64)
    assert plan is not None
    for batch in plan.batches:
        assert batch.totals == (8 * 4096, 8 * 1024)


def test_planned_batches_stay_inside_the_regime_they_claim():
    plan = plan_cell(8, 4096, 1024, 2048, kv_block=64)
    assert plan is not None
    for batch in plan.batches:
        assert classify(batch.rows, 2048) == batch.regime


def test_a_plan_carries_two_columns():
    plan = plan_cell(8, 4096, 1024, 2048, kv_block=64)
    assert plan is not None
    assert all(len(batch.columns) == 2 for batch in plan.batches)


def test_the_minimum_plan_solves_both_prices():
    """One pure rung plus two mixed rungs is what collection affords, and it
    has to be enough."""
    plan = plan_cell(8, 4096, 1024, 2048, kv_block=64)
    assert plan is not None
    ms = [_measure(8, 4096, 1024, b.rows, 9.6e-7, 4.9e-6, topk=2048, noise=1e-3) for b in plan.batches]
    fit = solve_cell(8, 4096, 1024, True, ms)
    assert fit.c_idx == pytest.approx(9.6e-7)
    assert fit.c_mla == pytest.approx(4.9e-6)
    assert fit.accepted
