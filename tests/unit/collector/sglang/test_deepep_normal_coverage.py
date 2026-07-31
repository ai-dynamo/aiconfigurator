# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Axis resolution and coverage gating for the DeepEP normal-mode collector.

The coverage gate is the only layer that can see the ``(sms, num_tokens)`` axes:
``provenance.case_plan_hash`` covers shape-level case ids alone and
``provenance.derive_table_status`` consults failures rather than coverage, so an
under-covered sweep otherwise finalizes as a ``status: complete`` table with a
matching case-plan hash. These tests need no GPU.
"""

import csv

import pytest

from collector.wideep.sglang.collect_deepep_normal import (
    DEEPEP_NORMAL_DEFAULT_SMS,
    DEEPEP_NORMAL_TOKENS,
    _sms_list,
    _tokens_list,
    _verify_axis_coverage,
)

_FIELDNAMES = (
    "node_num",
    "hidden_size",
    "num_token",
    "num_topk",
    "num_experts",
    "dispatch_sms",
    "dispatch_transmit_us",
)


def _write_perf_csv(path, rows):
    """Write a perf CSV shaped like the one ``helper.log_perf`` appends to."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)


_SHAPES = [(2048, 128, 8), (7168, 256, 8)]
_SMS = [4, 20]
_TOKENS = [1, 64, 1024]


def _rows(shapes, sms_list, tokens):
    return [
        {
            "node_num": 1,
            "hidden_size": hidden,
            "num_token": num_token,
            "num_topk": num_topk,
            "num_experts": num_experts,
            "dispatch_sms": sms,
            "dispatch_transmit_us": 42.0,
        }
        for (hidden, num_experts, num_topk) in shapes
        for sms in sms_list
        for num_token in tokens
    ]


@pytest.mark.unit
class TestSmsList:
    def test_default_is_the_grid_the_consumer_reads(self):
        # MoEDispatch defaults to sms=12, which takes the 2-D (sms, num_tokens)
        # grid branch, so the whole axis has to be collected. All six committed
        # 0.5.10 datasets carry these six values at node_num=1.
        assert DEEPEP_NORMAL_DEFAULT_SMS == (4, 8, 12, 16, 20, 24)
        assert _sms_list() == [4, 8, 12, 16, 20, 24]

    @pytest.mark.parametrize("raw,expected", [("20", [20]), ("4,8,12", [4, 8, 12]), ("4 8  12", [4, 8, 12])])
    def test_override_parses_commas_and_whitespace(self, monkeypatch, raw, expected):
        monkeypatch.setenv("DEEPEP_NORMAL_SMS", raw)
        assert _sms_list() == expected

    def test_blank_override_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("DEEPEP_NORMAL_SMS", "   ")
        assert _sms_list() == list(DEEPEP_NORMAL_DEFAULT_SMS)

    def test_odd_sms_rejected_at_startup(self, monkeypatch):
        # Buffer.set_num_sms asserts evenness, and it is only reached on the
        # default-config iteration of the tuning loop -- mid-collective, well into
        # a shape. Reject before any GPU work.
        monkeypatch.setenv("DEEPEP_NORMAL_SMS", "4,7,20")
        with pytest.raises(RuntimeError, match=r"must be even.*\[7\]"):
            _sms_list()

    def test_all_odd_values_reported_together(self, monkeypatch):
        monkeypatch.setenv("DEEPEP_NORMAL_SMS", "3,4,9")
        with pytest.raises(RuntimeError, match=r"\[3, 9\]"):
            _sms_list()


@pytest.mark.unit
class TestTokensList:
    def test_default_is_the_committed_ladder(self):
        assert _tokens_list() == list(DEEPEP_NORMAL_TOKENS)
        assert _tokens_list()[-1] == 131072

    def test_override_allows_trimming_the_expensive_tail(self, monkeypatch):
        monkeypatch.setenv("DEEPEP_NORMAL_TOKENS", "1,2,4,8")
        assert _tokens_list() == [1, 2, 4, 8]

    def test_blank_override_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("DEEPEP_NORMAL_TOKENS", "")
        assert _tokens_list() == list(DEEPEP_NORMAL_TOKENS)


@pytest.mark.unit
class TestVerifyAxisCoverage:
    def test_full_grid_passes_and_counts_shapes(self, tmp_path):
        path = tmp_path / "wideep_deepep_normal_perf.txt"
        _write_perf_csv(path, _rows(_SHAPES, _SMS, _TOKENS))
        assert _verify_axis_coverage(path, _SMS, _TOKENS) == 2

    def test_missing_sms_is_rejected(self, tmp_path):
        # The campaign trap: a sweep that ran one sms instead of the full grid.
        path = tmp_path / "wideep_deepep_normal_perf.txt"
        _write_perf_csv(path, _rows(_SHAPES, [20], _TOKENS))
        with pytest.raises(RuntimeError, match=r"under-covers the swept axes") as excinfo:
            _verify_axis_coverage(path, _SMS, _TOKENS)
        assert "missing sms=[4]" in str(excinfo.value)
        # Every short shape is named, not just the first.
        assert "hidden=2048" in str(excinfo.value)
        assert "hidden=7168" in str(excinfo.value)

    def test_missing_tokens_is_rejected(self, tmp_path):
        # A shape truncated mid-ladder, e.g. an OOM at the top of the token sweep.
        path = tmp_path / "wideep_deepep_normal_perf.txt"
        _write_perf_csv(path, _rows(_SHAPES, _SMS, [1, 64]))
        with pytest.raises(RuntimeError, match=r"missing sms=\[\] tokens=\[1024\]"):
            _verify_axis_coverage(path, _SMS, _TOKENS)

    def test_partial_shape_does_not_mask_a_complete_one(self, tmp_path):
        path = tmp_path / "wideep_deepep_normal_perf.txt"
        rows = _rows([_SHAPES[0]], _SMS, _TOKENS)
        rows += _rows([_SHAPES[1]], [20], _TOKENS)
        _write_perf_csv(path, rows)
        with pytest.raises(RuntimeError) as excinfo:
            _verify_axis_coverage(path, _SMS, _TOKENS)
        message = str(excinfo.value)
        assert "hidden=7168" in message
        assert "hidden=2048" not in message

    def test_absent_shape_does_not_fire(self, tmp_path):
        """A shape skipped for `num_experts % num_ranks` or a buffer-alloc failure
        is absent from the CSV entirely. That is whole-shape loss, reported by the
        caller against the orchestrator's tally, not an axis being short."""
        path = tmp_path / "wideep_deepep_normal_perf.txt"
        _write_perf_csv(path, _rows([_SHAPES[0]], _SMS, _TOKENS))
        assert _verify_axis_coverage(path, _SMS, _TOKENS) == 1

    def test_header_only_file_is_rejected(self, tmp_path):
        """Dropping `metrics_out` used to yield a run where every shape reported
        success and nothing was written. Zero data rows must never pass."""
        path = tmp_path / "wideep_deepep_normal_perf.txt"
        _write_perf_csv(path, [])
        with pytest.raises(RuntimeError, match=r"wrote no data rows"):
            _verify_axis_coverage(path, _SMS, _TOKENS)
