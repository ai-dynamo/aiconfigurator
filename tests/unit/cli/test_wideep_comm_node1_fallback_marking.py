# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A config using the WideEP communication node-1 fallback must say so where it is read.

The WideEP DeepEP tables answer an uncollected multi-node scale from
node_num=1 rows. That is allowed -- refusing removes WideEP from most systems
-- but the answer carries dedicated provenance, and these tests pin the path
from that provenance to the ranked config table, which is what a user actually
looks at. A warning in the log does not count.
"""

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiconfigurator.cli.report_and_save import (
    WIDEEP_COMM_NODE1_FALLBACK_MARK,
    _plot_worker_setup_table,
    _row_uses_wideep_comm_node1_fallback,
    log_final_summary,
)
from aiconfigurator.sdk.common import WIDEEP_COMM_NODE1_FALLBACK_COLUMN, WIDEEP_COMM_NODE1_FALLBACK_SOURCE
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary

pytestmark = pytest.mark.unit


def _agg_row(**overrides):
    row = {
        "backend": "sglang",
        "tokens/s/gpu": 100.0,
        "tokens/s/user": 50.0,
        "tokens/s/gpu_cluster": 100.0,
        "request_rate": 2.0,
        "ttft": 100.0,
        "request_latency": 200.0,
        "tpot": 10.0,
        "concurrency": 4.0,
        "num_total_gpus": 8,
        "tp": 1,
        "pp": 1,
        "dp": 8,
        "moe_tp": 1,
        "moe_ep": 8,
        "bs": 64,
        "power_w": 400.0,
    }
    row.update(overrides)
    return row


def _disagg_row(**overrides):
    row = _agg_row()
    for prefix in ("(p)", "(d)"):
        row.update(
            {
                f"{prefix}tp": 1,
                f"{prefix}pp": 1,
                f"{prefix}dp": 8,
                f"{prefix}moe_tp": 1,
                f"{prefix}moe_ep": 8,
                f"{prefix}bs": 32,
                f"{prefix}workers": 1,
            }
        )
    row.update(overrides)
    return row


def _table(rows, *, is_moe=True):
    return _plot_worker_setup_table(
        "agg",
        pd.DataFrame(rows),
        total_gpus=8,
        tpot_target=50.0,
        top=5,
        is_moe=is_moe,
        request_latency_target=None,
        show_power=False,
    )


class TestRankedTableWideepCommNode1FallbackMarking:
    def test_wideep_comm_node1_fallback_row_is_marked_and_explained(self):
        text = _table([_agg_row(**{WIDEEP_COMM_NODE1_FALLBACK_COLUMN: True})])
        assert f"1{WIDEEP_COMM_NODE1_FALLBACK_MARK}" in text
        assert "WideEP communication node-1 fallback" in text
        assert "single-node" in text

    def test_measured_row_carries_no_mark_and_no_footnote(self):
        """A clean run must not grow a caveat it has not earned."""
        text = _table([_agg_row(**{WIDEEP_COMM_NODE1_FALLBACK_COLUMN: False})])
        assert WIDEEP_COMM_NODE1_FALLBACK_MARK not in text
        assert "WideEP communication node-1 fallback" not in text

    def test_marking_is_per_row_not_per_table(self):
        """The whole point is telling proxied rows apart from measured ones."""
        rows = [
            _agg_row(**{"tokens/s/gpu_cluster": 200.0, WIDEEP_COMM_NODE1_FALLBACK_COLUMN: True}),
            _agg_row(**{"tokens/s/gpu_cluster": 100.0, WIDEEP_COMM_NODE1_FALLBACK_COLUMN: False}),
        ]
        lines = [ln for ln in _table(rows).splitlines() if "sglang" in ln]
        assert len(lines) == 2
        assert f"1{WIDEEP_COMM_NODE1_FALLBACK_MARK}" in lines[0]
        assert f"2{WIDEEP_COMM_NODE1_FALLBACK_MARK}" not in lines[1]

    def test_disagg_rows_are_marked_too(self):
        """Disagg rows are composed from worker candidates rather than a summary,
        so they take a separate path to the flag; an unmarked disagg table would
        read as measured."""
        text = _table([_disagg_row(**{WIDEEP_COMM_NODE1_FALLBACK_COLUMN: True})])
        assert "(p)parallel" in text  # confirm we exercised the disagg branch
        assert f"1{WIDEEP_COMM_NODE1_FALLBACK_MARK}" in text
        assert "WideEP communication node-1 fallback" in text

    def test_rows_predating_the_flag_are_not_marked(self):
        """No provenance is not evidence of extrapolation."""
        text = _table([_agg_row()])
        assert WIDEEP_COMM_NODE1_FALLBACK_MARK not in text


class TestRowUsesWideepCommNode1Fallback:
    def test_reads_the_explicit_flag(self):
        assert _row_uses_wideep_comm_node1_fallback({WIDEEP_COMM_NODE1_FALLBACK_COLUMN: True})
        assert not _row_uses_wideep_comm_node1_fallback({WIDEEP_COMM_NODE1_FALLBACK_COLUMN: False})

    def test_falls_back_to_the_per_ops_source_breakdown(self):
        row = {"_per_ops_source": {"mix_step": {"moe_dispatch": WIDEEP_COMM_NODE1_FALLBACK_SOURCE, "gemm": "silicon"}}}
        assert _row_uses_wideep_comm_node1_fallback(row)

    def test_generic_estimate_is_not_a_wideep_fallback(self):
        row = {"_per_ops_source": {"mix_step": {"gemm": "estimated"}}}
        assert not _row_uses_wideep_comm_node1_fallback(row)

    def test_mixed_alone_does_not_mark(self):
        """ "mixed" is the merge of differing sources for one op and arises
        benignly; treating it as extrapolation would over-report."""
        row = {"_per_ops_source": {"mix_step": {"gemm": "mixed", "moe_dispatch": "silicon"}}}
        assert not _row_uses_wideep_comm_node1_fallback(row)

    def test_missing_provenance_is_not_a_fallback(self):
        assert not _row_uses_wideep_comm_node1_fallback({})


class TestSummaryWideepCommNode1FallbackDetection:
    def _summary(self):
        return InferenceSummary(RuntimeConfig(batch_size=1, isl=128, osl=16))

    def test_wideep_comm_node1_fallback_in_a_phase_dict_is_detected(self):
        summary = self._summary()
        summary.set_generation_source_dict({"moe_dispatch": WIDEEP_COMM_NODE1_FALLBACK_SOURCE, "gemm": "silicon"})
        assert summary.uses_wideep_comm_node1_fallback()

    def test_wideep_comm_node1_fallback_nested_in_per_ops_source_is_detected(self):
        summary = self._summary()
        summary.set_per_ops_source({"decode": {"moe_dispatch": WIDEEP_COMM_NODE1_FALLBACK_SOURCE}})
        assert summary.uses_wideep_comm_node1_fallback()

    def test_generic_estimated_source_is_not_detected(self):
        summary = self._summary()
        summary.set_generation_source_dict({"gemm": "estimated"})
        assert not summary.uses_wideep_comm_node1_fallback()

    def test_all_measured_has_no_fallback(self):
        summary = self._summary()
        summary.set_context_source_dict({"gemm": "silicon"})
        summary.set_generation_source_dict({"gemm": "silicon", "attn": "empirical"})
        assert not summary.uses_wideep_comm_node1_fallback()

    def test_empty_summary_has_no_fallback(self):
        assert not self._summary().uses_wideep_comm_node1_fallback()


def _run_final_summary(caplog, *, uses_wideep_comm_node1_fallback: bool) -> str:
    task = MagicMock()
    task.primary_model_path = "deepseek-ai/DeepSeek-V3"
    task.is_moe = True
    task.tpot = 50.0
    task.request_latency = None
    task.backend_name = "sglang"
    task.total_gpus = 8
    best_configs = {
        "agg": pd.DataFrame([_agg_row(**{WIDEEP_COMM_NODE1_FALLBACK_COLUMN: uses_wideep_comm_node1_fallback})])
    }

    with caplog.at_level(logging.INFO, logger="aiconfigurator.cli.report_and_save"):
        log_final_summary(
            chosen_exp="agg",
            best_throughputs={"agg": 100.0},
            best_configs=best_configs,
            pareto_fronts={"agg": None},
            tasks={"agg": task},
            mode="default",
            top_n=1,
        )
    return "\n".join(r.message for r in caplog.records)


def test_headline_best_config_states_the_caveat(caplog):
    """The summary box numbers are the ones read in isolation, so the caveat
    has to sit next to them and not only under the table."""
    logged = _run_final_summary(caplog, uses_wideep_comm_node1_fallback=True)
    assert "WIDEEP COMM NODE-1 FALLBACK" in logged
    assert "optimistic" in logged


def test_headline_best_config_stays_quiet_when_measured(caplog):
    assert "WIDEEP COMM NODE-1 FALLBACK" not in _run_final_summary(caplog, uses_wideep_comm_node1_fallback=False)
