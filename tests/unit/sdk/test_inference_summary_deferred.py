# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deferred summary-row materialization (PR #1369 sweep optimization).

run_static/run_agg store a deferred row instead of building the summary
DataFrame eagerly.  Consumers may call get_static_info() before
get_summary_df() (the demo CLI and the former webapp static flow did), so
get_static_info() must materialize the deferred row itself rather than
assert on a DataFrame that was never built.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary

pytestmark = pytest.mark.unit


def _deferred_static_summary() -> InferenceSummary:
    """Mimic run_static(): deferred row set, no set_summary_df call."""
    summary = InferenceSummary(RuntimeConfig(isl=4000, osl=1000))
    data = dict.fromkeys(common.ColumnsStatic, 0)
    data.update({"model": "m", "tokens/s": 100.0, "tpot": 10.0})
    row = [[data[c] for c in common.ColumnsStatic]]
    summary.set_deferred_row(row, common.ColumnsStatic)
    summary.set_context_latency_dict({"gemm": 1.0})
    summary.set_generation_latency_dict({"gemm": 2.0})
    summary._memory = {"total": 10.0, "weights": 8.0}
    return summary


class TestDeferredRowMaterialization:
    def test_get_static_info_before_get_summary_df(self):
        """Regression: must not assert when the DataFrame is still deferred."""
        summary = _deferred_static_summary()
        perf_info, mem_info, *_ = summary.get_static_info()
        assert "throughput 100.00 tokens/s" in perf_info
        assert "Memory Usage" in mem_info

    def test_get_summary_df_builds_lazily_from_deferred_row(self):
        summary = _deferred_static_summary()
        df = summary.get_summary_df()
        assert df is not None
        assert df.loc[0, "tokens/s"] == 100.0
        # Second call returns the already-built frame
        assert summary.get_summary_df() is df
