# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report-table header semantics for the queueing (pass-calendar) columns.

The header IS the semantics contract: legacy mode shows the blended-mean
estimate as TTFT(avg); percentile mode names the exact stored quantile.
"""

import pandas as pd
import pytest

from aiconfigurator.cli.report_and_save import _itl_cell, _itl_display, _ttft_cell, _ttft_header

pytestmark = pytest.mark.unit


class TestQueueingHeaders:
    def test_legacy_mode_headers(self):
        assert _ttft_header(None) == "TTFT(avg)"
        assert _itl_display(None) == ("ITL(P99)", "itl_p99")

    def test_percentile_headers_name_the_quantile(self):
        assert _ttft_header(0.5) == "TTFT(P50)"
        assert _ttft_header(0.999) == "TTFT(P999)"
        assert _itl_display(0.5) == ("ITL(P50)", "itl_p50")
        assert _itl_display(0.99) == ("ITL(P99)", "itl_p99")


class TestQueueingCells:
    def test_ttft_cell_legacy_reads_legacy_column(self):
        row = {"ttft": 123.456, "ttft_steady_p50": 99.0}
        assert _ttft_cell(row, None) == "123.46"

    def test_ttft_cell_percentile_reads_stored_quantile(self):
        row = {"ttft": 123.456, "ttft_steady_p50": 99.0}
        assert _ttft_cell(row, 0.5) == "99.00"

    def test_ttft_cell_falls_back_without_queueing_columns(self):
        assert _ttft_cell({"ttft": 123.456}, 0.5) == "123.46"

    def test_itl_cell_handles_missing_distribution(self):
        assert _itl_cell({"itl_p99": pd.NA}, 0.99) == "-"
        assert _itl_cell({"itl_p99": 42.0}, 0.99) == "42.00"
