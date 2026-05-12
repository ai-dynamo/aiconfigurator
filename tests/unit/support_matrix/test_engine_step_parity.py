# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from tools.support_matrix.support_matrix import _compare_pareto_dfs

pytestmark = pytest.mark.unit


def test_compare_pareto_dfs_allows_metric_drift_within_tolerance():
    python_df = pd.DataFrame(
        [
            {"model": "model-a", "tp": 2, "ttft": 100.0, "tpot": 10.0, "tokens/s/gpu": 50.0},
            {"model": "model-a", "tp": 1, "ttft": 120.0, "tpot": 12.0, "tokens/s/gpu": 45.0},
        ]
    )
    rust_df = pd.DataFrame(
        [
            {"model": "model-a", "tp": 1, "ttft": 121.0, "tpot": 12.1, "tokens/s/gpu": 44.9},
            {"model": "model-a", "tp": 2, "ttft": 101.0, "tpot": 10.2, "tokens/s/gpu": 49.5},
        ]
    )

    assert _compare_pareto_dfs(python_df, rust_df, rtol=0.05, atol=1e-3) is None


def test_compare_pareto_dfs_reports_metric_drift_outside_tolerance():
    python_df = pd.DataFrame([{"model": "model-a", "tp": 1, "ttft": 100.0, "tpot": 10.0}])
    rust_df = pd.DataFrame([{"model": "model-a", "tp": 1, "ttft": 125.0, "tpot": 10.0}])

    mismatch = _compare_pareto_dfs(python_df, rust_df, rtol=0.05, atol=1e-3)

    assert mismatch is not None
    assert "ttft[0]" in mismatch
    assert "beyond tolerance" in mismatch


def test_compare_pareto_dfs_reports_configuration_mismatch():
    python_df = pd.DataFrame([{"model": "model-a", "tp": 1, "ttft": 100.0, "tpot": 10.0}])
    rust_df = pd.DataFrame([{"model": "model-a", "tp": 2, "ttft": 100.0, "tpot": 50.0}])

    mismatch = _compare_pareto_dfs(python_df, rust_df)

    assert mismatch is not None
    assert "selected different configurations" in mismatch


def test_compare_pareto_dfs_allows_different_rows_with_similar_frontier_envelope():
    python_df = pd.DataFrame(
        [
            {"model": "model-a", "tp": 1, "tokens/s/user": 100.0, "ttft": 120.0, "tpot": 10.0},
            {"model": "model-a", "tp": 2, "tokens/s/user": 90.0, "ttft": 110.0, "tpot": 11.0},
        ]
    )
    rust_df = pd.DataFrame([{"model": "model-a", "tp": 4, "tokens/s/user": 95.0, "ttft": 115.0, "tpot": 10.5}])

    assert _compare_pareto_dfs(python_df, rust_df) is None
