# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PARQUET_DIFF = Path(__file__).resolve().parents[3] / "tools" / "perf_database" / "parquet_diff.py"


@pytest.fixture
def parquet_diff_module():
    spec = importlib.util.spec_from_file_location("parquet_diff", PARQUET_DIFF)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_legacy_perf_policy_flags_added_and_modified_text_files(parquet_diff_module):
    entries = [
        parquet_diff_module.DiffEntry("A", "src/aiconfigurator/systems/data/h100/gemm_perf.txt"),
        parquet_diff_module.DiffEntry("M", "src/aiconfigurator/systems/data/h100/moe_perf.txt"),
        parquet_diff_module.DiffEntry("D", "src/aiconfigurator/systems/data/h100/nccl_perf.txt"),
        parquet_diff_module.DiffEntry("A", "src/aiconfigurator/systems/data/h100/gemm_perf.parquet"),
    ]

    legacy_changes = parquet_diff_module.find_legacy_perf_changes(entries)

    assert [entry.status for entry in legacy_changes] == ["A", "M"]
    assert parquet_diff_module.should_fail_strict([], legacy_changes)


def test_legacy_perf_policy_allows_text_file_deletions(parquet_diff_module):
    entries = [
        parquet_diff_module.DiffEntry("D", "src/aiconfigurator/systems/data/h100/gemm_perf.txt"),
    ]

    legacy_changes = parquet_diff_module.find_legacy_perf_changes(entries)
    report = parquet_diff_module.render_report(
        base_ref="origin/main",
        head_ref="HEAD",
        entries=entries,
        comparisons=[],
        legacy_perf_changes=legacy_changes,
    )

    assert legacy_changes == []
    assert not parquet_diff_module.should_fail_strict([], legacy_changes)
    assert "- Legacy `*_perf.txt` files added or modified: 0" in report
