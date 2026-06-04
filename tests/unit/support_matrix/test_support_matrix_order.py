# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tools.support_matrix.support_matrix import STATUS_PASS, SupportMatrix

pytestmark = pytest.mark.unit


def _result(system: str) -> tuple[str, str, str, str, str, str, str, None, str]:
    """Build a minimal passing support-matrix result row for a system."""
    return ("test/model", "TestArchitecture", system, "trtllm", "1.0.0", "agg", STATUS_PASS, None, "test-command")


def test_save_results_to_csv_writes_manifest_in_display_order(tmp_path):
    """Split support-matrix output should preserve product-priority file order."""
    results = [
        _result("b60"),
        _result("a100_sxm"),
        _result("l40s"),
        _result("h100_sxm"),
        _result("h200_sxm"),
        _result("rtx_pro_6000_server"),
        _result("gb300"),
        _result("b300_sxm"),
        _result("gb200"),
        _result("b200_sxm"),
    ]

    support_matrix = SupportMatrix.__new__(SupportMatrix)
    support_matrix.save_results_to_csv(results, str(tmp_path))

    with (tmp_path / "index.json").open() as f:
        manifest = json.load(f)

    assert manifest["files"] == [
        "b200_sxm.csv",
        "gb200.csv",
        "b300_sxm.csv",
        "gb300.csv",
        "rtx_pro_6000_server.csv",
        "h200_sxm.csv",
        "h100_sxm.csv",
        "l40s.csv",
        "a100_sxm.csv",
        "b60.csv",
    ]
