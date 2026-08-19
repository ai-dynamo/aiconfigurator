# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""_scan_one_file must skip blank AND literal `<unknown>` kernel_source rows,
exactly as the module docstring promises — neither may leak into records (and
from there into the generated manifest as a `shared` tier)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "perf_database"))
from generate_perf_data_reuse_manifest import _scan_one_file

pytestmark = pytest.mark.unit


def test_scan_skips_blank_and_unknown_kernel_source(tmp_path):
    table = tmp_path / "gemm_perf.txt"
    table.write_text(
        "framework,version,kernel_source,latency,m\n"
        "trtllm,1.0.0,good_kernel,1.5,16\n"
        "trtllm,1.0.0,<unknown>,1.5,32\n"
        "trtllm,1.0.0,,1.5,64\n"
    )
    out = _scan_one_file("h200_sxm", "trtllm", "1.0.0", table)
    assert out.rows_scanned == 3
    assert out.rows_unnamed_kernel_source == 2
    kernel_sources = {rec[2] for rec in out.records}
    assert kernel_sources == {"good_kernel"}
