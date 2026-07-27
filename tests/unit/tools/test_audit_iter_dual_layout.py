# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""_iter_data_files must yield identical tuples for legacy and family layouts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "perf_database"))
from audit_kernel_source import _iter_data_files

pytestmark = pytest.mark.unit


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"stub")


def test_iter_data_files_both_layouts(tmp_path):
    _touch(tmp_path, "h200_sxm/sglang/0.5.12/gemm_perf.parquet")  # legacy
    _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")  # family
    _touch(tmp_path, "h200_sxm/comm/nccl/2.19/nccl_perf.parquet")  # comm family
    got = {(s, b, v, p.name) for s, b, v, p in _iter_data_files(tmp_path)}
    assert ("h200_sxm", "sglang", "0.5.12", "gemm_perf.parquet") in got
    assert ("h200_sxm", "sglang", "0.5.14", "gemm_perf.parquet") in got
    # nccl stays excluded exactly as the legacy _SKIP_BACKEND_DIRS did
    assert not any(b == "nccl" for _, b, _, _ in got)
