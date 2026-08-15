# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic PerfDatabase construction sanity.

The per-call query behaviour this file used to pin on the stub/comprehensive
fixtures (query_gemm exact hits, empirical util calibration, cross-profile
borrowing, interpolation faithfulness, query_trtllm_alltoall normalization
and its not-enabled zero, custom-allreduce/nccl/p2p mode math, the HYBRID
fallback) retired to the compiled engine with #1357 PR-5 —
query_trtllm_alltoall is a tombstone, the rest are engine-routed shims whose
values are anchored by tests/cross_package/test_query_shim_baseline.py and
the frozen parity goldens.
"""

import pytest

pytestmark = pytest.mark.unit


def test_system_spec_was_loaded_correctly(stub_perf_db):
    """
    Sanity check: PerfDatabase.system_spec should be exactly what our patched yaml.load returned.
    """
    spec = stub_perf_db.system_spec
    assert isinstance(spec, dict)
    assert spec["gpu"]["bfloat16_tc_flops"] == 1_000.0
    assert spec["node"]["inter_node_bw"] == 100.0
