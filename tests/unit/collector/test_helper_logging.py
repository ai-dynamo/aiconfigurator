# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from collector.helper import _collector_log_dir_name

pytestmark = pytest.mark.unit


def test_collector_log_dir_name_bounds_long_scope():
    name = _collector_log_dir_name([f"very_long_op_name_{i:02d}" for i in range(20)], "20260601_041331")

    assert len(name) < 255
    assert name.endswith("_20260601_041331")


def test_collector_log_dir_name_keeps_short_scope_readable():
    assert _collector_log_dir_name(["gemm", "moe"], "20260601_041331") == "gemm+moe_20260601_041331"
