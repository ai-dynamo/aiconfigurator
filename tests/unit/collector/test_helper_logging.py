# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from collector.helper import log_scope_dirname

pytestmark = pytest.mark.unit


def test_short_scope_keeps_verbatim_join():
    assert log_scope_dirname(["gemm", "moe"]) == "gemm+moe"


def test_long_scope_is_capped_below_filesystem_limit():
    ops = [f"dsv4_very_long_module_name_{i}" for i in range(19)]
    name = log_scope_dirname(ops)
    assert name == f"{ops[0]}+18ops"
    assert len(name) + len("_20260709_175206") < 255
