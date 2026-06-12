# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from spica.planner import SCALING_POLICIES, throughput_intervals


def test_scaling_policy_decode():
    p = SCALING_POLICIES["hybrid_180_5"]
    assert p.enable_throughput and p.enable_load
    assert p.throughput_interval_s == 180
    assert p.load_interval_s == 5
    assert SCALING_POLICIES["disabled"].enable_throughput is False
    assert SCALING_POLICIES["throughput_600_5"].throughput_interval_s == 600


def test_throughput_intervals_distinct_enabled_only():
    ids = ["disabled", "load_180_5", "throughput_180_5", "hybrid_600_5", "throughput_600_5"]
    # enabled-throughput candidates: throughput_180_5 (180), hybrid_600_5 (600),
    # throughput_600_5 (600) -> {180, 600}
    assert throughput_intervals(ids) == [180, 600]


def test_throughput_intervals_empty_when_none_enabled():
    assert throughput_intervals(["disabled", "load_180_5", "load_180_10"]) == []
