# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode of the ``planner_scaling_policy`` composite presets.

Each preset id decodes to the four planner fields it stands for
(matching the Main Sweep Search Space table in the design proposal):
``enable_throughput_scaling``, ``enable_load_scaling``,
``throughput_adjustment_interval_seconds``, ``load_adjustment_interval_seconds``.

Only policies with throughput scaling enabled drive the load-predictor sweep,
so :func:`throughput_intervals` extracts the distinct throughput intervals to
sweep over.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingPolicy:
    enable_throughput: bool
    enable_load: bool
    throughput_interval_s: int | None
    load_interval_s: int | None


# preset id -> decoded policy (the {bool, bool, tput_interval, load_interval}
# tuples from the design proposal's planner_scaling_policy row).
SCALING_POLICIES: dict[str, ScalingPolicy] = {
    "disabled": ScalingPolicy(False, False, None, None),
    "throughput_180_5": ScalingPolicy(True, False, 180, 5),
    "throughput_600_5": ScalingPolicy(True, False, 600, 5),
    "load_180_5": ScalingPolicy(False, True, 180, 5),
    "load_180_10": ScalingPolicy(False, True, 180, 10),
    "hybrid_180_5": ScalingPolicy(True, True, 180, 5),
    "hybrid_600_5": ScalingPolicy(True, True, 600, 5),
}


def throughput_intervals(policy_ids: list[str]) -> list[int]:
    """Distinct throughput-adjustment intervals (seconds), sorted, across the
    given policy candidates that enable throughput scaling.

    Returns an empty list when no candidate enables throughput scaling — the
    load-predictor sweep is then unnecessary (it only matters for predictive
    throughput scaling).
    """
    intervals = {
        SCALING_POLICIES[p].throughput_interval_s
        for p in policy_ids
        if p in SCALING_POLICIES and SCALING_POLICIES[p].enable_throughput
    }
    return sorted(iv for iv in intervals if iv is not None)
