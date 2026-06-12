# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spica: Replay-backed smart sweeper for Dynamo deployments (Profiler V2)."""

from __future__ import annotations

from .config import (
    Candidate,
    OptimizationGoal,
    OptimizationTarget,
    SearchSpace,
    SLATarget,
    SmartSearchConfig,
    SweepConfig,
    Workload,
)
from .search import run_smart_search

__all__ = [
    "Candidate",
    "OptimizationGoal",
    "OptimizationTarget",
    "SearchSpace",
    "SLATarget",
    "SmartSearchConfig",
    "SweepConfig",
    "Workload",
    "run_smart_search",
]
