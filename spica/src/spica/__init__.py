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
from .load_predictor_sweep import (
    LoadPredictorResult,
    sweep_load_predictor,
    window_loss,
)
from .parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
    enumerate_disagg_configs,
    enumerate_parallel_configs,
    enumerate_worker_shapes,
)
from .planner import SCALING_POLICIES, ScalingPolicy, throughput_intervals
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
    # parallel-config enumeration
    "ParallelShape",
    "ReplicaParallelConfig",
    "DisaggParallelConfig",
    "enumerate_parallel_configs",
    "enumerate_worker_shapes",
    "enumerate_disagg_configs",
    # planner scaling-policy decode
    "SCALING_POLICIES",
    "ScalingPolicy",
    "throughput_intervals",
    # load-predictor sweep
    "LoadPredictorResult",
    "sweep_load_predictor",
    "window_loss",
]
