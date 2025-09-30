# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data schema for generator.

This module defines backend-agnostic dataclasses that capture
all fields needed by backend generators, decoupled intermidiate variables from other modules.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class DynamoConfig:
    """
    User-provided override config (flat dict). Also exposes helpers to split
    overrides by worker role for Jinja templates.
    """
    extras: Dict[str, Any] = field(default_factory=dict)

    def split_by_worker_type(self):
        """Return (common, {'decode':{},'prefill':{},'agg':{}})."""
        common, role = {}, {"decode": {}, "prefill": {}, "agg": {}}
        for k, v in self.extras.items():
            for r in role:
                p = f"{r}_"
                if k.startswith(p):
                    role[r][k[len(p):]] = v
                    break
            else:
                common[k] = v
        return common, role

@dataclass
class RuntimeView:
    """Runtime properties used for generation."""
    isl: int
    osl: int
    is_moe: bool
    kv_cache_mode: Optional[str] = None  # e.g. 'fp8', 'auto'
    nextn: int = 0
    nextn_accept_rates: Optional[List[float]] = None

@dataclass
class ParallelSpecBase:
    """Common spec shared by Agg and Disagg roles."""
    tp: int
    pp: int
    dp: int
    bs: int
    moe_tp: Optional[int] = None
    moe_ep: Optional[int] = None

@dataclass
class BestAggConfig(ParallelSpecBase):
    """Best Agg config using the common spec base."""

@dataclass
class RoleParallelSpec(ParallelSpecBase):
    """Spec for a Disagg role with worker count."""
    workers: int = 1

@dataclass
class BestDisaggConfig:
    """Best Disagg config with prefill and decode specs."""
    prefill: RoleParallelSpec
    decode: RoleParallelSpec

@dataclass
class GeneratorContext:
    """
    Validated view consumed by backend generators.
    """
    model_name: str
    backend: str
    version: str
    runtime: RuntimeView
    overrides: DynamoConfig
    best_agg: Optional[BestAggConfig] = None
    best_disagg: Optional[BestDisaggConfig] = None
