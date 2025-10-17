# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data schema for generator.

This module defines backend-agnostic dataclasses that capture
all fields needed by backend generators, decoupled intermidiate variables from other modules.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DynamoConfig:
    """
    User-provided override config (flat dict). Also exposes helpers to split
    overrides by worker role for Jinja templates.
    """

    extras: dict[str, Any] = field(default_factory=dict)

    def split_by_worker_type(self):
        """Return (common, {'decode':{},'prefill':{},'agg':{}})."""
        common, role = {}, {"decode": {}, "prefill": {}, "agg": {}}
        for k, v in self.extras.items():
            for r in role:
                p = f"{r}_"
                if k.startswith(p):
                    role[r][k[len(p) :]] = v
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
    kv_cache_mode: str | None = None  # e.g. 'fp8', 'auto'
    nextn: int = 0
    nextn_accept_rates: list[float] | None = None


@dataclass
class ModeConfig:
    """Config payload for a specific serving mode."""

    workers: int
    bs: int
    tp: int
    pp: int
    dp: int | None = 1
    moe_tp: int | None = 1
    moe_ep: int | None = 1
    memory: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


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
    modes: dict[str, ModeConfig] = field(default_factory=dict)
