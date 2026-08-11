# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention-lane resolver for AIConfigurator.

Produces an ordered tuple of attention kernel lane names for a given
(backend, version, sm_version, override) context. The tuple encodes
the precedence order that callers use when selecting a per-lane perf table.

AIC-1715/1716.
"""

from __future__ import annotations

import functools
import importlib.resources as pkg_resources
import logging
import os
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Canonical set of named measurement lanes (excludes "default" which is always appended last).
_KNOWN_LANES: frozenset[str] = frozenset({"fa3", "triton", "trtllm_mha", "flashinfer", "fla"})


@functools.cache
def _load_defaults(systems_root: Optional[str]) -> dict:
    """Load and cache attention_lane_defaults.yaml, keyed by *systems_root*.

    When *systems_root* is ``None`` the default package systems directory is
    used — the same location PerfDatabase resolves via
    ``_normalize_systems_paths(None)``.
    """
    if systems_root is None:
        systems_root = os.fspath(pkg_resources.files("aiconfigurator_core") / "systems")
    path = os.path.join(systems_root, "attention_lane_defaults.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(
            "attention_lane_defaults.yaml not found at %s; no framework defaults available",
            path,
        )
        return {}


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable integer tuple."""
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


def resolve_attention_lane_order(
    backend: str,
    version: str,
    sm_version: int,
    override: Optional[str],
    systems_root: Optional[str] = None,
) -> tuple[str, ...]:
    """Return an ordered tuple of attention lane names for the given context.

    Precedence rules (applied in order, each lane included at most once):

    1. *override* — always first when given.
    2. The framework-default lane for (*backend*, floor-matched *version*,
       *sm_version*) from ``attention_lane_defaults.yaml``, if present and
       not already listed.
    3. The remaining known lanes ``{"fa3","triton","trtllm_mha","flashinfer",
       "fla"}`` minus already-listed entries, in sorted (alphabetical) order.
    4. ``"default"`` — always last, exactly once.

    Floor-match on version: the highest version key in the map that is
    ``<= version`` (dotted-numeric comparison).  Unknown backend or no
    matching version/sm entry: skip step 2 and log one WARNING.
    """
    defaults = _load_defaults(systems_root)

    listed: list[str] = []

    # Step 1: override first.
    if override is not None:
        listed.append(override)

    # Step 2: framework-default lane from the map.
    backend_map = defaults.get(backend)
    if backend_map is None:
        logger.warning(
            "resolve_attention_lane_order: unknown backend %r — no framework defaults available",
            backend,
        )
    else:
        req_ver = _parse_version(version)
        valid = [(v_str, _parse_version(v_str)) for v_str in backend_map if _parse_version(v_str) <= req_ver]
        if valid:
            best_v_str = max(valid, key=lambda x: x[1])[0]
            map_lane = backend_map[best_v_str].get(sm_version)
            if map_lane is None:
                logger.warning(
                    "resolve_attention_lane_order: no entry for sm_version=%d in %r/%r",
                    sm_version,
                    backend,
                    best_v_str,
                )
            elif map_lane not in listed:
                listed.append(map_lane)

    # Step 3: remaining known lanes in sorted order.
    for lane in sorted(_KNOWN_LANES):
        if lane not in listed:
            listed.append(lane)

    # Step 4: "default" always last, exactly once.
    while "default" in listed:
        listed.remove("default")
    listed.append("default")

    return tuple(listed)
