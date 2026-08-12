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


def _donor_tier(pinned) -> tuple[str, ...]:
    """The lanes NOT pinned by explicit intent, in default precedence order.

    Remaining known lanes alphabetically, then ``"default"`` last. Callers that
    hold a perf table may re-order this tier (e.g. by measured coverage — see
    ``operations.attention.lane_walk_order``); this module ranks it by name
    only, because the resolver is deliberately table-blind.
    """
    return tuple(lane for lane in sorted(_KNOWN_LANES) if lane not in pinned) + ("default",)


class LaneOrder(tuple):
    """A resolved lane order that REMEMBERS how long its pinned head is.

    The value is an ordinary tuple of lane names — callers index, iterate,
    compare and serialize it exactly as before — plus one extra bit:
    ``pinned_count``, the length of the intent-ordered prefix (override, then
    the framework-default map lane). Downstream re-ranking
    (:func:`operations.attention.lane_walk_order`) needs that boundary, and it
    CANNOT be recovered from the flat tuple: a pinned ``("fa3", …)`` head and
    the unpinned alphabetical donor tier — which also starts with ``fa3`` — are
    byte-identical, so a reconstruction silently demotes the pin.
    """

    def __new__(cls, lanes, pinned_count: int = 0):
        self = super().__new__(cls, lanes)
        self.pinned_count = pinned_count
        return self


def split_attention_lane_tiers(lane_order: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a resolved order into ``(pinned, donors)``.

    *pinned* is the intent-ordered head this module produced from the override
    and the framework-default map entry — carried explicitly on the
    :class:`LaneOrder` this module returns, never re-derived; *donors* is the
    generic tail (:func:`_donor_tier`). A plain tuple this module did not
    produce — e.g. a hand-specified order such as ``("triton",)``, or a walk
    order that was already expanded — yields ``(lane_order, ())`` so explicit
    orders are always honoured verbatim.
    """
    pinned_count = getattr(lane_order, "pinned_count", None)
    lane_order = tuple(lane_order)
    if pinned_count is None:
        return lane_order, ()
    return lane_order[:pinned_count], lane_order[pinned_count:]


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable integer tuple.

    Leading numeric segments are parsed; the first non-numeric segment and
    everything after it is ignored so that PEP 440 suffixes such as
    ``"0.5.14.post1"`` or ``"0.5.14.dev3"`` produce ``(0, 5, 14)`` rather
    than falling back to ``(0,)``.
    """
    parts: list[int] = []
    for segment in v.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            break  # trailing non-numeric segment — stop here
    return tuple(parts) if parts else (0,)


def resolve_attention_lane_tiers(
    backend: str,
    version: str,
    sm_version: int,
    override: Optional[str],
    systems_root: Optional[str] = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(pinned, donors)`` for the given context.

    *pinned* is the explicit-intent head — the override and the
    framework-default map lane, in that precedence, ``"default"`` excluded (it
    is always the donor tier's last element). *donors* is the generic tail from
    :func:`_donor_tier`. :func:`resolve_attention_lane_order` concatenates the
    two; this function is what makes the boundary between them knowable
    downstream (see :class:`LaneOrder`).
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
        else:
            logger.warning(
                "resolve_attention_lane_order: version %r is below all entries for backend %r"
                " — no framework default available",
                version,
                backend,
            )

    # Steps 3-4: the donor tier — remaining known lanes alphabetically, then
    # "default" last exactly once (so a pinned "default" moves to the tail).
    pinned = tuple(lane for lane in listed if lane != "default")
    return pinned, _donor_tier(pinned)


def resolve_attention_lane_order(
    backend: str,
    version: str,
    sm_version: int,
    override: Optional[str],
    systems_root: Optional[str] = None,
) -> LaneOrder:
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

    The return value is the flat tuple of steps 1-4 (:class:`LaneOrder`, which
    is a ``tuple``), carrying the length of its pinned head so downstream
    re-ranking can leave that head alone.
    """
    pinned, donors = resolve_attention_lane_tiers(backend, version, sm_version, override, systems_root)
    return LaneOrder(pinned + donors, len(pinned))
