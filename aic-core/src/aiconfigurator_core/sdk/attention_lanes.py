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
import re
from collections.abc import Mapping
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Canonical set of named measurement lanes (excludes "default" which is always appended last).
_KNOWN_LANES: frozenset[str] = frozenset({"fa3", "triton", "trtllm_mha", "flashinfer", "fla"})

# Valid lane VALUES anywhere in attention_lane_defaults.yaml: the named lanes
# plus "default" (used verbatim by the shipped vllm block).
_KNOWN_LANE_VALUES: frozenset[str] = _KNOWN_LANES | frozenset({"default"})

# Leading digits of a version segment, e.g. the "0" in "0rc23" or "14" in "14a1".
_LEADING_DIGITS = re.compile(r"^(\d+)")


def _validate_architecture_defaults(architectures: object) -> None:
    """Fail loudly on a malformed ``architectures:`` section (AIC-1762).

    Schema: ``{architecture: {backend: {version_str: {sm_int: lane_str}}}}``,
    where ``lane_str`` is one of :data:`_KNOWN_LANE_VALUES`. Unlike a miss
    (unlisted architecture/backend/version/sm, which is the normal "fall back
    to the global map" path), a STRUCTURALLY wrong entry — a typo'd lane name,
    a non-int sm key, a non-mapping level — must never resolve to ``None`` and
    silently masquerade as an ordinary miss; it raises instead, per the repo's
    fail-closed convention.
    """
    # ValueError (not TypeError) for a config-shape error matches the
    # established convention elsewhere in this codebase for YAML schema
    # validation (see perf_database.py's reuse-block loader).
    if not isinstance(architectures, Mapping):
        raise ValueError(  # noqa: TRY004
            f"attention_lane_defaults.yaml: 'architectures' must be a mapping, got {type(architectures).__name__}"
        )
    for arch, backend_map in architectures.items():
        if not isinstance(backend_map, Mapping):
            raise ValueError(  # noqa: TRY004
                f"attention_lane_defaults.yaml: architectures[{arch!r}] must be a mapping, "
                f"got {type(backend_map).__name__}"
            )
        for backend, version_map in backend_map.items():
            if not isinstance(version_map, Mapping):
                raise ValueError(  # noqa: TRY004
                    f"attention_lane_defaults.yaml: architectures[{arch!r}][{backend!r}] must be a mapping, "
                    f"got {type(version_map).__name__}"
                )
            for version, sm_map in version_map.items():
                if not isinstance(version, str):
                    raise ValueError(  # noqa: TRY004
                        f"attention_lane_defaults.yaml: architectures[{arch!r}][{backend!r}] version key "
                        f"{version!r} must be a string"
                    )
                if not isinstance(sm_map, Mapping):
                    raise ValueError(  # noqa: TRY004
                        f"attention_lane_defaults.yaml: architectures[{arch!r}][{backend!r}][{version!r}] must be "
                        f"a mapping, got {type(sm_map).__name__}"
                    )
                for sm, lane in sm_map.items():
                    if not isinstance(sm, int):
                        raise ValueError(  # noqa: TRY004
                            f"attention_lane_defaults.yaml: architectures[{arch!r}][{backend!r}][{version!r}] "
                            f"sm key {sm!r} must be an int"
                        )
                    if lane not in _KNOWN_LANE_VALUES:
                        raise ValueError(
                            f"attention_lane_defaults.yaml: architectures[{arch!r}][{backend!r}][{version!r}]"
                            f"[{sm!r}] lane {lane!r} is not a known lane; expected one of "
                            f"{sorted(_KNOWN_LANE_VALUES)}"
                        )


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
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(
            "attention_lane_defaults.yaml not found at %s; no framework defaults available",
            path,
        )
        return {}
    if "architectures" in data:
        _validate_architecture_defaults(data["architectures"])
    return data


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

    A segment may also GLUE its suffix directly onto the last numeric
    component instead of dot-separating it — ``"1.3.0rc23"``,
    ``"1.3.0a1"``, ``"1.3.0.post1"``'s sibling forms. Without stripping the
    glued suffix first, ``int("0rc23")`` raises immediately and the whole
    segment (including its leading ``"0"``) is dropped, so ``"1.3.0rc23"``
    would parse as ``(1, 3)`` — shorter than, and therefore sorting BELOW,
    the plain ``"1.3.0"`` release it is a candidate for. That silently
    disqualifies a floor-matched framework-default lane keyed on the glued
    form whenever the requested version is itself glued-suffixed. Take the
    segment's leading digits (if any) before giving up on it, so
    ``"1.3.0rc23"`` parses as ``(1, 3, 0)``.
    """
    parts: list[int] = []
    for segment in v.split("."):
        try:
            parts.append(int(segment))
            continue
        except ValueError:
            pass
        leading_digits = _LEADING_DIGITS.match(segment)
        if leading_digits:
            parts.append(int(leading_digits.group(1)))
        break  # first non-cleanly-numeric segment — stop after its leading digits, if any
    return tuple(parts) if parts else (0,)


def resolve_attention_lane_tiers(
    backend: str,
    version: str,
    sm_version: int,
    override: Optional[str],
    systems_root: Optional[str] = None,
    architecture: Optional[str] = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(pinned, donors)`` for the given context.

    *pinned* is the explicit-intent head — the override, then either the
    per-architecture default or (when that does not apply) the global
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

    # Step 2 (AIC-1762): per-architecture default, consulted only when no
    # override was given -- explicit intent stays first-class either way.
    # Same version floor-match contract as the global map (step 3), scoped
    # under architectures.<architecture>.<backend>. When it resolves, it
    # FULLY REPLACES step 3 for this resolution: the global map is the
    # mechanism-level fallback for architectures (or backends/versions/sms
    # within one) that have no dedicated entry, not a second pin stacked
    # alongside a resolved architecture default.
    arch_lane: Optional[str] = None
    if override is None and architecture is not None:
        arch_backend_map = defaults.get("architectures", {}).get(architecture, {}).get(backend)
        if arch_backend_map is not None:
            req_ver = _parse_version(version)
            valid = [(v_str, _parse_version(v_str)) for v_str in arch_backend_map if _parse_version(v_str) <= req_ver]
            if valid:
                best_v_str = max(valid, key=lambda x: x[1])[0]
                arch_lane = arch_backend_map[best_v_str].get(sm_version)

    if arch_lane is not None:
        if arch_lane not in listed:
            listed.append(arch_lane)
    else:
        # Step 3 (UNCHANGED): framework-default lane from the global map.
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

    # Steps 4-5: the donor tier — remaining known lanes alphabetically, then
    # "default" last exactly once (so a pinned "default" moves to the tail).
    pinned = tuple(lane for lane in listed if lane != "default")
    return pinned, _donor_tier(pinned)


def resolve_attention_lane_order(
    backend: str,
    version: str,
    sm_version: int,
    override: Optional[str],
    systems_root: Optional[str] = None,
    architecture: Optional[str] = None,
) -> LaneOrder:
    """Return an ordered tuple of attention lane names for the given context.

    Precedence rules (applied in order, each lane included at most once):

    1. *override* — always first when given.
    2. The per-architecture default lane for (*architecture*, *backend*,
       floor-matched *version*, *sm_version*) from the ``architectures:``
       section of ``attention_lane_defaults.yaml`` (AIC-1762), if *override*
       is ``None`` and *architecture* resolves to one. When this resolves it
       REPLACES step 3 below for this call (mutually exclusive, not additive).
    3. Otherwise, the global framework-default lane for (*backend*,
       floor-matched *version*, *sm_version*) from the same file, if present
       and not already listed. This is the mechanism-level fallback: it is
       what runs whenever step 2 does not apply (no architecture given, an
       unlisted architecture, no entry for this backend, or *version* below
       that architecture's floor) — every architecture without its own
       ``architectures:`` entry resolves through this step exactly as before
       AIC-1762, byte-identical.
    4. The remaining known lanes ``{"fa3","triton","trtllm_mha","flashinfer",
       "fla"}`` minus already-listed entries, in sorted (alphabetical) order.
    5. ``"default"`` — always last, exactly once.

    Floor-match on version: the highest version key in the map that is
    ``<= version`` (dotted-numeric comparison).  Unknown backend or no
    matching version/sm entry: skip step 3 and log one WARNING (step 2 is
    silent on a miss — an architecture with no per-arch entry is the normal
    case, not a warning-worthy condition).

    The return value is the flat tuple of steps 1-5 (:class:`LaneOrder`, which
    is a ``tuple``), carrying the length of its pinned head so downstream
    re-ranking can leave that head alone.
    """
    pinned, donors = resolve_attention_lane_tiers(backend, version, sm_version, override, systems_root, architecture)
    return LaneOrder(pinned + donors, len(pinned))
