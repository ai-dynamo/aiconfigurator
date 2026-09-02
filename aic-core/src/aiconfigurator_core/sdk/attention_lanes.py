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
from typing import Optional

import yaml

from aiconfigurator_core.sdk.errors import UnsupportedAttentionBackendError

logger = logging.getLogger(__name__)

# Canonical user-facing attention-backend vocabulary. ``"default"`` selects
# the framework default and is not a named measurement lane.
ATTENTION_BACKEND_CHOICES: tuple[str, ...] = ("fa3", "triton", "trtllm_mha", "flashinfer", "fla", "default")
_KNOWN_LANES: frozenset[str] = frozenset(choice for choice in ATTENTION_BACKEND_CHOICES if choice != "default")

# User-facing override vocabulary (the CLI's ``--attention-backend`` choices,
# minus the universal ``"default"``) -> the stored ``kernel_source`` labels
# each backend collects under, most specific first. sglang collects under the
# user-facing names themselves; vllm prefixes its labels. vllm's flashinfer
# entry lists every label the shipped tables use across versions (0.24.0
# splits the lane into trtllm prefill/decode kernels, 0.22.0-era tables carry
# the single ``vllm_flashinfer`` label) — the table walk serves the first
# listed lane that carries the queried slice, so one pinned order covers all
# shipped versions and both the context and generation tables. A backend
# absent from this map supports NO override (trtllm collects ``torch_flow*``
# lanes that correspond to no user-facing choice).
_OVERRIDE_LANES: dict[str, dict[str, tuple[str, ...]]] = {
    "sglang": {
        "fa3": ("fa3",),
        "triton": ("triton",),
        "trtllm_mha": ("trtllm_mha",),
        "flashinfer": ("flashinfer",),
        "fla": ("fla",),
    },
    "vllm": {
        "triton": ("vllm_triton_attn",),
        "flashinfer": (
            "vllm_flashinfer_trtllmprefill",
            "vllm_flashinfer_trtllmdecode",
            "vllm_flashinfer",
        ),
    },
}


def resolve_attention_override_lanes(backend: str, override: str) -> tuple[str, ...]:
    """Stored ``kernel_source`` labels for a user-facing *override* on *backend*.

    Raises :class:`UnsupportedAttentionBackendError` for a (backend, override)
    pair outside :data:`_OVERRIDE_LANES` — an explicit override the backend's
    tables cannot serve must fail loudly, never fall through to a donor lane.
    ``"default"`` (the framework default) is accepted for every backend.
    """
    if override == "default":
        return ("default",)
    supported = _OVERRIDE_LANES.get(backend, {})
    lanes = supported.get(override)
    if lanes is None:
        raise UnsupportedAttentionBackendError(
            f"attention_backend={override!r} is not supported on backend {backend!r}; "
            f"supported values: {sorted(supported) + ['default']}."
        )
    return lanes


# Leading digits of a version segment, e.g. the "0" in "0rc23" or "14" in "14a1".
_LEADING_DIGITS = re.compile(r"^(\d+)")


@functools.cache
def _load_defaults(systems_root: Optional[str]) -> dict:
    """Load and cache attention_lane_defaults.yaml, keyed by *systems_root*.

    When *systems_root* is ``None`` the default package systems directory is
    used — the same location PerfDatabase resolves via
    ``_normalize_systems_paths(None)``. A custom systems root may override the
    map by shipping its own file; when it does not, the packaged map remains
    the framework-default contract while only the perf-data root changes.
    """
    packaged_path = pkg_resources.files("aiconfigurator_core") / "systems" / "attention_lane_defaults.yaml"
    path = (
        os.fspath(packaged_path) if systems_root is None else os.path.join(systems_root, "attention_lane_defaults.yaml")
    )
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        if systems_root is not None:
            logger.warning(
                "attention_lane_defaults.yaml not found at %s; falling back to packaged attention-lane defaults",
                path,
            )
            try:
                with packaged_path.open(encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except FileNotFoundError:
                path = os.fspath(packaged_path)
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

    ``framework_default_matched`` records whether the framework-default map
    had an entry for this exact (backend, floor-matched version, sm_version).
    It is NOT derivable from the tuple either: a map entry of ``"default"``
    pins nothing (``"default"`` always rides the donor tail), yet it is still
    positive evidence — unlike a version/backend with no entry at all, where
    donor density is no evidence of the framework default and the caller must
    fail closed (``operations.attention.resolved_lane_order_for_op``).
    """

    def __new__(cls, lanes, pinned_count: int = 0, framework_default_matched: bool = False):
        self = super().__new__(cls, lanes)
        self.pinned_count = pinned_count
        self.framework_default_matched = framework_default_matched
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
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Return ``(pinned, donors, framework_default_matched)`` for the context.

    *pinned* is the explicit-intent head — the override (translated to the
    backend's stored ``kernel_source`` labels, see
    :func:`resolve_attention_override_lanes`)
    and the framework-default map lane, in that precedence, ``"default"``
    excluded (it is always the donor tier's last element). *donors* is the
    generic tail from :func:`_donor_tier`.  *framework_default_matched* is
    True when the map had an entry for this exact (backend, floor-matched
    version, sm_version) — even one whose lane is ``"default"`` and therefore
    pins nothing. :func:`resolve_attention_lane_order` concatenates the first
    two; this function is what makes the boundary between them (and the
    map-evidence bit) knowable downstream (see :class:`LaneOrder`).

    Raises :class:`UnsupportedAttentionBackendError` when *override* names a
    lane the backend's tables do not collect.
    """
    defaults = _load_defaults(systems_root)

    listed: list[str] = []
    framework_default_matched = False

    # Step 1: override first, translated to the stored kernel_source labels.
    if override is not None:
        listed.extend(resolve_attention_override_lanes(backend, override))

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
            else:
                framework_default_matched = True
                if map_lane not in listed:
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
    return pinned, _donor_tier(pinned), framework_default_matched


def resolve_attention_lane_order(
    backend: str,
    version: str,
    sm_version: int,
    override: Optional[str],
    systems_root: Optional[str] = None,
) -> LaneOrder:
    """Return an ordered tuple of attention lane names for the given context.

    Precedence rules (applied in order, each lane included at most once):

    1. *override* — always first when given, translated to the backend's
       stored ``kernel_source`` labels
       (:func:`resolve_attention_override_lanes`; an override the backend does
       not support raises
       :class:`UnsupportedAttentionBackendError`).
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
    re-ranking can leave that head alone, plus whether the framework-default
    map matched so the caller can fail closed when it did not.
    """
    pinned, donors, framework_default_matched = resolve_attention_lane_tiers(
        backend, version, sm_version, override, systems_root
    )
    return LaneOrder(pinned + donors, len(pinned), framework_default_matched)
