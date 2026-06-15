# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request -> facts resolution (Phase 4a).

Maps a generation request's ``model_path`` and ``system_name`` onto the facts
data layer: a logical model-profile id (``models.yaml``) and a hardware profile
key (``hardware.yaml``), then delegates to :func:`resolve_facts`.

This wires resolution into the pipeline but APPLIES NOTHING — the resolved facts
are threaded onto the pipeline result and not yet used to alter rendered output.
"""
from __future__ import annotations

import dataclasses
import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from aiconfigurator.generator.utils import load_backend_version_matrix

from .resolve import ResolvedFacts, resolve_facts

_FACTS_DIR = Path(__file__).parent

# Map SDK system ids (and family variants) onto hardware.yaml profile keys.
# Unknown systems fall back to the raw name so resolve_facts raises clearly.
_SYSTEM_TO_HW: dict[str, str] = {
    "h100_sxm": "h100",
    "h100_pcie": "h100",
    "h100": "h100",
    "h200_sxm": "h200",
    "h200": "h200",
    "b200_sxm": "b200",
    "b200": "b200",
    "b300_sxm": "b200",
    "b300": "b200",
    "gb200": "gb200",
    "gb200_sxm": "gb200",
    "gb300": "gb200",
}

# Sections of the params dict that may carry the SDK system id.
_SYSTEM_SECTIONS = ("K8sConfig", "NodeConfig", "ServiceConfig")
# Default transport profile when the request does not specify one.
_DEFAULT_TRANSPORT = "nvlink"


def hardware_key_for_system(system_name: str) -> str:
    """Normalize an SDK system id (e.g. ``"h200_sxm"``) to a hardware key.

    Unknown systems fall back to the raw name so that downstream
    :func:`resolve_facts` raises a clear error for truly unknown hardware.
    """
    if not system_name:
        return system_name
    return _SYSTEM_TO_HW.get(system_name, system_name)


@lru_cache(maxsize=1)
def _model_match_table() -> list[tuple[str, str]]:
    """Return ``[(glob_pattern, profile_id), ...]`` from models.yaml."""
    models_path = _FACTS_DIR / "models.yaml"
    with models_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    table: list[tuple[str, str]] = []
    for profile_id, profile in (data.get("models") or {}).items():
        match = profile.get("match") or {}
        for pattern in match.get("model_paths") or []:
            table.append((pattern, profile_id))
    return table


def model_profile_for_path(model_path: str | None) -> str | None:
    """Return the logical model-profile id matching ``model_path``, or None.

    Matches the request's ``model_path`` against each profile's
    ``match.model_paths`` globs (fnmatch). Returns ``None`` for the generic
    path (no matching profile).
    """
    if not model_path:
        return None
    for pattern, profile_id in _model_match_table():
        if fnmatch.fnmatch(model_path, pattern):
            return profile_id
    return None


def _system_name(params: dict[str, Any]) -> str | None:
    """Extract the SDK system id from the params dict, if present."""
    for section in _SYSTEM_SECTIONS:
        sec = params.get(section)
        if isinstance(sec, dict):
            value = sec.get("system_name")
            if value:
                return value
    return None


def resolve_facts_for_request(
    params: dict[str, Any],
    backend: str,
    dynamo_version: str | None,
) -> ResolvedFacts | None:
    """Resolve facts for one generation request.

    Returns ``None`` if no ``system_name`` can be found in the params (nothing
    to resolve). Tolerant of a missing/blank ``dynamo_version`` (backend_version
    is unused in Phase 4a, so resolution must not crash when it is absent).
    """
    system_name = _system_name(params)
    if not system_name:
        return None

    hardware = hardware_key_for_system(system_name)
    model_path = (params.get("ServiceConfig") or {}).get("model_path")
    model_profile_id = model_profile_for_path(model_path)
    version = dynamo_version or ""

    # backend_version is unused in Phase 4a, so a missing/blank/unknown
    # dynamo_version must NOT crash resolution. resolve_facts() looks the
    # version up first and raises if the (version, backend) pair is absent;
    # when that lookup would fail, resolve with a pivot version that exists
    # purely to obtain the hardware/transport/model facts, then blank out the
    # backend_version on the result so we never surface a bogus value.
    # ``hardware`` here is the bare profile KEY (e.g. "gb200"); stash it on the
    # result so the apply step can match model ``defaults`` ``match.system``
    # (ResolvedFacts.hardware is the profile dict and carries no key inside it).
    matrix = load_backend_version_matrix(str(_FACTS_DIR / "runtimes" / "dynamo.yaml"))
    if version and backend in matrix.get(version, {}):
        return dataclasses.replace(
            resolve_facts(
                model_profile_id=model_profile_id,
                hardware=hardware,
                transport=_DEFAULT_TRANSPORT,
                dynamo_version=version,
                backend=backend,
            ),
            hardware_key=hardware,
        )

    pivot = next((v for v, b in matrix.items() if backend in b), None)
    facts = resolve_facts(
        model_profile_id=model_profile_id,
        hardware=hardware,
        transport=_DEFAULT_TRANSPORT,
        dynamo_version=pivot or version,
        backend=backend,
    )
    # Reflect the request's (possibly blank/unknown) version, drop the pivot's
    # backend_version since it does not correspond to the requested version.
    return dataclasses.replace(facts, dynamo_version=version, backend_version="", hardware_key=hardware)
