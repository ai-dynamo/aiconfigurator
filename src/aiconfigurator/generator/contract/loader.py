# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest loading and matching for the generator contract layer.

Minimal subset salvaged for the typed TRT-LLM engine-config builder. Exposes
``MANIFESTS_DIR``, ``load_yaml`` and ``_select_version_manifest`` (plus their
direct version-matching helpers). The full contract pipeline (model/variant/
cluster profile lookup, manifest bundles) is intentionally omitted here.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml
from packaging.version import InvalidVersion, Version

_GENERATOR_DIR = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = _GENERATOR_DIR / "manifests"


@cache
def load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_version(value: str | None) -> Version | None:
    if not value:
        return None
    raw = str(value).strip()
    if raw.lower().startswith("v") and len(raw) > 1:
        raw = raw[1:]
    if raw.endswith(".x"):
        raw = raw[:-2] + ".0"
    try:
        return Version(raw)
    except InvalidVersion:
        return None


def _version_matches(pattern: str, requested: str | None) -> bool:
    if not requested:
        return True
    pattern = str(pattern).strip()
    requested = str(requested).strip()
    if pattern == requested:
        return True
    if pattern.endswith(".x"):
        return requested.startswith(pattern[:-1])
    parsed_pattern = _parse_version(pattern)
    parsed_requested = _parse_version(requested)
    return bool(parsed_pattern and parsed_requested and parsed_pattern == parsed_requested)


def _select_version_manifest(directory: Path, requested: str | None) -> dict[str, Any] | None:
    candidates: list[tuple[Version | None, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.yaml")):
        manifest = load_yaml(str(path))
        version = manifest.get("version") or manifest.get("backend_version") or path.stem
        if _version_matches(str(version), requested):
            return manifest
        candidates.append((_parse_version(str(version)), manifest))
    if not requested:
        parsed = [(v, m) for v, m in candidates if v is not None]
        if parsed:
            return max(parsed, key=lambda item: item[0])[1]
        return candidates[-1][1] if candidates else None
    requested_v = _parse_version(requested)
    if requested_v is None:
        return None
    floor = [(v, m) for v, m in candidates if v is not None and v <= requested_v]
    if floor:
        return max(floor, key=lambda item: item[0])[1]
    return None
