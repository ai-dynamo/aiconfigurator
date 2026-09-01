# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every enumerated slot must load (review blocker, 2026-08).

`get_supported_databases` is the promise surface (CLI prechecks, backend-auto
sweeps, the support matrix). A slot that enumerates but cannot load — e.g. a
fleet-derived `next` advertised on systems that hold no data for it — is a
broken promise, so the invariant is checked pairwise over the whole
enumeration. Version-agnostic by construction: no literal is pinned.
"""

import pytest

from aiconfigurator_core.sdk import perf_database

pytestmark = pytest.mark.unit


def _enumerated_combos() -> list[tuple[str, str, str]]:
    supported = perf_database.get_supported_databases()
    return [
        (system, backend, version)
        for system, backends in sorted(supported.items())
        for backend, versions in sorted(backends.items())
        for version in versions
    ]


@pytest.mark.parametrize(
    "system,backend,version",
    _enumerated_combos(),
    ids=[f"{s}-{b}-{v}" for s, b, v in _enumerated_combos()],
)
def test_every_enumerated_slot_loads(system: str, backend: str, version: str):
    database = perf_database.get_database(system, backend, version)
    assert database is not None, f"enumerated slot {system}/{backend}/{version} failed to load"
    assert database.version == version


def test_fleet_next_excludes_override_system_data():
    """An override-governed system's data drop (e.g. b60/vLLM 0.26) must not
    advertise a fleet `next` that defaults-governed systems cannot load."""
    doc_overrides = {"b60"}
    supported = perf_database.get_supported_databases()
    b60_versions = set(supported.get("b60", {}).get("vllm", []))
    for system, backends in supported.items():
        if system in doc_overrides:
            continue
        for version in backends.get("vllm", []):
            slots = perf_database.get_version_slots(system, "vllm") or {}
            if version == slots.get("next"):
                # a defaults-governed next must never be a b60-only version
                assert version not in (b60_versions - set(slots.values())), (
                    f"{system}: fleet next {version} is backed only by the b60 override"
                )
