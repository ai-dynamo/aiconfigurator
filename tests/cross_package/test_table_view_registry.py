# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completeness of the table-view attribute registry across its sites.

The view registry used to live in four hand-synchronized, stringly-typed
places: the Rust dispatch match arms (with per-attribute parquet basenames),
Python's ``VIEW_KEY_LAYERS``, each op class's ``load_data`` literals, and the
retired migration baseline's codec. Two of the drift modes were SILENT (an
attribute missing from ``VIEW_KEY_LAYERS`` only fails on machines that have
that family's data; a basename typo in a Rust arm makes the view answer None
with no error). The engine exports the registry
(``aiconfigurator_core.table_view_attributes()``, same single-source pattern
as ``gemm_quant_util_levels``); these tests pin every other site to it:

* ``VIEW_KEY_LAYERS`` covers exactly the exported attributes;
* every exported basename is a known ``PerfDataFilename`` value (a typo in
  the registry cannot survive; a typo in a MATCH ARM is caught by the
  synthetic-parquet view-shape suites, which pin each fold's dispatch);
* every exported attribute actually dispatches (fetching it against a real
  pinned database returns a table or None — never "unknown attribute").
"""

from __future__ import annotations

import pytest

import aiconfigurator_core
from aiconfigurator_core.sdk.common import PerfDataFilename
from aiconfigurator_core.sdk.engine_table_view import VIEW_KEY_LAYERS

pytestmark = pytest.mark.unit


def _exported() -> dict[str, list[str]]:
    return dict(aiconfigurator_core.table_view_attributes())


def test_view_key_layers_match_the_engine_registry() -> None:
    exported = set(_exported())
    layered = set(VIEW_KEY_LAYERS)
    assert layered == exported, (
        f"VIEW_KEY_LAYERS drifted from the engine registry: "
        f"missing={sorted(exported - layered)} extra={sorted(layered - exported)}"
    )


def test_registry_basenames_are_known_perf_data_filenames() -> None:
    known = {member.value for member in PerfDataFilename}
    unknown = {
        (attribute, basename)
        for attribute, basenames in _exported().items()
        for basename in basenames
        if basename not in known
    }
    assert not unknown, f"registry basenames not in PerfDataFilename: {sorted(unknown)}"


def test_every_registry_attribute_dispatches_on_a_pinned_database() -> None:
    """Fetch each exported attribute against the h200 pin: a registry entry
    the Rust dispatch does not accept raises "unknown table-view attribute"
    here in every CI run — not None on some machine without the data."""
    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
    from aiconfigurator_core.sdk.perf_database import get_database

    database = get_database("h200_sxm", "trtllm", "1.2.0rc5", allow_unlisted_version=True)
    assert database is not None
    for attribute in _exported():
        table = fetch_table_view(database, attribute)
        assert table is None or isinstance(table, dict)
