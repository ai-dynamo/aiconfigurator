# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the empirical-provenance pipeline.

The estimation MATH that used to be tested here (``UtilGrid`` interpolation,
``grid_for``/``grid_from_reference`` caching, ``estimate`` transfer ladder,
``require_data_slice``) retired to the compiled engine with #1357 PR-5; its
behaviour is anchored by ``tests/cross_package/test_query_shim_baseline.py``
and the frozen parity goldens. What remains observable from Python — and
tested here — is the provenance pipeline the engine reports back through.
"""

import pytest

from aiconfigurator.sdk.operations.util_empirical import (
    PROVENANCE_ORDER,
    capture_provenance,
    clear_grid_cache,
    note_provenance,
    worst_provenance,
)

pytestmark = pytest.mark.unit


def test_note_provenance_is_noop_outside_capture():
    # Must not raise or leak state when no capture is active.
    note_provenance("xquant")

    with capture_provenance() as tags:
        pass

    assert tags == set()


def test_capture_provenance_collects_tags_fired_within_block():
    with capture_provenance() as tags:
        note_provenance("xshape")
        note_provenance("xquant")
        note_provenance("xshape")  # duplicates collapse: it's a set of tags

    assert tags == {"xshape", "xquant"}
    assert worst_provenance(tags) == "xquant"

    # Tags fired after the capture ends are not recorded.
    note_provenance("xop")
    assert tags == {"xshape", "xquant"}


def test_nested_captures_isolate_their_sinks():
    with capture_provenance() as outer:
        note_provenance("empirical")
        with capture_provenance() as inner:
            note_provenance("xop")
        note_provenance("xshape")

    assert inner == {"xop"}
    assert outer == {"empirical", "xshape"}


def test_worst_provenance_ranks_by_decreasing_confidence():
    # Later tags in PROVENANCE_ORDER rely on more aggressive transfer.
    assert list(PROVENANCE_ORDER) == ["silicon", "empirical", "xshape", "xquant", "xprofile", "xop"]

    assert worst_provenance(set()) == "silicon"
    assert worst_provenance({"empirical"}) == "empirical"
    assert worst_provenance({"empirical", "xshape"}) == "xshape"
    assert worst_provenance({"xshape", "xquant"}) == "xquant"
    assert worst_provenance({"xquant", "xprofile"}) == "xprofile"
    assert worst_provenance({"xprofile", "xop"}) == "xop"
    # Unknown tags rank as least-aggressive rather than raising.
    assert worst_provenance({"mystery", "xquant"}) == "xquant"


def test_clear_grid_cache_is_a_compat_noop():
    # Kept so the clear_all_op_caches eviction contract stays valid through
    # the deprecation window.
    assert clear_grid_cache() is None
