# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missing-``sms`` handling in ``_query_wideep_deepep_normal_table``.

The ``lookup_node == 1 and sms == 20`` fast path selects a single dispatch_sms
slice out of the node table. ``load_wideep_deepep_normal_data`` builds that
table from nested ``defaultdict``s, so plain ``node_data[sms]`` on an
uncollected sms auto-vivifies an empty branch *into the shared class-level
cache* instead of failing. The read must use ``.get()`` so a miss stays
read-only and surfaces as the usual structured interpolation error.

Lives here rather than in ``test_moe_dispatch.py`` (the other WideEP DeepEP
tests) because that module imports ``torch`` at module scope, which is not a
dependency of the SDK under test.
"""

from collections import defaultdict
from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import InterpolationDataNotAvailableError
from aiconfigurator.sdk.operations import MoEDispatch, PerformanceResult

pytestmark = pytest.mark.unit

HIDDEN, TOPK, EXPERTS = 7168, 8, 256


def _normal_table(sms_values):
    """Build a deepep-normal table in the exact nested-defaultdict shape that
    ``load_wideep_deepep_normal_data`` produces, populated only for ``sms_values``."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    for sms in sms_values:
        for num_tokens, latency in ((64, 100.0), (128, 200.0)):
            data[1][HIDDEN][TOPK][EXPERTS][sms][num_tokens] = {
                "latency": latency,
                "power": 0.0,
                "energy": 0.0,
            }
    return data


def _db(table):
    db = MagicMock()
    db._wideep_deepep_normal_data = table
    db._default_database_mode = common.DatabaseMode.SILICON
    db._interp_pr = lambda lat, energy=0.0: PerformanceResult(lat)
    return db


def _query(db, sms, num_tokens=128):
    return MoEDispatch._query_wideep_deepep_normal_table(
        db,
        node_num=1,
        num_tokens=num_tokens,
        num_experts=EXPERTS,
        topk=TOPK,
        hidden_size=HIDDEN,
        sms=sms,
        database_mode=common.DatabaseMode.SILICON,
    )


class TestWideEpDeepEpNormalMissingSms:
    """A node_num=1 slice that lacks sms=20 must fail cleanly and read-only."""

    @pytest.fixture(autouse=True)
    def _no_load(self, monkeypatch):
        monkeypatch.setattr(MoEDispatch, "load_data", lambda database: None)

    def test_missing_sms_raises_structured_error(self):
        """sms=20 absent -> structured interpolation miss, never a bare KeyError."""
        db = _db(_normal_table([16]))

        with pytest.raises(InterpolationDataNotAvailableError) as excinfo:
            _query(db, sms=20)

        # ValueError-derived, so existing `except ValueError` callers still classify
        # it as a data miss rather than a programming bug.
        assert isinstance(excinfo.value, ValueError)

    def test_missing_sms_does_not_mutate_cached_table(self):
        """The failed lookup must not auto-vivify an empty sms branch into the table."""
        table = _normal_table([16])
        node_slice = table[1][HIDDEN][TOPK][EXPERTS]
        assert sorted(node_slice) == [16]

        with pytest.raises(InterpolationDataNotAvailableError):
            _query(_db(table), sms=20)

        assert sorted(node_slice) == [16], "missing-sms read must leave the shared table untouched"

    def test_present_sms_still_resolves(self):
        """Sanity: when sms=20 was collected, the fast path still returns its value."""
        db = _db(_normal_table([20]))
        assert float(_query(db, sms=20)) == pytest.approx(0.2)  # 200us -> ms
