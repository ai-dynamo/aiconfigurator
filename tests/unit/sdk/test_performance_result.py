# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for PerformanceResult: latency, energy, and source tag propagation.

Covers the new ``source`` attribute introduced in the per-op silicon vs
empirical attribution feature -- default value, _merge_source helper, and
propagation through ``+``, ``*``, ``/``, ``abs()``.
"""

import pytest

from aiconfigurator.sdk.performance_result import PerformanceResult


def pr(latency, energy=0.0, source="silicon"):
    """Convenience constructor with explicit defaults."""
    return PerformanceResult(latency, energy=energy, source=source)


# -------------------------------------------------------------------------
# Construction & defaults
# -------------------------------------------------------------------------


class TestConstruction:
    def test_default_source_is_silicon(self):
        r = PerformanceResult(10.0)
        assert r.source == "silicon"

    def test_explicit_source_empirical(self):
        r = pr(10.0, source="empirical")
        assert r.source == "empirical"

    def test_explicit_source_mixed(self):
        r = pr(10.0, source="mixed")
        assert r.source == "mixed"

    def test_float_value_is_latency(self):
        r = pr(7.5, energy=100.0)
        assert float(r) == 7.5

    def test_energy_stored(self):
        r = pr(5.0, energy=250.0)
        assert r.energy == 250.0


# -------------------------------------------------------------------------
# _merge_source
# -------------------------------------------------------------------------


class TestMergeSource:
    @pytest.mark.parametrize("src", ["silicon", "empirical", "mixed"])
    def test_same_source_preserved(self, src):
        assert PerformanceResult._merge_source(src, src) == src

    @pytest.mark.parametrize(
        "a,b",
        [
            ("silicon", "empirical"),
            ("empirical", "silicon"),
            ("silicon", "mixed"),
            ("mixed", "empirical"),
        ],
    )
    def test_different_sources_become_mixed(self, a, b):
        assert PerformanceResult._merge_source(a, b) == "mixed"


# -------------------------------------------------------------------------
# __add__: PerformanceResult + PerformanceResult
# -------------------------------------------------------------------------


class TestAddPerformanceResult:
    def test_latency_summed(self):
        result = pr(3.0) + pr(4.0)
        assert float(result) == pytest.approx(7.0)

    def test_energy_summed(self):
        result = pr(3.0, energy=100.0) + pr(4.0, energy=200.0)
        assert result.energy == pytest.approx(300.0)

    def test_same_source_preserved(self):
        result = pr(1.0, source="silicon") + pr(2.0, source="silicon")
        assert result.source == "silicon"

    def test_same_empirical_preserved(self):
        result = pr(1.0, source="empirical") + pr(2.0, source="empirical")
        assert result.source == "empirical"

    def test_different_sources_become_mixed(self):
        result = pr(1.0, source="silicon") + pr(2.0, source="empirical")
        assert result.source == "mixed"

    def test_result_is_performance_result(self):
        result = pr(1.0) + pr(2.0)
        assert isinstance(result, PerformanceResult)


# -------------------------------------------------------------------------
# __add__: PerformanceResult + scalar
# -------------------------------------------------------------------------


class TestAddScalar:
    def test_latency_updated(self):
        result = pr(5.0) + 3.0
        assert float(result) == pytest.approx(8.0)

    def test_energy_preserved(self):
        result = pr(5.0, energy=150.0) + 3.0
        assert result.energy == pytest.approx(150.0)

    def test_source_preserved(self):
        result = pr(5.0, source="empirical") + 3.0
        assert result.source == "empirical"


# -------------------------------------------------------------------------
# __radd__: sum() support
# -------------------------------------------------------------------------


class TestRadd:
    def test_sum_latency(self):
        results = [pr(1.0), pr(2.0), pr(3.0)]
        total = sum(results)
        assert float(total) == pytest.approx(6.0)

    def test_sum_energy(self):
        results = [pr(1.0, energy=10.0), pr(2.0, energy=20.0), pr(3.0, energy=30.0)]
        total = sum(results)
        assert total.energy == pytest.approx(60.0)

    def test_sum_same_source_preserved(self):
        results = [pr(1.0, source="empirical")] * 3
        total = sum(results)
        assert total.source == "empirical"

    def test_sum_mixed_sources_become_mixed(self):
        results = [pr(1.0, source="silicon"), pr(2.0, source="empirical")]
        total = sum(results)
        assert total.source == "mixed"


# -------------------------------------------------------------------------
# __mul__ / __rmul__
# -------------------------------------------------------------------------


class TestMul:
    def test_latency_scaled(self):
        result = pr(4.0) * 2
        assert float(result) == pytest.approx(8.0)

    def test_energy_scaled(self):
        result = pr(4.0, energy=100.0) * 2
        assert result.energy == pytest.approx(200.0)

    def test_source_preserved(self):
        result = pr(4.0, source="empirical") * 2
        assert result.source == "empirical"

    def test_rmul_equivalent(self):
        r = pr(4.0, energy=100.0, source="empirical")
        assert float(2 * r) == float(r * 2)
        assert (2 * r).energy == (r * 2).energy
        assert (2 * r).source == (r * 2).source


# -------------------------------------------------------------------------
# __truediv__ / __rtruediv__
# -------------------------------------------------------------------------


class TestDiv:
    def test_latency_divided(self):
        result = pr(8.0) / 2
        assert float(result) == pytest.approx(4.0)

    def test_energy_divided(self):
        result = pr(8.0, energy=200.0) / 2
        assert result.energy == pytest.approx(100.0)

    def test_source_preserved(self):
        result = pr(8.0, source="empirical") / 2
        assert result.source == "empirical"

    def test_rtruediv_returns_plain_float(self):
        r = pr(4.0)
        result = 8.0 / r
        assert isinstance(result, float)
        assert not isinstance(result, PerformanceResult)
        assert result == pytest.approx(2.0)


# -------------------------------------------------------------------------
# __abs__
# -------------------------------------------------------------------------


class TestAbs:
    def test_negative_latency(self):
        result = abs(pr(-5.0, energy=-100.0, source="empirical"))
        assert float(result) == pytest.approx(5.0)

    def test_negative_energy(self):
        result = abs(pr(-5.0, energy=-100.0))
        assert result.energy == pytest.approx(100.0)

    def test_source_preserved(self):
        result = abs(pr(-5.0, source="empirical"))
        assert result.source == "empirical"


# -------------------------------------------------------------------------
# power property
# -------------------------------------------------------------------------


class TestPower:
    def test_normal_power(self):
        r = pr(10.0, energy=3500.0)
        assert r.power == pytest.approx(350.0)

    def test_zero_latency_returns_zero(self):
        r = pr(0.0, energy=100.0)
        assert r.power == 0.0
