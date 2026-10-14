# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the attention lane resolver.

AIC-1715/1716: resolve_attention_lane_order returns an ordered tuple of
attention lane names given (backend, version, sm_version, override).
"""

import logging

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Minimal YAML fixture mirroring the shipped attention_lane_defaults.yaml
# ---------------------------------------------------------------------------

_MINIMAL_YAML = """\
# Test fixture: minimal copy of attention_lane_defaults.yaml
sglang:
  "0.5.14":
    90: fa3
    100: triton
    103: triton
    120: flashinfer
vllm:
  "0.24.0":
    90: default
    100: default
    103: default
"""


@pytest.fixture
def systems_root(tmp_path):
    """Write a minimal attention_lane_defaults.yaml under tmp_path and return the path."""
    (tmp_path / "attention_lane_defaults.yaml").write_text(_MINIMAL_YAML, encoding="utf-8")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Helper: import the resolver fresh each test (avoids cross-test cache leaks
# at the _load_defaults level since each test gets a unique tmp_path string).
# ---------------------------------------------------------------------------


def _resolve(backend, version, sm_version, override, systems_root):
    from aiconfigurator_core.sdk.attention_lanes import resolve_attention_lane_order

    return resolve_attention_lane_order(backend, version, sm_version, override, systems_root)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_KNOWN_LANES_SORTED = ("fa3", "fla", "flashinfer", "triton", "trtllm_mha")


# ---------------------------------------------------------------------------
# _parse_version: dotted AND glued PEP-440-style suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version,expected",
    [
        # Plain dotted versions: unaffected baseline.
        ("0.5.14", (0, 5, 14)),
        ("1.3.0", (1, 3, 0)),
        # Dot-separated suffixes: the whole trailing segment is dropped, the
        # numeric prefix before it is kept (pre-existing, still covered).
        ("0.5.14.post1", (0, 5, 14)),
        ("0.5.14.dev3", (0, 5, 14)),
        # Glued suffixes: the suffix rides directly on the last numeric
        # segment with no separating dot. The segment's LEADING digits must
        # still be captured -- this is the R1 regression: naive int()-or-stop
        # parsing dropped the entire segment (including its leading digits)
        # on the first ValueError, so "1.3.0rc23" used to parse as (1, 3)
        # instead of (1, 3, 0), sorting BELOW the "1.3.0" it is a release
        # candidate for.
        ("1.3.0rc23", (1, 3, 0)),
        ("1.3.0a1", (1, 3, 0)),
        ("1.3.0b2", (1, 3, 0)),
        ("0.5.14rc23", (0, 5, 14)),
        # Glued suffix on a non-final segment: still stops after capturing
        # that segment's leading digits (matches the pre-existing "stop at
        # first non-numeric segment" contract -- only the DROP-THE-DIGITS
        # part of that contract was the bug).
        ("1.0rc5.2", (1, 0)),
        # A segment with no leading digits at all: nothing to capture, same
        # as the pre-existing dot-separated-suffix behavior.
        ("1.3.dev", (1, 3)),
        # Degenerate: no numeric segments anywhere falls back to (0,).
        ("dev", (0,)),
    ],
)
def test_parse_version_dotted_and_glued_forms(version, expected):
    from aiconfigurator_core.sdk.attention_lanes import _parse_version

    assert _parse_version(version) == expected, f"_parse_version({version!r})"


def test_parse_version_glued_suffix_sorts_at_or_above_its_base_release():
    """The concrete failure mode: a glued release-candidate string must sort
    >= the release it is a candidate for, so it still floor-matches a map
    entry keyed on the plain release version."""
    from aiconfigurator_core.sdk.attention_lanes import _parse_version

    assert _parse_version("1.3.0rc23") >= _parse_version("1.3.0")


def test_override_is_first(systems_root):
    """When override is given, it must appear first in the result."""
    result = _resolve("sglang", "0.5.14", 103, "trtllm_mha", systems_root)
    assert result[0] == "trtllm_mha", f"override must be first; got {result}"


def test_sglang_0514_sm103_triton_head(systems_root):
    """sglang/0.5.14/sm103 → triton is the framework default; must be first when no override."""
    result = _resolve("sglang", "0.5.14", 103, None, systems_root)
    assert result[0] == "triton", f"expected triton head for sglang/0.5.14/sm103; got {result}"


def test_floor_match_version(systems_root):
    """Version '0.5.15' should floor-match to the '0.5.14' entry and put triton first (sm103)."""
    result = _resolve("sglang", "0.5.15", 103, None, systems_root)
    assert result[0] == "triton", f"floor-match '0.5.15'→'0.5.14' should yield triton first; got {result}"


def test_unknown_backend_sorted_known_lanes_warning(systems_root, caplog):
    """An unknown backend: no map default; result is sorted known lanes then 'default'; warning logged."""
    with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.attention_lanes"):
        result = _resolve("unknown_backend", "0.5.14", 103, None, systems_root)

    # All known lanes must appear, sorted, before "default"
    assert result[:-1] == _KNOWN_LANES_SORTED, f"expected sorted known lanes before 'default'; got {result}"
    assert result[-1] == "default"

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "must log a WARNING for unknown backend"


def test_determinism(systems_root):
    """Calling the resolver twice with the same inputs must return the same tuple."""
    result1 = _resolve("sglang", "0.5.14", 103, None, systems_root)
    result2 = _resolve("sglang", "0.5.14", 103, None, systems_root)
    assert result1 == result2, "resolver must be deterministic"


def test_default_exactly_once_and_last(systems_root):
    """'default' must appear exactly once and as the last element, for all representative calls."""
    calls = [
        ("sglang", "0.5.14", 103, None),
        ("sglang", "0.5.14", 103, "fa3"),
        ("sglang", "0.5.14", 103, "default"),  # override == "default"
        ("unknown", "0.0.0", 0, None),
    ]
    for backend, version, sm, override in calls:
        result = _resolve(backend, version, sm, override, systems_root)
        assert result.count("default") == 1, (
            f"'default' must appear exactly once; got {result} for ({backend!r}, {version!r}, {sm}, {override!r})"
        )
        assert result[-1] == "default", (
            f"'default' must be last; got {result} for ({backend!r}, {version!r}, {sm}, {override!r})"
        )


def test_override_equal_to_map_default_not_duplicated(systems_root):
    """When override equals the map default, the lane must appear exactly once."""
    # sglang/0.5.14/sm103 → triton; override is also "triton"
    result = _resolve("sglang", "0.5.14", 103, "triton", systems_root)
    assert result.count("triton") == 1, f"triton must not be duplicated; got {result}"
    assert result[0] == "triton", f"triton must still be first (it is the override); got {result}"


def test_version_below_all_yaml_entries_warns(systems_root, caplog):
    """Known backend with version below all YAML keys: no map hit, warning logged, sorted known lanes."""
    # sglang YAML starts at "0.5.14"; "0.5.9" is below it → no valid floor-match
    with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.attention_lanes"):
        result = _resolve("sglang", "0.5.9", 103, None, systems_root)

    assert result[:-1] == _KNOWN_LANES_SORTED, f"expected sorted known lanes before 'default'; got {result}"
    assert result[-1] == "default"

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "must log a WARNING when version is below all YAML entries for a known backend"


def test_pep440_suffix_floor_matches_numeric_prefix(systems_root):
    """'0.5.14.post1' must floor-match the '0.5.14' entry and yield triton for sm103."""
    result = _resolve("sglang", "0.5.14.post1", 103, None, systems_root)
    assert result[0] == "triton", f"'0.5.14.post1' should floor-match '0.5.14' and yield triton head; got {result}"


def test_glued_pep440_suffix_floor_matches_numeric_prefix(systems_root):
    """'0.5.14rc23' (suffix GLUED to the last segment, no dot) must floor-match
    the '0.5.14' entry and yield triton for sm103 -- same contract as the
    dot-separated '0.5.14.post1' form above.

    Regression: naive parsing (int() on each dot-split segment, stop at the
    first failure) drops the WHOLE glued segment on failure, so
    '0.5.14rc23' used to parse as (0, 5) -- shorter than, and sorting BELOW,
    the clean '0.5.14' -> (0, 5, 14) it is a release candidate for. That took
    '0.5.14rc23' below the '0.5.14' floor-match entirely, silently dropping
    the framework-default lane.
    """
    result = _resolve("sglang", "0.5.14rc23", 103, None, systems_root)
    assert result[0] == "triton", f"'0.5.14rc23' should floor-match '0.5.14' and yield triton head; got {result}"


def test_pinned_head_is_carried_not_reconstructed(systems_root):
    """The resolved order must REPORT its pinned head, for every head lane.

    Regression (AIC-1715/1716): the split used to be reconstructed by scanning
    the flat tuple for the donor tier, which cannot see a pinned ``fa3`` — the
    donor tier starts with ``fa3`` too, so a pin of it is byte-identical to
    "nothing pinned" and the pinned head was handed to the density ranking as a
    donor. Only an explicitly carried tier boundary distinguishes the two.
    """
    from aiconfigurator_core.sdk.attention_lanes import split_attention_lane_tiers

    cases = {
        ("sglang", 90, None): ("fa3",),  # framework-default map lane
        ("sglang", 90, "fa3"): ("fa3",),  # override == map lane, listed once
        ("sglang", 999, "fa3"): ("fa3",),  # override only (no sm entry)
        ("sglang", 103, "fa3"): ("fa3", "triton"),  # override, then map lane
        ("sglang", 103, None): ("triton",),  # control: non-fa3 head
        ("sglang", 999, None): (),  # control: nothing pinned
    }
    for (backend, sm, override), expected_pinned in cases.items():
        order = _resolve(backend, "0.5.14", sm, override, systems_root)
        pinned, donors = split_attention_lane_tiers(order)
        assert pinned == expected_pinned, f"pinned head for ({backend!r}, {sm}, {override!r}); got {pinned}"
        assert pinned + donors == order, "the two tiers must reconstitute the flat order exactly"


def test_real_shipped_yaml_sglang_0514_sm103_triton():
    """Sanity-pin against the real shipped YAML: sglang/0.5.14/sm103 → triton head."""
    result = _resolve("sglang", "0.5.14", 103, None, None)
    assert result[0] == "triton", f"real shipped YAML must yield triton head for sglang/0.5.14/sm103; got {result}"
    assert result[-1] == "default", "real shipped YAML result must end with 'default'"


def test_custom_systems_root_without_lane_defaults_falls_back_to_packaged_copy(tmp_path, caplog):
    """Custom perf roots inherit packaged framework defaults when absent."""
    custom_root = tmp_path / "custom-systems"
    custom_root.mkdir()

    with caplog.at_level(logging.WARNING, logger="aiconfigurator_core.sdk.attention_lanes"):
        result = _resolve("sglang", "0.5.14", 103, None, str(custom_root))

    assert result[0] == "triton", f"packaged sglang/0.5.14/sm103 default must survive; got {result}"
    assert "falling back to packaged attention-lane defaults" in caplog.text
