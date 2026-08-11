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


def test_real_shipped_yaml_sglang_0514_sm103_triton():
    """Sanity-pin against the real shipped YAML: sglang/0.5.14/sm103 → triton head."""
    result = _resolve("sglang", "0.5.14", 103, None, None)
    assert result[0] == "triton", f"real shipped YAML must yield triton head for sglang/0.5.14/sm103; got {result}"
    assert result[-1] == "default", "real shipped YAML result must end with 'default'"
