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


def _resolve(backend, version, sm_version, override, systems_root, architecture=None):
    from aiconfigurator_core.sdk.attention_lanes import resolve_attention_lane_order

    return resolve_attention_lane_order(backend, version, sm_version, override, systems_root, architecture)


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


# ---------------------------------------------------------------------------
# AIC-1762: per-architecture defaults (final-review finding I1)
#
# ONE architecture is declared in the shipped YAML: Qwen3_5MoeForCausalLM
# (Qwen3.8-Max) -> sglang -> "0.5.17" -> {90: fa3, 100: trtllm_mha,
# 103: trtllm_mha, 120: flashinfer}. These tests exercise the new precedence
# step directly against a synthetic fixture (mirroring the shipped schema)
# AND pin the real shipped file at the bottom.
# ---------------------------------------------------------------------------

_ARCH_YAML = """\
# Test fixture: minimal copy of attention_lane_defaults.yaml + architectures:
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
architectures:
  Qwen3_5MoeForCausalLM:
    sglang:
      "0.5.17":
        90: fa3
        100: trtllm_mha
        103: trtllm_mha
        120: flashinfer
"""

_MAX_ARCH = "Qwen3_5MoeForCausalLM"
_CONDGEN_ARCH = "Qwen3_5MoeForConditionalGeneration"  # the 397B/35B/122B architecture — deliberately NOT declared


@pytest.fixture
def arch_systems_root(tmp_path):
    """systems_root holding both the global maps and one architecture entry."""
    (tmp_path / "attention_lane_defaults.yaml").write_text(_ARCH_YAML, encoding="utf-8")
    return str(tmp_path)


@pytest.mark.parametrize(
    "sm_version,expected_head",
    [(90, "fa3"), (100, "trtllm_mha"), (103, "trtllm_mha"), (120, "flashinfer")],
)
def test_architecture_default_heads_by_sm_when_no_override(arch_systems_root, sm_version, expected_head):
    """Max at sglang/0.5.17, no override: the architecture's own per-sm lane heads
    the order — trtllm_mha at sm100/103 (NOT the inherited triton), per I1."""
    result = _resolve("sglang", "0.5.17", sm_version, None, arch_systems_root, architecture=_MAX_ARCH)
    assert result[0] == expected_head, f"sm{sm_version}: expected {expected_head} head; got {result}"
    assert result[-1] == "default"


def test_architecture_default_pinned_head_is_the_architecture_lane_alone(arch_systems_root):
    """The architecture default REPLACES the global map for this resolution —
    it is not stacked as a second pin ahead of the (wrong, inherited) triton
    entry. Only trtllm_mha is pinned; triton is an ordinary alphabetical donor."""
    from aiconfigurator_core.sdk.attention_lanes import split_attention_lane_tiers

    result = _resolve("sglang", "0.5.17", 103, None, arch_systems_root, architecture=_MAX_ARCH)
    pinned, donors = split_attention_lane_tiers(result)
    assert pinned == ("trtllm_mha",), f"expected trtllm_mha as the sole pinned lane; got {pinned}"
    assert "triton" in donors, "triton must still be reachable as an ordinary donor, just not pinned"


def test_architecture_default_only_applies_to_its_own_backend():
    """The shipped/fixture architecture entry only declares sglang; a vllm
    query for the same architecture must fall straight to the global vllm map
    -- byte-identical to omitting the architecture argument. (The global
    vllm map's lane VALUE is itself the string "default", which the donor-
    tier rule treats as "nothing real pinned" -- see test_default_exactly_once_and_last
    -- so both sides resolve to the plain alphabetical + "default" tail, not
    a single-element ("default",) tuple; the equality is what this test
    actually guards.)"""
    result_arch = _resolve("vllm", "0.24.0", 100, None, None, architecture=_MAX_ARCH)
    result_noarch = _resolve("vllm", "0.24.0", 100, None, None, architecture=None)
    assert result_arch == result_noarch == _KNOWN_LANES_SORTED + ("default",), (
        f"vllm must be untouched by the sglang-only architecture entry; got {result_arch}"
    )


def test_override_wins_over_architecture_default(arch_systems_root):
    """An explicit override still wins outright, even for an architecture with
    its own default -- 'explicit intent stays first-class' (owner design)."""
    result = _resolve("sglang", "0.5.17", 103, "triton", arch_systems_root, architecture=_MAX_ARCH)
    assert result[0] == "triton", f"override must win over the architecture default; got {result}"
    assert result.count("triton") == 1, "override lane must not be duplicated by a later step"


def test_conditional_generation_397b_architecture_is_byte_identical_to_no_architecture(arch_systems_root):
    """Qwen3_5MoeForConditionalGeneration (397B/35B/122B) is deliberately NOT
    declared under architectures: — passing it must not change anything
    relative to omitting the architecture argument entirely (the whole point
    of I1's fix: only Max's serving-true dispatch differs)."""
    with_arch = _resolve("sglang", "0.5.14", 103, None, arch_systems_root, architecture=_CONDGEN_ARCH)
    without_arch = _resolve("sglang", "0.5.14", 103, None, arch_systems_root, architecture=None)
    assert with_arch == without_arch == ("triton", "fa3", "fla", "flashinfer", "trtllm_mha", "default")


def test_unlisted_architecture_is_byte_identical_to_no_architecture(arch_systems_root):
    """An architecture with no entry at all (not just 397B specifically) must
    resolve identically to the pre-AIC-1762 (no architecture argument) call."""
    with_arch = _resolve("sglang", "0.5.14", 103, None, arch_systems_root, architecture="SomeUnrelatedArchitecture")
    without_arch = _resolve("sglang", "0.5.14", 103, None, arch_systems_root, architecture=None)
    assert with_arch == without_arch


def test_version_below_architecture_floor_falls_back_to_global_map(arch_systems_root):
    """Max's own architecture entry starts at 0.5.17; a query at 0.5.14 (below
    that floor) must fall through to the global sglang map (triton @ sm103),
    NOT silently reuse the architecture's own data across a floor it declares."""
    below_floor = _resolve("sglang", "0.5.14", 103, None, arch_systems_root, architecture=_MAX_ARCH)
    global_only = _resolve("sglang", "0.5.14", 103, None, arch_systems_root, architecture=None)
    assert below_floor == global_only == ("triton", "fa3", "fla", "flashinfer", "trtllm_mha", "default")


@pytest.mark.parametrize(
    "malformed_yaml,match",
    [
        # Unknown lane name (typo: hyphen instead of underscore).
        (
            'architectures:\n  Qwen3_5MoeForCausalLM:\n    sglang:\n      "0.5.17":\n        100: trtllm-mha\n',
            "not a known lane",
        ),
        # sm key is a string, not an int (YAML would need explicit quoting to do this).
        (
            'architectures:\n  Qwen3_5MoeForCausalLM:\n    sglang:\n      "0.5.17":\n        "100": trtllm_mha\n',
            "must be an int",
        ),
        # version level is a scalar, not a mapping.
        (
            'architectures:\n  Qwen3_5MoeForCausalLM:\n    sglang:\n      "0.5.17": trtllm_mha\n',
            "must be a mapping",
        ),
    ],
)
def test_malformed_architecture_entry_fails_loudly(tmp_path, malformed_yaml, match):
    """A structurally broken architectures: entry must raise, never silently
    resolve to None and masquerade as an ordinary "unlisted architecture" miss
    (the repo's fail-closed convention)."""
    (tmp_path / "attention_lane_defaults.yaml").write_text(malformed_yaml, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _resolve("sglang", "0.5.17", 100, None, str(tmp_path), architecture=_MAX_ARCH)


def test_real_shipped_yaml_qwen38max_sm100_trtllm_mha():
    """Sanity-pin against the REAL shipped YAML (not a fixture): Max at
    sglang/0.5.17/sm100, no override -> trtllm_mha heads the order."""
    result = _resolve("sglang", "0.5.17", 100, None, None, architecture=_MAX_ARCH)
    assert result[0] == "trtllm_mha", f"real shipped YAML must yield trtllm_mha head for Max @ sm100; got {result}"
    assert result[-1] == "default"


def test_real_shipped_yaml_397b_still_triton_at_sm103():
    """Sanity-pin: the 397B architecture on the REAL shipped YAML is untouched
    — still triton @ sm103 (the pre-AIC-1762 answer)."""
    result = _resolve("sglang", "0.5.14", 103, None, None, architecture=_CONDGEN_ARCH)
    assert result[0] == "triton", f"397B must still resolve triton @ sm103; got {result}"
