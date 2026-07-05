# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for collector.capabilities: hardware capability floors and the hang denylist."""

from dataclasses import dataclass

import pytest

from collector import capabilities
from collector.capabilities import case_dtypes, filter_cases, unsupported_reason

pytestmark = pytest.mark.unit


@dataclass
class FakeCase:
    """Minimal typed case exposing the attributes capabilities.py inspects."""

    label: str = "case"
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    use_fp8_kv_cache: bool = False

    def __str__(self):
        return f"FakeCase(label={self.label}, dtype={self.dtype})"


def test_dtype_floor_filters_fp8_below_sm89_but_keeps_bfloat16():
    fp8_case = FakeCase(label="fp8", dtype="fp8")
    bf16_case = FakeCase(label="bf16", dtype="bfloat16")

    kept, dropped = filter_cases([fp8_case, bf16_case], op="gemm", sm_version=80)

    assert kept == [bf16_case]
    assert [(case.label, kind) for case, kind, _reason in dropped] == [("fp8", "capability")]
    assert dropped[0][2] == "dtype fp8 requires SM>=89"


def test_nvfp4_floor_is_sm100():
    nvfp4_case = FakeCase(dtype="nvfp4")

    kept_sm100, dropped_sm100 = filter_cases([nvfp4_case], op="moe", sm_version=100)
    assert kept_sm100 == [nvfp4_case]
    assert dropped_sm100 == []

    kept_sm90, dropped_sm90 = filter_cases([nvfp4_case], op="moe", sm_version=90)
    assert kept_sm90 == []
    assert [(kind, reason) for _case, kind, reason in dropped_sm90] == [
        ("capability", "dtype nvfp4 requires SM>=100"),
    ]


def test_op_floor_drops_dsa_context_module_wholesale_below_sm90():
    cases = [FakeCase(dtype="bfloat16"), FakeCase(dtype="fp8")]

    kept, dropped = filter_cases(cases, op="dsa_context_module", sm_version=89)

    assert kept == []
    assert all(kind == "capability" for _case, kind, _reason in dropped)
    assert {reason for _case, _kind, reason in dropped} == {"op dsa_context_module requires SM>=90"}


def test_unknown_dtype_and_unknown_sm_are_permissive():
    unknown_dtype_case = FakeCase(dtype="future_dtype")
    nvfp4_case = FakeCase(dtype="nvfp4")

    assert unsupported_reason(unknown_dtype_case, op="gemm", sm_version=80) is None
    assert unsupported_reason(nvfp4_case, op="dsa_context_module", sm_version=None) is None

    kept, dropped = filter_cases([unknown_dtype_case, nvfp4_case], op="dsa_context_module", sm_version=None)
    assert kept == [unknown_dtype_case, nvfp4_case]
    assert dropped == []


def test_fp8_kv_cache_flag_implies_fp8_floor():
    flagged_case = FakeCase(use_fp8_kv_cache=True)

    assert case_dtypes(flagged_case) == ["fp8"]

    kept, dropped = filter_cases([flagged_case], op="attention_context", sm_version=80)
    assert kept == []
    assert [(kind, reason) for _case, kind, reason in dropped] == [
        ("capability", "dtype fp8 requires SM>=89"),
    ]


def test_denylist_drops_cases_by_substring_match(monkeypatch):
    monkeypatch.setattr(capabilities, "_load_denylist", lambda: (("label=hangs", "deadlocks in NCCL init"),))
    hanging_case = FakeCase(label="hangs", dtype="bfloat16")
    good_case = FakeCase(label="good", dtype="bfloat16")

    kept, dropped = filter_cases([hanging_case, good_case], op="moe", sm_version=100)

    assert kept == [good_case]
    assert [(case.label, kind, reason) for case, kind, reason in dropped] == [
        ("hangs", "denylist", "deadlocks in NCCL init"),
    ]


def test_denylist_reason_defaults_to_matched_substring(monkeypatch):
    monkeypatch.setattr(capabilities, "_load_denylist", lambda: (("label=hangs", ""),))

    reason = capabilities.denylist_reason(FakeCase(label="hangs"))

    assert reason == "denylisted (contains 'label=hangs')"


@pytest.mark.parametrize(
    "entry",
    [
        {"contains": "tp=32"},  # neither field
        {"contains": "tp=32", "reason": "deadlocks in NCCL init"},  # missing added
        {"contains": "tp=32", "added": "2026-07-04"},  # missing reason
        {"contains": "tp=32", "reason": " ", "added": "2026-07-04"},  # blank reason
    ],
)
def test_denylist_loader_requires_dated_reason(monkeypatch, tmp_path, entry):
    """Hang suppression must stay auditable: entries without a reason+date fail loudly."""
    import yaml

    (tmp_path / "denylist.yaml").write_text(yaml.safe_dump({"schema_version": 1, "entries": [entry]}))
    monkeypatch.setattr(capabilities, "_CASES_DIR", tmp_path)
    capabilities._load_denylist.cache_clear()
    try:
        with pytest.raises(ValueError, match="'reason' and 'added'"):
            capabilities._load_denylist()
    finally:
        capabilities._load_denylist.cache_clear()


def test_denylist_loader_accepts_dated_reason(monkeypatch, tmp_path):
    import yaml

    (tmp_path / "denylist.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "entries": [{"contains": "tp=32", "reason": "deadlocks in NCCL init", "added": "2026-07-04"}],
            }
        )
    )
    monkeypatch.setattr(capabilities, "_CASES_DIR", tmp_path)
    capabilities._load_denylist.cache_clear()
    try:
        assert capabilities._load_denylist() == (("tp=32", "deadlocks in NCCL init"),)
    finally:
        capabilities._load_denylist.cache_clear()
