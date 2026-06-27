# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Support-matrix default: run SILICON first, re-run only explicit data gaps in HYBRID,
and keep successful rescue distinct from measured-silicon support.
AIC_SM_ALLOW_HYBRID=0 disables the rescue (pure-silicon matrix)."""

import pandas as pd
import pytest

from aiconfigurator.sdk.operations.util_empirical import note_provenance
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError
from tools.support_matrix import support_matrix as support_matrix_module
from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_HYBRID_PASS,
    STATUS_PASS,
    SupportMatrix,
    TestConstraints,
)

pytestmark = pytest.mark.unit


def _run_rescue(
    monkeypatch,
    *,
    silicon_ok,
    hybrid_ok=True,
    silicon_tier=None,
    hybrid_tier=None,
    allow_hybrid=None,
):
    """Drive run_single_test with a faked _run_mode keyed on database_mode. Rescue is on by
    default; pass allow_hybrid="0" to disable it."""
    calls: list[str] = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs["database_mode"])
        if kwargs["database_mode"] == "SILICON":
            if silicon_ok:
                if silicon_tier:
                    note_provenance(silicon_tier)
                return pd.DataFrame({"x": [1.0]})
            raise PerfDataNotAvailableError("No silicon data for this op")
        # HYBRID retry
        if not hybrid_ok:
            raise PerfDataNotAvailableError("No empirical utilisation data to estimate this op")
        if hybrid_tier:  # an empirical transfer fired (note inside the capture_provenance block)
            note_provenance(hybrid_tier)
        return pd.DataFrame({"x": [1.0]})

    monkeypatch.delenv("AIC_SM_DATABASE_MODE", raising=False)
    if allow_hybrid is not None:
        monkeypatch.setenv("AIC_SM_ALLOW_HYBRID", allow_hybrid)
    else:
        monkeypatch.delenv("AIC_SM_ALLOW_HYBRID", raising=False)
    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _m: TestConstraints(total_gpus=8, isl=256, osl=256, prefix=0, ttft=2_000_000, tpot=50_000),
    )
    statuses, _errors, _cmds, prov = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B",
        system="b200_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec={"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}},
        modes_to_test=["agg"],
        include_commands=True,
    )
    return statuses["agg"], prov["agg"], calls, _cmds["agg"]


def test_silicon_pass_stays_silicon_no_hybrid_retry(monkeypatch):
    status, src, calls, command = _run_rescue(monkeypatch, silicon_ok=True)
    assert status == STATUS_PASS and src == "silicon"
    assert calls == ["SILICON"]  # silicon passed -> no hybrid pass run
    assert "--database-mode SILICON" in command


def test_silicon_run_with_empirical_provenance_fails_invariant(monkeypatch):
    status, src, calls, command = _run_rescue(monkeypatch, silicon_ok=True, silicon_tier="xshape")

    assert status == STATUS_FAIL
    assert src == ""
    assert calls == ["SILICON"]
    assert "--database-mode SILICON" in command


def test_silicon_fail_hybrid_transfer_tagged_with_tier(monkeypatch):
    status, src, calls, command = _run_rescue(monkeypatch, silicon_ok=False, hybrid_tier="xshape")
    assert status == STATUS_HYBRID_PASS and src == "xshape"
    assert calls == ["SILICON", "HYBRID"]
    assert "--database-mode HYBRID" in command


def test_silicon_fail_hybrid_pass_no_tier_is_empirical(monkeypatch):
    # Shared-layer rows are already visible to SILICON. A no-tier HYBRID rescue is an
    # analytic empirical path that did not emit finer-grained provenance.
    status, src, calls, command = _run_rescue(monkeypatch, silicon_ok=False, hybrid_tier=None)
    assert status == STATUS_HYBRID_PASS and src == "empirical"
    assert "--database-mode HYBRID" in command


def test_silicon_fail_hybrid_fail_stays_fail(monkeypatch):
    status, src, calls, command = _run_rescue(monkeypatch, silicon_ok=False, hybrid_ok=False)
    assert status == STATUS_FAIL and src == ""
    assert calls == ["SILICON", "HYBRID"]
    assert "--database-mode SILICON" in command


def test_allow_hybrid_off_is_pure_silicon_no_rescue(monkeypatch):
    # AIC_SM_ALLOW_HYBRID=0 -> a silicon FAIL is NOT rescued; HYBRID is never run.
    status, src, calls, command = _run_rescue(monkeypatch, silicon_ok=False, hybrid_tier="xshape", allow_hybrid="0")
    assert status == STATUS_FAIL and src == ""
    assert calls == ["SILICON"]
    assert "--database-mode SILICON" in command


def test_programming_error_is_not_hidden_by_hybrid_retry(monkeypatch):
    calls: list[str] = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs["database_mode"])
        if kwargs["database_mode"] == "SILICON":
            raise TypeError("unexpected schema")
        return pd.DataFrame({"x": [1.0]})

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _m: TestConstraints(total_gpus=8, isl=256, osl=256, prefix=0, ttft=2_000_000, tpot=50_000),
    )
    statuses, errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B",
        system="b200_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec={"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}},
        modes_to_test=["agg"],
    )

    assert statuses == {"agg": STATUS_FAIL}
    assert "TypeError: unexpected schema" in errors["agg"]
    assert calls == ["SILICON"]
