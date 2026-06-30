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
    silicon_error=None,
):
    """Drive run_single_test with a faked _run_mode keyed on database_mode. Rescue is on by
    default; pass allow_hybrid="0" to disable it."""
    calls: list[str] = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs["database_mode"])
        if kwargs["database_mode"] == "SILICON":
            if silicon_error is not None:
                raise silicon_error
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
    statuses, errors, commands, prov = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B",
        system="b200_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec={"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}},
        modes_to_test=["agg"],
        include_commands=True,
    )
    return statuses["agg"], errors["agg"], prov["agg"], calls, commands["agg"]


@pytest.mark.parametrize(
    ("run_kwargs", "expected"),
    [
        pytest.param(
            {"silicon_ok": True},
            (STATUS_PASS, "silicon", ["SILICON"], "SILICON", None),
            id="silicon-pass",
        ),
        pytest.param(
            {"silicon_ok": True, "silicon_tier": "xshape"},
            (STATUS_FAIL, "", ["SILICON"], "SILICON", "SILICON support run emitted empirical provenance"),
            id="silicon-provenance-invariant",
        ),
        pytest.param(
            {"silicon_ok": False, "hybrid_tier": "xshape"},
            (STATUS_HYBRID_PASS, "xshape", ["SILICON", "HYBRID"], "HYBRID", None),
            id="hybrid-transfer",
        ),
        pytest.param(
            {"silicon_ok": False},
            (STATUS_HYBRID_PASS, "empirical", ["SILICON", "HYBRID"], "HYBRID", None),
            id="hybrid-own-data",
        ),
        pytest.param(
            {"silicon_ok": False, "hybrid_ok": False},
            (STATUS_FAIL, "", ["SILICON", "HYBRID"], "SILICON", "No silicon data"),
            id="hybrid-miss",
        ),
        pytest.param(
            {"silicon_ok": False, "hybrid_tier": "xshape", "allow_hybrid": "0"},
            (STATUS_FAIL, "", ["SILICON"], "SILICON", "No silicon data"),
            id="hybrid-disabled",
        ),
        pytest.param(
            {"silicon_ok": False, "silicon_error": TypeError("unexpected schema")},
            (STATUS_FAIL, "", ["SILICON"], "SILICON", "TypeError: unexpected schema"),
            id="programming-error",
        ),
    ],
)
def test_silicon_first_hybrid_rescue_state_machine(monkeypatch, run_kwargs, expected):
    status, error, source, calls, command = _run_rescue(monkeypatch, **run_kwargs)
    expected_status, expected_source, expected_calls, command_mode, error_fragment = expected

    assert status == expected_status
    assert source == expected_source
    assert calls == expected_calls
    assert f"--database-mode {command_mode}" in command
    if error_fragment is None:
        assert error is None
    else:
        assert error_fragment in error
