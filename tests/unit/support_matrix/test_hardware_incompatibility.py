# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_HW_INCOMPATIBLE,
    STATUS_PASS,
    SupportMatrix,
    get_hardware_incompatibility,
)

pytestmark = pytest.mark.unit


def _system_spec(*, sm_version: int, fp8: bool = False, fp4: bool = False) -> dict:
    gpu = {"sm_version": sm_version}
    if fp8:
        gpu["fp8_tc_flops"] = 1
    if fp4:
        gpu["fp4_tc_flops"] = 1
    return {"gpu": gpu}


def test_fp8_model_is_hardware_incompatible_without_fp8_support():
    incompatibility = get_hardware_incompatibility(
        model="Qwen/Qwen3-32B-FP8",
        system="a100_sxm",
        backend="trtllm",
        system_spec=_system_spec(sm_version=80),
    )

    assert incompatibility is not None
    assert incompatibility.missing_datatypes == ("FP8",)
    assert incompatibility.reason == "a100_sxm (SM80) does not support FP8 required by Qwen/Qwen3-32B-FP8"


def test_fp8_model_is_allowed_when_system_advertises_fp8_support():
    incompatibility = get_hardware_incompatibility(
        model="Qwen/Qwen3-32B-FP8",
        system="l40s",
        backend="trtllm",
        system_spec=_system_spec(sm_version=89, fp8=True),
    )

    assert incompatibility is None


def test_fp8_model_is_allowed_for_b60_software_fallback_without_native_fp8_flops():
    incompatibility = get_hardware_incompatibility(
        model="nvidia/Llama-3.1-70B-Instruct-FP8",
        system="b60",
        backend="vllm",
        system_spec={"gpu": {"bfloat16_tc_flops": 1}},
    )

    assert incompatibility is None


def test_fp4_model_is_hardware_incompatible_without_fp4_support():
    incompatibility = get_hardware_incompatibility(
        model="nvidia/Qwen3-235B-A22B-NVFP4",
        system="h100_sxm",
        backend="trtllm",
        system_spec=_system_spec(sm_version=90, fp8=True),
    )

    assert incompatibility is not None
    assert incompatibility.missing_datatypes == ("FP4",)
    assert "does not support FP4" in incompatibility.reason


@pytest.mark.parametrize("model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
def test_mxfp4_model_is_not_native_fp4_hardware_incompatible_on_hopper(model):
    incompatibility = get_hardware_incompatibility(
        model=model,
        system="h100_sxm",
        backend="trtllm",
        system_spec=_system_spec(sm_version=90, fp8=True),
    )

    assert incompatibility is None


def test_run_single_test_short_circuits_hardware_incompatible_model(monkeypatch):
    def fail_run_mode(**_kwargs):
        raise AssertionError("TaskRunner should not run for hardware-incompatible combinations")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fail_run_mode))

    status_dict, error_dict = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B-FP8",
        system="a100_sxm",
        backend="trtllm",
        version="1.0.0",
        system_spec=_system_spec(sm_version=80),
    )

    assert status_dict == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "does not support FP8" in error_dict["agg"]
    assert STATUS_PASS not in status_dict.values()


def test_run_single_test_marks_missing_silicon_data_as_unsupported(monkeypatch):
    def missing_data_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. "
            "Showing last exception: Failed to query moe data. "
            "Missing silicon data for the requested lookup."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(missing_data_run_mode))

    status_dict, error_dict = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B",
        system="rtx_pro_6000_server",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec=_system_spec(sm_version=120, fp8=True, fp4=True),
    )

    assert status_dict == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert error_dict["agg"].startswith("Deterministic unsupported configuration:")
    assert "Missing silicon data for the requested lookup" in error_dict["agg"]


def test_run_single_test_marks_unsupported_quant_mode_as_unsupported(monkeypatch):
    def unsupported_quant_run_mode(**_kwargs):
        raise ValueError(
            "Unsupported gemm quant mode 'fp8_static' for system='rtx_pro_6000_server', "
            "backend='vllm', version='0.19.0'. Supported gemm modes: ['bfloat16', 'fp8']"
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(unsupported_quant_run_mode))

    status_dict, error_dict = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B",
        system="rtx_pro_6000_server",
        backend="vllm",
        version="0.19.0",
        system_spec=_system_spec(sm_version=120, fp8=True, fp4=True),
    )

    assert status_dict == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "Unsupported gemm quant mode" in error_dict["agg"]


def test_run_single_test_keeps_unclassified_errors_as_fail(monkeypatch):
    def transient_run_mode(**_kwargs):
        raise RuntimeError("temporary worker crash")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(transient_run_mode))

    status_dict, error_dict = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B",
        system="rtx_pro_6000_server",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec=_system_spec(sm_version=120, fp8=True, fp4=True),
    )

    assert status_dict == {"agg": STATUS_FAIL, "disagg": STATUS_FAIL}
    assert "temporary worker crash" in error_dict["agg"]
