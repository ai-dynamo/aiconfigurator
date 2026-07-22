# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import tools.support_matrix.support_matrix as support_matrix_module
from tools.support_matrix.support_matrix import (
    STATUS_FRAMEWORK_INCOMPATIBLE,
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


def test_fp4_model_is_hardware_incompatible_below_sm80():
    incompatibility = get_hardware_incompatibility(
        model="nvidia/Qwen3-235B-A22B-NVFP4",
        system="v100",
        backend="trtllm",
        system_spec=_system_spec(sm_version=70, fp8=False),
    )

    assert incompatibility is not None
    assert "FP4" in incompatibility.missing_datatypes
    assert "does not support" in incompatibility.reason


def test_fp4_model_is_allowed_on_ampere_via_software_fallback():
    incompatibility = get_hardware_incompatibility(
        model="nvidia/Qwen3-235B-A22B-NVFP4",
        system="a100_sxm",
        backend="trtllm",
        system_spec=_system_spec(sm_version=80, fp8=True),
    )

    assert incompatibility is None


def test_fp4_model_is_allowed_on_hopper_via_software_fallback():
    incompatibility = get_hardware_incompatibility(
        model="nvidia/Qwen3-235B-A22B-NVFP4",
        system="h100_sxm",
        backend="trtllm",
        system_spec=_system_spec(sm_version=90, fp8=True),
    )

    assert incompatibility is None


def test_sglang_dsa_model_is_hardware_incompatible_below_sm90():
    incompatibility = get_hardware_incompatibility(
        model="zai-org/GLM-5",
        system="l40s",
        backend="sglang",
        system_spec=_system_spec(sm_version=89, fp8=True),
    )

    assert incompatibility is not None
    assert incompatibility.missing_datatypes == ()
    assert "SGLang DSA/NSA module collectors require SM90+" in incompatibility.reason


def test_sglang_dsa_model_is_allowed_on_sm90_plus():
    incompatibility = get_hardware_incompatibility(
        model="zai-org/GLM-5",
        system="h100_sxm",
        backend="sglang",
        system_spec=_system_spec(sm_version=90, fp8=True),
    )

    assert incompatibility is None


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


def test_run_single_test_propagates_hardware_preflight_failures(monkeypatch):
    def fail_get_model_info(_model):
        raise RuntimeError("metadata unavailable")

    def fail_run_mode(**_kwargs):
        raise AssertionError("TaskRunner should not run after hardware-preflight failure")

    monkeypatch.setattr(support_matrix_module, "_get_model_info", fail_get_model_info)
    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fail_run_mode))

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        SupportMatrix.run_single_test(
            model="Qwen/Qwen3-32B",
            system="l40s",
            backend="sglang",
            version="0.5.12",
            system_spec=_system_spec(sm_version=89, fp8=True),
        )


@pytest.mark.parametrize(
    ("model", "backend", "version", "message"),
    [
        (
            "Qwen/Qwen3-32B-FP8",
            "sglang",
            "0.5.10",
            "Unsupported gemm quant mode 'fp8_block' for system='rtx_pro_6000_server'",
        ),
        (
            "nvidia/Kimi-K2.5-NVFP4",
            "sglang",
            "0.5.10",
            "Unsupported moe quant mode 'nvfp4' for system='rtx_pro_6000_server'",
        ),
        (
            "MiniMaxAI/MiniMax-M2.5",
            "trtllm",
            "1.3.0rc10",
            "Unsupported moe quant mode 'fp8_block' for system='rtx_pro_6000_server'",
        ),
        (
            "moonshotai/Kimi-K2.5",
            "trtllm",
            "1.3.0rc10",
            "Unsupported moe quant mode 'int4_wo' for system='rtx_pro_6000_server'",
        ),
        (
            "deepseek-ai/DeepSeek-V4-Flash",
            "trtllm",
            "1.3.0rc10",
            "Unsupported moe quant mode 'w4a8_mxfp4_mxfp8' for system='rtx_pro_6000_server'",
        ),
        (
            "zai-org/GLM-5",
            "vllm",
            "0.19.0",
            "File does not exist at "
            "src/aiconfigurator/systems/data/rtx_pro_6000_server/vllm/0.19.0/dsa_context_module_perf.txt",
        ),
        (
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "vllm",
            "0.19.0",
            "File does not exist at src/aiconfigurator/systems/data/rtx_pro_6000_server/nccl/2.28.9/nccl_perf.txt",
        ),
        (
            "google/gemma-4-26B-A4B",
            "sglang",
            "0.5.10",
            "Failed to query context attention data for b=1, s=128.0, prefix=128.0",
        ),
        (
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "trtllm",
            "1.3.0rc10",
            "Failed to query moe data for num_tokens=128.0, hidden_size=5120",
        ),
    ],
)
def test_run_single_test_marks_known_rtx_pro_sm120_framework_gaps(monkeypatch, model, backend, version, message):
    def fail_run_mode(**_kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fail_run_mode))

    status_dict, error_dict = SupportMatrix.run_single_test(
        model=model,
        system="rtx_pro_6000_server",
        backend=backend,
        version=version,
        system_spec=_system_spec(sm_version=120, fp8=True, fp4=True),
        modes_to_test=("agg",),
    )

    assert status_dict == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert message in error_dict["agg"]
