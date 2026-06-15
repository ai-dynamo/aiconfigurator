# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.support_matrix import support_matrix as support_matrix_module
from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_FRAMEWORK_INCOMPATIBLE,
    STATUS_HW_INCOMPATIBLE,
    SupportMatrix,
    TestConstraints,
)

pytestmark = pytest.mark.unit


def _b200_system_spec() -> dict:
    return {"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}}


def _l40s_system_spec() -> dict:
    return {"gpu": {"sm_version": 89, "fp8_tc_flops": 1}}


def _patch_large_constraints(monkeypatch) -> None:
    monkeypatch.setattr(
        support_matrix_module,
        "_get_test_constraints",
        lambda _model: TestConstraints(total_gpus=128, isl=256, osl=256, prefix=128, ttft=2_000_000, tpot=50_000),
    )


def test_dsv4_vllm_019_unsupported_mxfp8_quant_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError(
            "Unsupported moe quant mode 'w4a8_mxfp4_mxfp8' for system='b200_sxm', backend='vllm', version='0.19.0'."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="deepseek-ai/DeepSeek-V4-Flash",
        system="b300_sxm",
        backend="vllm",
        version="0.19.0",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "Unsupported moe quant mode" in errors["agg"]


def test_dsv4_vllm_019_missing_mhc_data_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. Showing last exception: "
            "DeepSeek-V4 mHC module data not loaded for system='b200_sxm', "
            "backend='vllm', version='0.19.0'."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="sgl-project/DeepSeek-V4-Pro-FP8",
        system="b200_sxm",
        backend="vllm",
        version="0.19.0",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "DeepSeek-V4 mHC module data not loaded" in errors["disagg"]


def test_non_dsv4_vllm_019_error_remains_fail(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError("Unsupported moe quant mode 'w4a8_mxfp4_mxfp8'")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, _errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-30B-A3B",
        system="b200_sxm",
        backend="vllm",
        version="0.19.0",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FAIL, "disagg": STATUS_FAIL}


@pytest.mark.parametrize("backend,version", [("sglang", "0.5.10"), ("vllm", "0.14.0")])
def test_l40s_sm89_fp8_block_gemm_gap_is_hardware_incompatible(monkeypatch, backend, version):
    def fake_run_mode(**_kwargs):
        raise ValueError(
            f"Unsupported gemm quant mode 'fp8_block' for system='l40s', backend='{backend}', version='{version}'."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B-FP8",
        system="l40s",
        backend=backend,
        version=version,
        system_spec=_l40s_system_spec(),
    )

    assert statuses == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "Unsupported gemm quant mode 'fp8_block'" in errors["agg"]


def test_l40s_trtllm_fp8_block_moe_gap_is_hardware_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError("Unsupported moe quant mode 'fp8_block' for system='l40s', backend='trtllm', version='1.0.0'.")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-30B-A3B-FP8",
        system="l40s",
        backend="trtllm",
        version="1.0.0",
        system_spec=_l40s_system_spec(),
    )

    assert statuses == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "Unsupported moe quant mode 'fp8_block'" in errors["disagg"]


def test_l40s_fp8_block_other_backend_error_is_hardware_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError("Unsupported gemm quant mode 'fp8_block'")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B-FP8",
        system="l40s",
        backend="trtllm",
        version="1.0.0",
        system_spec=_l40s_system_spec(),
    )

    assert statuses == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "Unsupported gemm quant mode 'fp8_block'" in errors["agg"]


@pytest.mark.parametrize("model", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
def test_l40s_gpt_oss_mxfp4_moe_gap_is_hardware_incompatible(monkeypatch, model):
    def fake_run_mode(**_kwargs):
        raise ValueError(
            "Unsupported moe quant mode 'w4a16_mxfp4' for system='l40s', backend='sglang', version='0.5.10'."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model=model,
        system="l40s",
        backend="sglang",
        version="0.5.10",
        system_spec=_l40s_system_spec(),
    )

    assert statuses == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "Unsupported moe quant mode 'w4a16_mxfp4'" in errors["agg"]


def test_l40s_sglang_fp8_attention_gap_is_hardware_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError(
            "Unsupported context_attention quant mode 'fp8' for system='l40s', backend='sglang', version='0.5.10'."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-32B-FP8-Static-PerTensor",
        system="l40s",
        backend="sglang",
        version="0.5.10",
        system_spec=_l40s_system_spec(),
    )

    assert statuses == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "Unsupported context_attention quant mode 'fp8'" in errors["agg"]


def test_l40s_sglang_dsa_missing_data_gap_is_hardware_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "File does not exist at src/aiconfigurator/systems/data/l40s/sglang/0.5.10/dsa_context_module_perf.parquet"
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="zai-org/GLM-5",
        system="l40s",
        backend="sglang",
        version="0.5.10",
        system_spec=_l40s_system_spec(),
    )

    assert statuses == {"agg": STATUS_HW_INCOMPATIBLE, "disagg": STATUS_HW_INCOMPATIBLE}
    assert "does not support sparse DSA attention required by zai-org/GLM-5 on sglang" in errors["agg"]


def test_kimi_moonshot_trtllm_b200_int4_wo_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError(
            "Unsupported moe quant mode 'int4_wo' for system='b200_sxm', backend='trtllm', version='1.3.0rc10'."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="moonshotai/Kimi-K2.5",
        system="b200_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "Unsupported moe quant mode 'int4_wo'" in errors["agg"]


def test_kimi_moonshot_trtllm_int4_wo_other_system_remains_fail(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError("Unsupported moe quant mode 'int4_wo'")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, _errors = SupportMatrix.run_single_test(
        model="moonshotai/Kimi-K2.5",
        system="b300_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FAIL, "disagg": STATUS_FAIL}
