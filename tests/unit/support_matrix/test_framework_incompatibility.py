# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.support_matrix import support_matrix as support_matrix_module
from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_FRAMEWORK_INCOMPATIBLE,
    SupportMatrix,
    TestConstraints,
)

pytestmark = pytest.mark.unit


def _b200_system_spec() -> dict:
    return {"gpu": {"sm_version": 100, "fp8_tc_flops": 1, "fp4_tc_flops": 1}}


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


def test_run_mode_reports_missing_taskrunner_result(monkeypatch):
    monkeypatch.setattr(SupportMatrix, "_create_task_config", staticmethod(lambda **_kwargs: object()))
    monkeypatch.setattr(support_matrix_module.TaskRunner, "run", lambda _self, _task_config: None)

    with pytest.raises(RuntimeError, match="TaskRunner returned no result"):
        SupportMatrix._run_mode(
            mode="agg",
            model="Qwen/Qwen3-0.6B",
            system="b200_sxm",
            backend="sglang",
            version="0.5.12",
            constraints=TestConstraints(total_gpus=4, isl=256, osl=256, prefix=128, ttft=1500, tpot=50),
            engine_step_backend=None,
        )


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
