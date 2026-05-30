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


@pytest.mark.parametrize(
    ("model", "error"),
    [
        (
            "deepseek-ai/DeepSeek-V4-Flash",
            "DeepSeek-V4 mHC module data not loaded for system='gb200', backend='sglang', version='0.5.10'.",
        ),
        (
            "deepseek-ai/DeepSeek-V4-Pro",
            "No DeepSeek-V4 context attention silicon data for native_heads=128, loaded keys=[64]. "
            "Failed to query DeepSeek-V4 context attention module for b=1, s=128, prefix=128, "
            "num_heads=16, native_heads=128, tp_size=8, compress_ratio=128.",
        ),
        (
            "XiaomiMiMo/MiMo-V2-Flash",
            "Failed to query context attention data for b=1, s=128, prefix=128, n=8, n_kv=1, "
            "head_size=192, window_size=0.",
        ),
        (
            "google/gemma-4-26B-A4B",
            "Failed to query context attention data for b=1, s=128, prefix=128, n=2, n_kv=1, "
            "head_size=512, window_size=0.",
        ),
        (
            "openai/gpt-oss-120b",
            "Failed to query moe data for num_tokens=128, hidden_size=2880, inter_size=2880, topk=4, "
            "num_experts=128, moe_tp_size=8, moe_ep_size=1, "
            "quant_mode=<MoEQuantMode.int4_wo: QuantMapping(memory=0.5, compute=1, name='int4_wo')>.",
        ),
    ],
)
def test_gb200_sglang_0510_known_runtime_gaps_are_framework_incompatible(monkeypatch, model, error):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(f"No results found for any parallel configuration. Showing last exception: {error}")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model=model,
        system="gb200",
        backend="sglang",
        version="0.5.10",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert error.split(" for ")[0] in errors["agg"]


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
