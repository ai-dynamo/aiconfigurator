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


def test_gb200_sglang_gemma4_h512_context_gap_reports_sglang_runtime_error(monkeypatch):
    error = (
        "Failed to query context attention data for b=1, s=128, prefix=128, n=2, n_kv=1, head_size=512, window_size=0."
    )

    def fake_run_mode(**_kwargs):
        raise RuntimeError(f"No results found for any parallel configuration. Showing last exception: {error}")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="google/gemma-4-26B-A4B",
        system="gb200",
        backend="sglang",
        version="0.5.10",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "SGLang 0.5.10 on GB200 rejects Gemma 4 context attention with head_size=512" in errors["agg"]
    assert "Missing TRTLLM-GEN kernel (context)" in errors["disagg"]


@pytest.mark.parametrize(
    ("backend", "version", "model", "error"),
    [
        (
            "trtllm",
            "1.3.0rc10",
            "Qwen/Qwen3-VL-2B-Instruct",
            "division by zero Failed to query context attention data for b=1, s=0, prefix=0, "
            "n=4, n_kv=4, head_size=64.",
        ),
        (
            "vllm",
            "0.19.0",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Failed to query gemm data for m=0, n=216, k=1152.",
        ),
    ],
)
def test_gb200_qwen3_vl_zero_token_planner_errors_are_not_framework_gaps(monkeypatch, backend, version, model, error):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(f"No results found for any parallel configuration. Showing last exception: {error}")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model=model,
        system="gb200",
        backend=backend,
        version=version,
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FAIL, "disagg": STATUS_FAIL}
    assert error.split(".")[0] in errors["agg"]


def test_gb200_qwen3_vl_moe_parallelism_assertion_is_not_framework_gap(monkeypatch):
    error = "tp_size (16) * attention_dp_size (1) should be equal to moe_tp_size (1) * moe_ep_size (1)"

    def fake_run_mode(**_kwargs):
        raise RuntimeError(f"No results found for any parallel configuration. Showing last exception: {error}")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-VL-235B-A22B-Instruct",
        system="gb200",
        backend="sglang",
        version="0.5.10",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FAIL, "disagg": STATUS_FAIL}
    assert error in errors["agg"]


def test_gb200_trtllm_gemma4_h512_context_gap_is_framework_incompatible(monkeypatch):
    error = (
        "Failed to query context attention data for b=1, s=128, prefix=128, n=2, n_kv=1, head_size=512, window_size=0."
    )

    def fake_run_mode(**_kwargs):
        raise RuntimeError(f"No results found for any parallel configuration. Showing last exception: {error}")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="google/gemma-4-26B-A4B",
        system="gb200",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "TRT-LLM 1.3.0rc10 on GB200 rejects Gemma 4 context attention with head_size=512" in errors["agg"]
    assert "Head size 512 is not supported by MMHA" in errors["disagg"]


def test_gb200_trtllm_mimo_h192_context_gap_reports_runtime_error(monkeypatch):
    error = (
        "Failed to query context attention data for b=1, s=128, prefix=128, n=64, n_kv=4, head_size=192, window_size=0."
    )

    def fake_run_mode(**_kwargs):
        raise RuntimeError(f"No results found for any parallel configuration. Showing last exception: {error}")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="XiaomiMiMo/MiMo-V2-Flash",
        system="gb200",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "TRT-LLM 1.3.0rc10 on GB200 fails required MiMo context attention with head_size=192" in errors["agg"]
    assert "illegal memory access" in errors["disagg"]


@pytest.mark.parametrize(
    ("model", "hidden_size", "inter_size", "expected"),
    [
        (
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            1024,
            2688,
            "FlashInfer FP4 static weight preparation",
        ),
        (
            "nvidia/nemotron-ultra-rl-050826",
            2048,
            5120,
            "Routing kernel expects topK experts <= 10, got 22",
        ),
    ],
)
def test_gb200_vllm_019_nemotron_nvfp4_gaps_report_runtime_errors(
    monkeypatch, model, hidden_size, inter_size, expected
):
    error = (
        "Failed to query moe data for num_tokens=128, "
        f"hidden_size={hidden_size}, inter_size={inter_size}, topk=22, num_experts=512, "
        "moe_tp_size=8, moe_ep_size=1, "
        "quant_mode=<MoEQuantMode.nvfp4: QuantMapping(memory=0.5625, compute=4, name='nvfp4')>, "
        "workload_distribution='power_law_1.01'."
    )

    def fake_run_mode(**_kwargs):
        raise RuntimeError(f"No results found for any parallel configuration. Showing last exception: {error}")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model=model,
        system="gb200",
        backend="vllm",
        version="0.19.0",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "vLLM 0.19.0 on GB200 rejects" in errors["agg"]
    assert expected in errors["disagg"]


@pytest.mark.parametrize(
    ("model", "hidden_size"),
    [
        ("sgl-project/DeepSeek-V4-Flash-FP8", 4096),
        ("sgl-project/DeepSeek-V4-Pro-FP8", 7168),
    ],
)
def test_gb200_vllm_019_dsv4_mhc_gap_reports_missing_vllm_module(monkeypatch, model, hidden_size):
    error = (
        "DeepSeek-V4 mHC module data not loaded for system='gb200', backend='vllm', version='0.19.0'. "
        f"Failed to query DeepSeek-V4 mHC module for num_tokens=128, hidden_size={hidden_size}, "
        "hc_mult=4, sinkhorn_iters=20, op='pre'."
    )

    def fake_run_mode(**_kwargs):
        raise RuntimeError(f"No results found for any parallel configuration. Showing last exception: {error}")

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model=model,
        system="gb200",
        backend="vllm",
        version="0.19.0",
        system_spec=_b200_system_spec(),
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "cannot run the DeepSeek-V4 mHC pre module" in errors["agg"]
    assert "No module named 'vllm.model_executor.layers.mhc'" in errors["disagg"]


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
