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


def test_h100_sglang_dsv32_reduced_head_decode_gap_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. Showing last exception: "
            "Generation DSA module data not available for system='h100_sxm', backend='sglang', "
            "version='0.5.10', architecture='DeepseekV32ForCausalLM', "
            "kv_cache_dtype=KVCacheQuantMode.fp8, gemm_quant_mode=GEMMQuantMode.fp8_block, "
            "num_heads=16, s=257, b=2. Missing silicon data for the requested lookup."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="deepseek-ai/DeepSeek-V3.2",
        system="h100_sxm",
        backend="sglang",
        version="0.5.10",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "Unsupported h_q" in errors["agg"]
    assert "collector cannot produce valid rows" in errors["agg"]


def test_h100_vllm_dsv32_dsa_gap_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. Showing last exception: "
            "Context DSA module data not available for system='h100_sxm', backend='vllm', "
            "version='0.19.0', architecture='DeepseekV32ForCausalLM', "
            "fmha_quant_mode=FMHAQuantMode.bfloat16, kvcache_quant_mode=KVCacheQuantMode.fp8, "
            "gemm_quant_mode=GEMMQuantMode.fp8_block, num_heads=16, s=128, prefix=128, b=1. "
            "Missing silicon data for the requested lookup."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="deepseek-ai/DeepSeek-V3.2",
        system="h100_sxm",
        backend="vllm",
        version="0.19.0",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "vLLM 0.19.0 DeepSeek-V3.2 DSA fp8 path" in errors["agg"]
    assert "Traceback" not in errors["agg"]


def test_h100_vllm_mimo_fp8_kv_attention_gap_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. Showing last exception: "
            "Failed to query context attention data for b=1, s=128, prefix=128, n=8, n_kv=1, "
            "head_size=192, window_size=0, "
            "kvcache_quant_mode=<KVCacheQuantMode.fp8: QuantMapping(memory=1, compute=0, name='fp8')>, "
            "fmha_quant_mode=<FMHAQuantMode.bfloat16: QuantMapping(memory=2, compute=1, name='bfloat16')>. "
            "Missing silicon data for the requested lookup."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="XiaomiMiMo/MiMo-V2-Flash",
        system="h100_sxm",
        backend="vllm",
        version="0.19.0",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "query and key must have the same dtype" in errors["agg"]
    assert "Traceback" not in errors["agg"]


def test_h100_sglang_glm5_fp8_dsa_graph_gap_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. Showing last exception: "
            "Context DSA module data not available for system='h100_sxm', backend='sglang', "
            "version='0.5.10', architecture='GlmMoeDsaForCausalLM', "
            "kvcache_quant_mode=KVCacheQuantMode.fp8, gemm_quant_mode=GEMMQuantMode.fp8_block, "
            "num_heads=8, s=128, prefix=128, b=1. Missing silicon data for the requested lookup."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="zai-org/GLM-5-FP8",
        system="h100_sxm",
        backend="sglang",
        version="0.5.10",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "Unsupported h_q" in errors["agg"]
    assert "dsa_generation_module exposes fp8" in errors["agg"]


def test_h100_sglang_gemma4_head_dim_512_gap_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. Showing last exception: "
            "Failed to query context attention data for b=1, s=128, prefix=128, "
            "n=2, n_kv=1, head_size=512, window_size=0, kv_cache_dtype=KVCacheQuantMode.bfloat16."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="google/gemma-4-26B-A4B",
        system="h100_sxm",
        backend="sglang",
        version="0.5.10",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "head_dim <= 256" in errors["agg"]
    assert "head_dim=512" in errors["agg"]


def test_h100_trtllm_qwen3_vl_head_dim_72_gap_is_framework_incompatible(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. Showing last exception: "
            "Failed to query context attention data for b=1, s=784, prefix=0, "
            "n=2, n_kv=2, head_size=72, window_size=0. Missing silicon data for the requested lookup."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="Qwen/Qwen3-VL-32B-Instruct",
        system="h100_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "head_dim=72" in errors["agg"]
    assert "attention_context" in errors["agg"]


def test_h100_trtllm_dsa_fp8_gap_has_clean_reason(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError(
            "Unsupported dsa_generation_module quant mode 'fp8' for system='h100_sxm', "
            "backend='trtllm', version='1.3.0rc10'. Supported dsa_generation_module modes: ['bfloat16']"
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="zai-org/GLM-5-FP8",
        system="h100_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "sparse DSA generation rejects fp8 KV cache" in errors["agg"]
    assert "requires BF16 KV tensors" in errors["agg"]
    assert "Traceback" not in errors["agg"]


def test_h100_trtllm_dsv4_mhc_gap_has_clean_reason(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "DeepSeek-V4 mHC module data not loaded for system='h100_sxm', backend='trtllm', "
            "version='1.3.0rc10'. Failed to query DeepSeek-V4 mHC module for num_tokens=128.0."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="sgl-project/DeepSeek-V4-Pro-FP8",
        system="h100_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "DeepSeek-V4 mHC module data is not available" in errors["agg"]
    assert "Traceback" not in errors["agg"]


def test_h100_trtllm_moe_shape_gap_has_clean_reason(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise RuntimeError(
            "No results found for any parallel configuration. Showing last exception: "
            "Failed to query moe data for num_tokens=128.0, hidden_size=5120, inter_size=8192, "
            "topk=1, num_experts=128, moe_tp_size=8, moe_ep_size=1. Missing silicon data for the requested lookup."
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        system="h100_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "MoE data is missing" in errors["agg"]
    assert "Traceback" not in errors["agg"]


def test_h100_trtllm_kimi_int4_wo_gap_has_clean_reason(monkeypatch):
    def fake_run_mode(**_kwargs):
        raise ValueError(
            "Unsupported moe quant mode 'int4_wo' for system='h100_sxm', backend='trtllm', version='1.3.0rc10'. "
            "Supported moe modes: ['bfloat16', 'fp8', 'fp8_block', 'w4a16_mxfp4']"
        )

    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))
    _patch_large_constraints(monkeypatch)

    statuses, errors = SupportMatrix.run_single_test(
        model="moonshotai/Kimi-K2.5",
        system="h100_sxm",
        backend="trtllm",
        version="1.3.0rc10",
        system_spec={"gpu": {"sm_version": 90, "fp8_tc_flops": 1}},
    )

    assert statuses == {"agg": STATUS_FRAMEWORK_INCOMPATIBLE, "disagg": STATUS_FRAMEWORK_INCOMPATIBLE}
    assert "does not expose the int4_wo MoE path" in errors["agg"]
    assert "Traceback" not in errors["agg"]


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
