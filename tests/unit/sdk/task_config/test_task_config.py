# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the new flat TaskConfig in sdk/task_config.py.

End-to-end sweep correctness is covered by the integration parity test
against the legacy CLI; these tests focus on construction, defaulting,
prefix discipline, and the build_* helpers.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.task_config import TaskConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Construction defaults
# ---------------------------------------------------------------------------


def test_default_task_config_is_agg_with_default_workload():
    t = TaskConfig()
    assert t.serving_mode == "agg"
    assert t.isl == 4000
    assert t.osl == 1000
    assert t.ttft == 1000.0
    assert t.tpot == 50.0


def test_agg_with_model_resolves_identity_and_backend():
    t = TaskConfig(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        total_gpus=8,
    )
    assert t.is_moe is True
    assert t.model_family == "DEEPSEEK"
    assert t.nextn is not None
    assert t.backend_version is not None  # resolved to latest
    # Search space defaults populated
    assert t.agg_tp_candidates == [1, 2, 4, 8]
    assert t.agg_pp_candidates == [1]


def test_agg_resolves_quant_from_hf():
    t = TaskConfig(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    # DeepSeek-V3 is fp8_block from HF config
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8_block


def test_agg_quant_preset_overrides_hf():
    t = TaskConfig(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        quant_preset="bfloat16",
    )
    assert t.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16


def test_agg_explicit_quant_overrides_preset():
    t = TaskConfig(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        quant_preset="bfloat16",
        gemm_quant_mode=common.GEMMQuantMode.fp8,
    )
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8
    # other modes follow preset
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16


# ---------------------------------------------------------------------------
# Disagg construction
# ---------------------------------------------------------------------------


def test_disagg_with_separate_role_specs():
    t = TaskConfig(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
        total_gpus=32,
    )
    assert t.is_moe is True
    assert t.prefill_tp_candidates is not None
    assert t.decode_tp_candidates is not None
    assert t.num_gpu_per_replica is not None
    assert t.max_gpu_per_replica == 128
    assert t.max_prefill_workers == 32


def test_disagg_wideep_sets_larger_replica_budget():
    t = TaskConfig(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="gb200",
        prefill_enable_wideep=True,
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="gb200",
        decode_enable_wideep=True,
    )
    assert t.max_gpu_per_replica == 512
    assert t.num_gpu_per_replica is None  # wideep doesn't set a fixed list


# ---------------------------------------------------------------------------
# Strict prefix discipline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_wideep", True),
        ("enable_chunked_prefill", True),
        ("enable_eplb", True),
        ("quant_preset", "fp8"),
        ("gemm_quant_mode", common.GEMMQuantMode.fp8),
    ],
)
def test_disagg_rejects_top_level_worker_field_leakage(field, value):
    """Setting top-level worker fields in disagg mode must raise (no silent override)."""
    with pytest.raises(ValueError, match="top-level worker fields"):
        TaskConfig(
            serving_mode="disagg",
            prefill_model_path="x",
            prefill_system_name="h200_sxm",
            decode_model_path="x",
            decode_system_name="h200_sxm",
            **{field: value},
        )


# ---------------------------------------------------------------------------
# from_yaml: flat format (the new canonical YAML)
# ---------------------------------------------------------------------------


def test_from_yaml_flat_agg():
    yaml_data = {
        "serving_mode": "agg",
        "model_path": "deepseek-ai/DeepSeek-V3",
        "system_name": "h200_sxm",
        "backend_name": "trtllm",
        "backend_version": "1.2.0rc5",
        "total_gpus": 8,
        "isl": 4000,
        "osl": 1000,
        "ttft": 1000.0,
        "tpot": 40.0,
        "nextn": 1,
        "gemm_quant_mode": "fp8_block",
        "kvcache_quant_mode": "bfloat16",
        "agg_num_gpu_candidates": [4, 8],
        "agg_tp_candidates": [1, 2, 4, 8],
        "agg_pp_candidates": [1],
    }
    t = TaskConfig.from_yaml(yaml_data)
    assert t.serving_mode == "agg"
    assert t.model_path == "deepseek-ai/DeepSeek-V3"
    assert t.backend_version == "1.2.0rc5"
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16
    assert t.agg_num_gpu_candidates == [4, 8]
    assert t.agg_tp_candidates == [1, 2, 4, 8]
    assert t.nextn == 1


def test_from_yaml_flat_agg_minimal():
    """Just model_path + system_name; everything else defaults."""
    t = TaskConfig.from_yaml(
        {
            "serving_mode": "agg",
            "model_path": "deepseek-ai/DeepSeek-V3",
            "system_name": "h200_sxm",
        }
    )
    assert t.serving_mode == "agg"
    # Latest backend_version auto-resolved
    assert t.backend_version is not None
    # Quant inferred from HF config
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8_block


def test_from_yaml_flat_disagg():
    yaml_data = {
        "serving_mode": "disagg",
        "isl": 4000,
        "osl": 1000,
        "ttft": 1000.0,
        "tpot": 40.0,
        "total_gpus": 32,
        "prefill_model_path": "deepseek-ai/DeepSeek-V3",
        "prefill_system_name": "h200_sxm",
        "prefill_backend_name": "trtllm",
        "prefill_gemm_quant_mode": "fp8_block",
        "prefill_kvcache_quant_mode": "bfloat16",
        "decode_model_path": "deepseek-ai/DeepSeek-V3",
        "decode_system_name": "h200_sxm",
        "decode_backend_name": "trtllm",
        "decode_gemm_quant_mode": "fp8_block",
        "decode_kvcache_quant_mode": "bfloat16",
        "num_gpu_per_replica": [8, 16, 32, 64, 128],
        "max_gpu_per_replica": 128,
        "max_prefill_workers": 32,
        "max_decode_workers": 32,
        "prefill_latency_correction": 1.1,
        "decode_latency_correction": 1.08,
        "prefill_max_batch_size": 1,
        "decode_max_batch_size": 512,
    }
    t = TaskConfig.from_yaml(yaml_data)
    assert t.serving_mode == "disagg"
    assert t.prefill_model_path == "deepseek-ai/DeepSeek-V3"
    assert t.decode_model_path == "deepseek-ai/DeepSeek-V3"
    assert t.prefill_gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert t.decode_gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert t.max_gpu_per_replica == 128
    assert t.prefill_latency_correction == 1.1


def test_from_yaml_with_cli_overrides():
    t = TaskConfig.from_yaml(
        {
            "serving_mode": "agg",
            "model_path": "deepseek-ai/DeepSeek-V3",
            "system_name": "h200_sxm",
            "isl": 4000,
            "ttft": 1000.0,
        },
        isl=8000,
        ttft=500.0,
    )
    assert t.isl == 8000
    assert t.ttft == 500.0


def test_from_yaml_warns_on_unknown_keys(caplog):
    """Unknown keys are warned about but not silently swallowed."""
    import logging

    with caplog.at_level(logging.WARNING):
        TaskConfig.from_yaml(
            {
                "serving_mode": "agg",
                "model_path": "deepseek-ai/DeepSeek-V3",
                "system_name": "h200_sxm",
                "totally_made_up_field": 42,
                "another_typo": "value",
            }
        )
    assert "totally_made_up_field" in caplog.text
    assert "another_typo" in caplog.text


def test_from_yaml_disagg_rejects_legacy_shared_model_path():
    """Legacy YAML shape with top-level model_path is not silently mirrored to roles."""
    with pytest.raises(ValueError, match="top-level worker fields"):
        TaskConfig.from_yaml(
            {
                "serving_mode": "disagg",
                "model_path": "deepseek-ai/DeepSeek-V3",  # legacy shared form
                "system_name": "h200_sxm",
                "total_gpus": 32,
            }
        )


# ---------------------------------------------------------------------------
# Builders consumed by sweep.py
# ---------------------------------------------------------------------------


def test_build_runtime_config_carries_workload():
    t = TaskConfig(isl=2048, osl=512, ttft=300.0, tpot=20.0)
    rt = t.build_runtime_config(batch_size=64)
    assert rt.isl == 2048
    assert rt.osl == 512
    assert rt.ttft == 300.0
    assert rt.tpot == 20.0
    assert rt.batch_size == 64


def test_build_model_config_agg_uses_resolved_quant():
    t = TaskConfig(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
        quant_preset="bfloat16",
    )
    mc = t.build_model_config(role="agg")
    assert mc.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert mc.nextn == t.nextn


def test_sweep_agg_kwargs_shape():
    t = TaskConfig(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="h200_sxm",
    )
    kwargs = t.sweep_agg_kwargs(database="fake-db")
    assert kwargs["model_path"] == "deepseek-ai/DeepSeek-V3"
    assert kwargs["backend_name"] == "trtllm"
    assert kwargs["database"] == "fake-db"
    assert isinstance(kwargs["parallel_config_list"], list)
    assert len(kwargs["parallel_config_list"]) > 0


def test_sweep_disagg_kwargs_shape():
    t = TaskConfig(
        serving_mode="disagg",
        prefill_model_path="deepseek-ai/DeepSeek-V3",
        prefill_system_name="h200_sxm",
        decode_model_path="deepseek-ai/DeepSeek-V3",
        decode_system_name="h200_sxm",
    )
    kwargs = t.sweep_disagg_kwargs(prefill_database="p-db", decode_database="d-db")
    assert kwargs["model_path"] == "deepseek-ai/DeepSeek-V3"
    assert kwargs["prefill_database"] == "p-db"
    assert kwargs["decode_database"] == "d-db"
    assert kwargs["prefill_latency_correction"] == 1.1
    assert kwargs["decode_latency_correction"] == 1.08
    assert kwargs["decode_max_num_tokens"] == 512
    assert len(kwargs["prefill_num_worker_list"]) == 32
    assert len(kwargs["decode_num_worker_list"]) == 32


def test_sweep_kwargs_mode_mismatch_raises():
    t_agg = TaskConfig(serving_mode="agg", model_path="deepseek-ai/DeepSeek-V3", system_name="h200_sxm")
    with pytest.raises(ValueError, match="got 'agg'"):
        t_agg.sweep_disagg_kwargs(prefill_database=None, decode_database=None)
