# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the new flat TaskConfig in sdk/task_config.py.

End-to-end sweep correctness is covered by the integration parity test
against the legacy CLI; these tests focus on construction, defaulting,
prefix discipline, and the build_* helpers.
"""

from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.task_config import TaskConfig

pytestmark = pytest.mark.unit


EXAMPLE_YAML = Path(__file__).parents[4] / "src" / "aiconfigurator" / "cli" / "example.yaml"


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
# from_yaml: legacy format compatibility
# ---------------------------------------------------------------------------


def _load_example(name: str) -> dict:
    with EXAMPLE_YAML.open() as f:
        return yaml.safe_load(f)[name]


def test_from_yaml_legacy_agg_full():
    t = TaskConfig.from_yaml(_load_example("exp_agg_full"))
    assert t.serving_mode == "agg"
    assert t.model_path == "deepseek-ai/DeepSeek-V3"
    assert t.system_name == "h200_sxm"
    assert t.backend_version == "1.2.0rc5"  # explicit in YAML
    # quant from nested worker_config
    assert t.gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert t.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16
    # search candidates from nested worker_config
    assert t.agg_num_gpu_candidates == [4, 8]
    assert t.agg_tp_candidates == [1, 2, 4, 8]
    # nextn from config block
    assert t.nextn == 1


def test_from_yaml_legacy_agg_simplified():
    t = TaskConfig.from_yaml(_load_example("exp_agg_simplified"))
    assert t.serving_mode == "agg"
    assert t.model_path == "deepseek-ai/DeepSeek-V3"
    assert t.system_name == "h200_sxm"
    # No explicit backend_version; should resolve to latest
    assert t.backend_version is not None


def test_from_yaml_legacy_disagg_full_mirrors_shared_fields_to_roles():
    t = TaskConfig.from_yaml(_load_example("exp_disagg_full"))
    assert t.serving_mode == "disagg"
    # Top-level model_path replicated to both roles
    assert t.prefill_model_path == "deepseek-ai/DeepSeek-V3"
    assert t.decode_model_path == "deepseek-ai/DeepSeek-V3"
    # Top-level system_name replicated; decode_system_name preserved
    assert t.prefill_system_name == "h200_sxm"
    assert t.decode_system_name == "h200_sxm"
    # Nested replica_config keys mapped
    assert t.max_gpu_per_replica == 128
    assert t.max_prefill_workers == 32
    assert t.max_decode_workers == 32
    # advanced_tuning_config
    assert t.prefill_latency_correction == 1.1
    assert t.decode_latency_correction == 1.08
    assert t.decode_max_batch_size == 512
    # Per-role quant from nested *_worker_config
    assert t.prefill_gemm_quant_mode == common.GEMMQuantMode.fp8_block
    assert t.decode_gemm_quant_mode == common.GEMMQuantMode.fp8_block


def test_from_yaml_legacy_disagg_simplified_quant_from_hf():
    t = TaskConfig.from_yaml(_load_example("exp_disagg_simplified"))
    assert t.serving_mode == "disagg"
    assert t.prefill_model_path == "nvidia/DeepSeek-V3.1-NVFP4"
    # nvidia/DeepSeek-V3.1-NVFP4 should resolve to nvfp4 quant
    assert t.prefill_gemm_quant_mode == common.GEMMQuantMode.nvfp4
    assert t.decode_gemm_quant_mode == common.GEMMQuantMode.nvfp4
    # nextn from config
    assert t.nextn == 2


def test_from_yaml_with_cli_overrides():
    t = TaskConfig.from_yaml(_load_example("exp_agg_simplified"), isl=8000, ttft=500.0)
    assert t.isl == 8000
    assert t.ttft == 500.0


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
