# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import csv
import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest

from collector.case_generator import (
    get_attention_head_configs,
    get_moe_quantization_specs,
    moe_model_allows_quantization,
)
from collector.model_cases import (
    BASE_OP_CASES_DIR,
    build_collection_case_plan,
    default_architecture_cases_path,
    load_yaml_file,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SUPPORT_MATRIX_ROOT = REPO_ROOT / "src" / "aiconfigurator" / "systems" / "support_matrix"


def _load_mla_adapter(module_path: str, globals_dict: dict):
    source_path = REPO_ROOT / module_path
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_mla_test_cases"
    )
    namespace = dict(globals_dict)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_build_mla_test_cases"]


def test_model_case_plan_merges_required_base_and_framework_specific_ops():
    plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")

    assert plan.model_architecture == "DeepseekV3ForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("DeepseekV3ForCausalLM")]
    assert plan.has_op("gemm")
    assert not plan.has_op("attention_context")
    assert not plan.has_op("attention_generation")
    assert "moe" in plan.selected_ops
    assert "mla_context" in plan.selected_ops
    assert "wideep_mla_context" in plan.selected_ops
    assert "wideep_moe" in plan.selected_ops
    assert "trtllm_moe_wideep" not in plan.selected_ops


def test_attention_head_configs_preserve_real_model_structures_without_cross_mixing():
    from collector.case_generator import get_attention_context_shape_sweeps, get_attention_generation_shape_sweeps

    expected_model_structures = {
        # Gemma 4 local/global attention.
        (16, 8, 256, 1024),
        (16, 2, 512, 0),
        # Llama 4 local attention.
        (40, 8, 128, 8192),
        # MiMo-V2 global/local attention.
        (64, 4, 192, 0),
        (64, 8, 192, 128),
    }
    impossible_cross_model_mixes = {
        (64, 8, 256, 1024),
        (40, 8, 192, 8192),
        (16, 8, 512, 1024),
        (64, 4, 64, 128),
    }

    for phase, get_shape_sweeps in (
        ("context", get_attention_context_shape_sweeps),
        ("generation", get_attention_generation_shape_sweeps),
    ):
        configs = {
            (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
            for sweep in get_shape_sweeps("sglang")
            for config in get_attention_head_configs(sweep, phase=phase)
        }

        assert expected_model_structures <= configs
        assert configs.isdisjoint(impossible_cross_model_mixes)


def test_native_attention_profiles_drop_non_integral_local_gqa(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "example/unregistered-model")
    shape_sweep = {
        "num_attention_heads": 6,
        "num_key_value_heads": 3,
        "head_dim": 128,
        "window_size": 0,
        "tensor_parallel_sizes": [1, 2],
    }

    assert [
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for config in get_attention_head_configs(shape_sweep, phase="generation")
    ] == [
        (6, 3, 128, 0),
    ]


def test_targeted_attention_profile_uses_model_topology(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-32B-FP8")
    configs = {
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for sweep in get_attention_context_shape_sweeps("sglang")
        for config in get_attention_head_configs(sweep, phase="context")
    }

    assert configs == {
        (64, 8, 128, 0),
        (32, 4, 128, 0),
        (16, 2, 128, 0),
        (8, 1, 128, 0),
        (4, 1, 128, 0),
        (2, 1, 128, 0),
        (1, 1, 128, 0),
    }


def test_retired_kimi_generic_attention_profile_was_redundant_for_legacy_full_grids(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps, get_attention_generation_shape_sweeps

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    retired_kimi_configs = {
        (64, 64, 128, 0),
        (32, 32, 128, 0),
        (16, 16, 128, 0),
        (8, 8, 128, 0),
        (4, 4, 128, 0),
        (2, 2, 128, 0),
        (1, 1, 128, 0),
    }

    for backend in ("sglang", "trtllm"):
        for phase, get_shape_sweeps in (
            ("context", get_attention_context_shape_sweeps),
            ("generation", get_attention_generation_shape_sweeps),
        ):
            configs = {
                (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
                for sweep in get_shape_sweeps(backend)
                for config in get_attention_head_configs(sweep, phase=phase)
            }
            assert retired_kimi_configs <= configs


def test_added_model_attention_profiles_resolve_targeted_topology(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps

    profiles = (
        (("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"), 8, 2, 256, (1, 2, 4, 8)),
        (("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"), 16, 4, 256, (1, 2, 4, 8, 16)),
        (("Qwen/Qwen3.5-122B-A10B",), 32, 2, 256, (1, 2, 4, 8, 16, 32)),
        (("MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.7"), 48, 8, 128, (1, 2, 4, 8, 16)),
        (("Qwen/Qwen3-30B-A3B",), 32, 4, 128, (1, 2, 4, 8)),
    )

    for model_paths, num_heads, num_kv_heads, head_dim, tp_sizes in profiles:
        expected = {
            (num_heads // tp, (num_kv_heads + tp - 1) // tp, head_dim, 0)
            for tp in tp_sizes
            if (num_heads // tp) % ((num_kv_heads + tp - 1) // tp) == 0
        }
        for model_path in model_paths:
            monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
            configs = {
                (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
                for sweep in get_attention_context_shape_sweeps("sglang")
                for config in get_attention_head_configs(sweep, phase="context")
            }
            assert configs == expected, model_path


def test_added_model_moe_profiles_resolve_targeted_aliases(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    expected_by_model = {
        "Qwen/Qwen3.5-122B-A10B": ("Qwen/Qwen3.5-122B-A10B", 3072, 1024, 8, 256),
        "Qwen/Qwen3-235B-A22B-Instruct-2507": ("Qwen/Qwen3-235B-A22B", 4096, 1536, 8, 128),
        "MiniMaxAI/MiniMax-M2": ("MiniMaxAI/MiniMax-M2.5", 3072, 1536, 8, 256),
    }
    for model_path, expected in expected_by_model.items():
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        cases = get_common_moe_test_cases()
        assert {
            (case.model_name, case.hidden_size, case.inter_size, case.topk, case.num_experts) for case in cases
        } == {expected}


def test_mimo_attention_profile_matches_aic_full_attention_window(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "XiaomiMiMo/MiMo-7B-Base")
    configs = {
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for sweep in get_attention_context_shape_sweeps("vllm")
        for config in get_attention_head_configs(sweep, phase="context")
    }

    assert configs == {
        (32, 8, 128, 0),
        (16, 4, 128, 0),
        (8, 2, 128, 0),
        (4, 1, 128, 0),
        (2, 1, 128, 0),
        (1, 1, 128, 0),
    }


def test_qwen_vl_attention_profiles_stop_at_sdk_valid_tp16(monkeypatch):
    from collector.case_generator import get_attention_context_shape_sweeps

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-VL-32B-Instruct")
    configs = {
        (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
        for sweep in get_attention_context_shape_sweeps("vllm")
        for config in get_attention_head_configs(sweep, phase="context")
    }

    assert configs == {
        (64, 8, 128, 0),
        (32, 4, 128, 0),
        (16, 2, 128, 0),
        (8, 1, 128, 0),
        (4, 1, 128, 0),
    }


def test_full_encoder_attention_profiles_combine_defaults_and_model_deltas(monkeypatch):
    from collector.case_generator import get_attention_encoder_head_configs, get_attention_encoder_shape_sweeps

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    for backend in ("sglang", "trtllm", "vllm"):
        sweeps = get_attention_encoder_shape_sweeps(backend)
        keys = {
            (config.num_heads, config.head_dim)
            for sweep in sweeps
            for config in get_attention_encoder_head_configs(sweep)
        }
        default_keys = {
            (num_heads, head_dim)
            for sweep in sweeps
            for head_dim in sweep["head_dims"]
            for num_heads in sweep["head_counts"]
        }

        assert default_keys <= keys
        assert keys - default_keys == {(1, 64), (1, 72)}


def test_targeted_encoder_attention_profile_is_model_exact(monkeypatch):
    from collector.case_generator import get_attention_encoder_head_configs, get_attention_encoder_shape_sweeps

    for model_path, head_dim in (
        ("Qwen/Qwen3-VL-4B-Instruct", 64),
        ("Qwen/Qwen3-VL-32B-Instruct", 72),
        ("Qwen/Qwen3-VL-235B-A22B-Instruct", 72),
        ("moonshotai/Kimi-K2.5", 72),
        ("nvidia/Kimi-K2.5-NVFP4", 72),
    ):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        configs = [
            config
            for sweep in get_attention_encoder_shape_sweeps("vllm")
            for config in get_attention_encoder_head_configs(sweep)
        ]

        assert {(config.num_heads, config.head_dim) for config in configs} == {
            (16, head_dim),
            (8, head_dim),
            (4, head_dim),
            (2, head_dim),
            (1, head_dim),
        }


def test_base_gemm_cases_are_readable_shape_specs():
    plan = build_collection_case_plan(backend="sglang", model_path="Qwen/Qwen3-32B")
    assert plan.has_op("gemm")
    assert plan.has_op("attention_context")
    assert plan.has_op("attention_generation")

    def base_specs(op_name):
        data = load_yaml_file(BASE_OP_CASES_DIR / f"{op_name}.yaml")
        return data["all_frameworks_op_cases"][op_name]["cases"]

    gemm_specs = base_specs("gemm")
    assert len(gemm_specs) == 1
    spec = gemm_specs[0]
    assert spec["id"] == "base_transformer_gemm_shape_sweep"
    assert spec["token_counts"][:5] == [1, 2, 3, 4, 5]
    assert spec["feature_sizes"][:3] == [32, 64, 128]

    context_spec = base_specs("attention_context")[0]
    assert context_spec["id"] == "base_attention_context_shape_sweep"
    assert context_spec["kv_head_options"] == ["self", 1, 2, 4, 8]
    generation_spec = base_specs("attention_generation")[0]
    assert generation_spec["id"] == "base_attention_generation_shape_sweep"
    assert generation_spec["xqa_query_head_counts"][-1] == 128


def test_moe_model_quantization_policy_is_yaml_backed():
    assert not moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", "w4a8_mxfp4_mxfp8")
    assert not moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", "bfloat16")
    assert not moe_model_allows_quantization("sglang", "Qwen/Qwen3-235B-A22B", "w4a8_mxfp4_mxfp8")

    assert moe_model_allows_quantization("sglang", "openai/gpt-oss-120b", "w4a16_mxfp4")
    assert moe_model_allows_quantization("sglang", "openai/gpt-oss-120b", "w4a8_mxfp4_mxfp8")
    assert not moe_model_allows_quantization("sglang", "openai/gpt-oss-120b", "bfloat16")

    assert moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K2.5", "int4_wo")
    assert not moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K2.5", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K2.5", "bfloat16")
    assert not moe_model_allows_quantization("trtllm", "Qwen/Qwen3-235B-A22B", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("trtllm", "openai/gpt-oss-20b", "fp8")


def test_dsv4_moe_quantization_policy_prunes_unrelated_modes():
    expected_by_backend = {
        "sglang": {
            "deepseek-ai/DeepSeek-V4-Flash": set(),
            "deepseek-ai/DeepSeek-V4-Pro": set(),
            "sgl-project/DeepSeek-V4-Flash-FP8": {"fp8_block"},
            "sgl-project/DeepSeek-V4-Pro-FP8": {"fp8_block"},
        },
        "trtllm": {
            "deepseek-ai/DeepSeek-V4-Flash": {"w4a8_mxfp4_mxfp8"},
            "deepseek-ai/DeepSeek-V4-Pro": {"w4a8_mxfp4_mxfp8"},
            "sgl-project/DeepSeek-V4-Flash-FP8": {"fp8_block"},
            "sgl-project/DeepSeek-V4-Pro-FP8": {"fp8_block"},
        },
        "vllm": {
            "deepseek-ai/DeepSeek-V4-Flash": set(),
            "deepseek-ai/DeepSeek-V4-Pro": set(),
            "sgl-project/DeepSeek-V4-Flash-FP8": {"fp8_block"},
            "sgl-project/DeepSeek-V4-Pro-FP8": {"fp8_block"},
        },
    }

    for backend, expected_by_artifact in expected_by_backend.items():
        available_modes = {spec.name for spec in get_moe_quantization_specs(backend)}
        for model_path, expected in expected_by_artifact.items():
            allowed = {mode for mode in available_modes if moe_model_allows_quantization(backend, model_path, mode)}
            assert allowed == expected, (backend, model_path)


def test_kimi_moe_quantization_is_artifact_specific():
    expected_by_artifact = {
        "moonshotai/Kimi-K2-Instruct": {"fp8_block"},
        "moonshotai/Kimi-K2.5": {"int4_wo"},
        "nvidia/Kimi-K2.5-NVFP4": {"nvfp4"},
    }

    for backend in ("sglang", "trtllm", "vllm"):
        available_modes = {spec.name for spec in get_moe_quantization_specs(backend)}
        for model_path, expected in expected_by_artifact.items():
            allowed = {mode for mode in available_modes if moe_model_allows_quantization(backend, model_path, mode)}
            assert allowed == expected, (backend, model_path)


def test_deepseek_minimax_and_nemotron_moe_quantization_is_artifact_specific():
    shared_expected = {
        "deepseek-ai/DeepSeek-V3": {"fp8_block"},
        "deepseek-ai/DeepSeek-R1": {"fp8_block"},
        "deepseek-ai/DeepSeek-V3.2": {"fp8_block"},
        "nvidia/DeepSeek-V3.1-NVFP4": {"nvfp4"},
        "MiniMaxAI/MiniMax-M2": {"fp8_block"},
        "MiniMaxAI/MiniMax-M2.5": {"fp8_block"},
        "MiniMaxAI/MiniMax-M2.7": {"fp8_block"},
        "nvidia/MiniMax-M2.5-NVFP4": {"nvfp4"},
        "nvidia/MiniMax-M2.7-NVFP4": {"nvfp4"},
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {"bfloat16"},
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": {"nvfp4"},
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": {"bfloat16"},
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": {"nvfp4"},
    }

    for backend in ("sglang", "trtllm", "vllm"):
        available_modes = {spec.name for spec in get_moe_quantization_specs(backend)}
        expected_by_artifact = {
            **shared_expected,
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8": set() if backend == "sglang" else {"fp8"},
        }
        if backend == "vllm":
            expected_by_artifact["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"] = set()
            expected_by_artifact["nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"] = set()
        for model_path, expected in expected_by_artifact.items():
            allowed = {mode for mode in available_modes if moe_model_allows_quantization(backend, model_path, mode)}
            assert allowed == expected, (backend, model_path)


def test_gptoss_mxfp4_modes_are_additive_on_blackwell():
    from collector.case_generator import get_moe_quantization_modes

    def selected_modes(backend, sm_version):
        return {
            mode
            for mode in get_moe_quantization_modes(backend, sm_version=sm_version)
            if moe_model_allows_quantization(backend, "openai/gpt-oss-120b", mode)
        }

    assert selected_modes("sglang", 100) == {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}
    assert selected_modes("trtllm", 90) == {"w4a16_mxfp4"}
    assert selected_modes("trtllm", 100) == {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}


def test_sglang_mxfp4_quant_labels_select_explicit_activation_precision():
    source_path = REPO_ROOT / "collector/sglang/collect_moe.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_mxfp4_activation_precision"
    )
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), "exec"), namespace)

    assert namespace["_mxfp4_activation_precision"]("w4a16_mxfp4") == "bf16"
    assert namespace["_mxfp4_activation_precision"]("w4a8_mxfp4_mxfp8") == "default"


def test_attention_shape_specs_are_yaml_backed_with_backend_overrides():
    from collector.case_generator import get_attention_context_shape_sweeps, get_attention_generation_shape_sweeps

    def by_id(sweeps, sweep_id):
        return next(sweep for sweep in sweeps if sweep["id"] == sweep_id)

    context_id = "base_attention_context_shape_sweep"
    generation_id = "base_attention_generation_shape_sweep"
    sglang_context = by_id(get_attention_context_shape_sweeps("sglang"), context_id)
    trtllm_context = by_id(get_attention_context_shape_sweeps("trtllm"), context_id)
    vllm_context = by_id(get_attention_context_shape_sweeps("vllm"), context_id)
    vllm_xpu_context = by_id(get_attention_context_shape_sweeps("vllm_xpu"), context_id)
    vllm_generation = by_id(get_attention_generation_shape_sweeps("vllm"), generation_id)

    assert sglang_context["head_dims"] == [64, 128, 192, 256]
    assert trtllm_context["head_dims"] == [64, 128, 192, 256]
    assert trtllm_context["query_head_counts"][:6] == [1, 2, 3, 4, 5, 6]
    assert vllm_context["head_dims"] == [64, 128, 192, 256]
    assert vllm_context["query_head_counts"][-1] == 64
    assert trtllm_context["window_sizes"] == [0, 128, 1024]
    assert vllm_context["window_sizes"] == [0, 128, 1024, 8192]
    assert vllm_xpu_context["batch_sizes"] == [1, 2, 4, 8, 16, 32]
    assert vllm_xpu_context["kv_head_options"] == [1, 2, 4, 8]
    assert vllm_generation["mha_query_head_counts"][-1] == 64
    assert vllm_generation["xqa_query_head_counts"][-1] == 64


def test_gemm_common_cases_expand_from_base_op_yaml_shape_specs():
    from collector.case_generator import (
        ComputeScaleCommonTestCase,
        GemmCommonTestCase,
        get_compute_scale_case_specs,
        get_gemm_case_specs,
        get_gemm_type_specs,
    )

    cases = get_gemm_case_specs()
    xpu_cases = get_gemm_case_specs("vllm_xpu")

    assert len(cases) == 35742
    assert cases[0] == GemmCommonTestCase(x=32768, n=65536, k=51200)
    assert cases[-1] == GemmCommonTestCase(x=1, n=32, k=32)
    assert not any(case.n == 65536 and case.k == 65536 for case in cases)

    assert len(xpu_cases) == 7581
    assert xpu_cases[0] == GemmCommonTestCase(x=8192, n=12288, k=12288)
    assert xpu_cases[-1] == GemmCommonTestCase(x=1, n=32, k=32)
    assert get_gemm_type_specs("vllm_xpu") == ["bfloat16", "fp8"]

    compute_scale_cases = get_compute_scale_case_specs()
    assert len(compute_scale_cases) == 1628
    assert compute_scale_cases[0] == ComputeScaleCommonTestCase(m=32768, k=51200)
    assert compute_scale_cases[-1] == ComputeScaleCommonTestCase(m=1, k=65536)


def test_cross_model_common_cases_expand_from_base_op_yaml_sweeps(monkeypatch):
    from collector.case_generator import (
        get_common_gdn_test_cases,
        get_common_mamba2_test_cases,
        get_common_mhc_test_cases,
        get_common_moe_test_cases,
        get_context_mla_case_specs,
        get_generation_mla_case_specs,
    )

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    moe_cases = get_common_moe_test_cases()
    # +117 vs pre-GLM-5.2: nvidia/GLM-5.2-NVFP4 registered in moe.model_paths
    # (same MoE dims as GLM-5-NVFP4). The sglang collector's get_moe_test_cases
    # dedups GLM-5-NVFP4 vs GLM-5.2-NVFP4; this backend-agnostic common layer
    # keeps both.
    assert len(moe_cases) == 4326
    assert any(
        case.model_name == "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
        and case.hidden_size == 1024
        and case.inter_size == 2688
        for case in moe_cases
    )
    assert any(
        case.model_name == "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
        and case.hidden_size == 2048
        and case.inter_size == 5120
        for case in moe_cases
    )
    assert len(get_context_mla_case_specs()) == 220
    assert len(get_generation_mla_case_specs()) == 362
    mamba_cases = get_common_mamba2_test_cases()
    assert len(mamba_cases) == 12
    assert {case.model_name for case in mamba_cases} >= {"MAMBA2_GENERIC_4K", "MAMBA2_GENERIC_1K"}
    assert len(get_common_gdn_test_cases()) == 16
    mhc_cases = get_common_mhc_test_cases()
    assert len(mhc_cases) == 8
    assert {(case.model_name, case.phase, case.hidden_size, case.hc_mult) for case in mhc_cases} == {
        (model_name, phase, hidden_size, 4)
        for model_name, hidden_size in (
            ("deepseek-ai/DeepSeek-V4-Flash", 4096),
            ("sgl-project/DeepSeek-V4-Flash-FP8", 4096),
            ("deepseek-ai/DeepSeek-V4-Pro", 7168),
            ("sgl-project/DeepSeek-V4-Pro-FP8", 7168),
        )
        for phase in ("pre", "post")
    }
    assert {(case.phase, case.hidden_size, case.hc_mult) for case in mhc_cases} == {
        (phase, hidden_size, 4) for hidden_size in (4096, 7168) for phase in ("pre", "post")
    }


def test_mla_collectors_dedupe_on_loader_physical_keys(monkeypatch):
    from types import SimpleNamespace

    from collector.case_generator import get_context_mla_case_specs, get_generation_mla_case_specs

    sglang_adapter = _load_mla_adapter(
        "collector/sglang/collect_mla.py",
        {
            "KV_LORA_RANK": 512,
            "QK_NOPE_HEAD_DIM": 128,
            "QK_ROPE_HEAD_DIM": 64,
            "MLA_PAGE_SIZE": 64,
            "MAX_KV_LOC": ((2**31 - 1) // (512 + 64)) - 64,
        },
    )
    trtllm_adapter = _load_mla_adapter(
        "collector/trtllm/collect_mla.py",
        {
            "Scenario": lambda: SimpleNamespace(
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
            "_mla_tokens_per_block": lambda: 32,
        },
    )

    def physical_keys(cases):
        return {(case[3], case[4] // case[6], case[1], case[0]) for case in cases}

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    full_specs = (get_context_mla_case_specs(), get_generation_mla_case_specs())
    for adapter, kwargs in (
        (sglang_adapter, {"tp_sizes": (1, 2, 4, 8, 16, 32, 64)}),
        (trtllm_adapter, {}),
    ):
        context_cases = adapter(full_specs[0], dtype_list=("bf16", "fp8"), **kwargs)
        generation_cases = adapter(full_specs[1], dtype_list=("bf16", "fp8"), **kwargs)
        assert len(context_cases) == len(physical_keys(context_cases))
        assert len(generation_cases) == len(physical_keys(generation_cases))
        assert 1 in {case[4] // case[6] for case in context_cases}
        assert 1 in {case[4] // case[6] for case in generation_cases}


def test_kimi_mla_plan_includes_generation_bmm_helpers():
    required_ops = {"mla_context", "mla_generation", "mla_bmm_gen_pre", "mla_bmm_gen_post"}
    for backend in ("sglang", "trtllm"):
        plan = build_collection_case_plan(backend=backend, model_path="moonshotai/Kimi-K2.5")
        assert required_ops <= plan.selected_ops


def test_dsa_module_prefix_context_sweeps_are_yaml_backed():
    from collector.case_generator import get_mla_module_sweep_spec

    assert 128 in get_mla_module_sweep_spec("sglang").context_prefix_lengths
    assert get_mla_module_sweep_spec("trtllm").context_prefix_lengths == [0, 128]
    assert get_mla_module_sweep_spec("vllm").context_prefix_lengths == [0, 128]


def test_vllm_moe_quantization_metadata_is_yaml_backed():
    from collector.case_generator import (
        get_moe_quantization_modes,
        get_moe_quantization_module_config,
        moe_model_allows_quantization,
    )

    assert get_moe_quantization_modes("vllm", sm_version=90, runtime_features={"per_block_fp8": True}) == [
        "bfloat16",
        "int4_wo",
        "fp8",
        "fp8_block",
    ]
    assert get_moe_quantization_modes(
        "vllm",
        sm_version=100,
        runtime_features={"per_block_fp8": True, "nvfp4": True, "mxfp4": True},
    ) == ["bfloat16", "int4_wo", "fp8", "fp8_block", "nvfp4", "w4a16_mxfp4"]
    assert get_moe_quantization_modes(
        "vllm",
        sm_version=120,
        runtime_version="0.24.0",
        runtime_features={"per_block_fp8": True, "nvfp4": True, "mxfp4": True},
    ) == ["bfloat16", "int4_wo", "fp8", "fp8_block", "nvfp4", "w4a16_mxfp4"]

    assert moe_model_allows_quantization("vllm", "openai/gpt-oss-20b", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("vllm", "openai/gpt-oss-20b", "bfloat16")
    assert not moe_model_allows_quantization("vllm", "Qwen/Qwen3-235B-A22B", "w4a16_mxfp4")
    assert moe_model_allows_quantization("vllm", "Qwen/Qwen3-235B-A22B", "bfloat16")
    assert get_moe_quantization_module_config("vllm", "w4a16_mxfp4", model_name="openai/gpt-oss-20b") == {
        "has_bias": True,
        "activation": "swigluoai",
    }
    assert get_moe_quantization_module_config("vllm", "w4a16_mxfp4", model_name="Qwen/Qwen3-235B-A22B") == {}
    assert get_moe_quantization_module_config("vllm", "int4_wo", model_name="moonshotai/Kimi-K2.5") == {
        "group_size": 32
    }


def test_vllm_xpu_moe_metadata_is_yaml_backed(monkeypatch):
    from collector.case_generator import (
        get_moe_backend_model_activation,
        get_moe_backend_test_cases,
        get_moe_quantization_modes,
        moe_model_allows_quantization,
    )

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    cases = get_moe_backend_test_cases("vllm_xpu")

    assert len(cases) == 327
    assert {case.model_name for case in cases} == {
        "Qwen/Qwen1.5-MoE-A2.7B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "meta-llama/Llama-4-Scout-17B-16E",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }
    assert not any(case.model_name == "Qwen/Qwen3-30B-A3B" and case.tp >= 8 for case in cases)
    assert get_moe_backend_model_activation("vllm_xpu", "openai/gpt-oss-20b") == "swigluoai"
    assert get_moe_backend_model_activation("vllm_xpu", "Qwen/Qwen1.5-MoE-A2.7B") == "silu"

    assert get_moe_quantization_modes("vllm_xpu", sm_version=0, runtime_features={}) == [
        "bfloat16",
        "w4a16_mxfp4",
    ]
    assert get_moe_quantization_modes(
        "vllm_xpu",
        sm_version=0,
        runtime_features={"torch_fp8_e4m3fn": True},
    ) == ["bfloat16", "fp8", "w4a16_mxfp4"]
    assert moe_model_allows_quantization("vllm_xpu", "openai/gpt-oss-20b", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("vllm_xpu", "openai/gpt-oss-20b", "bfloat16")
    assert not moe_model_allows_quantization("vllm_xpu", "Qwen/Qwen1.5-MoE-A2.7B", "w4a16_mxfp4")

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "openai/gpt-oss-20b")
    targeted_cases = get_moe_backend_test_cases("vllm_xpu")
    assert targeted_cases
    assert {case.model_name for case in targeted_cases} == {"openai/gpt-oss-20b"}


def test_mla_bmm_cases_expand_from_base_op_yaml():
    from collector.case_generator import MLABMMCommonTestCase, get_mla_bmm_case_specs

    pre_cases = get_mla_bmm_case_specs("sglang", "mla_bmm_gen_pre")
    post_cases = get_mla_bmm_case_specs("sglang", "mla_bmm_gen_post")

    assert len(pre_cases) == 400
    assert len(post_cases) == 448
    assert pre_cases[0] == MLABMMCommonTestCase(
        num_tokens=1,
        num_heads=128,
        dtype="bfloat16",
        num_warmups=2,
        num_runs=10,
    )
    assert pre_cases[1] == MLABMMCommonTestCase(
        num_tokens=1,
        num_heads=128,
        dtype="fp8",
        num_warmups=2,
        num_runs=10,
    )
    assert post_cases[-1] == MLABMMCommonTestCase(
        num_tokens=20480,
        num_heads=1,
        dtype="fp8",
        num_warmups=2,
        num_runs=10,
    )


def test_mla_module_metadata_and_micro_sweeps_are_yaml_backed():
    from collector.case_generator import (
        get_mla_module_model_specs,
        get_mla_module_precision_specs,
        get_mla_module_sweep_spec,
    )

    sweep = get_mla_module_sweep_spec()
    dsa_specs = get_mla_module_model_specs(attention_type="dsa", apply_model_filter=False)
    kimi_specs = get_mla_module_model_specs(
        attention_type="mla",
        backend="vllm",
        wideep_mla=False,
        apply_model_filter=False,
    )
    wideep_specs = get_mla_module_model_specs(attention_type="mla", wideep_mla=True, apply_model_filter=False)
    vllm_specs = get_mla_module_model_specs(backend="vllm")

    assert sweep.batch_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    assert sweep.sequence_lengths[-2:] == [8192, 16384]
    assert sweep.inner_sweep_head_counts == [128, 64, 32, 16, 8]
    assert sweep.top_level_head_counts == [128, 64, 32, 16, 8]
    assert sweep.module_precision_combos == [("bfloat16", "bfloat16", "bfloat16")]

    trtllm_sweep = get_mla_module_sweep_spec("trtllm")
    assert trtllm_sweep.context_batch_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert trtllm_sweep.context_sequence_lengths[-1] == 32768
    assert trtllm_sweep.generation_sequence_lengths[-1] == 131072
    assert trtllm_sweep.inner_sweep_head_counts == [128, 64, 32, 16, 8, 4, 2, 1]
    assert trtllm_sweep.generation_max_tokens == 33554432

    assert [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs("sglang", phase="context", sm_version=90)
    ] == [
        ("bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16", "fp8", "bfloat16"),
        ("bfloat16", "bfloat16", "fp8_block"),
        ("bfloat16", "fp8", "fp8_block"),
    ]
    assert [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs("vllm", phase="context", sm_version=100)
    ] == [
        ("bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16", "fp8", "bfloat16"),
        ("fp8", "fp8", "bfloat16"),
        ("bfloat16", "bfloat16", "fp8_block"),
        ("bfloat16", "fp8", "fp8_block"),
        ("fp8", "fp8", "fp8_block"),
        ("bfloat16", "bfloat16", "nvfp4"),
        ("bfloat16", "fp8", "nvfp4"),
        ("fp8", "fp8", "nvfp4"),
    ]
    assert get_mla_module_sweep_spec("sglang").context_sequence_lengths[-2:] == [8192, 16384]

    vllm_sweep = get_mla_module_sweep_spec("vllm")
    assert vllm_sweep.context_sequence_lengths[-1] == 32768
    assert vllm_sweep.generation_sequence_lengths[-1] == 131072
    assert vllm_sweep.inner_sweep_head_counts == [128, 64, 32, 16, 8, 4, 2, 1]
    assert vllm_sweep.generation_max_tokens == 33554432
    assert vllm_sweep.generation_large_cache_tokens == 16777216
    assert [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs("vllm", phase="generation", sm_version=90)
    ] == [
        ("bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16", "fp8", "bfloat16"),
        ("bfloat16", "bfloat16", "fp8_block"),
        ("bfloat16", "fp8", "fp8_block"),
    ]

    # vLLM 0.24.0 FP8 prefill-query compute is declared for the dense-MLA
    # prefill path only: the sparse DSA builders have no prefill-query
    # quantization concept, so the fp8 compute combos are scoped
    # attention_types: [mla] and a DSA plan must not expand them.
    assert [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs("vllm", phase="context", sm_version=100, attention_type="dsa")
    ] == [
        ("bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16", "fp8", "bfloat16"),
        ("bfloat16", "bfloat16", "fp8_block"),
        ("bfloat16", "fp8", "fp8_block"),
        ("bfloat16", "bfloat16", "nvfp4"),
        ("bfloat16", "fp8", "nvfp4"),
    ]
    assert [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs("vllm", phase="context", sm_version=100, attention_type="mla")
    ] == [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs("vllm", phase="context", sm_version=100)
    ]

    with pytest.raises(ValueError, match="attention_type"):
        get_mla_module_precision_specs("vllm", phase="context", sm_version=100, attention_type="dense")

    assert {spec.model_path for spec in dsa_specs} == {
        "deepseek-ai/DeepSeek-V3.2",
        "zai-org/GLM-5",
        "zai-org/GLM-5-FP8",
        "nvidia/GLM-5-NVFP4",
        "nvidia/GLM-5.2-NVFP4",
    }
    assert {spec.native_num_heads for spec in dsa_specs if spec.architecture == "GlmMoeDsaForCausalLM"} == {64}
    assert {(spec.model_path, spec.architecture, spec.native_num_heads) for spec in kimi_specs} == {
        ("moonshotai/Kimi-K2-Instruct", "DeepseekV3ForCausalLM", 64),
        ("moonshotai/Kimi-K2.5", "KimiK25ForConditionalGeneration", 64),
        ("nvidia/Kimi-K2.5-NVFP4", "KimiK25ForConditionalGeneration", 64),
    }
    assert {spec.model_path for spec in wideep_specs} == {
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "nvidia/DeepSeek-V3.1-NVFP4",
    }
    assert {(spec.attention_type, spec.model_path, spec.architecture) for spec in vllm_specs} == {
        ("mla", "deepseek-ai/DeepSeek-V3", "DeepseekV3ForCausalLM"),
        ("dsa", "deepseek-ai/DeepSeek-V3.2", "DeepseekV32ForCausalLM"),
        ("dsa", "zai-org/GLM-5", "GlmMoeDsaForCausalLM"),
    }


def test_mla_module_targeted_artifacts_keep_requested_checkpoint(monkeypatch):
    from collector.case_generator import get_mla_module_model_specs

    for model_path, attention_type, architecture in (
        ("nvidia/DeepSeek-V3.1-NVFP4", "mla", "DeepseekV3ForCausalLM"),
        ("moonshotai/Kimi-K2-Instruct", "mla", "DeepseekV3ForCausalLM"),
        ("moonshotai/Kimi-K2.5", "mla", "KimiK25ForConditionalGeneration"),
        ("nvidia/Kimi-K2.5-NVFP4", "mla", "KimiK25ForConditionalGeneration"),
        ("nvidia/GLM-5-NVFP4", "dsa", "GlmMoeDsaForCausalLM"),
    ):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        specs = get_mla_module_model_specs(attention_type=attention_type, backend="vllm")
        assert [(spec.model_path, spec.architecture) for spec in specs] == [(model_path, architecture)]


def test_vllm_mla_module_artifacts_have_local_configs():
    from collector.case_generator import get_mla_module_model_specs

    config_root = REPO_ROOT / "src" / "aiconfigurator" / "model_configs"
    for spec in get_mla_module_model_specs(backend="vllm", apply_model_filter=False):
        config_path = config_root / f"{spec.model_path.replace('/', '--')}_config.json"
        assert config_path.is_file(), f"{spec.model_path} would require a runtime Hub download"


def test_shape_only_mla_alias_uses_canonical_model(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "moonshotai/Kimi-K2-Instruct")
    from collector.case_generator import get_context_mla_case_specs

    kimi_specs = get_context_mla_case_specs()
    assert kimi_specs
    assert {spec.model_name for spec in kimi_specs} == {"moonshotai/Kimi-K2.5"}
    assert {spec.num_heads for spec in kimi_specs} == {64, 128}


def test_model_cases_path_can_infer_model_path():
    model_cases_path = default_architecture_cases_path("DeepseekV4ForCausalLM")

    plan = build_collection_case_plan(backend="sglang", model_cases_path=str(model_cases_path))

    assert plan.model_path == "sgl-project/DeepSeek-V4-Flash-FP8"
    assert plan.model_architecture == "DeepseekV4ForCausalLM"
    assert "dsv4_csa_context_module" in plan.selected_ops
    assert "dsv4_csa_topk_calib" in plan.selected_ops
    assert "mhc_module" in plan.selected_ops
    assert {
        "dsv4_paged_mqa_logits_module",
        "dsv4_hca_attn_module",
        "dsv4_csa_attn_module",
    }.isdisjoint(plan.selected_ops)


def test_plan_rejects_model_declared_ops_unknown_to_backend_registry(tmp_path):
    """A typo in a model-declared op name must fail plan building loudly, not
    silently collect nothing for the intended benchmark."""
    case_file = tmp_path / "FakeArchForCausalLM_cases.yaml"
    case_file.write_text(
        "architecture: FakeArchForCausalLM\n"
        "model_path: fake/model\n"
        "model_ops:\n"
        "  - attention_context\n"
        "  - attention_contxt_typo\n"
    )

    with pytest.raises(ValueError, match="attention_contxt_typo"):
        build_collection_case_plan(backend="vllm", model_cases_path=str(case_file))


def test_dsv4_plan_only_uses_backend_specific_case_plan():
    model_path = "deepseek-ai/DeepSeek-V4-Pro"
    expected_ops = build_collection_case_plan(backend="sglang", model_path=model_path).ops

    result = subprocess.run(
        [
            sys.executable,
            "collector/collect.py",
            "--backend",
            "sglang",
            "--model-path",
            model_path,
            "--plan-only",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ops"] == expected_ops
    assert "dsv4_csa_topk_calib" in payload["ops"]
    assert "wideep_moe" in payload["ops"]


def test_vllm_024_schedules_consumed_dsv4_modules_only():
    from collector.vllm.registry import REGISTRY

    consumed_dsv4_ops = {
        "dsv4_csa_context_module",
        "dsv4_hca_context_module",
        "dsv4_csa_generation_module",
        "dsv4_hca_generation_module",
    }
    registry_only_ops = {"dsv4_paged_mqa_logits_module", "dsv4_hca_attn_module", "mhc_module"}
    plan = build_collection_case_plan(backend="vllm", model_path="sgl-project/DeepSeek-V4-Pro-FP8")

    assert plan.ops == [
        "dsv4_csa_context_module",
        "dsv4_csa_generation_module",
        "dsv4_hca_context_module",
        "dsv4_hca_generation_module",
        "gemm",
        "moe",
    ]
    assert consumed_dsv4_ops <= plan.selected_ops
    assert registry_only_ops.isdisjoint(plan.selected_ops)
    assert consumed_dsv4_ops | registry_only_ops <= {entry.op for entry in REGISTRY}


def test_model_architecture_can_select_case_file():
    plan = build_collection_case_plan(backend="trtllm", model_architecture="Qwen3MoeForCausalLM")

    assert plan.model_path == "Qwen/Qwen3-235B-A22B"
    assert plan.model_architecture == "Qwen3MoeForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Qwen3MoeForCausalLM")]
    assert "moe" in plan.selected_ops
    assert "mla_module" not in plan.selected_ops


def test_model_path_alias_resolves_architecture_case_file():
    plan = build_collection_case_plan(backend="trtllm", model_path="Qwen/Qwen3-235B-A22B-FP8")

    assert plan.model_path == "Qwen/Qwen3-235B-A22B-FP8"
    assert plan.model_architecture == "Qwen3MoeForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Qwen3MoeForCausalLM")]
    assert "moe" in plan.selected_ops


def test_encoder_attention_plan_matches_sdk_model_and_backend_support():
    dense_plan = build_collection_case_plan(backend="sglang", model_path="Qwen/Qwen3-32B")
    assert dense_plan.ops == ["attention_context", "attention_generation", "gemm"]
    assert not dense_plan.has_op("encoder_attention")

    for backend in ("sglang", "trtllm", "vllm"):
        assert build_collection_case_plan(
            backend=backend,
            model_path="Qwen/Qwen3-VL-32B-Instruct",
        ).has_op("encoder_attention")
    assert not build_collection_case_plan(
        backend="vllm_xpu",
        model_path="Qwen/Qwen3-VL-32B-Instruct",
    ).has_op("encoder_attention")


def test_vllm_024_model_plans_only_schedule_representable_attention_paths():
    kimi_path = "moonshotai/Kimi-K2.5"
    assert build_collection_case_plan(backend="vllm", model_path=kimi_path).ops == [
        "encoder_attention",
        "gemm",
        "mla_context_module",
        "mla_generation_module",
        "moe",
    ]
    assert build_collection_case_plan(backend="vllm", model_path="moonshotai/Kimi-K2-Instruct").ops == [
        "gemm",
        "mla_context_module",
        "mla_generation_module",
        "moe",
    ]
    assert build_collection_case_plan(backend="sglang", model_path=kimi_path).ops == [
        "gemm",
        "mla_bmm_gen_post",
        "mla_bmm_gen_pre",
        "mla_context",
        "mla_generation",
        "moe",
    ]
    assert build_collection_case_plan(backend="trtllm", model_path=kimi_path).ops == [
        "gemm",
        "mla_bmm_gen_post",
        "mla_bmm_gen_pre",
        "mla_context",
        "mla_generation",
        "moe",
        "trtllm_moe_wideep",
    ]
    assert build_collection_case_plan(backend="vllm_xpu", model_path=kimi_path).ops == ["gemm", "moe"]

    models_with_unrepresentable_vllm_attention = (
        "XiaomiMiMo/MiMo-V2-Flash",
        "google/gemma-4-26B-A4B",
        "openai/gpt-oss-120b",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    )
    legacy_plan = ["attention_context", "attention_generation", "gemm", "moe"]
    for model_path in models_with_unrepresentable_vllm_attention:
        assert build_collection_case_plan(backend="vllm", model_path=model_path).ops == ["gemm", "moe"]
        assert build_collection_case_plan(backend="vllm_xpu", model_path=model_path).ops == ["gemm", "moe"]
        for backend in ("sglang", "trtllm"):
            assert build_collection_case_plan(backend=backend, model_path=model_path).ops == legacy_plan


def test_compute_scale_is_selected_only_for_static_fp8_artifact():
    static_model = "Qwen/Qwen3-32B-FP8-Static-PerTensor"
    non_static_models = ("Qwen/Qwen3-32B", "Qwen/Qwen3-32B-FP8", "Qwen/Qwen3-0.6B")

    for backend in ("sglang", "trtllm", "vllm"):
        static_plan = build_collection_case_plan(backend=backend, model_path=static_model, sm_version=100)
        assert "compute_scale" in static_plan.selected_ops
        assert "compute_scale" in build_collection_case_plan(backend=backend, full=True).selected_ops

        for model_path in non_static_models:
            plan = build_collection_case_plan(backend=backend, model_path=model_path, sm_version=100)
            assert "compute_scale" not in plan.selected_ops

    xpu_plan = build_collection_case_plan(backend="vllm_xpu", model_path=static_model)
    assert "compute_scale" not in xpu_plan.selected_ops


def test_model_plans_do_not_request_ops_missing_from_backend_registry():
    deepseek_vllm = build_collection_case_plan(backend="vllm", model_path="deepseek-ai/DeepSeek-V3")
    kimi_vllm = build_collection_case_plan(backend="vllm", model_path="moonshotai/Kimi-K2.5")
    nemotron_sglang = build_collection_case_plan(
        backend="sglang",
        model_path="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    )
    nemotron_trtllm = build_collection_case_plan(
        backend="trtllm",
        model_path="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    )

    assert not deepseek_vllm.has_op("mla_context")
    assert not deepseek_vllm.has_op("mla_generation")
    assert deepseek_vllm.has_op("mla_context_module")
    assert not kimi_vllm.has_op("attention_context")
    assert not kimi_vllm.has_op("attention_generation")
    assert kimi_vllm.has_op("encoder_attention")
    assert kimi_vllm.has_op("mla_context_module")
    assert kimi_vllm.has_op("mla_generation_module")
    assert not kimi_vllm.has_op("mla_context")
    assert not kimi_vllm.has_op("mla_generation")
    assert not nemotron_sglang.has_op("mamba2")
    assert nemotron_trtllm.has_op("mamba2")


def test_full_mode_aggregates_all_model_case_files():
    plan = build_collection_case_plan(backend="sglang", full=True)

    assert plan.model_path is None
    assert len(plan.model_cases_paths) >= 18
    assert "wideep_mla_context" in plan.selected_ops
    assert "dsv4_csa_context_module" in plan.selected_ops
    assert "gdn" in plan.selected_ops


def test_full_mode_ops_are_a_union_of_model_plan_ops():
    for backend in ("sglang", "trtllm"):
        full_plan = build_collection_case_plan(backend=backend, full=True)
        for model_path in ("deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2.5", "Qwen/Qwen3-32B"):
            model_plan = build_collection_case_plan(backend=backend, model_path=model_path)
            assert model_plan.selected_ops <= full_plan.selected_ops, f"{backend}/{model_path}"


def test_mla_module_metadata_preserves_legacy_backends_and_canonicalizes_vllm():
    from collector.case_generator import get_mla_module_model_specs

    original_artifacts = {
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-R1",
        "nvidia/DeepSeek-V3.1-NVFP4",
    }

    def paths(backend):
        return {
            spec.model_path
            for spec in get_mla_module_model_specs(
                attention_type="mla",
                backend=backend,
                apply_model_filter=backend == "vllm",
            )
        }

    assert paths("vllm") == {"deepseek-ai/DeepSeek-V3"}
    assert paths("sglang") == original_artifacts
    assert paths("trtllm") == original_artifacts


def test_support_matrix_models_have_model_case_aliases():
    case_aliases = set()
    for path in (REPO_ROOT / "collector" / "cases" / "models").glob("*_cases.yaml"):
        data = path.read_text(encoding="utf-8")
        for line in data.splitlines():
            stripped = line.strip()
            if stripped.startswith("model_path: "):
                case_aliases.add(stripped.removeprefix("model_path: ").strip())
            elif stripped.startswith("- "):
                case_aliases.add(stripped.removeprefix("- ").strip())

    support_matrix_models = set()
    for path in SUPPORT_MATRIX_ROOT.glob("*.csv"):
        with path.open(encoding="utf-8") as f:
            support_matrix_models.update(row["HuggingFaceID"] for row in csv.DictReader(f))

    assert support_matrix_models <= case_aliases


def test_support_matrix_moe_alias_generates_targeted_cases(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-235B-A22B-FP8")

    cases = get_common_moe_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {"Qwen/Qwen3-235B-A22B"}


def test_qwen3_30b_fp8_alias_reuses_canonical_case_and_tp_constraints(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-30B-A3B-FP8")

    cases = get_common_moe_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {"Qwen/Qwen3-30B-A3B"}
    assert all(case.tp < 8 for case in cases)


def test_quant_sensitive_moe_artifacts_use_quant_equivalent_representatives(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    expected_representatives = {
        "nvidia/DeepSeek-V3.1-NVFP4": "nvidia/DeepSeek-V3.1-NVFP4",
        "nvidia/MiniMax-M2.5-NVFP4": "nvidia/MiniMax-M2.5-NVFP4",
        "nvidia/MiniMax-M2.7-NVFP4": "nvidia/MiniMax-M2.5-NVFP4",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    }

    for model_path, expected_representative in expected_representatives.items():
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        cases = get_common_moe_test_cases()
        assert cases and {case.model_name for case in cases} == {expected_representative}


def test_nemotron_ultra_quant_artifact_keeps_moe_path_but_reuses_mamba_profile(monkeypatch):
    from collector.case_generator import get_common_mamba2_test_cases, get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8")

    moe_cases = get_common_moe_test_cases()
    mamba_cases = get_common_mamba2_test_cases()

    assert moe_cases and {case.model_name for case in moe_cases} == {"nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8"}
    assert {case.hidden_size for case in moe_cases} == {2048, 8192}
    assert mamba_cases and {case.model_name for case in mamba_cases} == {
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
    }


def test_unverified_nemotron_rl_artifact_has_no_moe_profile(monkeypatch):
    from collector.case_generator import get_common_mamba2_test_cases, get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/nemotron-ultra-rl-050826")

    assert get_common_moe_test_cases() == []
    assert get_common_mamba2_test_cases()


def test_support_matrix_mamba_alias_generates_targeted_cases(monkeypatch):
    from collector.case_generator import get_common_mamba2_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")

    cases = get_common_mamba2_test_cases()

    assert cases
    assert {case.model_name for case in cases} == {"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"}


def test_collector_case_yaml_numeric_lists_are_sorted():
    def is_number(value):
        return isinstance(value, int | float) and not isinstance(value, bool)

    def walk_numeric_lists(value, path):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield from walk_numeric_lists(nested, (*path, str(key)))
        elif isinstance(value, list):
            if len(value) > 1 and all(is_number(item) for item in value):
                yield path, value
            for index, nested in enumerate(value):
                yield from walk_numeric_lists(nested, (*path, str(index)))

    violations = []
    for path in sorted((REPO_ROOT / "collector" / "cases").glob("**/*.yaml")):
        for yaml_path, values in walk_numeric_lists(load_yaml_file(path), ()):
            adjacent_values = list(pairwise(values))
            ascending = all(left <= right for left, right in adjacent_values)
            descending = all(left >= right for left, right in adjacent_values)
            if not (ascending or descending):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{'.'.join(yaml_path)} = {values}")

    assert violations == []
