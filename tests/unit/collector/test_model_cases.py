# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import csv
import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

from collector.case_generator import (
    get_attention_head_configs,
    get_moe_quantization_specs,
    moe_model_allows_quantization,
)
from collector.helper import create_test_case_id
from collector.model_cases import (
    CaseSelector,
    OpCasePlan,
    build_collection_case_plan,
    default_architecture_cases_path,
    default_sm_exceptions_path,
    expected_failure_for_test_case,
    filter_test_cases,
    filter_test_cases_with_report,
    load_yaml_file,
)

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
    assert "gemm" in plan.op_cases
    assert "attention_context" not in plan.op_cases
    assert "attention_generation" not in plan.op_cases
    assert "moe" in plan.op_cases
    assert "mla_context" in plan.op_cases
    assert "wideep_mla_context" in plan.op_cases
    assert "wideep_moe" in plan.op_cases
    assert "trtllm_moe_wideep" not in plan.op_cases


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
            for config in get_attention_head_configs(get_shape_sweeps("sglang")[0], phase=phase)
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
        for config in get_attention_head_configs(
            get_attention_context_shape_sweeps("sglang")[0],
            phase="context",
        )
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
                for config in get_attention_head_configs(get_shape_sweeps(backend)[0], phase=phase)
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
                for config in get_attention_head_configs(
                    get_attention_context_shape_sweeps("sglang")[0],
                    phase="context",
                )
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
        for config in get_attention_head_configs(
            get_attention_context_shape_sweeps("vllm")[0],
            phase="context",
        )
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
        for config in get_attention_head_configs(
            get_attention_context_shape_sweeps("vllm")[0],
            phase="context",
        )
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
        sweep = get_attention_encoder_shape_sweeps(backend)[0]
        configs = get_attention_encoder_head_configs(sweep)
        keys = {(config.num_heads, config.head_dim) for config in configs}
        default_keys = {(num_heads, head_dim) for head_dim in sweep["head_dims"] for num_heads in sweep["head_counts"]}

        assert default_keys <= keys
        assert keys - default_keys == {(1, 64), (1, 72)}


def test_targeted_encoder_attention_profile_is_model_exact(monkeypatch):
    from collector.case_generator import get_attention_encoder_head_configs, get_attention_encoder_shape_sweeps

    sweep = get_attention_encoder_shape_sweeps("vllm")[0]
    for model_path, head_dim in (
        ("Qwen/Qwen3-VL-4B-Instruct", 64),
        ("Qwen/Qwen3-VL-32B-Instruct", 72),
        ("Qwen/Qwen3-VL-235B-A22B-Instruct", 72),
        ("moonshotai/Kimi-K2.5", 72),
        ("nvidia/Kimi-K2.5-NVFP4", 72),
    ):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        configs = get_attention_encoder_head_configs(sweep)

        assert {(config.num_heads, config.head_dim) for config in configs} == {
            (16, head_dim),
            (8, head_dim),
            (4, head_dim),
            (2, head_dim),
            (1, head_dim),
        }


def test_base_gemm_cases_are_readable_shape_specs():
    plan = build_collection_case_plan(backend="sglang", model_path="Qwen/Qwen3-32B")
    gemm_plan = plan.op_cases["gemm"]
    context_plan = plan.op_cases["attention_context"]
    generation_plan = plan.op_cases["attention_generation"]

    assert len(gemm_plan.include.case_specs) == 1
    spec = gemm_plan.include.case_specs[0]
    assert spec["id"] == "base_transformer_gemm_shape_sweep"
    assert spec["token_counts"][:5] == [1, 2, 3, 4, 5]
    assert spec["feature_sizes"][:3] == [32, 64, 128]

    assert context_plan.include.case_specs[0]["id"] == "base_attention_context_shape_sweep"
    assert context_plan.include.case_specs[0]["kv_head_options"] == ["self", 1, 2, 4, 8]
    assert generation_plan.include.case_specs[0]["id"] == "base_attention_generation_shape_sweep"
    assert generation_plan.include.case_specs[0]["xqa_query_head_counts"][-1] == 128

    filtered = filter_test_cases(
        ["case0", "case1"], plan=gemm_plan, full_module_name="sglang.gemm", run_func_name="run_gemm"
    )

    assert filtered == ["case0", "case1"]


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
            "sgl-project/DeepSeek-V4-Flash-FP8": set(),
            "sgl-project/DeepSeek-V4-Pro-FP8": set(),
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

    sglang_context = get_attention_context_shape_sweeps("sglang")[0]
    trtllm_context = get_attention_context_shape_sweeps("trtllm")[0]
    vllm_context = get_attention_context_shape_sweeps("vllm")[0]
    vllm_xpu_context = get_attention_context_shape_sweeps("vllm_xpu")[0]
    vllm_generation = get_attention_generation_shape_sweeps("vllm")[0]

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
    assert len(moe_cases) == 4209
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
        assert required_ops <= set(plan.op_cases)
        for op_name in ("mla_context", "mla_generation"):
            assert plan.op_cases[op_name].include.rules[0]["match"] == {"tp_size": {"in": [1, 2, 4, 8]}}
        for op_name in ("mla_bmm_gen_pre", "mla_bmm_gen_post"):
            assert plan.op_cases[op_name].include.rules[0]["match"] == {"num_heads": {"in": [64, 32, 16, 8]}}


def test_kimi_targeted_mla_selector_keeps_consumer_local_head_grid(monkeypatch):
    from types import SimpleNamespace

    from collector.case_generator import get_context_mla_case_specs

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "moonshotai/Kimi-K2.5")
    adapter = _load_mla_adapter(
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
    generated = adapter(get_context_mla_case_specs(), dtype_list=("bf16",))
    op_plan = build_collection_case_plan(
        backend="trtllm",
        model_path="moonshotai/Kimi-K2.5",
    ).op_cases["mla_context"]
    selected = filter_test_cases(
        generated,
        plan=op_plan,
        full_module_name="trtllm.mla_context",
        run_func_name="run_mla",
    )

    assert {case[4] for case in selected} == {64, 128}
    assert {case[6] for case in selected} == {1, 2, 4, 8}
    assert {case[4] // case[6] for case in selected} == {128, 64, 32, 16, 8}
    assert any(case[4] == 64 and case[6] == 8 for case in selected)


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
        moe_shape_satisfies_constraints,
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

    assert moe_shape_satisfies_constraints(
        "vllm",
        "fp8_block",
        hidden_size=4096,
        inter_size=1536,
        tensor_parallel_size=1,
        topk=8,
    )
    assert not moe_shape_satisfies_constraints(
        "vllm",
        "fp8_block",
        hidden_size=4100,
        inter_size=1536,
        tensor_parallel_size=1,
        topk=8,
    )
    assert not moe_shape_satisfies_constraints(
        "vllm",
        "nvfp4",
        hidden_size=4096,
        inter_size=1536,
        tensor_parallel_size=1,
        topk=22,
    )


def test_vllm_sm120_nvfp4_moe_gap_is_an_sm_exception():
    plan = build_collection_case_plan(backend="vllm", model_path="Qwen/Qwen3-235B-A22B", sm_version=120)
    cases = [
        ["bfloat16", [1, 2], 4096, 1536, 8, 128, 1, 1, "Qwen/Qwen3-235B-A22B", "balanced", 0.0],
        ["nvfp4", [1, 2], 4096, 1536, 8, 128, 1, 1, "Qwen/Qwen3-235B-A22B", "balanced", 0.0],
    ]

    filtered, skipped = filter_test_cases_with_report(
        cases,
        plan=plan.op_cases["moe"],
        full_module_name="vllm.moe",
        run_func_name="run_moe_torch",
        runtime_version="0.19.0",
    )

    assert filtered == [cases[0]]
    assert skipped[0]["reason_type"] == "framework_version_unsupported"
    assert "FlashInfer TRT-LLM FP4 MoE path" in skipped[0]["reason"]


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
        ("bfloat16", "bfloat16", "fp8_block"),
        ("bfloat16", "fp8", "fp8_block"),
        ("bfloat16", "bfloat16", "nvfp4"),
        ("bfloat16", "fp8", "nvfp4"),
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

    assert {spec.model_path for spec in dsa_specs} == {
        "deepseek-ai/DeepSeek-V3.2",
        "zai-org/GLM-5",
        "zai-org/GLM-5-FP8",
        "nvidia/GLM-5-NVFP4",
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
    assert "dsv4_csa_context_module" in plan.op_cases
    assert "dsv4_csa_topk_calib" in plan.op_cases
    assert "mhc_module" in plan.op_cases
    assert {
        "dsv4_paged_mqa_logits_module",
        "dsv4_hca_attn_module",
        "dsv4_csa_attn_module",
    }.isdisjoint(plan.op_cases)


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
    assert consumed_dsv4_ops <= plan.op_cases.keys()
    assert registry_only_ops.isdisjoint(plan.op_cases)
    assert consumed_dsv4_ops | registry_only_ops <= {entry.op for entry in REGISTRY}


def test_model_architecture_can_select_case_file():
    plan = build_collection_case_plan(backend="trtllm", model_architecture="Qwen3MoeForCausalLM")

    assert plan.model_path == "Qwen/Qwen3-235B-A22B"
    assert plan.model_architecture == "Qwen3MoeForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Qwen3MoeForCausalLM")]
    assert "moe" in plan.op_cases
    assert "mla_module" not in plan.op_cases


def test_model_path_alias_resolves_architecture_case_file():
    plan = build_collection_case_plan(backend="trtllm", model_path="Qwen/Qwen3-235B-A22B-FP8")

    assert plan.model_path == "Qwen/Qwen3-235B-A22B-FP8"
    assert plan.model_architecture == "Qwen3MoeForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("Qwen3MoeForCausalLM")]
    assert "moe" in plan.op_cases


def test_encoder_attention_plan_matches_sdk_model_and_backend_support():
    dense_plan = build_collection_case_plan(backend="sglang", model_path="Qwen/Qwen3-32B")
    assert dense_plan.ops == ["attention_context", "attention_generation", "gemm"]
    assert "encoder_attention" not in dense_plan.op_cases

    for backend in ("sglang", "trtllm", "vllm"):
        assert (
            "encoder_attention"
            in build_collection_case_plan(
                backend=backend,
                model_path="Qwen/Qwen3-VL-32B-Instruct",
            ).op_cases
        )
    assert (
        "encoder_attention"
        not in build_collection_case_plan(
            backend="vllm_xpu",
            model_path="Qwen/Qwen3-VL-32B-Instruct",
        ).op_cases
    )


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
        assert "compute_scale" in static_plan.op_cases
        assert "compute_scale" in build_collection_case_plan(backend=backend, full=True).op_cases

        for model_path in non_static_models:
            plan = build_collection_case_plan(backend=backend, model_path=model_path, sm_version=100)
            assert "compute_scale" not in plan.op_cases

    xpu_plan = build_collection_case_plan(backend="vllm_xpu", model_path=static_model)
    assert "compute_scale" not in xpu_plan.op_cases


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

    assert "mla_context" not in deepseek_vllm.op_cases
    assert "mla_generation" not in deepseek_vllm.op_cases
    assert "mla_context_module" in deepseek_vllm.op_cases
    assert "attention_context" not in kimi_vllm.op_cases
    assert "attention_generation" not in kimi_vllm.op_cases
    assert "encoder_attention" in kimi_vllm.op_cases
    assert "mla_context_module" in kimi_vllm.op_cases
    assert "mla_generation_module" in kimi_vllm.op_cases
    assert "mla_context" not in kimi_vllm.op_cases
    assert "mla_generation" not in kimi_vllm.op_cases
    assert "mamba2" not in nemotron_sglang.op_cases
    assert "mamba2" in nemotron_trtllm.op_cases


def test_full_mode_aggregates_all_model_case_files():
    plan = build_collection_case_plan(backend="sglang", full=True)

    assert plan.model_path is None
    assert len(plan.model_cases_paths) >= 18
    assert "wideep_mla_context" in plan.op_cases
    assert "dsv4_csa_context_module" in plan.op_cases
    assert "gdn" in plan.op_cases


def test_full_mode_unions_model_selectors_instead_of_last_model_wins():
    for backend in ("sglang", "trtllm"):
        plan = build_collection_case_plan(backend=backend, full=True)
        for op in ("mla_context", "mla_generation", "mla_bmm_gen_pre", "mla_bmm_gen_post"):
            assert plan.op_cases[op].include.all_cases, f"{backend}/{op} was narrowed by a later model document"


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


def test_sm_exceptions_can_drop_framework_specific_op(tmp_path: Path):
    exceptions = tmp_path / "sm100_exceptions.yaml"
    exceptions.write_text(
        """
schema_version: 1
sm_version: 100
framework_specific_op_exceptions:
  sglang:
    wideep_moe:
      drop: true
""",
        encoding="utf-8",
    )

    plan = build_collection_case_plan(
        backend="sglang",
        model_path="deepseek-ai/DeepSeek-V3",
        sm_version=100,
        sm_exceptions_path=str(exceptions),
    )

    assert plan.sm_version == 100
    assert plan.sm_exceptions_path == exceptions.resolve()
    assert "wideep_moe" not in plan.op_cases
    assert "wideep_mla_context" in plan.op_cases


def test_sm_exception_catalog_does_not_activate_unrelated_ops():
    base_plan = build_collection_case_plan(backend="sglang", model_path="Qwen/Qwen3-32B")
    sm120_plan = build_collection_case_plan(
        backend="sglang",
        model_path="Qwen/Qwen3-32B",
        sm_version=120,
    )

    assert sm120_plan.ops == base_plan.ops == ["attention_context", "attention_generation", "gemm"]
    assert "moe" not in sm120_plan.op_cases
    assert "wideep_mla_context" not in sm120_plan.op_cases


def test_gpu_type_resolves_default_sm_exception_file(tmp_path: Path, monkeypatch):
    from collector import model_cases as model_case_module

    exceptions = tmp_path / "sm100_exceptions.yaml"
    exceptions.write_text(
        """
schema_version: 1
sm_version: 100
framework_specific_op_exceptions:
  sglang:
    wideep_moe:
      drop: true
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_case_module, "SM_EXCEPTIONS_DIR", tmp_path)

    plan = model_case_module.build_collection_case_plan(
        backend="sglang",
        model_path="deepseek-ai/DeepSeek-V3",
        gpu_type="b200_sxm",
    )

    assert plan.gpu_type == "b200_sxm"
    assert plan.sm_version == 100
    assert plan.sm_exceptions_path == default_sm_exceptions_path(100)
    assert plan.sm_exceptions_path == exceptions
    assert "wideep_moe" not in plan.op_cases


def test_sm_exception_files_list_matching_gpu_types():
    systems_dir = REPO_ROOT / "src" / "aiconfigurator" / "systems"
    expected_gpu_types_by_sm = {}
    for system_path in sorted(systems_dir.glob("*.yaml")):
        system_data = load_yaml_file(system_path)
        gpu = system_data.get("gpu", {})
        if not isinstance(gpu, dict) or gpu.get("sm_version") is None:
            continue
        expected_gpu_types_by_sm.setdefault(int(gpu["sm_version"]), []).append(system_path.stem)

    for sm_version, gpu_types in expected_gpu_types_by_sm.items():
        exception_path = default_sm_exceptions_path(sm_version)
        if not exception_path.exists():
            continue
        exception_data = load_yaml_file(exception_path)
        assert exception_data.get("gpu_types") == gpu_types


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


def test_filter_test_cases_supports_case_ids_contains_and_indices():
    cases = ["tp=1 ep=1", "tp=2 ep=1", "tp=4 ep=1", "tp=8 ep=1"]
    case_id = create_test_case_id(cases[1], "run_moe_torch", "sglang.moe")
    plan = OpCasePlan(
        include=CaseSelector(case_ids={case_id}, contains={"tp=4"}, indices={3}),
        exclude=CaseSelector(contains={"tp=8"}),
    )

    filtered = filter_test_cases(cases, plan=plan, full_module_name="sglang.moe", run_func_name="run_moe_torch")

    assert filtered == ["tp=2 ep=1", "tp=4 ep=1"]


def test_yaml_include_selector_can_narrow_a_new_op_plan(tmp_path: Path):
    cases = ["case0", "case1", "case2"]
    selected_case_id = create_test_case_id(cases[1], "run_gemm", "sglang.gemm")
    model_cases = tmp_path / "SelectorArchitecture_cases.yaml"
    model_cases.write_text(
        f"""
schema_version: 1
architecture: SelectorArchitecture
model_path: test/selector-model
include_base: true
base_ops:
  - gemm
all_frameworks_op_cases:
  gemm:
    case_ids:
      - '{selected_case_id}'
""",
        encoding="utf-8",
    )

    plan = build_collection_case_plan(
        backend="sglang",
        model_cases_path=str(model_cases),
    )
    gemm_plan = plan.op_cases["gemm"]

    assert not gemm_plan.include.all_cases
    assert gemm_plan.include.case_ids == {selected_case_id}
    assert filter_test_cases(
        cases,
        plan=gemm_plan,
        full_module_name="sglang.gemm",
        run_func_name="run_gemm",
    ) == [cases[1]]


def test_filter_test_cases_supports_index_ranges_and_limit():
    cases = ["case0", "case1", "case2", "case3"]
    plan = OpCasePlan(include=CaseSelector(index_ranges=[(1, 3)], limit=2))

    filtered = filter_test_cases(cases, plan=plan, full_module_name="vllm.gemm", run_func_name="run_gemm")

    assert filtered == ["case1", "case2"]


def test_filter_test_cases_supports_structured_exception_rules():
    cases = [
        ["bfloat16", 1, 65536, 65536],
        ["fp8", 1, 65536, 65536],
        ["fp8", 16, 65536, 65536],
    ]
    plan = OpCasePlan(
        exclude=CaseSelector(
            rules=[
                {
                    "reason_type": "framework_version_unsupported",
                    "version_prefixes": ["1.3.0rc10"],
                    "fields": ["gemm_type", "token_count", "output_features", "input_features"],
                    "match": {
                        "gemm_type": "fp8",
                        "token_count": {"lte": 8},
                        "output_features": {"gte": 51200},
                        "input_features": {"gte": 51200},
                    },
                }
            ]
        )
    )

    filtered = filter_test_cases(
        cases,
        plan=plan,
        full_module_name="trtllm.gemm",
        run_func_name="run_gemm",
        runtime_version="1.3.0rc10",
    )

    assert filtered == [cases[0], cases[2]]


def test_filter_test_cases_supports_not_in_structured_exception_rule():
    cases = [
        ["nvfp4", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"],
        ["nvfp4", "Qwen/Qwen3-235B-A22B"],
        ["bfloat16", "Qwen/Qwen3-235B-A22B"],
    ]
    plan = OpCasePlan(
        exclude=CaseSelector(
            rules=[
                {
                    "reason_type": "framework_version_unsupported",
                    "version_prefixes": ["0.5.10"],
                    "fields": ["moe_type", "model_path"],
                    "match": {
                        "moe_type": "nvfp4",
                        "model_path": {
                            "not_in": [
                                "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                                "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
                            ]
                        },
                    },
                }
            ]
        )
    )

    filtered = filter_test_cases(
        cases,
        plan=plan,
        full_module_name="sglang.moe",
        run_func_name="run_moe_torch",
        runtime_version="0.5.10",
    )

    assert filtered == [cases[0], cases[2]]


def test_filter_test_cases_reports_expected_sm_exception_reasons():
    cases = [
        ["bfloat16", 1, 65536, 65536],
        ["fp8", 1, 65536, 65536],
    ]
    plan = OpCasePlan(
        exclude=CaseSelector(
            rules=[
                {
                    "reason_type": "framework_version_unsupported",
                    "reason": "tiny-M FP8 large GEMM crashes on this framework build",
                    "fields": ["gemm_type", "token_count", "output_features", "input_features"],
                    "match": {"gemm_type": "fp8"},
                }
            ]
        )
    )

    filtered, skipped = filter_test_cases_with_report(
        cases,
        plan=plan,
        full_module_name="trtllm.gemm",
        run_func_name="run_gemm",
    )

    assert filtered == [cases[0]]
    assert skipped == [
        {
            "case_id": create_test_case_id(cases[1], "run_gemm", "trtllm.gemm"),
            "index": 1,
            "source": "sm_exception",
            "selector": "rule",
            "reason_type": "framework_version_unsupported",
            "reason": "tiny-M FP8 large GEMM crashes on this framework build",
        }
    ]


def test_sm89_exception_excludes_sglang_head_dim_256_context_attention_before_execution():
    plan = build_collection_case_plan(
        backend="sglang",
        model_path="google/gemma-4-26B-A4B",
        gpu_type="l40s",
    )
    case = [1, 256, 2, 1, 256, False, False, True, 1024]

    for version in ("0.5.9", "0.5.10", "0.5.12"):
        filtered, skipped = filter_test_cases_with_report(
            [case],
            plan=plan.op_cases["attention_context"],
            full_module_name="sglang.attention_context",
            run_func_name="run_attention_torch",
            runtime_version=version,
        )

        assert filtered == []
        assert skipped[0]["reason_type"] == "framework_version_unsupported"
        assert "head_dim=256" in skipped[0]["reason"]

    filtered, skipped = filter_test_cases_with_report(
        [case],
        plan=plan.op_cases["attention_context"],
        full_module_name="sglang.attention_context",
        run_func_name="run_attention_torch",
        runtime_version="0.5.8",
    )
    assert filtered == [case]
    assert skipped == []


def test_sm90_vllm_024_excludes_sm100_only_fp8_head_dim_512_attention():
    sm90_plan = build_collection_case_plan(backend="vllm", sm_version=90, full=True)
    sm100_plan = build_collection_case_plan(backend="vllm", sm_version=100, full=True)
    cases = {
        "attention_context": [1, 1024, 8, 1, 512, True, True, 0],
        "attention_generation": [128, 31, 8, 1, 512, True, False, 0],
    }

    for op, case in cases.items():
        filtered, skipped = filter_test_cases_with_report(
            [case],
            plan=sm90_plan.op_cases[op],
            full_module_name=f"vllm.{op}",
            run_func_name="run_attention_torch",
            runtime_version="0.24.0",
        )
        assert filtered == []
        assert skipped[0]["reason_type"] == "framework_version_unsupported"
        assert "SM100-only" in skipped[0]["reason"]

        bf16_case = [*case]
        bf16_case[5] = False
        assert filter_test_cases(
            [bf16_case],
            plan=sm90_plan.op_cases[op],
            full_module_name=f"vllm.{op}",
            run_func_name="run_attention_torch",
            runtime_version="0.24.0",
        ) == [bf16_case]
        for plan, version in ((sm90_plan, "0.23.0"), (sm100_plan, "0.24.0")):
            assert filter_test_cases(
                [case, bf16_case],
                plan=plan.op_cases[op],
                full_module_name=f"vllm.{op}",
                run_func_name="run_attention_torch",
                runtime_version=version,
            ) == [case, bf16_case]


def test_expected_failure_matching_does_not_consider_exclude_selector():
    case = ["remaining"]
    plan = OpCasePlan(exclude=CaseSelector(indices={1}))

    assert (
        expected_failure_for_test_case(
            case,
            plan=plan,
            full_module_name="sglang.unmigrated_op",
            run_func_name="run_case",
            index=1,
        )
        is None
    )


def test_sm120_exception_filters_trtllm_gptoss_mxfp4():
    plan = build_collection_case_plan(
        backend="trtllm",
        model_path="openai/gpt-oss-120b",
        gpu_type="rtx_pro_6000_server",
    )
    case = [
        "w4a16_mxfp4",
        [1, 2, 4],
        2880,
        2880,
        4,
        128,
        1,
        1,
        False,
        "openai/gpt-oss-120b",
        "power_law",
        1.01,
    ]

    filtered, skipped = filter_test_cases_with_report(
        [case],
        plan=plan.op_cases["moe"],
        full_module_name="trtllm.moe",
        run_func_name="run_moe_torch",
        runtime_version="1.3.0rc10",
    )

    assert filtered == []
    assert skipped == [
        {
            "case_id": create_test_case_id(case, "run_moe_torch", "trtllm.moe"),
            "index": 0,
            "source": "sm_exception",
            "selector": "rule",
            "reason_type": "framework_version_unsupported",
            "reason": "TRT-LLM 1.3.0rc5/rc10 TRTLLMGenFusedMoE rejects GPT-OSS MXFP4 paths on SM120.",
        }
    ]
    filtered, skipped = filter_test_cases_with_report(
        [case],
        plan=plan.op_cases["moe"],
        full_module_name="trtllm.moe",
        run_func_name="run_moe_torch",
        runtime_version="1.3.0rc4",
    )
    assert filtered == [case]
    assert skipped == []


def test_sm100_exception_filters_trtllm_int4_wo():
    plan = build_collection_case_plan(
        backend="trtllm",
        model_path="moonshotai/Kimi-K2.5",
        gpu_type="b200_sxm",
    )
    case = [
        "int4_wo",
        [1],
        7168,
        2048,
        8,
        384,
        8,
        1,
        False,
        "moonshotai/Kimi-K2.5",
        "power_law",
        1.2,
    ]

    filtered, skipped = filter_test_cases_with_report(
        [case],
        plan=plan.op_cases["moe"],
        full_module_name="trtllm.moe",
        run_func_name="run_moe_torch",
        runtime_version="1.3.0rc10",
    )

    assert filtered == []
    assert skipped == [
        {
            "case_id": create_test_case_id(case, "run_moe_torch", "trtllm.moe"),
            "index": 0,
            "source": "sm_exception",
            "selector": "rule",
            "reason_type": "framework_version_unsupported",
            "reason": (
                "TRT-LLM 1.3.0rc10 SM100 CutlassFusedMoE rejects plain W4A16/int4_wo "
                "in create_moe with ValueError Unsupported quantization mode [1]."
            ),
        }
    ]
    filtered, skipped = filter_test_cases_with_report(
        [case],
        plan=plan.op_cases["moe"],
        full_module_name="trtllm.moe",
        run_func_name="run_moe_torch",
        runtime_version="1.3.0rc9",
    )
    assert filtered == [case]
    assert skipped == []


def test_filter_test_cases_supports_computed_rule_conditions():
    cases = [
        [1, 4096, 64, 4, 128, False, False, False],
        [1, 4096, 64, 2, 128, False, False, False],
        [2, 8192, 96, 96, 128, True, True, True],
    ]
    plan = OpCasePlan(
        exclude=CaseSelector(
            rules=[
                {
                    "reason_type": "hardware_unsupported",
                    "fields": [
                        "batch_size",
                        "sequence_length",
                        "num_heads",
                        "num_key_value_heads",
                        "head_dim",
                        "use_fp8_kv_cache",
                        "use_fp8_context_fmha",
                        "is_context_phase",
                    ],
                    "conditions": [
                        {
                            "ratio": {
                                "numerator": "num_heads",
                                "denominator": "num_key_value_heads",
                                "gt": 16,
                            }
                        }
                    ],
                },
                {
                    "reason_type": "framework_version_unsupported",
                    "fields": [
                        "batch_size",
                        "sequence_length",
                        "num_heads",
                        "num_key_value_heads",
                        "head_dim",
                        "use_fp8_kv_cache",
                        "use_fp8_context_fmha",
                        "is_context_phase",
                    ],
                    "match": {"use_fp8_context_fmha": True},
                    "conditions": [
                        {
                            "product": {
                                "fields": ["batch_size", "sequence_length"],
                                "gte": 16384,
                            }
                        }
                    ],
                },
            ]
        )
    )

    filtered = filter_test_cases(
        cases,
        plan=plan,
        full_module_name="trtllm.attention_context",
        run_func_name="run_attention_torch",
        runtime_version="1.3.0rc10",
    )

    assert filtered == [cases[0]]


def test_known_exceptions_classify_runtime_failures(tmp_path: Path):
    exceptions = tmp_path / "sm120_exceptions.yaml"
    exceptions.write_text(
        """
schema_version: 1
sm_version: 120
known_exceptions:
  - framework: sglang
    op: moe
    reason_type: framework_version_unsupported
    source: collector/sglang/collect_moe.py
    case_fields:
      - moe_type
      - num_tokens
      - hidden_size
      - inter_size
      - topk
      - num_experts
      - tensor_parallel_size
      - expert_parallel_size
      - model_path
    reason: expected SM120 shared-memory overflow
    threshold_groups:
      - label: Qwen3
        match: {hidden_size: 2048, inter_size: 768, topk: 8, num_experts: 128}
        thresholds:
          - {tensor_parallel_size: 4, expert_parallel_size: 16, num_tokens_gte: 16}
""",
        encoding="utf-8",
    )
    plan = build_collection_case_plan(
        backend="sglang",
        model_path="Qwen/Qwen3-30B-A3B",
        sm_exceptions_path=str(exceptions),
    )
    case = ["fp8_block", 32, 2048, 768, 8, 128, 4, 16, "Qwen/Qwen3-30B-A3B"]

    expected = expected_failure_for_test_case(
        case,
        plan=plan.op_cases["moe"],
        full_module_name="sglang.moe",
        run_func_name="run_moe_torch",
        runtime_version="0.5.10",
    )

    assert expected == {
        "case_id": create_test_case_id(case, "run_moe_torch", "sglang.moe"),
        "source": "known_exception",
        "selector": "rule",
        "reason_type": "framework_version_unsupported",
        "reason": "expected SM120 shared-memory overflow",
        "reference_source": "collector/sglang/collect_moe.py",
        "label": "Qwen3",
    }
