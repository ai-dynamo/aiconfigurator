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
    get_gemm_case_specs,
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


def _load_gdn_getter(module_path: str):
    from collector.case_generator import get_common_gdn_test_cases

    source_path = REPO_ROOT / module_path
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_gdn_test_cases"
    )
    namespace = {"get_common_gdn_test_cases": get_common_gdn_test_cases}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["get_gdn_test_cases"]


def test_model_case_plan_merges_required_base_and_framework_specific_ops():
    plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")

    assert plan.model_architecture == "DeepseekV3ForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("DeepseekV3ForCausalLM")]
    assert plan.has_op("gemm")
    assert not plan.has_op("attention_context")
    assert not plan.has_op("attention_generation")
    assert "moe" in plan.selected_ops
    assert "mla_context" in plan.selected_ops
    assert "wideep_mla_context" not in plan.selected_ops
    # The sglang plan keeps moe_ep out (separate WideEP 0.5.10 runtime); the
    # trtllm plan activates it (same image as stock trtllm) — see
    # tests/unit/collector/trtllm/test_collect_moe_ep.py.
    assert "moe_ep" not in plan.selected_ops
    assert "trtllm_moe_wideep" not in plan.selected_ops  # retired op name


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


@pytest.mark.parametrize(
    ("model_path", "d_model", "global_k_heads", "global_v_heads", "tp_sizes"),
    [
        ("Qwen/Qwen3.5-27B", 5120, 16, 48, (1, 2, 4, 8)),
        ("Qwen/Qwen3.5-35B-A3B", 2048, 16, 32, (1, 2, 4, 8, 16)),
    ],
)
def test_qwen35_gdn_getters_expand_tp_local_physical_keys(
    monkeypatch, model_path, d_model, global_k_heads, global_v_heads, tp_sizes
):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    expected = {
        (phase, d_model, 4, global_k_heads // tp, 128, global_v_heads // tp, 128)
        for phase in ("context", "generation")
        for tp in tp_sizes
    }

    for module_path in ("collector/sglang/collect_gdn.py", "collector/vllm/collect_gdn.py"):
        cases = _load_gdn_getter(module_path)()
        assert {(case[0], case[1], case[2], case[3], case[4], case[5], case[6]) for case in cases} == expected


def test_gdn_tp_declarations_fail_loud_and_dedupe_on_loader_key(monkeypatch):
    from collector import case_generator

    invalid = {
        "model_path": "example/invalid",
        "d_model": 2048,
        "d_conv": 4,
        "num_k_heads": 16,
        "head_k_dim": 128,
        "num_v_heads": 32,
        "head_v_dim": 128,
        "tensor_parallel_sizes": [3],
    }
    monkeypatch.setattr(case_generator, "_model_case_values", lambda op_name: [invalid])
    with pytest.raises(ValueError, match="both global head counts to be divisible"):
        case_generator.get_common_gdn_test_cases()

    def profile(model_path, d_model, num_k_heads, num_v_heads, tp):
        return {
            "model_path": model_path,
            "d_model": d_model,
            "d_conv": 4,
            "num_k_heads": num_k_heads,
            "head_k_dim": 128,
            "num_v_heads": num_v_heads,
            "head_v_dim": 128,
            "tensor_parallel_sizes": [tp],
        }

    profiles = [
        profile("example/first", 2048, 16, 32, 4),
        profile("example/duplicate", 2048, 4, 8, 1),
        profile("example/distinct-d-model", 4096, 4, 8, 1),
    ]
    monkeypatch.setattr(case_generator, "_model_case_values", lambda op_name: profiles)
    cases = case_generator.get_common_gdn_test_cases()

    assert len(cases) == 4
    assert {(case.phase, case.d_model, case.num_k_heads, case.num_v_heads, case.model_name) for case in cases} == {
        (phase, 2048, 4, 8, "example/first") for phase in ("context", "generation")
    } | {(phase, 4096, 4, 8, "example/distinct-d-model") for phase in ("context", "generation")}


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
    assert moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", "w4a8_mxfp4_mxfp8")
    assert not moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", "bfloat16")
    assert not moe_model_allows_quantization("sglang", "Qwen/Qwen3-235B-A22B", "w4a8_mxfp4_mxfp8")
    assert moe_model_allows_quantization("sglang", "nvidia/GLM-5.2-NVFP4", "nvfp4")
    assert not moe_model_allows_quantization("sglang", "nvidia/GLM-5.2-NVFP4", "bfloat16")
    assert moe_model_allows_quantization("sglang", "zai-org/GLM-5-FP8", "fp8_block")
    assert not moe_model_allows_quantization("sglang", "zai-org/GLM-5-FP8", "nvfp4")

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
            "deepseek-ai/DeepSeek-V4-Flash": {"w4a8_mxfp4_mxfp8"},
            "deepseek-ai/DeepSeek-V4-Pro": {"w4a8_mxfp4_mxfp8"},
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
            "deepseek-ai/DeepSeek-V4-Flash": {"w4a8_mxfp4_mxfp8"},
            "deepseek-ai/DeepSeek-V4-Pro": {"w4a8_mxfp4_mxfp8"},
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

    from collector.case_generator import get_moe_quantization_modes, get_moe_quantization_module_config

    assert get_moe_quantization_module_config("sglang", "int4_wo", model_name="moonshotai/Kimi-K2.5") == {
        "group_size": 32
    }
    sm90_modes = get_moe_quantization_modes("sglang", sm_version=90)
    assert "int4_wo" in sm90_modes
    assert "nvfp4" not in sm90_modes
    assert "w4a16_mxfp4" in sm90_modes
    # int4_wo opened to SM100/103 (2026-07-11): serving auto-selects
    # flashinfer_trtllm for Kimi INT4 there (server_args.py:3736).
    assert "int4_wo" in get_moe_quantization_modes("sglang", sm_version=100)
    assert "nvfp4" in get_moe_quantization_modes("sglang", sm_version=100)


def test_sglang_marlin_is_declared_only_for_weight_only_modes():
    def backend_maps(value):
        if isinstance(value, dict):
            if "sglang_moe_backends" in value:
                yield value["sglang_moe_backends"]
            for child in value.values():
                yield from backend_maps(child)
        elif isinstance(value, list):
            for child in value:
                yield from backend_maps(child)

    base = load_yaml_file(REPO_ROOT / "collector/cases/base_ops/moe.yaml")
    maps = [base["common_case_values"]["moe_sglang"]["backends"]]
    for path in (REPO_ROOT / "collector/cases/models").glob("*_cases.yaml"):
        maps.extend(backend_maps(load_yaml_file(path)))

    marlin_modes = {
        mode for backends in maps for mode, mapping in backends.items() if "marlin" in json.dumps(mapping).lower()
    }
    # Marlin is a weight-only (bf16-activation) runner: INT4-WO everywhere it
    # is declared, plus MXFP4 w4a16 on SM120 where SGLang 0.5.14 serving
    # itself selects Marlin (server_args.py:3876-3887). NVFP4 and
    # mxfp8-activation modes must never declare it (FP4/INT4 identity
    # reversal).
    assert marlin_modes == {"int4_wo", "w4a16_mxfp4"}


@pytest.mark.unit
def test_sglang_registry_marks_unvalidated_dsa_and_moe_platforms_explicitly():
    from collector.sglang.registry import REGISTRY

    sm90 = build_collection_case_plan(backend="sglang", full=True, sm_version=90)
    sm100 = build_collection_case_plan(backend="sglang", full=True, sm_version=100)
    entries = {entry.op: entry for entry in REGISTRY}

    # SM90 unparked by the h100/h200 probe collections (2026-08-14..15,
    # pipelines 62700025 + 62872230): 67,532 context + 4,896 generation
    # skip rows, fa3/flashmla buckets clean.
    for op in ("dsa_context_module_skip_indexer", "dsa_generation_module_skip_indexer"):
        assert op in sm90.selected_ops
        assert op in sm100.selected_ops
        assert entries[op].unverified_sms == (120,)

    # SM103 unparked by the B300 hardware probe (2026-07-13, pipeline
    # 57716023): sampled dsa cases ran clean across all three kernel buckets.
    for op in ("dsa_context_module", "dsa_generation_module"):
        assert entries[op].unverified_sms == (120,)

    # The SM120 MoE bring-up audit (RTX 6000 Pro, 2026-07-05) cleared the moe
    # maturity marker: every planned (quant mode x backend) family was probed
    # on hardware with verified constructed-method provenance.
    assert entries["moe"].unverified_sms == ()

    # SM120 sparse-round audit (RTX 6000 Pro, 2026-07-06): the framework
    # itself rejects the DSA/GLM-5 sparse family on SM120 (TRTLLM-GEN
    # fmhaRunner "Unsupported architecture"; DeepGEMM attention.hpp:184;
    # sgl-kernel sparse attention SM90a/SM100f-only), the CSA context pool
    # derivation is fail-closed pending an SM120 Torch-indexer workspace
    # policy, and the topk-calib producer is not yet SM120-variant-aware.
    for op in (
        "dsv4_csa_topk_calib",
        "dsv4_paged_mqa_logits_module",
        "glm5_mqa_logits_module",
        "glm5_topk_module",
        "glm5_dsa_attn_module",
    ):
        assert entries[op].unverified_sms == (120,)

    # SM89 bring-up round 2 (L40S, 2026-07-07): stock 0.5.14 cannot run
    # DSV4 on SM89 at all — the server_args DeepseekV4 hook leaves
    # SGLANG_OPT_DEEPGEMM_HC_PRENORM default-True on non-SM120/non-HIP
    # platforms while deep_gemm is unimported (NameError in both mHC
    # directions), the compressed FlashMLA family has no SM89 target
    # (in-kernel CUDA InternalError on every HCA/CSA-generation case), and
    # the CSA context pool derivation is fail-closed below SM90.
    assert entries["dsv4_csa_context_module"].unverified_sms == (89, 120)
    for op in (
        "mhc_module",
        "dsv4_hca_context_module",
        "dsv4_hca_generation_module",
        "dsv4_csa_generation_module",
    ):
        assert entries[op].unverified_sms == (89,)

    # Probed clean on SM120 and SM89 with only classified failure tails
    # (GDN int32 kernel-limit raises, decode grid-Y boundary, top-cell
    # OOMs): no markers.
    assert entries["gdn"].unverified_sms == ()


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
        "MiniMaxAI/MiniMax-M3": {"bfloat16"},
        # MIXED_PRECISION artifact: routed experts NVFP4 gs16 (the MoE axis);
        # its MXFP8 attention/dense/shared-expert side is not a MoE mode.
        "nvidia/MiniMax-M3-NVFP4": {"nvfp4"},
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
            expected_by_artifact["nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"] = {"fp8"}
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
    assert selected_modes("sglang", 109) == {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}
    assert selected_modes("sglang", 110) == {"w4a16_mxfp4"}
    assert selected_modes("sglang", 119) == {"w4a16_mxfp4"}
    assert selected_modes("sglang", 90) == {"w4a16_mxfp4"}
    assert selected_modes("trtllm", 90) == {"w4a16_mxfp4"}
    assert selected_modes("trtllm", 100) == {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}


def test_sglang_dsv4_native_w4a8_mode_is_sm100_interval_gated():
    from collector.case_generator import get_moe_quantization_modes

    def selected_modes(sm_version):
        return {
            mode
            for mode in get_moe_quantization_modes("sglang", sm_version=sm_version)
            if moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", mode)
        }

    assert selected_modes(90) == set()
    assert selected_modes(100) == {"w4a8_mxfp4_mxfp8"}
    assert selected_modes(103) == {"w4a8_mxfp4_mxfp8"}
    assert selected_modes(109) == {"w4a8_mxfp4_mxfp8"}
    assert selected_modes(110) == set()
    assert selected_modes(119) == set()
    assert selected_modes(120) == set()


def test_vllm_dsv4_native_w4a8_mode_is_sm100_interval_gated():
    from collector.case_generator import get_moe_quantization_modes

    def selected_modes(sm_version, *, mxfp4=True):
        return {
            mode
            for mode in get_moe_quantization_modes(
                "vllm",
                sm_version=sm_version,
                runtime_features={"per_block_fp8": True, "nvfp4": True, "mxfp4": mxfp4},
            )
            if moe_model_allows_quantization("vllm", "deepseek-ai/DeepSeek-V4-Flash", mode)
        }

    # The trtllm-gen MXFP4xMXFP8 kernel exists only on the SM100 capability
    # family (10.x); SM90 serves this artifact as Marlin W4A16 and SM110+
    # routes elsewhere, so the label must not be collected there.
    assert selected_modes(90) == set()
    assert selected_modes(100) == {"w4a8_mxfp4_mxfp8"}
    assert selected_modes(103) == {"w4a8_mxfp4_mxfp8"}
    assert selected_modes(109) == {"w4a8_mxfp4_mxfp8"}
    assert selected_modes(110) == set()
    assert selected_modes(119) == set()
    assert selected_modes(120) == set()
    # The mxfp4 runtime-feature gate must hold even inside the SM interval.
    assert selected_modes(100, mxfp4=False) == set()


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

    # Base sweep expansion first (order preserved for checkpoint stability),
    # then model_case_values.gemm rows.
    assert len(cases) == 37296
    assert cases[0] == GemmCommonTestCase(x=32768, n=65536, k=51200)
    assert cases[35741] == GemmCommonTestCase(x=1, n=32, k=32)
    assert cases[-1] == GemmCommonTestCase(x=1, n=1, k=4096)
    assert not any(case.n == 65536 and case.k == 65536 for case in cases)

    assert len(xpu_cases) == 9618
    assert xpu_cases[0] == GemmCommonTestCase(x=8192, n=65536, k=12288)
    assert xpu_cases[9176] == GemmCommonTestCase(x=1, n=32, k=32)
    assert xpu_cases[-1] == GemmCommonTestCase(x=1, n=1, k=4096)
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
    # +117 per new GLM model path: GLM-5.1 (BF16/FP8/NVFP4) and GLM-5.2
    # (BF16/FP8) share GLM-5's MoE dims. nvidia/GLM-5.1-NVFP4 is also
    # registered in moe.yaml base_ops.
    # +114 for Kimi-K3's LatentMoE row (3584/3072, 896x16, w4a16_mxfp4).
    # +198 from Step-3.7-Flash: 99 cases for each physical BF16/FP8 artifact.
    # +117 for the vLLM Nemotron Super FP8 latent-MoE row (1024/2688, 512x22).
    # +114 for MiniMax-M3's MoE row (6144/3072, 128x4).
    # +114 for the nvidia/MiniMax-M3-NVFP4 row (same 6144/3072, 128x4
    # geometry; quant-distinct artifact — NVFP4 routed experts — so it is a
    # separate row, never merged with the BF16 parent).
    assert len(moe_cases) == 5568
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
    # MiniMax-M3's model-owned MoE dims must be present directly (both the
    # BF16 parent and the quant-distinct NVFP4 artifact row), not merely
    # inside the aggregate count above.
    assert any(
        case.model_name == "MiniMaxAI/MiniMax-M3" and case.hidden_size == 6144 and case.inter_size == 3072
        for case in moe_cases
    )
    assert any(
        case.model_name == "nvidia/MiniMax-M3-NVFP4" and case.hidden_size == 6144 and case.inter_size == 3072
        for case in moe_cases
    )
    # Step-3.7-Flash: assert both physical artifact identities, the shape, and
    # the routing contract. MoE loads the model config by model_name, so the
    # BF16 artifact must not alias to the FP8 representative.
    step_cases = [case for case in moe_cases if "Step-3.7-Flash" in case.model_name]
    assert {case.model_name for case in step_cases} == {
        "stepfun-ai/Step-3.7-Flash",
        "stepfun-ai/Step-3.7-Flash-FP8",
    }
    assert sum(case.model_name == "stepfun-ai/Step-3.7-Flash" for case in step_cases) == 99
    assert sum(case.model_name == "stepfun-ai/Step-3.7-Flash-FP8" for case in step_cases) == 99
    assert all(
        case.hidden_size == 4096 and case.inter_size == 1280 and case.topk == 8 and case.num_experts == 288
        for case in step_cases
    )
    # Sigmoid gate + correction bias before top-k, renormalized and scaled by
    # 3.0. The defaults (softmax, no bias, no scaling) would benchmark a
    # different MoE invocation from the one that is actually served.
    assert all(
        case.sglang_moe_scoring_func == "sigmoid"
        and case.sglang_moe_has_correction_bias
        and case.sglang_moe_renormalize
        and case.sglang_moe_routed_scaling_factor == 3.0
        for case in step_cases
    )

    # Kimi-K3 declares the native 96-head MLA profile (DeepSeek geometry),
    # expanding the MLA spec grids.
    assert len(get_context_mla_case_specs()) == 330
    assert len(get_generation_mla_case_specs()) == 543
    mamba_cases = get_common_mamba2_test_cases()
    assert len(mamba_cases) == 12
    assert {case.model_name for case in mamba_cases} >= {"MAMBA2_GENERIC_4K", "MAMBA2_GENERIC_1K"}
    assert len(get_common_gdn_test_cases()) == 74
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

    large_generation_spec = SimpleNamespace(
        model_name="large-generation-regression",
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        num_heads=128,
        batch_size=1024,
        input_len=4096,
        is_context_phase=False,
    )
    assert sglang_adapter([large_generation_spec], dtype_list=("bf16",), tp_sizes=(1,), backend="fa3")


def test_kimi_mla_plan_includes_generation_bmm_helpers():
    required_ops = {"mla_context", "mla_generation", "mla_bmm_gen_pre", "mla_bmm_gen_post"}
    for backend in ("sglang", "trtllm"):
        plan = build_collection_case_plan(backend=backend, model_path="moonshotai/Kimi-K2.5")
        assert required_ops <= plan.selected_ops


def test_kimi_k3_moe_is_planned_per_framework_and_never_for_trtllm():
    # K3 has no trtllm serving lane. moe activation is framework-specific
    # (sglang/vllm), so a K3-scoped trtllm run plans NO moe at all — a
    # planned-op zero-case expansion with no logged drop is structurally
    # impossible (case_authoring.md; review 2026-08-04).
    for backend in ("sglang", "vllm"):
        plan = build_collection_case_plan(backend=backend, model_path="moonshotai/Kimi-K3")
        assert "moe" in plan.selected_ops, backend
    trtllm_plan = build_collection_case_plan(backend="trtllm", model_path="moonshotai/Kimi-K3")
    assert "moe" not in trtllm_plan.selected_ops

    # Cross-model trtllm sweeps (getter runs with no model filter) still see
    # the K3 moe row: the declared empty trtllm allowlist rejects EVERY mode,
    # and the trtllm getter logs the fully-dropped model instead of silently
    # expanding to zero.
    from collector.case_generator import get_moe_quantization_modes, moe_model_allows_quantization

    modes = get_moe_quantization_modes(
        "trtllm",
        sm_version=100,
        runtime_features={"per_block_fp8": True, "nvfp4": True, "mxfp4": True},
    )
    assert modes  # the sweep itself is non-empty
    for mode in modes:
        assert not moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K3", mode), mode


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
    ) == ["bfloat16", "int4_wo", "fp8", "fp8_block", "nvfp4", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"]
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

    # 600/672 since the Kimi-K3 96-head family (96/48/24/12) joined the
    # base head_counts grid alongside the DeepSeek 128-family (2026-08-02).
    assert len(pre_cases) == 600
    assert len(post_cases) == 672
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
    trtllm_specs = get_mla_module_model_specs(backend="trtllm")
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
        "zai-org/GLM-5.1",
        "zai-org/GLM-5.1-FP8",
        "nvidia/GLM-5.1-NVFP4",
        "zai-org/GLM-5.2",
        "zai-org/GLM-5.2-FP8",
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
        ("msa", "MiniMaxAI/MiniMax-M3", "MiniMaxM3ForCausalLM"),
    }
    assert trtllm_specs == vllm_specs


def test_vllm_msa_module_precision_combos_declare_fp8_kv_cache():
    """The vLLM MSA collector declares its precision combos in-file (not via
    mla_module.yaml). vLLM 0.24.0's M3 sparse backend accepts an fp8 (e4m3)
    main KV cache on every platform — Triton in-kernel dequant off the SM100
    family, MSA attend on it (supported_kv_cache_dtypes common/
    sparse_attention.py:56-62, view :352, select_main_impl_cls :391-422
    @v0.24.0) — so every gemm tier pairs with both KV dtypes; only the gemm
    axis is SM-gated."""
    source_path = REPO_ROOT / "collector/vllm/collect_msa_module.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_get_precision_combos"
    )

    def combos(sm, phase):
        namespace = {"get_sm_version": lambda: sm}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), "exec"), namespace)
        return set(namespace["_get_precision_combos"](phase))

    sm90_expected = {
        ("bfloat16", "bfloat16", "bfloat16"),
        ("bfloat16", "fp8", "bfloat16"),
        ("bfloat16", "bfloat16", "fp8_block"),
        ("bfloat16", "fp8", "fp8_block"),
    }
    for phase in ("context", "generation"):
        assert combos(90, phase) == sm90_expected
        assert combos(100, phase) == sm90_expected | {
            ("bfloat16", "bfloat16", "nvfp4"),
            ("bfloat16", "fp8", "nvfp4"),
        }


def test_mla_module_targeted_artifacts_keep_requested_checkpoint(monkeypatch):
    from collector.case_generator import get_mla_module_model_specs

    for backend, model_path, attention_type, architecture in (
        ("trtllm", "nvidia/DeepSeek-V3.1-NVFP4", "mla", "DeepseekV3ForCausalLM"),
        ("trtllm", "nvidia/GLM-5-NVFP4", "dsa", "GlmMoeDsaForCausalLM"),
        ("vllm", "nvidia/DeepSeek-V3.1-NVFP4", "mla", "DeepseekV3ForCausalLM"),
        ("vllm", "moonshotai/Kimi-K2-Instruct", "mla", "DeepseekV3ForCausalLM"),
        ("vllm", "moonshotai/Kimi-K2.5", "mla", "KimiK25ForConditionalGeneration"),
        ("vllm", "nvidia/Kimi-K2.5-NVFP4", "mla", "KimiK25ForConditionalGeneration"),
        ("vllm", "nvidia/GLM-5-NVFP4", "dsa", "GlmMoeDsaForCausalLM"),
    ):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        specs = get_mla_module_model_specs(attention_type=attention_type, backend=backend)
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
    assert "moe_ep" not in payload["ops"]


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
    # TRT-LLM 1.3.0rc20 serves the full K2.5 VLM including MoonViT3d, so the
    # trtllm plan carries the vision encoder_attention profile like vLLM.
    assert build_collection_case_plan(backend="trtllm", model_path=kimi_path).ops == [
        "encoder_attention",
        "gemm",
        "mla_bmm_gen_post",
        "mla_bmm_gen_pre",
        "mla_context",
        "mla_generation",
        "moe",
        "moe_ep",
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
    assert "wideep_mla_context" not in plan.selected_ops
    assert "dsv4_csa_context_module" in plan.selected_ops
    assert "gdn" in plan.selected_ops


def test_full_mode_ops_are_a_union_of_model_plan_ops():
    for backend in ("sglang", "trtllm"):
        full_plan = build_collection_case_plan(backend=backend, full=True)
        for model_path in ("deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2.5", "Qwen/Qwen3-32B"):
            model_plan = build_collection_case_plan(backend=backend, model_path=model_path)
            assert model_plan.selected_ops <= full_plan.selected_ops, f"{backend}/{model_path}"


def test_mla_module_metadata_canonicalizes_consumer_keyed_backends():
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
                apply_model_filter=backend in {"trtllm", "vllm"},
            )
        }

    assert paths("vllm") == {"deepseek-ai/DeepSeek-V3"}
    assert paths("sglang") == original_artifacts
    assert paths("trtllm") == {"deepseek-ai/DeepSeek-V3"}


def test_trtllm_mla_module_getter_requests_backend_canonicalization():
    from types import SimpleNamespace

    source_path = REPO_ROOT / "collector/trtllm/collect_mla_module.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_module_test_cases"
    )
    calls = []

    def get_model_specs(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(model_path="deepseek-ai/DeepSeek-V3")]

    namespace = {
        "get_context_test_cases": lambda _attention_type: [[128, 1, 8, "bfloat16", "bfloat16", "bfloat16"]],
        "get_generation_test_cases": lambda _attention_type: [],
        "get_mla_module_model_specs": get_model_specs,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)

    cases = namespace["_build_module_test_cases"]("mla", "context")

    assert calls == [{"attention_type": "mla", "backend": "trtllm"}]
    assert cases == [[128, 1, 8, "bfloat16", "bfloat16", "bfloat16", "deepseek-ai/DeepSeek-V3", "mla"]]


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
        "nvidia/MiniMax-M3-NVFP4": "nvidia/MiniMax-M3-NVFP4",
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    }

    for model_path, expected_representative in expected_representatives.items():
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        cases = get_common_moe_test_cases()
        assert cases and {case.model_name for case in cases} == {expected_representative}


def test_nemotron_super_fp8_vllm_moe_case_covers_missing_consumer_key(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    model_path = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    cases = get_common_moe_test_cases(backend="vllm")

    assert cases
    assert {case.model_name for case in cases} == {model_path}
    assert any(
        case.hidden_size == 1024
        and case.inter_size == 2688
        and case.topk == 22
        and case.num_experts == 512
        and case.tp == 1
        and case.ep == 4
        and case.token_expert_distribution == "power_law"
        and case.power_law_alpha == 1.01
        for case in cases
    )
    assert moe_model_allows_quantization("vllm", model_path, "fp8")
    assert not moe_model_allows_quantization("vllm", model_path, "bfloat16")

    config_path = REPO_ROOT / "src/aiconfigurator/model_configs" / f"{model_path.replace('/', '--')}_config.json"
    assert config_path.is_file()


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


def test_step3p7_plans_correlated_attention_topologies(monkeypatch):
    """Step-3.7 must plan (64 Q, window 0) and (96 Q, window 512), not their cross-product.

    The pinned vLLM Step3p5 block substitutes 96 query heads on
    sliding_attention layers while global layers keep 64. A single row
    cross-producting num_attention_heads=64 with window_sizes [0, 512] plans
    64-head SWA cases that never run and omits the 96-head SWA cases that do --
    without changing the total case count, so an aggregate assertion misses it.
    """
    from collector.case_generator import get_attention_context_shape_sweeps

    # Attention is shape-only, so both artifacts intentionally share the rows.
    for model_path in ("stepfun-ai/Step-3.7-Flash-FP8", "stepfun-ai/Step-3.7-Flash"):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
        configs = {
            (config.num_heads, config.num_kv_heads, config.head_dim, config.window_size)
            for sweep in get_attention_context_shape_sweeps("vllm")
            for config in get_attention_head_configs(sweep, phase="context")
        }

        assert configs == {
            # global attention: 64 Q / 8 KV, sharded over TP 1/2/4/8
            (64, 8, 128, 0),
            (32, 4, 128, 0),
            (16, 2, 128, 0),
            (8, 1, 128, 0),
            # sliding attention (window 512): 96 Q / 8 KV
            (96, 8, 128, 512),
            (48, 4, 128, 512),
            (24, 2, 128, 512),
            (12, 1, 128, 512),
        }, model_path


def test_qwen35_gemm_model_rows_add_exact_below_grid_widths():
    """model_case_values.gemm supplies exact widths under the base feature grid
    (scalar expert gate n=1, GDN b/a projections); token density comes from the
    base sweeps and cases dedupe on the physical (x, n, k) tuple."""
    specs = get_gemm_case_specs()
    shapes = {(case.n, case.k) for case in specs}
    assert {(1, 2048), (8, 2048), (16, 4096), (12, 5120)} <= shapes

    base_tokens = set()
    for sweep in load_yaml_file(BASE_OP_CASES_DIR / "gemm.yaml")["all_frameworks_op_cases"]["gemm"]["cases"]:
        base_tokens.update(int(token) for token in sweep["token_counts"])
    assert {case.x for case in specs if (case.n, case.k) == (1, 2048)} == base_tokens

    assert len(specs) == len({(case.x, case.n, case.k) for case in specs})


def test_vllm_msa_persist_row_raises_when_log_perf_fails():
    """A false return from log_perf (lock exhaustion / write failure) must
    fail the case: a worker that returns normally lets the checkpoint advance
    with no row persisted, silently shrinking the dataset. Mirrors the
    TRT-LLM/SGLang collectors' behavior."""
    source_path = REPO_ROOT / "collector/vllm/collect_msa_module.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_persist_msa_row")

    def run(log_perf_result):
        calls = {}

        def fake_log_perf(**kwargs):
            calls.update(kwargs)
            return log_perf_result

        namespace = {"log_perf": fake_log_perf}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), "exec"), namespace)
        namespace["_persist_msa_row"](
            item={"latency": "1.0"},
            vllm_version="0.24.0",
            device_name="test-device",
            op_name="msa_generation_module",
            kernel_source="MiniMaxM3SparseTritonImpl",
            perf_filename="msa_generation_module_perf.txt",
            power_stats=None,
        )
        return calls

    calls = run(True)
    assert calls["perf_filename"] == "msa_generation_module_perf.txt"

    with pytest.raises(RuntimeError, match="failed to persist MSA row"):
        run(False)
