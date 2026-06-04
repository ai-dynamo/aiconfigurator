# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from itertools import pairwise
from pathlib import Path

from collector.case_generator import moe_model_allows_quantization
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


def test_model_case_plan_merges_base_op_and_framework_specific_ops():
    plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")

    assert plan.model_architecture == "DeepseekV3ForCausalLM"
    assert plan.model_cases_paths == [default_architecture_cases_path("DeepseekV3ForCausalLM")]
    assert "gemm" in plan.op_cases
    assert "attention_context" in plan.op_cases
    assert "moe" in plan.op_cases
    assert "mla_context" in plan.op_cases
    assert "wideep_mla_context" in plan.op_cases
    assert "wideep_moe" in plan.op_cases
    assert "trtllm_moe_wideep" not in plan.op_cases


def test_base_gemm_cases_are_readable_shape_specs():
    plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")
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
    assert moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", "w4a8_mxfp4_mxfp8")
    assert moe_model_allows_quantization("sglang", "deepseek-ai/DeepSeek-V4-Flash", "bfloat16")
    assert not moe_model_allows_quantization("sglang", "Qwen/Qwen3-235B-A22B", "w4a8_mxfp4_mxfp8")

    assert moe_model_allows_quantization("sglang", "openai/gpt-oss-120b", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("sglang", "openai/gpt-oss-120b", "bfloat16")

    assert moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K2.5", "w4a16_mxfp4")
    assert moe_model_allows_quantization("trtllm", "moonshotai/Kimi-K2.5", "bfloat16")
    assert not moe_model_allows_quantization("trtllm", "Qwen/Qwen3-235B-A22B", "w4a16_mxfp4")
    assert not moe_model_allows_quantization("trtllm", "openai/gpt-oss-20b", "fp8")


def test_attention_shape_specs_are_yaml_backed_with_backend_overrides():
    from collector.case_generator import get_attention_context_shape_sweeps, get_attention_generation_shape_sweeps

    sglang_context = get_attention_context_shape_sweeps("sglang")[0]
    trtllm_context = get_attention_context_shape_sweeps("trtllm")[0]
    vllm_context = get_attention_context_shape_sweeps("vllm")[0]
    vllm_xpu_context = get_attention_context_shape_sweeps("vllm_xpu")[0]
    vllm_generation = get_attention_generation_shape_sweeps("vllm")[0]

    assert sglang_context["head_dims"] == [128, 192, 256]
    assert trtllm_context["head_dims"] == [64, 128, 192, 256]
    assert trtllm_context["query_head_counts"][:6] == [1, 2, 3, 4, 5, 6]
    assert vllm_context["head_dims"] == [64, 128, 192, 256, 512]
    assert vllm_context["query_head_counts"][-1] == 64
    assert trtllm_context["window_sizes"] == [0, 1024]
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
    assert len(moe_cases) == 4548
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
    assert len(get_context_mla_case_specs()) == 550
    assert len(get_generation_mla_case_specs()) == 885
    assert len(get_common_mamba2_test_cases()) == 8
    assert len(get_common_gdn_test_cases()) == 16
    assert len(get_common_mhc_test_cases()) == 8


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
        runtime_version="0.19.0",
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
    wideep_specs = get_mla_module_model_specs(attention_type="mla", wideep_mla=True, apply_model_filter=False)

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
    assert {spec.model_path for spec in wideep_specs} == {
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3",
        "nvidia/DeepSeek-V3.1-NVFP4",
    }


def test_model_cases_path_can_infer_model_path():
    model_cases_path = default_architecture_cases_path("DeepseekV4ForCausalLM")

    plan = build_collection_case_plan(backend="sglang", model_cases_path=str(model_cases_path))

    assert plan.model_path == "sgl-project/DeepSeek-V4-Flash-FP8"
    assert plan.model_architecture == "DeepseekV4ForCausalLM"
    assert "dsv4_csa_context_module" in plan.op_cases
    assert "mhc_module" in plan.op_cases


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


def test_full_mode_aggregates_all_model_case_files():
    plan = build_collection_case_plan(backend="sglang", full=True)

    assert plan.model_path is None
    assert len(plan.model_cases_paths) >= 18
    assert "wideep_mla_context" in plan.op_cases
    assert "dsv4_csa_context_module" in plan.op_cases
    assert "gdn" in plan.op_cases


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
    assert {case.model_name for case in cases} == {"Qwen/Qwen3-235B-A22B-FP8"}


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

    expected = expected_failure_for_test_case(
        case,
        plan=plan.op_cases["moe"],
        full_module_name="trtllm.moe",
        run_func_name="run_moe_torch",
        runtime_version="1.3.0rc10",
    )

    assert expected == {
        "case_id": create_test_case_id(case, "run_moe_torch", "trtllm.moe"),
        "source": "sm_exception",
        "selector": "rule",
        "reason_type": "framework_version_unsupported",
        "reason": "TRT-LLM 1.3.0rc5/rc10 TRTLLMGenFusedMoE rejects GPT-OSS MXFP4 paths on SM120.",
    }
    assert (
        expected_failure_for_test_case(
            case,
            plan=plan.op_cases["moe"],
            full_module_name="trtllm.moe",
            run_func_name="run_moe_torch",
            runtime_version="1.3.0rc4",
        )
        is None
    )


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

    expected = expected_failure_for_test_case(
        case,
        plan=plan.op_cases["moe"],
        full_module_name="trtllm.moe",
        run_func_name="run_moe_torch",
        runtime_version="1.3.0rc10",
    )

    assert expected == {
        "case_id": create_test_case_id(case, "run_moe_torch", "trtllm.moe"),
        "source": "sm_exception",
        "selector": "rule",
        "reason_type": "framework_version_unsupported",
        "reason": (
            "TRT-LLM 1.3.0rc10 SM100 CutlassFusedMoE rejects plain W4A16/int4_wo "
            "in create_moe with ValueError Unsupported quantization mode [1]."
        ),
    }
    assert (
        expected_failure_for_test_case(
            case,
            plan=plan.op_cases["moe"],
            full_module_name="trtllm.moe",
            run_func_name="run_moe_torch",
            runtime_version="1.3.0rc9",
        )
        is None
    )


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
