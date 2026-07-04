# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_moe.py"

MOE_RUNTIME_DEFAULTS = {
    "sglang_moe_backends": {},
    "sglang_moe_activation": "silu",
    "sglang_moe_is_gated": True,
    "sglang_moe_has_bias": False,
    "sglang_moe_gemm1_alpha": None,
    "sglang_moe_gemm1_clamp_limit": None,
    "sglang_moe_swiglu_limit": None,
    "sglang_moe_scoring_func": "softmax",
    "sglang_moe_routing_method_type": None,
    "sglang_moe_routed_scaling_factor": None,
    "sglang_moe_renormalize": True,
    "sglang_moe_has_correction_bias": False,
    "sglang_moe_num_expert_group": None,
    "sglang_moe_topk_group": None,
    "sglang_moe_apply_router_weight_on_input": False,
}


def _load_functions(*names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded


def _gptoss_case(*, tp: int, ep: int):
    return SimpleNamespace(
        **(
            MOE_RUNTIME_DEFAULTS
            | {
                "sglang_moe_has_bias": True,
                "sglang_moe_gemm1_alpha": 1.702,
                "sglang_moe_gemm1_clamp_limit": 7.0,
            }
        ),
        num_tokens_list=[128],
        hidden_size=2880,
        inter_size=2880,
        topk=4,
        num_experts=128,
        tp=tp,
        ep=ep,
        model_name="openai/gpt-oss-120b",
        token_expert_distribution="balanced",
        power_law_alpha=None,
        architecture="GptOssForCausalLM",
    )


def _populate_gptoss_cases(
    cases,
    *,
    sm_version=100,
    allowed_mode="w4a8_mxfp4_mxfp8",
    module_config=None,
    resolved_backend=None,
):
    allowed_modes = {allowed_mode} if isinstance(allowed_mode, str) else set(allowed_mode)
    loaded = _load_functions(
        "get_moe_test_cases",
        namespace={
            "itertools": itertools,
            "get_sm_version": lambda: sm_version,
            "get_common_moe_test_cases": lambda: cases,
            "moe_model_allows_quantization": (lambda _backend, _model, mode: mode in allowed_modes),
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: module_config or {},
            "get_sglang_moe_backend": lambda _case, mode, sm: (
                resolved_backend
                or (
                    "marlin"
                    if sm == 90 and mode == "int4_wo"
                    else "flashinfer_trtllm"
                    if mode == "nvfp4" and sm in {100, 103}
                    else "flashinfer_mxfp4"
                    if mode in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}
                    else "triton"
                )
            ),
            "_SM120_NEMOTRON_NVFP4_MODELS": set(),
        },
    )
    return loaded["get_moe_test_cases"]()


def _glm5_case(model_name: str):
    return SimpleNamespace(
        **(
            MOE_RUNTIME_DEFAULTS
            | {
                "sglang_moe_scoring_func": "sigmoid",
                "sglang_moe_routing_method_type": "DeepSeekV3",
                "sglang_moe_routed_scaling_factor": 2.5,
                "sglang_moe_has_correction_bias": True,
                "sglang_moe_num_expert_group": 1,
                "sglang_moe_topk_group": 1,
            }
        ),
        num_tokens_list=[128],
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        tp=1,
        ep=1,
        model_name=model_name,
        token_expert_distribution="balanced",
        power_law_alpha=None,
        architecture="GlmMoeDsaForCausalLM",
    )


def test_glm52_nvfp4_replaces_consumer_equivalent_glm5_moe_cases():
    populated = _populate_gptoss_cases(
        [
            _glm5_case("nvidia/GLM-5-NVFP4"),
            _glm5_case("nvidia/GLM-5.2-NVFP4"),
        ],
        allowed_mode="nvfp4",
    )

    assert populated
    assert {case[8] for case in populated} == {"nvidia/GLM-5.2-NVFP4"}


def test_kimi_int4_population_restores_all_hopper_ep_slices():
    from collector.case_generator import get_common_moe_test_cases

    common_cases = [case for case in get_common_moe_test_cases() if case.model_name == "moonshotai/Kimi-K2.5"]
    populated = _populate_gptoss_cases(
        common_cases,
        sm_version=90,
        allowed_mode="int4_wo",
        module_config={"group_size": 32},
    )

    assert len(populated) == 3078
    assert sum(case[7] == 64 for case in populated) == 243
    assert sum(case[7] == 128 for case in populated) == 162
    assert not _populate_gptoss_cases(
        common_cases,
        sm_version=100,
        allowed_mode="int4_wo",
        module_config={"group_size": 32},
    )


def test_fp8_block_population_uses_tp_local_intermediate_alignment():
    invalid = _glm5_case("example/fp8")
    invalid.hidden_size = 128
    invalid.inter_size = 384
    invalid.tp = 2
    invalid.num_experts = 8
    invalid.topk = 2
    valid = SimpleNamespace(**vars(invalid))
    valid.inter_size = 512

    assert not _populate_gptoss_cases([invalid], sm_version=90, allowed_mode="fp8_block")
    assert len(_populate_gptoss_cases([valid], sm_version=90, allowed_mode="fp8_block")) == 1


def test_gemma_gelu_population_restores_platform_vector_alignment():
    invalid = _gptoss_case(tp=16, ep=1)
    invalid.model_name = "google/gemma-4-26B-A4B"
    invalid.hidden_size = 2816
    invalid.inter_size = 704
    invalid.topk = 8
    invalid.sglang_moe_activation = "gelu"
    valid = SimpleNamespace(**vars(invalid))
    valid.tp = 8

    assert not _populate_gptoss_cases([invalid], sm_version=90, allowed_mode="bfloat16")
    assert len(_populate_gptoss_cases([valid], sm_version=90, allowed_mode="bfloat16")) == 1

    blackwell_valid = SimpleNamespace(**vars(invalid))
    blackwell_valid.tp = 4
    assert not _populate_gptoss_cases([valid], sm_version=100, allowed_mode="bfloat16")
    assert len(_populate_gptoss_cases([blackwell_valid], sm_version=100, allowed_mode="bfloat16")) == 1


def test_bfloat16_flashinfer_cutlass_population_uses_tp_local_inter_alignment():
    invalid = _gptoss_case(tp=16, ep=1)
    invalid.model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    invalid.hidden_size = 2688
    invalid.inter_size = 1856
    invalid.topk = 6
    invalid.sglang_moe_activation = "relu2"
    invalid.sglang_moe_is_gated = False
    valid = SimpleNamespace(**vars(invalid))
    valid.tp = 8
    ultra = SimpleNamespace(**vars(invalid))
    ultra.model_name = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
    ultra.hidden_size = 8192
    ultra.inter_size = 5120

    assert not _populate_gptoss_cases(
        [invalid], sm_version=90, allowed_mode="bfloat16", resolved_backend="flashinfer_cutlass"
    )
    assert (
        len(
            _populate_gptoss_cases(
                [valid, ultra], sm_version=90, allowed_mode="bfloat16", resolved_backend="flashinfer_cutlass"
            )
        )
        == 2
    )
    assert (
        len(
            _populate_gptoss_cases(
                [invalid], sm_version=100, allowed_mode="bfloat16", resolved_backend="flashinfer_trtllm"
            )
        )
        == 1
    )


def test_dsv4_w4a16_population_uses_sm90_fp4_expert_alignment():
    invalid = _gptoss_case(tp=16, ep=1)
    invalid.model_name = "deepseek-ai/DeepSeek-V4-Pro"
    invalid.architecture = "DeepseekV4ForCausalLM"
    invalid.hidden_size = 7168
    invalid.inter_size = 3072
    invalid.topk = 6
    invalid.num_experts = 384
    valid = SimpleNamespace(**vars(invalid))
    valid.tp = 8

    assert not _populate_gptoss_cases([invalid], sm_version=90, allowed_mode="w4a16_mxfp4")
    populated = _populate_gptoss_cases([valid], sm_version=90, allowed_mode="w4a16_mxfp4")
    assert len(populated) == 1
    assert populated[0][-1] is True

    blackwell = _populate_gptoss_cases([valid], sm_version=100, allowed_mode="w4a8_mxfp4_mxfp8")
    assert len(blackwell) == 1
    assert blackwell[0][0] == "w4a8_mxfp4_mxfp8"


def test_sm90_population_excludes_nvfp4_instead_of_using_marlin():
    assert not _populate_gptoss_cases(
        [_glm5_case("nvidia/GLM-5.2-NVFP4")],
        sm_version=90,
        allowed_mode="nvfp4",
    )


def test_moe_population_deduplicates_equal_persisted_keys_and_rejects_semantic_conflicts():
    first = _gptoss_case(tp=4, ep=8)
    duplicate = SimpleNamespace(**vars(first))
    duplicate.model_name = "example/equivalent-gptoss"

    assert len(_populate_gptoss_cases([first, duplicate])) == 1

    duplicate.sglang_moe_activation = "gelu"
    with pytest.raises(ValueError, match="share one perf DB key but require different execution semantics"):
        _populate_gptoss_cases([first, duplicate])


def test_case_generator_preserves_representative_sglang_moe_runtime_contracts():
    from collector.case_generator import get_common_moe_test_cases, get_sglang_moe_backend

    cases = get_common_moe_test_cases()

    def model_case(model_name):
        return next(case for case in cases if case.model_name == model_name)

    deepseek = model_case("deepseek-ai/DeepSeek-V3")
    assert (deepseek.sglang_moe_scoring_func, deepseek.sglang_moe_routed_scaling_factor) == ("sigmoid", 2.5)
    assert get_sglang_moe_backend(deepseek, "fp8_block", 100) == "flashinfer_trtllm"

    gemma = model_case("google/gemma-4-26B-A4B")
    assert gemma.sglang_moe_activation == "gelu"

    gpt_oss = model_case("openai/gpt-oss-120b")
    assert (gpt_oss.sglang_moe_has_bias, gpt_oss.sglang_moe_gemm1_alpha, gpt_oss.sglang_moe_gemm1_clamp_limit) == (
        True,
        1.702,
        7.0,
    )

    kimi = model_case("moonshotai/Kimi-K2.5")
    assert get_sglang_moe_backend(kimi, "int4_wo", 90) == "marlin"
    assert (
        kimi.sglang_moe_scoring_func,
        kimi.sglang_moe_routing_method_type,
        kimi.sglang_moe_routed_scaling_factor,
        kimi.sglang_moe_has_correction_bias,
        kimi.sglang_moe_num_expert_group,
        kimi.sglang_moe_topk_group,
    ) == ("sigmoid", "DeepSeekV3", 2.827, True, 1, 1)

    dsv4 = model_case("deepseek-ai/DeepSeek-V4-Pro")
    assert (dsv4.sglang_moe_scoring_func, dsv4.sglang_moe_has_correction_bias) == ("sqrtsoftplus", True)

    llama4 = model_case("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    assert (
        llama4.sglang_moe_scoring_func,
        llama4.sglang_moe_routing_method_type,
        llama4.sglang_moe_renormalize,
        llama4.sglang_moe_apply_router_weight_on_input,
    ) == ("sigmoid", "Llama4", False, True)

    nemotron = model_case("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
    assert (nemotron.sglang_moe_activation, nemotron.sglang_moe_is_gated) == ("relu2", False)
    assert get_sglang_moe_backend(nemotron, "nvfp4", 100) == "flashinfer_trtllm"

    source = SOURCE_PATH.read_text()
    assert "use_grouped_topk=num_expert_group is not None and topk_group is not None" in source
    assert "num_expert_group=num_expert_group" in source


@pytest.mark.parametrize(("tp", "ep"), [(4, 8), (32, 1), (32, 8)])
def test_gptoss_mxfp4_population_retains_tp_and_ep_buckets(tp, ep):
    populated = _populate_gptoss_cases([_gptoss_case(tp=tp, ep=ep)])

    assert len(populated) == 1
    assert populated[0][0] == "w4a8_mxfp4_mxfp8"
    assert populated[0][6:8] == [tp, ep]
    assert populated[0][12] == "flashinfer_mxfp4"
    assert populated[0][15:18] == [True, 1.702, 7.0]
    assert populated[0][-1] is False


@pytest.mark.parametrize(
    ("mode", "sm_version", "expected"),
    [
        ("nvfp4", 100, "flashinfer_trtllm"),
        ("nvfp4", 103, "flashinfer_trtllm"),
        ("nvfp4", 120, "flashinfer_cutlass"),
        ("int4_wo", 90, "marlin"),
        ("w4a16_mxfp4", 90, "flashinfer_mxfp4"),
        ("w4a8_mxfp4_mxfp8", 100, "flashinfer_mxfp4"),
    ],
)
def test_yaml_backend_map_matches_sglang_0514(mode, sm_version, expected):
    from collector.case_generator import get_sglang_moe_backend

    assert get_sglang_moe_backend(SimpleNamespace(sglang_moe_backends={}), mode, sm_version) == expected


@pytest.mark.parametrize("mode", ["nvfp4", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"])
def test_fp4_modes_reject_marlin_backend(mode):
    from collector.case_generator import get_sglang_moe_backend

    test_case = SimpleNamespace(sglang_moe_backends={mode: {90: "marlin"}})
    with pytest.raises(ValueError, match="Marlin is only valid for int4_wo"):
        get_sglang_moe_backend(test_case, mode, 90)


def test_sm90_nvfp4_has_no_backend():
    from collector.case_generator import get_sglang_moe_backend

    with pytest.raises(ValueError, match="No SGLang MoE backend"):
        get_sglang_moe_backend(SimpleNamespace(sglang_moe_backends={}), "nvfp4", 90)


@pytest.mark.parametrize("persisted", [True, False])
@pytest.mark.parametrize(
    ("moe_type", "moe_backend", "model_name", "sm_version", "kernel_source"),
    [
        ("int4_wo", "marlin", "moonshotai/Kimi-K2.5", 90, "sglang_marlin_moe"),
        ("w4a16_mxfp4", "flashinfer_mxfp4", "openai/gpt-oss-120b", 90, "sglang_flashinfer_mxfp4_moe"),
        (
            "w4a8_mxfp4_mxfp8",
            "flashinfer_mxfp4",
            "openai/gpt-oss-120b",
            100,
            "sglang_flashinfer_mxfp4_moe",
        ),
    ],
)
def test_quantized_moe_uses_framework_path_and_fails_closed(
    persisted,
    moe_type,
    moe_backend,
    model_name,
    sm_version,
    kernel_source,
):
    framework_calls = []
    logged = []
    fake_torch = SimpleNamespace(
        bfloat16=object(),
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(
            set_device=lambda _device: None,
            empty_cache=lambda: None,
            memory_allocated=lambda _device: 26,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=100),
            get_device_name=lambda _device: "H200",
        ),
    )

    def framework_benchmark(**kwargs):
        framework_calls.append(kwargs)
        return 1.25, {"power": 100.0}, moe_backend

    run = _load_functions(
        "run_moe_torch",
        namespace={
            "torch": fake_torch,
            "_benchmark_framework_quantized_moe": framework_benchmark,
            "benchmark": lambda *_args, **_kwargs: pytest.fail("quantized MoE must not use the raw benchmark"),
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {"group_size": 32},
            "get_sm_version": lambda: sm_version,
            "_fmoe_kernels_mod": SimpleNamespace(_B_DESC_CACHE=SimpleNamespace(clear=lambda: None)),
            "gc": SimpleNamespace(collect=lambda: None),
            "log_perf": lambda **kwargs: logged.append(kwargs) or persisted,
            "pkg_resources": SimpleNamespace(get_distribution=lambda _name: SimpleNamespace(version="0.5.14")),
            "EXIT_CODE_RESTART": 10,
        },
    )["run_moe_torch"]

    args = (
        moe_type,
        128,
        7168,
        2048,
        8,
        384,
        1,
        1,
        model_name,
    )
    if persisted:
        assert (
            run(
                *args,
                distributed="balanced",
                moe_backend=moe_backend,
                perf_filename="moe.csv",
            )
            == 10
        )
    else:
        with pytest.raises(RuntimeError, match="Failed to persist SGLang MoE performance row"):
            run(
                *args,
                distributed="balanced",
                moe_backend=moe_backend,
                perf_filename="moe.csv",
            )

    assert framework_calls[0]["model_name"] == model_name
    assert logged[0]["kernel_source"] == kernel_source


def test_raw_moe_case_cleans_gpu_state_and_fails_closed():
    cleared = []
    collected = []
    emptied = []
    fake_torch = SimpleNamespace(
        bfloat16=object(),
        device=lambda value: value,
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(
            set_device=lambda _device: None,
            empty_cache=lambda: emptied.append(True),
            memory_allocated=lambda _device: 0,
            get_device_properties=lambda _device: SimpleNamespace(total_memory=100),
            get_device_name=lambda _device: "H200",
        ),
    )
    run = _load_functions(
        "run_moe_torch",
        namespace={
            "torch": fake_torch,
            "benchmark": lambda *_args, **_kwargs: (1.25, {"power": 100.0}),
            "build_rank0_workloads": lambda **_kwargs: pytest.fail("EP=1 must not build rank-local workloads"),
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {},
            "_benchmark_framework_quantized_moe": lambda **_kwargs: pytest.fail("BF16 must use the raw benchmark"),
            "_fmoe_kernels_mod": SimpleNamespace(_B_DESC_CACHE=SimpleNamespace(clear=lambda: cleared.append(True))),
            "gc": SimpleNamespace(collect=lambda: collected.append(True)),
            "get_sm_version": lambda: 90,
            "log_perf": lambda **_kwargs: False,
            "pkg_resources": SimpleNamespace(get_distribution=lambda _name: SimpleNamespace(version="0.5.14")),
            "EXIT_CODE_RESTART": 10,
        },
    )["run_moe_torch"]

    with pytest.raises(RuntimeError, match="Failed to persist SGLang MoE performance row"):
        run(
            "bfloat16",
            128,
            4096,
            14336,
            2,
            8,
            1,
            1,
            "mistralai/Mixtral-8x7B-v0.1",
            distributed="balanced",
            perf_filename="moe.csv",
        )

    assert cleared == [True]
    assert collected == [True]
    assert emptied == [True]


@pytest.mark.parametrize(
    ("moe_type", "hidden_size", "inter_size", "tp", "is_gated", "error"),
    [
        ("fp8_block", 128, 384, 2, True, "fp8_block.*local_inter_size"),
    ],
)
def test_runtime_rejects_misaligned_quantized_cases(moe_type, hidden_size, inter_size, tp, is_gated, error):
    fake_torch = SimpleNamespace(
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(set_device=lambda _device: None),
    )
    run = _load_functions(
        "run_moe_torch",
        namespace={"torch": fake_torch, "get_sm_version": lambda: 90},
    )["run_moe_torch"]

    with pytest.raises(ValueError, match=error):
        run(
            moe_type,
            128,
            hidden_size,
            inter_size,
            2,
            8,
            tp,
            1,
            "example/model",
            moe_backend="triton" if moe_type == "fp8_block" else "marlin",
            is_gated=is_gated,
            perf_filename="moe.csv",
        )


@pytest.mark.parametrize("moe_type", ["nvfp4", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"])
def test_runtime_rejects_fp4_modes_on_marlin(moe_type):
    fake_torch = SimpleNamespace(
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(set_device=lambda _device: None),
    )
    run = _load_functions(
        "run_moe_torch",
        namespace={"torch": fake_torch, "get_sm_version": lambda: 90},
    )["run_moe_torch"]

    with pytest.raises(ValueError, match="Marlin is only valid for int4_wo"):
        run(
            moe_type,
            128,
            128,
            128,
            2,
            8,
            1,
            1,
            "example/model",
            moe_backend="marlin",
            perf_filename="moe.csv",
        )


def test_runtime_rejects_misaligned_sm90_dsv4_w4a16_case():
    fake_torch = SimpleNamespace(
        set_default_device=lambda _device: None,
        cuda=SimpleNamespace(set_device=lambda _device: None),
    )
    run = _load_functions(
        "run_moe_torch",
        namespace={"torch": fake_torch, "get_sm_version": lambda: 90},
    )["run_moe_torch"]

    with pytest.raises(ValueError, match=r"SM90 DeepSeek-V4 W4A16.*local_inter_size"):
        run(
            "w4a16_mxfp4",
            128,
            7168,
            3072,
            6,
            384,
            16,
            1,
            "deepseek-ai/DeepSeek-V4-Pro",
            moe_backend="flashinfer_mxfp4",
            is_fp4_experts=True,
            perf_filename="moe.csv",
        )


def test_framework_moe_router_logits_are_explicitly_float32():
    source = ast.get_source_segment(
        SOURCE_PATH.read_text(),
        next(
            node
            for node in ast.parse(SOURCE_PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_benchmark_framework_quantized_moe"
        ),
    )

    assert "balanced_logits(num_tokens, num_experts, topk).to(device=device, dtype=torch.float32)" in source
    assert "device=device, dtype=torch.float32" in source.split("power_law_logits_v3", maxsplit=1)[1]


def test_framework_int4_builds_grouped_compressed_tensors_config():
    source = ast.get_source_segment(
        SOURCE_PATH.read_text(),
        next(
            node
            for node in ast.parse(SOURCE_PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_benchmark_framework_quantized_moe"
        ),
    )

    assert "CompressedTensorsConfig.from_config" in source
    assert '"strategy": "group"' in source
    assert '"group_size": int4_group_size' in source
