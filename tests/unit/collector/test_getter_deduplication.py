# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from collector.case_generator import MoeCommonTestCase

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


class _Dummy:
    def __init__(self, *_args, **kwargs):
        self.__dict__.update(kwargs)


def _noop(*_args, **_kwargs):
    return None


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_collector(monkeypatch, module_name: str, relative_path: str):
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _install_trtllm_stubs(monkeypatch):
    _stub_module(
        monkeypatch,
        "torch",
        Tensor=object,
        bfloat16="bfloat16",
        float8_e4m3fn="float8_e4m3fn",
        float32="float32",
        cuda=types.SimpleNamespace(empty_cache=_noop, Stream=_noop),
        device=lambda value: value,
    )
    _stub_module(monkeypatch, "tensorrt_llm", __version__="1.3.0rc10")
    for package in (
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.models",
        "tensorrt_llm._torch.modules",
        "tensorrt_llm.models",
    ):
        _stub_module(monkeypatch, package)

    _stub_module(monkeypatch, "tensorrt_llm._torch.autotuner", AutoTuner=_Dummy, autotune=_noop)
    _stub_module(monkeypatch, "tensorrt_llm._torch.model_config", ModelConfig=_Dummy)
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.models.modeling_deepseekv3",
        DeepseekV3Gate=_Dummy,
    )
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.modules.fused_moe",
        RenormalizeMoeRoutingMethod=_Dummy,
        create_moe=_noop,
    )
    _stub_module(monkeypatch, "tensorrt_llm.mapping", Mapping=_Dummy)
    _stub_module(monkeypatch, "tensorrt_llm.models.modeling_utils", QuantAlgo=_Dummy(), QuantConfig=_Dummy)
    _stub_module(
        monkeypatch,
        "collector.helper",
        EXIT_CODE_RESTART=1,
        balanced_logits=_noop,
        benchmark_with_power=_noop,
        get_sm_version=lambda: 100,
        log_perf=_noop,
        power_law_logits_v3=_noop,
    )


def _install_vllm_stubs(monkeypatch):
    torch = _stub_module(
        monkeypatch,
        "torch",
        Tensor=object,
        bfloat16="bfloat16",
        float8_e4m3fn="float8_e4m3fn",
        float32="float32",
        uint8="uint8",
        device=lambda value: value,
    )
    torch_nn = _stub_module(monkeypatch, "torch.nn")
    torch.nn = torch_nn
    _stub_module(monkeypatch, "torch.nn.functional")

    _stub_module(monkeypatch, "vllm")
    for package in (
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
    ):
        _stub_module(monkeypatch, package)
    sys.modules["vllm.model_executor.layers.fused_moe"].fused_experts = _noop
    _stub_module(monkeypatch, "vllm.config", VllmConfig=_Dummy, set_current_vllm_config=_noop)
    _stub_module(monkeypatch, "vllm.forward_context", get_forward_context=_noop, set_forward_context=_noop)
    _stub_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.config",
        fp8_w8a8_moe_quant_config=_noop,
        int4_w4a16_moe_quant_config=_noop,
    )
    _stub_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.layer",
        determine_expert_map=_noop,
    )
    _stub_module(monkeypatch, "vllm.version", __version__="0.24.0")
    _stub_module(
        monkeypatch,
        "collector.helper",
        balanced_logits=_noop,
        benchmark_with_power=_noop,
        get_sm_version=lambda: 100,
        log_perf=_noop,
        power_law_logits_v3=_noop,
    )


def _moe_case(model_name: str, *, distribution: str = "balanced", alpha: float = 0.0, ep: int = 1):
    return MoeCommonTestCase(
        num_tokens_list=[1, 8],
        hidden_size=4096,
        inter_size=2048,
        topk=2,
        num_experts=8,
        tp=1,
        ep=ep,
        model_name=model_name,
        token_expert_distribution=distribution,
        power_law_alpha=alpha,
    )


def _persisted_distribution(distribution: str, alpha: float | None) -> str:
    return f"power_law_{alpha}" if distribution == "power_law" else distribution


def test_trtllm_moe_getter_dedupes_equal_resolved_invocations(monkeypatch):
    _install_trtllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.trtllm.collect_moe", "collector/trtllm/collect_moe.py")
    common_cases = [
        _moe_case("model-a"),
        _moe_case("model-b"),
        _moe_case("model-a", distribution="power_law", alpha=1.2),
    ]
    module_configs = {
        "model-a": {"group_size": 32},
        "model-b": {"group_size": 32},
    }
    monkeypatch.setattr(module, "get_common_moe_test_cases", lambda: common_cases)
    monkeypatch.setattr(module, "moe_model_allows_quantization", lambda _backend, _model, mode: mode == "int4_wo")
    monkeypatch.setattr(
        module,
        "get_moe_quantization_module_config",
        lambda _backend, _mode, *, model_name: module_configs[model_name],
    )

    cases = module.get_moe_test_cases()

    assert {(case[9], case[10], case[11]) for case in cases} == {
        ("model-a", "balanced", 0.0),
        ("model-a", "power_law", 1.2),
    }


@pytest.mark.parametrize(
    ("conflicting_model", "conflicting_config"),
    [
        ("model-c", {"group_size": 128}),
        ("vendor/Nemotron-3-test", {"group_size": 32}),
    ],
)
def test_trtllm_moe_getter_rejects_consumer_key_collision(
    monkeypatch,
    conflicting_model,
    conflicting_config,
):
    _install_trtllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.trtllm.collect_moe", "collector/trtllm/collect_moe.py")
    common_cases = [
        _moe_case("model-a"),
        replace(_moe_case(conflicting_model), num_tokens_list=[8, 16]),
    ]
    module_configs = {
        "model-a": {"group_size": 32},
        conflicting_model: conflicting_config,
    }
    monkeypatch.setattr(module, "get_common_moe_test_cases", lambda: common_cases)
    monkeypatch.setattr(module, "moe_model_allows_quantization", lambda _backend, _model, mode: mode == "int4_wo")
    monkeypatch.setattr(
        module,
        "get_moe_quantization_module_config",
        lambda _backend, _mode, *, model_name: module_configs[model_name],
    )

    with pytest.raises(ValueError, match="TRT-LLM MoE population collision") as exc_info:
        module.get_moe_test_cases()

    assert "model-a" in str(exc_info.value)
    assert conflicting_model in str(exc_info.value)


def test_trtllm_dsv4_moe_getter_retains_tp_and_ep_buckets(monkeypatch):
    _install_trtllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.trtllm.collect_moe", "collector/trtllm/collect_moe.py")
    tp_ep_pairs = {(tp, ep) for tp in (4, 16, 32) for ep in (1, 2, 4, 8)}
    base_case = MoeCommonTestCase(
        num_tokens_list=[128],
        hidden_size=4096,
        inter_size=2048,
        topk=6,
        num_experts=256,
        tp=1,
        ep=1,
        model_name="deepseek-ai/DeepSeek-V4-Flash",
        token_expert_distribution="balanced",
        power_law_alpha=None,
        architecture="DeepseekV4ForCausalLM",
    )
    monkeypatch.setattr(
        module,
        "get_common_moe_test_cases",
        lambda: [replace(base_case, tp=tp, ep=ep) for tp, ep in sorted(tp_ep_pairs)],
    )
    monkeypatch.setattr(
        module,
        "moe_model_allows_quantization",
        lambda _backend, _model, mode: mode == "w4a8_mxfp4_mxfp8",
    )
    monkeypatch.setattr(module, "get_moe_quantization_module_config", lambda *_args, **_kwargs: {})

    cases = module.get_moe_test_cases()

    assert {(case[6], case[7]) for case in cases} == tp_ep_pairs
    assert {case[0] for case in cases} == {"w4a8_mxfp4_mxfp8"}


def test_vllm_moe_getter_dedupes_equal_resolved_invocations(monkeypatch):
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    common_cases = [
        _moe_case("model-a"),
        _moe_case("model-b"),
        _moe_case("model-a", distribution="power_law", alpha=1.2),
        _moe_case("model-b", ep=2),
    ]
    module_configs = {
        "model-a": {"activation": "silu", "has_bias": False},
        "model-b": {"activation": "silu", "has_bias": False},
    }
    monkeypatch.setattr(module, "get_common_moe_test_cases", lambda **_kwargs: common_cases)
    monkeypatch.setattr(module, "get_moe_quantization_modes", lambda *_args, **_kwargs: ["w4a16_mxfp4"])
    monkeypatch.setattr(module, "moe_model_allows_quantization", lambda *_args: True)
    monkeypatch.setattr(
        module,
        "_load_model_moe_config",
        lambda _model_name: {"model_type": "qwen3_moe", "hidden_act": "silu", "norm_topk_prob": True},
    )
    monkeypatch.setattr(
        module,
        "get_moe_quantization_module_config",
        lambda _backend, _mode, *, model_name: module_configs[model_name],
    )

    cases = module.get_moe_test_cases()
    model_names = [case[8] for case in cases]

    assert len(cases) == 3
    assert model_names.count("model-a") == 2
    assert model_names.count("model-b") == 1


def test_vllm_moe_getter_rejects_consumer_key_collision(monkeypatch):
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    common_cases = [
        _moe_case("model-a"),
        replace(_moe_case("model-c"), num_tokens_list=[8, 16]),
    ]
    module_configs = {
        "model-a": {"activation": "silu", "has_bias": False},
        "model-c": {"activation": "swigluoai", "has_bias": True},
    }
    monkeypatch.setattr(module, "get_common_moe_test_cases", lambda **_kwargs: common_cases)
    monkeypatch.setattr(module, "get_moe_quantization_modes", lambda *_args, **_kwargs: ["w4a16_mxfp4"])
    monkeypatch.setattr(module, "moe_model_allows_quantization", lambda *_args: True)
    monkeypatch.setattr(
        module,
        "_load_model_moe_config",
        lambda _model_name: {"model_type": "qwen3_moe", "hidden_act": "silu", "norm_topk_prob": True},
    )
    monkeypatch.setattr(
        module,
        "get_moe_quantization_module_config",
        lambda _backend, _mode, *, model_name: module_configs[model_name],
    )

    with pytest.raises(ValueError, match="vLLM MoE population collision") as exc_info:
        module.get_moe_test_cases()

    assert "model-a" in str(exc_info.value)
    assert "model-c" in str(exc_info.value)


def test_trtllm_repository_moe_getter_has_unique_consumer_keys(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    _install_trtllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.trtllm.collect_moe", "collector/trtllm/collect_moe.py")

    consumer_keys = []
    for case in module.get_moe_test_cases():
        moe_type, num_tokens_list, hidden_size, inter_size, topk, num_experts, tp, ep = case[:8]
        min_latency_mode, _, distribution, alpha = case[8:]
        table = "low_latency" if min_latency_mode else "default"
        distribution = _persisted_distribution(distribution, alpha)
        consumer_keys.extend(
            (table, moe_type, distribution, topk, num_experts, hidden_size, inter_size, tp, ep, num_tokens)
            for num_tokens in num_tokens_list
        )

    assert consumer_keys
    assert len(consumer_keys) == len(set(consumer_keys))


def test_vllm_repository_moe_getter_has_unique_consumer_keys(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")

    consumer_keys = []
    for case in module.get_moe_test_cases():
        moe_type, num_tokens_list, hidden_size, inter_size, topk, num_experts, tp, ep = case[:8]
        _, distribution, alpha = case[8:]
        distribution = _persisted_distribution(distribution, alpha)
        consumer_keys.extend(
            (moe_type, distribution, topk, num_experts, hidden_size, inter_size, tp, ep, num_tokens)
            for num_tokens in num_tokens_list
        )

    assert consumer_keys
    assert len(consumer_keys) == len(set(consumer_keys))


def test_vllm_sm90_repository_moe_getter_excludes_unconsumable_dsv4_cases(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    monkeypatch.setattr(module, "get_sm_version", lambda: 90)

    cases = module.get_moe_test_cases()
    native_dsv4_models = {
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
    }
    converted_dsv4_models = {
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "sgl-project/DeepSeek-V4-Pro-FP8",
    }

    assert len(cases) == 1887
    assert sum(len(case[1]) for case in cases) == 50949
    # Native artifacts stay excluded (their w4a8_mxfp4_mxfp8 label has no
    # consumable vLLM path); the converted FP8 artifacts are collected as
    # fp8_block only — the layout vLLM serves with the documented
    # expert_dtype override.
    assert not any(case[8] in native_dsv4_models for case in cases)
    converted_modes = {case[0] for case in cases if case[8] in converted_dsv4_models}
    assert converted_modes == {"fp8_block"}


@pytest.mark.parametrize(
    ("model_path", "moe_type"),
    [
        ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "bfloat16"),
        ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8", "fp8"),
    ],
)
def test_vllm_nemotron_ultra_uses_latent_moe_width(monkeypatch, model_path, moe_type):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    monkeypatch.setattr(module, "get_sm_version", lambda: 90)

    cases = module.get_moe_test_cases()

    assert len(cases) == 42
    assert sum(len(case[1]) for case in cases) == 1134
    assert {case[0] for case in cases} == {moe_type}
    assert {case[2] for case in cases} == {2048}


@pytest.mark.parametrize(
    "model_path",
    [
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    ],
)
def test_vllm_nemotron_topk22_nvfp4_artifacts_are_not_scheduled(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")

    assert module.get_moe_test_cases() == []


@pytest.mark.parametrize(
    "model_path",
    [
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    ],
)
def test_nemotron_ultra_declares_backend_specific_moe_geometry(monkeypatch, model_path):
    from collector.case_generator import get_common_moe_test_cases

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    assert {case.hidden_size for case in get_common_moe_test_cases()} == {2048, 8192}
    assert {case.hidden_size for case in get_common_moe_test_cases(backend="vllm")} == {2048}
    assert {case.hidden_size for case in get_common_moe_test_cases(backend="sglang")} == {8192}
    assert {case.hidden_size for case in get_common_moe_test_cases(backend="trtllm")} == {8192}


def test_vllm_moe_declares_representable_parallel_topologies(monkeypatch):
    from collector.case_generator import get_common_moe_test_cases

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    cases = get_common_moe_test_cases(backend="vllm")
    topologies = {(case.tp, case.ep) for case in cases}

    assert all(tp == 1 or ep == 1 for tp, ep in topologies)
    assert any(tp > 1 and ep == 1 for tp, ep in topologies)
    assert any(tp == 1 and ep > 1 for tp, ep in topologies)


def test_vllm_moe_cuda_graph_fails_closed():
    tree = ast.parse((REPO_ROOT / "collector/vllm/collect_moe.py").read_text())
    run_moe = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_moe_torch")
    benchmark_calls = [
        node
        for node in ast.walk(run_moe)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "benchmark_with_power"
    ]

    assert len(benchmark_calls) == 1
    assert all(keyword.arg != "allow_graph_fail" for keyword in benchmark_calls[0].keywords)


def test_vllm_moe_resolves_dynamic_experts_inside_token_loop():
    tree = ast.parse((REPO_ROOT / "collector/vllm/collect_moe.py").read_text())
    run_moe = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_moe_torch")
    token_loop = next(
        node
        for node in ast.walk(run_moe)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "num_tokens"
    )

    selected_leaf_calls = [
        node
        for node in ast.walk(token_loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_select_experts_impl"
    ]
    source_assignments = [
        node
        for node in ast.walk(token_loop)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "source" for target in node.targets)
    ]

    assert len(selected_leaf_calls) == 1
    assert len(source_assignments) == 2


@pytest.mark.parametrize(
    ("group_fields", "message"),
    [
        ({"topk_group": 4}, "missing n_group"),
        ({"n_group": 8}, "missing topk_group"),
        ({"n_group": "invalid", "topk_group": 1}, "requires integer group fields"),
        ({"n_group": 0, "topk_group": 1}, "invalid n_group=0"),
        ({"n_group": 4, "topk_group": 8}, "invalid n_group=4, topk_group=8"),
    ],
)
def test_vllm_grouped_topk_config_fails_closed(monkeypatch, group_fields, message):
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    model_config = {"model_type": "deepseek_v3", "hidden_act": "silu", **group_fields}
    monkeypatch.setattr(module, "_load_model_moe_config", lambda _model_name: model_config)

    with pytest.raises(ValueError, match=message):
        module._resolve_moe_runtime_config("model", {})


def test_vllm_standard_topk_does_not_require_group_fields(monkeypatch):
    _install_vllm_stubs(monkeypatch)
    module = _load_collector(monkeypatch, "collector.vllm.collect_moe", "collector/vllm/collect_moe.py")
    monkeypatch.setattr(module, "_load_model_moe_config", lambda _model_name: {"model_type": "qwen3_moe"})

    runtime_config = module._resolve_moe_runtime_config("model", {})

    assert runtime_config["use_grouped_topk"] is False
    assert runtime_config["num_expert_group"] is None
    assert runtime_config["topk_group"] is None
