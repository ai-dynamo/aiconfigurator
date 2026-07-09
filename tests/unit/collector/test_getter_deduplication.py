# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
from collector.case_generator import MoeCommonTestCase

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
AIC_PROJECT_ROOT = REPO_ROOT / "packages" / "aiconfigurator"


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
    spec = importlib.util.spec_from_file_location(module_name, AIC_PROJECT_ROOT / relative_path)
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
    _stub_module(monkeypatch, "vllm.version", __version__="0.19.0")
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
    monkeypatch.setattr(module, "get_common_moe_test_cases", lambda: common_cases)
    monkeypatch.setattr(module, "get_moe_quantization_modes", lambda *_args, **_kwargs: ["w4a16_mxfp4"])
    monkeypatch.setattr(module, "moe_model_allows_quantization", lambda *_args: True)
    monkeypatch.setattr(module, "moe_shape_satisfies_constraints", lambda *_args, **_kwargs: True)
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
    monkeypatch.setattr(module, "get_common_moe_test_cases", lambda: common_cases)
    monkeypatch.setattr(module, "get_moe_quantization_modes", lambda *_args, **_kwargs: ["w4a16_mxfp4"])
    monkeypatch.setattr(module, "moe_model_allows_quantization", lambda *_args: True)
    monkeypatch.setattr(module, "moe_shape_satisfies_constraints", lambda *_args, **_kwargs: True)
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
    monkeypatch.setattr(module, "per_block_cast_to_fp8", _noop)
    monkeypatch.setattr(module, "_nvfp4_available", True)
    monkeypatch.setattr(module, "_mxfp4_available", True)

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
    monkeypatch.setattr(module, "per_block_cast_to_fp8", _noop)
    monkeypatch.setattr(module, "_nvfp4_available", True)

    assert module.get_moe_test_cases() == []
