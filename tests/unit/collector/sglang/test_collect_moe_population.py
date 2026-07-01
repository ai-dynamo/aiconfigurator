# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import itertools
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_moe.py"
MODEL_CONFIG_DIR = REPO_ROOT / "src" / "aiconfigurator" / "model_configs"


def _load_functions(*names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded


def _gptoss_case(*, tp: int, ep: int):
    return SimpleNamespace(
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


def _populate_gptoss_cases(cases, *, sm_version=100, allowed_mode="w4a8_mxfp4_mxfp8"):
    allowed_modes = {allowed_mode} if isinstance(allowed_mode, str) else set(allowed_mode)
    loaded = _load_functions(
        "get_moe_test_cases",
        namespace={
            "itertools": itertools,
            "get_sm_version": lambda: sm_version,
            "get_common_moe_test_cases": lambda: cases,
            "moe_model_allows_quantization": (lambda _backend, _model, mode: mode in allowed_modes),
            "_uses_relu2_moe_activation": lambda _model: False,
            "get_moe_quantization_module_config": lambda *_args, **_kwargs: {},
            "_SM120_NEMOTRON_NVFP4_MODELS": set(),
        },
    )
    return loaded["get_moe_test_cases"]()


def _glm5_case(model_name: str):
    return SimpleNamespace(
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


@pytest.mark.parametrize(("tp", "ep"), [(4, 8), (32, 1), (32, 8)])
def test_gptoss_mxfp4_population_retains_tp_and_ep_buckets(tp, ep):
    populated = _populate_gptoss_cases([_gptoss_case(tp=tp, ep=ep)])

    assert len(populated) == 1
    assert populated[0][0] == "w4a8_mxfp4_mxfp8"
    assert populated[0][6:8] == [tp, ep]
    assert populated[0][-1] is None


@pytest.mark.parametrize(
    ("mode", "sm_version", "expected"),
    [
        ("nvfp4", 90, "marlin"),
        ("nvfp4", 100, "flashinfer_trtllm"),
        ("nvfp4", 103, "flashinfer_trtllm"),
        ("nvfp4", 120, "flashinfer_cutlass"),
        ("w4a16_mxfp4", 90, "flashinfer_mxfp4"),
        ("w4a8_mxfp4_mxfp8", 100, "flashinfer_mxfp4"),
    ],
)
def test_quantized_framework_backend_map_matches_sglang_0514(mode, sm_version, expected):
    resolve = _load_functions("_resolve_framework_moe_backend")["_resolve_framework_moe_backend"]

    assert resolve(mode, sm_version) == expected


@pytest.mark.parametrize(
    ("model_name", "activation", "scoring", "routing_method", "scale"),
    [
        ("nvidia/Kimi-K2.5-NVFP4", "silu", "sigmoid", "DeepSeekV3", 2.827),
        ("nvidia/MiniMax-M2.5-NVFP4", "silu", "sigmoid", None, None),
        ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "relu2", "sigmoid", "DeepSeekV3", 5.0),
        ("Qwen/Qwen3-235B-A22B", "silu", "softmax", "Renormalize", None),
        ("deepseek-ai/DeepSeek-V4-Pro", "silu", "sqrtsoftplus", "DeepSeekV3", 2.5),
    ],
)
def test_framework_routing_comes_from_local_model_contract(model_name, activation, scoring, routing_method, scale):
    enum = SimpleNamespace(DeepSeekV3="DeepSeekV3", Renormalize="Renormalize")
    route = _load_functions(
        "_framework_routing",
        namespace={
            "_MODEL_CONFIG_DIR": MODEL_CONFIG_DIR,
            "json": json,
            "RoutingMethodType": enum,
            "SimpleNamespace": SimpleNamespace,
        },
    )["_framework_routing"](model_name)

    assert route.activation == activation
    assert route.scoring_func == scoring
    assert route.routing_method == routing_method
    assert route.routed_scale == scale


@pytest.mark.parametrize("fail_during_benchmark", [False, True])
def test_mxfp4_parallel_patch_covers_benchmark_and_restores_helpers(fail_during_benchmark):
    def original_helper(name):
        def helper(*_args, **_kwargs):
            return name

        return helper

    moe_layer = SimpleNamespace(
        get_tp_group=original_helper("layer_tp_group"),
        is_allocation_symmetric=original_helper("layer_symmetric"),
        get_moe_expert_parallel_world_size=original_helper("layer_ep_world"),
        get_moe_expert_parallel_rank=original_helper("layer_ep_rank"),
        get_moe_tensor_parallel_world_size=original_helper("layer_tp_world"),
        get_moe_tensor_parallel_rank=original_helper("layer_tp_rank"),
        create_kt_config_from_server_args=original_helper("layer_kt_config"),
    )
    standard_dispatch = SimpleNamespace(
        get_tp_group=original_helper("dispatch_tp_group"),
        is_allocation_symmetric=original_helper("dispatch_symmetric"),
        get_moe_expert_parallel_world_size=original_helper("dispatch_ep_world"),
        get_moe_expert_parallel_rank=original_helper("dispatch_ep_rank"),
    )
    mxfp4 = SimpleNamespace(
        get_tp_group=original_helper("mxfp4_tp_group"),
        is_allocation_symmetric=original_helper("mxfp4_symmetric"),
    )
    modules = (moe_layer, standard_dispatch, mxfp4)
    originals = [(module, name, value) for module in modules for name, value in vars(module).items()]

    def benchmark_config(*_args, **_kwargs):
        assert moe_layer.get_moe_expert_parallel_world_size() == 8
        assert moe_layer.get_moe_expert_parallel_rank() == 0
        assert moe_layer.get_moe_tensor_parallel_world_size() == 4
        assert moe_layer.get_moe_tensor_parallel_rank() == 0
        assert moe_layer.create_kt_config_from_server_args(object(), 0) is None
        assert standard_dispatch.get_moe_expert_parallel_world_size() == 8
        assert standard_dispatch.get_moe_expert_parallel_rank() == 0
        for module in modules:
            assert module.get_tp_group() is None
            assert not module.is_allocation_symmetric()
        if fail_during_benchmark:
            raise RuntimeError("benchmark failed")
        return 1.25, {"power": 100.0}

    fake_torch = SimpleNamespace(dtype=object, cuda=SimpleNamespace(manual_seed_all=lambda _seed: None))
    loaded = _load_functions(
        "_patch_mxfp4_single_process_parallel",
        "benchmark",
        namespace={
            "contextmanager": contextmanager,
            "nullcontext": nullcontext,
            "torch": fake_torch,
            "_moe_layer_mod": moe_layer,
            "_std_dispatch_mod": standard_dispatch,
            "_mxfp4_mod": mxfp4,
            "_HAS_SGLANG_MXFP4": True,
            "_HAS_MARLIN_MOE": False,
            "benchmark_config": benchmark_config,
        },
    )
    benchmark = loaded["benchmark"]
    kwargs = {
        "num_tokens": 128,
        "num_experts": 8,
        "shard_intermediate_size": 512,
        "hidden_size": 256,
        "topk": 2,
        "dtype": object(),
        "use_fp8_w8a8": False,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_mxfp4_w4a16": True,
        "moe_tp_size": 4,
        "moe_ep_size": 8,
    }

    if fail_during_benchmark:
        with pytest.raises(RuntimeError, match="benchmark failed"):
            benchmark(**kwargs)
    else:
        assert benchmark(**kwargs) == (1.25, {"power": 100.0})

    for module, name, original in originals:
        assert getattr(module, name) is original
