# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.case_generator import (
    AttentionHeadConfig,
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_collector_function(name, namespace):
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    loaded = dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded[name]


def _model_configs(monkeypatch, model_path, sm_version, phase):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    sweeps = (
        get_attention_context_shape_sweeps("sglang")
        if phase == "context"
        else get_attention_generation_shape_sweeps("sglang")
    )
    return [
        config
        for sweep in sweeps
        for config in get_attention_head_configs(
            sweep,
            phase=phase,
            backend="sglang",
            sm_version=sm_version,
        )
    ]


def _profiles(*profiles):
    return {"head_profiles": list(profiles)}


def _profile(*, source="fa3", window_size=0, v_head_dim=128, chunk=None):
    profile = {
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "v_head_dim": v_head_dim,
        "window_size": window_size,
        "tensor_parallel_sizes": [1],
        "sglang_backends": {90: source},
    }
    if chunk is not None:
        profile["sglang_attention_chunk_size"] = chunk
    return profile


def test_framework_neutral_attention_population_keeps_the_existing_contract(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    configs = get_attention_head_configs(
        _profiles(_profile()),
        phase="context",
    )

    assert configs == [AttentionHeadConfig(8, 2, 128, 0)]


@pytest.mark.parametrize(
    ("model_path", "tp_heads", "windows"),
    [
        ("openai/gpt-oss-120b", [64, 32, 16, 8, 4, 2, 1], [0, 128]),
        ("meta-llama/Llama-4-Scout-17B-16E", [40, 20, 10, 5], [0, 8192]),
    ],
)
def test_framework_neutral_model_profiles_keep_their_historical_order(monkeypatch, model_path, tp_heads, windows):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    sweep = get_attention_context_shape_sweeps("sglang")[0]
    configs = get_attention_head_configs(sweep, phase="context")

    assert [(config.num_heads, config.window_size) for config in configs] == [
        (num_heads, window_size) for num_heads in tp_heads for window_size in windows
    ]


def test_sglang_runtime_metadata_does_not_expand_the_legacy_key(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")

    with pytest.raises(ValueError, match="legacy-key/source pair"):
        get_attention_head_configs(
            _profiles(_profile(v_head_dim=128), _profile(v_head_dim=64)),
            phase="context",
            backend="sglang",
            sm_version=90,
        )

    with pytest.raises(ValueError, match="legacy-key/source pair"):
        get_attention_head_configs(
            _profiles(
                _profile(window_size=8192),
                _profile(window_size=8192, chunk=8192),
            ),
            phase="context",
            backend="sglang",
            sm_version=90,
        )


def test_sglang_global_chunk_is_a_runtime_noop_for_population(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    configs = get_attention_head_configs(
        _profiles(_profile(), _profile(chunk=8192)),
        phase="context",
        backend="sglang",
        sm_version=90,
    )

    assert len(configs) == 1


def test_sglang_source_is_recordable_without_becoming_an_sdk_key(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    configs = get_attention_head_configs(
        _profiles(_profile(source="fa3"), _profile(source="triton")),
        phase="context",
        backend="sglang",
        sm_version=90,
    )

    assert [config.kernel_source for config in configs] == ["fa3", "triton"]


def test_sglang_default_backend_map_follows_0514_serving_selection(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")

    def _sources(sm_version, *, sink=False):
        profile = _profile()
        del profile["sglang_backends"]
        if sink:
            profile["sglang_has_attention_sink"] = True
        configs = get_attention_head_configs(
            _profiles(profile),
            phase="context",
            backend="sglang",
            sm_version=sm_version,
        )
        return {config.kernel_source for config in configs}

    # server_args._get_default_attn_backend (MHA): non-Hopper/non-SM100 CUDA
    # defaults to flashinfer unless the model has attention sinks.
    assert _sources(90) == {"fa3"}
    assert _sources(100) == {"trtllm_mha"}
    assert _sources(103) == {"trtllm_mha"}
    assert _sources(89) == {"flashinfer"}
    assert _sources(120) == {"flashinfer"}
    assert _sources(89, sink=True) == {"triton"}
    assert _sources(120, sink=True) == {"triton"}

    for unsupported_sm in (80, 86):
        with pytest.raises(ValueError, match=r"No SGLang 0\.5\.14 attention backend mapping"):
            _sources(unsupported_sm)


def test_sglang_sm90_full_structural_population_is_stable(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)

    for phase, sweep_getter, expected in (
        ("context", get_attention_context_shape_sweeps, 147),
        ("generation", get_attention_generation_shape_sweeps, 132),
    ):
        configs = [
            config
            for sweep in sweep_getter("sglang")
            for config in get_attention_head_configs(
                sweep,
                phase=phase,
                backend="sglang",
                sm_version=90,
            )
        ]
        population_keys = {
            (
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.window_size,
                config.kernel_source,
            )
            for config in configs
        }

        assert len(configs) == expected
        assert len(population_keys) == expected


def test_sglang_0514_model_runtime_contracts(monkeypatch):
    mimo = {
        config.window_size: config
        for config in _model_configs(monkeypatch, "XiaomiMiMo/MiMo-V2-Flash", 90, "context")
        if config.num_heads == 64
    }
    assert {(config.head_dim, config.v_head_dim) for config in mimo.values()} == {(192, 128)}
    assert not mimo[0].has_attention_sink
    assert mimo[128].has_attention_sink
    assert mimo[128].runtime_window_size == 128
    assert {config.kernel_source for config in mimo.values()} == {"fa3"}

    gemma = {
        config.window_size: config
        for config in _model_configs(monkeypatch, "google/gemma-4-26B-A4B", 90, "context")
        if config.num_heads == 16
    }
    assert gemma[1024].runtime_window_size == 1023
    assert {config.scaling for config in gemma.values()} == {1.0}
    assert {config.kernel_source for config in gemma.values()} == {"triton"}

    llama = {
        config.window_size: config
        for config in _model_configs(monkeypatch, "meta-llama/Llama-4-Scout-17B-16E", 90, "context")
        if config.num_heads == 40
    }
    assert set(llama) == {0, 8192}
    assert all(config.attention_chunk_size == 8192 for config in llama.values())
    assert all(config.runtime_window_size == -1 for config in llama.values())

    gpt_oss = {
        config.window_size: config
        for config in _model_configs(monkeypatch, "openai/gpt-oss-120b", 90, "generation")
        if config.num_heads == 64
    }
    assert gpt_oss[128].runtime_window_size == 127
    assert all(config.has_attention_sink for config in gpt_oss.values())
    assert {config.kernel_source for config in gpt_oss.values()} == {"fa3"}

    assert {
        config.kernel_source for config in _model_configs(monkeypatch, "nvidia/Nemotron-H-56B-Base-8K", 100, "context")
    } == {"flashinfer"}
    assert {config.kernel_source for config in _model_configs(monkeypatch, "Qwen/Qwen3.5-27B", 100, "context")} == {
        "triton"
    }
    assert _model_configs(monkeypatch, "moonshotai/Kimi-K2.5", 90, "context") == []


def test_sglang_attention_runtime_fields_are_not_persisted_dimensions():
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]

    pool = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "MHATokenToKVPool")
    radix = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "RadixAttention")
    log_perf = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "log_perf")
    layer_call = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "layer")

    assert "v_head_dim" in {keyword.arg for keyword in pool.keywords}
    assert {"v_head_dim", "sliding_window_size", "use_irope"} <= {keyword.arg for keyword in radix.keywords}
    assert "sinks" in {keyword.arg for keyword in layer_call.keywords}

    item_list = next(keyword.value for keyword in log_perf.keywords if keyword.arg == "item_list")
    row = item_list.elts[0]
    row_fields = {key.value for key in row.keys if isinstance(key, ast.Constant)}
    assert "v_head_dim" not in row_fields
    assert "attention_chunk_size" not in row_fields
    assert "runtime_window_size" not in row_fields
    source_kw = next(keyword.value for keyword in log_perf.keywords if keyword.arg == "kernel_source")
    assert isinstance(source_kw, ast.Name) and source_kw.id == "attn_backend_name"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sm_version", "num_heads", "num_kv_heads", "head_dim", "max_kv_elements"),
    [
        (100, 8, 2, 128, 1_000_000),
        # KV storage remains below the declared budget while the old broad
        # SM120 Q/O predicate would have removed this backend-specific case.
        (120, 8, 1, 2, 100),
    ],
)
def test_blackwell_context_cases_are_not_silently_removed(
    sm_version,
    num_heads,
    num_kv_heads,
    head_dim,
    max_kv_elements,
):
    sweep = {
        "batch_sizes": [1],
        "sequence_lengths": [10],
        "max_tokens_self_attention": 1_000_000,
        "max_tokens_grouped_query_attention": 1_000_000,
        "max_batch_size_self_attention": 1_000,
        "max_kv_elements": max_kv_elements,
        "precision_cases": [{"fp8_kv_cache": True, "fp8_context_fmha": True}],
    }
    head = SimpleNamespace(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        window_size=0,
        v_head_dim=head_dim,
        runtime_window_size=-1,
        attention_chunk_size=None,
        has_attention_sink=False,
        scaling=None,
        kernel_source="triton",
        architecture="TestForCausalLM",
    )
    getter = _load_collector_function(
        "get_context_attention_test_cases",
        {
            "_int_list": lambda values: [int(value) for value in values],
            "get_sm_version": lambda: sm_version,
            "get_attention_context_shape_sweeps": lambda _backend: [sweep],
            "get_attention_head_configs": lambda *_args, **_kwargs: [head],
        },
    )

    cases = getter()

    assert len(cases) == 1
    assert cases[0][5:7] == [True, True]


@pytest.mark.unit
def test_fp8_live_attention_rejects_backends_that_only_accept_bfloat16_inputs():
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch"
    )
    source = ast.get_source_segment(source_path.read_text(encoding="utf-8"), function)

    assert 'attn_backend_name in {"flashinfer", "trtllm_mha"}' in source
    assert 'attn_backend_name != "trtllm_mha"' not in source


@pytest.mark.parametrize(
    "relative_path",
    [
        "collector/trtllm/collect_attn.py",
        "collector/vllm/collect_attn.py",
        "collector/vllm/collect_attn_xpu.py",
    ],
)
def test_untouched_attention_collectors_keep_framework_neutral_calls(relative_path):
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_attention_head_configs"
    ]

    assert len(calls) == 2
    assert all(not any(keyword.arg in {"backend", "sm_version"} for keyword in call.keywords) for call in calls)
