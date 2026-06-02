# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types


def _install_fake_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _import_collect_attn(monkeypatch):
    _install_fake_module(monkeypatch, "pkg_resources")

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float8_e4m3fn = "float8_e4m3fn"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    _install_fake_module(monkeypatch, "sglang")
    _install_fake_module(monkeypatch, "sglang.srt")
    _install_fake_module(monkeypatch, "sglang.srt.layers")
    _install_fake_module(monkeypatch, "sglang.srt.layers.attention")
    _install_fake_module(monkeypatch, "sglang.srt.configs")
    _install_fake_module(monkeypatch, "sglang.srt.mem_cache")
    _install_fake_module(monkeypatch, "sglang.srt.model_executor")

    _install_fake_module(monkeypatch, "sglang.srt.layers.dp_attention")
    _install_fake_module(
        monkeypatch,
        "sglang.srt.configs.model_config",
        AttentionArch=types.SimpleNamespace(MHA="mha"),
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.layers.attention.flashattention_backend",
        FlashAttentionBackend=object,
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.layers.radix_attention",
        RadixAttention=object,
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.mem_cache.memory_pool",
        MHATokenToKVPool=object,
        ReqToTokenPool=object,
    )
    _install_fake_module(
        monkeypatch,
        "sglang.srt.model_executor.forward_batch_info",
        ForwardBatch=object,
        ForwardMode=object,
    )

    _install_fake_module(
        monkeypatch,
        "collector.case_generator",
        get_attention_context_shape_sweeps=lambda _backend: [],
        get_attention_generation_shape_sweeps=lambda _backend: [],
    )
    _install_fake_module(
        monkeypatch,
        "collector.helper",
        benchmark_with_power=lambda **_kwargs: None,
        get_sm_version=lambda: 100,
        log_perf=lambda **_kwargs: None,
    )

    module_name = "collector.sglang.collect_attn"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, "collector/sglang/collect_attn.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_dense_attention_mocks_include_sglang_0512_fields(monkeypatch):
    mod = _import_collect_attn(monkeypatch)

    model_config = mod.MockModelConfig(num_attention_heads=8, num_key_value_heads=4, head_dim=192)
    server_args = mod.MockServerArgs(page_size=64)
    model_runner = mod.MockModelRunner(
        device="cuda:0",
        num_heads=8,
        num_kv_heads=4,
        head_dim=192,
    )

    assert model_config.v_head_dim == 192
    assert model_config.swa_v_head_dim == 192
    assert model_config.global_v_head_dim == 192
    assert server_args.enable_mis is False
    assert model_runner.linear_attn_model_spec is None


def test_runtime_sliding_window_size_uses_none_for_full_attention(monkeypatch):
    mod = _import_collect_attn(monkeypatch)

    assert mod._runtime_sliding_window_size(0, use_triton_attention=True) is None
    assert mod._runtime_sliding_window_size(-1, use_triton_attention=True) is None
    assert mod._runtime_sliding_window_size(1024, use_triton_attention=True) == 1024

    assert mod._runtime_sliding_window_size(0, use_triton_attention=False) == 0
