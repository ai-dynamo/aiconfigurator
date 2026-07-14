# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from collector.fpm_forward.population import _dsv4_vllm_points
from collector.fpm_forward.types import FPMPoint
from collector.helper import create_test_case_id

pytestmark = pytest.mark.unit


@pytest.fixture
def dsv4_module(monkeypatch):
    def install_module(name: str, *, package: bool = False, **attrs):
        module = ModuleType(name)
        if package:
            module.__path__ = []
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class QuantizeMethodBase:
        pass

    class DeepseekV4Attention:
        pass

    def context(*_args, **_kwargs):
        return nullcontext()

    torch = install_module(
        "torch",
        OutOfMemoryError=RuntimeError,
        bfloat16=object(),
        device=lambda value: value,
        full=lambda *_args, **_kwargs: object(),
        inference_mode=context,
    )
    torch.cuda = SimpleNamespace(
        OutOfMemoryError=RuntimeError,
        empty_cache=lambda: None,
        get_device_name=lambda _device: "fake-gpu",
        synchronize=lambda: None,
    )
    install_module("vllm", package=True)
    install_module("vllm.config", set_current_vllm_config=context)
    install_module("vllm.forward_context", set_forward_context=context)
    install_module("vllm.model_executor", package=True)
    install_module("vllm.model_executor.layers", package=True)
    install_module("vllm.model_executor.layers.quantization", package=True)
    install_module(
        "vllm.model_executor.layers.quantization.base_config",
        QuantizeMethodBase=QuantizeMethodBase,
    )
    install_module("vllm.models", package=True)
    install_module("vllm.models.deepseek_v4", package=True)
    install_module("vllm.models.deepseek_v4.attention", DeepseekV4Attention=DeepseekV4Attention)
    install_module("vllm.models.deepseek_v4.nvidia", package=True)
    install_module("vllm.models.deepseek_v4.nvidia.model", _select_dsv4_attn_cls=lambda _config: None)
    install_module("vllm.platforms", current_platform=SimpleNamespace())
    install_module("vllm.utils", package=True)
    install_module(
        "vllm.utils.deep_gemm",
        fp8_fp4_paged_mqa_logits=lambda *_args, **_kwargs: None,
        get_paged_mqa_logits_metadata=lambda *_args, **_kwargs: None,
    )
    install_module("vllm.utils.platform_utils", num_compute_units=lambda *_args, **_kwargs: 1)
    install_module("vllm.utils.torch_utils", set_default_torch_dtype=context)
    install_module("vllm.version", __version__="0.24.0")
    install_module("vllm.v1", package=True)
    install_module("vllm.v1.worker", package=True)
    install_module("vllm.v1.worker.workspace", init_workspace_manager=lambda *_args, **_kwargs: None)
    install_module(
        "collector.vllm.utils",
        BatchSpec=SimpleNamespace,
        create_common_attn_metadata=lambda *_args, **_kwargs: None,
        create_vllm_config=lambda *_args, **_kwargs: None,
        setup_distributed=lambda *_args, **_kwargs: None,
    )

    module_name = "collector.vllm.collect_dsv4_attn"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_context_cases_carry_prefix_and_obey_both_token_budgets(dsv4_module, monkeypatch):
    mod = dsv4_module
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(mod, "_selected_dsv4_models", lambda: ("model",))
    monkeypatch.setattr(mod, "_DSV4_MODULE_BATCH_SIZES", [1, 1024])
    monkeypatch.setattr(mod, "_DSV4_MODULE_SEQ_LENGTHS", [64, mod.MAX_SEQ_LEN, mod.MAX_SEQ_LEN + 1])
    monkeypatch.setattr(mod, "_DSV4_MODULE_TP_SIZES", [1])

    cases = mod.get_dsv4_csa_context_test_cases()

    assert cases
    assert {len(case) for case in cases} == {10}
    assert {case[9] for case in cases if case[0] == 64 and case[1] == 1} == {
        0,
        128,
        2048,
        4096,
        mod.MAX_SEQ_LEN - 64,
    }
    assert {case[9] for case in cases if case[0] == mod.MAX_SEQ_LEN} == {0}
    assert all(case[0] <= mod.MAX_SEQ_LEN for case in cases)
    assert all(case[9] + case[0] <= mod.MAX_SEQ_LEN for case in cases)
    assert all(case[1] * case[0] <= mod.MAX_CONTEXT_QUERY_TOKENS for case in cases)
    assert all(case[1] * (case[9] + case[0]) <= mod.MAX_GENERATION_KV_TOKENS for case in cases)

    task_ids = {create_test_case_id(case, "run_dsv4_attn_worker", mod.__name__) for case in cases}
    assert len(task_ids) == len(cases)

    generation_cases = mod.get_dsv4_csa_generation_test_cases()
    assert generation_cases
    assert {len(case) for case in generation_cases} == {9}

    forwarded = {}
    monkeypatch.setattr(mod, "_init_cuda", lambda _device: None)
    monkeypatch.setattr(mod, "_bench_attention_shape", lambda **kwargs: forwarded.update(kwargs))
    nonzero_prefix_case = next(case for case in cases if case[9] == 128)
    mod.run_dsv4_attn_worker(
        *nonzero_prefix_case,
        perf_filename="dsv4_csa_context_module_perf.txt",
    )
    assert forwarded["prefix_len"] == 128


def test_fpm_population_matches_default_op_level_attention_shapes(dsv4_module, monkeypatch):
    mod = dsv4_module
    monkeypatch.setattr(mod, "MAX_SEQ_LEN", mod.DEFAULT_MAX_SEQ_LEN)
    prefill, decode = _dsv4_vllm_points("vllm", {})
    context_shapes = mod._vllm_dsv4_attention_filter_shapes(
        "context",
        mod._DSV4_MODULE_BATCH_SIZES,
        mod._DSV4_MODULE_SEQ_LENGTHS,
    )
    generation_shapes = mod._vllm_dsv4_attention_filter_shapes(
        "generation",
        mod._DSV4_MODULE_BATCH_SIZES,
        mod._DSV4_MODULE_SEQ_LENGTHS,
    )

    assert set(prefill) == {
        FPMPoint("prefill", batch_size, seq_len, prefix_len) for batch_size, seq_len, prefix_len in context_shapes
    }
    assert set(decode) == {
        FPMPoint("decode", batch_size, 1, seq_len) for batch_size, seq_len, _prefix_len in generation_shapes
    }


def test_smoke_has_zero_and_nonzero_prefix_without_endpoint_sweep(dsv4_module, monkeypatch):
    mod = dsv4_module
    monkeypatch.setattr(sys, "argv", ["collect.py", "--smoke"])
    monkeypatch.setattr(mod, "_selected_dsv4_models", lambda: ("model",))
    monkeypatch.setattr(mod, "_DSV4_MODULE_BATCH_SIZES", [1])
    monkeypatch.setattr(mod, "_DSV4_MODULE_TP_SIZES", [1])

    cases = mod.get_dsv4_hca_context_test_cases()

    assert {(case[0], case[9]) for case in cases} == {(64, 0), (64, 128)}


@pytest.mark.parametrize("kernel", ["paged_mqa_logits", "hca_attn"])
def test_sparse_cases_queue_short_shapes_for_runtime_observation(dsv4_module, monkeypatch, kernel):
    mod = dsv4_module
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(mod, "_selected_dsv4_models", lambda: ("model",))
    monkeypatch.setattr(mod, "_DSV4_SPARSE_BS_LIST", [1])
    monkeypatch.setattr(mod, "_DSV4_SPARSE_ISL_LIST", [1])
    monkeypatch.setattr(mod, "_DSV4_SPARSE_PAST_KV_LIST", [0])
    monkeypatch.setattr(mod, "_DSV4_SPARSE_TP_LIST_ATTN", [1])
    monkeypatch.setattr(mod, "_DSV4_SPARSE_TP_LIST_INDEXER", [1])

    assert mod._build_vllm_dsv4_sparse_test_cases(kernel) == [[1, 1, 0, 1, kernel, "model"]]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"kv_cache_dtype": "bfloat16"}, "kv_cache_dtype"),
        ({"compute_dtype": "float16"}, "compute_dtype"),
        ({"attention_backend": "FLASH_ATTN"}, "attention_backend must be unset"),
    ],
)
def test_dsv4_worker_rejects_ignored_runtime_inputs(dsv4_module, override, message):
    args = {
        "seq_len": 64,
        "batch_size": 1,
        "tp_size": 1,
        "kv_cache_dtype": "fp8",
        "compute_dtype": "bfloat16",
        "gemm_type": "fp8_block",
        "model_path": "model",
        "attn_kind": "csa",
        "attention_backend": None,
        "perf_filename": "dsv4_csa_context_module_perf.txt",
    }
    args.update(override)

    with pytest.raises(ValueError, match=message):
        dsv4_module.run_dsv4_attn_worker(**args)


@pytest.mark.parametrize("kernel", ["paged_mqa_logits", "hca_attn"])
def test_sparse_worker_observes_short_shapes(dsv4_module, monkeypatch, kernel):
    calls = {}
    monkeypatch.setattr(dsv4_module, "_init_cuda", lambda device: calls.setdefault("device", device))
    monkeypatch.setattr(dsv4_module, "_bench_sparse_kernel_shape", lambda **kwargs: calls.update(kwargs))

    dsv4_module.run_dsv4_sparse_kernel_worker(
        1,
        1,
        0,
        1,
        kernel,
        "model",
        perf_filename="perf.txt",
    )

    assert calls["kernel"] == kernel
    assert calls["isl"] == 1
    assert calls["past_kv"] == 0


def test_sparse_worker_raises_beyond_max_position_embeddings(dsv4_module, monkeypatch):
    monkeypatch.setattr(dsv4_module, "_init_cuda", lambda device: None)
    monkeypatch.setattr(
        dsv4_module, "_bench_sparse_kernel_shape", lambda **kwargs: pytest.fail("must raise before benchmarking")
    )

    over_limit = dsv4_module._DSV4_SPARSE_MAX_FULL_S + 1
    with pytest.raises(ValueError, match="max_position_embeddings"):
        dsv4_module.run_dsv4_sparse_kernel_worker(
            1,
            1,
            over_limit,
            1,
            "paged_mqa_logits",
            "model",
            perf_filename="perf.txt",
        )


@pytest.mark.parametrize(
    ("mode", "seq_len", "prefix_len", "metadata_seq_len", "query_len", "logged_isl", "logged_step"),
    [
        ("context", 64, 128, 192, 64, 64, 128),
        ("generation", 64, 0, 65, 1, 1, 64),
    ],
)
def test_bench_uses_total_cache_length_and_fresh_query_length(
    dsv4_module,
    monkeypatch,
    mode,
    seq_len,
    prefix_len,
    metadata_seq_len,
    query_len,
    logged_isl,
    logged_step,
):
    mod = dsv4_module
    calls = {}

    class FakeAttention:
        prefix = "model.layers.0.attn"
        n_local_heads = 8

        def __call__(self, *_args):
            return None

    class FakeLayer:
        def get_attn_backend(self):
            return SimpleNamespace(get_name=lambda: "FAKE_BACKEND")

        def get_kv_cache_spec(self, _config):
            return object()

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                hidden_size=8,
            )
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={"model.layers.0.attn": FakeLayer()},
        ),
    )

    @contextmanager
    def create_module(**kwargs):
        calls["create"] = kwargs
        yield FakeAttention(), config

    @contextmanager
    def benchmark(**_kwargs):
        yield {"latency_ms": 1.25, "power_stats": None}

    monkeypatch.setattr(mod, "_tp_simulation", lambda _tp: nullcontext())
    monkeypatch.setattr(mod, "_create_dsv4_attention_module", create_module)

    def make_metadata(**kwargs):
        calls["metadata"] = kwargs
        return SimpleNamespace(positions=object())

    monkeypatch.setattr(mod, "_make_common_metadata", make_metadata)
    monkeypatch.setattr(mod, "_build_metadata_and_bind_caches", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mod, "set_current_vllm_config", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(mod, "set_forward_context", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(mod, "benchmark_with_power", benchmark)
    monkeypatch.setattr(mod, "log_perf", lambda **kwargs: calls.setdefault("log", kwargs))
    monkeypatch.setattr(mod.torch, "full", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(mod.torch.cuda, "get_device_name", lambda _device: "fake-gpu")
    monkeypatch.setattr(mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(mod.gc, "collect", lambda: None)

    latency = mod._bench_attention_shape(
        model_path="model",
        attn_kind="csa",
        mode=mode,
        batch_size=2,
        seq_len=seq_len,
        prefix_len=prefix_len,
        tp_size=1,
        gemm_type="fp8_block",
        device="cuda:0",
        perf_filename="perf.txt",
        warming_up=1,
        test_ite=1,
    )

    assert latency == 1.25
    assert calls["create"]["seq_len"] == metadata_seq_len
    assert calls["create"]["query_len"] == query_len
    assert calls["metadata"]["seq_len"] == metadata_seq_len
    assert calls["metadata"]["query_len"] == query_len
    row = calls["log"]["item_list"][0]
    assert row["isl"] == logged_isl
    assert row["step"] == logged_step
    # kernel_source must be the registered layer's actual backend, not an
    # assumed name.
    assert calls["log"]["kernel_source"] == "FAKE_BACKEND"


@pytest.mark.parametrize(("mode", "expected_prefix"), [("context", 128), ("generation", 0)])
def test_cli_past_kv_is_context_prefix_only(dsv4_module, monkeypatch, mode, expected_prefix):
    mod = dsv4_module
    calls = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dsv4_attn.py",
            "--mode",
            mode,
            "--seq-len",
            "64",
            "--past-kv",
            "128",
        ],
    )
    monkeypatch.setattr(mod, "_init_cuda", lambda _device: None)
    monkeypatch.setattr(mod, "_bench_attention_shape", lambda **kwargs: calls.update(kwargs))

    mod.main()

    assert calls["prefix_len"] == expected_prefix
    assert calls["seq_len"] == 64
