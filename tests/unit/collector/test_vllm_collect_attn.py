# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import os
import typing
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ATTN_SOURCE = ROOT / "collector/vllm/collect_attn.py"
UTILS_SOURCE = ROOT / "collector/vllm/utils.py"
BF16, FP8, UINT8 = object(), object(), object()


def _load_function(path: Path, name: str, namespace: dict):
    """Compile the real function body without importing unavailable vLLM."""
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _return_self(self, *_args, **_kwargs):
    return self


class FakeTensor:
    ndim = 5
    __getitem__ = contiguous = permute = to = transpose = _return_self


class Spec:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)


class FullAttentionSpec(Spec):
    pass


class SlidingWindowSpec(Spec):
    pass


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]


@pytest.mark.parametrize("use_fp8_kv_cache", [False, True])
def test_generation_uses_total_runtime_length_and_production_call_order(use_fp8_kv_cache):
    calls, events = {}, []
    slot_mapping = object()
    history_slot_mapping = SimpleNamespace(numel=lambda: 128)

    torch = SimpleNamespace(
        bfloat16=BF16,
        cuda=SimpleNamespace(set_device=lambda _device: None, get_device_name=lambda _device: "fake-gpu"),
        tensor=lambda *_args, **_kwargs: FakeTensor(),
        randn=lambda *_args, **_kwargs: FakeTensor(),
        cat=lambda *_args, **_kwargs: FakeTensor(),
        empty_like=lambda *_args, **_kwargs: FakeTensor(),
    )

    class Platform:
        def fp8_dtype(self):
            return FP8

        def get_attn_backend_cls(self, _backend, selector, *, num_heads):
            calls["selector"] = selector
            calls["selector_num_heads"] = num_heads
            return "fake.backend"

    class Backend:
        forward_includes_kv_cache_update = False

        get_name = staticmethod(lambda: "FLASH_ATTN")
        get_required_kv_cache_layout = staticmethod(lambda: None)
        get_kv_cache_stride_order = staticmethod(lambda: (0, 1, 2, 3, 4))

    class Builder:
        def __init__(self, *_args, **_kwargs):
            pass

        def build(self, **_kwargs):
            return object()

    class Impl:
        supports_quant_query_input = False

        def __init__(self, *_args, **_kwargs):
            pass

        def do_kv_cache_update(self, layer, key, value, cache, slots):
            events.append(("update", layer, key, value, cache, slots))

        def forward(self, layer, query, key, value, cache, metadata, *, output):
            events.append(("forward", layer, query, key, value, cache, metadata, output))

    class MockAttentionLayer:
        def __init__(self, _device):
            self._q_scale = self._k_scale = self._v_scale = FakeTensor()
            self._q_scale_float = self._k_scale_float = self._v_scale_float = 1.0

    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, num_gpu_blocks=8192),
        model_config=SimpleNamespace(get_sliding_window=lambda: None),
    )

    def create_config(**kwargs):
        calls["config"] = kwargs
        return config

    def create_metadata(batch_spec, _block_size, _device):
        calls["batch"] = batch_spec
        return SimpleNamespace(slot_mapping=slot_mapping)

    @contextmanager
    def benchmark(**kwargs):
        kwargs["kernel_func"]()
        kwargs["kernel_func"]()
        yield {"latency_ms": 1.25, "power_stats": None}

    namespace = {
        "__file__": str(ATTN_SOURCE),
        "os": os,
        "torch": torch,
        "BatchSpec": BatchSpec,
        "MockAttentionLayer": MockAttentionLayer,
        "set_random_seed": lambda _seed: None,
        "create_vllm_config": create_config,
        "set_current_vllm_config": lambda _config: nullcontext(),
        "AttentionSelectorConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "current_platform": Platform(),
        "resolve_obj_by_qualname": lambda _path: Backend,
        "AttentionBackendEnum": {"FLASH_ATTN": object()},
        "create_standard_kv_cache_spec": lambda *_args: SimpleNamespace(dtype=UINT8 if use_fp8_kv_cache else BF16),
        "create_common_attn_metadata": create_metadata,
        "create_kv_cache_and_block_mappings": lambda **_kwargs: (FakeTensor(), history_slot_mapping),
        "set_kv_cache_layout": lambda _layout: None,
        "get_attention_backend": lambda _backend: (Builder, Impl),
        "benchmark_with_power": benchmark,
        "log_perf": lambda **kwargs: calls.setdefault("log", kwargs),
        "vllm_version": "0.24.0",
    }
    run = _load_function(ATTN_SOURCE, "run_attention_torch", namespace)
    with ExitStack() as stack:
        run(
            stack,
            batch_size=2,
            input_len=64,
            num_heads=8,
            num_kv_heads=2,
            head_dim=128,
            use_fp8_kv_cache=use_fp8_kv_cache,
            is_context_phase=False,
            perf_filename="generation_attention_perf.txt",
        )

    assert calls["batch"] == BatchSpec(seq_lens=[65, 65], query_lens=[1, 1])
    assert calls["config"]["max_model_len"] == 65
    assert calls["selector"].kv_cache_dtype == ("fp8" if use_fp8_kv_cache else "auto")
    assert calls["log"]["item_list"][0]["isl"] == 1
    assert calls["log"]["item_list"][0]["step"] == 64
    assert [event[0] for event in events] == ["update", "update", "forward", "update", "forward"]
    assert events[0][5] is history_slot_mapping
    for update, forward in zip(events[1::2], events[2::2], strict=True):
        assert update[1:5] == (forward[1], forward[3], forward[4], forward[5])
        assert update[5] is slot_mapping


@pytest.mark.parametrize("randomize_blocks", [False, True])
def test_dense_kv_cache_history_slots_follow_block_table(randomize_blocks):
    torch = pytest.importorskip("torch")

    namespace = {
        "torch": torch,
        "CommonAttentionMetadata": object,
        "cdiv": lambda value, divisor: (value + divisor - 1) // divisor,
    }
    create_cache = _load_function(UTILS_SOURCE, "create_kv_cache_and_block_mappings", namespace)
    metadata = SimpleNamespace(
        num_reqs=2,
        seq_lens_cpu=torch.tensor([70, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([69, 4], dtype=torch.int32),
        block_table_tensor=torch.full((2, 2), -1, dtype=torch.int32),
        slot_mapping=torch.full((2,), -1, dtype=torch.long),
    )

    cache, history_slots = create_cache(
        block_size=64,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_blocks=8,
        common_attn_metadata=metadata,
        randomize_blocks=randomize_blocks,
    )

    expected_history_slots = []
    for request, context_len in enumerate([69, 4]):
        for offset in range(context_len):
            block_id = int(metadata.block_table_tensor[request, offset // 64])
            expected_history_slots.append(block_id * 64 + offset % 64)

    assert cache.shape == (2, 8, 64, 1, 8)
    assert sorted(metadata.block_table_tensor[0, :2].tolist() + metadata.block_table_tensor[1, :1].tolist()) == [
        1,
        2,
        3,
    ]
    assert metadata.slot_mapping.tolist() == [
        int(metadata.block_table_tensor[0, 1]) * 64 + 5,
        int(metadata.block_table_tensor[1, 0]) * 64 + 4,
    ]
    assert history_slots.tolist() == expected_history_slots


def test_mla_fp8_history_uses_framework_cache_writer():
    torch = pytest.importorskip("torch")
    calls = []

    class Ops:
        @staticmethod
        def concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, *, kv_cache_dtype, scale):
            calls.append((kv_c, k_pe, slots.clone(), kv_cache_dtype, scale.clone()))
            kv_cache.view(-1, kv_cache.shape[-1])[slots] = 17

    namespace = {
        "torch": torch,
        "ops": Ops,
        "CommonAttentionMetadata": object,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "cdiv": lambda value, divisor: (value + divisor - 1) // divisor,
    }
    create_cache = _load_function(UTILS_SOURCE, "create_and_prepopulate_kv_cache_mla", namespace)
    metadata = SimpleNamespace(
        seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([2], dtype=torch.int32),
        block_table_tensor=torch.full((1, 1), -1, dtype=torch.int32),
        slot_mapping=torch.full((1,), -1, dtype=torch.long),
    )
    kv_c = torch.full((2, 4), 1.5, dtype=torch.bfloat16)
    k_pe = torch.full((2, 1, 2), -1.5, dtype=torch.bfloat16)

    cache = create_cache(
        kv_c_contexts=[kv_c],
        k_pe_contexts=[k_pe],
        block_size=64,
        head_size=6,
        dtype=torch.uint8,
        device=torch.device("cpu"),
        num_blocks=4,
        common_attn_metadata=metadata,
        randomize_blocks=False,
        kv_cache_dtype="fp8",
        scale=2.0,
    )

    assert len(calls) == 1
    assert calls[0][0] is kv_c
    assert torch.equal(calls[0][1], k_pe.squeeze(1))
    assert calls[0][2].tolist() == [64, 65]
    assert calls[0][3] == "fp8"
    assert calls[0][4].item() == 2.0
    assert torch.equal(cache[1, :2], torch.full((2, 6), 17, dtype=torch.uint8))


@pytest.mark.parametrize(
    ("use_fp8", "window", "expected_cls", "expected_dtype", "cache_dtype"),
    [
        (False, None, FullAttentionSpec, BF16, "auto"),
        (True, None, FullAttentionSpec, UINT8, "fp8"),
        (True, 128, SlidingWindowSpec, UINT8, "fp8"),
    ],
)
def test_standard_kv_cache_spec_matches_vllm_contract(use_fp8, window, expected_cls, expected_dtype, cache_dtype):
    namespace = {
        "VllmConfig": object,
        "FullAttentionSpec": FullAttentionSpec,
        "SlidingWindowSpec": SlidingWindowSpec,
        "get_kv_quant_mode": lambda dtype: ("quant", dtype),
        "kv_cache_dtype_str_to_dtype": lambda dtype, _model: BF16 if dtype == "auto" else UINT8,
    }
    create_spec = _load_function(UTILS_SOURCE, "create_standard_kv_cache_spec", namespace)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, cache_dtype=cache_dtype),
        parallel_config=object(),
        model_config=SimpleNamespace(
            dtype=BF16,
            get_num_kv_heads=lambda _parallel: 2,
            get_head_size=lambda: 128,
            get_sliding_window=lambda: window,
        ),
    )

    spec = create_spec(config, use_fp8)

    assert type(spec) is expected_cls
    assert (spec.block_size, spec.num_kv_heads, spec.head_size, spec.head_size_v) == (64, 2, 128, 128)
    assert spec.dtype is expected_dtype
    assert spec.kv_quant_mode == ("quant", cache_dtype)
    if window is not None:
        assert spec.sliding_window == window
