# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from collector.model_cases import build_collection_case_plan
from collector.registry_types import PerfFile
from collector.trtllm.registry import REGISTRY

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]

DSV4_MODULE_OPS = {
    "dsv4_csa_context_module": (PerfFile.DSV4_CSA_CONTEXT_MODULE, "get_dsv4_csa_context_test_cases"),
    "dsv4_hca_context_module": (PerfFile.DSV4_HCA_CONTEXT_MODULE, "get_dsv4_hca_context_test_cases"),
    "dsv4_csa_generation_module": (PerfFile.DSV4_CSA_GENERATION_MODULE, "get_dsv4_csa_generation_test_cases"),
    "dsv4_hca_generation_module": (PerfFile.DSV4_HCA_GENERATION_MODULE, "get_dsv4_hca_generation_test_cases"),
}


def _load_module_with_torch_stub(monkeypatch):
    # exec_module runs the collector's import fallbacks, which append to
    # sys.path; snapshot it so the mutation does not leak into later tests.
    monkeypatch.setattr(sys, "path", list(sys.path))
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    module_path = REPO_ROOT / "collector" / "trtllm" / "collect_dsv4_attn.py"
    spec = importlib.util.spec_from_file_location("trtllm_dsv4_target", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_wires_all_four_dsv4_module_ops():
    entries = {entry.op: entry for entry in REGISTRY}
    for op, (perf_file, get_func) in DSV4_MODULE_OPS.items():
        entry = entries[op]
        assert entry.module == "collector.trtllm.collect_dsv4_attn"
        assert entry.run_func == "run_dsv4_attn_worker"
        # get_func <-> perf_filename pairing decides which population lands
        # in which table; a CSA getter wired to the HCA file would otherwise
        # pass and mislabel every row.
        assert entry.get_func == get_func
        assert entry.perf_filename == perf_file
        # Pre-Blackwell rejection comes from the framework at runtime
        # (classified failure). SM120 is parked with hardware probe evidence
        # (RTX PRO 6000 campaign 2026-08-07: 100% classified failures).
        assert entry.unverified is False
        assert entry.unverified_sms == (120,)


def test_trtllm_dsv4_plan_schedules_attention_modules():
    plan = build_collection_case_plan(backend="trtllm", model_path="sgl-project/DeepSeek-V4-Pro-FP8")
    assert set(DSV4_MODULE_OPS) <= plan.selected_ops
    assert "mhc_module" in plan.selected_ops


def test_dsv4_case_population_shape_and_budget(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)

    for mode, expected_kind, getter in (
        ("context", "csa", module.get_dsv4_csa_context_test_cases),
        ("context", "hca", module.get_dsv4_hca_context_test_cases),
        ("generation", "csa", module.get_dsv4_csa_generation_test_cases),
        ("generation", "hca", module.get_dsv4_hca_generation_test_cases),
    ):
        cases = getter()
        assert cases
        ids = [case["id"] for case in cases]
        assert len(ids) == len(set(ids)), f"duplicate ids in {mode}/{expected_kind}"
        for case in cases:
            params = case["params"]
            if mode == "context":
                sl, bs, tp, kv, comp, gemm, model, kind, prefix = params
                assert prefix + sl <= module.MAX_SEQ_LEN
                assert bs * sl <= module.MAX_CONTEXT_QUERY_TOKENS
                assert bs * (prefix + sl) <= module.MAX_GENERATION_KV_TOKENS
            else:
                sl, bs, tp, kv, comp, gemm, model, kind = params
                assert bs * sl <= module.MAX_GENERATION_KV_TOKENS
            assert sl <= module.MAX_SEQ_LEN
            assert (kv, comp, gemm) == ("fp8", "bfloat16", "fp8_block")
            assert kind == expected_kind
            assert tp in (1, 2, 4, 8)


def test_dsv4_worker_infers_mode_from_perf_filename(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)

    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(module, "run_dsv4_attn", fake_run)
    module.run_dsv4_attn_worker(
        64,
        1,
        2,
        "fp8",
        "bfloat16",
        "fp8_block",
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "csa",
        128,
        perf_filename="/out/dsv4_csa_context_module_perf.txt",
    )
    assert captured["mode"] == "context"
    assert captured["prefix_len"] == 128
    assert captured["tp_size"] == 2

    module.run_dsv4_attn_worker(
        1024,
        4,
        8,
        "fp8",
        "bfloat16",
        "fp8_block",
        "sgl-project/DeepSeek-V4-Pro-FP8",
        "hca",
        perf_filename="/out/dsv4_hca_generation_module_perf.txt",
    )
    assert captured["mode"] == "generation"
    assert captured["attn_kind"] == "hca"


def test_module_cache_reuses_same_geometry_and_evicts_on_change(monkeypatch):
    """Size-1 cache: consecutive same-(model, kind, tp) cases share one build;
    a geometry change evicts and rebuilds (bounded memory)."""
    module = _load_module_with_torch_stub(monkeypatch)

    builds = []

    def fake_build(*, model_path, attn_kind, tp_size, device):
        builds.append((model_path, attn_kind, tp_size))
        return (object(), object(), {"local_heads": 8, "native_heads": 64})

    monkeypatch.setattr(module, "create_dsv4_attention_module", fake_build)
    module._MODULE_CACHE.clear()

    a1 = module._cached_dsv4_attention_module("m/flash", "csa", 4, "cuda:0")
    a2 = module._cached_dsv4_attention_module("m/flash", "csa", 4, "cuda:0")
    assert a1 is a2
    assert len(builds) == 1

    b1 = module._cached_dsv4_attention_module("m/flash", "hca", 4, "cuda:0")
    assert len(builds) == 2
    assert list(module._MODULE_CACHE) == [("m/flash", "hca", 4, "cuda:0")]
    assert b1 is not a1
    module._MODULE_CACHE.clear()


def test_generation_geometry_matches_serving_invariant(monkeypatch):
    """position == past-seen/cached == persisted step (model_engine.py:4148,
    4164-4169 @1.3.0rc23): the decode dummy request registers seq_len+1 beam
    tokens, so all three sides of the triple are seq_len. Guards the
    off-by-one found in review (position was seq_len - 1)."""
    module = _load_module_with_torch_stub(monkeypatch)
    for seq_len in (1, 4, 512, 65536):
        geo = module.generation_request_geometry(seq_len)
        assert geo["position"] == geo["num_cached_tokens"] == geo["persisted_step"] == seq_len


# ---------------------------------------------------------------------------
# Constructor-level serving-parity capture (review item 5, PR #1486):
# stub the tensorrt_llm surface and assert the EXACT arguments handed to the
# cache-manager constructor, add_dummy_requests and the attention Metadata
# constructor for a context and a generation case.
# ---------------------------------------------------------------------------


class _CapturingManager:
    instances: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.ctor_args = args
        self.ctor_kwargs = kwargs
        self.dummy_calls: list = []
        _CapturingManager.instances.append(self)

    def add_dummy_requests(self, request_ids, token_nums, is_gen):
        self.dummy_calls.append({"request_ids": request_ids, "token_nums": token_nums, "is_gen": is_gen})

    def shutdown(self):
        pass


class _CapturingMetadata:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prepared = False
        _CapturingMetadata.instances.append(self)

    def prepare(self):
        self.prepared = True


def _install_trtllm_stubs(monkeypatch, module):
    captured = {"kv_cache_config": []}

    def _mod(name):
        m = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    tl = _mod("tensorrt_llm")
    _mod("tensorrt_llm._torch")
    iface = _mod("tensorrt_llm._torch.attention_backend.interface")
    _mod("tensorrt_llm._torch.attention_backend")
    utils_mod = _mod("tensorrt_llm._torch.attention_backend.utils")
    bindings = _mod("tensorrt_llm.bindings")
    _mod("tensorrt_llm.bindings.internal")
    bm = _mod("tensorrt_llm.bindings.internal.batch_manager")
    llm_args = _mod("tensorrt_llm.llmapi.llm_args")
    _mod("tensorrt_llm.llmapi")
    tl.__version__ = "1.3.0rc23"

    iface.AttentionRuntimeFeatures = lambda **kw: SimpleNamespace(**kw)
    iface.KVCacheParams = lambda **kw: SimpleNamespace(**kw)
    utils_mod.get_attention_backend = lambda backend, sparse_cfg: SimpleNamespace(
        Metadata=_CapturingMetadata, __name__="FakeDSV4Backend"
    )
    bindings.DataType = SimpleNamespace(FP8="FP8")
    bm.CacheType = SimpleNamespace(SELFKONLY="SELFKONLY")

    def _kv_cache_config(**kw):
        captured["kv_cache_config"].append(kw)
        return SimpleNamespace(**kw)

    llm_args.KvCacheConfig = _kv_cache_config
    # the create function re-imports this locally (relative-or-plain fallback)
    mla_stub = _mod("collect_mla_module")
    mla_stub.get_kv_cache_manager_cls = lambda mc, cfg: _CapturingManager

    # torch stub surface used by the create function
    torch_stub = sys.modules["torch"]
    torch_stub.int32 = "int32"
    torch_stub.int8 = "int8"
    torch_stub.cuda = SimpleNamespace(mem_get_info=lambda dev: (100 * 2**30, 128 * 2**30), empty_cache=lambda: None)
    torch_stub.tensor = lambda data, **kw: SimpleNamespace(data=data, **kw)
    torch_stub.device = lambda d: d
    return captured


def _fake_model_config():
    pretrained = SimpleNamespace(kv_lora_rank=448, qk_rope_head_dim=64, vocab_size=129280, hidden_size=7168)
    sparse = SimpleNamespace(to_sparse_metadata_params=lambda pretrained_config: {"sparse": True})
    return SimpleNamespace(
        pretrained_config=pretrained,
        sparse_attention_config=sparse,
        attn_backend="TRTLLM",
        mapping="MAPPING",
        extra_attrs={},
    )


def _run_create(module, monkeypatch, **kw):
    _CapturingManager.instances.clear()
    _CapturingMetadata.instances.clear()
    captured = _install_trtllm_stubs(monkeypatch, module)
    manager, metadata, attention_cls = module.create_dsv4_kv_cache_and_metadata(
        model_config=_fake_model_config(),
        attn_module=SimpleNamespace(indexer=None),
        device="cuda:0",
        **kw,
    )
    return manager, metadata, captured


def test_context_metadata_constructor_args_match_serving(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)
    bs, sl, prefix = 2, 64, 128
    manager, metadata, captured = _run_create(
        module, monkeypatch, batch_size=bs, seq_len=sl, is_context=True, prefix_len=prefix
    )
    # manager construction (pyexecutor/_util.py:1843-1867 @1.3.0rc23)
    mk = manager.ctor_kwargs
    assert mk["num_kv_heads"] == 1 and mk["num_layers"] == 1
    assert mk["head_dim"] == 448 + 64
    assert mk["tokens_per_block"] == 128
    assert mk["max_seq_len"] == 512  # engine-envelope floor (prefix+sl+1=193 -> 512)
    assert mk["vocab_size"] == 129280
    # dummy requests register the REAL request size, not the floored envelope
    dc = manager.dummy_calls[0]
    assert dc["token_nums"] == [prefix + sl] * bs and dc["is_gen"] is False
    # Metadata population (model_engine.py:2475-2489, :3960-3998)
    kw = metadata.kwargs
    assert kw["seq_lens"].data == [sl] * bs  # chunk-local fresh tokens
    assert kw["num_contexts"] == bs
    assert kw["kv_cache_params"].num_cached_tokens_per_seq == [prefix] * bs  # begin_compute prefix
    assert kw["prompt_lens"] == [sl] * bs  # chunk-local (SM100 cached-KV walker)
    assert kw["enable_context_mla_with_cached_kv"] is True
    assert kw["runtime_features"].cache_reuse is True
    assert kw["request_ids"] == list(range(bs))
    assert metadata.prepared


def test_generation_metadata_constructor_args_match_serving(monkeypatch):
    module = _load_module_with_torch_stub(monkeypatch)
    bs, past_kv = 4, 512
    manager, metadata, captured = _run_create(
        module, monkeypatch, batch_size=bs, seq_len=past_kv, is_context=False, prefix_len=0
    )
    geo = module.generation_request_geometry(past_kv)
    dc = manager.dummy_calls[0]
    # decode dummy request registers past_kv + 1 beam tokens -> serving's
    # past_seen = max_beam_num_tokens - 1 == past_kv (model_engine.py:4308)
    assert dc["token_nums"] == [past_kv + 1] * bs and dc["is_gen"] is True
    kw = metadata.kwargs
    assert kw["seq_lens"].data == [1] * bs  # one new token per decode step
    assert kw["num_contexts"] == 0
    # cached == past_seen == position triple (model_engine.py:4315,4332-4335)
    assert kw["kv_cache_params"].num_cached_tokens_per_seq == [geo["num_cached_tokens"]] * bs
    assert kw["prompt_lens"] == [past_kv] * bs
    assert kw["enable_context_mla_with_cached_kv"] is False
    assert kw["runtime_features"].cache_reuse is False
    assert metadata.prepared
