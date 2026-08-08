# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from collector.model_cases import build_collection_case_plan
from collector.registry_types import PerfFile
from collector.trtllm.registry import REGISTRY

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]

DSV4_MODULE_OPS = {
    "dsv4_csa_context_module": PerfFile.DSV4_CSA_CONTEXT_MODULE,
    "dsv4_hca_context_module": PerfFile.DSV4_HCA_CONTEXT_MODULE,
    "dsv4_csa_generation_module": PerfFile.DSV4_CSA_GENERATION_MODULE,
    "dsv4_hca_generation_module": PerfFile.DSV4_HCA_GENERATION_MODULE,
}


def _load_module_with_torch_stub(monkeypatch):
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
    for op, perf_file in DSV4_MODULE_OPS.items():
        entry = entries[op]
        assert entry.module == "collector.trtllm.collect_dsv4_attn"
        assert entry.run_func == "run_dsv4_attn_worker"
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

    for mode, getter in (
        ("context", module.get_dsv4_csa_context_test_cases),
        ("generation", module.get_dsv4_hca_generation_test_cases),
    ):
        cases = getter()
        assert cases
        ids = [case["id"] for case in cases]
        assert len(ids) == len(set(ids)), f"duplicate ids in {mode}"
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
            assert kind in module.ATTN_KIND_TO_COMPRESS_RATIO
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
