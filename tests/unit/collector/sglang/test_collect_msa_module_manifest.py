# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-time inner-manifest binding for the SGLang MSA collector (review
4969690316 Standards-2): every queued case carries the EXACT (b, s, prefix)
list it will execute — task IDs, resume checkpoints, and failure records bind
the retained shapes; the worker never re-derives the grid."""

import importlib.util
import sys
import types

import pytest

pytestmark = pytest.mark.unit


def _import_module(monkeypatch, sm: int = 100, cuda: bool = False):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None, set_device=lambda *_: None)
    torch.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    mod_name = "collector.sglang.collect_msa_module"
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    spec = importlib.util.spec_from_file_location(mod_name, "collector/sglang/collect_msa_module.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "get_sm_version", lambda: sm)
    return mod


def test_every_case_carries_a_nonempty_exact_manifest(monkeypatch):
    mod = _import_module(monkeypatch)
    for mode, getter in (
        ("context", mod.get_msa_context_module_test_cases),
        ("generation", mod.get_msa_generation_module_test_cases),
    ):
        cases = getter()
        assert cases, mode
        for case in cases:
            assert len(case) == 9, "case layout: positional prefix + target_tp + manifest"
            manifest = case[8]
            assert manifest, "empty-manifest cases must not be queued"
            # The manifest IS the plan: re-resolving with the same inputs
            # reproduces it exactly (deterministic binding).
            expected = mod._inner_manifest(
                mode,
                batch_size=case[1],
                num_heads=case[2],
                kv_dtype=case[3],
                model_path=case[6],
            )
            assert manifest == expected


def test_generation_manifest_respects_the_msa_budget(monkeypatch):
    mod = _import_module(monkeypatch)
    from collector.case_generator import get_mla_module_sweep_spec

    budget = get_mla_module_sweep_spec("sglang").generation_msa_max_tokens
    assert budget == 33554432
    for case in mod.get_msa_generation_module_test_cases():
        assert all(b * kv <= budget for (b, kv, _p) in case[8])


def test_worker_requires_the_manifest(monkeypatch):
    """The worker never re-derives the grid: without a manifest (and without
    --quick) it must raise instead of silently rebuilding shapes."""
    mod = _import_module(monkeypatch)
    with pytest.raises(RuntimeError, match=r"empty-manifest|manifest"):
        mod.run_msa_module(
            num_heads=8,
            model_path="MiniMaxAI/MiniMax-M3",
            kv_cache_dtype="bfloat16",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            is_prefill=False,
            gpu_id=0,
            inner_manifest=(),
        )
