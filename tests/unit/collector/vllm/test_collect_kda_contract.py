# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Phase coverage of the shared KDA case generator is pinned once, in
# tests/unit/collector/sglang/test_collect_kda_contract.py (both backend
# getters adapt the same generator).

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_kda.py"


def test_kda_context_conv_int32_overflow_guard_is_resolved_not_present():
    # Resolved at the 0.27.0 era bump: the old unverified FIXME
    # (kernel-limit) guard `nt * proj >= 2 ** 31` was deleted after 0.27.0's
    # causal_conv1d was verified int64 throughout and GB300 silicon passed
    # the formerly-vetoed cells. Pin that the guard does not silently come
    # back without re-verification, and that the resolution comment keeps
    # its evidence citations.
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    guard_tests = {ast.unparse(node.test) for node in ast.walk(tree) if isinstance(node, ast.If)}
    assert "nt * proj >= 2 ** 31" not in guard_tests
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "stride_x_token: tl.int64" in source


def test_kda_context_conv_passes_serve_parity_metadata():
    # Serving prefill hands the step-cached GDNAttentionMetadata into every
    # layer's causal_conv1d_fn call; omitting it selects the non-serving
    # metadata=None branch that rebuilds token offsets with numpy + a D2H
    # sync inside every timed call (the 0.1.dev19262 pollute-flat bug). Pin
    # that run_kda_context_benchmark passes metadata= structurally.
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_kda_context_benchmark":
            conv_calls = [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "causal_conv1d_fn"
            ]
            # Materialized before asserting: a generator is always truthy and
            # all() over an empty one is vacuously true, so an assertion built
            # on a generator passes even when no causal_conv1d_fn call exists.
            assert len(conv_calls) == 1, (
                f"run_kda_context_benchmark must call causal_conv1d_fn at exactly one site, found {len(conv_calls)}"
            )
            assert any(kw.arg == "metadata" for kw in conv_calls[0].keywords), (
                "run_kda_context_benchmark's causal_conv1d_fn call must pass metadata= "
                "(serve-parity GDNAttentionMetadata); the metadata=None branch adds a "
                "numpy + D2H floor inside every timed call"
            )
            return
    raise AssertionError("run_kda_context_benchmark not found / no causal_conv1d_fn call")


def test_kda_context_seq_len_one_routes_through_decode_kernels(monkeypatch):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    run_entrypoint = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_kda_torch"
    )

    calls = []

    def record(kind):
        return lambda **kwargs: calls.append((kind, kwargs))

    namespace = {
        "WORKER_RESTART": 23,
        "run_kda_context_benchmark": record("context"),
        "run_kda_generation_benchmark": record("decode"),
        "run_kda_verify_benchmark": record("verify"),
    }
    module = ast.Module(body=[run_entrypoint], type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    version_module = ModuleType("vllm.version")
    version_module.__version__ = "0.27.0"
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.version", version_module)

    kwargs = {
        "phase": "context",
        "d_model": 7168,
        "d_conv": 4,
        "num_k_heads": 12,
        "head_k_dim": 128,
        "num_v_heads": 12,
        "head_v_dim": 128,
        "batch_size_list": [1, 2],
        "seq_len_list": [1, 2, 4],
        "model_name": "moonshotai/Kimi-K3",
        "perf_filename": "unused.txt",
    }
    assert namespace["run_kda_torch"](**kwargs) == 23
    assert [(kind, call["seq_len_list"] if kind == "context" else call["row_phase"]) for kind, call in calls] == [
        ("decode", "context"),
        ("context", [2, 4]),
    ]

    with pytest.raises(ValueError, match="sequence lengths must be positive"):
        namespace["run_kda_torch"](**{**kwargs, "seq_len_list": [0]})


def test_kda_dispatch_mirrors_serving():
    # The collector must dispatch prefill like serving (FlashKDA when
    # supported, Triton fallback) and probe the fused decode kernel via the
    # same predicate serving uses — never pin a kernel unconditionally.
    # AST name references (not substring greps), so docstrings/comments
    # cannot satisfy the contract — mirrors the sglang twin test.
    import ast

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "is_flashkda_supported" in referenced
    assert "is_fused_kda_decode_supported" in referenced
