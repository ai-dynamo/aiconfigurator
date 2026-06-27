# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Collector V1 coverage guards for attention case generation."""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_encoder_head_configs,
    get_attention_encoder_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "collector_v1"

ATTENTION_V1_REF = "a4827ce203e9fbc24fe6c6779a7eaa2a7dc79f1a"
ENCODER_INITIAL_REF = "36808ecced9af9d0d71d944c716ae96d1d4a2a47"

ATTENTION_COUNTS = {
    ("sglang", "context"): (33714, 50901),
    ("sglang", "generation"): (19484, 39556),
    ("trtllm", "context"): (63192, 75483),
    ("trtllm", "generation"): (40240, 54318),
    ("vllm", "context"): (40392, 50932),
    ("vllm", "generation"): (36288, 53638),
    ("vllm_xpu", "context"): (16188, 17838),
    ("vllm_xpu", "generation"): (26322, 30728),
}

ATTENTION_MODULES = {
    "sglang": "collector/sglang/collect_attn.py",
    "trtllm": "collector/trtllm/collect_attn.py",
    "vllm": "collector/vllm/collect_attn.py",
    "vllm_xpu": "collector/vllm/collect_attn_xpu.py",
}


def _load_function_group(module_path: str, targets: tuple[str, ...], namespace: dict) -> dict:
    """Compile target getters and their local helper functions without framework imports."""

    source_path = REPO_ROOT / module_path
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = set()
    pending = list(targets)
    while pending:
        name = pending.pop()
        if name in required or name not in functions:
            continue
        required.add(name)
        calls = {
            node.func.id
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        pending.extend(calls & functions.keys())

    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in required]
    loaded = dict(namespace)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded


def _load_fixture(backend: str, op: str) -> tuple[dict, set[tuple]]:
    path = FIXTURE_ROOT / backend / f"{op}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        keys = {tuple(json.loads(line)) for line in stream}
    return header, keys


def _attention_physical_key(backend: str, phase: str, case: list) -> tuple:
    if backend == "sglang":
        batch, sequence, heads, kv_heads, head_dim, fp8_kv, fp8_fmha, is_context, window = case
    elif backend == "trtllm":
        batch, sequence, heads, kv_heads, head_dim, window, fp8_kv, fp8_fmha, is_context = case
    else:
        batch, sequence, heads, kv_heads, head_dim, fp8_kv, is_context, window = case
        fp8_fmha = False

    assert bool(is_context) == (phase == "context")
    normalized_kv_heads = 0 if kv_heads == heads else kv_heads
    if phase == "context":
        return (
            "fp8" if fp8_fmha else "bfloat16",
            "fp8" if fp8_kv else "bfloat16",
            normalized_kv_heads,
            head_dim,
            window,
            heads,
            sequence,
            batch,
        )
    return (
        "fp8" if fp8_kv else "bfloat16",
        normalized_kv_heads,
        head_dim,
        window,
        heads,
        batch,
        sequence + 1,
    )


def _current_attention_keys(backend: str, phase: str) -> set[tuple]:
    namespace = _load_function_group(
        ATTENTION_MODULES[backend],
        ("get_context_attention_test_cases", "get_generation_attention_test_cases"),
        {
            "get_attention_context_shape_sweeps": get_attention_context_shape_sweeps,
            "get_attention_generation_shape_sweeps": get_attention_generation_shape_sweeps,
            "get_attention_head_configs": get_attention_head_configs,
            "get_sm_version": lambda: 100,
            "tensorrt_llm": SimpleNamespace(__version__="1.3.0rc10"),
            "vllm_version": "0.19.0",
        },
    )
    getter = namespace[f"get_{phase}_attention_test_cases"]
    return {_attention_physical_key(backend, phase, case) for case in getter()}


@pytest.mark.parametrize(
    ("backend", "phase", "v1_count", "cleaned_count"),
    [(*scope, *counts) for scope, counts in ATTENTION_COUNTS.items()],
)
def test_current_attention_getters_are_exact_v1_physical_key_supersets(
    backend: str,
    phase: str,
    v1_count: int,
    cleaned_count: int,
):
    header, legacy_keys = _load_fixture(backend, f"attention_{phase}")
    current_keys = _current_attention_keys(backend, phase)

    assert header["source_git_ref"] == ATTENTION_V1_REF
    assert len(legacy_keys) == v1_count
    assert legacy_keys <= current_keys
    assert len(current_keys) == cleaned_count
    assert len(current_keys - legacy_keys) == cleaned_count - v1_count


@pytest.mark.parametrize("backend", ["sglang", "trtllm", "vllm"])
def test_current_encoder_getters_are_exact_initial_physical_key_supersets(backend: str):
    module_path = f"collector/{backend}/collect_attn_encoder.py"
    namespace = _load_function_group(
        module_path,
        ("get_encoder_attention_test_cases",),
        {
            "get_attention_encoder_shape_sweeps": get_attention_encoder_shape_sweeps,
            "get_attention_encoder_head_configs": get_attention_encoder_head_configs,
        },
    )
    current_keys = {
        ("bfloat16", head_dim, heads, sequence, batch)
        for batch, sequence, heads, head_dim in namespace["get_encoder_attention_test_cases"]()
    }
    header, legacy_keys = _load_fixture(backend, "encoder_attention")

    assert header["source_git_ref"] == ENCODER_INITIAL_REF
    assert len(legacy_keys) == 7008
    assert legacy_keys <= current_keys
    assert len(current_keys) == 7679
    assert len(current_keys - legacy_keys) == 671
