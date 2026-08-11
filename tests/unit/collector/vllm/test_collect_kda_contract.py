# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Phase coverage of the shared KDA case generator is pinned once, in
# tests/unit/collector/sglang/test_collect_kda_contract.py (both backend
# getters adapt the same generator).

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_kda.py"


def test_kda_context_raises_on_conv_int32_offset_overflow():
    # The vllm guard is deliberately NOT the sglang bound: sglang's Triton
    # kernel int32-offsets across the 3-block buffer (silicon-proven
    # `nt * 3*proj`); the vllm bound is the unverified per-block `nt * proj`
    # (see the FIXME(kernel-limit) at the guard). Pin the exact expression
    # structurally so a future bound edit is a conscious act.
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    guard_tests = {ast.unparse(node.test) for node in ast.walk(tree) if isinstance(node, ast.If)}
    assert "nt * proj >= 2 ** 31" in guard_tests


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
