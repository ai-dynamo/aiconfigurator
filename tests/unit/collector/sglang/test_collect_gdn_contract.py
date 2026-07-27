# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_gdn.py"


def test_gdn_context_does_not_silently_drop_fixed_capacity_shapes():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gdn_context_benchmark"
    )
    referenced_names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}

    assert "MAX_GDN_CONTEXT_TOKENS" not in referenced_names
    assert "MAX_GDN_CONTEXT_VALUE_ELEMENTS" not in referenced_names
    assert "skipped_points" not in referenced_names


def test_gdn_context_raises_on_conv_int32_offset_overflow():
    # Verified framework kernel limit, not a silent skip: stock 0.5.14
    # _causal_conv1d_fwd_kernel int32 token-offset overflow at 2**31 packed
    # elements (causal_conv1d_triton.py:373-379; RTX 6000 Pro memcheck
    # 2026-07-06). The guard must RAISE inside the sweep loop so the cell
    # contributes to the failing group summary instead of corrupting the CUDA
    # context and aborting the remaining cells.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "total_tokens * conv_channels >= 2**31" in source
    assert "int32 token-offset overflow" in source
    assert "causal_conv1d_triton.py:373-379" in source
