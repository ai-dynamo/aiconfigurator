# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_kda.py"


def test_kda_context_does_not_silently_drop_fixed_capacity_shapes():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_kda_context_benchmark"
    )
    referenced_names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}

    assert "MAX_KDA_CONTEXT_TOKENS" not in referenced_names
    assert "skipped_points" not in referenced_names


def test_kda_context_raises_on_conv_int32_offset_overflow():
    # Same verified framework kernel limit as GDN: the Triton causal_conv1d
    # forward computes token-major offsets in int32
    # (causal_conv1d_triton.py:373-379). KDA runs the conv per Q/K/V block,
    # but each block is a strided VIEW over the full 3-block mixed_qkv buffer,
    # so the offsets span total_tokens * conv_channels elements — proven on
    # silicon: cells in [2**31, 3*2**31) IMA'd on both Hopper/SM90 and
    # B200/SM100 under the older per-block (proj_size) bound. The guard must
    # RAISE inside
    # the sweep loop so the cell lands in the classified failure log instead
    # of corrupting the CUDA context.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "total_tokens * conv_channels >= 2**31" in source
    assert "int32 token-offset overflow" in source
    assert "causal_conv1d_triton.py:373-379" in source


def test_kda_case_phases_cover_context_generation_verify():
    # The registry getter must emit all three phases for every declared shape;
    # verify rows carry the draft-token width in the seq_len slot. The backend
    # getter is a field-order adapter over the shared spec generator (which is
    # importable without torch), so assert against the actual emitted specs
    # instead of grepping the module source (review fix: the phase strings
    # also appear in the docstring, so a source grep could never fail).
    from collector.case_generator import get_common_kda_test_cases

    phases = {case.phase for case in get_common_kda_test_cases()}
    assert phases == {"context", "generation", "verify"}
