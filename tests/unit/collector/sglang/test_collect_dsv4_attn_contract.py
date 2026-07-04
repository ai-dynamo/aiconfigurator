# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from collections.abc import Iterable
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_dsv4_attn.py"


def test_dsv4_worker_fails_closed_when_phase_has_no_valid_shapes():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_subprocess_entry"
    )
    namespace = {
        "Iterable": Iterable,
        "_expand_grid": lambda: ([], [1]),
        "_filter_pairs": lambda *_args: [],
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

    with pytest.raises(RuntimeError, match=r"no valid prefix/sl values; ok=0 error=0 skip=0 total=0"):
        namespace["_subprocess_entry"](
            mode="generation",
            attn_kind="csa",
            model_path="sgl-project/DeepSeek-V4-Flash-FP8",
            kv_cache_dtype="fp8",
            batch_size=1,
            output_path="unused",
        )


def test_dsv4_generation_enters_sglang_model_capture_mode():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_dsv4_mla_module"
    )
    function_source = ast.get_source_segment(source, function)

    assert "model_capture_mode() if not is_prefill" in function_source
