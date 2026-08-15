# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import sys
import types
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


def _load_function(source_path: Path, name: str, namespace: dict | None = None):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded[name]


class TestResolveFlashinferGdnDecode:
    """_resolve_flashinfer_gdn_decode (CodeRabbit collect_gdn.py:369, Major):
    SM100+ sglang serving mandates the FlashInfer bf16-state GDN decode
    kernel (server_args.py:4884-4915 @0.5.14); an import failure there must
    surface as a classified failure the caller raises, never a silent skip
    that omits the serving-selected lane. SM<100 stays a legitimate no-op."""

    def _resolve(self, sm_version: int):
        return _load_function(
            SOURCE_PATH,
            "_resolve_flashinfer_gdn_decode",
            {"get_sm_version": lambda: sm_version},
        )

    def test_not_applicable_below_sm100(self):
        resolve = self._resolve(90)

        assert resolve() == (None, None)

    def test_boundary_sm100_takes_mandatory_lane_branch(self, monkeypatch):
        # SM100 itself (the boundary) must already take the mandatory-lane
        # branch: the guard is a strict `< 100`, so only SM99 and below skip.
        monkeypatch.delitem(sys.modules, "flashinfer.gdn_decode", raising=False)
        monkeypatch.delitem(sys.modules, "flashinfer", raising=False)
        resolve = self._resolve(100)

        kernel_fn, error_message = resolve()

        assert kernel_fn is None
        assert error_message is not None
        assert "SM100" in error_message

    def test_classified_error_when_unavailable_on_sm100(self, monkeypatch):
        # flashinfer is genuinely not installed in this dev/CI venv, so this
        # reproduces the real gap without needing a sys.modules trick: the
        # previous code returned a bare None here (CodeRabbit finding), so
        # the caller happily skipped the row and the case reported success.
        monkeypatch.delitem(sys.modules, "flashinfer.gdn_decode", raising=False)
        monkeypatch.delitem(sys.modules, "flashinfer", raising=False)
        resolve = self._resolve(103)

        kernel_fn, error_message = resolve()

        assert kernel_fn is None
        assert error_message is not None
        assert "SM103" in error_message
        assert "collection environment gap" in error_message

    def test_returns_kernel_when_available_on_sm100(self, monkeypatch):
        sentinel = object()
        fake_gdn_decode = types.ModuleType("flashinfer.gdn_decode")
        fake_gdn_decode.gated_delta_rule_decode_pretranspose = sentinel
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.gdn_decode = fake_gdn_decode
        monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
        monkeypatch.setitem(sys.modules, "flashinfer.gdn_decode", fake_gdn_decode)
        resolve = self._resolve(103)

        kernel_fn, error_message = resolve()

        assert kernel_fn is sentinel
        assert error_message is None


def test_run_gdn_generation_benchmark_raises_classified_error_not_silent_skip():
    """Structural guard: the flashinfer sibling-lane branch in
    run_gdn_generation_benchmark must raise the classified error resolved by
    _resolve_flashinfer_gdn_decode rather than silently falling through when
    the kernel is unavailable (the `if kernel is not None` guard alone would
    silently drop the row on SM100+, matching the CodeRabbit finding)."""
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gdn_generation_benchmark"
    )

    flashinfer_ifs = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "flashinfer_gdn_decode_fn"
    ]
    assert len(flashinfer_ifs) == 1
    guard = flashinfer_ifs[0]

    # ast.unparse's `elif` is represented as a single-statement `orelse`
    # whose test references the error variable and whose body raises.
    assert len(guard.orelse) == 1
    error_branch = guard.orelse[0]
    assert isinstance(error_branch, ast.If)
    assert isinstance(error_branch.test, ast.Compare)
    assert isinstance(error_branch.test.left, ast.Name)
    assert error_branch.test.left.id == "flashinfer_gdn_decode_error"
    assert any(isinstance(statement, ast.Raise) for statement in error_branch.body)
