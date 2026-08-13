# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fp8_block GEMM weight/scale pre-packing (AIC-1744).

``collector/sglang/collect_gemm.py`` imports ``sglang``/``sgl_kernel`` (and
``torch``) at module scope and only runs on a GPU node with those installed.
This test loads ``_prepare_fp8_block_weights`` (plus the ``cdiv``/
``scale_shape`` helpers it calls) straight from the source AST and executes
the real function bodies against a minimal fake ``torch`` -- the same
technique already used for this class of collector file by
``tests/unit/collector/sglang/test_collect_moe_population.py`` and
``tests/unit/collector/test_vllm_collect_attn.py``.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_gemm.py"


def _load_functions(*names: str, namespace: dict | None = None) -> dict:
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded


FLOAT8, FLOAT32, INT32 = object(), object(), object()


class _FakeTensor:
    """Shape/dtype-tracking stand-in; values are never read by this test."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def clamp(self, min=None, max=None):  # noqa: A002 - mirrors torch.Tensor.clamp's kwarg names
        return self

    def to(self, dtype):
        return _FakeTensor(self.shape, dtype)

    def __sub__(self, _other):
        return self

    def __mul__(self, _other):
        return self

    __rmul__ = __mul__


def _fake_torch():
    return SimpleNamespace(
        float8_e4m3fn=FLOAT8,
        float32=FLOAT32,
        finfo=lambda _dtype: SimpleNamespace(min=-448.0, max=448.0),
        rand=lambda *shape, device=None: _FakeTensor(tuple(shape), FLOAT32),
        randn=lambda shape, device=None, dtype=None: _FakeTensor(tuple(shape), dtype),
    )


def _prepare_fp8_block_weights(namespace):
    return _load_functions(
        "_prepare_fp8_block_weights",
        "cdiv",
        "scale_shape",
        namespace=namespace,
    )["_prepare_fp8_block_weights"]


def test_fp8_block_weights_are_prepacked_once_when_ue8m0():
    calls = []

    def fake_requant(w, s, block):
        calls.append((w.shape, s.shape, tuple(block)))
        return w, s.to(INT32)  # stand-in for the packed UE8M0 scale

    prepare = _prepare_fp8_block_weights({"torch": _fake_torch(), "requant_weight_ue8m0": fake_requant})

    b_fp8, scale_b = prepare(256, 256, "cpu", ue8m0=True)

    assert len(calls) == 1
    assert calls[0] == ((256, 256), (2, 2), (128, 128))
    assert scale_b.dtype is INT32
    assert b_fp8.dtype is FLOAT8


def test_fp8_block_weights_stay_raw_fp32_when_not_ue8m0():
    calls = []

    def fake_requant(w, s, block):
        calls.append((w.shape, s.shape, tuple(block)))
        return w, s.to(INT32)

    prepare = _prepare_fp8_block_weights({"torch": _fake_torch(), "requant_weight_ue8m0": fake_requant})

    b_fp8, scale_b = prepare(256, 256, "cpu", ue8m0=False)

    assert calls == []
    assert scale_b.dtype is FLOAT32
    assert b_fp8.dtype is FLOAT8


def test_create_gemm_fp8_block_branch_delegates_to_the_prepack_helper():
    """Locks in that ``create_gemm`` no longer builds ``scale_b`` inline.

    Weight/scale packing must happen exactly once, at setup, via
    ``_prepare_fp8_block_weights`` -- never inside the ``fp8_block``
    ``gemm_op`` closure, which the timed benchmark replays every call.
    """
    source = SOURCE_PATH.read_text()
    start = source.index('elif gemm_type == "fp8_block":')
    end = source.index('elif gemm_type == "fp8":', start)
    branch_source = source[start:end]

    assert "_prepare_fp8_block_weights(" in branch_source
    assert "requant_weight_ue8m0" not in branch_source
    assert "torch.randn(scale_shape(" not in branch_source

    def_start = branch_source.index("def gemm_op():")
    gemm_op_source = branch_source[def_start:]

    assert "sglang_per_token_group_quant_fp8(" in gemm_op_source
    assert "fp8_gemm_deepgemm(" in gemm_op_source
    assert "_prepare_fp8_block_weights(" not in gemm_op_source
    assert "requant_weight_ue8m0" not in gemm_op_source
