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

``_load_create_gemm`` extends the same technique one level further: it
AST-extracts ``create_gemm`` itself (nested inside ``run_gemm``) and execs
it into a namespace that seeds its free variables (``gemm_type``, ``M``,
``N``, ``K``, ``device``) as globals. Running the real ``create_gemm``/
``gemm_op`` closures this way -- rather than pattern-matching their source
text -- lets a behavioral test catch ANY reintroduction of weight-scale
packing inside ``gemm_op``, including through a renamed import alias or a
new attribute path: the sandboxed globals resolve only the names a test
explicitly wires in, so anything else raises ``NameError`` instead of
silently slipping past a substring check.
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


FLOAT8, FLOAT32, INT32, BFLOAT16 = object(), object(), object(), object()


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


def _normalize_fake_shape(shape: tuple) -> tuple:
    """Mirror ``torch.randn``/``torch.empty``'s dual call shape.

    Real torch accepts either N size ints (``randn(M, K, ...)``, used by
    ``create_gemm``) or a single size tuple (``randn((M, K), ...)``, used by
    ``_prepare_fp8_block_weights``). Both must produce the same shape here.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def _fake_torch():
    return SimpleNamespace(
        float8_e4m3fn=FLOAT8,
        float32=FLOAT32,
        bfloat16=BFLOAT16,
        finfo=lambda _dtype: SimpleNamespace(min=-448.0, max=448.0),
        rand=lambda *shape, device=None: _FakeTensor(tuple(shape), FLOAT32),
        randn=lambda *shape, device=None, dtype=None: _FakeTensor(_normalize_fake_shape(shape), dtype),
        empty=lambda *shape, device=None, dtype=None: _FakeTensor(_normalize_fake_shape(shape), dtype),
    )


def _prepare_fp8_block_weights(namespace):
    return _load_functions(
        "_prepare_fp8_block_weights",
        "cdiv",
        "scale_shape",
        namespace=namespace,
    )["_prepare_fp8_block_weights"]


def _load_create_gemm(namespace: dict):
    """AST-extract ``create_gemm`` (nested inside ``run_gemm``) plus the
    module-level helpers its ``fp8_block`` branch calls, and exec them all
    into one shared namespace/``__globals__`` in a single ``exec`` call.

    ``create_gemm`` takes no arguments in the real module -- ``gemm_type``,
    ``M``, ``N``, ``K``, ``device`` (and ``fp4_backend``) are free variables
    closed over from ``run_gemm``'s enclosing scope. Once extracted to top
    level, those same names resolve as globals instead, so the caller seeds
    them via ``namespace``. This runs the REAL ``create_gemm``/``gemm_op``
    bodies -- including whatever names they reference -- entirely CPU-side,
    with no reimplementation of either function.
    """
    tree = ast.parse(SOURCE_PATH.read_text(), filename=str(SOURCE_PATH))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_prepare_fp8_block_weights", "cdiv", "scale_shape")
    ]
    run_gemm = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gemm")
    create_gemm = next(
        node for node in ast.walk(run_gemm) if isinstance(node, ast.FunctionDef) and node.name == "create_gemm"
    )
    loaded = dict(namespace)
    exec(compile(ast.Module(body=[*helpers, create_gemm], type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded["create_gemm"]


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


def test_fp8_block_gemm_op_never_repacks_weights_after_setup():
    """Behavioral guard added for the review finding on b813b55.

    ``test_create_gemm_fp8_block_branch_delegates_to_the_prepack_helper``
    above only checks ``gemm_op``'s source TEXT for the literal substring
    ``"requant_weight_ue8m0"``. The reviewer showed that's evadable: import
    the same function under a renamed alias (``from ... import
    requant_weight_ue8m0 as _pack_weight_scale_evasive``) and call the alias
    inside ``gemm_op`` -- no banned substring appears, so the text check
    passes while per-call packing is back.

    This test instead runs the REAL ``create_gemm``/``gemm_op`` closures
    (``_load_create_gemm``, AST-extracted like every other function this
    file tests) inside a sandbox that defines only the names the *current*
    fp8_block path is allowed to call. A counting recorder sits at the
    packing boundary (``requant_weight_ue8m0``): setup (``create_gemm()``)
    must call it exactly once, and calling the returned ``gemm_op()`` any
    number of times afterward must add zero further calls. Any
    reintroduction of packing inside ``gemm_op`` -- same name, renamed
    alias, or a new module-attribute path -- either increments the recorder
    (caught by the count) or references a name absent from this sandbox
    (caught by ``NameError``, itself a test failure) -- it cannot slip
    through unnoticed the way a source-substring check can.

    Mutation-tested (see task-1-report.md, "Fix round 1"): reintroducing the
    packing call inside ``gemm_op`` under a renamed import alias makes this
    test fail; the unmodified source makes it pass.
    """
    calls = []

    def fake_requant(w, s, block):
        calls.append((w.shape, s.shape, tuple(block)))
        return w, s.to(INT32)

    def fake_quant_fp8(a, **_kwargs):
        return _FakeTensor(a.shape, FLOAT8), _FakeTensor((1,), INT32)

    def fake_fp8_gemm_deepgemm(x_fp8, x_scale, y_fp8, y_scale, out, m, n, k):
        return out

    namespace = {
        "torch": _fake_torch(),
        "requant_weight_ue8m0": fake_requant,
        "sglang_per_token_group_quant_fp8": fake_quant_fp8,
        "fp8_gemm_deepgemm": fake_fp8_gemm_deepgemm,
        "DEEPGEMM_SCALE_UE8M0": True,
        "gemm_type": "fp8_block",
        "M": 4,
        "N": 256,
        "K": 256,
        "device": "cpu",
        "fp4_backend": None,
    }
    create_gemm = _load_create_gemm(namespace)

    gemm_op = create_gemm()
    assert len(calls) == 1  # setup-time packing, exactly once

    for _ in range(5):
        gemm_op()
    assert len(calls) == 1  # zero additional packing calls from the timed op
