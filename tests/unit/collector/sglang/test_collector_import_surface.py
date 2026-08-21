# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-surface smoke tests for the three sglang collector modules
(AIC-1762 Task 4c/4d follow-up, 2026-08-18).

First real-GPU 0.5.17 run: ``collect_moe.py`` died at MODULE IMPORT on every
system with ``AttributeError: module 'sglang.srt.server_args' has no
attribute '_global_server_args'``. The collector classified it and exited 0,
so the jobs looked green with no moe parquet. This was a NON-import
attribute access (a bare module-attribute read inside a module-level ``if``
guard) that the earlier per-symbol audit pattern (import statements +
explicitly-exercised APIs) never covered -- and the ``sys.modules`` fakes
used elsewhere in this test suite for capability-probe functions
(``unittest.mock.MagicMock``-based) could not have caught it either, because
``MagicMock`` auto-creates any attribute access instead of raising.

These tests use STRICT fakes instead: every ``sglang.*`` module is a plain
``types.ModuleType`` populated with only the exact names the collector
touches at import time (module-level, not inside function bodies) --
``types.ModuleType`` raises ``AttributeError`` on anything not explicitly
set, the same failure mode the real framework produces for a genuinely
absent attribute. Two shapes are modeled, derived from the sweep documented
in each collect_*.py file's module-level comment blocks: one matching
sglang 0.5.14's real surface, one matching 0.5.17's. Only the names that
actually differ between versions are called out per-shape; everything
unchanged between versions (``FusedMoE``, ``TopK``, ``Fp8Config``, etc.) is
shared, since re-verifying THEIR existence is not this failure class.

``torch``/``pkg_resources``/``sgl_kernel`` are faked permissively
(``MagicMock``) -- they are unrelated, always-absent-in-this-venv
dependencies, not the subject of this bug class; strictness is scoped to
the sglang surface specifically, per the actual incident.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[4]
COLLECTOR_DIR = REPO_ROOT / "collector"


# ---------------------------------------------------------------------------
# Shared strict-fake-tree infrastructure
# ---------------------------------------------------------------------------
def _build_module_tree(specs: dict[str, dict]) -> dict[str, types.ModuleType]:
    """Build a strict fake module tree: every dotted path in `specs`, plus
    every implied parent prefix, becomes a plain ``types.ModuleType``
    (raises ``AttributeError`` on anything not explicitly set below).
    Verified empirically (2026-08-18) that pure ``sys.modules`` injection --
    no ``__path__``, no parent-attribute wiring -- is sufficient for both
    ``import a.b.c as x`` and ``from a.b.c import y`` to resolve.
    """
    modules: dict[str, types.ModuleType] = {}

    def ensure(path: str) -> types.ModuleType:
        if path not in modules:
            modules[path] = types.ModuleType(path)
        return modules[path]

    for path, attrs in specs.items():
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            ensure(".".join(parts[:i]))
        leaf = modules[path]
        for name, value in attrs.items():
            setattr(leaf, name, value)
    return modules


class _Dummy:
    """Stand-in for an sglang class the collector only constructs/references,
    never subclasses or introspects at import time."""

    def __init__(self, *args, **kwargs):
        pass


def _dummy(*args, **kwargs):
    return None


def _install_fake_sglang(monkeypatch, specs: dict[str, dict], *, lenient_extras: tuple[str, ...] = ()) -> None:
    for path, module in _build_module_tree(specs).items():
        monkeypatch.setitem(sys.modules, path, module)
    for name in ("torch", "pkg_resources", *lenient_extras):
        monkeypatch.setitem(sys.modules, name, MagicMock())


def _import_fresh(monkeypatch, dotted_name: str, source_path: Path) -> types.ModuleType:
    """Exec `source_path` under module name `dotted_name`, bypassing any
    cached entry, and let import-time exceptions propagate to the caller."""
    monkeypatch.delitem(sys.modules, dotted_name, raising=False)
    spec = importlib.util.spec_from_file_location(dotted_name, source_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, dotted_name, module)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# collect_moe.py -- the module that actually broke
# ---------------------------------------------------------------------------
_MOE_COMMON_SURFACE = {
    "sglang.srt.layers.moe.fused_moe_triton.layer": {"FusedMoE": _Dummy},
    "sglang.srt.layers.moe.token_dispatcher.standard": {},
    "sglang.srt.layers.moe.topk": {
        "StandardTopKOutput": _Dummy,
        "TopK": _Dummy,
        "TopKConfig": _Dummy,
        "TopKOutputFormat": _Dummy,
        "select_experts": _dummy,
    },
    "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe": {},
    "sglang.srt.layers.quantization.fp8": {"Fp8Config": _Dummy},
    "sglang.srt.layers.quantization.modelopt_quant": {
        "ModelOptFp4Config": _Dummy,
        "ModelOptFp8Config": _Dummy,
    },
    "sglang.srt.layers.quantization.mxfp4": {"Mxfp4Config": _Dummy},
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe": {"fused_moe": _dummy},
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config": {
        "get_config_dtype_str": _dummy,
        "get_default_config": _dummy,
        "get_moe_configs": _dummy,
    },
    "sglang.srt.layers.moe.moe_runner.base": {"MoeRunnerConfig": _Dummy},
    "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors": {"CompressedTensorsConfig": _Dummy},
    "sglang.srt.utils": {"is_hip": _dummy},
}

# sglang.srt.server_args: 0.5.14 has a bare `_global_server_args` module
# global; 0.5.17 replaces it with get_global_server_args()/
# set_global_server_args_for_scheduler()/ServerArgs (server_args.py:
# 9294-9316 @0.5.17) and drops the bare global entirely.
# sglang.srt.runtime_context: 0.5.14's 227-line file exposes only
# get_context/get_parallel (no _server_args slot on the RuntimeContext it
# returns -- confirmed by reading the whole file, not just grepping, after
# an earlier version of this fix wrongly assumed get_context alone was a
# safe version probe); 0.5.17 adds get_flags/get_exec/get_server_args.
# fused_moe_triton_kernels: relocated from srt/layers/moe/moe_runner/
# triton_utils/ (0.5.14) to the new top-level sglang.kernels.ops.moe
# package (0.5.17).
# layers.moe.utils: MOE_RUNNER_BACKEND (0.5.14 bare global, gone at 0.5.17)
# -- not touched at import time either way, included for completeness.
SGLANG_0514_MOE_SURFACE = {
    **_MOE_COMMON_SURFACE,
    "sglang.srt.server_args": {"_global_server_args": None},
    "sglang.srt.runtime_context": {"get_context": _dummy, "get_parallel": _dummy},
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels": {"_B_DESC_CACHE": {}},
    "sglang.srt.layers.moe.utils": {
        "MoeRunnerBackend": _Dummy,
        "RoutingMethodType": _Dummy,
        "MOE_RUNNER_BACKEND": None,
    },
}

SGLANG_0517_MOE_SURFACE = {
    **_MOE_COMMON_SURFACE,
    "sglang.srt.server_args": {
        "ServerArgs": _Dummy,
        "get_global_server_args": _dummy,
        "set_global_server_args_for_scheduler": _dummy,
        "set_global_server_args_for_tokenizer": _dummy,
        # Deliberately NO _global_server_args -- the whole bug.
    },
    "sglang.srt.runtime_context": {
        "get_context": _dummy,
        "get_parallel": _dummy,
        "get_flags": _dummy,
        "get_exec": _dummy,
        "get_server_args": _dummy,
    },
    "sglang.kernels.ops.moe.fused_moe_triton_kernels": {"_B_DESC_CACHE": {}},
    "sglang.srt.layers.moe.utils": {"MoeRunnerBackend": _Dummy, "RoutingMethodType": _Dummy},
}

_COLLECT_MOE_PATH = COLLECTOR_DIR / "sglang" / "collect_moe.py"
_COLLECT_MOE_DOTTED = "collector.sglang.collect_moe"

# Verbatim pre-fix excerpt: collector/sglang/collect_moe.py lines 60-86 at
# commit f5b49140a581809b3031775883fc1c2e55132c96 (the commit immediately
# before the server_args fix), truncated right after the module-level
# `_global_server_args` mock block whose bare attribute read (line 78
# there) is the exact statement that crashed the first real-GPU 0.5.17 run.
# "vs 0.5.14-shaped" on this exact pre-fix code proves the bug is
# 0.5.17-specific, not a general break. Embedded as a string constant
# instead of a `git show` read: that commit is a branch-head object that
# squash-merge-only main and base-branch rebases leave unreachable on fresh
# clones -- this test's `git show` hard-errored in CI with
# CalledProcessError 128 after a base-branch rebase orphaned the pinned
# commit.
_PRE_FIX_MOE_SNIPPET = """\
import gc
import importlib
import itertools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import MagicMock

import pkg_resources

# Mock global server args before importing MOE modules (required by SGLang 0.5.5+)
# The fused_moe_triton_config module now requires get_global_server_args() to be set
import sglang.srt.server_args as _server_args_module
import torch

if _server_args_module._global_server_args is None:
    _mock_server_args = MagicMock()
    _mock_server_args.enable_deterministic_inference = False
    _mock_server_args.enable_fused_moe_sum_all_reduce = (
        False  # SGLang 0.5.14; prevents fused all-reduce in single-GPU benchmarks
    )
    _mock_server_args.kt_weight_path = None
    _mock_server_args.flashinfer_mxfp4_moe_precision = "default"
    _server_args_module._global_server_args = _mock_server_args
"""


class TestCollectMoeImportSurface:
    def test_imports_cleanly_against_0514_shaped_sglang(self, monkeypatch):
        _install_fake_sglang(monkeypatch, SGLANG_0514_MOE_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)

    def test_imports_cleanly_against_0517_shaped_sglang(self, monkeypatch):
        _install_fake_sglang(monkeypatch, SGLANG_0517_MOE_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)

    def test_0517_shaped_fake_reproduces_the_real_bug_on_prefix_code(self, monkeypatch, tmp_path):
        """Proves red on a scratch revert, per the review's explicit ask:
        exec the embedded verbatim pre-fix excerpt (_PRE_FIX_MOE_SNIPPET)
        against the exact same 0.5.17-shaped fake the tests above use, and
        confirm it fails with the exact real-world AttributeError -- not an
        assumption about what "should" happen."""
        scratch_file = tmp_path / "collect_moe.py"
        scratch_file.write_text(_PRE_FIX_MOE_SNIPPET)

        _install_fake_sglang(monkeypatch, SGLANG_0517_MOE_SURFACE)

        with pytest.raises(AttributeError, match="_global_server_args"):
            _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, scratch_file)

    def test_prefix_code_was_fine_against_0514_shaped_sglang(self, monkeypatch, tmp_path):
        """Companion to the test above: the pre-fix code was never broken in
        general, only against 0.5.17's shape -- confirms the bug (and this
        regression test) is version-specific, not a false alarm that would
        have failed either way."""
        scratch_file = tmp_path / "collect_moe.py"
        scratch_file.write_text(_PRE_FIX_MOE_SNIPPET)

        _install_fake_sglang(monkeypatch, SGLANG_0514_MOE_SURFACE)

        _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, scratch_file)


# ---------------------------------------------------------------------------
# collect_gemm.py
# ---------------------------------------------------------------------------
_GEMM_COMMON_SURFACE = {
    "sglang.srt.layers.deep_gemm_wrapper": {"DEEPGEMM_SCALE_UE8M0": False, "gemm_nt_f8f8bf16": _dummy},
    "sglang.srt.layers.quantization.fp8_utils": {"requant_weight_ue8m0": _dummy},
}

# fp8_kernel.py (sglang_per_token_group_quant_fp8): relocated from
# srt/layers/quantization/ (0.5.14) to the new top-level
# sglang.kernels.ops.quantization package (0.5.17).
SGLANG_0514_GEMM_SURFACE = {
    **_GEMM_COMMON_SURFACE,
    "sglang.srt.layers.quantization.fp8_kernel": {"sglang_per_token_group_quant_fp8": _dummy},
}
SGLANG_0517_GEMM_SURFACE = {
    **_GEMM_COMMON_SURFACE,
    "sglang.kernels.ops.quantization.fp8_kernel": {"sglang_per_token_group_quant_fp8": _dummy},
}

_COLLECT_GEMM_PATH = COLLECTOR_DIR / "sglang" / "collect_gemm.py"
_COLLECT_GEMM_DOTTED = "collector.sglang.collect_gemm"
# collect_gemm.py does `import torch.nn.functional as F` -- needs real
# submodule entries in sys.modules, not just attribute access on a mocked
# `torch`, plus the separate sgl_kernel package it unconditionally imports.
_GEMM_LENIENT_EXTRAS = ("torch.nn", "torch.nn.functional", "sgl_kernel")


class TestCollectGemmImportSurface:
    """gemm ran clean end-to-end on real GPU hardware (per the report this
    task responds to); these tests are the forward-looking regression guard
    -- no prefix-code bug to reproduce here, unlike collect_moe.py."""

    def test_imports_cleanly_against_0514_shaped_sglang(self, monkeypatch):
        _install_fake_sglang(monkeypatch, SGLANG_0514_GEMM_SURFACE, lenient_extras=_GEMM_LENIENT_EXTRAS)
        _import_fresh(monkeypatch, _COLLECT_GEMM_DOTTED, _COLLECT_GEMM_PATH)

    def test_imports_cleanly_against_0517_shaped_sglang(self, monkeypatch):
        _install_fake_sglang(monkeypatch, SGLANG_0517_GEMM_SURFACE, lenient_extras=_GEMM_LENIENT_EXTRAS)
        _import_fresh(monkeypatch, _COLLECT_GEMM_DOTTED, _COLLECT_GEMM_PATH)


# ---------------------------------------------------------------------------
# collect_gdn.py
# ---------------------------------------------------------------------------
_COLLECT_GDN_PATH = COLLECTOR_DIR / "sglang" / "collect_gdn.py"
_COLLECT_GDN_DOTTED = "collector.sglang.collect_gdn"


class TestCollectGdnImportSurface:
    def test_imports_cleanly_with_zero_sglang_modules_present(self, monkeypatch):
        """collect_gdn.py's only sglang references at module level are
        inside `if TYPE_CHECKING:` (inert at runtime) -- every real sglang
        import is deferred to call time, inside run_gdn_torch() (already
        covered by this file's own version-conditional-import fix and gdn's
        clean real-GPU run). This test locks that structural fact: importing
        the module must not require sglang to be present AT ALL. If a
        future change moved an sglang touch to module level without a
        version guard, this test would start failing the same way
        collect_moe.py's did."""
        monkeypatch.delitem(sys.modules, "sglang", raising=False)
        _install_fake_sglang(monkeypatch, {})  # torch/pkg_resources only, no sglang.* entries
        _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, _COLLECT_GDN_PATH)
