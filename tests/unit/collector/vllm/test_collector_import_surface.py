# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-surface smoke tests for the three vllm collector modules bumped
0.24.0 -> 0.27.1 (AIC-1782 Task V1), mirroring
tests/unit/collector/sglang/test_collector_import_surface.py's strict-fake
pattern and its origin incident (a module-level surface that broke on the
first real 0.5.17 run, invisible to a lenient-mock test).

These tests use STRICT fakes: every ``vllm.*`` module is a plain
``types.ModuleType`` populated with only the exact names the collector
touches -- ``types.ModuleType`` raises ``AttributeError`` on anything not
explicitly set (converted to ``ImportError`` by ``from X import Y`` for a
missing name), the same failure mode the real framework produces for a
genuinely absent attribute. Two shapes are modeled per module, derived from
the sweep documented in each collect_*.py file's module-level comment block:
one matching vLLM 0.24.0's real surface, one matching 0.27.1's. Only names
that actually differ between versions are called out per-shape; everything
unchanged between versions is shared.

``torch``/``collector.vllm.utils`` are faked permissively (a bare
``types.SimpleNamespace`` for the latter, ``MagicMock`` for the former) --
neither is the subject of this bug class. ``collector/vllm/utils.py``'s own
module-level vllm imports (CacheConfig, ModelConfig, VllmConfig,
init_distributed_environment, ensure_model_parallel_initialized, cdiv,
kv_cache_dtype_str_to_dtype, AttentionBackendEnum, CommonAttentionMetadata,
FullAttentionSpec, SlidingWindowSpec, get_kv_quant_mode, ...) were
independently verified unchanged at v0.27.1 as part of this task's research
(collect_gemm.py imports from it at module level, collect_moe.py inside
run_moe_torch) -- stubbing it out here keeps each test focused on the
collector file actually under test, not on re-proving that separately-
verified surface every time.
"""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[4]
COLLECTOR_DIR = REPO_ROOT / "collector"

# HEAD at the start of AIC-1782 Task V1 -- "vs 0.24.0-shaped" on this exact
# prefix code proves each fix is version-specific, not a general break.
PRE_FIX_COMMIT = "86f07df2dd3df12179238e62f3480c7ac6ad83ee"


# ---------------------------------------------------------------------------
# Shared strict-fake-tree infrastructure (mirrors the sglang test file)
# ---------------------------------------------------------------------------
def _build_module_tree(specs: dict[str, dict]) -> dict[str, types.ModuleType]:
    """Build a strict fake module tree: every dotted path in `specs`, plus
    every implied parent prefix, becomes a plain ``types.ModuleType``.
    Pure ``sys.modules`` injection (no ``__path__``, no parent-attribute
    wiring) is sufficient for both ``import a.b.c as x`` and
    ``from a.b.c import y`` to resolve (verified for the sglang round,
    2026-08-18; reused unchanged here)."""
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
    """Stand-in for a vllm class the collector only constructs/references,
    never subclasses or introspects at import time."""

    def __init__(self, *args, **kwargs):
        pass


def _dummy(*args, **kwargs):
    return None


class _NullContext:
    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


_null_context = _NullContext()


def _install_fake_vllm(monkeypatch, specs: dict[str, dict]) -> None:
    for path, module in _build_module_tree(specs).items():
        monkeypatch.setitem(sys.modules, path, module)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    # See module docstring: collector/vllm/utils.py's own vllm surface is
    # verified separately; stub it so each test here stays focused.
    monkeypatch.setitem(
        sys.modules,
        "collector.vllm.utils",
        types.SimpleNamespace(setup_distributed=_dummy, with_exit_stack=lambda f: f),
    )


def _import_fresh(monkeypatch, dotted_name: str, source_path: Path) -> types.ModuleType:
    """Exec `source_path` under module name `dotted_name`, bypassing any
    cached entry, and let import-time exceptions propagate to the caller."""
    monkeypatch.delitem(sys.modules, dotted_name, raising=False)
    spec = importlib.util.spec_from_file_location(dotted_name, source_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, dotted_name, module)
    spec.loader.exec_module(module)
    return module


def _prefix_source_text(relative_path: str) -> str:
    """The real file's content as of PRE_FIX_COMMIT (git-blob read, no
    working-tree mutation)."""
    result = subprocess.run(
        ["git", "show", f"{PRE_FIX_COMMIT}:{relative_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# collect_moe.py -- the one surface that actually moved (FusedMoE ->
# FusedMoEFactory rename inside run_moe_torch())
# ---------------------------------------------------------------------------
_MOE_COMMON_SURFACE = {
    "vllm.version": {"__version__": "0.0.0"},
    "vllm.config": {"VllmConfig": _Dummy, "set_current_vllm_config": _null_context},
    "vllm.forward_context": {"get_forward_context": _dummy, "set_forward_context": _null_context},
    "vllm.model_executor.layers.fused_moe.experts.fallback": {"FallbackExperts": _Dummy},
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors": {
        "CompressedTensorsConfig": _Dummy
    },
    "vllm.model_executor.layers.quantization.fp8": {"Fp8Config": _Dummy},
    "vllm.model_executor.layers.quantization.modelopt": {"ModelOptFp8Config": _Dummy},
    "vllm.model_executor.layers.quantization.mxfp4": {"Mxfp4Config": _Dummy},
    "vllm.v1.worker.workspace": {"init_workspace_manager": _dummy},
}

# vllm.model_executor.layers.fused_moe.layer: the module-level FusedMoE
# factory FUNCTION at 0.24.0 (layer.py:102-103, flagged "# TODO: rename
# this" immediately above its def), renamed to FusedMoEFactory at 0.27.1
# (layer.py:99, TODO removed; fused_moe/__init__.py exports FusedMoEFactory,
# not FusedMoE -- no back-compat alias at either version).
VLLM_0240_MOE_SURFACE = {
    **_MOE_COMMON_SURFACE,
    "vllm.model_executor.layers.fused_moe.layer": {"FusedMoE": _dummy},
}
VLLM_0271_MOE_SURFACE = {
    **_MOE_COMMON_SURFACE,
    "vllm.model_executor.layers.fused_moe.layer": {"FusedMoEFactory": _dummy},
}

_COLLECT_MOE_PATH = COLLECTOR_DIR / "vllm" / "collect_moe.py"
_COLLECT_MOE_DOTTED = "collector.vllm.collect_moe"

# run_moe_torch() runs its whole vllm import preamble (10 statements, 9 of
# them vllm-rooted) as the first lines of the function body, strictly before
# any other code -- including this collector-owned, vllm-independent guard.
# Requesting moe_tp_size>1 AND moe_ep_size>1 together makes the FIRST line
# after the imports raise this exact, deterministic ValueError, so reaching
# it (rather than an ImportError/AttributeError from a fake module) proves
# every import in the preamble resolved.
_SENTINEL_KWARGS = dict(moe_tp_size=2, moe_ep_size=2, model_name="x", perf_filename="x")
_SENTINEL_ARGS = ("bfloat16", [1], 1, 1, 1, 1)
_SENTINEL_MATCH = "does not combine logical TP and EP"


class TestCollectMoeImportSurface:
    def test_imports_cleanly_against_0240_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, VLLM_0240_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)
        with pytest.raises(ValueError, match=_SENTINEL_MATCH):
            module.run_moe_torch(*_SENTINEL_ARGS, **_SENTINEL_KWARGS)

    def test_imports_cleanly_against_0271_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, VLLM_0271_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, _COLLECT_MOE_PATH)
        with pytest.raises(ValueError, match=_SENTINEL_MATCH):
            module.run_moe_torch(*_SENTINEL_ARGS, **_SENTINEL_KWARGS)

    def test_0271_shaped_fake_reproduces_a_break_on_prefix_code(self, monkeypatch, tmp_path):
        """Red-proof, per the sglang round's precedent: exec the git blob
        from immediately before this bump (PRE_FIX_COMMIT, unconditional
        ``from vllm...import FusedMoE``) against the 0.27.1-shaped fake
        (which has no ``FusedMoE`` name) and confirm it fails on import,
        not an assumption about what "should" happen."""
        scratch_file = tmp_path / "collect_moe.py"
        scratch_file.write_text(_prefix_source_text("collector/vllm/collect_moe.py"))

        _install_fake_vllm(monkeypatch, VLLM_0271_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, scratch_file)

        with pytest.raises(ImportError, match="FusedMoE"):
            module.run_moe_torch(*_SENTINEL_ARGS, **_SENTINEL_KWARGS)

    def test_prefix_code_was_fine_against_0240_shaped_vllm(self, monkeypatch, tmp_path):
        """Companion to the test above: the prefix code was never broken in
        general, only against 0.27.1's shape -- confirms the break (and this
        regression test) is version-specific."""
        scratch_file = tmp_path / "collect_moe.py"
        scratch_file.write_text(_prefix_source_text("collector/vllm/collect_moe.py"))

        _install_fake_vllm(monkeypatch, VLLM_0240_MOE_SURFACE)
        module = _import_fresh(monkeypatch, _COLLECT_MOE_DOTTED, scratch_file)

        with pytest.raises(ValueError, match=_SENTINEL_MATCH):
            module.run_moe_torch(*_SENTINEL_ARGS, **_SENTINEL_KWARGS)


# ---------------------------------------------------------------------------
# collect_gemm.py -- byte-identical framework surface at both pinned
# versions (no version branching needed); forward-looking regression guard
# ---------------------------------------------------------------------------
_GEMM_SURFACE = {
    "vllm.envs": {"VLLM_BATCH_INVARIANT": False},
    "vllm.version": {"__version__": "0.0.0"},
    "vllm._custom_ops": {"scaled_fp4_quant": _dummy},
    "vllm.config": {"VllmConfig": _Dummy, "set_current_vllm_config": _null_context},
    "vllm.model_executor.kernels.linear.scaled_mm.flashinfer": {
        "FlashInferFp8DeepGEMMDynamicBlockScaledKernel": _Dummy
    },
    "vllm.model_executor.layers.linear": {"RowParallelLinear": _Dummy},
    "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors": {
        "CompressedTensorsConfig": _Dummy
    },
    "vllm.model_executor.layers.quantization.fp8": {"Fp8Config": _Dummy},
    "vllm.utils.deep_gemm": {"per_block_cast_to_fp8": _dummy},
}

_COLLECT_GEMM_PATH = COLLECTOR_DIR / "vllm" / "collect_gemm.py"
_COLLECT_GEMM_DOTTED = "collector.vllm.collect_gemm"


class TestCollectGemmImportSurface:
    """Every surface collect_gemm.py touches -- directly and via
    collector/vllm/utils.py -- was found byte-identical or line-shift-only
    between v0.24.0 and v0.27.1 (this task's research); one fake tree
    therefore covers both pinned versions. These are the forward-looking
    regression guard, not a red/pre-fix reproduction -- there is no fix to
    prove here, unlike collect_moe.py above."""

    def test_imports_cleanly_against_0240_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, _GEMM_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GEMM_DOTTED, _COLLECT_GEMM_PATH)

    def test_imports_cleanly_against_0271_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, _GEMM_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GEMM_DOTTED, _COLLECT_GEMM_PATH)


# ---------------------------------------------------------------------------
# collect_gdn.py -- vllm.model_executor.layers.fla relocated wholesale to
# the new top-level vllm.third_party.flash_linear_attention package
# ---------------------------------------------------------------------------
_GDN_COMMON_SURFACE = {
    "vllm.version": {"__version__": "0.0.0"},
    "vllm.config": {"VllmConfig": _Dummy, "set_current_vllm_config": _null_context},
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn": {"ChunkGatedDeltaRule": _Dummy},
    "vllm.model_executor.layers.mamba.ops.causal_conv1d": {
        "causal_conv1d_fn": _dummy,
        "causal_conv1d_update": _dummy,
    },
    "vllm.v1.attention.backends.gdn_attn": {"GDNAttentionMetadata": _Dummy},
    "vllm.v1.attention.backends.utils": {"compute_causal_conv1d_metadata": _dummy},
}

# vllm.model_executor.layers.fla (0.24.0) relocated wholesale to the new
# top-level vllm.third_party.flash_linear_attention (0.27.1, verified via
# source clone: vllm/third_party/flash_linear_attention/ops/__init__.py);
# the only symbol this collector touches there,
# fused_recurrent_gated_delta_rule_packed_decode, is byte-identical at the
# new location and re-exported from ops/__init__.py at both versions.
VLLM_0240_GDN_SURFACE = {
    **_GDN_COMMON_SURFACE,
    "vllm.model_executor.layers.fla.ops": {"fused_recurrent_gated_delta_rule_packed_decode": _dummy},
}
VLLM_0271_GDN_SURFACE = {
    **_GDN_COMMON_SURFACE,
    "vllm.third_party.flash_linear_attention.ops": {"fused_recurrent_gated_delta_rule_packed_decode": _dummy},
}

_COLLECT_GDN_PATH = COLLECTOR_DIR / "vllm" / "collect_gdn.py"
_COLLECT_GDN_DOTTED = "collector.vllm.collect_gdn"


class TestCollectGdnImportSurface:
    def test_imports_cleanly_against_0240_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, VLLM_0240_GDN_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, _COLLECT_GDN_PATH)

    def test_imports_cleanly_against_0271_shaped_vllm(self, monkeypatch):
        _install_fake_vllm(monkeypatch, VLLM_0271_GDN_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, _COLLECT_GDN_PATH)

    def test_0271_shaped_fake_reproduces_a_break_on_prefix_code(self, monkeypatch, tmp_path):
        """Red-proof: the pre-fix code (unconditional import from the old
        vllm.model_executor.layers.fla.ops location) must fail on import
        against the 0.27.1-shaped fake, which only has the new
        vllm.third_party.flash_linear_attention.ops path."""
        scratch_file = tmp_path / "collect_gdn.py"
        scratch_file.write_text(_prefix_source_text("collector/vllm/collect_gdn.py"))

        _install_fake_vllm(monkeypatch, VLLM_0271_GDN_SURFACE)

        with pytest.raises(ImportError, match="fla"):
            _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, scratch_file)

    def test_prefix_code_was_fine_against_0240_shaped_vllm(self, monkeypatch, tmp_path):
        """Companion to the test above: the prefix code was never broken in
        general, only against 0.27.1's shape."""
        scratch_file = tmp_path / "collect_gdn.py"
        scratch_file.write_text(_prefix_source_text("collector/vllm/collect_gdn.py"))

        _install_fake_vllm(monkeypatch, VLLM_0240_GDN_SURFACE)
        _import_fresh(monkeypatch, _COLLECT_GDN_DOTTED, scratch_file)
