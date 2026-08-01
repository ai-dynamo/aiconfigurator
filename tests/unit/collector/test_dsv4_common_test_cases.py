# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest

from collector import case_generator as common_test_cases

pytestmark = pytest.mark.unit


_FLASH = "deepseek-ai/DeepSeek-V4-Flash"
_PRO = "deepseek-ai/DeepSeek-V4-Pro"
_FLASH_FP8 = "sgl-project/DeepSeek-V4-Flash-FP8"
_PRO_FP8 = "sgl-project/DeepSeek-V4-Pro-FP8"
_SUPPORTED_MODELS = (_FLASH, _PRO, _FLASH_FP8, _PRO_FP8)


def test_dsv4_context_structural_manifest_owns_model_position_admission():
    manifest = common_test_cases._dsv4_context_structural_manifest(
        batch_size=2,
        seq_lens=[16, 8, 1],
        prefix_lens=[0, 8, 16],
        max_position_embeddings=16,
    )

    assert manifest == ((0, (16, 8, 1)), (8, (8, 1)))


def test_dsv4_no_filter_expands_flash_and_pro_modules_calib_stays_canonical(monkeypatch):
    """Full/raw plans expand BOTH default artifacts for module/sparse ops —
    their tables key model geometry ([native][local] / native heads) since
    #1423/#1431 — while topk calib stays pinned to the single canonical model
    because its persisted keys carry no geometry (#1429 follow-up)."""
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    module_cases = common_test_cases.get_dsv4_csa_context_test_cases()
    assert module_cases
    assert {case[6] for case in module_cases} == {_FLASH_FP8, _PRO_FP8}
    assert common_test_cases.get_dsv4_paged_mqa_logits_test_cases() == [
        [_FLASH_FP8, "paged_mqa_logits"],
        [_PRO_FP8, "paged_mqa_logits"],
    ]
    assert common_test_cases.get_dsv4_topk_calib_test_cases() == [[_FLASH_FP8, "topk"]]


def test_dsv4_targeted_noncanonical_model_drops_calib_with_logged_reason(monkeypatch, capsys):
    """A targeted run for a non-canonical model schedules its module cases but
    zero calib cases, and the drop is logged — the calib table can hold only
    the canonical model's rows."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _PRO)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    assert common_test_cases.get_dsv4_csa_context_test_cases()
    assert common_test_cases.get_dsv4_topk_calib_test_cases() == []
    assert "calib keys carry no model geometry" in capsys.readouterr().out

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _FLASH_FP8)
    assert common_test_cases.get_dsv4_topk_calib_test_cases() == [[_FLASH_FP8, "topk"]]


@pytest.mark.parametrize("model_path", _SUPPORTED_MODELS)
def test_dsv4_attn_cases_include_same_gemm_types_for_supported_models(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setattr(common_test_cases, "_has_native_fp4_experts", lambda: True)

    cases = common_test_cases.get_dsv4_csa_context_test_cases()

    assert cases
    assert {case[5] for case in cases} == {"bfloat16", "fp8_block"}
    assert {case[6] for case in cases} == {model_path}


@pytest.mark.parametrize("model_path", _SUPPORTED_MODELS)
def test_dsv4_attn_cases_honor_model_filter(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    monkeypatch.setattr(sys, "argv", ["pytest"])

    cases = common_test_cases.get_dsv4_hca_generation_test_cases()

    assert cases
    assert {case[6] for case in cases} == {model_path}
    assert {case[7] for case in cases} == {"hca"}


def test_dsv4_sparse_smoke_cases_honor_model_filter(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _FLASH)
    monkeypatch.setattr(sys, "argv", ["collect.py", "--smoke"])

    cases = common_test_cases.get_dsv4_paged_mqa_logits_test_cases()

    # Sparse kernels are now one case per model ([model_path, kernel]); the worker
    # derives the (prefix, isl, bs) shapes 1:1 from the owning module CSV at
    # runtime instead of a separate smoke sweep grid.
    assert cases == [[_FLASH, "paged_mqa_logits"]]


@pytest.mark.parametrize("model_path", (_FLASH_FP8, _PRO_FP8))
def test_dsv4_mhc_preserves_requested_fp8_artifact(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)

    cases = common_test_cases.get_common_mhc_test_cases()

    assert len(cases) == 2
    assert {case.model_name for case in cases} == {model_path}


def test_dsv4_cases_skip_unrelated_model_filter(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "MiniMaxAI/MiniMax-M2.5")
    monkeypatch.setattr(sys, "argv", ["pytest"])

    assert common_test_cases.get_dsv4_csa_context_test_cases() == []
    assert common_test_cases.get_dsv4_hca_attn_test_cases() == []


def test_sglang_sparse_modules_topk_calib_uses_canonical_model(monkeypatch):
    """The sglang registry's topk-calib getter (deepseekv4_sparse_modules)
    applies the same canonical-calib restriction as the shared case_generator
    getter — otherwise a full/raw run would schedule a second model's calib
    case and silently overwrite the geometry-free calib table."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "collector" / "sglang" / "deepseekv4_sparse_modules.py"
    tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
    wanted = {
        "_dsv4_sparse_kernel_cases",
        "get_dsv4_topk_calib_test_cases",
        "get_dsv4_paged_mqa_logits_test_cases",
    }
    ns = {
        "_dsv4_context_derived_shapes": lambda _m: [(0, 1, 1)],
        "_dsv4_generation_derived_shapes": lambda _m: [(0, 1, 1)],
    }
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in body} == wanted
    exec(compile(ast.Module(body=body, type_ignores=[]), str(src), "exec"), ns)

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    # full/raw: sparse kernels expand both default models, calib only canonical
    assert {c[0] for c in ns["get_dsv4_paged_mqa_logits_test_cases"]()} == {_FLASH_FP8, _PRO_FP8}
    assert {c[0] for c in ns["get_dsv4_topk_calib_test_cases"]()} == {_FLASH_FP8}
    # targeted non-canonical model: calib case is dropped (logged upstream)
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", _PRO)
    assert ns["get_dsv4_topk_calib_test_cases"]() == []
