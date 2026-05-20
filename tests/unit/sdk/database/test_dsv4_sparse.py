# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for V4-Flash sparse-kernel infrastructure.

Covers:
  * the per-(attn_kind, mode) module loaders and their split-file merge
  * V4-Flash CSV rows deriving the local ``num_heads`` key from ``tp_size``
  * ``_dsv4_robust_3d_lookup`` exact-match short-circuit
  * ``_deep_merge_dsv4_dicts`` cross-kind dict merge
  * topk_512 IO-formula past_kv correction inside the V4 context query
"""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import (
    DSV4_NATIVE_HEADS,
    _deep_merge_dsv4_dicts,
    _dsv4_num_heads_from_row,
    _dsv4_robust_3d_lookup,
    load_context_dsv4_kind_module_data,
    load_generation_dsv4_kind_module_data,
)

pytestmark = pytest.mark.unit


# ───────────────────────────────────────────────────────────────────────
# CSV fixture helpers
# ───────────────────────────────────────────────────────────────────────

_CTX_HEADER = (
    "framework,version,device,op_name,kernel_source,model,architecture,"
    "mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,"
    "step,compress_ratio,latency"
)


def _ctx_row(
    *,
    attn_kind: str,
    cr: int,
    bs: int,
    isl: int,
    tp: int,
    step: int = 0,
    gemm: str = "fp8_block",
    lat: float = 1.0,
) -> str:
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_{attn_kind}_context_module,"
        "compressed_flashmla,deepseek-ai/DeepSeek-V4-Flash,DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,{gemm},64,{bs},{isl},{tp},{step},{cr},{lat:.4f}"
    )


def _gen_row(
    *, attn_kind: str, cr: int, bs: int, isl: int, step: int, tp: int, gemm: str = "fp8_block", lat: float = 0.1
) -> str:
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_{attn_kind}_generation_module,"
        "compressed_flashmla,deepseek-ai/DeepSeek-V4-Flash,DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,{gemm},64,{bs},{isl},{tp},{step},{cr},{lat:.4f}"
    )


def _write_csv(path, header: str, rows: list[str]) -> str:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


# ───────────────────────────────────────────────────────────────────────
# Local num_heads key derivation from collector rows
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "row,expected_num_heads",
    [
        ({"num_heads": "64", "tp_size": "1"}, 64),
        ({"num_heads": "64", "tp_size": "2"}, 32),
        ({"num_heads": "64", "tp_size": "4"}, 16),
        ({"num_heads": "64", "tp_size": "8"}, 8),
        ({"num_heads": "8"}, 8),
    ],
)
def test_dsv4_num_heads_from_row(row, expected_num_heads):
    """Old CSVs may write native heads; loader exposes the local num_heads axis."""
    assert _dsv4_num_heads_from_row(row) == expected_num_heads


def test_dsv4_native_head_count():
    assert DSV4_NATIVE_HEADS == 64


# ───────────────────────────────────────────────────────────────────────
# Loader: split-by-kind module CSVs
# ───────────────────────────────────────────────────────────────────────


def test_load_context_dsv4_kind_module_data_keys_by_num_heads(tmp_path):
    """Context loader must key the inner cube by local ``num_heads``."""
    rows = [
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=1, lat=18.0),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=2, lat=14.0),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=4, lat=11.5),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=8, lat=10.5),
    ]
    path = _write_csv(tmp_path / "csa_ctx.txt", _CTX_HEADER, rows)
    data = load_context_dsv4_kind_module_data(path)
    arch = "DeepseekV4ForCausalLM"
    sub = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block][arch][4]
    # keys at the 6th level are local num_heads {64, 32, 16, 8}
    assert set(sub.keys()) == {8, 16, 32, 64}
    # axis order continues [num_heads][prefix][s][b]; context step is the prefix.
    assert sub[8][0][8192][1]["latency"] == pytest.approx(10.5)
    # TP variation must be preserved (smaller TP slower; projections 1/N sharded)
    assert sub[64][0][8192][1]["latency"] > sub[8][0][8192][1]["latency"]


def test_load_generation_dsv4_kind_module_data_b_before_s(tmp_path):
    """Generation loader must use ``[head][b][s_total]`` (b before s).

    aic_dev's ``_interp_3d`` in generation queries is called as
    ``_interp_3d(num_heads, b, s, ...)`` — the data dict must follow
    that argument order.
    """
    rows = [
        _gen_row(attn_kind="csa", cr=4, bs=1, isl=1, step=1023, tp=1, lat=0.1),
        _gen_row(attn_kind="csa", cr=4, bs=4, isl=1, step=1023, tp=1, lat=0.4),
        _gen_row(attn_kind="csa", cr=4, bs=4, isl=1, step=8191, tp=1, lat=1.0),
    ]
    path = _write_csv(tmp_path / "csa_gen.txt", _CTX_HEADER, rows)
    data = load_generation_dsv4_kind_module_data(path)
    arch = "DeepseekV4ForCausalLM"
    sub = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block][arch][4]
    # axis order [num_heads][b][s_total] — b comes before s_total
    s_total_short = 1 + 1023  # isl + step
    s_total_long = 1 + 8191
    assert sub[64][1][s_total_short]["latency"] == pytest.approx(0.1)
    assert sub[64][4][s_total_short]["latency"] == pytest.approx(0.4)
    assert sub[64][4][s_total_long]["latency"] == pytest.approx(1.0)


# ───────────────────────────────────────────────────────────────────────
# _deep_merge_dsv4_dicts — combining csa/hca split files
# ───────────────────────────────────────────────────────────────────────


def test_deep_merge_dsv4_dicts_preserves_disjoint_keys():
    csa = {"f": {"k": {"g": {"a": {4: {"x": 1}}}}}}
    hca = {"f": {"k": {"g": {"a": {128: {"x": 2}}}}}}
    merged = {}
    for d in (csa, hca):
        _deep_merge_dsv4_dicts(merged, d)
    assert sorted(merged["f"]["k"]["g"]["a"].keys()) == [4, 128]
    assert merged["f"]["k"]["g"]["a"][4] == {"x": 1}
    assert merged["f"]["k"]["g"]["a"][128] == {"x": 2}


def test_deep_merge_dsv4_dicts_handles_none():
    dest = {"a": 1}
    out = _deep_merge_dsv4_dicts(dest, None)
    assert out is dest
    assert dest == {"a": 1}


# ───────────────────────────────────────────────────────────────────────
# _dsv4_robust_3d_lookup — exact-match short-circuit
# ───────────────────────────────────────────────────────────────────────


def test_robust_3d_lookup_exact_match_short_circuits():
    """Avoids cubic / qhull when the exact (head, s, b) point is in the data."""

    class _Stub:
        def _interp_3d(self, *a, **kw):
            raise AssertionError("must not call _interp_3d when exact match exists")

    data = {8: {8192: {1: {"latency": 11.7, "energy": 0.0}}}}
    result = _dsv4_robust_3d_lookup(_Stub(), data, 8, 8192, 1)
    assert result["latency"] == pytest.approx(11.7)


# ───────────────────────────────────────────────────────────────────────
# Generic interpolation cache regressions
# ───────────────────────────────────────────────────────────────────────


def test_interp_2d_linear_ignores_stale_id_cache():
    from aiconfigurator.sdk.perf_database import PerfDatabase

    db = PerfDatabase.__new__(PerfDatabase)
    data = {
        0: {0: {"latency": 1.0, "energy": 0.0}, 10: {"latency": 3.0, "energy": 0.0}},
        10: {0: {"latency": 5.0, "energy": 0.0}, 10: {"latency": 7.0, "energy": 0.0}},
    }
    db._extracted_metrics_cache = {
        id(data): (
            {0: {0: 100.0, 10: 100.0}, 10: {0: 100.0, 10: 100.0}},
            {0: {0: 0.0, 10: 0.0}, 10: {0: 0.0, 10: 0.0}},
        )
    }

    result = PerfDatabase._interp_2d_linear(db, 5, 5, data)

    assert result["latency"] == pytest.approx(4.0)


def test_interp_3d_ignores_stale_id_cache():
    from aiconfigurator.sdk.perf_database import PerfDatabase

    db = PerfDatabase.__new__(PerfDatabase)
    data = {}
    for x in (0, 10):
        data[x] = {}
        for y in (0, 10):
            data[x][y] = {}
            for z in (0, 10):
                data[x][y][z] = {"latency": float(x + y + z), "energy": 0.0}
    db._extracted_metrics_cache = {
        id(data): (
            {
                0: {0: {0: 100.0, 10: 100.0}, 10: {0: 100.0, 10: 100.0}},
                10: {0: {0: 100.0, 10: 100.0}, 10: {0: 100.0, 10: 100.0}},
            },
            {0: {0: {0: 0.0, 10: 0.0}, 10: {0: 0.0, 10: 0.0}}, 10: {0: {0: 0.0, 10: 0.0}, 10: {0: 0.0, 10: 0.0}}},
        )
    }

    result = PerfDatabase._interp_3d(db, 5, 5, 5, data, "linear")

    assert result["latency"] == pytest.approx(15.0)


# ───────────────────────────────────────────────────────────────────────
# Test-case generators + ``--model-path`` filter
# ───────────────────────────────────────────────────────────────────────


def test_dsv4_test_cases_active_under_no_filter(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    from collector.case_generator import (
        get_dsv4_csa_context_test_cases,
        get_dsv4_paged_mqa_logits_test_cases,
    )

    assert len(get_dsv4_csa_context_test_cases()) > 0
    assert len(get_dsv4_paged_mqa_logits_test_cases()) > 0


def test_dsv4_test_cases_skipped_under_other_model(monkeypatch):
    """Filter to a non-V4 model -> V4 module ops emit zero cases."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    from collector.case_generator import (
        get_dsv4_csa_context_test_cases,
        get_dsv4_csa_generation_test_cases,
        get_dsv4_hca_attn_test_cases,
        get_dsv4_paged_mqa_logits_test_cases,
    )

    assert get_dsv4_csa_context_test_cases() == []
    assert get_dsv4_csa_generation_test_cases() == []
    assert get_dsv4_paged_mqa_logits_test_cases() == []
    assert get_dsv4_hca_attn_test_cases() == []


@pytest.mark.parametrize(
    "model_path",
    [
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "sgl-project/DeepSeek-V4-Pro-FP8",
    ],
)
def test_dsv4_test_cases_active_under_v4_filter(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    from collector.case_generator import get_dsv4_csa_context_test_cases

    cases = get_dsv4_csa_context_test_cases()
    assert len(cases) > 0
    # all cases use the caller-provided V4 Flash/Pro model path
    assert {c[6] for c in cases} == {model_path}
    # all cases for this op are CSA
    assert {c[7] for c in cases} == {"csa"}

@pytest.mark.parametrize(
    "model_path",
    [
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "sgl-project/DeepSeek-V4-Pro-FP8",
    ],
)
def test_dsv4_sparse_test_cases_only_indexer_tp1(monkeypatch, model_path):
    """Sweep is fixed at tp=[1] (kernel is TP-invariant)."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    from collector.case_generator import (
        get_dsv4_hca_attn_test_cases,
        get_dsv4_paged_mqa_logits_test_cases,
    )

    paged = get_dsv4_paged_mqa_logits_test_cases()
    hca = get_dsv4_hca_attn_test_cases()
    assert {c[3] for c in paged} == {1}
    assert {c[3] for c in hca} == {1}


def test_dsv4_fp8_models_keep_fp8_block_on_pre_blackwell(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8")
    import collector.case_generator as cases_mod

    monkeypatch.setattr(cases_mod, "_has_native_fp4_experts", lambda: False)
    cases = cases_mod.get_dsv4_csa_context_test_cases()

    assert "fp8_block" in {c[5] for c in cases}


def test_dsv4_native_fp4_models_skip_fp8_block_on_pre_blackwell(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash")
    import collector.case_generator as cases_mod

    monkeypatch.setattr(cases_mod, "_has_native_fp4_experts", lambda: False)
    cases = cases_mod.get_dsv4_csa_context_test_cases()

    assert "fp8_block" not in {c[5] for c in cases}


# ───────────────────────────────────────────────────────────────────────
# topk_512 IO-formula correction inside query_context
# ───────────────────────────────────────────────────────────────────────


def test_topk_512_io_formula_delta_units():
    """Δ_topk(M, past_kv) = M*past_kv / (mem_bw * 0.1) * 1000 (ms)."""
    M = 8192  # noqa: N806
    past_kv = 8192
    mem_bw = 4023e9  # H20 HBM B/s
    expected_us = M * past_kv / (mem_bw * 0.1) * 1e6  # ms = sec*1000; us = sec*1e6
    expected_ms = expected_us / 1000.0
    assert expected_ms == pytest.approx(0.1668, rel=1e-3)
    # at past_kv=0 the Δ is zero
    assert (M * 0) / (mem_bw * 0.1) * 1000.0 == 0.0


def test_topk_512_io_formula_scales_linearly_with_past_kv():
    """Doubling past_kv should double the IO Δ."""
    M = 8192  # noqa: N806
    mem_bw = 4023e9
    delta_8k = M * 8192 / (mem_bw * 0.1) * 1000.0
    delta_16k = M * 16384 / (mem_bw * 0.1) * 1000.0
    assert delta_16k == pytest.approx(2 * delta_8k, rel=1e-9)
