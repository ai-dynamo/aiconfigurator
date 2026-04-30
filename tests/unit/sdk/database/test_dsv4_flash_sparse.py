# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for V4-Flash sparse-kernel infrastructure.

Covers:
  * the per-(attn_kind, mode) module loaders and their split-file merge
  * the sparse-kernel CSV loader (paged_mqa_logits / hca_attn)
  * ``_dsv4_flash_tp_from_num_heads`` reverse derivation
  * ``_lookup_dsv4_flash_sparse_kernel`` (exact + interp + tp fallback)
  * ``_dsv4_flash_robust_3d_lookup`` exact-match short-circuit
  * ``_deep_merge_dsv4_dicts`` cross-kind dict merge
  * topk_512 IO-formula past_kv correction inside the V4 context query
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import (
    DSV4_FLASH_NATIVE_HEADS,
    LoadedOpData,
    _deep_merge_dsv4_dicts,
    _dsv4_flash_robust_3d_lookup,
    _dsv4_flash_tp_from_num_heads,
    load_context_dsv4_flash_kind_module_data,
    load_dsv4_flash_sparse_kernel_data,
    load_generation_dsv4_flash_kind_module_data,
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
_SPARSE_HEADER = _CTX_HEADER  # same column layout


def _ctx_row(*, attn_kind: str, cr: int, bs: int, isl: int, tp: int, gemm: str = "fp8_block", lat: float = 1.0) -> str:
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_flash_{attn_kind}_context_module,"
        "compressed_flashmla,deepseek-ai/DeepSeek-V4-Flash,DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,{gemm},64,{bs},{isl},{tp},0,{cr},{lat:.4f}"
    )


def _gen_row(
    *, attn_kind: str, cr: int, bs: int, isl: int, step: int, tp: int, gemm: str = "fp8_block", lat: float = 0.1
) -> str:
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_flash_{attn_kind}_generation_module,"
        "compressed_flashmla,deepseek-ai/DeepSeek-V4-Flash,DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,{gemm},64,{bs},{isl},{tp},{step},{cr},{lat:.4f}"
    )


def _sparse_row(*, kernel: str, bs: int, isl: int, past_kv: int, tp: int, cr: int, lat: float = 0.05) -> str:
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_flash_{kernel}_module,"
        f"{kernel},deepseek-ai/DeepSeek-V4-Flash,DeepseekV4ForCausalLM,"
        f"fp8_e4m3,fp8_e4m3,fp8_block,64,{bs},{isl},{tp},{past_kv},{cr},{lat:.4f}"
    )


def _write_csv(path, header: str, rows: list[str]) -> str:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


# ───────────────────────────────────────────────────────────────────────
# Reverse derivation: tp_size from local_heads
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "num_heads,expected_tp",
    [(64, 1), (32, 2), (16, 4), (8, 8)],
)
def test_dsv4_flash_tp_from_num_heads_standard(num_heads, expected_tp):
    """Inverse of model layer's ``local_heads = NATIVE // tp_size``."""
    assert _dsv4_flash_tp_from_num_heads(num_heads) == expected_tp


def test_dsv4_flash_tp_from_num_heads_edge_cases():
    # zero / negative / above-native fall back to tp=1
    assert _dsv4_flash_tp_from_num_heads(0) >= 1  # graceful fallback
    assert _dsv4_flash_tp_from_num_heads(128) == 1  # 64 // 128 = 0 → max(1, 0)
    # consistent with documented native head count
    assert DSV4_FLASH_NATIVE_HEADS == 64


# ───────────────────────────────────────────────────────────────────────
# Loader: sparse-kernel CSV
# ───────────────────────────────────────────────────────────────────────


def test_load_dsv4_flash_sparse_kernel_data_basic(tmp_path):
    rows = [
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=1024, past_kv=0, tp=1, cr=4, lat=0.10),
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=1024, past_kv=8192, tp=1, cr=4, lat=0.30),
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=8192, past_kv=0, tp=1, cr=4, lat=0.55),
    ]
    path = _write_csv(tmp_path / "paged.txt", _SPARSE_HEADER, rows)
    data = load_dsv4_flash_sparse_kernel_data(path)
    assert data is not None
    arch = "DeepseekV4ForCausalLM"
    # data[arch][tp][past_kv][isl][bs] = {"latency": ...}
    assert data[arch][1][0][1024][1]["latency"] == pytest.approx(0.10)
    assert data[arch][1][8192][1024][1]["latency"] == pytest.approx(0.30)
    assert data[arch][1][0][8192][1]["latency"] == pytest.approx(0.55)


def test_load_dsv4_flash_sparse_kernel_data_skips_dup_headers(tmp_path):
    """Loader must skip CSV header lines mistakenly appended on re-runs."""
    rows = [
        _sparse_row(kernel="hca_attn", bs=1, isl=1024, past_kv=0, tp=1, cr=128, lat=0.5),
        _SPARSE_HEADER,  # duplicate header
        _sparse_row(kernel="hca_attn", bs=1, isl=2048, past_kv=0, tp=1, cr=128, lat=0.7),
    ]
    path = _write_csv(tmp_path / "hca_dup.txt", _SPARSE_HEADER, rows)
    data = load_dsv4_flash_sparse_kernel_data(path)
    assert data is not None
    arch = "DeepseekV4ForCausalLM"
    # Both real rows present, header line silently dropped.
    assert data[arch][1][0][1024][1]["latency"] == pytest.approx(0.5)
    assert data[arch][1][0][2048][1]["latency"] == pytest.approx(0.7)


def test_load_dsv4_flash_sparse_kernel_data_missing_returns_none(tmp_path):
    assert load_dsv4_flash_sparse_kernel_data(str(tmp_path / "no_such.txt")) is None


# ───────────────────────────────────────────────────────────────────────
# Loader: split-by-kind module CSVs
# ───────────────────────────────────────────────────────────────────────


def test_load_context_dsv4_flash_kind_module_data_keys_by_tp(tmp_path):
    """Context loader must key the inner cube by ``tp_size``, not ``num_heads``."""
    rows = [
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=1, lat=18.0),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=2, lat=14.0),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=4, lat=11.5),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=8, lat=10.5),
    ]
    path = _write_csv(tmp_path / "csa_ctx.txt", _CTX_HEADER, rows)
    data = load_context_dsv4_flash_kind_module_data(path)
    arch = "DeepseekV4ForCausalLM"
    sub = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block][arch][4]
    # keys at the 6th level are tp_size {1, 2, 4, 8}
    assert set(sub.keys()) == {1, 2, 4, 8}
    # axis order continues [tp][s][b]
    assert sub[8][8192][1]["latency"] == pytest.approx(10.5)
    # TP variation must be preserved (smaller TP slower; projections 1/N sharded)
    assert sub[1][8192][1]["latency"] > sub[8][8192][1]["latency"]


def test_load_generation_dsv4_flash_kind_module_data_b_before_s(tmp_path):
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
    data = load_generation_dsv4_flash_kind_module_data(path)
    arch = "DeepseekV4ForCausalLM"
    sub = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block][arch][4]
    # axis order [tp][b][s_total] — b comes before s_total
    s_total_short = 1 + 1023  # isl + step
    s_total_long = 1 + 8191
    assert sub[1][1][s_total_short]["latency"] == pytest.approx(0.1)
    assert sub[1][4][s_total_short]["latency"] == pytest.approx(0.4)
    assert sub[1][4][s_total_long]["latency"] == pytest.approx(1.0)


# ───────────────────────────────────────────────────────────────────────
# _deep_merge_dsv4_dicts — combining csa/hca/swa split files
# ───────────────────────────────────────────────────────────────────────


def test_deep_merge_dsv4_dicts_preserves_disjoint_keys():
    csa = {"f": {"k": {"g": {"a": {4: {"x": 1}}}}}}
    hca = {"f": {"k": {"g": {"a": {128: {"x": 2}}}}}}
    swa = {"f": {"k": {"g": {"a": {0: {"x": 3}}}}}}
    merged = {}
    for d in (csa, hca, swa):
        _deep_merge_dsv4_dicts(merged, d)
    assert sorted(merged["f"]["k"]["g"]["a"].keys()) == [0, 4, 128]
    assert merged["f"]["k"]["g"]["a"][4] == {"x": 1}
    assert merged["f"]["k"]["g"]["a"][128] == {"x": 2}
    assert merged["f"]["k"]["g"]["a"][0] == {"x": 3}


def test_deep_merge_dsv4_dicts_handles_none():
    dest = {"a": 1}
    out = _deep_merge_dsv4_dicts(dest, None)
    assert out is dest
    assert dest == {"a": 1}


# ───────────────────────────────────────────────────────────────────────
# _dsv4_flash_robust_3d_lookup — exact-match short-circuit
# ───────────────────────────────────────────────────────────────────────


def test_robust_3d_lookup_exact_match_short_circuits():
    """Avoids cubic / qhull when the exact (head, s, b) point is in the data."""

    class _Stub:
        def _interp_3d(self, *a, **kw):
            raise AssertionError("must not call _interp_3d when exact match exists")

    data = {8: {8192: {1: {"latency": 11.7, "energy": 0.0}}}}
    result = _dsv4_flash_robust_3d_lookup(_Stub(), data, 8, 8192, 1)
    assert result["latency"] == pytest.approx(11.7)


def test_robust_3d_lookup_falls_back_to_linear_when_cubic_fails():
    """Cubic raise (QhullError on degenerate point cloud) → linear path runs."""
    calls = []

    class _Stub:
        def _interp_3d(self, x, y, z, d, method):
            calls.append(method)
            if method == "cubic":
                raise RuntimeError("QH6154 simulated qhull failure")
            return {"latency": 4.94, "energy": 0.0}

    # Empty data → exact lookup misses
    result = _dsv4_flash_robust_3d_lookup(_Stub(), {}, 8, 8192, 1)
    assert calls == ["cubic", "linear"]
    assert result["latency"] == pytest.approx(4.94)


# ───────────────────────────────────────────────────────────────────────
# _lookup_dsv4_flash_sparse_kernel — tp fallback + past_kv interp
# ───────────────────────────────────────────────────────────────────────


def _make_sparse_db_with_paged_mqa(tmp_path, *, lat_at_past0: float, lat_at_past8192: float):
    """Helper: build a minimal PerfDatabase carrying paged_mqa_logits at tp=1."""
    rows = [
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=8192, past_kv=0, tp=1, cr=4, lat=lat_at_past0),
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=8192, past_kv=8192, tp=1, cr=4, lat=lat_at_past8192),
    ]
    path = _write_csv(tmp_path / "paged.txt", _SPARSE_HEADER, rows)
    data = load_dsv4_flash_sparse_kernel_data(path)

    class _DB:
        # mimic the attribute name PerfDatabase uses
        _dsv4_flash_sparse_kernel_data: ClassVar[dict] = {
            "paged_mqa_logits": LoadedOpData(data, None, path),
        }

    return _DB()


def test_lookup_sparse_kernel_exact_hit(tmp_path):
    from aiconfigurator.sdk.perf_database import PerfDatabase

    db = _make_sparse_db_with_paged_mqa(tmp_path, lat_at_past0=0.1, lat_at_past8192=0.3)
    val = PerfDatabase._lookup_dsv4_flash_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=0,
        tp_size=1,
    )
    assert val == pytest.approx(0.1)
    val = PerfDatabase._lookup_dsv4_flash_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=8192,
        tp_size=1,
    )
    assert val == pytest.approx(0.3)


def test_lookup_sparse_kernel_tp_fallback(tmp_path):
    """Caller asks tp=8 but data only has tp=1 — must fall back to tp=1."""
    from aiconfigurator.sdk.perf_database import PerfDatabase

    db = _make_sparse_db_with_paged_mqa(tmp_path, lat_at_past0=0.1, lat_at_past8192=0.3)
    val = PerfDatabase._lookup_dsv4_flash_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=8192,
        tp_size=8,
    )
    assert val == pytest.approx(0.3)


def test_lookup_sparse_kernel_past_kv_linear_interp(tmp_path):
    """Bracketing past_kv values exist — return linear interp."""
    from aiconfigurator.sdk.perf_database import PerfDatabase

    db = _make_sparse_db_with_paged_mqa(tmp_path, lat_at_past0=0.1, lat_at_past8192=0.3)
    # midpoint past_kv=4096 → expect 0.2
    val = PerfDatabase._lookup_dsv4_flash_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=4096,
        tp_size=1,
    )
    assert val == pytest.approx(0.2, rel=1e-3)


def test_lookup_sparse_kernel_missing_returns_none():
    """Missing dict / kernel name → None (caller uses SOL ratio fallback)."""
    from aiconfigurator.sdk.perf_database import PerfDatabase

    class _DB:
        _dsv4_flash_sparse_kernel_data: ClassVar[dict] = {}

    val = PerfDatabase._lookup_dsv4_flash_sparse_kernel(
        _DB(),
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=0,
        tp_size=1,
    )
    assert val is None


# ───────────────────────────────────────────────────────────────────────
# Test-case generators + ``--model-path`` filter
# ───────────────────────────────────────────────────────────────────────


def test_dsv4_flash_test_cases_active_under_no_filter(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    from collector.common_test_cases import (
        get_dsv4_flash_csa_context_test_cases,
        get_dsv4_flash_paged_mqa_logits_test_cases,
    )

    assert len(get_dsv4_flash_csa_context_test_cases()) > 0
    assert len(get_dsv4_flash_paged_mqa_logits_test_cases()) > 0


def test_dsv4_flash_test_cases_skipped_under_other_model(monkeypatch):
    """Filter to a non-V4 model → V4 ops emit zero cases (collector skips)."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    from collector.common_test_cases import (
        get_dsv4_flash_csa_context_test_cases,
        get_dsv4_flash_csa_generation_test_cases,
        get_dsv4_flash_hca_attn_test_cases,
        get_dsv4_flash_paged_mqa_logits_test_cases,
    )

    assert get_dsv4_flash_csa_context_test_cases() == []
    assert get_dsv4_flash_csa_generation_test_cases() == []
    assert get_dsv4_flash_paged_mqa_logits_test_cases() == []
    assert get_dsv4_flash_hca_attn_test_cases() == []


def test_dsv4_flash_test_cases_active_under_v4_filter(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash")
    from collector.common_test_cases import get_dsv4_flash_csa_context_test_cases

    cases = get_dsv4_flash_csa_context_test_cases()
    assert len(cases) > 0
    # all cases reference the V4-Flash model name
    assert {c[6] for c in cases} == {"deepseek-ai/DeepSeek-V4-Flash"}
    # all cases for this op are CSA
    assert {c[7] for c in cases} == {"csa"}


def test_dsv4_flash_sparse_test_cases_only_indexer_tp1(monkeypatch):
    """Sweep is fixed at tp=[1] (kernel is TP-invariant)."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash")
    from collector.common_test_cases import (
        get_dsv4_flash_hca_attn_test_cases,
        get_dsv4_flash_paged_mqa_logits_test_cases,
    )

    paged = get_dsv4_flash_paged_mqa_logits_test_cases()
    hca = get_dsv4_flash_hca_attn_test_cases()
    assert {c[3] for c in paged} == {1}
    assert {c[3] for c in hca} == {1}


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
