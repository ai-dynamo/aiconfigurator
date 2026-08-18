# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DeepSeek-V4 sparse-kernel infrastructure.

Covers:
  * the per-(attn_kind, mode) module loaders (test-only parsers)
  * the sparse-kernel CSV loader (paged_mqa_logits / hca_attn)
  * ``_lookup_sparse_kernel`` (exact + engine resolve + tp fallback)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import (
    load_context_dsv4_kind_module_data,
    load_dsv4_sparse_kernel_data,
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
_SPARSE_HEADER = _CTX_HEADER  # same column layout
_FLASH_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
_PRO_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_FLASH_NATIVE_HEADS = 64
_PRO_NATIVE_HEADS = 128


def _native_heads_for_model(model: str) -> int:
    return _PRO_NATIVE_HEADS if "Pro" in model else _FLASH_NATIVE_HEADS


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
    model: str = _FLASH_MODEL,
    num_heads: int | None = None,
) -> str:
    # The collector writes rank-LOCAL heads (native // tp, the unified #1429
    # convention); callers may override to simulate malformed files.
    heads = max(1, _native_heads_for_model(model) // tp) if num_heads is None else num_heads
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_{attn_kind}_context_module,"
        f"compressed_flashmla,{model},DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,{gemm},{heads},{bs},{isl},{tp},{step},{cr},{lat:.4f}"
    )


def _gen_row(
    *,
    attn_kind: str,
    cr: int,
    bs: int,
    isl: int,
    step: int,
    tp: int,
    gemm: str = "fp8_block",
    lat: float = 0.1,
    model: str = _FLASH_MODEL,
    num_heads: int | None = None,
    version: str = "test",
) -> str:
    heads = max(1, _native_heads_for_model(model) // tp) if num_heads is None else num_heads
    return (
        f"SGLang,{version},NVIDIA H20-3e,dsv4_{attn_kind}_generation_module,"
        f"compressed_flashmla,{model},DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,{gemm},{heads},{bs},{isl},{tp},{step},{cr},{lat:.4f}"
    )


def _sparse_row(
    *,
    kernel: str,
    bs: int,
    isl: int,
    past_kv: int,
    tp: int,
    cr: int,
    lat: float = 0.05,
    model: str = _FLASH_MODEL,
) -> str:
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_{kernel}_module,"
        f"{kernel},{model},DeepseekV4ForCausalLM,"
        f"fp8_e4m3,fp8_e4m3,fp8_block,{_native_heads_for_model(model)},{bs},{isl},{tp},{past_kv},{cr},{lat:.4f}"
    )


def _write_csv(path, header: str, rows: list[str]) -> str:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


# ───────────────────────────────────────────────────────────────────────
# Loader: sparse-kernel CSV
# ───────────────────────────────────────────────────────────────────────


def test_load_dsv4_sparse_kernel_data_basic(tmp_path):
    rows = [
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=1024, past_kv=0, tp=1, cr=4, lat=0.10),
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=1024, past_kv=8192, tp=1, cr=4, lat=0.30),
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=8192, past_kv=0, tp=1, cr=4, lat=0.55),
    ]
    path = _write_csv(tmp_path / "paged.txt", _SPARSE_HEADER, rows)
    data = load_dsv4_sparse_kernel_data(path)
    assert data is not None
    # data[native_heads][tp][past_kv][isl][bs] = {"latency": ...}
    assert data[_FLASH_NATIVE_HEADS][1][0][1024][1]["latency"] == pytest.approx(0.10)
    assert data[_FLASH_NATIVE_HEADS][1][8192][1024][1]["latency"] == pytest.approx(0.30)
    assert data[_FLASH_NATIVE_HEADS][1][0][8192][1]["latency"] == pytest.approx(0.55)


def test_load_dsv4_sparse_kernel_data_skips_dup_headers(tmp_path):
    """Loader must skip CSV header lines mistakenly appended on re-runs."""
    rows = [
        _sparse_row(kernel="hca_attn", bs=1, isl=1024, past_kv=0, tp=1, cr=128, lat=0.5),
        _SPARSE_HEADER,  # duplicate header
        _sparse_row(kernel="hca_attn", bs=1, isl=2048, past_kv=0, tp=1, cr=128, lat=0.7),
    ]
    path = _write_csv(tmp_path / "hca_dup.txt", _SPARSE_HEADER, rows)
    data = load_dsv4_sparse_kernel_data(path)
    assert data is not None
    # Both real rows present, header line silently dropped.
    assert data[_FLASH_NATIVE_HEADS][1][0][1024][1]["latency"] == pytest.approx(0.5)
    assert data[_FLASH_NATIVE_HEADS][1][0][2048][1]["latency"] == pytest.approx(0.7)


def test_load_dsv4_sparse_kernel_data_missing_returns_none(tmp_path):
    assert load_dsv4_sparse_kernel_data(str(tmp_path / "no_such.txt")) is None


# ───────────────────────────────────────────────────────────────────────
# Loader: split-by-kind module CSVs
# ───────────────────────────────────────────────────────────────────────


def test_load_context_dsv4_kind_module_data_keys_by_native_and_local_head(tmp_path):
    """The files' ``num_heads`` column is the rank-LOCAL head count (#1429
    unified convention); the loader derives the model-native count
    ``num_heads * tp_size`` and keys the head identity as [native][local].
    Axis order after the heads is [cr][prefix][s][b]."""
    # Pro native=128 sharded at tp=1/2/4/8 -> local heads 128/64/32/16.
    rows = [
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=1, lat=18.0, model=_PRO_MODEL),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=2, lat=14.0, model=_PRO_MODEL),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=4, lat=11.5, model=_PRO_MODEL),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=8, lat=10.5, model=_PRO_MODEL),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=8, step=128, lat=12.5, model=_PRO_MODEL),
    ]
    path = _write_csv(tmp_path / "csa_ctx.txt", _CTX_HEADER, rows)
    data = load_context_dsv4_kind_module_data(path)
    quant = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    # one native bucket (Pro=128) with the tp sweep as local keys
    assert set(quant.keys()) == {_PRO_NATIVE_HEADS}
    locals_ = quant[_PRO_NATIVE_HEADS]
    assert set(locals_.keys()) == {128, 64, 32, 16}
    # axis order after the heads is [cr][prefix][s][b]
    assert locals_[16][4][0][8192][1]["latency"] == pytest.approx(10.5)
    assert locals_[16][4][128][8192][1]["latency"] == pytest.approx(12.5)
    # more local heads (less sharded) is slower
    assert locals_[128][4][0][8192][1]["latency"] > locals_[16][4][0][8192][1]["latency"]


def test_load_generation_dsv4_kind_module_data_b_before_s(tmp_path):
    """Generation loader must use ``[head][b][s_total]`` (b before s).

    Generation queries resolve with axes ``(num_heads, b, s_total)`` — the
    data dict nesting must follow that axis order.
    """
    rows = [
        _gen_row(attn_kind="csa", cr=4, bs=1, isl=1, step=1023, tp=1, lat=0.1),
        _gen_row(attn_kind="csa", cr=4, bs=4, isl=1, step=1023, tp=1, lat=0.4),
        _gen_row(attn_kind="csa", cr=4, bs=4, isl=1, step=8191, tp=1, lat=1.0),
    ]
    path = _write_csv(tmp_path / "csa_gen.txt", _CTX_HEADER, rows)
    data = load_generation_dsv4_kind_module_data(path)
    # tp=1 rows -> local == native; axis order [native][local][cr][b][s_total]
    sub = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block][_FLASH_NATIVE_HEADS][_FLASH_NATIVE_HEADS][4]
    # axis order after [native][local][cr] is [b][s_total]; b first
    s_total_short = 1 + 1023  # isl + step
    s_total_long = 1 + 8191
    assert sub[1][s_total_short]["latency"] == pytest.approx(0.1)
    assert sub[4][s_total_short]["latency"] == pytest.approx(0.4)
    assert sub[4][s_total_long]["latency"] == pytest.approx(1.0)


def test_load_context_dsv4_kind_module_data_keeps_native_heads_separate(tmp_path):
    rows = [
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=1, lat=18.0, model=_FLASH_MODEL),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=1, lat=23.0, model=_PRO_MODEL),
    ]
    path = _write_csv(tmp_path / "csa_ctx_models.txt", _CTX_HEADER, rows)
    data = load_context_dsv4_kind_module_data(path)
    data = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    # [native][local][cr][prefix][s][b]; tp=1 rows -> local == native, prefix=0
    assert data[_FLASH_NATIVE_HEADS][_FLASH_NATIVE_HEADS][4][0][8192][1]["latency"] == pytest.approx(18.0)
    assert data[_PRO_NATIVE_HEADS][_PRO_NATIVE_HEADS][4][0][8192][1]["latency"] == pytest.approx(23.0)


def test_load_generation_dsv4_kind_module_data_keeps_native_heads_separate(tmp_path):
    rows = [
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=1, lat=0.2, model=_FLASH_MODEL),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=1, lat=0.6, model=_PRO_MODEL),
    ]
    path = _write_csv(tmp_path / "hca_gen_models.txt", _CTX_HEADER, rows)
    data = load_generation_dsv4_kind_module_data(path)
    data = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    # [native][local][cr][b][s_total]; tp=1 -> local == native
    assert data[_FLASH_NATIVE_HEADS][_FLASH_NATIVE_HEADS][128][1][1024]["latency"] == pytest.approx(0.2)
    assert data[_PRO_NATIVE_HEADS][_PRO_NATIVE_HEADS][128][1][1024]["latency"] == pytest.approx(0.6)


# Retired with #1357 PR-5: the ``_lookup_sparse_kernel`` query helper (tp
# fallback, past_kv linear interpolation, batch bracketing, past-frontier
# util holds) moved to the compiled engine; anchored by the frozen parity
# goldens and tests/cross_package/test_query_shim_baseline.py.


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
    """Filter to a non-V4 model → V4 ops emit zero cases (collector skips)."""
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
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "sgl-project/DeepSeek-V4-Pro-FP8",
    ],
)
def test_dsv4_test_cases_active_under_v4_filter(monkeypatch, model_path):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    from collector.case_generator import get_dsv4_csa_context_test_cases

    cases = get_dsv4_csa_context_test_cases()
    assert len(cases) > 0
    # all cases use the caller-provided DeepSeek-V4 model path
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
def test_dsv4_sparse_test_cases_emit_one_kernel_case_per_model(monkeypatch, model_path):
    """SCHEME A: sparse-kernel cases are ``[model_path, kernel]`` (one per model);
    TP is no longer a case axis — the worker fixes tp=1 internally because the
    kernel is TP-invariant."""
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    from collector.case_generator import (
        get_dsv4_hca_attn_test_cases,
        get_dsv4_paged_mqa_logits_test_cases,
    )

    paged = get_dsv4_paged_mqa_logits_test_cases()
    hca = get_dsv4_hca_attn_test_cases()
    assert {c[1] for c in paged} == {"paged_mqa_logits"}
    assert {c[1] for c in hca} == {"hca_attn"}
    assert {c[0] for c in paged} == {model_path}
    assert {c[0] for c in hca} == {model_path}


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


def test_load_dsv4_kind_module_data_rejects_stale_native_semantics(tmp_path):
    """Files still storing the pre-#1131 NATIVE convention — ``num_heads``
    constant across a tp sweep — must fail loudly instead of collapsing tp
    shards onto wrong (native, local) coordinates.  The shipped sglang 0.5.10
    tables were migrated in-place (#1429); this guards against unmigrated
    external files."""
    ctx_rows = [
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=1, lat=18.0, model=_PRO_MODEL, num_heads=128),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=8, lat=10.5, model=_PRO_MODEL, num_heads=128),
    ]
    path = _write_csv(tmp_path / "csa_ctx_stale.txt", _CTX_HEADER, ctx_rows)
    with pytest.raises(ValueError, match="pre-#1131 NATIVE semantics"):
        load_context_dsv4_kind_module_data(path)

    gen_rows = [
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.5, model=_PRO_MODEL, num_heads=128),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.3, model=_PRO_MODEL, num_heads=128),
    ]
    path = _write_csv(tmp_path / "hca_gen_stale.txt", _CTX_HEADER, gen_rows)
    with pytest.raises(ValueError, match="pre-#1131 NATIVE semantics"):
        load_generation_dsv4_kind_module_data(path)


def test_load_dsv4_kind_module_data_stale_guard_is_per_model(tmp_path):
    """The stale-NATIVE guard fingerprints each model separately: one stale
    artifact fails the load even when another model's rows are well-formed,
    and two well-formed models bucket into their own [native][local] cells."""
    stale_plus_good = [
        # Pro written native-style (constant 128 across tp) -> stale.
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.5, model=_PRO_MODEL, num_heads=128),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.3, model=_PRO_MODEL, num_heads=128),
        # Flash written local-style (64/tp) -> fine on its own.
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.4, model=_FLASH_MODEL),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.2, model=_FLASH_MODEL),
    ]
    path = _write_csv(tmp_path / "hca_gen_mixed.txt", _CTX_HEADER, stale_plus_good)
    with pytest.raises(ValueError, match="DeepSeek-V4-Pro"):
        load_generation_dsv4_kind_module_data(path)

    good_rows = [
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.5, model=_PRO_MODEL),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.3, model=_PRO_MODEL),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.4, model=_FLASH_MODEL),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.2, model=_FLASH_MODEL),
    ]
    path = _write_csv(tmp_path / "hca_gen_two_models.txt", _CTX_HEADER, good_rows)
    data = load_generation_dsv4_kind_module_data(path)
    q = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    assert q[_PRO_NATIVE_HEADS][64][128][1][1024]["latency"] == pytest.approx(0.5)  # Pro tp2 -> local 64
    assert q[_PRO_NATIVE_HEADS][16][128][1][1024]["latency"] == pytest.approx(0.3)  # Pro tp8 -> local 16
    assert q[_FLASH_NATIVE_HEADS][32][128][1][1024]["latency"] == pytest.approx(0.4)  # Flash tp2
    assert q[_FLASH_NATIVE_HEADS][8][128][1][1024]["latency"] == pytest.approx(0.2)  # Flash tp8


def test_load_dsv4_kind_module_data_stale_guard_is_per_version(tmp_path):
    """The stale-NATIVE fingerprint is checked per (model, version): the
    shared layer pools sibling-version files into one row stream, so a stale
    native-writing sibling of the SAME model must be caught even when the
    pooled union of both versions no longer looks heads-constant."""
    rows = [
        # stale sibling version: native semantics (heads constant across tp).
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.5, num_heads=64, version="0.5.10"),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.3, num_heads=64, version="0.5.10"),
        # migrated primary version: local semantics (heads = 64 // tp).
        _gen_row(attn_kind="hca", cr=128, bs=2, isl=1, step=1023, tp=2, lat=0.4, num_heads=32, version="0.5.16"),
        _gen_row(attn_kind="hca", cr=128, bs=2, isl=1, step=1023, tp=8, lat=0.2, num_heads=8, version="0.5.16"),
    ]
    path = _write_csv(tmp_path / "hca_gen_versions.txt", _CTX_HEADER, rows)
    with pytest.raises(ValueError, match=r"version='0\.5\.10'"):
        load_generation_dsv4_kind_module_data(path)

    # Two well-formed versions of the same model pool cleanly.
    good = [
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.5, version="0.5.10"),
        _gen_row(attn_kind="hca", cr=128, bs=2, isl=1, step=1023, tp=8, lat=0.2, version="0.5.16"),
    ]
    path = _write_csv(tmp_path / "hca_gen_versions_ok.txt", _CTX_HEADER, good)
    data = load_generation_dsv4_kind_module_data(path)
    q = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    assert q[_FLASH_NATIVE_HEADS][32][128][1][1024]["latency"] == pytest.approx(0.5)
    assert q[_FLASH_NATIVE_HEADS][8][128][2][1024]["latency"] == pytest.approx(0.2)


# ───────────────────────────────────────────────────────────────────────
# Shipped-data guardrail: every packaged DSV4 module table is rank-local
# ───────────────────────────────────────────────────────────────────────


def test_shipped_dsv4_module_tables_are_rank_local():
    """Repo-wide #1429 invariant: every shipped DSV4 module parquet stores
    rank-LOCAL ``num_heads`` (within one model, ``num_heads * tp_size`` is
    constant across its tp sweep).  A file regressing to the retired NATIVE
    convention would poison [native][local] keying for that system, so fail
    here before it ships.  Uses parquet column reads only — the whole scan is
    a few dozen small files."""
    pq = pytest.importorskip("pyarrow.parquet")
    import aiconfigurator_core

    data_root = Path(aiconfigurator_core.__file__).parent / "systems" / "data"
    module_files = sorted(
        p
        for p in data_root.glob("*/sparse_attention/*/*/dsv4_*_module_perf.parquet")
        if any(kind in p.name for kind in ("csa_context", "csa_generation", "hca_context", "hca_generation"))
    )
    assert module_files, f"no shipped DSV4 module tables found under {data_root}"

    offenders = []
    for path in module_files:
        table = pq.read_table(path, columns=["model", "num_heads", "tp_size"])
        pairs_by_model: dict[str, set[tuple[int, int]]] = {}
        for model, heads, tp in zip(
            table["model"].to_pylist(), table["num_heads"].to_pylist(), table["tp_size"].to_pylist(), strict=True
        ):
            pairs_by_model.setdefault(str(model), set()).add((int(heads), max(1, int(tp))))
        for model, pairs in pairs_by_model.items():
            tps = {tp for _, tp in pairs}
            heads_constant = len({h for h, _ in pairs}) == 1
            product_constant = len({h * tp for h, tp in pairs}) == 1
            if len(tps) > 1 and heads_constant and not product_constant:
                offenders.append(f"{path.relative_to(data_root)}: {model} {sorted(pairs)}")
    assert not offenders, "stale NATIVE-semantics DSV4 module rows shipped:\n" + "\n".join(offenders)


def test_load_dsv4_kind_module_data_requires_tp_size_column(tmp_path):
    """A module file with no parseable tp_size column must fail loudly: every
    row would collapse to tp=1, the stale-NATIVE fingerprint could never
    trigger, and native = num_heads * tp_size would be silently wrong."""
    header_no_tp = (
        "framework,version,device,op_name,kernel_source,model,architecture,"
        "mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,"
        "step,compress_ratio,latency"
    )
    row = (
        f"SGLang,test,NVIDIA H20-3e,dsv4_hca_generation_module,"
        f"compressed_flashmla,{_FLASH_MODEL},DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,fp8_block,64,1,1,1023,128,0.1000"
    )
    path = _write_csv(tmp_path / "hca_gen_no_tp.txt", header_no_tp, [row])
    with pytest.raises(ValueError, match="tp_size"):
        load_generation_dsv4_kind_module_data(path)


# ───────────────────────────────────────────────────────────────────────
# #1460 review fixes: mixed unparseable tp_size; per-native topk calib
# ───────────────────────────────────────────────────────────────────────


def test_load_dsv4_module_rejects_mixed_unparseable_tp_size(tmp_path):
    """A row with an empty tp_size among rows that carry one must fail: the
    silent tp=1 fallback would derive native = num_heads and file the row in a
    wrong native bucket (#1460 review)."""
    rows = [
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=2, lat=14.0, model=_PRO_MODEL),
    ]
    # Same schema, tp_size cell left empty.
    broken = rows[0].rsplit(",", 3)
    broken_row = ",".join([broken[0].rsplit(",", 1)[0], "", broken[1], broken[2], broken[3]])
    path = _write_csv(tmp_path / "csa_ctx_mixed_tp.txt", _CTX_HEADER, rows + [broken_row])
    with pytest.raises(ValueError, match="without a parseable"):
        load_context_dsv4_kind_module_data(path)


# ``_build_topk_calib_from_rows`` / ``_dsv4_topk_calib_for_native`` /
# ``_dsv4_topk_delta_ms`` retired with #1357 PR-5: the compiled engine builds
# the topk-calib DELTA table from the raw rows itself (native keying and the
# #1460 no-borrow/garbage-key skip live in aic-core/rust; anchored by the
# parity goldens).
