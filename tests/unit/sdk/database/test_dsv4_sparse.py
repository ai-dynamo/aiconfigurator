# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DeepSeek-V4 sparse-kernel infrastructure.

Covers:
  * the per-(attn_kind, mode) module loaders and their split-file merge
  * the sparse-kernel CSV loader (paged_mqa_logits / hca_attn)
  * ``_lookup_sparse_kernel`` (exact + engine resolve + tp fallback)
  * ``_deep_merge_dsv4_dicts`` cross-kind dict merge
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    _deep_merge_dsv4_dicts,
)
from aiconfigurator.sdk.perf_database import (
    LoadedOpData,
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
    gemm: str = "fp8_block",
    lat: float = 1.0,
    model: str = _FLASH_MODEL,
    num_heads: int | None = None,
) -> str:
    # SCHEME A: the collector writes the rank-LOCAL head count (native // tp);
    # callers may override it to simulate different shardings on one model.
    heads = _native_heads_for_model(model) // tp if num_heads is None else num_heads
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_{attn_kind}_context_module,"
        f"compressed_flashmla,{model},DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,{gemm},{heads},{bs},{isl},{tp},0,{cr},{lat:.4f}"
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
) -> str:
    return (
        f"SGLang,test,NVIDIA H20-3e,dsv4_{attn_kind}_generation_module,"
        f"compressed_flashmla,{model},DeepseekV4ForCausalLM,"
        f"bfloat16,fp8_e4m3,{gemm},{_native_heads_for_model(model)},{bs},{isl},{tp},{step},{cr},{lat:.4f}"
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


def test_load_context_dsv4_kind_module_data_keys_by_local_head(tmp_path):
    """SCHEME A: TP is folded into the rank-LOCAL ``num_heads`` (native // tp),
    so the loader keys the head axis by local head count — there is NO separate
    tp_size key. Axis order after the head is [cr][prefix][s][b]."""
    # Pro native=128 sharded at tp=1/2/4/8 -> local heads 128/64/32/16.
    rows = [
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=1, lat=18.0, model=_PRO_MODEL, num_heads=128),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=2, lat=14.0, model=_PRO_MODEL, num_heads=64),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=4, lat=11.5, model=_PRO_MODEL, num_heads=32),
        _ctx_row(attn_kind="csa", cr=4, bs=1, isl=8192, tp=8, lat=10.5, model=_PRO_MODEL, num_heads=16),
    ]
    path = _write_csv(tmp_path / "csa_ctx.txt", _CTX_HEADER, rows)
    data = load_context_dsv4_kind_module_data(path)
    quant = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    # head axis keyed by local head count {128, 64, 32, 16} (no tp_size axis)
    assert set(quant.keys()) == {128, 64, 32, 16}
    # axis order after the head is [cr][prefix][s][b]
    assert quant[16][4][0][8192][1]["latency"] == pytest.approx(10.5)
    # more local heads (less sharded) is slower
    assert quant[128][4][0][8192][1]["latency"] > quant[16][4][0][8192][1]["latency"]


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
    sub = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block][_FLASH_NATIVE_HEADS][4]
    # SCHEME A axis order after [head][cr] is [b][s_total] (no tp axis); b first
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
    # SCHEME A: [local_head][cr][prefix][s][b]; tp=1 rows -> local head == native, prefix=0
    assert data[_FLASH_NATIVE_HEADS][4][0][8192][1]["latency"] == pytest.approx(18.0)
    assert data[_PRO_NATIVE_HEADS][4][0][8192][1]["latency"] == pytest.approx(23.0)


def test_load_generation_dsv4_kind_module_data_keeps_native_heads_separate(tmp_path):
    rows = [
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=1, lat=0.2, model=_FLASH_MODEL),
        _gen_row(attn_kind="hca", cr=128, bs=1, isl=1, step=1023, tp=1, lat=0.6, model=_PRO_MODEL),
    ]
    path = _write_csv(tmp_path / "hca_gen_models.txt", _CTX_HEADER, rows)
    data = load_generation_dsv4_kind_module_data(path)
    data = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    # SCHEME A: [local_head][cr][b][s_total]; tp=1 -> local head == native, no tp axis
    assert data[_FLASH_NATIVE_HEADS][128][1][1024]["latency"] == pytest.approx(0.2)
    assert data[_PRO_NATIVE_HEADS][128][1][1024]["latency"] == pytest.approx(0.6)


# ───────────────────────────────────────────────────────────────────────
# _deep_merge_dsv4_dicts — combining csa/hca split files
# ───────────────────────────────────────────────────────────────────────


def test_deep_merge_dsv4_dicts_preserves_disjoint_keys():
    csa = {"f": {"k": {"g": {4: {"x": 1}}}}}
    hca = {"f": {"k": {"g": {128: {"x": 2}}}}}
    merged = {}
    for d in (csa, hca):
        _deep_merge_dsv4_dicts(merged, d)
    assert sorted(merged["f"]["k"]["g"].keys()) == [4, 128]
    assert merged["f"]["k"]["g"][4] == {"x": 1}
    assert merged["f"]["k"]["g"][128] == {"x": 2}


# ───────────────────────────────────────────────────────────────────────
# _lookup_dsv4_sparse_kernel — tp fallback + past_kv interp
# ───────────────────────────────────────────────────────────────────────


def _make_sparse_db_with_paged_mqa(tmp_path, *, lat_at_past0: float, lat_at_past8192: float):
    """Helper: build a minimal PerfDatabase-like stub carrying paged_mqa_logits at tp=1.

    ``_lookup_sparse_kernel`` calls ``interpolation.*`` directly rather
    than ``database._interp_*`` wrappers, so the stub only needs the
    data attribute and the per-database extracted-metrics cache slot."""
    rows = [
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=8192, past_kv=0, tp=1, cr=4, lat=lat_at_past0),
        _sparse_row(kernel="paged_mqa_logits", bs=1, isl=8192, past_kv=8192, tp=1, cr=4, lat=lat_at_past8192),
    ]
    path = _write_csv(tmp_path / "paged.txt", _SPARSE_HEADER, rows)
    data = load_dsv4_sparse_kernel_data(path)

    class _DB:
        _dsv4_sparse_kernel_data: ClassVar[dict] = {
            "paged_mqa_logits": LoadedOpData(data, None, path),
        }

    return _DB()


def _sparse_value(latency: float) -> dict[str, float]:
    return {"latency": latency}


def _sparse_sampled_batch_caps_grid(*, offset: float = 0.0) -> dict:
    """Mock sparse-kernel data with sampled DeepSeek-V4 batch caps."""
    return {
        1024: {
            1: _sparse_value(offset + 1.00),
            2: _sparse_value(offset + 3.00),
            4: _sparse_value(offset + 6.00),
            8: _sparse_value(offset + 12.00),
        },
        2048: {
            1: _sparse_value(offset + 2.00),
            2: _sparse_value(offset + 4.80),
            4: _sparse_value(offset + 8.00),
        },
        4096: {
            1: _sparse_value(offset + 3.00),
            2: _sparse_value(offset + 5.80),
        },
        8192: {
            1: _sparse_value(offset + 4.00),
        },
    }


def _make_sparse_db_from_grid(per_tp_dict: dict):
    class _DB:
        _dsv4_sparse_kernel_data: ClassVar[dict] = {
            "paged_mqa_logits": LoadedOpData(
                {_FLASH_NATIVE_HEADS: {1: per_tp_dict}},
                None,
                "mock_paged_mqa_logits",
            ),
        }

    return _DB()


def test_lookup_sparse_kernel_exact_hit(tmp_path):
    db = _make_sparse_db_with_paged_mqa(tmp_path, lat_at_past0=0.1, lat_at_past8192=0.3)
    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=0,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )
    assert val == pytest.approx(0.1)
    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=8192,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )
    assert val == pytest.approx(0.3)


def test_lookup_sparse_kernel_tp_fallback(tmp_path):
    """Caller asks tp=8 but data only has tp=1 — must fall back to tp=1."""

    db = _make_sparse_db_with_paged_mqa(tmp_path, lat_at_past0=0.1, lat_at_past8192=0.3)
    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=8192,
        tp_size=8,
        native_heads=_FLASH_NATIVE_HEADS,
    )
    assert val == pytest.approx(0.3)


def test_lookup_sparse_kernel_past_kv_linear_interp(tmp_path):
    """Bracketing past_kv values exist — return linear interp."""

    db = _make_sparse_db_with_paged_mqa(tmp_path, lat_at_past0=0.1, lat_at_past8192=0.3)
    # midpoint past_kv=4096 → expect 0.2
    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=4096,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )
    assert val == pytest.approx(0.2, rel=1e-3)


def test_lookup_sparse_kernel_uses_requested_native_heads(tmp_path):
    rows = [
        _sparse_row(kernel="hca_attn", bs=1, isl=8192, past_kv=0, tp=1, cr=128, lat=0.4, model=_FLASH_MODEL),
        _sparse_row(kernel="hca_attn", bs=1, isl=8192, past_kv=0, tp=1, cr=128, lat=0.9, model=_PRO_MODEL),
    ]
    path = _write_csv(tmp_path / "hca_models.txt", _SPARSE_HEADER, rows)
    data = load_dsv4_sparse_kernel_data(path)

    class _DB:
        _dsv4_sparse_kernel_data: ClassVar[dict] = {
            "hca_attn": LoadedOpData(data, None, path),
        }

    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        _DB(),
        kernel="hca_attn",
        bs=1,
        isl=8192,
        past_kv=0,
        tp_size=1,
        native_heads=_PRO_NATIVE_HEADS,
    )
    assert val == pytest.approx(0.9)


def test_lookup_sparse_kernel_holds_util_on_isolated_leaves():
    """Two isolated leaves that bracket past_kv but share no (isl, batch) grid.

    Neither past_kv branch can resolve (isl=1536, bs=2) in-data, so the engine
    falls to util-hold: snap outer axes to the nearest collected path
    (past=0 -> isl=1024), anchor util on its boundary leaf b=1
    (util = SOL(0,1024,1)/1.0 = 1024^2/2), and scale by the pair-count SOL at
    the query: SOL(2048,1536,2)/util = 8650752/524288 = 16.5.
    (Previously this cloud went to scattered cubic griddata.)
    """

    class _DB:
        _dsv4_sparse_kernel_data: ClassVar[dict] = {
            "paged_mqa_logits": LoadedOpData(
                {
                    _FLASH_NATIVE_HEADS: {
                        1: {
                            0: {1024: {1: _sparse_value(1.0)}},
                            4096: {2048: {2: _sparse_value(4.0)}},
                        }
                    }
                },
                None,
                "mock_paged_mqa_logits",
            ),
        }

    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        _DB(),
        kernel="paged_mqa_logits",
        bs=2,
        isl=1536,
        past_kv=2048,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )

    sol_q = 2 * (2048 * 1536 + 1536**2 / 2)
    anchor_util = 1 * (1024**2 / 2) / 1.0
    assert val == pytest.approx(sol_q / anchor_util)  # 16.5


def test_lookup_sparse_kernel_brackets_batch_and_drops_ragged_isl_branch():
    """bs=3 at isl=2682 on the batch-capped grid.

    The isl brackets are {2048, 4096}; the 4096 row is batch-capped at b=2 so
    it cannot cover bs=3 and is dropped. The surviving 2048 row brackets
    b in {2, 4} (4.8 + (8.0-4.8)/2 = 6.4), then the dropped isl axis is
    corrected by the pair-count SOL ratio at the query's other coordinates:
    6.4 * SOL(0,2682,3)/SOL(0,2048,3). A plain survivor clamp measured -41%
    median on one-sided seq-row LOO folds; the SOL-ratio correction is the
    engine's single-survivor contract.
    """
    db = _make_sparse_db_from_grid({0: _sparse_sampled_batch_caps_grid()})
    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=3,
        isl=2682,
        past_kv=0,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )

    def sol(p, i, b):
        return b * (p * i + i * i / 2.0)

    b3_at_2048 = 4.80 + (8.00 - 4.80) * (3 - 2) / (4 - 2)  # 6.4
    assert val == pytest.approx(b3_at_2048 * sol(0, 2682, 3) / sol(0, 2048, 3))


def test_lookup_sparse_kernel_holds_util_beyond_all_batches():
    """bs=5 exceeds every collected batch at every isl -> util-hold.

    No isl branch covers bs=5 so in-data resolution fails entirely. The hold
    snaps isl to the nearest collected row (2048), anchors util on its
    boundary batch b=4 (util = SOL(0,2048,4)/8.0), and scales by the
    pair-count SOL at the query, so the isl growth 2048->2682 rides the
    quadratic SOL rather than a linear batch extrapolation.
    """
    db = _make_sparse_db_from_grid({0: _sparse_sampled_batch_caps_grid()})
    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=5,
        isl=2682,
        past_kv=0,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )

    anchor_util = 4 * (2048**2 / 2) / 8.00
    sol_q = 5 * (2682**2 / 2)
    assert val == pytest.approx(sol_q / anchor_util, rel=1e-4)  # ~17.15


def test_lookup_sparse_kernel_brackets_batch_within_covering_isl_row():
    """bs=5, isl=1565.2: only the isl=1024 row reaches b=8 and can bracket
    bs=5.

    The isl brackets are {1024, 2048}; the 2048 row is capped at b=4 so it
    drops, and the 1024 row bracket-blends b in {4, 8}: 6 + (12-6)/4 = 7.5 --
    a measured bracket instead of the legacy x5/4 linear batch scaling --
    then the dropped isl axis is SOL-ratio corrected (single-survivor
    contract): 7.5 * SOL(0,1565.2,5)/SOL(0,1024,5).
    """
    isl = 1565.2
    db = _make_sparse_db_from_grid({0: _sparse_sampled_batch_caps_grid()})
    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=5,
        isl=isl,
        past_kv=0,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )

    def sol(p, i, b):
        return b * (p * i + i * i / 2.0)

    b5_at_1024 = 6.00 + (12.00 - 6.00) * (5 - 4) / (8 - 4)  # 7.5
    assert val == pytest.approx(b5_at_1024 * sol(0, isl, 5) / sol(0, 1024, 5))


def test_lookup_sparse_kernel_blends_past_kv_branches():
    """past_kv=2048 midway between two collected grids blends both branches.

    Each past_kv branch resolves like the covering-isl-row case above
    (b in {4, 8} bracket at isl=1024: 7.5 and offset+4 -> 11.5), each gets
    the single-survivor SOL-ratio correction along the dropped isl axis
    (the ratio is evaluated at the QUERY's coordinates, past=2048, so it is
    common to both branches and factors out of the blend), then the past_kv
    bracket blends at weight 1/2: 9.5 * SOL(2048,1565.2,5)/SOL(2048,1024,5).
    """
    isl = 1565.2
    db = _make_sparse_db_from_grid(
        {
            0: _sparse_sampled_batch_caps_grid(offset=0.0),
            4096: _sparse_sampled_batch_caps_grid(offset=4.0),
        }
    )
    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        db,
        kernel="paged_mqa_logits",
        bs=5,
        isl=isl,
        past_kv=2048,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )

    def sol(p, i, b):
        return b * (p * i + i * i / 2.0)

    b5_at_1024 = 6.00 + (12.00 - 6.00) * (5 - 4) / (8 - 4)
    at_past_0 = b5_at_1024
    at_past_4096 = b5_at_1024 + 4.0
    blend = (at_past_0 + at_past_4096) / 2  # 9.5
    assert val == pytest.approx(blend * sol(2048, isl, 5) / sol(2048, 1024, 5))


def test_lookup_sparse_kernel_missing_returns_none():
    """Missing dict / kernel name → None (caller uses SOL ratio fallback)."""

    class _DB:
        _dsv4_sparse_kernel_data: ClassVar[dict] = {}

    val = ContextDeepSeekV4AttentionModule._lookup_sparse_kernel(
        _DB(),
        kernel="paged_mqa_logits",
        bs=1,
        isl=8192,
        past_kv=0,
        tp_size=1,
        native_heads=_FLASH_NATIVE_HEADS,
    )
    assert val is None


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
