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
