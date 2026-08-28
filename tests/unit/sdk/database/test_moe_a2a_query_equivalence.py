# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 query-equivalence gate: ``query_moe_a2a`` vs the legacy comm tables.

Shipped-data sweeps on real databases. The legacy query facades
(``query_wideep_deepep_normal``/``ll``, ``query_trtllm_alltoall``) are
tombstones since #1357 PR-5, so the legacy side of each comparison is the
RAW loaded table row itself, probed at EXACT collected token points (three
per slice: min / median / max — off-grid interpolation and the beyond-range
util-hold retired to the compiled engine and are anchored by the frozen
parity goldens). For EVERY slice of the legacy tables the engine-routed
``query_moe_a2a`` must reproduce the raw row at rel <= 1e-9:

- sglang h200_sxm 0.5.6.post2: ``deepep_ht`` dispatch+combine == the DeepEP
  normal row (summed legs, us -> ms); ``deepep_ll`` dispatch+combine == the
  DeepEP LL row (sms=0 path).
- trtllm gb200 1.3.0rc10: every (kernel_source, op_name, moe_dtype) slice ==
  the alltoall row. Mapped comm_dtype is the slice's run dtype for
  prepare/dispatch/standard combine and "fp4" for the low-precision combine
  kernel. NO carve-outs.

The tolerance covers only float non-associativity: the unified store keeps
ms leaves (converted at adapt time) while the legacy DeepEP tables carry us.
"""

import os
from pathlib import Path

import pytest

from aiconfigurator_core.sdk import engine
from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
from aiconfigurator_core.sdk.operations.base import resolve_op_data_path
from aiconfigurator_core.sdk.operations.moe_comm import MoEAllToAll
from aiconfigurator_core.sdk.perf_database import get_database

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SYSTEMS_DATA_ROOT = REPO_ROOT / "aic-core" / "src" / "aiconfigurator_core" / "systems" / "data"

DEEPEP_NORMAL_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_deepep_normal_perf.parquet"
)
DEEPEP_LL_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_deepep_ll_perf.parquet"
)
TRTLLM_ALLTOALL_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "gb200"), "trtllm", "1.3.0rc10", "trtllm_alltoall_perf.parquet"
)

REL_TOL = 1e-9


def _query_a2a(db, comm_backend, phase, comm_dtype, ep_size, node_num, hidden, topk, experts, tok, sms=0):
    """The retired ``query_moe_a2a`` shim's exact twin: build the unified op
    and evaluate it through the engine's single-op plumbing."""
    op = MoEAllToAll(
        "moe_a2a_query",
        1.0,
        phase=phase,
        comm_backend=comm_backend,
        hidden_size=hidden,
        topk=topk,
        num_experts=experts,
        moe_ep_size=ep_size,
        node_num=node_num,
        comm_dtype=comm_dtype,
        sms=sms,
    )
    return engine._evaluate_single_op(db, op, is_context=True, batch_size=1, s=1, x=int(tok))


def _iter_slices(nested, depth):
    """Yield ``(key_tuple, node)`` for every node ``depth`` dict levels down."""
    if depth == 0:
        yield (), nested
        return
    for key, sub in nested.items():
        for rest, node in _iter_slices(sub, depth - 1):
            yield (key, *rest), node


def _exact_token_probes(token_keys):
    """Three exact collected points per slice: min / median / max."""
    keys = sorted(token_keys)
    return sorted({keys[0], keys[len(keys) // 2], keys[-1]})


# ---------------------------------------------------------------------------
# (a) sglang h200_sxm 0.5.6.post2 — DeepEP normal (HT) and low-latency (LL)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.exists(DEEPEP_NORMAL_PATH) and os.path.exists(DEEPEP_LL_PATH)),
    reason="shipped h200_sxm sglang 0.5.6.post2 DeepEP parquets not present",
)
def test_l1_deepep_query_equivalence():
    db = get_database("h200_sxm", "sglang", "0.5.6.post2", allow_unlisted_version=True)
    assert db is not None

    # Legacy table: [node][hidden][topk][experts][sms] -> {tokens: leaf (us)}.
    legacy_normal = fetch_table_view(db, "_wideep_deepep_normal_data")
    assert legacy_normal
    normal_comparisons = 0
    for (node, hidden, topk, experts, sms), tokens in _iter_slices(legacy_normal, 5):
        ep = node * 8  # legacy 8-GPU HGX fleets
        for tok in _exact_token_probes(tokens):
            unified = _query_a2a(
                db, "deepep_ht", "dispatch", "default", ep, node, hidden, topk, experts, tok, sms=sms
            ) + _query_a2a(db, "deepep_ht", "combine", "default", ep, node, hidden, topk, experts, tok, sms=sms)
            context = f"deepep_ht {node=} {hidden=} {topk=} {experts=} {sms=} {tok=}"
            assert float(unified) == pytest.approx(tokens[tok]["latency"] / 1000.0, rel=REL_TOL), context
            normal_comparisons += 1
    # _exact_token_probes dedupes min/median/max, so short token axes yield
    # fewer than three probes — count what actually ran, not 3x slices.
    assert normal_comparisons == sum(len(_exact_token_probes(t)) for _, t in _iter_slices(legacy_normal, 5))
    assert normal_comparisons > 50

    # Legacy LL table: [node][hidden][topk][experts] -> {tokens: leaf (us)}; no
    # SM budget -> unified rows live under sms=0.
    legacy_ll = fetch_table_view(db, "_wideep_deepep_ll_data")
    assert legacy_ll
    ll_comparisons = 0
    for (node, hidden, topk, experts), tokens in _iter_slices(legacy_ll, 4):
        ep = node * 8
        for tok in _exact_token_probes(tokens):
            unified = _query_a2a(
                db, "deepep_ll", "dispatch", "default", ep, node, hidden, topk, experts, tok, sms=0
            ) + _query_a2a(db, "deepep_ll", "combine", "default", ep, node, hidden, topk, experts, tok, sms=0)
            context = f"deepep_ll {node=} {hidden=} {topk=} {experts=} {tok=}"
            assert float(unified) == pytest.approx(tokens[tok]["latency"] / 1000.0, rel=REL_TOL), context
            ll_comparisons += 1
    assert ll_comparisons == sum(len(_exact_token_probes(t)) for _, t in _iter_slices(legacy_ll, 4))


# ---------------------------------------------------------------------------
# (b) trtllm gb200 1.3.0rc10 — NVLink two-sided / one-sided alltoall
# ---------------------------------------------------------------------------


# kernel_source -> unified comm_backend (the legacy moe_backend steering —
# "wideep" -> NVLinkTwoSided, None -> NVLinkOneSided on SM100 — retired with
# query_trtllm_alltoall; the engine's MoEDispatch owns kernel selection now).
_KERNEL_MAP = {
    "NVLinkTwoSided": "nvlink_two_sided",
    "NVLinkOneSided": "nvlink_one_sided",
}

# op_name -> (unified phase, pinned comm_dtype); None passes the slice's run
# dtype through (Task 3 remap: lossless 1:1).
_OP_MAP = {
    "alltoall_prepare": ("prepare", None),
    "alltoall_dispatch": ("dispatch", None),
    "alltoall_combine": ("combine", None),
    "alltoall_combine_low_precision": ("combine", "fp4"),
}


@pytest.mark.skipif(
    not os.path.exists(TRTLLM_ALLTOALL_PATH),
    reason="shipped gb200 trtllm 1.3.0rc10 alltoall parquet not present",
)
def test_l1_trtllm_alltoall_query_equivalence():
    db = get_database("gb200", "trtllm", "1.3.0rc10", allow_unlisted_version=True)
    assert db is not None

    # Legacy table: [kernel][op][quant][node][hidden][topk][experts][ep] -> {tokens: leaf (ms)}.
    legacy_table = fetch_table_view(db, "_trtllm_alltoall_data")
    assert legacy_table

    comparisons = 0
    for (kernel_source, op_name, quant, node, hidden, topk, experts, ep), tokens in _iter_slices(legacy_table, 8):
        comm_backend = _KERNEL_MAP[kernel_source]
        phase, pinned_dtype = _OP_MAP[op_name]
        comm_dtype = pinned_dtype if pinned_dtype is not None else quant.name
        for tok in _exact_token_probes(tokens):
            unified = _query_a2a(db, comm_backend, phase, comm_dtype, ep, node, hidden, topk, experts, tok, sms=0)
            context = f"{kernel_source} {op_name} {quant.name} {node=} {hidden=} {topk=} {experts=} {ep=} {tok=}"
            assert float(unified) == pytest.approx(tokens[tok]["latency"], rel=REL_TOL), context
            comparisons += 1
    assert comparisons == sum(len(_exact_token_probes(t)) for _, t in _iter_slices(legacy_table, 8))
    assert comparisons > 100


@pytest.mark.skipif(
    not os.path.exists(TRTLLM_ALLTOALL_PATH),
    reason="shipped gb200 trtllm 1.3.0rc10 alltoall parquet not present",
)
def test_l1_fp8_block_normalization_matches_legacy():
    """Caller-side dtype alias: ``fp8_block`` reuses the fp8 comm tables.

    The legacy query normalized fp8_block -> fp8 before the table walk;
    ``query_moe_a2a`` must mirror it. The slice-driven sweep above only
    probes STORED dtype keys, so this caller-visible alias needs its own
    shipped-data probe (raw fp8 rows as the expectation).
    """
    from aiconfigurator_core.sdk import common

    db = get_database("gb200", "trtllm", "1.3.0rc10", allow_unlisted_version=True)
    assert db is not None

    legacy_table = fetch_table_view(db, "_trtllm_alltoall_data")
    assert legacy_table
    dispatch_fp8 = legacy_table["NVLinkTwoSided"]["alltoall_dispatch"][common.MoEQuantMode.fp8]
    (node, hidden, topk, experts, ep), tokens = min(_iter_slices(dispatch_fp8, 5), key=lambda kv: kv[0])

    comparisons = 0
    for tok in _exact_token_probes(tokens):
        unified = _query_a2a(db, "nvlink_two_sided", "dispatch", "fp8_block", ep, node, hidden, topk, experts, tok)
        assert float(unified) == pytest.approx(tokens[tok]["latency"], rel=REL_TOL), f"fp8_block {tok=}"
        comparisons += 1
    assert comparisons > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
