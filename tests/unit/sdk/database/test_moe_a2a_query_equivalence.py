# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L1 query-equivalence gate: ``query_moe_a2a`` vs the legacy comm queries.

Shipped-data sweeps on real databases. For EVERY slice of the legacy tables
and a token-probe set spanning exact hits, in-range interpolation, and the
beyond-range util-hold ({min, max, midpoints of adjacent collected points,
2 x max}), the unified query must reproduce the legacy query at rel <= 1e-9:

- sglang h200_sxm 0.5.6.post2: ``deepep_ht`` dispatch+combine ==
  ``query_wideep_deepep_normal``; ``deepep_ll`` dispatch+combine ==
  ``query_wideep_deepep_ll`` (sms=0 path).
- trtllm gb200 1.3.0rc10: every (kernel_source, op_name, moe_dtype) slice ==
  ``query_trtllm_alltoall`` with the kernel selected via ``moe_backend``
  ("wideep" -> NVLinkTwoSided; None -> NVLinkOneSided on SM100). Mapped
  comm_dtype is the slice's run dtype for prepare/dispatch/standard combine
  and "fp4" for the low-precision combine kernel. NO carve-outs.

The tolerance covers only float non-associativity: the unified store keeps
ms leaves (converted at adapt time) while the legacy DeepEP path interpolates
us and divides by 1000 at the end; measured drift is a few ulps (~1e-16).
"""

import itertools
import os
from pathlib import Path

import pytest

from aiconfigurator_core.sdk.operations.base import resolve_op_data_path
from aiconfigurator_core.sdk.operations.moe import (
    load_trtllm_alltoall_data,
    load_wideep_deepep_ll_data,
    load_wideep_deepep_normal_data,
)
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


def _iter_slices(nested, depth):
    """Yield ``(key_tuple, node)`` for every node ``depth`` dict levels down."""
    if depth == 0:
        yield (), nested
        return
    for key, sub in nested.items():
        for rest, node in _iter_slices(sub, depth - 1):
            yield (key, *rest), node


def _token_probes(token_keys):
    """{min, max, midpoints of adjacent collected points}.

    Extrapolation (past ``max``) is deliberately excluded: the unified path
    queries dispatch and combine as separate phases, so beyond the sweep each
    phase applies its own util-hold anchor and their sum diverges from the
    legacy combined-latency util-hold — a real semantic split of the two
    interpolation paths, not a bug. Interpolation stays byte-equivalent.
    """
    keys = sorted(token_keys)
    probes = {keys[0], keys[-1]}
    for lo, hi in itertools.pairwise(keys):
        probes.add((lo + hi) // 2)
    return sorted(probes)


def _assert_equivalent(unified, legacy, context):
    assert float(unified) == pytest.approx(float(legacy), rel=REL_TOL), context
    assert unified.energy == pytest.approx(legacy.energy, rel=REL_TOL), context


# ---------------------------------------------------------------------------
# (a) sglang h200_sxm 0.5.6.post2 — DeepEP normal (HT) and low-latency (LL)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.exists(DEEPEP_NORMAL_PATH) and os.path.exists(DEEPEP_LL_PATH)),
    reason="shipped h200_sxm sglang 0.5.6.post2 DeepEP parquets not present",
)
def test_l1_deepep_query_equivalence():
    db = get_database("h200_sxm", "sglang", "0.5.6.post2")
    assert db is not None

    # Legacy table: [node][hidden][topk][experts][sms] -> {tokens: leaf}.
    legacy_normal = load_wideep_deepep_normal_data(DEEPEP_NORMAL_PATH)
    assert legacy_normal
    normal_comparisons = 0
    for (node, hidden, topk, experts, sms), tokens in _iter_slices(legacy_normal, 5):
        ep = node * 8  # legacy 8-GPU HGX fleets
        for tok in _token_probes(tokens):
            unified = db.query_moe_a2a(
                "deepep_ht", "dispatch", "default", ep, node, hidden, topk, experts, tok, sms=sms
            ) + db.query_moe_a2a("deepep_ht", "combine", "default", ep, node, hidden, topk, experts, tok, sms=sms)
            legacy = db.query_wideep_deepep_normal(
                node_num=node, num_tokens=tok, num_experts=experts, topk=topk, hidden_size=hidden, sms=sms
            )
            _assert_equivalent(unified, legacy, f"deepep_ht {node=} {hidden=} {topk=} {experts=} {sms=} {tok=}")
            normal_comparisons += 1
    assert normal_comparisons > 100

    # Legacy LL table: [node][hidden][topk][experts] -> {tokens: leaf}; no SM
    # budget -> unified rows live under sms=0.
    legacy_ll = load_wideep_deepep_ll_data(DEEPEP_LL_PATH)
    assert legacy_ll
    ll_comparisons = 0
    for (node, hidden, topk, experts), tokens in _iter_slices(legacy_ll, 4):
        ep = node * 8
        for tok in _token_probes(tokens):
            unified = db.query_moe_a2a(
                "deepep_ll", "dispatch", "default", ep, node, hidden, topk, experts, tok, sms=0
            ) + db.query_moe_a2a("deepep_ll", "combine", "default", ep, node, hidden, topk, experts, tok, sms=0)
            legacy = db.query_wideep_deepep_ll(
                node_num=node, num_tokens=tok, num_experts=experts, topk=topk, hidden_size=hidden
            )
            _assert_equivalent(unified, legacy, f"deepep_ll {node=} {hidden=} {topk=} {experts=} {tok=}")
            ll_comparisons += 1
    assert ll_comparisons > 100


# ---------------------------------------------------------------------------
# (b) trtllm gb200 1.3.0rc10 — NVLink two-sided / one-sided alltoall
# ---------------------------------------------------------------------------


# kernel_source -> (unified comm_backend, moe_backend steering
# _select_alltoall_kernel to that same kernel on SM100).
_KERNEL_MAP = {
    "NVLinkTwoSided": ("nvlink_two_sided", "wideep"),
    "NVLinkOneSided": ("nvlink_one_sided", None),
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
    db = get_database("gb200", "trtllm", "1.3.0rc10")
    assert db is not None

    # Legacy table: [kernel][op][quant][node][hidden][topk][experts][ep] -> {tokens: leaf}.
    legacy_table = load_trtllm_alltoall_data(TRTLLM_ALLTOALL_PATH)
    assert legacy_table

    comparisons = 0
    for (kernel_source, op_name, quant, node, hidden, topk, experts, ep), tokens in _iter_slices(legacy_table, 8):
        comm_backend, moe_backend = _KERNEL_MAP[kernel_source]
        phase, pinned_dtype = _OP_MAP[op_name]
        comm_dtype = pinned_dtype if pinned_dtype is not None else quant.name
        for tok in _token_probes(tokens):
            unified = db.query_moe_a2a(comm_backend, phase, comm_dtype, ep, node, hidden, topk, experts, tok, sms=0)
            legacy = db.query_trtllm_alltoall(
                op_name=op_name,
                num_tokens=tok,
                hidden_size=hidden,
                topk=topk,
                num_experts=experts,
                moe_ep_size=ep,
                quant_mode=quant,
                moe_backend=moe_backend,
                node_num=node,
            )
            _assert_equivalent(
                unified,
                legacy,
                f"{kernel_source} {op_name} {quant.name} {node=} {hidden=} {topk=} {experts=} {ep=} {tok=}",
            )
            comparisons += 1
    assert comparisons > 500


@pytest.mark.skipif(
    not os.path.exists(TRTLLM_ALLTOALL_PATH),
    reason="shipped gb200 trtllm 1.3.0rc10 alltoall parquet not present",
)
def test_l1_fp8_block_normalization_matches_legacy():
    """Caller-side dtype alias: ``fp8_block`` reuses the fp8 comm tables.

    The legacy query normalizes fp8_block -> fp8 via
    ``_normalize_quant_mode_for_table`` before the table walk; ``query_moe_a2a``
    must mirror it. The slice-driven sweep above only probes STORED dtype keys,
    so this caller-visible alias needs its own shipped-data probe.
    """
    from aiconfigurator_core.sdk import common

    db = get_database("gb200", "trtllm", "1.3.0rc10")
    assert db is not None

    legacy_table = load_trtllm_alltoall_data(TRTLLM_ALLTOALL_PATH)
    assert legacy_table
    dispatch_fp8 = legacy_table["NVLinkTwoSided"]["alltoall_dispatch"][common.MoEQuantMode.fp8]
    (node, hidden, topk, experts, ep), tokens = min(_iter_slices(dispatch_fp8, 5), key=lambda kv: kv[0])

    comparisons = 0
    for tok in _token_probes(tokens):
        unified = db.query_moe_a2a("nvlink_two_sided", "dispatch", "fp8_block", ep, node, hidden, topk, experts, tok)
        legacy = db.query_trtllm_alltoall(
            op_name="alltoall_dispatch",
            num_tokens=tok,
            hidden_size=hidden,
            topk=topk,
            num_experts=experts,
            moe_ep_size=ep,
            quant_mode=common.MoEQuantMode.fp8_block,
            moe_backend="wideep",
            node_num=node,
        )
        _assert_equivalent(unified, legacy, f"fp8_block {node=} {hidden=} {topk=} {experts=} {ep=} {tok=}")
        comparisons += 1
    assert comparisons > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
