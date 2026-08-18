# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified ``MoEAllToAll`` op and ``query_moe_a2a``.

Query semantics against an injected ``db._moe_a2a_data`` store (the
``__dict__``-gated bind in ``load_data`` honors pre-set attributes): token
interpolation and scale_factor, comm_dtype fallback to the sole collected
dtype, typed misses, phase/backend validation, the silicon-only tier
contract (SOL/SOL_FULL/EMPIRICAL raise ``EmpiricalNotImplementedError``),
the ``attention_tp_size`` token divide (legacy fidelity with MoEDispatch's
``num_tokens // scale_num_tokens``), and the off-grid-sms 2-D interpolation
path against the legacy ``query_wideep_deepep_normal`` oracle.

The shipped-data section pins the comm-family placement: the legacy comm
sources feeding ``load_moe_a2a_data`` resolve under the ``comm/`` family dir
and the comm hard-exclusion keeps them primary-only (design §6.5 rule 5).
"""

import os
from pathlib import Path

import pytest

from aiconfigurator_core.sdk.operations import MoEAllToAll
from aiconfigurator_core.sdk.operations.base import resolve_op_data_path

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SYSTEMS_DATA_ROOT = REPO_ROOT / "aic-core" / "src" / "aiconfigurator_core" / "systems" / "data"

DEEPEP_NORMAL_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_deepep_normal_perf.parquet"
)
DEEPEP_LL_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_deepep_ll_perf.parquet"
)


def _leaf(latency, power=0.0):
    return {"latency": latency, "power": power, "energy": power * latency}


def _store(entries):
    """Build a nested moe_a2a store from ``(9-part key, {tokens: leaf})`` pairs."""
    data = {}
    for key, tokens in entries:
        node = data
        for part in key[:-1]:
            node = node.setdefault(part, {})
        node[key[-1]] = tokens
    return data


# Shared slice shape: ep=16, node=2, hidden=7168, topk=8, experts=256.
_SLICE = (16, 2, 7168, 8, 256)


def _build_injected_store():
    return _store(
        [
            # deepep_ht dispatch/combine: two-point token curves under sms=20.
            (
                ("deepep_ht", "dispatch", "default", *_SLICE, 20),
                {32: _leaf(0.10, power=100.0), 64: _leaf(0.20, power=100.0)},
            ),
            (
                ("deepep_ht", "combine", "default", *_SLICE, 20),
                {32: _leaf(0.30, power=100.0), 64: _leaf(0.50, power=100.0)},
            ),
            # deepep_ll collected under the sole untyped "default" slice (the
            # legacy-DeepEP shape; the one sanctioned sole-key fallback target).
            (("deepep_ll", "dispatch", "default", *_SLICE, 0), {8: _leaf(0.40)}),
            # nvlink_two_sided prepare phase (trtllm-only phase) + a multi-dtype
            # dispatch slice for the no-fallback test.
            (("nvlink_two_sided", "prepare", "fp8", *_SLICE, 0), {16: _leaf(0.05)}),
            (("nvlink_two_sided", "dispatch", "fp8", *_SLICE, 0), {16: _leaf(0.06)}),
            (("nvlink_two_sided", "dispatch", "bfloat16", *_SLICE, 0), {16: _leaf(0.07)}),
            # nvlink_one_sided dispatch collected under fp8 only (fp8_block
            # normalization target — the alias, not the sole-key fallback).
            (("nvlink_one_sided", "dispatch", "fp8", *_SLICE, 0), {16: _leaf(0.08)}),
            # nvlink_one_sided combine collected under nvfp4 only: a sole TYPED
            # key (the shipped GB200 shape) must MISS for other dtypes.
            (("nvlink_one_sided", "combine", "nvfp4", *_SLICE, 0), {16: _leaf(0.12)}),
            # combine with BOTH a real fp8_block key and an fp8 key: exact-first
            # ordering must keep the collected fp8_block row winning.
            (("nvlink_two_sided", "combine", "fp8_block", *_SLICE, 0), {16: _leaf(0.09)}),
            (("nvlink_two_sided", "combine", "fp8", *_SLICE, 0), {16: _leaf(0.11)}),
        ]
    )


@pytest.fixture
def a2a_db(stub_perf_db):
    """A stub PerfDatabase with an injected unified moe_a2a store.

    ``stub_perf_db`` warm-up already bound ``_moe_a2a_data`` (None on its
    unsupported stub backend); the assignment below replaces it and the
    ``__dict__`` gate in ``MoEAllToAll.load_data`` keeps the injected store.
    """
    stub_perf_db._moe_a2a_data = _build_injected_store()
    return stub_perf_db


def _make_op(scale_factor=1.0, **overrides):
    kwargs = {
        "phase": "dispatch",
        "comm_backend": "deepep_ht",
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "moe_ep_size": 16,
        "node_num": 2,
        "comm_dtype": "default",
        "sms": 20,
    }
    kwargs.update(overrides)
    return MoEAllToAll("test_a2a", scale_factor, **kwargs)


# ---------------------------------------------------------------------------
# Query semantics on the injected store
# ---------------------------------------------------------------------------
# Retired with #1357 PR-5 (single oracle = the compiled engine): the query
# semantics previously pinned here on the injected store — token
# interpolation/scale, exact leaf hits, comm-dtype fallback and fp8_block
# normalization, typed misses, attention_tp token division, off-grid sms 2D
# interpolation, and the estimation-tier EmpiricalNotImplementedError — live
# in aic-core/rust/.../operators/moe_a2a.rs, anchored by
# tests/cross_package/test_query_shim_baseline.py and the frozen parity
# goldens (the shims answer from DISK, so injected in-memory stores are
# invisible to them). Python-side boundary contracts stay below.
# ---------------------------------------------------------------------------


def test_ctor_rejects_unknown_backend():
    with pytest.raises(ValueError, match="comm_backend"):
        _make_op(comm_backend="bogus_backend")


def test_ctor_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        _make_op(phase="gather")


def test_ctor_and_query_reject_phase_outside_backend_comm_phases(a2a_db):
    # prepare is a known phase globally, but only the trtllm nvlink_two_sided
    # backend implements it — the registry's per-backend comm_phases must
    # reject the combination at the boundary, not as a later data miss.
    with pytest.raises(ValueError, match="does not implement phase 'prepare'"):
        _make_op(comm_backend="deepep_ht", phase="prepare")
    with pytest.raises(ValueError, match="does not implement phase 'prepare'"):
        a2a_db.query_moe_a2a("deepep_ll", "prepare", "default", 16, 2, 7168, 8, 256, 32)


def test_query_rejects_unknown_backend_and_phase(a2a_db):
    with pytest.raises(ValueError, match="comm_backend"):
        a2a_db.query_moe_a2a("bogus_backend", "dispatch", "default", 16, 2, 7168, 8, 256, 32)
    with pytest.raises(ValueError, match="phase"):
        a2a_db.query_moe_a2a("deepep_ht", "gather", "default", 16, 2, 7168, 8, 256, 32)


def test_get_weights_is_zero():
    assert _make_op().get_weights() == 0.0


# ---------------------------------------------------------------------------
# Shipped data: comm-family placement of the moe_a2a legacy sources
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.exists(DEEPEP_NORMAL_PATH) and os.path.exists(DEEPEP_LL_PATH)),
    reason="shipped h200_sxm sglang 0.5.6.post2 DeepEP parquets not present",
)
def test_shipped_legacy_comm_sources_resolve_in_comm_family_dir():
    """moe_a2a lives in the comm family: on shipped data its legacy sources
    resolve under ``<system>/comm/...`` and the comm hard-exclusion in
    ``_build_op_sources`` admits the primary only (no reuse channels)."""
    from aiconfigurator_core.sdk.perf_database import get_database

    comm_dir_fragment = f"{os.sep}comm{os.sep}"
    assert comm_dir_fragment in DEEPEP_NORMAL_PATH
    assert comm_dir_fragment in DEEPEP_LL_PATH

    db = get_database("h200_sxm", "sglang", "0.5.6.post2")
    assert db is not None
    MoEAllToAll.load_data(db)

    wrapper = db._moe_a2a_data
    assert wrapper is not None and wrapper.loaded
    # Legacy adapters fed the unified store from the comm-family files.
    assert {"deepep_ht", "deepep_ll"} <= set(wrapper.keys())

    for op_file in ("wideep_deepep_normal_perf.parquet", "wideep_deepep_ll_perf.parquet"):
        records = db.data_provenance[op_file]
        assert [record["channel"] for record in records] == ["primary"]
        assert comm_dir_fragment in records[0]["path"]
        assert records[0]["exists"]


@pytest.mark.skipif(
    not os.path.exists(DEEPEP_NORMAL_PATH),
    reason="shipped h200_sxm sglang 0.5.6.post2 DeepEP normal parquet not present",
)
def test_attention_tp_default_noop_on_shipped_l1_case():
    """Rerun one L1 sweep case through the op: with the default
    ``attention_tp_size`` the op must be byte-identical to the direct
    ``query_moe_a2a`` lookup and reproduce the legacy DeepEP-normal query
    (dispatch + combine) at the L1 tolerance."""
    from aiconfigurator_core.sdk import engine
    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
    from aiconfigurator_core.sdk.perf_database import get_database

    db = get_database("h200_sxm", "sglang", "0.5.6.post2")
    assert db is not None

    # First slice of the legacy table (via the engine view), deterministically.
    legacy_table = fetch_table_view(db, "_wideep_deepep_normal_data")
    assert legacy_table
    node = min(legacy_table)
    hidden = min(legacy_table[node])
    topk = min(legacy_table[node][hidden])
    experts = min(legacy_table[node][hidden][topk])
    sms = min(legacy_table[node][hidden][topk][experts])
    tok = min(legacy_table[node][hidden][topk][experts][sms])
    ep = node * 8  # legacy 8-GPU HGX fleets

    total = 0.0
    for phase in ("dispatch", "combine"):

        def _op(attention_tp_size: int = 1) -> MoEAllToAll:
            return MoEAllToAll(
                f"tp_noop_{phase}",
                1.0,
                phase=phase,
                comm_backend="deepep_ht",
                hidden_size=hidden,
                topk=topk,
                num_experts=experts,
                moe_ep_size=ep,
                node_num=node,
                sms=sms,
                attention_tp_size=attention_tp_size,
            )

        op_value = float(engine._evaluate_single_op(db, _op(), is_context=True, batch_size=1, s=1, x=int(tok)))
        explicit = float(
            engine._evaluate_single_op(db, _op(attention_tp_size=1), is_context=True, batch_size=1, s=1, x=int(tok))
        )
        assert op_value == explicit  # default attention_tp_size: byte-identical no-op
        total += op_value

    # query_wideep_deepep_normal is a tombstone since #1357 PR-5; the legacy
    # expectation is the RAW summed dispatch+combine leaf (us) in ms.
    legacy = legacy_table[node][hidden][topk][experts][sms][tok]["latency"] / 1000.0
    assert total == pytest.approx(legacy, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
