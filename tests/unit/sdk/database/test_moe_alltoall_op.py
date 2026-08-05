# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified ``MoEAllToAll`` op and ``query_moe_a2a``.

Query semantics against an injected ``db._moe_a2a_data`` store (the
``__dict__``-gated bind in ``load_data`` honors pre-set attributes): token
interpolation and scale_factor, comm_dtype fallback to the sole collected
dtype, typed misses, phase/backend validation, and the silicon-only tier
contract (SOL/SOL_FULL/EMPIRICAL raise ``EmpiricalNotImplementedError``).

The shipped-data section pins the comm-family placement: the legacy comm
sources feeding ``load_moe_a2a_data`` resolve under the ``comm/`` family dir
and the comm hard-exclusion keeps them primary-only (design §6.5 rule 5).
"""

import os
from pathlib import Path

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.errors import EmpiricalNotImplementedError, PerfDataNotAvailableError
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
            # deepep_ll collected under a single non-default dtype (fallback target).
            (("deepep_ll", "dispatch", "bfloat16", *_SLICE, 0), {8: _leaf(0.40)}),
            # nvlink_two_sided prepare phase (trtllm-only phase) + a multi-dtype
            # dispatch slice for the no-fallback test.
            (("nvlink_two_sided", "prepare", "fp8", *_SLICE, 0), {16: _leaf(0.05)}),
            (("nvlink_two_sided", "dispatch", "fp8", *_SLICE, 0), {16: _leaf(0.06)}),
            (("nvlink_two_sided", "dispatch", "bfloat16", *_SLICE, 0), {16: _leaf(0.07)}),
            # nvlink_one_sided dispatch collected under fp8 only (fp8_block
            # normalization target when it is the sole dtype).
            (("nvlink_one_sided", "dispatch", "fp8", *_SLICE, 0), {16: _leaf(0.08)}),
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


def test_token_midpoint_interpolation_scales_by_scale_factor(a2a_db):
    op = _make_op(scale_factor=2.0)
    result = op.query(a2a_db, x=48)  # midpoint of the {32, 64} token curve
    assert float(result) == pytest.approx(0.15 * 2.0, rel=1e-12)
    # power lerps flat at 100 W -> energy = 100 * 0.15, scaled with latency.
    assert result.energy == pytest.approx(100.0 * 0.15 * 2.0, rel=1e-12)
    assert result.source == "silicon"


def test_exact_token_hit_returns_leaf_value(a2a_db):
    result = a2a_db.query_moe_a2a("deepep_ht", "combine", "default", 16, 2, 7168, 8, 256, 64, sms=20)
    assert float(result) == pytest.approx(0.50, rel=1e-12)
    assert result.energy == pytest.approx(100.0 * 0.50, rel=1e-12)


def test_dtype_fallback_to_sole_available(a2a_db):
    # Requested "default" is absent; "bfloat16" is the only collected dtype.
    op = _make_op(comm_backend="deepep_ll", sms=0)
    result = op.query(a2a_db, x=8)
    assert float(result) == pytest.approx(0.40, rel=1e-12)


def test_multi_dtype_missing_requested_raises(a2a_db):
    # dispatch has {fp8, bfloat16}: no sole dtype to fall back to.
    with pytest.raises(PerfDataNotAvailableError, match="nvfp4"):
        a2a_db.query_moe_a2a("nvlink_two_sided", "dispatch", "nvfp4", 16, 2, 7168, 8, 256, 16, sms=0)


def test_fp8_block_normalizes_to_fp8_when_fp8_is_sole_dtype(a2a_db):
    # fp8_block is a behavioral mode reusing the fp8 comm tables (the same
    # normalization legacy query_trtllm_alltoall applies).
    result = a2a_db.query_moe_a2a("nvlink_one_sided", "dispatch", "fp8_block", 16, 2, 7168, 8, 256, 16, sms=0)
    assert float(result) == pytest.approx(0.08, rel=1e-12)


def test_fp8_block_normalizes_to_fp8_among_multiple_dtypes(a2a_db):
    # {fp8, bfloat16} collected: the sole-dtype fallback cannot answer here,
    # so this pins the fp8_block -> fp8 aliasing specifically (the reviewer's
    # gb200 repro shape).
    result = a2a_db.query_moe_a2a("nvlink_two_sided", "dispatch", "fp8_block", 16, 2, 7168, 8, 256, 16, sms=0)
    assert float(result) == pytest.approx(0.06, rel=1e-12)


def test_exact_fp8_block_key_wins_over_normalization(a2a_db):
    # combine has BOTH fp8_block (0.09) and fp8 (0.11): exact key first.
    result = a2a_db.query_moe_a2a("nvlink_two_sided", "combine", "fp8_block", 16, 2, 7168, 8, 256, 16, sms=0)
    assert float(result) == pytest.approx(0.09, rel=1e-12)


def test_prepare_phase_query(a2a_db):
    op = _make_op(comm_backend="nvlink_two_sided", phase="prepare", comm_dtype="fp8", sms=0)
    result = op.query(a2a_db, x=16)
    assert float(result) == pytest.approx(0.05, rel=1e-12)


def test_missing_slice_raises_named_miss(a2a_db):
    with pytest.raises(PerfDataNotAvailableError, match="requested slice"):
        a2a_db.query_moe_a2a("deepep_ht", "dispatch", "default", 999, 2, 7168, 8, 256, 32, sms=20)


def test_hybrid_missing_slice_raises_empirical_not_implemented(a2a_db):
    with pytest.raises(EmpiricalNotImplementedError, match="silicon data required"):
        a2a_db.query_moe_a2a(
            "deepep_ht",
            "dispatch",
            "default",
            999,
            2,
            7168,
            8,
            256,
            32,
            sms=20,
            database_mode=common.DatabaseMode.HYBRID,
        )


# ---------------------------------------------------------------------------
# Validation and tier contract
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


@pytest.mark.parametrize("mode", [common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL, common.DatabaseMode.EMPIRICAL])
def test_estimation_tiers_raise_empirical_not_implemented(a2a_db, mode):
    with pytest.raises(EmpiricalNotImplementedError) as excinfo:
        a2a_db.query_moe_a2a("deepep_ht", "dispatch", "default", 16, 2, 7168, 8, 256, 32, sms=20, database_mode=mode)
    message = str(excinfo.value)
    assert "silicon data required (estimation tier is a planned follow-up)" in message
    # Full query context is part of the message.
    for fragment in ("deepep_ht", "dispatch", "7168", "256"):
        assert fragment in message


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
