# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``PerfDatabase`` coverage-probe API (PR 2's enumerator contract).

``moe_a2a_coverage``: ``comm_backend -> {(ep_size, node_num)}`` where BOTH the
dispatch AND combine phases carry a non-empty token curve for the shape (any
comm_dtype, any sms; the prepare phase is neither required nor sufficient).
``moe_expert_compute_coverage``: ``{moe_ep_size}`` with a non-empty token curve
for the shape, unioned across kernel_source/distribution/num_slots at
``moe_tp_size == 1`` (the large-EP constraint).

Both probes are read-only key walks: no query execution, no table mutation
(non-vivifying even on defaultdict-backed stores), and an absent or unloaded
table — including a ``LoadedOpData`` wrapping ``None``, whose item access
raises ``PerfDataNotAvailableError`` — yields an empty result instead of
raising.

The shipped-data section smokes the probes on real databases (h200_sxm sglang
0.5.6.post2 and gb200 trtllm 1.3.0rc10) against the legacy-adapted comm and
wideep-compute tables.
"""

import os
from pathlib import Path

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import resolve_op_data_path
from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename, get_database

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
SGLANG_CONTEXT_MOE_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_context_moe_perf.parquet"
)
TRTLLM_WIDEEP_MOE_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "gb200"), "trtllm", "1.3.0rc10", "wideep_moe_perf.parquet"
)


def _vivifying_store():
    """Local twin of the retired parsers' auto-vivifying nested defaultdict
    store (the parsers are gone; the probe contract they exposed is not)."""
    from collections import defaultdict

    def factory():
        return defaultdict(factory)

    return defaultdict(factory)


def _store_leaf(data, key, leaf):
    node = data
    for part in key[:-1]:
        node = node[part]
    node[key[-1]] = leaf


def _leaf(latency, power=0.0):
    return {"latency": latency, "power": power, "energy": power * latency}


def _store(entries):
    """Build a nested store from ``(outer key tuple, {tokens: leaf})`` pairs."""
    data = {}
    for key, tokens in entries:
        node = data
        for part in key[:-1]:
            node = node.setdefault(part, {})
        node[key[-1]] = tokens
    return data


def _key_paths(node):
    """Every key path in a nested dict, as a set of tuples (values ignored)."""
    paths = set()
    for key, sub in node.items():
        paths.add((key,))
        if isinstance(sub, dict):
            paths.update((key, *rest) for rest in _key_paths(sub))
    return paths


# Probed shape: hidden=7168, topk=8, experts=256 (DeepSeek-V3/R1).
_SHAPE = (7168, 8, 256)


# ---------------------------------------------------------------------------
# moe_a2a_coverage on a synthetic store
# ---------------------------------------------------------------------------


def _build_a2a_store():
    """9-part key order: (comm_backend, phase, comm_dtype, ep_size, node_num,
    hidden_size, topk, num_experts, sms) -> {num_tokens: leaf}."""
    return _store(
        [
            # deepep_ht (16, 2): dispatch under (default, sms=20), combine
            # under (bfloat16, sms=0) — ANY comm_dtype / ANY sms counts.
            (("deepep_ht", "dispatch", "default", 16, 2, *_SHAPE, 20), {32: _leaf(0.10)}),
            (("deepep_ht", "combine", "bfloat16", 16, 2, *_SHAPE, 0), {32: _leaf(0.30)}),
            # deepep_ht (32, 4): dispatch only -> excluded (both phases required).
            (("deepep_ht", "dispatch", "default", 32, 4, *_SHAPE, 20), {32: _leaf(0.20)}),
            # deepep_ht (64, 8): both phases present but the combine token
            # curve is EMPTY -> excluded (non-empty curves required).
            (("deepep_ht", "dispatch", "default", 64, 8, *_SHAPE, 20), {32: _leaf(0.25)}),
            (("deepep_ht", "combine", "default", 64, 8, *_SHAPE, 20), {}),
            # nvlink_two_sided (8, 2): dispatch+combine with NO prepare rows
            # -> covered (prepare is not required).
            (("nvlink_two_sided", "dispatch", "fp8", 8, 2, *_SHAPE, 0), {16: _leaf(0.05)}),
            (("nvlink_two_sided", "combine", "fp8", 8, 2, *_SHAPE, 0), {16: _leaf(0.06)}),
            # nvlink_two_sided (16, 4): prepare ONLY -> not coverage
            # (prepare is not sufficient).
            (("nvlink_two_sided", "prepare", "fp8", 16, 4, *_SHAPE, 0), {16: _leaf(0.04)}),
            # deepep_ll: dispatch and combine cover DISJOINT pairs -> no pair
            # has both phases, so the backend is omitted entirely.
            (("deepep_ll", "dispatch", "default", 16, 2, *_SHAPE, 0), {8: _leaf(0.40)}),
            (("deepep_ll", "combine", "default", 32, 4, *_SHAPE, 0), {8: _leaf(0.50)}),
        ]
    )


@pytest.fixture
def a2a_cov_db(stub_perf_db):
    """A stub PerfDatabase with an injected unified moe_a2a store.

    ``stub_perf_db`` warm-up already bound ``_moe_a2a_data`` (None on its
    unsupported stub backend); the assignment below replaces it and the
    ``__dict__`` gate in ``MoEAllToAll.load_data`` keeps the injected store.
    """
    stub_perf_db._moe_a2a_data = _build_a2a_store()
    return stub_perf_db


def test_a2a_coverage_full_dict(a2a_cov_db):
    assert a2a_cov_db.moe_a2a_coverage(*_SHAPE) == {
        "deepep_ht": {(16, 2)},
        "nvlink_two_sided": {(8, 2)},
    }


def test_a2a_dispatch_only_pair_excluded(a2a_cov_db):
    assert (32, 4) not in a2a_cov_db.moe_a2a_coverage(*_SHAPE)["deepep_ht"]


def test_a2a_empty_token_curve_pair_excluded(a2a_cov_db):
    assert (64, 8) not in a2a_cov_db.moe_a2a_coverage(*_SHAPE)["deepep_ht"]


def test_a2a_pair_covered_across_different_dtype_and_sms(a2a_cov_db):
    # dispatch is collected under ("default", sms=20), combine under
    # ("bfloat16", sms=0): the pair still counts (ANY dtype, ANY sms).
    assert (16, 2) in a2a_cov_db.moe_a2a_coverage(*_SHAPE)["deepep_ht"]


def test_a2a_prepare_neither_required_nor_sufficient(a2a_cov_db):
    coverage = a2a_cov_db.moe_a2a_coverage(*_SHAPE)
    assert (8, 2) in coverage["nvlink_two_sided"]  # covered without any prepare row
    assert (16, 4) not in coverage["nvlink_two_sided"]  # prepare-only pair


def test_a2a_backend_without_covered_pair_omitted(a2a_cov_db):
    assert "deepep_ll" not in a2a_cov_db.moe_a2a_coverage(*_SHAPE)


def test_a2a_unknown_shape_returns_empty_dict(a2a_cov_db):
    assert a2a_cov_db.moe_a2a_coverage(4096, 8, 128) == {}


def test_a2a_absent_table_returns_empty_dict(stub_perf_db):
    # The stub backend is unsupported: warm-up bound _moe_a2a_data = None.
    assert stub_perf_db._moe_a2a_data is None
    assert stub_perf_db.moe_a2a_coverage(*_SHAPE) == {}


def test_a2a_unloaded_wrapper_returns_empty_dict(stub_perf_db):
    # A LoadedOpData wrapping None raises PerfDataNotAvailableError on item
    # access (raise_if_not_loaded) but is falsy; the probe must yield {}
    # without raising.
    stub_perf_db._moe_a2a_data = LoadedOpData(None, PerfDataFilename.moe_a2a, "/nonexistent/moe_a2a_perf.parquet")
    assert stub_perf_db.moe_a2a_coverage(*_SHAPE) == {}


def test_a2a_probe_does_not_vivify_defaultdict_store(stub_perf_db):
    # The raw loader store auto-vivifies on indexing; the probe must walk with
    # .get() so a miss at any level leaves the key structure untouched.
    data = _vivifying_store()
    _store_leaf(data, ("deepep_ht", "dispatch", "default", 16, 2, *_SHAPE, 20, 32), _leaf(0.1))
    _store_leaf(data, ("deepep_ht", "combine", "default", 16, 2, *_SHAPE, 20, 32), _leaf(0.2))
    stub_perf_db._moe_a2a_data = data
    before = _key_paths(data)

    assert stub_perf_db.moe_a2a_coverage(*_SHAPE) == {"deepep_ht": {(16, 2)}}
    assert stub_perf_db.moe_a2a_coverage(4096, 8, 128) == {}  # absent hidden_size
    assert stub_perf_db.moe_a2a_coverage(7168, 99, 256) == {}  # absent topk under a present hidden_size
    assert stub_perf_db.moe_a2a_coverage(7168, 8, 999) == {}  # absent num_experts under a present topk

    assert _key_paths(data) == before


# ---------------------------------------------------------------------------
# moe_expert_compute_coverage on a synthetic store
# ---------------------------------------------------------------------------


def _build_ep_store():
    """11-part key order: (kernel_source, quant, distribution, inference_phase,
    topk, num_experts, num_slots, hidden_size, inter_size, moe_tp_size,
    moe_ep_size) -> {num_tokens: leaf}."""
    fp8_block = common.MoEQuantMode.fp8_block
    return _store(
        [
            # The same shape spread across distributions, num_slots, and
            # kernel sources: the probe unions all of them.
            (("deepep_moe", fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 16), {32: _leaf(0.1)}),
            (("deepep_moe", fp8_block, "power_law_1.2", "context", 8, 256, 512, 7168, 2048, 1, 32), {32: _leaf(0.2)}),
            (("deepgemm", fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 64), {32: _leaf(0.3)}),
            # moe_tp_size == 2 -> filtered out (the large-EP family is EP-only).
            (("deepep_moe", fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 2, 128), {32: _leaf(0.4)}),
            # Empty token curve -> not coverage.
            (("deepep_moe", fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 256), {}),
            # generation data exists only under a DIFFERENT quant mode.
            (
                ("deepep_moe", common.MoEQuantMode.bfloat16, "uniform", "generation", 8, 256, 256, 7168, 2048, 1, 8),
                {32: _leaf(0.5)},
            ),
        ]
    )


@pytest.fixture
def ep_cov_db(stub_perf_db):
    """A stub PerfDatabase with an injected unified moe_ep store (same
    ``__dict__``-gated injection as ``a2a_cov_db``, via ``MoEExpertCompute.load_data``)."""
    stub_perf_db._moe_ep_data = _build_ep_store()
    return stub_perf_db


def test_ep_coverage_unions_kernel_sources_distributions_and_slots(ep_cov_db):
    covered = ep_cov_db.moe_expert_compute_coverage(7168, 2048, 8, 256, common.MoEQuantMode.fp8_block, "context")
    assert covered == {16, 32, 64}


def test_ep_coverage_pins_moe_tp_to_one(ep_cov_db):
    # ep=128 is collected at moe_tp_size == 2 only.
    assert 128 not in ep_cov_db.moe_expert_compute_coverage(
        7168, 2048, 8, 256, common.MoEQuantMode.fp8_block, "context"
    )


def test_ep_coverage_requires_non_empty_token_curve(ep_cov_db):
    # ep=256 has an (empty) token dict.
    assert 256 not in ep_cov_db.moe_expert_compute_coverage(
        7168, 2048, 8, 256, common.MoEQuantMode.fp8_block, "context"
    )


def test_ep_coverage_missing_phase_returns_empty_set(ep_cov_db):
    # fp8_block is collected for "context" only.
    assert (
        ep_cov_db.moe_expert_compute_coverage(7168, 2048, 8, 256, common.MoEQuantMode.fp8_block, "generation") == set()
    )


def test_ep_coverage_filters_by_inference_phase(ep_cov_db):
    bf16 = common.MoEQuantMode.bfloat16
    assert ep_cov_db.moe_expert_compute_coverage(7168, 2048, 8, 256, bf16, "generation") == {8}
    assert ep_cov_db.moe_expert_compute_coverage(7168, 2048, 8, 256, bf16, "context") == set()


def test_ep_coverage_unknown_shape_or_quant_returns_empty_set(ep_cov_db):
    assert ep_cov_db.moe_expert_compute_coverage(4096, 2048, 8, 256, common.MoEQuantMode.fp8_block, "context") == set()
    assert ep_cov_db.moe_expert_compute_coverage(7168, 2048, 8, 256, common.MoEQuantMode.nvfp4, "context") == set()


def test_ep_absent_table_returns_empty_set(stub_perf_db):
    assert stub_perf_db._moe_ep_data is None
    assert (
        stub_perf_db.moe_expert_compute_coverage(7168, 2048, 8, 256, common.MoEQuantMode.fp8_block, "context") == set()
    )


def test_ep_unloaded_wrapper_returns_empty_set(stub_perf_db):
    stub_perf_db._moe_ep_data = LoadedOpData(
        None, PerfDataFilename.moe_expert_compute, "/nonexistent/moe_expert_compute_perf.parquet"
    )
    assert (
        stub_perf_db.moe_expert_compute_coverage(7168, 2048, 8, 256, common.MoEQuantMode.fp8_block, "context") == set()
    )


def test_ep_probe_does_not_vivify_defaultdict_store(stub_perf_db):
    data = _vivifying_store()
    key = ("deepep_moe", common.MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 16, 32)
    _store_leaf(data, key, _leaf(0.1))
    stub_perf_db._moe_ep_data = data
    before = _key_paths(data)

    fp8_block = common.MoEQuantMode.fp8_block
    assert stub_perf_db.moe_expert_compute_coverage(7168, 2048, 8, 256, fp8_block, "context") == {16}
    assert stub_perf_db.moe_expert_compute_coverage(7168, 2048, 8, 256, common.MoEQuantMode.nvfp4, "context") == set()
    assert (
        stub_perf_db.moe_expert_compute_coverage(7168, 2048, 8, 256, fp8_block, "generation") == set()
    )  # absent phase
    assert stub_perf_db.moe_expert_compute_coverage(4096, 2048, 8, 256, fp8_block, "context") == set()  # absent hidden
    assert stub_perf_db.moe_expert_compute_coverage(7168, 4096, 8, 256, fp8_block, "context") == set()  # absent inter

    assert _key_paths(data) == before


# ---------------------------------------------------------------------------
# Shipped-data smoke: real databases, legacy-adapted tables
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.exists(DEEPEP_NORMAL_PATH) and os.path.exists(DEEPEP_LL_PATH)),
    reason="shipped h200_sxm sglang 0.5.6.post2 DeepEP parquets not present",
)
def test_shipped_h200_sglang_a2a_coverage():
    db = get_database("h200_sxm", "sglang", "0.5.6.post2", allow_unlisted_version=True)
    assert db is not None

    coverage = db.moe_a2a_coverage(*_SHAPE)
    # The legacy DeepEP tables were collected on 8-GPU HGX fleets at node
    # counts {1, 2, 4, 8}, i.e. pairs {(8, 1), (16, 2), (32, 4), (64, 8)} for
    # both backends; assert a stable core point rather than the exact set.
    assert coverage["deepep_ht"] >= {(16, 2)}
    assert coverage["deepep_ll"] >= {(16, 2)}

    assert db.moe_a2a_coverage(4096, 8, 128) == {}


@pytest.mark.skipif(
    not os.path.exists(TRTLLM_ALLTOALL_PATH),
    reason="shipped gb200 trtllm 1.3.0rc10 alltoall parquet not present",
)
def test_shipped_gb200_trtllm_a2a_coverage():
    db = get_database("gb200", "trtllm", "1.3.0rc10", allow_unlisted_version=True)
    assert db is not None

    coverage = db.moe_a2a_coverage(*_SHAPE)
    # The legacy gb200 alltoall parquet carries dispatch+combine rows for BOTH
    # NVLink kernels at this shape (two-sided at ep {2..64}, one-sided at
    # ep {2..32}, node_num derived per NVL4), so both backends report coverage.
    assert coverage["nvlink_two_sided"]
    assert coverage["nvlink_one_sided"]

    assert db.moe_a2a_coverage(4096, 8, 128) == {}


@pytest.mark.skipif(
    not os.path.exists(SGLANG_CONTEXT_MOE_PATH),
    reason="shipped h200_sxm sglang 0.5.6.post2 wideep context moe parquet not present",
)
def test_shipped_h200_sglang_ep_compute_coverage():
    db = get_database("h200_sxm", "sglang", "0.5.6.post2", allow_unlisted_version=True)
    assert db is not None

    # deepep_moe fp8_block context data covers ep {2..256} at this shape.
    covered = db.moe_expert_compute_coverage(7168, 2048, 8, 256, common.MoEQuantMode.fp8_block, "context")
    assert covered


@pytest.mark.skipif(
    not os.path.exists(TRTLLM_WIDEEP_MOE_PATH),
    reason="shipped gb200 trtllm 1.3.0rc10 wideep moe parquet not present",
)
def test_shipped_gb200_trtllm_ep_compute_coverage():
    db = get_database("gb200", "trtllm", "1.3.0rc10", allow_unlisted_version=True)
    assert db is not None

    # The legacy trtllm wideep table has no phase split; the adapter registers
    # each row under BOTH phases, so the generation probe answers here.
    covered = db.moe_expert_compute_coverage(7168, 2048, 8, 256, common.MoEQuantMode.nvfp4, "generation")
    assert covered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
