# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy comm adapters into the unified moe_a2a store (L0 equivalence gate).

Covers the three ``_load_legacy_a2a`` adapters in operations/moe_comm.py:

- sglang ``wideep_deepep_normal_perf`` -> ``deepep_ht`` (per-phase rows from the
  four ``*_transmit_us``/``*_notify_us`` columns, ``sms=dispatch_sms``),
- sglang ``wideep_deepep_ll_perf`` -> ``deepep_ll`` (per-phase rows from
  ``dispatch_avg_t_us``/``combine_avg_t_us`` — the LL table never had the
  four-way split, matching ``load_wideep_deepep_ll_data``; ``sms=0``),
- trtllm ``trtllm_alltoall_perf`` -> ``nvlink_two_sided``/``nvlink_one_sided``.

Both synthetic mapping-rule tests and shipped-data equivalence sweeps against
the legacy loaders (``operations/moe.py``), which are the oracles here.
"""

import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from aiconfigurator_core.sdk.operations.base import resolve_op_data_path
from aiconfigurator_core.sdk.operations.moe import (
    load_trtllm_alltoall_data,
    load_wideep_deepep_ll_data,
    load_wideep_deepep_normal_data,
)
from aiconfigurator_core.sdk.operations.moe_comm import load_moe_a2a_data

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

NORMAL_ROW = {
    "node_num": 2,
    "hidden_size": 7168,
    "num_token": 64,
    "num_topk": 8,
    "num_experts": 256,
    "dispatch_sms": 20,
    "dispatch_transmit_us": 120.5,
    "dispatch_notify_us": 10.25,
    "combine_transmit_us": 200.5,
    "combine_notify_us": 20.25,
}

LL_ROW = {
    "node_num": 4,
    "hidden_size": 7168,
    "num_token": 32,
    "num_topk": 8,
    "num_experts": 288,
    "dispatch_avg_t_us": 15.5,
    "combine_avg_t_us": 12.25,
}

ALLTOALL_ROW = {
    "kernel_source": "NVLinkTwoSided",
    "op_name": "alltoall_dispatch",
    "moe_dtype": "fp8",
    "num_tokens": 128,
    "hidden_size": 7168,
    "topk": 8,
    "num_experts": 256,
    "moe_ep_size": 16,
    "latency": 0.25,
}


def _row(base, **overrides):
    row = dict(base)
    row.update(overrides)
    return row


def _write_parquet(tmp_path, rows, name):
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)


def _load_adapted(tmp_path, **legacy_kwargs):
    """Run the loader with no new-schema data so only the adapters contribute."""
    missing = str(tmp_path / "moe_a2a_perf.parquet")
    return load_moe_a2a_data([(missing, None)], **legacy_kwargs)


def _leaf(store, key):
    """Walk the 10-part key with .get so a defaultdict store cannot vivify."""
    node = store
    for part in key:
        node = node.get(part)
        assert node is not None, f"missing adapted leaf at {part!r} of {key}"
    return node


def _iter_leaves(nested, upper_levels):
    """Yield (key_tuple, leaf) walking ``upper_levels`` dict levels + the token dict."""
    if upper_levels == 0:
        yield from (((num_tokens,), leaf) for num_tokens, leaf in nested.items())
        return
    for key, sub in nested.items():
        for rest, leaf in _iter_leaves(sub, upper_levels - 1):
            yield (key, *rest), leaf


# ---------------------------------------------------------------------------
# DeepEP (sglang) synthetic mapping rules
# ---------------------------------------------------------------------------


def test_deepep_normal_maps_per_phase_component_sums(tmp_path):
    path = _write_parquet(tmp_path, [dict(NORMAL_ROW)], "wideep_deepep_normal_perf.parquet")

    data = _load_adapted(tmp_path, legacy_normal_sources=[(path, None)])

    assert set(data.keys()) == {"deepep_ht"}
    assert set(data["deepep_ht"].keys()) == {"dispatch", "combine"}
    # comm_dtype="default", ep_size = node_num * 8 (legacy 8-GPU HGX fleets),
    # sms = dispatch_sms for both phase rows.
    dispatch = _leaf(data, ("deepep_ht", "dispatch", "default", 16, 2, 7168, 8, 256, 20, 64))
    combine = _leaf(data, ("deepep_ht", "combine", "default", 16, 2, 7168, 8, 256, 20, 64))
    assert dispatch["latency"] == (120.5 + 10.25) / 1000.0  # us -> ms
    assert combine["latency"] == (200.5 + 20.25) / 1000.0
    assert dispatch["power"] == 0.0 and dispatch["energy"] == 0.0
    assert set(dispatch.keys()) == {"latency", "power", "energy"}


def test_deepep_normal_power_column_populates_energy(tmp_path):
    path = _write_parquet(tmp_path, [_row(NORMAL_ROW, power=400.0)], "wideep_deepep_normal_perf.parquet")

    data = _load_adapted(tmp_path, legacy_normal_sources=[(path, None)])

    dispatch = _leaf(data, ("deepep_ht", "dispatch", "default", 16, 2, 7168, 8, 256, 20, 64))
    assert dispatch["power"] == 400.0
    assert dispatch["energy"] == 400.0 * ((120.5 + 10.25) / 1000.0)  # W*ms


def test_deepep_ll_maps_avg_columns_under_sms_zero(tmp_path):
    path = _write_parquet(tmp_path, [dict(LL_ROW)], "wideep_deepep_ll_perf.parquet")

    data = _load_adapted(tmp_path, legacy_ll_sources=[(path, None)])

    assert set(data.keys()) == {"deepep_ll"}
    dispatch = _leaf(data, ("deepep_ll", "dispatch", "default", 32, 4, 7168, 8, 288, 0, 32))
    combine = _leaf(data, ("deepep_ll", "combine", "default", 32, 4, 7168, 8, 288, 0, 32))
    assert dispatch["latency"] == 15.5 / 1000.0  # dispatch_avg_t_us, us -> ms
    assert combine["latency"] == 12.25 / 1000.0  # combine_avg_t_us, us -> ms


# ---------------------------------------------------------------------------
# TRT-LLM alltoall synthetic mapping rules
# ---------------------------------------------------------------------------


def test_trtllm_kernel_source_mapping_and_unmapped_skip(tmp_path):
    rows = [
        dict(ALLTOALL_ROW),
        _row(ALLTOALL_ROW, kernel_source="NVLinkOneSided"),
        _row(ALLTOALL_ROW, kernel_source="DeepEP"),  # no unified backend -> dropped
    ]
    path = _write_parquet(tmp_path, rows, "trtllm_alltoall_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_alltoall_sources=[(path, None)])

    assert set(data.keys()) == {"nvlink_two_sided", "nvlink_one_sided"}


def test_trtllm_op_name_phase_and_dtype_routing(tmp_path):
    rows = [
        _row(ALLTOALL_ROW, op_name="alltoall_prepare", moe_dtype="fp8", latency=0.01),
        _row(ALLTOALL_ROW, op_name="alltoall_dispatch", moe_dtype="fp8", latency=0.02),
        _row(ALLTOALL_ROW, op_name="alltoall_combine", moe_dtype="fp8", latency=0.03),
        _row(ALLTOALL_ROW, op_name="alltoall_combine_low_precision", moe_dtype="nvfp4", latency=0.04),
        _row(ALLTOALL_ROW, op_name="alltoall_unknown", latency=9.9),  # dropped
    ]
    path = _write_parquet(tmp_path, rows, "trtllm_alltoall_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_alltoall_sources=[(path, None)])

    backend = data["nvlink_two_sided"]
    assert set(backend.keys()) == {"prepare", "dispatch", "combine"}
    # prepare, dispatch and standard combine pass the run's moe_dtype through
    # (lossless 1:1 legacy fidelity); only the low-precision combine kernel
    # gets the pinned "fp4" payload key.
    assert set(backend["prepare"].keys()) == {"fp8"}
    assert set(backend["dispatch"].keys()) == {"fp8"}
    assert set(backend["combine"].keys()) == {"fp8", "fp4"}
    assert _leaf(data, ("nvlink_two_sided", "prepare", "fp8", 16, 4, 7168, 8, 256, 0, 128))["latency"] == 0.01
    assert _leaf(data, ("nvlink_two_sided", "combine", "fp8", 16, 4, 7168, 8, 256, 0, 128))["latency"] == 0.03
    assert _leaf(data, ("nvlink_two_sided", "combine", "fp4", 16, 4, 7168, 8, 256, 0, 128))["latency"] == 0.04


def test_trtllm_latency_already_ms_stored_raw(tmp_path):
    # The legacy trtllm_alltoall latency column is already in milliseconds
    # (query_trtllm_alltoall returns table values without the /1000 the DeepEP
    # path applies), so the adapter must NOT divide by 1000.
    path = _write_parquet(tmp_path, [dict(ALLTOALL_ROW)], "trtllm_alltoall_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_alltoall_sources=[(path, None)])

    leaf = _leaf(data, ("nvlink_two_sided", "dispatch", "fp8", 16, 4, 7168, 8, 256, 0, 128))
    assert leaf["latency"] == 0.25
    assert leaf["power"] == 0.0 and leaf["energy"] == 0.0


def test_trtllm_node_num_derived_from_ep_size(tmp_path):
    # Legacy GB200 NVL4 data has no num_nodes column: node_num = max(1, ep // 4).
    rows = [_row(ALLTOALL_ROW, moe_ep_size=ep) for ep in (2, 4, 8, 64)]
    path = _write_parquet(tmp_path, rows, "trtllm_alltoall_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_alltoall_sources=[(path, None)])

    for ep, node_num in ((2, 1), (4, 1), (8, 2), (64, 16)):
        leaf = _leaf(data, ("nvlink_two_sided", "dispatch", "fp8", ep, node_num, 7168, 8, 256, 0, 128))
        assert leaf["latency"] == 0.25


def test_trtllm_num_nodes_column_respected_when_present(tmp_path):
    # Mirrors load_trtllm_alltoall_data: an explicit num_nodes column wins over
    # the ep//4 derivation.
    path = _write_parquet(tmp_path, [_row(ALLTOALL_ROW, num_nodes=5, moe_ep_size=8)], "trtllm_alltoall_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_alltoall_sources=[(path, None)])

    node_nums = set(data["nvlink_two_sided"]["dispatch"]["fp8"][8].keys())
    assert node_nums == {5}


def test_trtllm_kernel_source_column_absent_defaults_to_two_sided(tmp_path):
    row = dict(ALLTOALL_ROW)
    del row["kernel_source"]
    path = _write_parquet(tmp_path, [row], "trtllm_alltoall_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_alltoall_sources=[(path, None)])

    assert set(data.keys()) == {"nvlink_two_sided"}


def test_trtllm_multi_dtype_combine_rows_map_lossless(tmp_path):
    # Standard-combine rows measured under different run dtypes keep distinct
    # keys — no collapse, every legacy leaf survives 1:1 with its own latency.
    rows = [
        _row(ALLTOALL_ROW, op_name="alltoall_combine", moe_dtype="bfloat16", latency=0.011),
        _row(ALLTOALL_ROW, op_name="alltoall_combine", moe_dtype="fp8", latency=0.012),
    ]
    path = _write_parquet(tmp_path, rows, "trtllm_alltoall_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_alltoall_sources=[(path, None)])

    combine = data["nvlink_two_sided"]["combine"]
    assert set(combine.keys()) == {"bfloat16", "fp8"}
    assert _leaf(data, ("nvlink_two_sided", "combine", "bfloat16", 16, 4, 7168, 8, 256, 0, 128))["latency"] == 0.011
    assert _leaf(data, ("nvlink_two_sided", "combine", "fp8", 16, 4, 7168, 8, 256, 0, 128))["latency"] == 0.012


def test_trtllm_power_column_populates_energy(tmp_path):
    # The shipped gb200 file has no power column; a legacy file that does must
    # feed the energy = power * latency(ms) leaf field.
    path = _write_parquet(tmp_path, [_row(ALLTOALL_ROW, power=350.0)], "trtllm_alltoall_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_alltoall_sources=[(path, None)])

    leaf = _leaf(data, ("nvlink_two_sided", "dispatch", "fp8", 16, 4, 7168, 8, 256, 0, 128))
    assert leaf["power"] == 350.0
    assert leaf["energy"] == 350.0 * 0.25  # W * ms, latency stored raw (already ms)


# ---------------------------------------------------------------------------
# Precedence wiring: new-schema rows overwrite legacy-adapted leaves
# ---------------------------------------------------------------------------


def test_new_schema_row_overwrites_legacy_adapted_leaf(tmp_path):
    legacy_path = _write_parquet(tmp_path, [dict(NORMAL_ROW)], "wideep_deepep_normal_perf.parquet")
    new_row = {
        "comm_backend": "deepep_ht",
        "phase": "dispatch",
        "comm_dtype": "default",
        "ep_size": 16,
        "node_num": 2,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "sms": 20,
        "num_tokens": 64,
        "latency": 999.0,  # us
        "power": 100.0,
    }
    new_path = _write_parquet(tmp_path, [new_row], "moe_a2a_perf.parquet")

    data = load_moe_a2a_data([(new_path, None)], legacy_normal_sources=[(legacy_path, None)])

    dispatch = _leaf(data, ("deepep_ht", "dispatch", "default", 16, 2, 7168, 8, 256, 20, 64))
    assert dispatch["latency"] == 999.0 / 1000.0  # the new-schema row won
    assert dispatch["power"] == 100.0
    # The combine leaf has no new-schema counterpart -> legacy value survives.
    combine = _leaf(data, ("deepep_ht", "combine", "default", 16, 2, 7168, 8, 256, 20, 64))
    assert combine["latency"] == (200.5 + 20.25) / 1000.0


# ---------------------------------------------------------------------------
# Shipped-data equivalence sweeps (legacy loaders are the oracles)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.exists(DEEPEP_NORMAL_PATH) and os.path.exists(DEEPEP_LL_PATH)),
    reason="shipped h200_sxm sglang 0.5.6.post2 DeepEP parquets not present",
)
def test_shipped_deepep_equivalence_sweep(tmp_path):
    """Every legacy DeepEP leaf maps onto adapted per-phase leaves.

    Per-phase latencies are checked bit-exactly against the raw parquet
    columns (the adapter's own arithmetic, reproduced independently). The
    dispatch+combine sum is checked against the legacy loader's summed-us
    leaf at rel=1e-12: exact equality is impossible there because the legacy
    sum rounds once over four addends while the per-phase split rounds each
    phase separately (a couple of ulps apart — measured max 3.8e-16 relative
    on shipped data), and 1e-12 still fails loudly on any unit (x1000) or
    phase-mapping error.
    """
    legacy_normal = load_wideep_deepep_normal_data(DEEPEP_NORMAL_PATH)
    legacy_ll = load_wideep_deepep_ll_data(DEEPEP_LL_PATH)
    assert legacy_normal is not None and legacy_ll is not None

    adapted = load_moe_a2a_data(
        [(str(tmp_path / "moe_a2a_perf.parquet"), None)],
        legacy_normal_sources=[(DEEPEP_NORMAL_PATH, None)],
        legacy_ll_sources=[(DEEPEP_LL_PATH, None)],
    )
    assert adapted is not None

    # Independent per-phase expectation, straight from the parquet columns.
    expected = {}
    specs = (
        (
            DEEPEP_NORMAL_PATH,
            "deepep_ht",
            {
                "dispatch": ("dispatch_transmit_us", "dispatch_notify_us"),
                "combine": ("combine_transmit_us", "combine_notify_us"),
            },
        ),
        (DEEPEP_LL_PATH, "deepep_ll", {"dispatch": ("dispatch_avg_t_us",), "combine": ("combine_avg_t_us",)}),
    )
    for path, comm_backend, phase_columns in specs:
        for row in pq.read_table(path).to_pylist():
            node_num = int(row["node_num"])
            sms = int(row["dispatch_sms"]) if comm_backend == "deepep_ht" else 0
            for phase, columns in phase_columns.items():
                latency_us = 0.0
                for column in columns:
                    latency_us += float(row[column])
                key = (
                    comm_backend,
                    phase,
                    "default",
                    node_num * 8,
                    node_num,
                    int(row["hidden_size"]),
                    int(row["num_topk"]),
                    int(row["num_experts"]),
                    sms,
                    int(row["num_token"]),
                )
                expected.setdefault(key, latency_us / 1000.0)

    visited = 0
    sweeps = (
        (legacy_normal, "deepep_ht", 5, True),  # [node][hidden][topk][experts][sms]{tokens}
        (legacy_ll, "deepep_ll", 4, False),  # [node][hidden][topk][experts]{tokens}
    )
    for legacy, comm_backend, upper_levels, has_sms in sweeps:
        table_visited = 0
        for legacy_key, legacy_leaf in _iter_leaves(legacy, upper_levels):
            if has_sms:
                node_num, hidden_size, topk, num_experts, sms, num_tokens = legacy_key
            else:
                node_num, hidden_size, topk, num_experts, num_tokens = legacy_key
                sms = 0
            base = (comm_backend, "default", node_num * 8, node_num, hidden_size, topk, num_experts, sms, num_tokens)
            dispatch = _leaf(adapted, (base[0], "dispatch", *base[1:]))
            combine = _leaf(adapted, (base[0], "combine", *base[1:]))
            assert dispatch["latency"] == expected[(base[0], "dispatch", *base[1:])]
            assert combine["latency"] == expected[(base[0], "combine", *base[1:])]
            assert dispatch["latency"] + combine["latency"] == pytest.approx(legacy_leaf["latency"] / 1000.0, rel=1e-12)
            assert dispatch["power"] == legacy_leaf["power"]
            assert combine["power"] == legacy_leaf["power"]
            table_visited += 1
        assert table_visited > 0
        visited += table_visited

    assert visited > 100  # silent-empty sweep guard
    # No stray adapted leaves beyond the per-phase mapped rows (2 per legacy leaf).
    assert len(list(_iter_leaves(adapted, 9))) == len(expected) == 2 * visited


@pytest.mark.skipif(
    not os.path.exists(TRTLLM_ALLTOALL_PATH),
    reason="shipped gb200 trtllm 1.3.0rc10 alltoall parquet not present",
)
def test_shipped_trtllm_alltoall_equivalence_sweep(tmp_path):
    """Every legacy alltoall leaf equals the adapted leaf under the mapped keys.

    Latencies must match bit-exactly (the column is already ms; the adapter
    stores it raw). The mapping is lossless 1:1: prepare/dispatch/standard
    combine are keyed by the run's moe_dtype and low-precision combine by
    "fp4", so no two legacy rows may share a mapped key and every legacy leaf
    must survive with its own latency — zero duplicates, zero shadowing.
    """
    kernel_to_backend = {"NVLinkTwoSided": "nvlink_two_sided", "NVLinkOneSided": "nvlink_one_sided"}
    op_to_phase_dtype = {
        "alltoall_prepare": ("prepare", None),
        "alltoall_dispatch": ("dispatch", None),
        "alltoall_combine": ("combine", None),
        "alltoall_combine_low_precision": ("combine", "fp4"),
    }

    legacy = load_trtllm_alltoall_data(TRTLLM_ALLTOALL_PATH)
    assert legacy is not None

    adapted = load_moe_a2a_data(
        [(str(tmp_path / "moe_a2a_perf.parquet"), None)],
        legacy_trtllm_alltoall_sources=[(TRTLLM_ALLTOALL_PATH, None)],
    )
    assert adapted is not None

    def mapped_key(kernel_source, op_name, dtype_name, node_num, hidden_size, topk, num_experts, ep_size, num_tokens):
        phase, pinned_dtype = op_to_phase_dtype[op_name]
        comm_dtype = pinned_dtype if pinned_dtype is not None else dtype_name
        return (
            kernel_to_backend[kernel_source],
            phase,
            comm_dtype,
            ep_size,
            node_num,
            hidden_size,
            topk,
            num_experts,
            0,  # legacy alltoall rows carry no SM budget
            num_tokens,
        )

    # Expectation straight from the parquet; the remap must be collision-free.
    rows = pq.read_table(TRTLLM_ALLTOALL_PATH).to_pylist()
    expected = {}
    for row in rows:
        ep_size = int(row["moe_ep_size"])
        node_num = int(row["num_nodes"]) if "num_nodes" in row else max(1, ep_size // 4)
        key = mapped_key(
            row.get("kernel_source", "NVLinkTwoSided"),
            row["op_name"],
            row["moe_dtype"],
            node_num,
            int(row["hidden_size"]),
            int(row["topk"]),
            int(row["num_experts"]),
            ep_size,
            int(row["num_tokens"]),
        )
        assert key not in expected, f"duplicate mapped key {key}"
        expected[key] = float(row["latency"])

    visited = 0
    # legacy layout: [kernel][op][quant][node][hidden][topk][experts][ep]{tokens}
    for legacy_key, legacy_leaf in _iter_leaves(legacy, 8):
        kernel_source, op_name, quant, node_num, hidden_size, topk, num_experts, ep_size, num_tokens = legacy_key
        key = mapped_key(
            kernel_source, op_name, quant.name, node_num, hidden_size, topk, num_experts, ep_size, num_tokens
        )
        leaf = _leaf(adapted, key)
        assert leaf["latency"] == expected[key]
        assert leaf["latency"] == legacy_leaf["latency"]  # zero shadowing: 1:1
        assert leaf["power"] == legacy_leaf["power"]
        assert leaf["energy"] == legacy_leaf["energy"]
        visited += 1

    assert visited > 500  # silent-empty sweep guard
    # Lossless 1:1: adapted leaves == legacy leaves == parquet rows, no strays.
    assert len(list(_iter_leaves(adapted, 9))) == visited == len(expected)
    # The mapping-specific populations the adapter must preserve:
    prepare_store = adapted.get("nvlink_two_sided", {}).get("prepare", {})
    assert len(list(_iter_leaves(prepare_store, 7))) > 0
    fp4_combines = [leaf for key, leaf in _iter_leaves(adapted, 9) if key[1] == "combine" and key[2] == "fp4"]
    assert len(fp4_combines) > 0
