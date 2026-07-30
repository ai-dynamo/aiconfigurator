# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified moe_a2a_perf loader (operations/moe_comm.py).

Key order under test: [comm_backend][phase][comm_dtype][ep_size][node_num]
[hidden_size][topk][num_experts][sms][num_tokens] -> {latency (ms), power (W),
energy (W*ms)}. The parquet ``latency`` column is in microseconds.
"""

import logging

import pandas as pd
import pytest

from aiconfigurator_core.sdk.operations.moe_comm import (
    _moe_a2a_store,
    _store_a2a_leaf,
    load_moe_a2a_data,
)

pytestmark = pytest.mark.unit

BASE_ROW = {
    "framework": "sglang",
    "version": "0.5.6",
    "device": "h200_sxm",
    "comm_backend": "deepep_ht",
    "phase": "dispatch",
    "comm_dtype": "fp8",
    "ep_size": 16,
    "node_num": 2,
    "hidden_size": 7168,
    "topk": 8,
    "num_experts": 256,
    "num_tokens": 128,
    "sms": 24,
    "transmit_us": 120.0,
    "notify_us": 30.0,
    "latency": 150.0,
    "power": 400.0,
}


def _row(**overrides):
    row = dict(BASE_ROW)
    row.update(overrides)
    return row


def _write_parquet(tmp_path, rows, name="moe_a2a_perf.parquet"):
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path, index=False)
    return str(path)


def test_new_schema_row_nested_structure_ms_conversion_and_energy(tmp_path):
    path = _write_parquet(tmp_path, [_row()])

    data = load_moe_a2a_data([(path, None)])

    assert set(data.keys()) == {"deepep_ht"}
    assert set(data["deepep_ht"].keys()) == {"dispatch"}
    leaf = data["deepep_ht"]["dispatch"]["fp8"][16][2][7168][8][256][24][128]
    assert leaf["latency"] == pytest.approx(0.150)  # 150 us -> ms
    assert leaf["power"] == 400.0
    assert leaf["energy"] == pytest.approx(400.0 * 0.150)  # W*ms
    assert set(leaf.keys()) == {"latency", "power", "energy"}


def test_prepare_phase_row_round_trips(tmp_path):
    row = _row(
        framework="trtllm",
        device="gb200_sxm",
        comm_backend="nvlink_two_sided",
        phase="prepare",
        comm_dtype="bfloat16",
        node_num=4,
        sms=0,
        latency=32.0,
        power=250.0,
    )
    path = _write_parquet(tmp_path, [row])

    data = load_moe_a2a_data([(path, None)])

    assert set(data["nvlink_two_sided"].keys()) == {"prepare"}
    leaf = data["nvlink_two_sided"]["prepare"]["bfloat16"][16][4][7168][8][256][0][128]
    assert leaf["latency"] == pytest.approx(0.032)
    assert leaf["energy"] == pytest.approx(250.0 * 0.032)


def test_ll_row_with_null_sms_lands_under_key_zero(tmp_path):
    # Mixed frame: the HT row keeps its SM budget, the LL row leaves sms null.
    rows = [
        _row(),
        _row(comm_backend="deepep_ll", sms=None, latency=50.0),
    ]
    path = _write_parquet(tmp_path, rows)

    data = load_moe_a2a_data([(path, None)])

    ll_by_sms = data["deepep_ll"]["dispatch"]["fp8"][16][2][7168][8][256]
    assert set(ll_by_sms.keys()) == {0}
    assert ll_by_sms[0][128]["latency"] == pytest.approx(0.050)
    ht_by_sms = data["deepep_ht"]["dispatch"]["fp8"][16][2][7168][8][256]
    assert set(ht_by_sms.keys()) == {24}


def test_absent_sms_and_power_columns_default_to_zero(tmp_path):
    row = _row()
    del row["sms"]
    del row["power"]
    path = _write_parquet(tmp_path, [row])

    data = load_moe_a2a_data([(path, None)])

    by_sms = data["deepep_ht"]["dispatch"]["fp8"][16][2][7168][8][256]
    assert set(by_sms.keys()) == {0}
    leaf = by_sms[0][128]
    assert leaf["latency"] == pytest.approx(0.150)
    assert leaf["power"] == 0.0
    assert leaf["energy"] == 0.0


def test_two_shapes_coexist(tmp_path):
    rows = [
        _row(),
        _row(hidden_size=4096, topk=4, num_experts=64, latency=90.0),
    ]
    path = _write_parquet(tmp_path, rows)

    data = load_moe_a2a_data([(path, None)])

    by_hidden = data["deepep_ht"]["dispatch"]["fp8"][16][2]
    assert set(by_hidden.keys()) == {7168, 4096}
    assert by_hidden[7168][8][256][24][128]["latency"] == pytest.approx(0.150)
    assert by_hidden[4096][4][64][24][128]["latency"] == pytest.approx(0.090)


def test_missing_file_returns_none(tmp_path):
    missing = str(tmp_path / "moe_a2a_perf.parquet")

    assert load_moe_a2a_data([(missing, None)]) is None


def test_legacy_source_kwargs_are_accepted(tmp_path):
    missing = str(tmp_path / "absent" / "moe_a2a_perf.parquet")
    legacy_kwargs = {
        "legacy_normal_sources": [(str(tmp_path / "wideep_deepep_normal_perf.parquet"), None)],
        "legacy_ll_sources": [(str(tmp_path / "wideep_deepep_ll_perf.parquet"), None)],
        "legacy_trtllm_alltoall_sources": [(str(tmp_path / "trtllm_alltoall_perf.parquet"), None)],
    }

    # Nothing loads anywhere -> None.
    assert load_moe_a2a_data([(missing, None)], **legacy_kwargs) is None

    # New-schema rows load identically with the legacy kwargs supplied.
    path = _write_parquet(tmp_path, [_row()])
    assert load_moe_a2a_data([(path, None)], **legacy_kwargs) == load_moe_a2a_data([(path, None)])


def test_empty_rows_file_returns_empty_store(tmp_path):
    path = _write_parquet(tmp_path, pd.DataFrame({column: [] for column in BASE_ROW}))

    data = load_moe_a2a_data([(path, None)])

    assert data is not None
    assert data == {}


def test_intra_source_collision_keeps_first_and_logs_debug(tmp_path, caplog):
    rows = [_row(latency=150.0), _row(latency=999.0)]
    path = _write_parquet(tmp_path, rows)

    with caplog.at_level(logging.DEBUG, logger="aiconfigurator_core.sdk.operations.moe_comm"):
        data = load_moe_a2a_data([(path, None)])

    leaf = data["deepep_ht"]["dispatch"]["fp8"][16][2][7168][8][256][24][128]
    assert leaf["latency"] == pytest.approx(0.150)
    assert any("value conflict in moe_a2a data" in message for message in caplog.messages)


def test_store_helper_overwrite_flag():
    # Task 3's legacy adapters store with overwrite=False; new-schema rows
    # store with overwrite=True and must replace a legacy leaf at the same key.
    data = _moe_a2a_store()
    key = ("deepep_ht", "dispatch", "fp8", 16, 2, 7168, 8, 256, 24, 128)
    legacy_leaf = {"latency": 1.0, "power": 0.0, "energy": 0.0}
    new_leaf = {"latency": 0.150, "power": 400.0, "energy": 60.0}

    _store_a2a_leaf(data, key, legacy_leaf, overwrite=False)
    assert data["deepep_ht"]["dispatch"]["fp8"][16][2][7168][8][256][24][128] == legacy_leaf

    _store_a2a_leaf(data, key, new_leaf, overwrite=True)
    assert data["deepep_ht"]["dispatch"]["fp8"][16][2][7168][8][256][24][128] == new_leaf

    # overwrite=False on an occupied key keeps the existing leaf.
    _store_a2a_leaf(data, key, legacy_leaf, overwrite=False)
    assert data["deepep_ht"]["dispatch"]["fp8"][16][2][7168][8][256][24][128] == new_leaf
