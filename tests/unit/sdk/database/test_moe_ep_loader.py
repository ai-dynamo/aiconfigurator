# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the unified moe_ep_perf loader (operations/moe_comm.py).

Key order under test: [kernel_source][quant(MoEQuantMode)][distribution]
[inference_phase][topk][num_experts][num_slots][hidden_size][inter_size]
[moe_tp_size][moe_ep_size][num_tokens] -> {latency (ms), power (W),
energy (W*ms)}. The ``latency`` column is already in milliseconds — new
schema (spec §4.2) and legacy compute tables alike.

Covers the ``_load_legacy_ep`` adapters:

- sglang ``wideep_context_moe_perf`` / ``wideep_generation_moe_perf`` ->
  ``kernel_source="deepep_moe"``, ``num_slots=num_experts``,
  ``inference_phase`` from which kwarg carried the source,
- trtllm ``wideep_moe_perf`` -> native ``kernel_source``/``num_slots``/
  ``_eplb`` distributions, each row registered under BOTH inference phases
  (the legacy table has no context/generation split).

Both synthetic mapping-rule tests and shipped-data equivalence sweeps against
the legacy loaders (``operations/moe.py``), which are the oracles here.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from aiconfigurator_core.sdk.common import MoEQuantMode
from aiconfigurator_core.sdk.operations.base import resolve_op_data_path
from aiconfigurator_core.sdk.operations.moe import (
    load_wideep_context_moe_data,
    load_wideep_generation_moe_data,
    load_wideep_moe_compute_data,
)
from aiconfigurator_core.sdk.operations.moe_comm import load_moe_ep_data

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SYSTEMS_DATA_ROOT = REPO_ROOT / "aic-core" / "src" / "aiconfigurator_core" / "systems" / "data"

SGLANG_CONTEXT_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_context_moe_perf.parquet"
)
SGLANG_GENERATION_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", "wideep_generation_moe_perf.parquet"
)
TRTLLM_WIDEEP_PATH = resolve_op_data_path(
    str(SYSTEMS_DATA_ROOT / "gb200"), "trtllm", "1.3.0rc10", "wideep_moe_perf.parquet"
)

NEW_ROW = {
    "framework": "SGLang",
    "version": "0.5.6.post2",
    "device": "NVIDIA H200",
    "kernel_source": "deepep_moe",
    "moe_dtype": "fp8_block",
    "distribution": "uniform",
    "inference_phase": "context",
    "topk": 8,
    "num_experts": 256,
    "num_slots": 288,
    "hidden_size": 7168,
    "inter_size": 2048,
    "moe_tp_size": 1,
    "moe_ep_size": 16,
    "num_tokens": 128,
    "latency": 0.25,  # already ms
    "power": 400.0,
}

# Column shape of the shipped sglang wideep files. op_name and the legacy
# kernel_source column ("deepepmoe") are present but ignored by the oracle
# loaders — the adapter must pin kernel_source="deepep_moe" regardless.
LEGACY_SGLANG_ROW = {
    "framework": "SGLang",
    "version": "0.5.6.post2",
    "device": "NVIDIA H200",
    "op_name": "moe_context",
    "kernel_source": "deepepmoe",
    "moe_dtype": "fp8_block",
    "num_tokens": 32,
    "hidden_size": 7168,
    "inter_size": 2048,
    "topk": 8,
    "num_experts": 256,
    "moe_tp_size": 1,
    "moe_ep_size": 2,
    "distribution": "uniform",
    "latency": 0.3651657,  # already ms
}

# Column shape of the shipped gb200 wideep_moe file (extra columns such as
# moe_kernel / simulation_mode are ignored by oracle and adapter alike).
LEGACY_TRTLLM_ROW = {
    "framework": "TRTLLM",
    "version": "1.3.0rc10",
    "device": "NVIDIA GB200",
    "op_name": "wideep_moe_eplb",
    "kernel_source": "wideep_compute_cutlass",
    "moe_dtype": "nvfp4",
    "moe_kernel": "cutlass",
    "num_tokens": 1,
    "hidden_size": 7168,
    "inter_size": 2048,
    "topk": 8,
    "num_experts": 256,
    "num_slots": 288,
    "moe_tp_size": 1,
    "moe_ep_size": 2,
    "distribution": "power_law_1.01_eplb",
    "latency": 0.0611904,  # already ms
}


# 12-part unified keys the fixture rows above map to.
NEW_KEY = ("deepep_moe", MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 288, 7168, 2048, 1, 16, 128)
LEGACY_SGLANG_KEY = ("deepep_moe", MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 2, 32)
TRT_KEY_BASE = ("wideep_compute_cutlass", MoEQuantMode.nvfp4, "power_law_1.01_eplb")
TRT_KEY_SHAPE = (8, 256, 288, 7168, 2048, 1, 2, 1)  # topk..num_tokens, phase spliced in between


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
    missing = str(tmp_path / "moe_ep_perf.parquet")
    return load_moe_ep_data([(missing, None)], **legacy_kwargs)


def _leaf(store, key):
    """Walk the 12-part key with .get so a defaultdict store cannot vivify."""
    node = store
    for part in key:
        node = node.get(part)
        assert node is not None, f"missing leaf at {part!r} of {key}"
    return node


def _iter_leaves(nested, upper_levels):
    """Yield (key_tuple, leaf) walking ``upper_levels`` dict levels + the token dict."""
    if upper_levels == 0:
        yield from (((num_tokens,), leaf) for num_tokens, leaf in nested.items())
        return
    for key, sub in nested.items():
        for rest, leaf in _iter_leaves(sub, upper_levels - 1):
            yield (key, *rest), leaf


def _sglang_mapped_key(row, inference_phase):
    """Unified key for one legacy sglang wideep row, derived from raw columns."""
    return (
        "deepep_moe",
        MoEQuantMode[row["moe_dtype"]],
        row["distribution"],
        inference_phase,
        int(row["topk"]),
        int(row["num_experts"]),
        int(row["num_experts"]),  # num_slots = num_experts (no EPLB axis in legacy sglang)
        int(row["hidden_size"]),
        int(row["inter_size"]),
        int(row["moe_tp_size"]),
        int(row["moe_ep_size"]),
        int(row["num_tokens"]),
    )


def _trtllm_mapped_key(row, inference_phase):
    """Unified key for one legacy trtllm wideep_moe row, derived from raw columns."""
    return (
        row.get("kernel_source", "moe_torch_flow"),
        MoEQuantMode[row["moe_dtype"]],
        row["distribution"],
        inference_phase,
        int(row["topk"]),
        int(row["num_experts"]),
        int(row["num_slots"]),
        int(row["hidden_size"]),
        int(row["inter_size"]),
        int(row["moe_tp_size"]),
        int(row["moe_ep_size"]),
        int(row["num_tokens"]),
    )


# ---------------------------------------------------------------------------
# New-schema rows
# ---------------------------------------------------------------------------


def test_new_schema_row_nested_structure_ms_latency_and_energy(tmp_path):
    path = _write_parquet(tmp_path, [dict(NEW_ROW)], "moe_ep_perf.parquet")

    data = load_moe_ep_data([(path, None)])

    assert set(data.keys()) == {"deepep_moe"}
    assert set(data["deepep_moe"].keys()) == {MoEQuantMode.fp8_block}  # enum key, not string
    leaf = _leaf(data, NEW_KEY)
    assert leaf["latency"] == 0.25  # column already ms — stored raw, no /1000
    assert leaf["power"] == 400.0
    assert leaf["energy"] == 400.0 * 0.25  # W*ms
    assert set(leaf.keys()) == {"latency", "power", "energy"}


def test_new_schema_phase_and_slots_are_distinct_axes(tmp_path):
    rows = [
        dict(NEW_ROW),
        _row(NEW_ROW, inference_phase="generation", latency=0.5),
        _row(NEW_ROW, num_slots=384, latency=0.75),
    ]
    path = _write_parquet(tmp_path, rows, "moe_ep_perf.parquet")

    data = load_moe_ep_data([(path, None)])

    by_phase = data["deepep_moe"][MoEQuantMode.fp8_block]["uniform"]
    assert set(by_phase.keys()) == {"context", "generation"}
    assert set(by_phase["context"][8][256].keys()) == {288, 384}
    base = ("deepep_moe", MoEQuantMode.fp8_block, "uniform")
    assert _leaf(data, (*base, "context", 8, 256, 288, 7168, 2048, 1, 16, 128))["latency"] == 0.25
    assert _leaf(data, (*base, "generation", 8, 256, 288, 7168, 2048, 1, 16, 128))["latency"] == 0.5
    assert _leaf(data, (*base, "context", 8, 256, 384, 7168, 2048, 1, 16, 128))["latency"] == 0.75


def test_new_schema_absent_power_column_defaults_to_zero(tmp_path):
    row = dict(NEW_ROW)
    del row["power"]
    path = _write_parquet(tmp_path, [row], "moe_ep_perf.parquet")

    data = load_moe_ep_data([(path, None)])

    leaf = _leaf(data, NEW_KEY)
    assert leaf["power"] == 0.0
    assert leaf["energy"] == 0.0


def test_new_schema_present_but_null_power_cells_load_as_no_power(tmp_path):
    # A power column that exists but is null on some rows (a merged file, or a
    # writer that only measured part of the sweep) must load cleanly: null and
    # NaN mean "not measured", exactly like an absent column. This used to
    # raise ValueError (parquet null -> "" -> float("")).
    rows = [
        _row(NEW_ROW, power=None),
        _row(NEW_ROW, num_tokens=256, power=400.0),
    ]
    path = _write_parquet(tmp_path, rows, "moe_ep_perf.parquet")

    data = load_moe_ep_data([(path, None)])

    null_leaf = _leaf(data, NEW_KEY)
    assert null_leaf["latency"] == 0.25  # the row still contributes latency
    assert null_leaf["power"] == 0.0
    assert null_leaf["energy"] == 0.0
    measured_leaf = _leaf(data, (*NEW_KEY[:-1], 256))
    assert measured_leaf["power"] == 400.0  # measured rows keep their power


def test_missing_file_returns_none(tmp_path):
    assert load_moe_ep_data([(str(tmp_path / "moe_ep_perf.parquet"), None)]) is None


def test_empty_rows_file_returns_empty_store(tmp_path):
    path = _write_parquet(tmp_path, pd.DataFrame({column: [] for column in NEW_ROW}), "moe_ep_perf.parquet")

    data = load_moe_ep_data([(path, None)])

    assert data is not None
    assert data == {}


def test_new_schema_intra_source_collision_keeps_first_and_logs_debug(tmp_path, caplog):
    rows = [_row(NEW_ROW, latency=0.25), _row(NEW_ROW, latency=9.9)]
    path = _write_parquet(tmp_path, rows, "moe_ep_perf.parquet")

    with caplog.at_level(logging.DEBUG, logger="aiconfigurator_core.sdk.operations.moe_comm"):
        data = load_moe_ep_data([(path, None)])

    leaf = _leaf(data, NEW_KEY)
    assert leaf["latency"] == 0.25
    assert any("value conflict in moe_ep data" in message for message in caplog.messages)


def test_legacy_source_kwargs_are_accepted(tmp_path):
    missing = str(tmp_path / "absent" / "moe_ep_perf.parquet")
    legacy_kwargs = {
        "legacy_context_sources": [(str(tmp_path / "wideep_context_moe_perf.parquet"), None)],
        "legacy_generation_sources": [(str(tmp_path / "wideep_generation_moe_perf.parquet"), None)],
        "legacy_trtllm_wideep_sources": [(str(tmp_path / "wideep_moe_perf.parquet"), None)],
    }

    # Nothing loads anywhere -> None.
    assert load_moe_ep_data([(missing, None)], **legacy_kwargs) is None

    # New-schema rows load identically with the (all-missing) legacy kwargs supplied.
    path = _write_parquet(tmp_path, [dict(NEW_ROW)], "moe_ep_perf.parquet")
    assert load_moe_ep_data([(path, None)], **legacy_kwargs) == load_moe_ep_data([(path, None)])


# ---------------------------------------------------------------------------
# sglang legacy adapter (context/generation two-file split)
# ---------------------------------------------------------------------------


def test_sglang_context_adapter_maps_row(tmp_path):
    path = _write_parquet(tmp_path, [dict(LEGACY_SGLANG_ROW)], "wideep_context_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_context_sources=[(path, None)])

    # kernel_source pinned to "deepep_moe" (the legacy column says "deepepmoe"
    # and the oracle loader never reads it); num_slots = num_experts.
    assert set(data.keys()) == {"deepep_moe"}
    by_phase = data["deepep_moe"][MoEQuantMode.fp8_block]["uniform"]
    assert set(by_phase.keys()) == {"context"}
    leaf = _leaf(data, LEGACY_SGLANG_KEY)
    assert leaf["latency"] == 0.3651657  # already ms — stored raw
    assert leaf["power"] == 0.0 and leaf["energy"] == 0.0
    assert set(leaf.keys()) == {"latency", "power", "energy"}


def test_sglang_generation_adapter_maps_row(tmp_path):
    row = _row(LEGACY_SGLANG_ROW, op_name="moe_generation", num_tokens=2, latency=0.1795488)
    path = _write_parquet(tmp_path, [row], "wideep_generation_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_generation_sources=[(path, None)])

    by_phase = data["deepep_moe"][MoEQuantMode.fp8_block]["uniform"]
    assert set(by_phase.keys()) == {"generation"}
    key = ("deepep_moe", MoEQuantMode.fp8_block, "uniform", "generation", 8, 256, 256, 7168, 2048, 1, 2, 2)
    assert _leaf(data, key)["latency"] == 0.1795488


def test_sglang_adapter_power_column_populates_energy(tmp_path):
    # The shipped sglang files have no power column; a legacy file that does
    # must feed the energy = power * latency(ms) leaf field.
    path = _write_parquet(tmp_path, [_row(LEGACY_SGLANG_ROW, power=420.0)], "wideep_context_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_context_sources=[(path, None)])

    leaf = _leaf(data, LEGACY_SGLANG_KEY)
    assert leaf["power"] == 420.0
    assert leaf["energy"] == 420.0 * 0.3651657


def test_sglang_adapter_duplicate_rows_first_wins_like_oracle(tmp_path):
    # load_wideep_context_moe_data guards with the skip-on-key-conflict
    # shared-layer contract (#1423), so on a duplicate legacy key the FIRST
    # row wins — the adapter must mirror that (same as the a2a adapters).
    rows = [_row(LEGACY_SGLANG_ROW, latency=0.1), _row(LEGACY_SGLANG_ROW, latency=0.2)]
    path = _write_parquet(tmp_path, rows, "wideep_context_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_context_sources=[(path, None)])

    assert _leaf(data, LEGACY_SGLANG_KEY)["latency"] == 0.1


def test_sglang_adapter_cross_source_conflict_first_source_wins(tmp_path):
    # The case that moved real numbers (case03, sglang 0.5.10): the primary
    # version file and a shared-layer/fallback file both carry a key. The
    # oracle's #1423 keep-first guard makes the primary (first-listed) source
    # win; the adapter must resolve the same way.
    primary = _write_parquet(tmp_path, [_row(LEGACY_SGLANG_ROW, latency=0.1)], "wideep_context_moe_perf.parquet")
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    fallback = _write_parquet(shared_dir, [_row(LEGACY_SGLANG_ROW, latency=0.9)], "wideep_context_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_context_sources=[(primary, None), (fallback, None)])

    assert _leaf(data, LEGACY_SGLANG_KEY)["latency"] == 0.1


# ---------------------------------------------------------------------------
# trtllm legacy adapter (single-table wideep_moe, no phase split)
# ---------------------------------------------------------------------------


def test_trtllm_adapter_registers_both_phases_with_identical_leaves(tmp_path):
    path = _write_parquet(tmp_path, [dict(LEGACY_TRTLLM_ROW)], "wideep_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_wideep_sources=[(path, None)])

    by_phase = data["wideep_compute_cutlass"][MoEQuantMode.nvfp4]["power_law_1.01_eplb"]
    assert set(by_phase.keys()) == {"context", "generation"}
    context = _leaf(data, (*TRT_KEY_BASE, "context", *TRT_KEY_SHAPE))
    generation = _leaf(data, (*TRT_KEY_BASE, "generation", *TRT_KEY_SHAPE))
    assert context == generation == {"latency": 0.0611904, "power": 0.0, "energy": 0.0}
    assert context is not generation  # independent leaf dicts, no shared mutable state


def test_trtllm_adapter_preserves_native_kernel_source_slots_and_eplb_distribution(tmp_path):
    rows = [
        dict(LEGACY_TRTLLM_ROW),
        _row(LEGACY_TRTLLM_ROW, num_slots=384, latency=0.5),
        _row(LEGACY_TRTLLM_ROW, distribution="power_law_1.01", latency=0.7),
        _row(LEGACY_TRTLLM_ROW, kernel_source="deepgemm", latency=0.9),
    ]
    path = _write_parquet(tmp_path, rows, "wideep_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_wideep_sources=[(path, None)])

    assert set(data.keys()) == {"wideep_compute_cutlass", "deepgemm"}
    cutlass = data["wideep_compute_cutlass"][MoEQuantMode.nvfp4]
    assert set(cutlass.keys()) == {"power_law_1.01_eplb", "power_law_1.01"}
    assert set(cutlass["power_law_1.01_eplb"]["context"][8][256].keys()) == {288, 384}
    key = ("deepgemm", MoEQuantMode.nvfp4, "power_law_1.01_eplb", "generation", 8, 256, 288, 7168, 2048, 1, 2, 1)
    assert _leaf(data, key)["latency"] == 0.9


def test_trtllm_adapter_kernel_source_absent_defaults_moe_torch_flow(tmp_path):
    # Mirrors load_wideep_moe_compute_data: no kernel_source column -> "moe_torch_flow".
    row = dict(LEGACY_TRTLLM_ROW)
    del row["kernel_source"]
    path = _write_parquet(tmp_path, [row], "wideep_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_wideep_sources=[(path, None)])

    assert set(data.keys()) == {"moe_torch_flow"}


def test_trtllm_adapter_duplicate_rows_first_wins_like_oracle(tmp_path):
    # load_wideep_moe_compute_data guards with the skip-on-key-conflict
    # shared-layer contract (#1423) -> first row wins.
    rows = [_row(LEGACY_TRTLLM_ROW, latency=0.1), _row(LEGACY_TRTLLM_ROW, latency=0.2)]
    path = _write_parquet(tmp_path, rows, "wideep_moe_perf.parquet")

    data = _load_adapted(tmp_path, legacy_trtllm_wideep_sources=[(path, None)])

    for phase in ("context", "generation"):
        assert _leaf(data, (*TRT_KEY_BASE, phase, *TRT_KEY_SHAPE))["latency"] == 0.1


# ---------------------------------------------------------------------------
# Precedence wiring: new-schema rows overwrite legacy-adapted leaves
# ---------------------------------------------------------------------------


def test_new_schema_row_overwrites_legacy_adapted_leaf(tmp_path):
    legacy_path = _write_parquet(tmp_path, [dict(LEGACY_SGLANG_ROW)], "wideep_context_moe_perf.parquet")
    new_row = _row(
        NEW_ROW,
        num_slots=256,
        moe_ep_size=2,
        num_tokens=32,
        latency=9.5,
        power=100.0,
    )
    new_path = _write_parquet(tmp_path, [new_row], "moe_ep_perf.parquet")

    data = load_moe_ep_data([(new_path, None)], legacy_context_sources=[(legacy_path, None)])

    leaf = _leaf(data, ("deepep_moe", MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 2, 32))
    assert leaf["latency"] == 9.5  # the new-schema row won
    assert leaf["power"] == 100.0


def test_new_schema_overwrites_only_named_phase_of_trtllm_legacy(tmp_path):
    # A legacy trtllm row lands under both phases; a new-schema context row
    # replaces the context leaf only — the generation twin keeps legacy data.
    legacy_path = _write_parquet(tmp_path, [dict(LEGACY_TRTLLM_ROW)], "wideep_moe_perf.parquet")
    new_row = _row(
        NEW_ROW,
        kernel_source="wideep_compute_cutlass",
        moe_dtype="nvfp4",
        distribution="power_law_1.01_eplb",
        inference_phase="context",
        num_slots=288,
        moe_ep_size=2,
        num_tokens=1,
        latency=7.0,
        power=50.0,
    )
    new_path = _write_parquet(tmp_path, [new_row], "moe_ep_perf.parquet")

    data = load_moe_ep_data([(new_path, None)], legacy_trtllm_wideep_sources=[(legacy_path, None)])

    assert _leaf(data, (*TRT_KEY_BASE, "context", *TRT_KEY_SHAPE)) == {"latency": 7.0, "power": 50.0, "energy": 350.0}
    assert _leaf(data, (*TRT_KEY_BASE, "generation", *TRT_KEY_SHAPE))["latency"] == 0.0611904


# ---------------------------------------------------------------------------
# Shipped-data equivalence sweeps (legacy loaders are the oracles)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.path.exists(SGLANG_CONTEXT_PATH) and os.path.exists(SGLANG_GENERATION_PATH)),
    reason="shipped h200_sxm sglang 0.5.6.post2 wideep MoE parquets not present",
)
def test_shipped_sglang_wideep_moe_equivalence_sweep(tmp_path):
    """Every legacy sglang wideep MoE leaf equals the adapted leaf, bit-exactly.

    The legacy ``latency`` column is already ms (same perf_interp query path
    as the regular moe table, no /1000 anywhere), so the adapter stores it
    raw and exact float equality must hold. The mapping is lossless 1:1
    (kernel_source pinned, num_slots = num_experts, phase from the file), so
    no two legacy rows may share a mapped key.
    """
    legacy_context = load_wideep_context_moe_data(SGLANG_CONTEXT_PATH)
    legacy_generation = load_wideep_generation_moe_data(SGLANG_GENERATION_PATH)
    assert legacy_context is not None and legacy_generation is not None

    adapted = _load_adapted(
        tmp_path,
        legacy_context_sources=[(SGLANG_CONTEXT_PATH, None)],
        legacy_generation_sources=[(SGLANG_GENERATION_PATH, None)],
    )
    assert adapted is not None

    # Expectation straight from the parquets; the remap must be collision-free.
    expected = {}
    for path, inference_phase in ((SGLANG_CONTEXT_PATH, "context"), (SGLANG_GENERATION_PATH, "generation")):
        for row in pq.read_table(path).to_pylist():
            key = _sglang_mapped_key(row, inference_phase)
            assert key not in expected, f"duplicate mapped key {key}"
            expected[key] = float(row["latency"])

    visited = 0
    sweeps = ((legacy_context, "context"), (legacy_generation, "generation"))
    for legacy, inference_phase in sweeps:
        table_visited = 0
        # legacy layout: [quant][distribution][topk][experts][hidden][inter][tp][ep]{tokens}
        for legacy_key, legacy_leaf in _iter_leaves(legacy, 8):
            quant, distribution, topk, num_experts, hidden_size, inter_size, moe_tp_size, moe_ep_size, num_tokens = (
                legacy_key
            )
            key = (
                "deepep_moe",
                quant,
                distribution,
                inference_phase,
                topk,
                num_experts,
                num_experts,  # num_slots = num_experts
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
                num_tokens,
            )
            leaf = _leaf(adapted, key)
            assert leaf["latency"] == expected[key]
            assert leaf["latency"] == legacy_leaf["latency"]  # bit-exact, zero shadowing
            assert leaf["power"] == legacy_leaf["power"]
            assert leaf["energy"] == legacy_leaf["energy"]
            table_visited += 1
        assert table_visited > 100  # silent-empty sweep guard, per file
        visited += table_visited

    # Lossless 1:1: adapted leaves == legacy leaves == parquet rows, no strays.
    assert len(list(_iter_leaves(adapted, 11))) == visited == len(expected)


@pytest.mark.skipif(
    not os.path.exists(TRTLLM_WIDEEP_PATH),
    reason="shipped gb200 trtllm 1.3.0rc10 wideep_moe parquet not present",
)
def test_shipped_trtllm_wideep_moe_equivalence_sweep(tmp_path):
    """Every legacy trtllm wideep_moe leaf is reproduced under BOTH phases.

    Latencies must match bit-exactly (the column is already ms; the adapter
    stores it raw). Native kernel_source, the num_slots axis ({256,288,384}
    on shipped data) and the ``_eplb`` distributions must survive unchanged;
    the adapted store holds exactly 2x the legacy leaves, nothing else.
    """
    legacy = load_wideep_moe_compute_data(TRTLLM_WIDEEP_PATH)
    assert legacy is not None

    adapted = _load_adapted(tmp_path, legacy_trtllm_wideep_sources=[(TRTLLM_WIDEEP_PATH, None)])
    assert adapted is not None

    # Expectation straight from the parquet; the remap must be collision-free.
    rows = pq.read_table(TRTLLM_WIDEEP_PATH).to_pylist()
    expected = {}
    for row in rows:
        for inference_phase in ("context", "generation"):
            key = _trtllm_mapped_key(row, inference_phase)
            assert key not in expected, f"duplicate mapped key {key}"
            expected[key] = float(row["latency"])

    visited = 0
    # legacy layout: [kernel][quant][distribution][topk][experts][hidden][inter][slots][tp][ep]{tokens}
    for legacy_key, legacy_leaf in _iter_leaves(legacy, 10):
        kernel_source, quant, distribution, topk, num_experts, hidden_size, inter_size, num_slots, tp, ep, tokens = (
            legacy_key
        )
        for inference_phase in ("context", "generation"):
            key = (
                kernel_source,
                quant,
                distribution,
                inference_phase,
                topk,
                num_experts,
                num_slots,
                hidden_size,
                inter_size,
                tp,
                ep,
                tokens,
            )
            leaf = _leaf(adapted, key)
            assert leaf["latency"] == expected[key]
            assert leaf["latency"] == legacy_leaf["latency"]  # bit-exact, zero shadowing
            assert leaf["power"] == legacy_leaf["power"]
            assert leaf["energy"] == legacy_leaf["energy"]
        visited += 1

    assert visited > 500  # silent-empty sweep guard
    # Both-phase registration: adapted leaves == 2x legacy leaves == expectation.
    adapted_leaves = list(_iter_leaves(adapted, 11))
    assert len(adapted_leaves) == 2 * visited == len(expected)

    # The mapping-specific populations the adapter must preserve:
    slots_seen = {key[6] for key, _ in adapted_leaves}
    assert slots_seen == {int(row["num_slots"]) for row in rows} == {256, 288, 384}
    distributions_seen = {key[2] for key, _ in adapted_leaves}
    assert distributions_seen == {row["distribution"] for row in rows}
    assert any(distribution.endswith("_eplb") for distribution in distributions_seen)
    kernels_seen = {key[0] for key, _ in adapted_leaves}
    assert kernels_seen == {row.get("kernel_source", "moe_torch_flow") for row in rows}
