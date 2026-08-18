# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-view behavior pins for the unified large-EP comm tables
(``_moe_a2a_data`` / ``_moe_ep_data``) and their legacy-file adapter legs.

Successor of the retired parser unit tests (``test_moe_a2a_loader.py``,
``test_moe_ep_loader.py``, ``test_moe_a2a_legacy_adapters.py``): the Python
parsers were deleted with the deprecation-cleanup PR, and the behaviors that
remain engine-visible are re-pinned here THROUGH the engine table view — a
synthetic parquet tree in, the loader-shaped nested dict out. Parser-internal
mechanics (store helpers, kwarg signatures, debug logs) retired with the
parsers; within-file precedence and the shipped-data equivalences live in the
Rust ``#[cfg(test)]`` modules (``moe_a2a.rs`` / ``moe_expert_compute.rs``
oracle tests) and the data-plane baseline replay.

Key orders under test (identical to the retired parsers):
- ``_moe_a2a_data``: [comm_backend][phase][comm_dtype][ep_size][node_num]
  [hidden_size][topk][num_experts][sms][num_tokens] -> {latency (ms), power,
  energy}. The new-schema ``latency`` column is MICROSECONDS (view divides by
  1000); the legacy trtllm_alltoall leg is already ms (stored raw).
- ``_moe_ep_data``: [kernel_source][quant][distribution][inference_phase]
  [topk][num_experts][num_slots][hidden_size][inter_size][moe_tp_size]
  [moe_ep_size][num_tokens] -> same leaf. ``latency`` already ms everywhere.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator.sdk.common import MoEQuantMode
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit


@pytest.fixture
def systems_root(tmp_path: Path) -> Path:
    """An ``h100_sxm`` systems tree with the full gpu/node spec shape the
    engine probe needs (mirrors test_table_view_data_shapes.py)."""
    root = tmp_path / "systems"
    root.mkdir()
    (root / "h100_sxm.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data/h100_sxm",
                "gpu": {
                    "sm_version": 90,
                    "mem_bw": 4_800_000_000_000.0,
                    "mem_bw_empirical_scaling_factor": 0.8,
                    "mem_empirical_constant_latency": 0.000003,
                    "bfloat16_tc_flops": 989_000_000_000_000.0,
                    "fp8_tc_flops": 1_978_000_000_000_000.0,
                },
                "node": {
                    "num_gpus_per_node": 8,
                    "inter_node_bw": 50_000_000_000.0,
                    "intra_node_bw": 450_000_000_000.0,
                    "p2p_latency": 0.00001,
                },
                "misc": {"nccl_version": "2.26.2"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _build_db(systems_root: Path, *, backend: str = "sglang", version: str = "1.0.0") -> PerfDatabase:
    return PerfDatabase(
        system="h100_sxm",
        backend=backend,
        version=version,
        systems_root=str(systems_root),
        database_mode="HYBRID",
    )


def _write_parquet(systems_root: Path, rel: str, rows: list[dict], *, types: dict | None = None) -> Path:
    """Write row dicts as parquet, preserving the first row's column order."""
    path = systems_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    arrays = {}
    for col in columns:
        values = [row[col] for row in rows]
        arrays[col] = pa.array(values, type=types[col]) if types and col in types else pa.array(values)
    pq.write_table(pa.table(arrays), path)
    return path


def _fetch(db: PerfDatabase, attribute: str):
    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

    return fetch_table_view(db, attribute)


def _leaf(store, key):
    node = store
    for part in key:
        node = node.get(part)
        assert node is not None, f"missing leaf at {part!r} of {key}"
    return node


# New-schema fixture rows (column shapes = the collector writers' headers).
A2A_ROW = {
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
    "latency": 150.0,  # us
    "power": 400.0,
}

EP_ROW = {
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

A2A_KEY = ("deepep_ht", "dispatch", "fp8", 16, 2, 7168, 8, 256, 24, 128)
EP_KEY = ("deepep_moe", MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 288, 7168, 2048, 1, 16, 128)

A2A_REL = "data/h100_sxm/moe_comm/sglang/1.0.0/moe_a2a_perf.parquet"
EP_REL = "data/h100_sxm/moe_comm/sglang/1.0.0/moe_expert_compute_perf.parquet"


def _row(base, **overrides):
    row = dict(base)
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# _moe_a2a_data: new-schema fold
# ---------------------------------------------------------------------------


def test_a2a_prepare_phase_row_round_trips(systems_root: Path) -> None:
    _write_parquet(systems_root, A2A_REL, [_row(A2A_ROW, phase="prepare", sms=0, node_num=4, latency=42.0)])
    loaded = _fetch(_build_db(systems_root), "_moe_a2a_data")
    leaf = _leaf(loaded, ("deepep_ht", "prepare", "fp8", 16, 4, 7168, 8, 256, 0, 128))
    assert leaf["latency"] == pytest.approx(0.042)  # us -> ms


def test_a2a_null_sms_lands_under_key_zero(systems_root: Path) -> None:
    _write_parquet(systems_root, A2A_REL, [_row(A2A_ROW, sms=None)], types={"sms": pa.int64()})
    loaded = _fetch(_build_db(systems_root), "_moe_a2a_data")
    assert _leaf(loaded, ("deepep_ht", "dispatch", "fp8", 16, 2, 7168, 8, 256, 0, 128))


def test_a2a_absent_sms_and_power_columns_default_to_zero(systems_root: Path) -> None:
    row = {k: v for k, v in A2A_ROW.items() if k not in ("sms", "power")}
    _write_parquet(systems_root, A2A_REL, [row])
    loaded = _fetch(_build_db(systems_root), "_moe_a2a_data")
    leaf = _leaf(loaded, ("deepep_ht", "dispatch", "fp8", 16, 2, 7168, 8, 256, 0, 128))
    assert leaf["power"] == 0.0
    assert leaf["energy"] == 0.0


def test_a2a_null_power_cells_load_as_no_power(systems_root: Path) -> None:
    _write_parquet(systems_root, A2A_REL, [_row(A2A_ROW, power=None)], types={"power": pa.float64()})
    loaded = _fetch(_build_db(systems_root), "_moe_a2a_data")
    leaf = _leaf(loaded, A2A_KEY)
    assert leaf["power"] == 0.0 and leaf["energy"] == 0.0


def test_a2a_non_finite_measured_power_refuses_load(systems_root: Path) -> None:
    _write_parquet(systems_root, A2A_REL, [_row(A2A_ROW, power=float("inf"))])
    with pytest.raises(Exception, match="power must be finite when measured"):
        _fetch(_build_db(systems_root), "_moe_a2a_data")


def test_a2a_null_latency_cell_refuses_load_with_named_error(systems_root: Path) -> None:
    _write_parquet(systems_root, A2A_REL, [_row(A2A_ROW, latency=None)], types={"latency": pa.float64()})
    with pytest.raises(Exception, match="latency is schema-required and must be finite"):
        _fetch(_build_db(systems_root), "_moe_a2a_data")


def test_a2a_non_finite_latency_cell_refuses_load(systems_root: Path) -> None:
    _write_parquet(systems_root, A2A_REL, [_row(A2A_ROW, latency=float("nan"))])
    with pytest.raises(Exception, match="latency is schema-required and must be finite"):
        _fetch(_build_db(systems_root), "_moe_a2a_data")


def test_a2a_missing_file_returns_none_and_empty_file_returns_empty(systems_root: Path) -> None:
    db = _build_db(systems_root)
    assert _fetch(db, "_moe_a2a_data") is None
    _write_parquet(systems_root, A2A_REL, [A2A_ROW])
    # a fresh db avoids the per-db view cache; the empty-vs-missing split is
    # observable per load
    empty_root_row = {k: [] for k in A2A_ROW}
    path = systems_root / A2A_REL
    pq.write_table(
        pa.table(
            {
                k: pa.array(v, type=pa.float64() if k in ("latency", "power") else None)
                for k, v in empty_root_row.items()
            }
        ),
        path,
    )
    assert _fetch(_build_db(systems_root), "_moe_a2a_data") == {}


# ---------------------------------------------------------------------------
# _moe_ep_data: new-schema fold
# ---------------------------------------------------------------------------


def test_ep_new_schema_key_structure_ms_latency_and_energy(systems_root: Path) -> None:
    _write_parquet(systems_root, EP_REL, [EP_ROW])
    loaded = _fetch(_build_db(systems_root), "_moe_ep_data")
    leaf = _leaf(loaded, EP_KEY)
    assert leaf["latency"] == pytest.approx(0.25)  # already ms — stored raw
    assert leaf["power"] == pytest.approx(400.0)
    assert leaf["energy"] == pytest.approx(100.0)
    assert set(leaf) == {"latency", "power", "energy"}


def test_ep_phase_and_slots_are_distinct_axes(systems_root: Path) -> None:
    _write_parquet(
        systems_root,
        EP_REL,
        [
            EP_ROW,
            _row(EP_ROW, inference_phase="generation", latency=0.5),
            _row(EP_ROW, num_slots=384, latency=0.75),
        ],
    )
    loaded = _fetch(_build_db(systems_root), "_moe_ep_data")
    assert _leaf(loaded, EP_KEY)["latency"] == pytest.approx(0.25)
    gen_key = list(EP_KEY)
    gen_key[3] = "generation"
    assert _leaf(loaded, tuple(gen_key))["latency"] == pytest.approx(0.5)
    slots_key = list(EP_KEY)
    slots_key[6] = 384
    assert _leaf(loaded, tuple(slots_key))["latency"] == pytest.approx(0.75)


def test_ep_absent_power_column_defaults_to_zero(systems_root: Path) -> None:
    row = {k: v for k, v in EP_ROW.items() if k != "power"}
    _write_parquet(systems_root, EP_REL, [row])
    leaf = _leaf(_fetch(_build_db(systems_root), "_moe_ep_data"), EP_KEY)
    assert leaf["power"] == 0.0 and leaf["energy"] == 0.0


def test_ep_null_power_cells_load_as_no_power(systems_root: Path) -> None:
    _write_parquet(systems_root, EP_REL, [_row(EP_ROW, power=None)], types={"power": pa.float64()})
    leaf = _leaf(_fetch(_build_db(systems_root), "_moe_ep_data"), EP_KEY)
    assert leaf["power"] == 0.0 and leaf["energy"] == 0.0


def test_ep_infinite_measured_power_refuses_load(systems_root: Path) -> None:
    _write_parquet(systems_root, EP_REL, [_row(EP_ROW, power=float("inf"))])
    with pytest.raises(Exception, match="power must be finite when measured"):
        _fetch(_build_db(systems_root), "_moe_ep_data")


def test_ep_nan_power_cell_is_lenient_no_power(systems_root: Path) -> None:
    """moe_comm's deliberate leniency: a NaN power cell reads as unmeasured
    (0.0), unlike the classic families' loud rule; only INF raises."""
    _write_parquet(systems_root, EP_REL, [_row(EP_ROW, power=float("nan"))])
    leaf = _leaf(_fetch(_build_db(systems_root), "_moe_ep_data"), EP_KEY)
    assert leaf["power"] == 0.0 and leaf["energy"] == 0.0


def test_ep_null_latency_cell_refuses_load_with_named_error(systems_root: Path) -> None:
    _write_parquet(systems_root, EP_REL, [_row(EP_ROW, latency=None)], types={"latency": pa.float64()})
    with pytest.raises(Exception, match="latency is schema-required and must be finite"):
        _fetch(_build_db(systems_root), "_moe_ep_data")


def test_ep_non_finite_latency_cell_refuses_load(systems_root: Path) -> None:
    _write_parquet(systems_root, EP_REL, [_row(EP_ROW, latency=float("inf"))])
    with pytest.raises(Exception, match="latency is schema-required and must be finite"):
        _fetch(_build_db(systems_root), "_moe_ep_data")


def test_ep_missing_file_returns_none(systems_root: Path) -> None:
    assert _fetch(_build_db(systems_root), "_moe_ep_data") is None


def test_ep_cross_source_conflict_first_source_wins(systems_root: Path) -> None:
    """Shared-layer precedence: the primary version's row wins the exact
    coordinate; the earlier sibling only fills shapes the primary lacks
    (design §6.1/§6.2 first-wins — this now exercises the ENGINE-side source
    resolution end to end)."""
    _write_parquet(systems_root, EP_REL, [EP_ROW])
    other = _row(EP_ROW, latency=9.0)
    other_shape = _row(EP_ROW, num_tokens=512, latency=3.5)
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/sglang/0.9.0/moe_expert_compute_perf.parquet",
        [other, other_shape],
    )
    loaded = _fetch(_build_db(systems_root), "_moe_ep_data")
    assert _leaf(loaded, EP_KEY)["latency"] == pytest.approx(0.25)  # primary wins
    fill_key = list(EP_KEY)
    fill_key[11] = 512
    assert _leaf(loaded, tuple(fill_key))["latency"] == pytest.approx(3.5)  # donor fills


# ---------------------------------------------------------------------------
# _moe_ep_data: legacy sglang / trtllm adapter legs
# ---------------------------------------------------------------------------

LEGACY_SGLANG_ROW = {
    "framework": "SGLang",
    "version": "0.5.6.post2",
    "device": "NVIDIA H200",
    "op_name": "moe_context",
    "kernel_source": "deepepmoe",  # ignored: adapter pins deepep_moe
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

LEGACY_SGLANG_KEY = ("deepep_moe", MoEQuantMode.fp8_block, "uniform", "context", 8, 256, 256, 7168, 2048, 1, 2, 32)

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


def test_ep_legacy_sglang_context_adapter_maps_row(systems_root: Path) -> None:
    """kernel_source pinned to deepep_moe, num_slots = num_experts, phase from
    the source file (context)."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/sglang/1.0.0/wideep_context_moe_perf.parquet",
        [LEGACY_SGLANG_ROW],
    )
    loaded = _fetch(_build_db(systems_root), "_moe_ep_data")
    assert _leaf(loaded, LEGACY_SGLANG_KEY)["latency"] == pytest.approx(0.3651657)


def test_ep_legacy_sglang_generation_adapter_maps_row_and_power(systems_root: Path) -> None:
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/sglang/1.0.0/wideep_generation_moe_perf.parquet",
        [_row(LEGACY_SGLANG_ROW, op_name="moe_generation", power=500.0)],
    )
    loaded = _fetch(_build_db(systems_root), "_moe_ep_data")
    key = list(LEGACY_SGLANG_KEY)
    key[3] = "generation"
    leaf = _leaf(loaded, tuple(key))
    assert leaf["power"] == pytest.approx(500.0)
    assert leaf["energy"] == pytest.approx(500.0 * 0.3651657)


def test_ep_legacy_trtllm_kernel_source_absent_defaults_moe_torch_flow(systems_root: Path) -> None:
    row = {k: v for k, v in LEGACY_TRTLLM_ROW.items() if k != "kernel_source"}
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/trtllm/1.0.0/wideep_moe_perf.parquet",
        [row],
    )
    loaded = _fetch(_build_db(systems_root, backend="trtllm"), "_moe_ep_data")
    key = ("moe_torch_flow", MoEQuantMode.nvfp4, "power_law_1.01_eplb", "context", 8, 256, 288, 7168, 2048, 1, 2, 1)
    assert _leaf(loaded, key)["latency"] == pytest.approx(0.0611904)


def test_ep_new_schema_overwrites_only_named_phase_of_trtllm_legacy(systems_root: Path) -> None:
    """The legacy trtllm row registers under BOTH phases; a new-schema row
    overwrites only its own phase, leaving the twin phase's adapted leaf."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/trtllm/1.0.0/wideep_moe_perf.parquet",
        [LEGACY_TRTLLM_ROW],
    )
    new_row = _row(
        EP_ROW,
        kernel_source="wideep_compute_cutlass",
        moe_dtype="nvfp4",
        distribution="power_law_1.01_eplb",
        inference_phase="context",
        num_slots=288,
        moe_ep_size=2,
        num_tokens=1,
        latency=0.9,
    )
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/trtllm/1.0.0/moe_expert_compute_perf.parquet",
        [new_row],
    )
    loaded = _fetch(_build_db(systems_root, backend="trtllm"), "_moe_ep_data")
    base = ("wideep_compute_cutlass", MoEQuantMode.nvfp4, "power_law_1.01_eplb")
    shape = (8, 256, 288, 7168, 2048, 1, 2, 1)
    assert _leaf(loaded, (*base, "context", *shape))["latency"] == pytest.approx(0.9)
    assert _leaf(loaded, (*base, "generation", *shape))["latency"] == pytest.approx(0.0611904)


# ---------------------------------------------------------------------------
# _moe_a2a_data: legacy adapter legs
# ---------------------------------------------------------------------------


def test_a2a_legacy_deepep_normal_power_column_populates_energy(systems_root: Path) -> None:
    row = {
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
        "power": 300.0,
    }
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/sglang/1.0.0/wideep_deepep_normal_perf.parquet",
        [row],
    )
    loaded = _fetch(_build_db(systems_root), "_moe_a2a_data")
    dispatch = _leaf(loaded, ("deepep_ht", "dispatch", "default", 16, 2, 7168, 8, 256, 20, 64))
    assert dispatch["latency"] == pytest.approx((120.5 + 10.25) / 1000.0)
    assert dispatch["power"] == pytest.approx(300.0)
    assert dispatch["energy"] == pytest.approx(300.0 * dispatch["latency"])


def test_a2a_legacy_trtllm_ms_raw_and_new_schema_us_share_one_leaf_space(systems_root: Path) -> None:
    """The unit divergence pin: legacy trtllm_alltoall rows are ALREADY ms
    (stored raw) while new-schema moe_a2a rows are us (divided by 1000) —
    both land in the same ms leaf space of one view."""
    legacy = {
        "kernel_source": "NVLinkTwoSided",
        "op_name": "alltoall_dispatch",
        "moe_dtype": "fp8",
        "num_tokens": 128,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "moe_ep_size": 16,
        "latency": 0.25,  # ms, stored raw
    }
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/trtllm/1.0.0/trtllm_alltoall_perf.parquet",
        [legacy],
    )
    new_row = _row(A2A_ROW, comm_backend="deepep_ht", latency=250.0)  # us
    _write_parquet(systems_root, "data/h100_sxm/moe_comm/trtllm/1.0.0/moe_a2a_perf.parquet", [new_row])
    loaded = _fetch(_build_db(systems_root, backend="trtllm"), "_moe_a2a_data")
    # legacy leg: node_num falls back to max(1, ep_size/4) = 4 when the
    # num_nodes column is absent (trtllm_alltoall.rs::legacy_num_nodes_fallback);
    # sms 0, comm_dtype from the run dtype
    legacy_leaf = _leaf(loaded, ("nvlink_two_sided", "dispatch", "fp8", 16, 4, 7168, 8, 256, 0, 128))
    assert legacy_leaf["latency"] == pytest.approx(0.25)
    new_leaf = _leaf(loaded, A2A_KEY)
    assert new_leaf["latency"] == pytest.approx(0.25)  # 250us -> 0.25ms


def test_a2a_legacy_trtllm_multi_dtype_combine_rows_map_lossless(systems_root: Path) -> None:
    """Standard combine keyed by the run dtype; low-precision combine keyed
    ``fp4`` — two run dtypes and the low-precision leg never collapse."""
    base = {
        "kernel_source": "NVLinkTwoSided",
        "op_name": "alltoall_combine",
        "moe_dtype": "fp8",
        "num_tokens": 128,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "moe_ep_size": 16,
        "latency": 0.1,
    }
    rows = [
        base,
        _row(base, moe_dtype="bf16", latency=0.2),
        _row(base, op_name="alltoall_combine_low_precision", latency=0.3),
        _row(base, op_name="alltoall_combine_low_precision", moe_dtype="bf16", latency=0.4),
    ]
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/trtllm/1.0.0/trtllm_alltoall_perf.parquet",
        rows,
    )
    loaded = _fetch(_build_db(systems_root, backend="trtllm"), "_moe_a2a_data")
    shape = (16, 4, 7168, 8, 256, 0, 128)
    assert _leaf(loaded, ("nvlink_two_sided", "combine", "fp8", *shape))["latency"] == pytest.approx(0.1)
    assert _leaf(loaded, ("nvlink_two_sided", "combine", "bf16", *shape))["latency"] == pytest.approx(0.2)
    fp4 = _leaf(loaded, ("nvlink_two_sided", "combine", "fp4", *shape))
    # last fp4-keyed row wins within one file (0.4 overwrote 0.3? No: the
    # retired adapter kept FIRST on duplicate keys) — both low-precision rows
    # key to fp4, first occurrence wins.
    assert fp4["latency"] == pytest.approx(0.3)


def test_a2a_legacy_trtllm_power_column_populates_energy(systems_root: Path) -> None:
    legacy = {
        "kernel_source": "NVLinkTwoSided",
        "op_name": "alltoall_dispatch",
        "moe_dtype": "fp8",
        "num_tokens": 128,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "moe_ep_size": 16,
        "latency": 0.25,
        "power": 200.0,
    }
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/trtllm/1.0.0/trtllm_alltoall_perf.parquet",
        [legacy],
    )
    loaded = _fetch(_build_db(systems_root, backend="trtllm"), "_moe_a2a_data")
    leaf = _leaf(loaded, ("nvlink_two_sided", "dispatch", "fp8", 16, 4, 7168, 8, 256, 0, 128))
    assert leaf["power"] == pytest.approx(200.0)
    assert leaf["energy"] == pytest.approx(200.0 * 0.25)
