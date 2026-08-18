# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retired-loader fidelity of the engine table views on MALFORMED data.

The shipped parquet files are covered bit-for-bit by the frozen data-plane
baseline (``tests/cross_package/test_data_plane_baseline.py``); what that
replay cannot see is how the folds treat data shapes the retired Python
parsers tolerated but the collector never ships: null cells, DOUBLE-typed
integer columns (a single null upcasts a pandas column), NaN sentinels,
negative collector-bug keys, and legacy ``INCOMPLETE.txt`` vetoes. Each test
here pins one such contract to the retired parser's exact behavior
(see the per-fold comments in ``perf_database/table_view.rs``):

* null cells in PLAIN-STRING key columns keep the row keyed under ``""``
  (``_read_perf_rows`` mapped None -> ``""``); enum-decoded columns stay
  fail-loud on both sides.
* DOUBLE-typed integer key columns load their truncated values
  (``int(float(x))``) — never a silently empty table.
* negative integer keys stay negative keys (skip-on-malformed loaders kept
  them; they never abort the whole view).
* a NaN ``tp_size`` fails the MLA module load LOUDLY (Python's ``int(nan)``
  raised) instead of silently disarming the #1429 rank-local guard.
* ``window_size`` defaults to 0 only when the COLUMN is absent; a null cell
  fails loudly and a DOUBLE cell loads its true value.
* the moe_comm-family lenient power/latency readers are storage-agnostic:
  an INT64 watts/latency column is a value, not null.
* the legacy ``INCOMPLETE.txt`` whole-dir veto applies to the self-resolved
  views (megamoe family walk, NCCL/OneCCL comm roots) exactly like
  ``resolve_op_data_path`` / ``_build_op_sources`` applied it.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit


@pytest.fixture
def systems_root(tmp_path: Path) -> Path:
    """An ``h100_sxm`` systems tree with the full gpu/node spec shape the
    engine probe needs (mirrors test_reuse_ordering's loaded-table fixture)."""
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


def _build_db(systems_root: Path, *, backend: str = "trtllm", version: str = "1.0.0") -> PerfDatabase:
    return PerfDatabase(
        system="h100_sxm",
        backend=backend,
        version=version,
        systems_root=str(systems_root),
        database_mode="HYBRID",
    )


def _write_parquet(systems_root: Path, rel: str, columns: dict[str, list]) -> Path:
    path = systems_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def _fetch(db: PerfDatabase, attribute: str):
    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

    return fetch_table_view(db, attribute)


# ---------------------------------------------------------------------------
# Null string cells in plain-string key columns keep the row (keyed "")
# ---------------------------------------------------------------------------


def test_null_plain_string_key_cell_keeps_row_keyed_empty(systems_root: Path) -> None:
    """A null ``distribution``/``kernel_source`` cell must not abort the MoE
    view: the retired loader read it as ``""`` and kept the row."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe/trtllm/1.0.0/moe_perf.parquet",
        {
            "moe_dtype": ["fp8", "fp8"],
            "num_tokens": [16, 32],
            "hidden_size": [1024, 1024],
            "inter_size": [4096, 4096],
            "topk": [2, 2],
            "num_experts": [8, 8],
            "moe_tp_size": [1, 1],
            "moe_ep_size": [1, 1],
            "distribution": pa.array([None, "uniform"], type=pa.string()),
            "kernel_source": pa.array(["moe_torch_flow", None], type=pa.string()),
            "latency": [1.5, 2.5],
        },
    )
    loaded = _fetch(_build_db(systems_root), "_moe_data")
    quant = common.MoEQuantMode.fp8
    assert loaded[quant][""][2][8][1024][4096][1][1][16]["latency"] == pytest.approx(1.5)
    assert loaded[quant]["uniform"][2][8][1024][4096][1][1][32]["latency"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# DOUBLE-typed integer key columns (pandas upcast) load truncated values
# ---------------------------------------------------------------------------


def test_double_typed_dsv4_key_columns_load_instead_of_emptying_table(systems_root: Path) -> None:
    """A float64 ``batch_size``/``isl``/... column loaded fine in Python
    (``int(4.0) == 4``); an INT64-only read would silently return an EMPTY
    table with ``loaded=True`` and no error."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/dsv4/trtllm/1.0.0/dsv4_csa_context_module_perf.parquet",
        {
            "batch_size": pa.array([1.0], type=pa.float64()),
            "isl": pa.array([1024.0], type=pa.float64()),
            "step": pa.array([0.0], type=pa.float64()),
            "compress_ratio": pa.array([4.0], type=pa.float64()),
            "latency": [0.5],
            "num_heads": pa.array([16.0], type=pa.float64()),
            "tp_size": pa.array([8.0], type=pa.float64()),
            "gemm_type": ["fp8"],
            "mla_dtype": ["fp8"],
            "kv_cache_dtype": ["fp8"],
            "model": ["sgl-project/DeepSeek-V4-Flash-FP8"],
            "version": ["1.0.0"],
        },
    )
    loaded = _fetch(_build_db(systems_root), "_context_deepseek_v4_attention_module_data")
    fmha = common.FMHAQuantMode.fp8
    kv = common.KVCacheQuantMode.fp8
    gemm = common.GEMMQuantMode.fp8
    # native = 16 * 8 = 128, local = 16
    assert loaded[fmha][kv][gemm][128][16][4][0][1024][1]["latency"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Negative sparse-kernel keys stay negative keys; null rows skip themselves
# ---------------------------------------------------------------------------


def test_negative_sparse_key_files_under_minus_one_and_null_rows_skip(systems_root: Path) -> None:
    """A ``step=-1`` collector sentinel keyed under -1 in Python (int(-1));
    it must not abort the whole DSv4 sparse family. A null latency skips only
    its own row."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/dsv4/trtllm/1.0.0/dsv4_paged_mqa_logits_module_perf.parquet",
        {
            "num_heads": [16, 16, 16],
            "tp_size": [1, 1, 1],
            "step": [-1, 64, 128],
            "isl": [1024, 1024, 1024],
            "batch_size": [1, 1, 1],
            "latency": pa.array([0.1, 0.2, None], type=pa.float64()),
        },
    )
    loaded = _fetch(_build_db(systems_root), "_dsv4_sparse_kernel_data.paged_mqa_logits")
    assert loaded[16][1][-1][1024][1]["latency"] == pytest.approx(0.1)
    assert loaded[16][1][64][1024][1]["latency"] == pytest.approx(0.2)
    assert 128 not in loaded[16][1]  # null latency skipped that row only


# ---------------------------------------------------------------------------
# NaN tp_size fails the MLA module load loudly (#1429 guard stays armed)
# ---------------------------------------------------------------------------


def test_nan_tp_size_fails_mla_module_load_loudly(systems_root: Path) -> None:
    _write_parquet(
        systems_root,
        "data/h100_sxm/mla/trtllm/1.0.0/mla_context_module_perf.parquet",
        {
            "model": ["deepseek-ai/DeepSeek-V3"],
            "num_heads": [128],
            "batch_size": [1],
            "isl": [1024],
            "latency": [0.5],
            "mla_dtype": ["fp8"],
            "kv_cache_dtype": ["fp8"],
            "gemm_type": ["fp8"],
            "tp_size": pa.array([float("nan")], type=pa.float64()),
        },
    )
    with pytest.raises(Exception, match="tp_size"):
        _fetch(_build_db(systems_root), "_context_mla_module_data")


# ---------------------------------------------------------------------------
# window_size: column-absent -> 0; DOUBLE cell -> true value; null -> loud
# ---------------------------------------------------------------------------


def _attention_columns(window_size) -> dict[str, list]:
    cols = {
        "attn_dtype": ["fp8"],
        "kv_cache_dtype": ["fp8"],
        "batch_size": [1],
        "isl": [1024],
        "num_heads": [32],
        "num_key_value_heads": [8],
        "head_dim": [128],
        "latency": [0.7],
    }
    if window_size is not None:
        cols["window_size"] = window_size
    return cols


def test_double_window_size_loads_true_value(systems_root: Path) -> None:
    """float64 window_size=128.0 keys at 128 like Python's int(128.0) — a
    silent 0 default would conflate SWA rows with global-attention rows."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/attention/trtllm/1.0.0/context_attention_perf.parquet",
        _attention_columns(pa.array([128.0], type=pa.float64())),
    )
    loaded = _fetch(_build_db(systems_root), "_context_attention_data")
    fmha = common.FMHAQuantMode.fp8
    kv = common.KVCacheQuantMode.fp8
    assert loaded[fmha][kv][8][128][128][32][1024][1]["latency"] == pytest.approx(0.7)


def test_null_window_size_cell_fails_loudly(systems_root: Path) -> None:
    _write_parquet(
        systems_root,
        "data/h100_sxm/attention/trtllm/1.0.0/context_attention_perf.parquet",
        _attention_columns(pa.array([None], type=pa.int64())),
    )
    with pytest.raises(Exception, match="window_size"):
        _fetch(_build_db(systems_root), "_context_attention_data")


# ---------------------------------------------------------------------------
# moe_comm lenient readers are storage-agnostic (INT64 watts / latency)
# ---------------------------------------------------------------------------


def test_integer_power_and_latency_columns_load_for_moe_a2a(systems_root: Path) -> None:
    """Python's ``float(raw)`` never cared about parquet storage: an INT64
    power cell is watts (not a silent 0.0) and an INT64 latency cell is a
    value (not a misdiagnosed \"corrupt\" null)."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/moe_comm/trtllm/1.0.0/moe_a2a_perf.parquet",
        {
            "comm_backend": ["deepep_ht"],
            "phase": ["dispatch"],
            "comm_dtype": ["default"],
            "ep_size": [16],
            "node_num": [2],
            "hidden_size": [7168],
            "topk": [8],
            "num_experts": [256],
            "sms": [24],
            "num_tokens": [128],
            "latency": pa.array([1500], type=pa.int64()),  # us, integer-typed
            "power": pa.array([450], type=pa.int64()),  # watts, integer-typed
        },
    )
    loaded = _fetch(_build_db(systems_root), "_moe_a2a_data")
    leaf = loaded["deepep_ht"]["dispatch"]["default"][16][2][7168][8][256][24][128]
    assert leaf["latency"] == pytest.approx(1.5)  # us -> ms
    assert leaf["power"] == pytest.approx(450.0)
    assert leaf["energy"] == pytest.approx(675.0)


# ---------------------------------------------------------------------------
# Legacy INCOMPLETE.txt vetoes on the self-resolved views (round-2 gap:
# the round-1 veto test covers only map-resolved views)
# ---------------------------------------------------------------------------


_MEGAMOE_COLUMNS = {
    "used_cuda_graph": [True],
    "includes_gate_topk": [False],
    "includes_routed_scale": [True],
    "kernel_dtype": ["fp8"],
    "moe_dtype": ["fp8"],
    "pre_dispatch": ["fused"],
    "source_policy": ["primary"],
    "distribution": ["uniform"],
    "topk": [8],
    "num_experts": [256],
    "hidden_size": [7168],
    "inter_size": [2048],
    "moe_ep_size": [16],
    "num_tokens": [64],
    "latency": [3.5],
    "routed_scaling_factor": [2.5],
    "phase": ["generation"],
}


def test_incomplete_family_dir_vetoes_megamoe_view(systems_root: Path) -> None:
    """The retired loader resolved megamoe through resolve_op_data_path,
    whose family walk skips a legacy-INCOMPLETE dir — the view must answer
    None there, never serve the vetoed rows."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/dsv4/sglang/0.5.16/dsv4_megamoe_module_perf.parquet",
        _MEGAMOE_COLUMNS,
    )
    db = _build_db(systems_root, backend="sglang", version="0.5.16")
    assert _fetch(db, "_dsv4_megamoe_module_data") is not None  # positive control

    (systems_root / "data/h100_sxm/dsv4/sglang/0.5.16/INCOMPLETE.txt").write_bytes(b"partial collection\n")
    from aiconfigurator_core.sdk.operations.base import clear_all_op_caches

    clear_all_op_caches()  # drop the memoized probe spec so sources re-resolve
    assert _fetch(db, "_dsv4_megamoe_module_data") is None


def test_incomplete_comm_dir_vetoes_nccl_view(systems_root: Path) -> None:
    """_build_op_sources refused an existing comm primary under the legacy
    INCOMPLETE veto ("Not admitting primary source"), leaving _nccl_data
    None; the view must reproduce that, never serve the vetoed rows."""
    _write_parquet(
        systems_root,
        "data/h100_sxm/comm/nccl/2.26.2/nccl_perf.parquet",
        {
            "nccl_dtype": ["half"],
            "num_gpus": [8],
            "message_size": [1048576],
            "op_name": ["all_reduce"],
            "latency": [0.3],
        },
    )
    db = _build_db(systems_root)
    loaded = _fetch(db, "_nccl_data")
    assert loaded[common.CommQuantMode.half]["all_reduce"][8][1048576]["latency"] == pytest.approx(0.3)

    (systems_root / "data/h100_sxm/comm/nccl/2.26.2/INCOMPLETE.txt").write_bytes(b"partial collection\n")
    from aiconfigurator_core.sdk.operations.base import clear_all_op_caches

    clear_all_op_caches()
    assert _fetch(db, "_nccl_data") is None


# ---------------------------------------------------------------------------
# Probe-spec memo: governed by the documented eviction levers (round 2)
# ---------------------------------------------------------------------------


def _gemm_columns(latency: float) -> dict[str, list]:
    return {
        "gemm_dtype": ["bfloat16"],
        "m": [128],
        "n": [256],
        "k": [512],
        "latency": [latency],
    }


def test_clear_database_runtime_caches_reaches_the_view_after_a_disk_update(systems_root: Path) -> None:
    """The documented lever contract: after clear_database_runtime_caches, a
    reload reads fresh rows from disk — including through the table views
    (the per-database memo must not pin a stale Rust snapshot)."""
    from aiconfigurator_core.sdk.perf_database import clear_database_runtime_caches

    rel = "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet"
    _write_parquet(systems_root, rel, _gemm_columns(1.0))
    db = _build_db(systems_root)
    quant = common.GEMMQuantMode.bfloat16
    assert _fetch(db, "_gemm_data")[quant][128][256][512]["latency"] == pytest.approx(1.0)

    _write_parquet(systems_root, rel, _gemm_columns(9.0))
    clear_database_runtime_caches("h100_sxm", "trtllm", "1.0.0")
    assert _fetch(db, "_gemm_data")[quant][128][256][512]["latency"] == pytest.approx(9.0)


def test_warmed_database_stays_picklable_and_deepcopyable(systems_root: Path) -> None:
    """The memo holds plain strings, never a pyo3 handle: a database that has
    served table views must survive pickle and deepcopy (cross-process
    caching / ProcessPool return paths)."""
    _write_parquet(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet", _gemm_columns(1.0))
    db = _build_db(systems_root)
    assert _fetch(db, "_gemm_data") is not None
    assert "_table_view_probe_spec" in db.__dict__

    clone = copy.deepcopy(db)
    assert clone.system == db.system
    restored = pickle.loads(pickle.dumps(db))
    assert restored.system == db.system


# ---------------------------------------------------------------------------
# load_data half-population hardening (round 2): a failed later fetch must
# not turn every retry into a bare KeyError
# ---------------------------------------------------------------------------


def test_moe_load_data_retry_recovers_after_a_failed_later_fetch(systems_root: Path, monkeypatch) -> None:
    from aiconfigurator_core.sdk import engine_table_view
    from aiconfigurator_core.sdk.operations.moe import MoE

    _write_parquet(
        systems_root,
        "data/h100_sxm/moe/sglang/0.5.16/moe_perf.parquet",
        {
            "moe_dtype": ["fp8"],
            "num_tokens": [16],
            "hidden_size": [1024],
            "inter_size": [4096],
            "topk": [2],
            "num_experts": [8],
            "moe_tp_size": [1],
            "moe_ep_size": [1],
            "distribution": ["uniform"],
            "kernel_source": ["moe_torch_flow"],
            "latency": [1.5],
        },
    )
    db = _build_db(systems_root, backend="sglang", version="0.5.16")

    real_load_view = engine_table_view.load_view
    state = {"fail_on": "_wideep_context_moe_data"}

    def flaky_load_view(database, attribute, filename_enum):
        if attribute == state["fail_on"]:
            state["fail_on"] = None
            raise RuntimeError("transient parquet failure")
        return real_load_view(database, attribute, filename_enum)

    monkeypatch.setattr(engine_table_view, "load_view", flaky_load_view)
    MoE.clear_cache()
    try:
        with pytest.raises(RuntimeError, match="transient parquet failure"):
            MoE.load_data(db)
        # The failed load must not have committed a partial cache: the retry
        # re-fetches everything instead of KeyError-ing at the bind lines.
        MoE.load_data(db)
        assert db._moe_data is not None
        assert db._wideep_context_moe_data is not None
    finally:
        MoE.clear_cache()


# ---------------------------------------------------------------------------
# Composite weights recurse through per-child overrides (round 2)
# ---------------------------------------------------------------------------


def test_composite_weights_respect_child_weight_shields() -> None:
    """A tombstoned MoEDispatch (deepep flavor, weight shielded to 0.0)
    nested inside Overlap/Fallback must not crash memory estimation: the
    composites recurse through each child's own get_weights."""
    from aiconfigurator.sdk.operations.moe import MoEDispatch
    from aiconfigurator.sdk.operations.overlap import FallbackOp, OverlapOp

    dispatch = MoEDispatch(
        "d",
        1.0,
        hidden_size=7168,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=16,
        attention_dp_size=1,
        pre_dispatch=False,
        moe_backend="deepep_moe",
    )
    assert dispatch.get_weights() == 0.0
    assert OverlapOp("o", group_a=[dispatch], group_b=[]).get_weights() == 0.0
    assert FallbackOp("f", primary=dispatch, fallback=[]).get_weights() == 0.0
