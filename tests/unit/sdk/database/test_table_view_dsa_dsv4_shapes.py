# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-view behavior pins for the DSA and DSV4 module tables.

Successor of the retired parser unit tests (the loader halves of
``test_dsa_module.py`` and ``test_dsv4_sparse.py``): the Python parsers were
deleted with the deprecation-cleanup PR; the behaviors that remain
engine-visible are re-pinned here THROUGH the engine table view. Everything
is parquet (the engine never read ``.txt``); the one deliberate semantics
translation is the retired "whitespace step" case, whose parquet-native
equivalent is a NULL ``step`` cell (defaults to prefix 0).

Key orders (identical to the retired parsers):
- ``_context_dsa_module_data``: [fmha][kv][gemm][arch][bucket][heads]
  [prefix][isl][b]; generation drops fmha+prefix and indexes total decode
  length [kv][gemm][arch][bucket][heads][b][isl+step].
- ``_context_deepseek_v4_attention_module_data``: [fmha][kv][gemm]
  [native][local][cr][prefix][s][b]; generation [kv][gemm][native][local]
  [cr][b][s_total].
- ``_dsv4_sparse_kernel_data.<sub>``: [native_heads][tp][past_kv][isl][b].
- ``_dsv4_csa_topk_calib_data``: [native_heads][step][isl][b][score_mode]
  (score_mode stays a STRING key; the other four coerce to int).
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit

GLM5_ARCHITECTURE = "GlmMoeDsaForCausalLM"
DEFAULT_DSA_ARCHITECTURE = "DeepseekV32ForCausalLM"
_FLASH_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
_PRO_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
_FLASH_NATIVE_HEADS = 64
_PRO_NATIVE_HEADS = 128


@pytest.fixture
def systems_root(tmp_path: Path) -> Path:
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
    # Synthetic table-shape fixtures intentionally omit Collector V3 sidecars.
    return PerfDatabase(
        system="h100_sxm",
        backend=backend,
        version=version,
        systems_root=str(systems_root),
        database_mode="HYBRID",
        strict_provenance=False,
    )


def _write_parquet(systems_root: Path, rel: str, rows: list[dict], *, types: dict | None = None) -> Path:
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


# ---------------------------------------------------------------------------
# DSA context/generation module views
# ---------------------------------------------------------------------------

DSA_CTX_REL = "data/h100_sxm/sparse_attention/sglang/1.0.0/dsa_context_module_perf.parquet"
DSA_GEN_REL = "data/h100_sxm/sparse_attention/sglang/1.0.0/dsa_generation_module_perf.parquet"


def _dsa_row(**overrides) -> dict:
    row = {
        "architecture": DEFAULT_DSA_ARCHITECTURE,
        "kernel_source": "default",
        "gemm_type": "bfloat16",
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "num_heads": 32,
        "batch_size": 1,
        "isl": 256,
        "step": 0,
        "latency": 10.0,
        "power": 10.0,
    }
    row.update(overrides)
    return row


def test_glm5_context_view_requires_step_column(systems_root: Path) -> None:
    row = {k: v for k, v in _dsa_row(architecture=GLM5_ARCHITECTURE).items() if k != "step"}
    _write_parquet(systems_root, DSA_CTX_REL, [row])
    with pytest.raises(Exception, match="requires a non-empty step column"):
        _fetch(_build_db(systems_root), "_context_dsa_module_data")


def test_glm5_context_view_accepts_numeric_zero_step(systems_root: Path) -> None:
    _write_parquet(systems_root, DSA_CTX_REL, [_dsa_row(architecture=GLM5_ARCHITECTURE)])
    data = _fetch(_build_db(systems_root), "_context_dsa_module_data")
    value = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
        GLM5_ARCHITECTURE
    ]["flashmla_kv"][32][0][256][1]
    assert value["latency"] == pytest.approx(10.0)


def test_default_context_view_null_step_defaults_prefix_zero(systems_root: Path) -> None:
    """Parquet-native successor of the retired "whitespace step" case: a NULL
    ``step`` cell on the default (non-GLM5) architecture reads as prefix 0."""
    _write_parquet(systems_root, DSA_CTX_REL, [_dsa_row(step=None)], types={"step": pa.int64()})
    data = _fetch(_build_db(systems_root), "_context_dsa_module_data")
    value = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
        DEFAULT_DSA_ARCHITECTURE
    ]["flashmla_kv"][32][0][256][1]
    assert value["latency"] == pytest.approx(10.0)


def test_dsa_context_cross_source_first_wins(systems_root: Path) -> None:
    """Primary version wins the exact coordinate; the earlier sibling fills
    only missing shapes (within one file, LAST row wins — the retired
    pandas-iterrows overwrite, pinned in dsa.rs)."""
    _write_parquet(systems_root, DSA_CTX_REL, [_dsa_row(latency=7.0), _dsa_row(latency=10.0)])
    _write_parquet(
        systems_root,
        "data/h100_sxm/sparse_attention/sglang/0.9.0/dsa_context_module_perf.parquet",
        [_dsa_row(latency=99.0), _dsa_row(batch_size=2, isl=512, latency=20.0)],
    )
    data = _fetch(_build_db(systems_root), "_context_dsa_module_data")
    head_data = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
        DEFAULT_DSA_ARCHITECTURE
    ]["flashmla_kv"][32][0]
    assert head_data[256][1] == {"latency": 10.0, "power": 10.0, "energy": 100.0}
    assert head_data[512][2] == {"latency": 20.0, "power": 10.0, "energy": 200.0}


def test_dsa_generation_view_indexes_total_decode_length(systems_root: Path) -> None:
    _write_parquet(
        systems_root,
        DSA_GEN_REL,
        [_dsa_row(isl=1, step=149, latency=20.0)],
    )
    data = _fetch(_build_db(systems_root), "_generation_dsa_module_data")
    value = data[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][DEFAULT_DSA_ARCHITECTURE][
        "flashmla_kv"
    ][32][1][150]
    assert value == {"latency": 20.0, "power": 10.0, "energy": 200.0}


def test_dsa_generation_cross_source_first_wins_on_total_sequence(systems_root: Path) -> None:
    _write_parquet(
        systems_root,
        DSA_GEN_REL,
        [_dsa_row(isl=1, step=149, latency=7.0), _dsa_row(isl=2, step=148, latency=10.0)],
    )
    _write_parquet(
        systems_root,
        "data/h100_sxm/sparse_attention/sglang/0.9.0/dsa_generation_module_perf.parquet",
        # Different isl/step decomposition, same indexed total sequence 150;
        # plus a genuinely new total 151 that fills.
        [_dsa_row(isl=2, step=148, latency=99.0), _dsa_row(batch_size=2, isl=1, step=150, latency=20.0)],
    )
    data = _fetch(_build_db(systems_root), "_generation_dsa_module_data")
    head_data = data[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][DEFAULT_DSA_ARCHITECTURE][
        "flashmla_kv"
    ][32]
    assert head_data[1][150]["latency"] == pytest.approx(10.0)  # primary, last-row-in-file
    assert head_data[2][151]["latency"] == pytest.approx(20.0)  # donor fills


# ---------------------------------------------------------------------------
# DSV4 kind-module views ([native][local] keying, #1429)
# ---------------------------------------------------------------------------

DSV4_CTX_CSA_REL = "data/h100_sxm/sparse_attention/sglang/1.0.0/dsv4_csa_context_module_perf.parquet"
DSV4_GEN_HCA_REL = "data/h100_sxm/sparse_attention/sglang/1.0.0/dsv4_hca_generation_module_perf.parquet"


def _native_heads_for_model(model: str) -> int:
    return _PRO_NATIVE_HEADS if "Pro" in model else _FLASH_NATIVE_HEADS


def _dsv4_row(
    *,
    kind: str,
    phase: str,
    cr: int,
    bs: int,
    isl: int,
    tp: int,
    step: int = 0,
    gemm: str = "fp8_block",
    lat: float = 0.1,
    model: str = _FLASH_MODEL,
    num_heads: int | None = None,
    version: str = "test",
) -> dict:
    heads = max(1, _native_heads_for_model(model) // tp) if num_heads is None else num_heads
    return {
        "framework": "SGLang",
        "version": version,
        "device": "NVIDIA H20-3e",
        "op_name": f"dsv4_{kind}_{phase}_module",
        "kernel_source": "compressed_flashmla",
        "model": model,
        "architecture": "DeepseekV4ForCausalLM",
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8_e4m3",
        "gemm_type": gemm,
        "num_heads": heads,
        "batch_size": bs,
        "isl": isl,
        "tp_size": tp,
        "step": step,
        "compress_ratio": cr,
        "latency": lat,
    }


def test_dsv4_context_view_keys_by_native_and_local_head(systems_root: Path) -> None:
    rows = [
        _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=1, lat=18.0, model=_PRO_MODEL),
        _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=2, lat=14.0, model=_PRO_MODEL),
        _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=4, lat=11.5, model=_PRO_MODEL),
        _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=8, lat=10.5, model=_PRO_MODEL),
        _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=8, step=128, lat=12.5, model=_PRO_MODEL),
    ]
    _write_parquet(systems_root, DSV4_CTX_CSA_REL, rows)
    data = _fetch(_build_db(systems_root), "_context_deepseek_v4_attention_module_data")
    quant = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    assert set(quant.keys()) == {_PRO_NATIVE_HEADS}
    locals_ = quant[_PRO_NATIVE_HEADS]
    assert set(locals_.keys()) == {128, 64, 32, 16}
    assert locals_[16][4][0][8192][1]["latency"] == pytest.approx(10.5)
    assert locals_[16][4][128][8192][1]["latency"] == pytest.approx(12.5)
    assert locals_[128][4][0][8192][1]["latency"] > locals_[16][4][0][8192][1]["latency"]


def test_dsv4_generation_view_b_before_s_and_total_length(systems_root: Path) -> None:
    rows = [
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=1, isl=1, step=1023, tp=1, lat=0.1),
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=4, isl=1, step=1023, tp=1, lat=0.4),
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=4, isl=1, step=8191, tp=1, lat=1.0),
    ]
    _write_parquet(systems_root, DSV4_GEN_HCA_REL, rows)
    data = _fetch(_build_db(systems_root), "_generation_deepseek_v4_attention_module_data")
    sub = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block][_FLASH_NATIVE_HEADS][_FLASH_NATIVE_HEADS][
        128
    ]
    assert sub[1][1024]["latency"] == pytest.approx(0.1)
    assert sub[4][1024]["latency"] == pytest.approx(0.4)
    assert sub[4][8192]["latency"] == pytest.approx(1.0)


def test_dsv4_views_keep_native_head_buckets_separate(systems_root: Path) -> None:
    _write_parquet(
        systems_root,
        DSV4_CTX_CSA_REL,
        [
            _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=1, lat=18.0, model=_FLASH_MODEL),
            _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=1, lat=23.0, model=_PRO_MODEL),
        ],
    )
    data = _fetch(_build_db(systems_root), "_context_deepseek_v4_attention_module_data")
    data = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    assert data[_FLASH_NATIVE_HEADS][_FLASH_NATIVE_HEADS][4][0][8192][1]["latency"] == pytest.approx(18.0)
    assert data[_PRO_NATIVE_HEADS][_PRO_NATIVE_HEADS][4][0][8192][1]["latency"] == pytest.approx(23.0)


def test_dsv4_view_rejects_stale_native_semantics(systems_root: Path) -> None:
    """Files still storing the pre-#1131 NATIVE convention (num_heads constant
    across a tp sweep) must fail loudly instead of collapsing tp shards onto
    wrong (native, local) coordinates."""
    rows = [
        _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=1, lat=18.0, model=_PRO_MODEL, num_heads=128),
        _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=8, lat=10.5, model=_PRO_MODEL, num_heads=128),
    ]
    _write_parquet(systems_root, DSV4_CTX_CSA_REL, rows)
    with pytest.raises(Exception, match="pre-#1131 NATIVE semantics"):
        _fetch(_build_db(systems_root), "_context_deepseek_v4_attention_module_data")


def test_dsv4_view_stale_guard_is_per_model(systems_root: Path) -> None:
    stale_plus_good = [
        _dsv4_row(
            kind="hca",
            phase="generation",
            cr=128,
            bs=1,
            isl=1,
            step=1023,
            tp=2,
            lat=0.5,
            model=_PRO_MODEL,
            num_heads=128,
        ),
        _dsv4_row(
            kind="hca",
            phase="generation",
            cr=128,
            bs=1,
            isl=1,
            step=1023,
            tp=8,
            lat=0.3,
            model=_PRO_MODEL,
            num_heads=128,
        ),
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.4, model=_FLASH_MODEL),
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.2, model=_FLASH_MODEL),
    ]
    _write_parquet(systems_root, DSV4_GEN_HCA_REL, stale_plus_good)
    with pytest.raises(Exception, match="DeepSeek-V4-Pro"):
        _fetch(_build_db(systems_root), "_generation_deepseek_v4_attention_module_data")

    good = [
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.5, model=_PRO_MODEL),
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.3, model=_PRO_MODEL),
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.4, model=_FLASH_MODEL),
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=1, isl=1, step=1023, tp=8, lat=0.2, model=_FLASH_MODEL),
    ]
    _write_parquet(systems_root, DSV4_GEN_HCA_REL, good)
    data = _fetch(_build_db(systems_root, version="1.0.0"), "_generation_deepseek_v4_attention_module_data")
    q = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    assert q[_PRO_NATIVE_HEADS][64][128][1][1024]["latency"] == pytest.approx(0.5)
    assert q[_PRO_NATIVE_HEADS][16][128][1][1024]["latency"] == pytest.approx(0.3)
    assert q[_FLASH_NATIVE_HEADS][32][128][1][1024]["latency"] == pytest.approx(0.4)
    assert q[_FLASH_NATIVE_HEADS][8][128][1][1024]["latency"] == pytest.approx(0.2)


def test_dsv4_view_stale_guard_is_per_version(systems_root: Path) -> None:
    """The stale-NATIVE fingerprint is checked per (model, version): a stale
    native-writing sibling of the SAME model must be caught even when the
    pooled union of versions no longer looks heads-constant."""
    rows = [
        _dsv4_row(
            kind="hca",
            phase="generation",
            cr=128,
            bs=1,
            isl=1,
            step=1023,
            tp=2,
            lat=0.5,
            num_heads=64,
            version="0.5.10",
        ),
        _dsv4_row(
            kind="hca",
            phase="generation",
            cr=128,
            bs=1,
            isl=1,
            step=1023,
            tp=8,
            lat=0.3,
            num_heads=64,
            version="0.5.10",
        ),
        _dsv4_row(
            kind="hca",
            phase="generation",
            cr=128,
            bs=2,
            isl=1,
            step=1023,
            tp=2,
            lat=0.4,
            num_heads=32,
            version="0.5.16",
        ),
        _dsv4_row(
            kind="hca", phase="generation", cr=128, bs=2, isl=1, step=1023, tp=8, lat=0.2, num_heads=8, version="0.5.16"
        ),
    ]
    _write_parquet(systems_root, DSV4_GEN_HCA_REL, rows)
    with pytest.raises(Exception, match=r'version="0\.5\.10"'):
        _fetch(_build_db(systems_root), "_generation_deepseek_v4_attention_module_data")

    good = [
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=1, isl=1, step=1023, tp=2, lat=0.5, version="0.5.10"),
        _dsv4_row(kind="hca", phase="generation", cr=128, bs=2, isl=1, step=1023, tp=8, lat=0.2, version="0.5.16"),
    ]
    _write_parquet(systems_root, DSV4_GEN_HCA_REL, good)
    data = _fetch(_build_db(systems_root, version="1.0.0"), "_generation_deepseek_v4_attention_module_data")
    q = data[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.fp8_block]
    assert q[_FLASH_NATIVE_HEADS][32][128][1][1024]["latency"] == pytest.approx(0.5)
    assert q[_FLASH_NATIVE_HEADS][8][128][2][1024]["latency"] == pytest.approx(0.2)


def test_dsv4_view_requires_tp_size_column(systems_root: Path) -> None:
    row = {
        k: v for k, v in _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=1).items() if k != "tp_size"
    }
    _write_parquet(systems_root, DSV4_CTX_CSA_REL, [row])
    with pytest.raises(Exception, match="tp_size"):
        _fetch(_build_db(systems_root), "_context_deepseek_v4_attention_module_data")


def test_dsv4_view_rejects_null_tp_size_among_populated_rows(systems_root: Path) -> None:
    """A null ``tp_size`` cell mixed into populated rows must fail loudly —
    silently defaulting would poison the [native][local] fingerprint (the
    exact #1460-review failure the retired parser guarded)."""
    rows = [
        _dsv4_row(kind="csa", phase="context", cr=4, bs=1, isl=8192, tp=1, lat=1.0),
        dict(_dsv4_row(kind="csa", phase="context", cr=4, bs=2, isl=8192, tp=1, lat=2.0), tp_size=None),
    ]
    _write_parquet(systems_root, DSV4_CTX_CSA_REL, rows, types={"tp_size": pa.int64()})
    with pytest.raises(Exception, match="tp_size"):
        _fetch(_build_db(systems_root), "_context_deepseek_v4_attention_module_data")


# ---------------------------------------------------------------------------
# DSV4 sparse-kernel sub-views
# ---------------------------------------------------------------------------


def test_dsv4_sparse_kernel_view_key_order(systems_root: Path) -> None:
    def sparse_row(*, bs, isl, past_kv, tp, lat):
        return {
            "framework": "SGLang",
            "version": "test",
            "device": "NVIDIA H20-3e",
            "op_name": "dsv4_paged_mqa_logits_module",
            "kernel_source": "paged_mqa_logits",
            "model": _FLASH_MODEL,
            "architecture": "DeepseekV4ForCausalLM",
            "mla_dtype": "fp8_e4m3",
            "kv_cache_dtype": "fp8_e4m3",
            "gemm_type": "fp8_block",
            "num_heads": _FLASH_NATIVE_HEADS,
            "batch_size": bs,
            "isl": isl,
            "tp_size": tp,
            "step": past_kv,
            "compress_ratio": 4,
            "latency": lat,
        }

    _write_parquet(
        systems_root,
        "data/h100_sxm/sparse_attention/sglang/1.0.0/dsv4_paged_mqa_logits_module_perf.parquet",
        [
            sparse_row(bs=1, isl=1024, past_kv=0, tp=1, lat=0.10),
            sparse_row(bs=1, isl=1024, past_kv=8192, tp=1, lat=0.30),
            sparse_row(bs=1, isl=8192, past_kv=0, tp=1, lat=0.55),
        ],
    )
    data = _fetch(_build_db(systems_root), "_dsv4_sparse_kernel_data.paged_mqa_logits")
    assert data[_FLASH_NATIVE_HEADS][1][0][1024][1]["latency"] == pytest.approx(0.10)
    assert data[_FLASH_NATIVE_HEADS][1][8192][1024][1]["latency"] == pytest.approx(0.30)
    assert data[_FLASH_NATIVE_HEADS][1][0][8192][1]["latency"] == pytest.approx(0.55)


def test_dsv4_sparse_kernel_view_missing_returns_none(systems_root: Path) -> None:
    # paged_mqa_logits is the sole surviving sparse sidecar view; with no
    # parquet written it must report absent (None), not raise.
    assert _fetch(_build_db(systems_root), "_dsv4_sparse_kernel_data.paged_mqa_logits") is None


def test_dsv4_retired_sparse_kernel_views_are_unknown_attributes(systems_root: Path) -> None:
    # hca_attn/csa_attn sidecars were retired (loaded-but-never-queried);
    # their attributes are gone from the registry and fail loudly.
    from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError

    with pytest.raises(PerfDataNotAvailableError, match="unknown table-view attribute"):
        _fetch(_build_db(systems_root), "_dsv4_sparse_kernel_data.csa_attn")


# ---------------------------------------------------------------------------
# DSV4 CSA topk-calib view (the last retired parser: load_dsv4_sparse_op_data
# under _TOPK_CALIB_KEYS)
# ---------------------------------------------------------------------------

CALIB_REL = "data/h100_sxm/sparse_attention/sglang/1.0.0/dsv4_csa_topk_calib_perf.parquet"


def _calib_row(**overrides) -> dict:
    row = {
        "num_heads": 64,
        "step": 0,
        "isl": 8192,
        "batch_size": 1,
        "score_mode": "v1_top_last",
        "latency": 0.05,
    }
    row.update(overrides)
    return row


def test_calib_view_nests_ints_then_string_score_mode(systems_root: Path) -> None:
    _write_parquet(
        systems_root,
        CALIB_REL,
        [
            _calib_row(),
            _calib_row(score_mode="v1_flat", latency=0.0),
            _calib_row(isl=1024, latency=0.01),
        ],
    )
    data = _fetch(_build_db(systems_root), "_dsv4_csa_topk_calib_data")
    assert data[64][0][8192][1]["v1_top_last"]["latency"] == 0.05
    assert data[64][0][8192][1]["v1_flat"]["latency"] == 0.0
    assert data[64][0][1024][1]["v1_top_last"]["latency"] == 0.01


def test_calib_view_skips_bad_key_rows_and_keeps_first_on_conflict(systems_root: Path) -> None:
    """The retired parser skipped rows with null/NaN key cells or a blank /
    NaN-sentinel score_mode (``_is_bad_key``) and kept the FIRST value on a
    duplicate coordinate (shared-layer first-wins)."""
    _write_parquet(
        systems_root,
        CALIB_REL,
        [
            _calib_row(latency=1.0),
            _calib_row(latency=2.0),  # duplicate coordinate: first wins
            _calib_row(step=None, latency=3.0),  # null int key -> row skipped
            _calib_row(score_mode="", latency=4.0),  # blank string key -> skipped
            _calib_row(score_mode="nan", latency=5.0),  # NaN sentinel -> skipped
        ],
        types={"step": pa.int64()},
    )
    data = _fetch(_build_db(systems_root), "_dsv4_csa_topk_calib_data")
    assert data == {64: {0: {8192: {1: {"v1_top_last": {"latency": 1.0}}}}}}


def test_calib_view_absent_file_and_empty_rows_return_none(systems_root: Path) -> None:
    """`root or None`: a missing file AND an existing file with zero usable
    rows both answer None (unlike the classic loaders, which distinguish)."""
    assert _fetch(_build_db(systems_root), "_dsv4_csa_topk_calib_data") is None
    _write_parquet(systems_root, CALIB_REL, [_calib_row(score_mode="")])
    assert _fetch(_build_db(systems_root), "_dsv4_csa_topk_calib_data") is None
