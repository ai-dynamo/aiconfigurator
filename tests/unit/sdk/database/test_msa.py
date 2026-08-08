# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) op: SOL, the own silicon tables, and the
cross-op (XOP) transfer from DSA.

MSA now has its own msa_*_module silicon tables (DSA-module row schema); the
empirical fallback remains the DSA cross-op transfer gated by the XOP transfer
kind. These tests cover the SOL path, the silicon table lookup, the
silicon-over-xop HYBRID priority, and the XOP gate."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import EmpiricalNotImplementedError
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataNotAvailableError

pytestmark = pytest.mark.unit

MSA_ARCHITECTURE = "MiniMaxM3ForCausalLM"


def _msa_value(latency: float) -> dict[str, float]:
    return {"latency": latency, "power": 10.0, "energy": latency * 10.0}


def _ctx_msa(**overrides):
    from aiconfigurator.sdk.operations.msa import ContextMSAModule

    # M3-like per-GPU shape: 8 q / 1 kv heads, head_dim 128, v 128, top-16 blocks * 128.
    kwargs = dict(
        num_heads=8,
        num_kv_heads=1,
        hidden_size=4096,
        head_dim=128,
        v_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=2048,
        block_size=128,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
    )
    kwargs.update(overrides)
    return ContextMSAModule("msa", 1.0, **kwargs)


def _gen_msa(**overrides):
    from aiconfigurator.sdk.operations.msa import GenerationMSAModule

    kwargs = dict(
        num_heads=8,
        num_kv_heads=1,
        hidden_size=4096,
        head_dim=128,
        v_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=2048,
        block_size=128,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
    )
    kwargs.update(overrides)
    return GenerationMSAModule("msa", 1.0, **kwargs)


# The DSA-schema loader nests ...[architecture][dsa_backend][num_heads]...;
# msa rows carry kernel_source="default" so fp8-KV rows land in the
# "flashmla_kv" bucket (bf16-KV rows back both) — the fixtures mirror that.
def _context_msa_data(msa_dict: dict, kv=common.KVCacheQuantMode.bfloat16) -> dict:
    return {
        common.FMHAQuantMode.bfloat16: {
            kv: {
                common.GEMMQuantMode.bfloat16: {
                    MSA_ARCHITECTURE: {"flashmla_kv": msa_dict},
                },
            },
        },
    }


def _generation_msa_data(msa_dict: dict) -> dict:
    return {
        common.KVCacheQuantMode.bfloat16: {
            common.GEMMQuantMode.bfloat16: {
                MSA_ARCHITECTURE: {"flashmla_kv": msa_dict},
            },
        },
    }


def _bf16_ctx_msa():
    return _ctx_msa(
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
    )


def test_msa_sol_scales_with_workload(comprehensive_perf_db):
    """SOL mode computes the three-group MSA SOL (gemm + fp8 indexer + sparse attn). Assert it
    RESPONDS to the workload rather than returning a constant: more new tokens (s) add work, and
    a longer cached prefix adds indexer/attention work (full_s > index_topk)."""
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SOL)
    try:
        op = _ctx_msa()
        small = float(op.query(comprehensive_perf_db, batch_size=8, s=512, prefix=0))
        large = float(op.query(comprehensive_perf_db, batch_size=8, s=2048, prefix=0))
        with_prefix = float(op.query(comprehensive_perf_db, batch_size=8, s=2048, prefix=2048))
        assert 0 < small < large  # scales with new-token count
        assert with_prefix > large  # cached prefix adds indexer work beyond index_topk
    finally:
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)


def test_msa_xop_gating(comprehensive_perf_db, monkeypatch):
    """The DSA-to-MSA utilization transfer is gated and tagged as XOP."""
    from aiconfigurator.sdk.operations import util_empirical

    util_queries = []

    def dsa_util(_database, **kwargs):
        util_queries.append(kwargs)
        return 0.5

    monkeypatch.setattr("aiconfigurator.sdk.operations.msa._dsa_context_util", dsa_util)
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.HYBRID)
    kw = dict(batch_size=8, s=2048, prefix=0)
    try:
        comprehensive_perf_db.set_transfer_policy(["xshape", "xquant"])  # no XOP
        with pytest.raises(EmpiricalNotImplementedError) as exc:
            _ctx_msa().query(comprehensive_perf_db, **kw)
        assert "xop" in str(exc.value).lower()  # gated at the policy, not a data miss
        assert util_queries == []

        comprehensive_perf_db.set_transfer_policy(None)  # XOP allowed
        with util_empirical.capture_provenance() as tags:
            assert float(_ctx_msa().query(comprehensive_perf_db, **kw)) > 0
        assert len(util_queries) == 1
        assert util_empirical.worst_provenance(tags) == "xop"
    finally:
        comprehensive_perf_db.set_transfer_policy(None)
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)


# ═══════════════════════════════════════════════════════════════════════
# Own silicon tables (msa_context_module / msa_generation_module)
# ═══════════════════════════════════════════════════════════════════════


class TestMSASiliconTables:
    def test_context_silicon_exact_hit(self, stub_perf_db):
        msa_dict = {8: {0: {256: {1: _msa_value(10.0)}}}}
        stub_perf_db._context_msa_module_data = LoadedOpData(
            _context_msa_data(msa_dict), common.PerfDataFilename.msa_context_module, "injected"
        )

        result = _bf16_ctx_msa().query(stub_perf_db, batch_size=1, s=256, prefix=0)

        assert float(result) == pytest.approx(10.0)
        assert result.source == "silicon"

    def test_context_silicon_prefix_axis_interpolates(self, stub_perf_db):
        # Two prefix points bracketing the query: perf_interp resolves the
        # 4-axis [heads][prefix][s][b] raw grid, so a mid-prefix query lands
        # strictly between the collected latencies.
        msa_dict = {8: {0: {256: {1: _msa_value(10.0)}}, 512: {256: {1: _msa_value(20.0)}}}}
        stub_perf_db._context_msa_module_data = LoadedOpData(
            _context_msa_data(msa_dict), common.PerfDataFilename.msa_context_module, "injected"
        )

        result = _bf16_ctx_msa().query(stub_perf_db, batch_size=1, s=256, prefix=256)

        assert 10.0 < float(result) < 20.0
        assert result.source == "silicon"

    def test_context_silicon_missing_quant_slice_raises_data_not_available(self, stub_perf_db):
        # Table exists but only for bf16 KV; the fp8-KV op must get the typed
        # "data not available" miss in SILICON mode.
        msa_dict = {8: {0: {256: {1: _msa_value(10.0)}}}}
        stub_perf_db._context_msa_module_data = LoadedOpData(
            _context_msa_data(msa_dict), common.PerfDataFilename.msa_context_module, "injected"
        )

        with pytest.raises(PerfDataNotAvailableError, match="Context MSA module data not available"):
            _ctx_msa().query(stub_perf_db, batch_size=1, s=256, prefix=0)

    def test_generation_silicon_exact_hit(self, stub_perf_db):
        # Generation grid is [num_heads][b][s], s = total decode length.
        msa_dict = {8: {1: {4097: _msa_value(0.5)}}}
        stub_perf_db._generation_msa_module_data = LoadedOpData(
            _generation_msa_data(msa_dict), common.PerfDataFilename.msa_generation_module, "injected"
        )

        result = _gen_msa().query(stub_perf_db, batch_size=1, s=4097, beam_width=1)

        assert float(result) == pytest.approx(0.5)
        assert result.source == "silicon"

    def test_generation_silicon_missing_table_raises_data_not_available(self, stub_perf_db):
        with pytest.raises(PerfDataNotAvailableError, match="Generation MSA module data not available"):
            _gen_msa().query(stub_perf_db, batch_size=1, s=4097, beam_width=1)

    def test_hybrid_prefers_silicon_over_xop(self, stub_perf_db, monkeypatch):
        """With MSA data present, HYBRID must answer from the table and never
        touch the DSA transfer."""
        from aiconfigurator.sdk.operations import util_empirical

        util_queries = []
        monkeypatch.setattr(
            "aiconfigurator.sdk.operations.msa._dsa_context_util",
            lambda _db, **kw: util_queries.append(kw) or 0.5,
        )
        msa_dict = {8: {0: {256: {1: _msa_value(10.0)}}}}
        stub_perf_db._context_msa_module_data = LoadedOpData(
            _context_msa_data(msa_dict), common.PerfDataFilename.msa_context_module, "injected"
        )
        stub_perf_db.set_default_database_mode(common.DatabaseMode.HYBRID)

        with util_empirical.capture_provenance() as tags:
            result = _bf16_ctx_msa().query(stub_perf_db, batch_size=1, s=256, prefix=0)

        assert float(result) == pytest.approx(10.0)
        assert result.source == "silicon"
        assert util_queries == []  # xop never consulted
        assert util_empirical.worst_provenance(tags) == "silicon"

    def test_hybrid_falls_back_to_xop_without_msa_table(self, stub_perf_db, monkeypatch):
        """No MSA table -> the pre-existing DSA cross-op transfer still fires
        (no regression of the legacy behavior)."""
        from aiconfigurator.sdk.operations import util_empirical

        monkeypatch.setattr("aiconfigurator.sdk.operations.msa._dsa_context_util", lambda _db, **kw: 0.5)
        stub_perf_db.set_default_database_mode(common.DatabaseMode.HYBRID)

        op = _bf16_ctx_msa()
        with util_empirical.capture_provenance() as tags:
            result = op.query(stub_perf_db, batch_size=1, s=256, prefix=0)

        sol = op._sol(stub_perf_db, 1, 256, 0, is_context=True)
        assert float(result) == pytest.approx(sol / 0.5)
        assert result.source == "empirical"
        assert util_empirical.worst_provenance(tags) == "xop"


# ═══════════════════════════════════════════════════════════════════════
# Loaders (DSA-schema delegation)
# ═══════════════════════════════════════════════════════════════════════


def _write_msa_parquet(path, op_name: str, rows: list[dict]) -> None:
    """Write rows with the exact collector schema (collect_msa_module.py)."""
    columns = {
        "model": [r.get("model", "MiniMaxAI/MiniMax-M3") for r in rows],
        "architecture": [r.get("architecture", MSA_ARCHITECTURE) for r in rows],
        "op_name": [op_name for _ in rows],
        "kernel_source": [r.get("kernel_source", "default") for r in rows],
        "mla_dtype": [r.get("mla_dtype", "bfloat16") for r in rows],
        "kv_cache_dtype": [r.get("kv_cache_dtype", "bfloat16") for r in rows],
        "gemm_type": [r.get("gemm_type", "bfloat16") for r in rows],
        "num_heads": [r["num_heads"] for r in rows],
        "batch_size": [r["batch_size"] for r in rows],
        "isl": [r["isl"] for r in rows],
        "tp_size": [r.get("tp_size", 8) for r in rows],
        "step": [r["step"] for r in rows],
        "latency": [r["latency"] for r in rows],
    }
    pq.write_table(pa.table(columns), path)


def test_context_msa_loader_keys_prefix_axis(tmp_path):
    from aiconfigurator.sdk.operations.msa import load_context_msa_module_data

    data_path = tmp_path / "msa_context_module_perf.parquet"
    _write_msa_parquet(
        data_path,
        "msa_context_module",
        [
            {"num_heads": 8, "batch_size": 1, "isl": 256, "step": 0, "latency": 10.0},
            {"num_heads": 8, "batch_size": 1, "isl": 256, "step": 512, "latency": 20.0},
        ],
    )

    data = load_context_msa_module_data(str(data_path))

    head_data = data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][
        MSA_ARCHITECTURE
    ]["flashmla_kv"][8]
    assert head_data[0][256][1]["latency"] == pytest.approx(10.0)  # [prefix][s][b]
    assert head_data[512][256][1]["latency"] == pytest.approx(20.0)


def test_generation_msa_loader_collapses_isl_step_to_seq(tmp_path):
    from aiconfigurator.sdk.operations.msa import load_generation_msa_module_data

    data_path = tmp_path / "msa_generation_module_perf.parquet"
    # collector semantics: isl=1, step=kv_len -> canonical s = isl + step.
    _write_msa_parquet(
        data_path,
        "msa_generation_module",
        [{"num_heads": 8, "batch_size": 4, "isl": 1, "step": 4096, "latency": 0.5}],
    )

    data = load_generation_msa_module_data(str(data_path))

    head_data = data[common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][MSA_ARCHITECTURE]["flashmla_kv"][
        8
    ]
    assert head_data[4][4097]["latency"] == pytest.approx(0.5)  # [b][s = isl + step]
