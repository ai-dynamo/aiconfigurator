# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk import operations as ops
from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.perf_database import (
    LoadedOpData,
    PerfDataNotAvailableError,
    _dsv4_flash_robust_3d_lookup,
    load_context_deepseek_v4_attention_module_data,
    load_generation_deepseek_v4_attention_module_data,
    load_mhc_module_data,
)

pytestmark = pytest.mark.unit


def _deepseek_v4_attn_kwargs(compress_ratio: int) -> dict:
    return {
        "b": 2,
        "s": 256,
        "prefix": 0,
        "num_heads": 16,
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "o_lora_rank": 1024,
        "head_dim": 512,
        "rope_head_dim": 64,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 1024,
        "window_size": 128,
        "compress_ratio": compress_ratio,
        "o_groups": 2,
        "kvcache_quant_mode": common.KVCacheQuantMode.fp8,
        "fmha_quant_mode": common.FMHAQuantMode.bfloat16,
        "gemm_quant_mode": common.GEMMQuantMode.fp8_block,
    }


def _deepseek_v4_value(latency: float) -> dict[str, float]:
    return {"latency": latency, "power": 10.0, "energy": latency * 10.0}


def _context_deepseek_v4_data(compress_ratio: int, attn_dict: dict) -> dict:
    return {
        common.FMHAQuantMode.bfloat16: {
            common.KVCacheQuantMode.fp8: {
                common.GEMMQuantMode.fp8_block: {
                    "DeepseekV4ForCausalLM": {
                        compress_ratio: attn_dict,
                    },
                },
            },
        },
    }


def _generation_deepseek_v4_data(compress_ratio: int, attn_dict: dict) -> dict:
    return {
        common.KVCacheQuantMode.fp8: {
            common.GEMMQuantMode.fp8_block: {
                "DeepseekV4ForCausalLM": {
                    compress_ratio: attn_dict,
                },
            },
        },
    }


def _dsv4_flash_sampled_batch_caps_grid() -> dict:
    """Mock the real V4-Flash sampled shape: b=1/2/4/8 with shrinking max s."""
    return {
        8: {
            1024: {
                1: _deepseek_v4_value(1.00),
                2: _deepseek_v4_value(3.00),
                4: _deepseek_v4_value(6.00),
                8: _deepseek_v4_value(12.00),
            },
            2048: {
                1: _deepseek_v4_value(2.00),
                2: _deepseek_v4_value(4.80),
                4: _deepseek_v4_value(8.00),
            },
            4096: {
                1: _deepseek_v4_value(3.00),
                2: _deepseek_v4_value(5.80),
            },
            8192: {
                1: _deepseek_v4_value(4.00),
            },
        }
    }


def _dsv4_flash_generation_sampled_grid() -> dict:
    """Generation shape is [tp][b][s_total]; collector's minimum s_total is 2."""
    return {
        8: {
            1: {
                2: _deepseek_v4_value(0.20),
                5: _deepseek_v4_value(0.50),
            },
            2: {
                2: _deepseek_v4_value(0.40),
                5: _deepseek_v4_value(1.00),
            },
        }
    }


def _dsv4_flash_sparse_kernel_grid(lat_without_prefix: float = 0.02, lat_with_prefix: float = 0.05) -> dict:
    return {
        "DeepseekV4ForCausalLM": {
            1: {
                0: {
                    54.0: {1: {"latency": lat_without_prefix}},
                },
                2816.0: {
                    54.0: {1: {"latency": lat_with_prefix}},
                },
            }
        }
    }


def test_deepseek_v4_module_loaders_are_placeholders(tmp_path):
    assert load_mhc_module_data(str(tmp_path / "mhc_module_perf.txt")) is None
    assert load_context_deepseek_v4_attention_module_data(str(tmp_path / "deepseek_v4_context_module_perf.txt")) is None
    assert (
        load_generation_deepseek_v4_attention_module_data(str(tmp_path / "deepseek_v4_generation_module_perf.txt"))
        is None
    )


class TestDeepSeekV4MHCModule:
    def test_mhc_sol_and_hybrid_return_positive(self, comprehensive_perf_db):
        for mode in (common.DatabaseMode.SOL, common.DatabaseMode.HYBRID):
            result = comprehensive_perf_db.query_mhc_module(
                num_tokens=512,
                hidden_size=7168,
                hc_mult=4,
                sinkhorn_iters=20,
                op="pre",
                quant_mode=common.GEMMQuantMode.bfloat16,
                database_mode=mode,
            )
            assert float(result) > 0

    def test_mhc_sol_full_shape(self, comprehensive_perf_db):
        result = comprehensive_perf_db.query_mhc_module(
            num_tokens=512,
            hidden_size=7168,
            hc_mult=4,
            sinkhorn_iters=20,
            op="both",
            quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert len(result) == 3
        sol_time, sol_math, sol_mem = result
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_mhc_weight_memory_uses_quant_mode(self, comprehensive_perf_db):
        bf16_op = ops.DeepSeekV4MHCModule(
            "mhc",
            1,
            "pre",
            7168,
            4,
            20,
            common.GEMMQuantMode.bfloat16,
        )
        fp8_op = ops.DeepSeekV4MHCModule(
            "mhc",
            1,
            "pre",
            7168,
            4,
            20,
            common.GEMMQuantMode.fp8_block,
        )
        assert fp8_op.get_weights() == pytest.approx(bf16_op.get_weights() / 2)

        bf16_sol = comprehensive_perf_db.query_mhc_module(
            num_tokens=512,
            hidden_size=7168,
            hc_mult=4,
            sinkhorn_iters=20,
            op="pre",
            quant_mode=common.GEMMQuantMode.bfloat16,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        fp8_sol = comprehensive_perf_db.query_mhc_module(
            num_tokens=512,
            hidden_size=7168,
            hc_mult=4,
            sinkhorn_iters=20,
            op="pre",
            quant_mode=common.GEMMQuantMode.fp8_block,
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert fp8_sol[2] < bf16_sol[2]


class TestDeepSeekV4AttentionModule:
    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_context_sol_returns_positive_for_all_attention_kinds(self, comprehensive_perf_db, compress_ratio):
        result = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **_deepseek_v4_attn_kwargs(compress_ratio),
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_generation_hybrid_falls_back_for_all_attention_kinds(self, comprehensive_perf_db, compress_ratio):
        kwargs = _deepseek_v4_attn_kwargs(compress_ratio)
        kwargs.pop("prefix")
        kwargs["s"] = 4096
        result = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **kwargs,
            database_mode=common.DatabaseMode.HYBRID,
        )
        assert float(result) > 0

    def test_generation_uses_pre_decode_kv_length(self, comprehensive_perf_db):
        base = _deepseek_v4_attn_kwargs(4)
        base.pop("prefix")
        current = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **{**base, "s": 512},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        next_step = comprehensive_perf_db.query_generation_deepseek_v4_attention_module(
            **{**base, "s": 513},
            database_mode=common.DatabaseMode.SOL_FULL,
        )

        assert next_step[1] > current[1]

    def test_generation_robust_lookup_extrapolates_query_below_min_sampled_s_total(self, mutable_comprehensive_perf_db):
        """Regression: query s_total=1 is below the collector's sampled s_total=2."""
        db = mutable_comprehensive_perf_db
        mock_grid = _dsv4_flash_generation_sampled_grid()

        result = _dsv4_flash_robust_3d_lookup(db, mock_grid, 8, 1, 1, batch_axis="y")

        expected = 0.20 + (0.50 - 0.20) * (1 - 2) / (5 - 2)
        assert math.isclose(result["latency"], expected, rel_tol=1e-6)
        assert math.isclose(result["energy"], expected * 10.0, rel_tol=1e-6)

    def test_generation_silicon_extrapolates_query_below_min_sampled_s_total(self, mutable_comprehensive_perf_db):
        """Full-query regression for generated query b=1, s_total=1, tp=8."""
        db = mutable_comprehensive_perf_db
        mock_grid = _dsv4_flash_generation_sampled_grid()
        db._generation_deepseek_v4_attention_module_data = LoadedOpData(
            _generation_deepseek_v4_data(4, mock_grid),
            common.PerfDataFilename.deepseek_v4_generation_module,
            "mock_dsv4_flash_generation_module_tp8",
        )
        kwargs = _deepseek_v4_attn_kwargs(4)
        kwargs.pop("prefix")

        result = db.query_generation_deepseek_v4_attention_module(
            **{
                **kwargs,
                "b": 1,
                "s": 1,
                "num_heads": 8,
            },
            database_mode=common.DatabaseMode.SILICON,
        )

        expected = 0.20 + (0.50 - 0.20) * (1 - 2) / (5 - 2)
        assert float(result) == pytest.approx(expected)
        assert result.energy == pytest.approx(expected * 10.0)

    def test_csa_topk_changes_attention_workload(self, comprehensive_perf_db):
        base = _deepseek_v4_attn_kwargs(4)
        low_topk = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "index_topk": 128, "s": 4096},
            database_mode=common.DatabaseMode.SOL,
        )
        high_topk = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "index_topk": 1024, "s": 4096},
            database_mode=common.DatabaseMode.SOL,
        )
        assert high_topk > low_topk

    def test_csa_context_uses_raw_piecewise_around_compressed_topk_boundary(self, mutable_comprehensive_perf_db):
        db = mutable_comprehensive_perf_db
        # Data keyed by tp_size (=4 for num_heads=16, recovered as 64 // num_heads).
        # See _dsv4_flash_tp_from_num_heads in perf_database.py.
        raw_attn_dict = {
            4: {
                4096: {2: _deepseek_v4_value(20.0)},
                8192: {2: _deepseek_v4_value(80.0)},
                12288: {2: _deepseek_v4_value(100.0)},
            }
        }
        extrapolated_attn_dict = {
            4: {
                4096: {2: _deepseek_v4_value(20.0)},
                4097: {2: _deepseek_v4_value(21.0)},
                8192: {2: _deepseek_v4_value(80.0)},
                12288: {2: _deepseek_v4_value(100.0)},
            }
        }
        db._raw_context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, raw_attn_dict), common.PerfDataFilename.deepseek_v4_context_module, "raw"
        )
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, extrapolated_attn_dict),
            common.PerfDataFilename.deepseek_v4_context_module,
            "extrapolated",
        )

        def fail_interp_3d(*args, **kwargs):
            raise AssertionError("_interp_3d should not be used when raw same-regime CSA anchors exist")

        db._interp_3d = fail_interp_3d

        base = _deepseek_v4_attn_kwargs(4)
        result = db.query_context_deepseek_v4_attention_module(
            **{**base, "s": 4097, "prefix": 0},
            database_mode=common.DatabaseMode.SILICON,
        )

        expected = 80.0 + (100.0 - 80.0) / (12288 - 8192) * (4097 - 8192)
        assert float(result) == pytest.approx(expected)
        assert result.energy == pytest.approx(expected * 10.0)

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_context_prefix_changes_sol_for_all_attention_kinds(self, comprehensive_perf_db, compress_ratio):
        base = {**_deepseek_v4_attn_kwargs(compress_ratio), "s": 512}
        no_prefix = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **base,
            database_mode=common.DatabaseMode.SOL,
        )
        with_prefix = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "prefix": 1024},
            database_mode=common.DatabaseMode.SOL,
        )
        assert with_prefix > no_prefix

    @pytest.mark.parametrize("compress_ratio", [0, 4, 128])
    def test_context_sol_increases_with_sequence_length(self, comprehensive_perf_db, compress_ratio):
        base = _deepseek_v4_attn_kwargs(compress_ratio)
        short = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "s": 256},
            database_mode=common.DatabaseMode.SOL,
        )
        long = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "s": 4096},
            database_mode=common.DatabaseMode.SOL,
        )
        assert long > short

    def test_csa_indexer_logits_scale_with_compressed_length_not_topk_only(self, comprehensive_perf_db):
        base = {**_deepseek_v4_attn_kwargs(4), "s": 4096}
        short_cache = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "prefix": 0, "index_topk": 16},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        long_cache = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "prefix": 4096, "index_topk": 16},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert long_cache[1] > short_cache[1]

    def test_kvcache_quant_changes_sol_memory(self, comprehensive_perf_db):
        base = {**_deepseek_v4_attn_kwargs(128), "s": 4096, "prefix": 4096}
        bf16 = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "kvcache_quant_mode": common.KVCacheQuantMode.bfloat16},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        fp8 = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "kvcache_quant_mode": common.KVCacheQuantMode.fp8},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert fp8[2] < bf16[2]

    def test_gemm_quant_changes_sol_math_and_memory(self, comprehensive_perf_db):
        base = {**_deepseek_v4_attn_kwargs(4), "s": 4096, "prefix": 1024}
        bf16 = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "gemm_quant_mode": common.GEMMQuantMode.bfloat16},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        fp8 = comprehensive_perf_db.query_context_deepseek_v4_attention_module(
            **{**base, "gemm_quant_mode": common.GEMMQuantMode.fp8_block},
            database_mode=common.DatabaseMode.SOL_FULL,
        )
        assert fp8[1] < bf16[1]
        assert fp8[2] < bf16[2]

    def test_robust_3d_lookup_uses_b2_when_b3_s2682_is_missing(self, mutable_comprehensive_perf_db):
        """Regression: b=3, s=2682 uses b=2 because b=4 only reaches s=2048."""
        db = mutable_comprehensive_perf_db
        mock_grid = _dsv4_flash_sampled_batch_caps_grid()

        result = _dsv4_flash_robust_3d_lookup(db, mock_grid, 8, 2682, 3)
        b2_at_2682 = 4.80 + (5.80 - 4.80) * (2682 - 2048) / (4096 - 2048)
        expected = b2_at_2682 * 3 / 2
        assert math.isclose(result["latency"], expected, rel_tol=1e-6)

    def test_robust_3d_lookup_uses_b4_for_bs5_real_mixed_batch_shape(self, mutable_comprehensive_perf_db):
        """Regression: real bs=5 mixed-prefix request uses b=4 at avg_isl=1565.2."""
        db = mutable_comprehensive_perf_db
        reqs = [(1266, 1792), (1292, 1280), (1251, 1792), (2225, 1280), (1792, 1792)]
        avg_isl = sum(isl for isl, _ in reqs) / len(reqs)
        avg_past_kv = sum(past_kv for _, past_kv in reqs) / len(reqs)
        bs = len(reqs)

        assert bs == 5
        assert avg_isl == pytest.approx(1565.2)
        assert avg_past_kv == pytest.approx(1587.2)

        mock_grid = _dsv4_flash_sampled_batch_caps_grid()
        result = _dsv4_flash_robust_3d_lookup(db, mock_grid, 8, avg_isl, bs)

        b4_at_avg_isl = 6.00 + (8.00 - 6.00) * (avg_isl - 1024) / (2048 - 1024)
        expected = b4_at_avg_isl * 5 / 4
        assert math.isclose(result["latency"], expected, rel_tol=1e-6)
        assert math.isclose(result["energy"], expected * 10.0, rel_tol=1e-6)

    def test_robust_3d_lookup_uses_b1_for_bs1_short_single_attn_shape(self, mutable_comprehensive_perf_db):
        """Regression: bs=1 has no smaller batch, so use b=1 along the s axis."""
        db = mutable_comprehensive_perf_db
        reqs = [(54, 2816)]
        avg_isl = sum(isl for isl, _ in reqs) / len(reqs)
        avg_past_kv = sum(past_kv for _, past_kv in reqs) / len(reqs)
        bs = len(reqs)

        assert bs == 1
        assert avg_isl == pytest.approx(54.0)
        assert avg_past_kv == pytest.approx(2816.0)

        mock_grid = _dsv4_flash_sampled_batch_caps_grid()
        result = _dsv4_flash_robust_3d_lookup(db, mock_grid, 8, avg_isl, bs)

        b1_at_avg_isl = 1.00 + (2.00 - 1.00) * (avg_isl - 1024) / (2048 - 1024)
        assert math.isclose(result["latency"], b1_at_avg_isl, rel_tol=1e-6)

    def test_context_silicon_handles_bs1_s54_prefix2816_single_attn_module(self, mutable_comprehensive_perf_db):
        """Full-query regression for the single bs=1, isl=54, prefix=2816 attention module."""
        db = mutable_comprehensive_perf_db
        module_grid = _dsv4_flash_sampled_batch_caps_grid()
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.deepseek_v4_context_module,
            "mock_dsv4_flash_context_module_tp8",
        )
        db._raw_context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.deepseek_v4_context_module,
            "mock_raw_dsv4_flash_context_module_tp8",
        )
        sparse_grid = _dsv4_flash_sparse_kernel_grid()
        db._dsv4_flash_sparse_kernel_data = {
            "paged_mqa_logits": LoadedOpData(
                sparse_grid,
                common.PerfDataFilename.dsv4_flash_paged_mqa_logits_module,
                "mock_dsv4_flash_paged_mqa_logits_module",
            ),
        }

        result = db.query_context_deepseek_v4_attention_module(
            **{
                **_deepseek_v4_attn_kwargs(4),
                "b": 1,
                "s": 54,
                "prefix": 2816,
                "num_heads": 8,
            },
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) > 0
        assert result.energy >= 0

    def test_context_silicon_errors_when_prefix_sparse_kernel_delta_missing(self, mutable_comprehensive_perf_db):
        """Prefix CSA needs paged_mqa_logits delta; do not silently query s+prefix."""
        db = mutable_comprehensive_perf_db
        module_grid = _dsv4_flash_sampled_batch_caps_grid()
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.deepseek_v4_context_module,
            "mock_dsv4_flash_context_module_tp8",
        )

        with pytest.raises(PerfDataNotAvailableError, match="paged_mqa_logits sparse-kernel correction"):
            db.query_context_deepseek_v4_attention_module(
                **{
                    **_deepseek_v4_attn_kwargs(4),
                    "b": 1,
                    "s": 54,
                    "prefix": 2816,
                    "num_heads": 8,
                },
                database_mode=common.DatabaseMode.SILICON,
            )

    def test_context_silicon_handles_b3_s2682_prefix0_num_heads8_from_sampled_batches(
        self, mutable_comprehensive_perf_db
    ):
        """Full-query regression for sampled b=2/b=4 data and query b=3."""
        db = mutable_comprehensive_perf_db
        module_grid = _dsv4_flash_sampled_batch_caps_grid()
        db._context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.deepseek_v4_context_module,
            "mock_dsv4_flash_context_module_tp8",
        )
        db._raw_context_deepseek_v4_attention_module_data = LoadedOpData(
            _context_deepseek_v4_data(4, module_grid),
            common.PerfDataFilename.deepseek_v4_context_module,
            "mock_raw_dsv4_flash_context_module_tp8",
        )

        result = db.query_context_deepseek_v4_attention_module(
            **{
                **_deepseek_v4_attn_kwargs(4),
                "b": 3,
                "s": 2682,
                "prefix": 0,
                "num_heads": 8,
            },
            database_mode=common.DatabaseMode.SILICON,
        )

        assert float(result) > 0
        assert result.energy >= 0


def test_deepseek_v4_static_sol_and_hybrid_run_end_to_end(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    db.system_spec["gpu"]["mem_capacity"] = 288400343040
    db.system_spec["misc"]["nccl_mem"] = {1: 0, 2: 0, 4: 0, 8: 0}
    db.system_spec["misc"]["other_mem"] = 0
    model_config = config.ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        nextn=1,
        nextn_accept_rates=[0.85, 0.3, 0.0, 0.0, 0.0],
        overwrite_num_layers=2,
    )
    model = get_model("sgl-project/DeepSeek-V4-Flash-FP8", model_config, backend_name="trtllm")
    backend = TRTLLMBackend()
    runtime = RuntimeConfig(batch_size=1, beam_width=1, isl=128, osl=4, prefix=0)

    for mode in (common.DatabaseMode.SOL, common.DatabaseMode.HYBRID):
        db.set_default_database_mode(mode)
        summary = backend.run_static(model, db, runtime, mode="static", stride=1)
        assert sum(summary.get_context_latency_dict().values()) > 0
        assert sum(summary.get_generation_latency_dict().values()) > 0


def test_sglang_deepseek_v4_pro_moe_workspace_uses_residual_hidden_size(mutable_comprehensive_perf_db):
    db = mutable_comprehensive_perf_db
    db.system_spec["gpu"]["mem_capacity"] = 198674743296  # GB200 189471 MiB
    db.system_spec["misc"]["nccl_mem"] = {1: 0, 2: 358612992, 4: 411041792, 8: 411041792}
    db.system_spec["misc"]["other_mem"] = 3758096384

    model_config = config.ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.w4a8_mxfp4_mxfp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        nextn=0,
    )
    model = get_model("deepseek-ai/DeepSeek-V4-Pro", model_config, backend_name="sglang")

    memory = SGLANGBackend()._get_memory_usage(
        model,
        db,
        batch_size=1,
        beam_width=1,
        isl=8192,
        osl=1024,
    )

    num_tokens = 8192
    attention_width = model._num_heads * model._head_size
    residual_width = model._hidden_size
    tp_activation_factor = 28
    attention_workspace = 2 * num_tokens * attention_width * tp_activation_factor
    moe_scale_workspace = (
        num_tokens
        * residual_width
        * model.config.attention_dp_size
        * model._num_experts
        * model._topk
        / model.config.moe_ep_size
        / 128
        * 4
    )
    expected_activation_gib = (attention_workspace + moe_scale_workspace) * 1.15 / (1 << 30)

    assert memory["activations"] == pytest.approx(expected_activation_gib)

    old_moe_scale_workspace = (
        num_tokens
        * attention_width
        * model.config.attention_dp_size
        * model._num_experts
        * model._topk
        / model.config.moe_ep_size
        / 128
        * 4
    )
    old_activation_gib = (attention_workspace + old_moe_scale_workspace) * 1.15 / (1 << 30)
    assert memory["activations"] < old_activation_gib
