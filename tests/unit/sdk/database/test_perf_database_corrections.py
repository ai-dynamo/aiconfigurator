# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


class TestCorrectData:
    """Test cases for per-op ``_correct_sol`` SOL clamping.

    The ``PerfDatabase._correct_data`` wrapper has been retired;
    callers invoke ``GEMM._correct_sol(db)`` / ``GenerationAttention._correct_sol(db)``
    directly. These tests exercise the SOL clamp on a database whose
    instance attributes have been mutated to artificially low values."""

    def test_correct_gemm_data(self, mutable_comprehensive_perf_db, caplog):
        """``GEMM._correct_sol`` clamps GEMM data to >= SOL."""
        from aiconfigurator.sdk.operations.gemm import GEMM

        db = mutable_comprehensive_perf_db
        quant_mode = common.GEMMQuantMode.bfloat16
        m, n, k = 64, 128, 256

        # Calculate what SOL should be
        sol_value = db.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL)

        # Set an artificially low value
        db._gemm_data[quant_mode][m][n][k] = sol_value * 0.5

        with caplog.at_level("DEBUG"):
            GEMM._correct_sol(db)

        assert db._gemm_data[quant_mode][m][n][k] >= sol_value
        assert f"sol {sol_value} > perf_db" in caplog.text or "gemm quant" in caplog.text

    def test_correct_generation_attention_data(self, mutable_comprehensive_perf_db, caplog):
        """``GenerationAttention._correct_sol`` clamps generation attention data to >= SOL."""
        from aiconfigurator.sdk.operations.attention import GenerationAttention

        db = mutable_comprehensive_perf_db
        kv_cache_quant_mode = common.KVCacheQuantMode.bfloat16
        n_kv = 0  # MHA case
        n, b, s = 16, 4, 64

        # Calculate SOL
        sol_value = db.query_generation_attention(
            b, s, n, n, kv_cache_quant_mode, database_mode=common.DatabaseMode.SOL
        )

        # Set an artificially low value
        db._generation_attention_data[kv_cache_quant_mode][n_kv][128][0][n][b][s] = sol_value * 0.5

        with caplog.at_level("DEBUG"):
            GenerationAttention._correct_sol(db)

        corrected_value = db._generation_attention_data[kv_cache_quant_mode][n_kv][128][0][n][b][s]
        assert corrected_value >= sol_value


class TestUpdateSupportMatrix:
    """Test cases for _update_support_matrix method."""

    def test_support_matrix_creation(self, comprehensive_perf_db):
        """Test that supported_quant_mode is properly created."""
        # ``supported_quant_mode`` is a ``_LazySupportMatrix`` (dict-like
        # view that resolves keys on first read) rather than a plain
        # dict. Both shapes support the same per-key access pattern the
        # rest of this test exercises.
        from aiconfigurator.sdk.perf_database import _LazySupportMatrix

        assert hasattr(comprehensive_perf_db, "supported_quant_mode")
        assert isinstance(comprehensive_perf_db.supported_quant_mode, dict | _LazySupportMatrix)

        # Check expected keys
        expected_keys = [
            "gemm",
            "context_attention",
            "generation_attention",
            "context_mla",
            "generation_mla",
            "mla_bmm",
            "nccl",
            "moe",
        ]
        for key in expected_keys:
            assert key in comprehensive_perf_db.supported_quant_mode
            assert isinstance(comprehensive_perf_db.supported_quant_mode[key], list)

        # Verify some expected quant modes
        assert "bfloat16" in comprehensive_perf_db.supported_quant_mode["gemm"]
        assert "bfloat16" in comprehensive_perf_db.supported_quant_mode["context_attention"]
