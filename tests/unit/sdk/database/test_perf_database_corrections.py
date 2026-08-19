# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

pytestmark = pytest.mark.unit

# The load-time SOL clamp this file used to pin (``GEMM._correct_sol`` /
# ``GenerationAttention._correct_sol``: loaded latencies floored at the SOL
# roofline) retired with the Python query math (#1357 PR-5). The loaded
# Python table is now the RAW collected data plane; the compiled engine
# applies the same clamp on its own load (aic-core/rust perf_database), and
# query values stay SOL-floored via the frozen parity goldens.


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
