# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from aiconfigurator.sdk import common


class TestMoE:
    """Test cases for query_moe method."""

    def test_query_moe_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for MoE."""
        num_tokens = 16
        hidden_size = 2048
        inter_size = 8192
        topk = 2
        num_experts = 8
        moe_tp_size = 2
        moe_ep_size = 2
        quant_mode = common.MoEQuantMode.float16
        workload_distribution = "uniform"

        result = comprehensive_perf_db.query_moe(
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant_mode,
            workload_distribution,
            database_mode=common.DatabaseMode.SOL,
        )

        # Calculate expected SOL result
        total_tokens = num_tokens * topk
        ops = total_tokens * hidden_size * inter_size * 3 * 2 // moe_ep_size // moe_tp_size
        mem_bytes = quant_mode.value.memory * (
            total_tokens * hidden_size * 3
            + total_tokens * inter_size * 3 // moe_tp_size
            + hidden_size * inter_size * 3 // moe_tp_size * min(num_experts // moe_ep_size, total_tokens)
        )
        sol_math = (
            ops / (comprehensive_perf_db.system_spec["gpu"]["float16_tc_flops"] * quant_mode.value.compute) * 1000
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_moe_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_moe(
            8,
            1024,
            4096,
            2,
            8,
            1,
            1,
            common.MoEQuantMode.float16,
            "uniform",
            database_mode=common.DatabaseMode.SOL_FULL,
        )

        sol_only = comprehensive_perf_db.query_moe(
            8,
            1024,
            4096,
            2,
            8,
            1,
            1,
            common.MoEQuantMode.float16,
            "uniform",
            database_mode=common.DatabaseMode.SOL,
        )
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_query_moe_non_database_mode(self, comprehensive_perf_db):
        """Test SILICON mode with data lookup."""
        num_tokens = 8
        hidden_size = 2048
        inter_size = 8192
        topk = 2
        num_experts = 8
        moe_tp_size = 2
        moe_ep_size = 2
        quant_mode = common.MoEQuantMode.float16
        workload_distribution = "uniform"

        result = comprehensive_perf_db.query_moe(
            num_tokens,
            hidden_size,
            inter_size,
            topk,
            num_experts,
            moe_tp_size,
            moe_ep_size,
            quant_mode,
            workload_distribution,
            database_mode=common.DatabaseMode.SILICON,
        )

        # Should use data from moe_data
        expected = comprehensive_perf_db._moe_data[quant_mode][workload_distribution][topk][num_experts][hidden_size][
            inter_size
        ][moe_tp_size][moe_ep_size][num_tokens]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_moe_different_workload_distributions(self, comprehensive_perf_db):
        """Test MoE with different workload distributions."""
        base_params = {
            "num_tokens": 8,
            "hidden_size": 2048,
            "inter_size": 8192,
            "topk": 2,
            "num_experts": 8,
            "moe_tp_size": 1,
            "moe_ep_size": 1,
            "quant_mode": common.MoEQuantMode.float16,
            "database_mode": common.DatabaseMode.SILICON,
        }

        uniform_result = comprehensive_perf_db.query_moe(**base_params, workload_distribution="uniform")
        imbalanced_result = comprehensive_perf_db.query_moe(**base_params, workload_distribution="imbalanced")

        # Both should return valid results
        assert uniform_result > 0
        assert imbalanced_result > 0

    def test_query_moe_edge_cases(self, comprehensive_perf_db):
        """Test edge cases for MoE."""
        # Single token
        result = comprehensive_perf_db.query_moe(
            1,
            1024,
            4096,
            1,
            8,
            1,
            1,
            common.MoEQuantMode.float16,
            "uniform",
            database_mode=common.DatabaseMode.SOL,
        )
        assert result > 0

        # Large EP size (all experts on one device)
        result = comprehensive_perf_db.query_moe(
            8,
            1024,
            4096,
            2,
            8,
            1,
            8,
            common.MoEQuantMode.float16,
            "uniform",
            database_mode=common.DatabaseMode.SOL,
        )
        assert result > 0


class TestMLABMM:
    """Test cases for query_mla_bmm method."""

    def test_query_mla_bmm_database_mode_pre(self, comprehensive_perf_db):
        """Test SOL mode calculation for MLA BMM pre operation."""
        num_tokens = 16
        num_heads = 4
        quant_mode = common.GEMMQuantMode.float16
        if_pre = True

        result = comprehensive_perf_db.query_mla_bmm(
            num_tokens, num_heads, quant_mode, if_pre, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result
        ops = 2 * num_tokens * num_heads * 128 * 512  # 2 for fma
        mem_bytes = num_heads * (num_tokens * 640 + 128 * 512) * quant_mode.value.memory
        sol_math = (
            ops / (comprehensive_perf_db.system_spec["gpu"]["float16_tc_flops"] * quant_mode.value.compute) * 1000
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_mla_bmm_database_mode_post(self, comprehensive_perf_db):
        """Test SOL mode calculation for MLA BMM post operation."""
        num_tokens = 8
        num_heads = 2
        quant_mode = common.GEMMQuantMode.fp8
        if_pre = False

        result = comprehensive_perf_db.query_mla_bmm(
            num_tokens, num_heads, quant_mode, if_pre, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result (same formula for pre/post)
        ops = 2 * num_tokens * num_heads * 128 * 512
        mem_bytes = num_heads * (num_tokens * 640 + 128 * 512) * quant_mode.value.memory
        sol_math = (
            ops / (comprehensive_perf_db.system_spec["gpu"]["float16_tc_flops"] * quant_mode.value.compute) * 1000
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_mla_bmm_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_mla_bmm(
            8, 4, common.GEMMQuantMode.float16, True, database_mode=common.DatabaseMode.SOL_FULL
        )

        sol_only = comprehensive_perf_db.query_mla_bmm(
            8, 4, common.GEMMQuantMode.float16, True, database_mode=common.DatabaseMode.SOL
        )
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_query_mla_bmm_non_database_mode_pre(self, comprehensive_perf_db):
        """Test SILICON mode for pre operation."""
        num_tokens = 8
        num_heads = 4
        quant_mode = common.GEMMQuantMode.float16

        result = comprehensive_perf_db.query_mla_bmm(
            num_tokens, num_heads, quant_mode, True, database_mode=common.DatabaseMode.SILICON
        )

        # Should use data from mla_bmm_data
        expected = comprehensive_perf_db._mla_bmm_data[quant_mode]["mla_gen_pre"][num_heads][num_tokens]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_mla_bmm_non_database_mode_post(self, comprehensive_perf_db):
        """Test SILICON mode for post operation."""
        num_tokens = 16
        num_heads = 2
        quant_mode = common.GEMMQuantMode.fp8

        result = comprehensive_perf_db.query_mla_bmm(
            num_tokens, num_heads, quant_mode, False, database_mode=common.DatabaseMode.SILICON
        )

        # Should use data from mla_bmm_data
        expected = comprehensive_perf_db._mla_bmm_data[quant_mode]["mla_gen_post"][num_heads][num_tokens]
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_mla_bmm_different_configs(self, comprehensive_perf_db):
        """Test MLA BMM with different configurations."""
        configs = [
            (1, 1, common.GEMMQuantMode.float16, True),
            (32, 8, common.GEMMQuantMode.float16, False),
            (16, 4, common.GEMMQuantMode.fp8, True),
            (8, 2, common.GEMMQuantMode.fp8, False),
        ]

        for num_tokens, num_heads, quant_mode, if_pre in configs:
            result = comprehensive_perf_db.query_mla_bmm(
                num_tokens, num_heads, quant_mode, if_pre, database_mode=common.DatabaseMode.SILICON
            )
            assert result > 0, (
                f"Failed for config: tokens={num_tokens}, heads={num_heads}, quant={quant_mode}, pre={if_pre}"
            )


class TestMemoryOperations:
    """Test cases for query_mem_op method."""

    def test_query_mem_op_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for memory operations."""
        mem_bytes = 1_000_000  # 1 MB

        result = comprehensive_perf_db.query_mem_op(mem_bytes, database_mode=common.DatabaseMode.SOL)

        # Calculate expected SOL result
        expected = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_mem_op_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        mem_bytes = 500_000

        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_mem_op(
            mem_bytes, database_mode=common.DatabaseMode.SOL_FULL
        )

        sol_only = comprehensive_perf_db.query_mem_op(mem_bytes, database_mode=common.DatabaseMode.SOL)
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_query_mem_op_non_database_mode(self, comprehensive_perf_db):
        """Test SILICON mode with empirical scaling."""
        mem_bytes = 2_000_000

        result = comprehensive_perf_db.query_mem_op(mem_bytes, database_mode=common.DatabaseMode.SILICON)

        # Calculate expected result with empirical factors
        bw = comprehensive_perf_db.system_spec["gpu"]["mem_bw"]
        scaling = comprehensive_perf_db.system_spec["gpu"]["mem_bw_empirical_scaling_factor"]
        constant = comprehensive_perf_db.system_spec["gpu"]["mem_empirical_constant_latency"]
        expected = (mem_bytes / (bw * scaling) + constant) * 1000

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_mem_op_edge_cases(self, comprehensive_perf_db):
        """Test edge cases for memory operations."""
        # Zero bytes
        result = comprehensive_perf_db.query_mem_op(0, database_mode=common.DatabaseMode.SOL)
        assert result == 0

        # Very small transfer
        result = comprehensive_perf_db.query_mem_op(1, database_mode=common.DatabaseMode.SILICON)
        assert result > 0  # Should include constant latency

        # Large transfer
        result = comprehensive_perf_db.query_mem_op(1_000_000_000, database_mode=common.DatabaseMode.SOL)
        assert result > 0


class TestP2P:
    """Test cases for query_p2p method."""

    def test_query_p2p_database_mode(self, comprehensive_perf_db):
        """Test SOL mode calculation for P2P transfers."""
        message_bytes = 1_000_000  # 1 MB

        result = comprehensive_perf_db.query_p2p(message_bytes, database_mode=common.DatabaseMode.SOL)

        # Calculate expected SOL result
        expected = message_bytes / comprehensive_perf_db.system_spec["node"]["inter_node_bw"] * 1000

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_p2p_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        message_bytes = 500_000

        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_p2p(
            message_bytes, database_mode=common.DatabaseMode.SOL_FULL
        )

        sol_only = comprehensive_perf_db.query_p2p(message_bytes, database_mode=common.DatabaseMode.SOL)
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_query_p2p_non_database_mode(self, comprehensive_perf_db):
        """Test SILICON mode with P2P latency."""
        message_bytes = 2_000_000

        result = comprehensive_perf_db.query_p2p(message_bytes, database_mode=common.DatabaseMode.SILICON)

        # Calculate expected result with P2P latency
        bw = comprehensive_perf_db.system_spec["node"]["inter_node_bw"]
        p2p_latency = comprehensive_perf_db.system_spec["node"]["p2p_latency"]
        expected = (message_bytes / bw + p2p_latency) * 1000

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_p2p_edge_cases(self, comprehensive_perf_db):
        """Test edge cases for P2P transfers."""
        # Zero bytes - should still have latency in SILICON mode
        result_sol = comprehensive_perf_db.query_p2p(0, database_mode=common.DatabaseMode.SOL)
        result_silicon = comprehensive_perf_db.query_p2p(0, database_mode=common.DatabaseMode.SILICON)

        assert result_sol == 0
        assert result_silicon > 0  # Should include P2P latency

        # Small message
        result = comprehensive_perf_db.query_p2p(64, database_mode=common.DatabaseMode.SILICON)
        assert result > 0
