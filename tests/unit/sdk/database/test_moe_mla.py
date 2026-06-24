# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import EmpiricalNotImplementedError
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


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
        quant_mode = common.MoEQuantMode.bfloat16
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
            ops / (comprehensive_perf_db.system_spec["gpu"]["bfloat16_tc_flops"] * quant_mode.value.compute) * 1000
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)
        assert result.source == "sol"

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
            common.MoEQuantMode.bfloat16,
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
            common.MoEQuantMode.bfloat16,
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
        quant_mode = common.MoEQuantMode.bfloat16
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
            "quant_mode": common.MoEQuantMode.bfloat16,
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
            common.MoEQuantMode.bfloat16,
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
            common.MoEQuantMode.bfloat16,
            "uniform",
            database_mode=common.DatabaseMode.SOL,
        )
        assert result > 0

    def test_query_moe_silicon_within_range_uses_interpolation(self, comprehensive_perf_db):
        """When num_tokens <= max(moe_dict), result comes from interpolation (or exact hit)."""
        # Fixture has token points [1, 2, 4, 8, 16, 32]; 8 is an exact grid point.
        num_tokens = 8
        result = comprehensive_perf_db.query_moe(
            num_tokens,
            2048,
            8192,
            2,
            8,
            2,
            2,
            common.MoEQuantMode.bfloat16,
            "uniform",
            database_mode=common.DatabaseMode.SILICON,
        )
        assert isinstance(result, PerformanceResult)
        assert float(result) > 0
        # Exact hit: fixture uses 0.1 * num_tokens per point
        expected = comprehensive_perf_db._moe_data[common.MoEQuantMode.bfloat16]["uniform"][2][8][2048][8192][2][2][
            num_tokens
        ]
        assert math.isclose(float(result), expected, rel_tol=1e-6)

    def test_query_moe_silicon_overflow_uses_util_extrapolation(self, comprehensive_perf_db):
        """When num_tokens > max(moe_dict), result comes from _estimate_overflow_with_last_token_util."""
        # Fixture max token point is 32; query beyond it to trigger overflow path.
        max_stored = 32
        num_tokens_overflow = 64
        result = comprehensive_perf_db.query_moe(
            num_tokens_overflow,
            2048,
            8192,
            2,
            8,
            2,
            2,
            common.MoEQuantMode.bfloat16,
            "uniform",
            database_mode=common.DatabaseMode.SILICON,
        )
        assert isinstance(result, PerformanceResult)
        assert float(result) > 0
        # Extrapolated latency should be greater than latency at max stored point
        latency_at_max = comprehensive_perf_db._moe_data[common.MoEQuantMode.bfloat16]["uniform"][2][8][2048][8192][2][
            2
        ][max_stored]
        assert float(result) > latency_at_max

    def test_query_moe_silicon_boundary_at_max_tokens(self, comprehensive_perf_db):
        """When num_tokens == max(moe_dict), interpolation path is used (exact hit), not overflow."""
        max_stored = 32
        result = comprehensive_perf_db.query_moe(
            max_stored,
            2048,
            8192,
            2,
            8,
            2,
            2,
            common.MoEQuantMode.bfloat16,
            "uniform",
            database_mode=common.DatabaseMode.SILICON,
        )
        assert isinstance(result, PerformanceResult)
        expected = comprehensive_perf_db._moe_data[common.MoEQuantMode.bfloat16]["uniform"][2][8][2048][8192][2][2][
            max_stored
        ]
        assert math.isclose(float(result), expected, rel_tol=1e-6)

    def test_query_moe_vllm_missing_bucket_raises_structured_error(self, mutable_comprehensive_perf_db):
        """Missing MoE token buckets must not leak raw IndexError/KeyError in SILICON mode."""
        db = mutable_comprehensive_perf_db
        db.backend = common.BackendName.vllm.value
        db._moe_data[common.MoEQuantMode.bfloat16]["uniform"][2][8][2048][8192][1][3] = {}

        with pytest.raises(PerfDataNotAvailableError) as exc_info:
            db.query_moe(
                22,
                2048,
                8192,
                2,
                8,
                1,
                3,
                common.MoEQuantMode.bfloat16,
                "uniform",
                database_mode=common.DatabaseMode.SILICON,
            )

        message = str(exc_info.value)
        assert "No MoE silicon data points" in message
        assert "Consider using HYBRID mode" in message
        assert "IndexError" not in message
        assert "KeyError" not in message

    def test_query_moe_vllm_missing_dimension_wraps_keyerror(self, mutable_comprehensive_perf_db):
        """Missing MoE shape dimensions must also be converted to typed missing-data errors."""
        db = mutable_comprehensive_perf_db
        db.backend = common.BackendName.vllm.value

        with pytest.raises(PerfDataNotAvailableError) as exc_info:
            db.query_moe(
                22,
                2048,
                8192,
                2,
                8,
                1,
                3,
                common.MoEQuantMode.bfloat16,
                "uniform",
                database_mode=common.DatabaseMode.SILICON,
            )

        message = str(exc_info.value)
        assert "Missing silicon data for the requested lookup" in message
        assert "Consider using HYBRID mode" in message
        assert "KeyError" not in message
        assert "IndexError" not in message

    def test_query_moe_vllm_missing_bucket_hybrid_raises(self, mutable_comprehensive_perf_db):
        """With the bucket empty and no cross-shape transfer reference, HYBRID raises
        EmpiricalNotImplementedError instead of fabricating a SOL/constant."""
        from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

        db = mutable_comprehensive_perf_db
        db.backend = common.BackendName.vllm.value
        db._moe_data[common.MoEQuantMode.bfloat16]["uniform"][2][8][2048][8192][1][3] = {}

        with pytest.raises(EmpiricalNotImplementedError):
            db.query_moe(
                22,
                2048,
                8192,
                2,
                8,
                1,
                3,
                common.MoEQuantMode.bfloat16,
                "uniform",
                database_mode=common.DatabaseMode.HYBRID,
            )


class TestMLABMM:
    """Test cases for query_mla_bmm method."""

    def test_query_mla_bmm_database_mode_pre(self, comprehensive_perf_db):
        """Test SOL mode calculation for MLA BMM pre operation."""
        num_tokens = 16
        num_heads = 4
        quant_mode = common.GEMMQuantMode.bfloat16
        if_pre = True

        result = comprehensive_perf_db.query_mla_bmm(
            num_tokens, num_heads, quant_mode, if_pre, database_mode=common.DatabaseMode.SOL
        )

        # Calculate expected SOL result
        ops = 2 * num_tokens * num_heads * 128 * 512  # 2 for fma
        mem_bytes = num_heads * (num_tokens * 640 + 128 * 512) * quant_mode.value.memory
        sol_math = (
            ops / (comprehensive_perf_db.system_spec["gpu"]["bfloat16_tc_flops"] * quant_mode.value.compute) * 1000
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
            ops / (comprehensive_perf_db.system_spec["gpu"]["bfloat16_tc_flops"] * quant_mode.value.compute) * 1000
        )
        sol_mem = mem_bytes / comprehensive_perf_db.system_spec["gpu"]["mem_bw"] * 1000
        expected = max(sol_math, sol_mem)

        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_query_mla_bmm_sol_full_mode(self, comprehensive_perf_db):
        """Test SOL_FULL mode returns (sol_time, sol_math, sol_mem)."""
        sol_time, sol_math, sol_mem = comprehensive_perf_db.query_mla_bmm(
            8, 4, common.GEMMQuantMode.bfloat16, True, database_mode=common.DatabaseMode.SOL_FULL
        )

        sol_only = comprehensive_perf_db.query_mla_bmm(
            8, 4, common.GEMMQuantMode.bfloat16, True, database_mode=common.DatabaseMode.SOL
        )
        assert sol_time > 0
        assert math.isclose(sol_time, float(sol_only), rel_tol=1e-6)
        assert math.isclose(sol_time, max(sol_math, sol_mem), rel_tol=1e-6)

    def test_query_mla_bmm_non_database_mode_pre(self, comprehensive_perf_db):
        """Test SILICON mode for pre operation."""
        num_tokens = 8
        num_heads = 4
        quant_mode = common.GEMMQuantMode.bfloat16

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
            (1, 1, common.GEMMQuantMode.bfloat16, True),
            (32, 8, common.GEMMQuantMode.bfloat16, False),
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


class TestMoECrossProfileTransfer:
    """Cross-PROFILE cross-quant transfer (MoE Tier 3): when the query quant has no
    data of any profile, borrow the nearest collected quant's util curve and rescale
    by the per-quant util-LEVEL ratio e(query)/e(ref). The stub db collects only
    bfloat16 (2,1) and fp8 (1,2)."""

    def test_quant_util_level_keyed_by_profile_and_structured(self):
        from aiconfigurator.sdk.operations.moe import _moe_quant_util_level

        mq = common.MoEQuantMode
        # keyed by (memory, compute) profile, NOT by enum name: quants sharing a profile
        # share a level (int4_wo & w4a16_mxfp4 both (0.5,1); fp8 & fp8_block both (1,2))
        assert _moe_quant_util_level(mq.int4_wo) == _moe_quant_util_level(mq.w4a16_mxfp4)
        assert _moe_quant_util_level(mq.fp8) == _moe_quant_util_level(mq.fp8_block)
        # structural fact the transfer ratios rely on: more aggressive weight quant runs
        # further below its (higher) roofline -> lower achieved-util level
        assert (
            _moe_quant_util_level(mq.w4a16_mxfp4) < _moe_quant_util_level(mq.fp8) < _moe_quant_util_level(mq.bfloat16)
        )

    def test_xprofile_quants_nearest_first(self):
        from aiconfigurator.sdk.operations.moe import _xprofile_moe_quants

        table = {common.MoEQuantMode.fp8: 1, common.MoEQuantMode.bfloat16: 1}
        # query nvfp4 (0.5625, 4): fp8 (1,2) is nearer than bf16 (2,1) in profile space
        ordered = _xprofile_moe_quants(common.MoEQuantMode.nvfp4, table)
        assert ordered[0] is common.MoEQuantMode.fp8
        # same-profile quants are excluded (fp8_block shares fp8's (1,2) profile)
        assert common.MoEQuantMode.fp8 not in _xprofile_moe_quants(common.MoEQuantMode.fp8_block, table)

    def test_absent_profile_fills_via_tier3(self, comprehensive_perf_db):
        """nvfp4 (profile (0.5625,4)) has no data and no same-profile sibling -> Tier 3
        borrows fp8 and returns a finite estimate instead of raising. The util-level
        correction (util<1, e(nvfp4)<e(fp8)) makes it strictly slower than SOL."""
        kwargs = dict(
            num_tokens=16,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.nvfp4,
            workload_distribution="uniform",
        )
        sol = float(comprehensive_perf_db.query_moe(**kwargs, database_mode=common.DatabaseMode.SOL))
        empirical = float(comprehensive_perf_db.query_moe(**kwargs, database_mode=common.DatabaseMode.EMPIRICAL))
        assert empirical > 0
        assert empirical > sol  # util < 1 always degrades SOL; transfer did fire

    def test_resolve_transfer_policy(self):
        tk = common.TransferKind
        assert common.resolve_transfer_policy(None) == common.ALL_TRANSFERS
        assert common.resolve_transfer_policy("conservative") == common.TRANSFER_PRESETS["conservative"]
        assert common.resolve_transfer_policy(["xshape", "xquant"]) == frozenset({tk.XSHAPE, tk.XQUANT})
        assert common.resolve_transfer_policy(tk.XPROFILE) == frozenset({tk.XPROFILE})
        # comma-separated string (the CLI / flat-YAML form) splits into kinds
        assert common.resolve_transfer_policy("xshape,xquant") == frozenset({tk.XSHAPE, tk.XQUANT})
        assert common.resolve_transfer_policy(" xshape , xprofile ") == frozenset({tk.XSHAPE, tk.XPROFILE})
        with pytest.raises(ValueError):
            common.resolve_transfer_policy("not_a_kind")

    def test_worst_provenance_picks_least_confident(self):
        from aiconfigurator.sdk.operations import util_empirical as ue

        assert ue.worst_provenance(set()) == "silicon"  # nothing fired
        assert ue.worst_provenance({"empirical"}) == "empirical"
        # least-confident (latest in PROVENANCE_ORDER) wins over a mixed set
        assert ue.worst_provenance({"xshape", "xop", "empirical"}) == "xop"
        assert ue.worst_provenance({"xshape", "xquant"}) == "xquant"
        # capture round-trip: note inside the block, collected after
        with ue.capture_provenance() as tags:
            ue.note_provenance("xprofile")
            ue.note_provenance("xshape")
        assert tags == {"xprofile", "xshape"} and ue.worst_provenance(tags) == "xprofile"
        ue.note_provenance("xop")  # outside any capture -> no-op, no error

    def test_transfer_policy_gates_cross_profile(self, comprehensive_perf_db):
        """With XPROFILE disabled, the absent-profile quant raises again instead of
        falling through to cross-profile transfer. The policy is read at query time."""
        kwargs = dict(
            num_tokens=16,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.nvfp4,
            workload_distribution="uniform",
            database_mode=common.DatabaseMode.EMPIRICAL,
        )
        try:
            comprehensive_perf_db.set_transfer_policy("balanced")  # xshape+xquant+xversion, no xprofile
            with pytest.raises(EmpiricalNotImplementedError):
                comprehensive_perf_db.query_moe(**kwargs)
            comprehensive_perf_db.set_transfer_policy(None)  # all on -> fills again
            assert float(comprehensive_perf_db.query_moe(**kwargs)) > 0
        finally:
            comprehensive_perf_db.set_transfer_policy(None)
