# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TrtLLMWideEPMoE operation."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import PerformanceResult, TrtLLMWideEPMoE

pytestmark = pytest.mark.unit


class TestTrtLLMWideEPMoE:
    """Test cases for TrtLLMWideEPMoE class."""

    @pytest.fixture
    def seam(self, monkeypatch):
        """Record the twin op handed to the #1357 PR-5 engine seam
        (``engine._evaluate_single_op``) — the retired Python query body's
        ``query_wideep_moe_compute`` orchestration now lives in the
        ``_engine_query_plan`` twin build + the compiled engine."""
        from aiconfigurator_core.sdk import engine as engine_module

        recorded = {}

        def fake_evaluate_single_op(database, op, **eval_kwargs):
            recorded["database"] = database
            recorded["op"] = op
            recorded["eval_kwargs"] = eval_kwargs
            return PerformanceResult(10.5, energy=2.5, source="silicon")

        monkeypatch.setattr(engine_module, "_evaluate_single_op", fake_evaluate_single_op)
        return recorded

    def test_initialization_with_default_num_slots(self):
        """Test TrtLLMWideEPMoE initialization with default num_slots."""
        moe = TrtLLMWideEPMoE(
            name="test_wideep_moe",
            scale_factor=2.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.bfloat16,
            workload_distribution="power_law_1.01_eplb",
            attention_dp_size=1,
        )

        assert moe._name == "test_wideep_moe"
        assert moe._scale_factor == 2.0
        assert moe._hidden_size == 2048
        assert moe._inter_size == 8192
        assert moe._topk == 2
        assert moe._num_experts == 8
        assert moe._num_slots == 8  # Should default to num_experts
        assert moe._moe_tp_size == 2
        assert moe._moe_ep_size == 2
        assert moe._is_gated  # Default value

    def test_initialization_with_custom_num_slots(self):
        """Test TrtLLMWideEPMoE initialization with custom num_slots."""
        moe = TrtLLMWideEPMoE(
            name="test_wideep_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            num_slots=16,  # Custom num_slots > num_experts
            moe_tp_size=1,
            moe_ep_size=1,
            quant_mode=common.MoEQuantMode.nvfp4,
            workload_distribution="power_law_1.01_eplb",
            attention_dp_size=2,
            is_gated=False,
        )

        assert moe._num_slots == 16
        assert not moe._is_gated
        assert moe._attention_dp_size == 2

    def test_weight_calculation_gated(self):
        """Test weight calculation for gated MoE."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=1024,
            inter_size=4096,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.bfloat16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_gated=True,
        )

        # For gated: 3 GEMMs * hidden_size * inter_size * num_experts * memory_bytes / tp / ep
        expected_weights = (1024 * 4096 * 8 * 2 * 3) // 2 // 2
        assert moe.get_weights() == expected_weights  # scale_factor = 1.0 (engine-computed, PR-6)

    def test_weight_calculation_non_gated(self):
        """Test weight calculation for non-gated MoE."""
        moe = TrtLLMWideEPMoE(
            name="test_moe",
            scale_factor=2.0,
            hidden_size=1024,
            inter_size=4096,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.bfloat16,
            workload_distribution="uniform",
            attention_dp_size=1,
            is_gated=False,
        )

        # For non-gated: 2 GEMMs * hidden_size * inter_size * num_experts * memory_bytes / tp / ep
        expected_weights = (1024 * 4096 * 8 * 2 * 2) // 2 // 2
        assert moe.get_weights() == expected_weights * 2.0  # scale_factor = 2.0 (engine-computed, PR-6)

    def _make(self, **overrides):
        base = dict(
            name="test_moe",
            scale_factor=1.0,
            hidden_size=2048,
            inter_size=8192,
            topk=2,
            num_experts=8,
            moe_tp_size=2,
            moe_ep_size=2,
            quant_mode=common.MoEQuantMode.bfloat16,
            workload_distribution="power_law_1.01_eplb",
            attention_dp_size=1,
        )
        base.update(overrides)
        return TrtLLMWideEPMoE(**base)

    def test_query_routes_unified_expert_compute_twin(self, seam):
        """The query shim hands the engine a MoEExpertCompute twin carrying
        the legacy parameterization verbatim (num_slots defaulting to
        num_experts) and the LOCAL token count (attention-dp globalization is
        the twin's job inside the engine)."""
        from aiconfigurator_core.sdk.operations.moe_comm import MoEExpertCompute

        moe = self._make(moe_tp_size=1)
        result = moe.query(object(), x=16)

        twin = seam["op"]
        assert isinstance(twin, MoEExpertCompute)
        assert seam["eval_kwargs"]["x"] == 16
        assert twin._hidden_size == 2048
        assert twin._inter_size == 8192
        assert twin._topk == 2
        assert twin._num_experts == 8
        assert twin._num_slots == 8  # defaults to num_experts
        assert twin._moe_ep_size == 2
        assert twin._quant_mode == common.MoEQuantMode.bfloat16
        assert twin._workload_distribution == "power_law_1.01_eplb"

        # The engine's value passes through untouched (it owns scale_factor).
        assert isinstance(result, PerformanceResult)
        assert float(result) == 10.5
        assert result.energy == 2.5

    def test_query_rejects_nondefault_moe_tp(self, seam):
        """The unified twin carries no tp axis and the retired math divided
        its SOL by moe_tp_size — tp != 1 must be a loud error, never a silent
        tp=1 value (CodeRabbit review on #1552)."""
        moe = self._make(moe_tp_size=2)
        with pytest.raises(NotImplementedError, match="moe_tp_size=1 only"):
            moe.query(object(), x=16)
        assert "op" not in seam  # rejected before reaching the engine seam

    def test_query_propagates_attention_dp_to_twin(self, seam):
        """attention_dp_size rides the twin (the engine globalizes tokens);
        x itself stays rank-local at the seam."""
        moe = self._make(moe_tp_size=1, moe_ep_size=1, workload_distribution="uniform", attention_dp_size=4)
        moe.query(object(), x=16)

        assert seam["eval_kwargs"]["x"] == 16
        assert seam["op"]._attention_dp_size == 4

    def test_query_propagates_scale_factor_to_twin(self, seam):
        moe = self._make(scale_factor=3.0, moe_tp_size=1, moe_ep_size=1, workload_distribution="uniform")
        moe.query(object(), x=16)

        assert seam["op"]._scale_factor == 3.0

    def test_query_with_quant_mode_override(self, seam):
        """The legacy per-call quant_mode override rebuilds the twin."""
        moe = self._make(moe_tp_size=1, moe_ep_size=1, workload_distribution="uniform")
        moe.query(object(), x=16, quant_mode=common.MoEQuantMode.nvfp4)

        assert seam["op"]._quant_mode == common.MoEQuantMode.nvfp4

    def test_query_with_custom_num_slots(self, seam):
        """Custom EPLB num_slots passes through to the twin."""
        moe = self._make(
            num_slots=12,
            moe_tp_size=1,
            moe_ep_size=1,
            workload_distribution="power_law_1.2_eplb",
        )
        moe.query(object(), x=16)

        assert seam["op"]._num_slots == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
