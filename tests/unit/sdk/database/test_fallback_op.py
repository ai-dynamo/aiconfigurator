# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FallbackOp and MLAModule operations.

The FallbackOp latency composition (try primary, fall back on typed
coverage miss, propagate programming errors, HYBRID's silicon-copy primary)
retired to the compiled engine with #1357 PR-5: ``FallbackOp.query`` is now a
deprecation shim that converts the whole composite (children included) and
evaluates it in Rust, so there is no Python seam where the per-child queries
happen. That behaviour is anchored by the frozen parity goldens and
tests/cross_package/test_query_shim_baseline.py. What stays Python-owned —
weight accounting and the query-shim planning (phase routing, beam-width
validation) — is tested here.
"""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.operations import FallbackOp, MLAModule, PerformanceResult

pytestmark = pytest.mark.unit


def _make_mock_op(latency: float, energy: float, weights: float = 0.0):
    """Create a mock operation that returns the given latency/energy/weights."""
    op = MagicMock()
    op._name = "mock_op"
    op.query.return_value = PerformanceResult(latency, energy=energy)
    op.get_weights.return_value = weights
    return op


class TestFallbackOp:
    """Test cases for FallbackOp class."""

    def test_get_weights_from_primary(self):
        """get_weights uses primary weights when available.

        Weights route through the engine (PR-6), so the children must be
        real spec-expressible ops: bf16 GEMM weighs n*k*2 bytes."""
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.operations import GEMM

        primary = GEMM("p", 1.0, 50, 5, common.GEMMQuantMode.bfloat16)  # 500 B
        fallback = GEMM("f", 1.0, 30, 5, common.GEMMQuantMode.bfloat16)  # 300 B

        op = FallbackOp("test", primary=primary, fallback=[fallback])
        assert op.get_weights() == 500.0

    def test_get_weights_from_fallback(self):
        """get_weights sums fallback weights when primary has none."""
        from aiconfigurator.sdk import common
        from aiconfigurator.sdk.operations import GEMM

        primary = GEMM("p", 1.0, 0, 5, common.GEMMQuantMode.bfloat16)  # 0 B
        fallback_1 = GEMM("f1", 1.0, 10, 5, common.GEMMQuantMode.bfloat16)  # 100 B
        fallback_2 = GEMM("f2", 1.0, 20, 5, common.GEMMQuantMode.bfloat16)  # 200 B

        op = FallbackOp("test", primary=primary, fallback=[fallback_1, fallback_2])
        assert op.get_weights() == 300.0


def _mla_module(is_context: bool, scale_factor: float = 1.0) -> MLAModule:
    from aiconfigurator.sdk import common

    return MLAModule(
        "test_ctx" if is_context else "test_gen",
        scale_factor,
        is_context,
        16,
        common.KVCacheQuantMode.fp8,
        common.FMHAQuantMode.bfloat16,
        common.GEMMQuantMode.fp8_block,
    )


class TestMLAModule:
    """MLAModule query-shim planning (restubbed at the #1357 PR-5 seam,
    ``engine._evaluate_single_op``): the instance's phase must drive the
    engine evaluation phase, and the legacy query kwargs must propagate."""

    @pytest.fixture
    def seam(self, monkeypatch):
        from aiconfigurator_core.sdk import engine as engine_module

        recorded = {}

        def fake_evaluate_single_op(database, op, **eval_kwargs):
            recorded["op"] = op
            recorded["eval_kwargs"] = eval_kwargs
            return PerformanceResult(10.0, energy=100.0, source="silicon")

        monkeypatch.setattr(engine_module, "_evaluate_single_op", fake_evaluate_single_op)
        return recorded

    def test_context_instance_evaluates_context_phase(self, seam):
        op = _mla_module(is_context=True)
        result = op.query(object(), batch_size=4, s=4000, prefix=0)

        assert seam["op"] is op
        assert seam["eval_kwargs"]["is_context"] is True
        assert seam["eval_kwargs"]["batch_size"] == 4
        assert seam["eval_kwargs"]["s"] == 4000
        assert seam["eval_kwargs"]["prefix"] == 0
        assert float(result) == 10.0

    def test_generation_instance_evaluates_generation_phase(self, seam):
        op = _mla_module(is_context=False)
        result = op.query(object(), batch_size=4, s=4000, beam_width=1)

        assert seam["op"] is op
        assert seam["eval_kwargs"]["is_context"] is False
        assert seam["eval_kwargs"]["batch_size"] == 4
        assert seam["eval_kwargs"]["s"] == 4000
        assert float(result) == 10.0

    def test_generation_rejects_beam_width_not_1(self):
        """Generation MLAModule raises ValueError for beam_width != 1."""
        mock_db = MagicMock()
        op = _mla_module(is_context=False)
        with pytest.raises(ValueError, match="beam_width=1"):
            op.query(mock_db, batch_size=4, s=4000, beam_width=2)
