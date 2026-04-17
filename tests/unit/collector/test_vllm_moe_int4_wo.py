# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for int4_wo (W4A16) support in the vLLM MoE collectors.

Requires a real torch+vllm environment (run on a GPU cluster).

  TestV1GetMoeTestCasesInt4Wo  - collect_moe_v1: int4_wo gated by _int4_wo_available
  TestV2GetMoeTestCasesInt4Wo  - collect_moe_v2: int4_wo always present
  TestV2RunMoeTorchInt4Wo      - collect_moe_v2: weight shapes, quant config, log_perf
"""

import contextlib
import dataclasses
import importlib.util
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

_COLLECTOR_DIR = Path(__file__).resolve().parents[3] / "collector"
if str(_COLLECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_COLLECTOR_DIR))


# ---------------------------------------------------------------------------
# Load collector modules directly from file so the "vllm" subpackage name
# doesn't collide with the installed vllm package.
# ---------------------------------------------------------------------------


def _load(filename: str):
    path = _COLLECTOR_DIR / "vllm" / filename
    spec = importlib.util.spec_from_file_location(f"_test_{Path(filename).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_v1 = _load("collect_moe_v1.py")
_v2 = _load("collect_moe_v2.py")


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Tc:
    """Minimal stand-in for MoeCommonTestCase."""

    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    tp: int
    ep: int
    model_name: str = "test/model"
    num_tokens_list: list = dataclasses.field(default_factory=lambda: [1, 4, 16])
    token_expert_distribution: str = "balanced"
    power_law_alpha: Optional[float] = 0.0


def _moe_types(cases):
    return {c[0] for c in cases}


@contextlib.contextmanager
def _fake_benchmark(kernel_func_kwarg="kernel_func"):
    """Context-manager replacement for benchmark_with_power.

    Calls kernel_func once so callers capture real tensor shapes, then yields
    a fake results dict.
    """

    def _bwp(device, kernel_func, **kw):
        @contextlib.contextmanager
        def _ctx():
            results = {}
            kernel_func()
            results["latency_ms"] = 1.0
            results["power_stats"] = {}
            yield results

        return _ctx()

    yield _bwp


# ---------------------------------------------------------------------------
# TestV1GetMoeTestCasesInt4Wo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestV1GetMoeTestCasesInt4Wo:
    """collect_moe_v1: int4_wo gated by _int4_wo_available; filtered by 128-alignment."""

    def _cases(self, test_cases, *, int4_wo_available=True, sm=70):
        with (
            patch.object(_v1, "get_common_moe_test_cases", return_value=test_cases),
            patch.object(_v1, "get_sm_version", return_value=sm),
            patch.object(_v1, "_int4_wo_available", int4_wo_available),
            patch.object(_v1, "_nvfp4_available", False),
            patch.object(_v1, "_mxfp4_available", False),
        ):
            return _v1.get_moe_test_cases()

    def _aligned(self):
        return [_Tc(hidden_size=4096, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1)]

    def test_included_when_available(self):
        assert "int4_wo" in _moe_types(self._cases(self._aligned(), int4_wo_available=True))

    def test_excluded_when_not_available(self):
        assert "int4_wo" not in _moe_types(self._cases(self._aligned(), int4_wo_available=False))

    def test_filtered_misaligned_hidden(self):
        # 4097 % 128 != 0
        tc = [_Tc(hidden_size=4097, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1)]
        assert "int4_wo" not in _moe_types(self._cases(tc))

    def test_filtered_misaligned_local_inter(self):
        # inter_size=513, tp=1 → local_inter=513, 513 % 128 != 0
        tc = [_Tc(hidden_size=4096, inter_size=513, topk=2, num_experts=4, tp=1, ep=1)]
        assert "int4_wo" not in _moe_types(self._cases(tc))

    def test_passes_aligned_dims(self):
        cases = [c for c in self._cases(self._aligned()) if c[0] == "int4_wo"]
        assert len(cases) > 0

    def test_perf_filename(self):
        for c in self._cases(self._aligned()):
            if c[0] == "int4_wo":
                assert c[9] == "moe_perf.txt"

    def test_float16_always_present(self):
        assert "float16" in _moe_types(self._cases(self._aligned(), int4_wo_available=False))


# ---------------------------------------------------------------------------
# TestV2GetMoeTestCasesInt4Wo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestV2GetMoeTestCasesInt4Wo:
    """collect_moe_v2: int4_wo unconditional (vllm>=0.17.0 always has the kernel)."""

    def _cases(self, test_cases, *, sm=70):
        with (
            patch.object(_v2, "get_common_moe_test_cases", return_value=test_cases),
            patch.object(_v2, "get_sm_version", return_value=sm),
            patch.object(_v2, "_nvfp4_available", False),
            patch.object(_v2, "_mxfp4_available", False),
        ):
            return _v2.get_moe_test_cases()

    def _aligned(self):
        return [_Tc(hidden_size=4096, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1)]

    def test_int4_wo_always_present(self):
        assert "int4_wo" in _moe_types(self._cases(self._aligned()))

    def test_filtered_misaligned_hidden(self):
        tc = [_Tc(hidden_size=4097, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1)]
        assert "int4_wo" not in _moe_types(self._cases(tc))

    def test_filtered_misaligned_local_inter(self):
        tc = [_Tc(hidden_size=4096, inter_size=513, topk=2, num_experts=4, tp=1, ep=1)]
        assert "int4_wo" not in _moe_types(self._cases(tc))

    def test_filtered_misaligned_local_inter_with_tp(self):
        # inter_size=1024, tp=3 → local_inter ≈ 341 (not divisible by 128)
        tc = [_Tc(hidden_size=4096, inter_size=1024, topk=2, num_experts=4, tp=3, ep=1)]
        assert "int4_wo" not in _moe_types(self._cases(tc))

    def test_passes_aligned_local_inter_with_tp(self):
        # inter_size=512, tp=4, ep=1 → local_inter=128, 128%128==0 → passes.
        # ep must be 1: v2 skips cases where tp > 1 AND ep > 1.
        tc = [_Tc(hidden_size=4096, inter_size=512, topk=2, num_experts=4, tp=4, ep=1)]
        int4_cases = [c for c in self._cases(tc) if c[0] == "int4_wo"]
        assert len(int4_cases) > 0

    def test_perf_filename(self):
        for c in self._cases(self._aligned()):
            if c[0] == "int4_wo":
                assert c[9] == "moe_perf.txt"

    def test_gpt_oss_excluded(self):
        """GPT-OSS models are mxfp4-only; int4_wo must not be collected for them."""
        tc = [
            _Tc(hidden_size=4096, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1, model_name="openai/gpt-oss-20b")
        ]
        assert "int4_wo" not in _moe_types(self._cases(tc))


# ---------------------------------------------------------------------------
# TestV2RunMoeTorchInt4Wo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestV2RunMoeTorchInt4Wo:
    """collect_moe_v2.run_moe_torch with moe_type='int4_wo'.

    benchmark_with_power is replaced by a stub that runs kernel_func once so
    the captured call args reflect real tensor shapes.  All vllm compute ops
    (fused_experts, int4_w4a16_moe_quant_config) are mocked to avoid GPU work.
    """

    # 128-aligned dimensions, small enough for fast CPU execution.
    _H = 256  # hidden_size
    _I = 256  # inter_size  (local_inter = I // TP = 256)
    _E = 4  # num_experts
    _TOPK = 2
    _TP = 1
    _EP = 1
    _GROUP = 128  # Marlin group_size

    def _run(self):
        """Run run_moe_torch with int4_wo and return the capturing mocks."""
        mock_quant = MagicMock(return_value=MagicMock(name="quant_cfg"))
        mock_fe = MagicMock()
        mock_log = MagicMock()

        def fake_balanced_logits(num_tokens, num_experts, topk):
            return torch.zeros(num_tokens, num_experts)

        def fake_bwp(device, kernel_func, **kw):
            @contextlib.contextmanager
            def _ctx():
                results = {}
                kernel_func()
                results["latency_ms"] = 1.0
                results["power_stats"] = {}
                yield results

            return _ctx()

        import torch

        with (
            patch.object(_v2, "int4_w4a16_moe_quant_config", mock_quant),
            patch.object(_v2, "fused_experts", mock_fe),
            patch.object(_v2, "log_perf", mock_log),
            patch.object(_v2, "balanced_logits", fake_balanced_logits),
            patch.object(_v2, "benchmark_with_power", fake_bwp),
            # Prevent set_default_device("cuda:0") from polluting other tests.
            patch.object(torch, "set_default_device"),
        ):
            _v2.run_moe_torch(
                "int4_wo",
                [1],  # single token count
                self._H,
                self._I,
                self._TOPK,
                self._E,
                self._TP,
                self._EP,
                "test/model",
                "moe_perf.txt",
                "balanced",  # avoids power_law routing complexity
                0.0,
                "cuda:0",
            )

        return mock_quant, mock_fe, mock_log

    def test_w1_dtype_and_shape(self):
        """w1 must be uint8-packed: (E, 2*inter, hidden//2)."""
        import torch

        _, mock_fe, _ = self._run()
        w1 = mock_fe.call_args[0][1]
        assert w1.dtype == torch.uint8
        assert w1.shape == (self._E, 2 * self._I, self._H // 2)

    def test_w2_dtype_and_shape(self):
        """w2 must be uint8-packed: (E, hidden, local_inter//2)."""
        import torch

        _, mock_fe, _ = self._run()
        w2 = mock_fe.call_args[0][2]
        assert w2.dtype == torch.uint8
        assert w2.shape == (self._E, self._H, self._I // 2)

    def test_w1_scale_shape(self):
        """w1_scale to int4_w4a16_moe_quant_config: (E, 2*inter, hidden//group_size)."""
        import torch

        mock_quant, _, _ = self._run()
        w1_scale = mock_quant.call_args.kwargs["w1_scale"]
        assert w1_scale.dtype == torch.float16
        assert w1_scale.shape == (self._E, 2 * self._I, self._H // self._GROUP)

    def test_w2_scale_shape(self):
        """w2_scale to int4_w4a16_moe_quant_config: (E, hidden, local_inter//group_size)."""
        import torch

        mock_quant, _, _ = self._run()
        w2_scale = mock_quant.call_args.kwargs["w2_scale"]
        assert w2_scale.dtype == torch.float16
        assert w2_scale.shape == (self._E, self._H, self._I // self._GROUP)

    def test_zero_points_are_none(self):
        """Symmetric quantization: w1_zp and w2_zp must be None."""
        mock_quant, _, _ = self._run()
        assert mock_quant.call_args.kwargs["w1_zp"] is None
        assert mock_quant.call_args.kwargs["w2_zp"] is None

    def test_block_shape(self):
        """block_shape=[0, group_size] encodes per-group-K quantization."""
        mock_quant, _, _ = self._run()
        assert mock_quant.call_args.kwargs["block_shape"] == [0, self._GROUP]

    def test_fused_experts_receives_quant_config(self):
        """fused_experts must receive the config object returned by int4_w4a16_moe_quant_config."""
        mock_quant, mock_fe, _ = self._run()
        assert mock_fe.call_args.kwargs["quant_config"] is mock_quant.return_value

    def test_log_perf_kernel_source(self):
        """Kernel source in perf database must be 'vllm_marlin_int4_moe'."""
        _, _, mock_log = self._run()
        assert mock_log.call_args.kwargs["kernel_source"] == "vllm_marlin_int4_moe"

    def test_log_perf_moe_dtype(self):
        """moe_dtype logged must be 'int4_wo'."""
        _, _, mock_log = self._run()
        item = mock_log.call_args.kwargs["item_list"][0]
        assert item["moe_dtype"] == "int4_wo"
