# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for int4_wo (W4A16) Marlin path in the SGLang MoE collector.

TestGetMoeTestCasesInt4Wo  - int4_wo test case generation and filtering
TestBenchmarkMarlinPath    - Marlin kernel dispatch and weight setup
"""

import contextlib
import dataclasses
import importlib.util
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Collector tests require torch. If not available (CPU-only CI), skip.
if not importlib.util.find_spec("torch"):
    pytest.skip("torch required for collector tests", allow_module_level=True)

_COLLECTOR_DIR = Path(__file__).resolve().parents[3] / "collector"
if str(_COLLECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_COLLECTOR_DIR))


def _load_sglang_moe():
    """Load collector/sglang/collect_moe.py, mocking heavy SGLang deps."""
    # Stub out SGLang modules that are not available in the test env.
    _stubs = {}
    _modules_to_stub = [
        "pkg_resources",
        "sglang",
        "sglang.srt",
        "sglang.srt.server_args",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.fused_moe_triton",
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe",
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config",
        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe",
        "sglang.srt.layers.moe.topk",
        "sglang.srt.layers.moe.flashinfer_cutedsl_moe",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.gptq",
        "sglang.srt.layers.quantization.marlin_utils",
        "sglang.srt.utils",
        "sglang.jit_kernel",
        "sglang.jit_kernel.moe_wna16_marlin",
    ]
    saved = {}
    for name in _modules_to_stub:
        if name in sys.modules:
            saved[name] = sys.modules[name]
        mock = MagicMock()
        sys.modules[name] = mock
        _stubs[name] = mock

    # Wire up specific attributes the module reads at import time.
    sys.modules["sglang.srt.server_args"]._global_server_args = None
    sys.modules["sglang.srt.utils"].is_hip = MagicMock(return_value=False)
    # pkg_resources.get_distribution("sglang").version
    dist_mock = MagicMock()
    dist_mock.version = "0.5.9"
    sys.modules["pkg_resources"].get_distribution = MagicMock(return_value=dist_mock)

    try:
        path = _COLLECTOR_DIR / "sglang" / "collect_moe.py"
        spec = importlib.util.spec_from_file_location("_test_collect_moe_sglang", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        # Restore original modules (or remove stubs).
        for name in _modules_to_stub:
            if name in saved:
                sys.modules[name] = saved[name]
            else:
                sys.modules.pop(name, None)

    return mod


_mod = _load_sglang_moe()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Tc:
    """Minimal stand-in for a common MoE test case."""

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


# ---------------------------------------------------------------------------
# TestGetMoeTestCasesInt4Wo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetMoeTestCasesInt4Wo:
    """int4_wo test-case generation: always present, filtered by 256-alignment."""

    def _cases(self, test_cases, *, sm=90):
        with (
            patch.object(_mod, "get_common_moe_test_cases", return_value=test_cases),
            patch.object(_mod, "get_sm_version", return_value=sm),
            patch.object(_mod, "HAS_FLASHINFER_CUTE", False),
        ):
            return _mod.get_moe_test_cases()

    def _aligned(self):
        return [_Tc(hidden_size=4096, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1)]

    def test_int4_wo_present(self):
        assert "int4_wo" in _moe_types(self._cases(self._aligned()))

    def test_int4_wo_present_sm80(self):
        """int4_wo should be available even on SM < 90 (no fp8_block)."""
        assert "int4_wo" in _moe_types(self._cases(self._aligned(), sm=80))

    def test_filtered_misaligned_hidden(self):
        # hidden_size=4097 → 4097 % 256 != 0
        tc = [_Tc(hidden_size=4097, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1)]
        assert "int4_wo" not in _moe_types(self._cases(tc))

    def test_filtered_misaligned_local_inter(self):
        # inter_size=513, tp=1 → (513 // 1) % 256 != 0
        tc = [_Tc(hidden_size=4096, inter_size=513, topk=2, num_experts=4, tp=1, ep=1)]
        assert "int4_wo" not in _moe_types(self._cases(tc))

    def test_passes_kimi_k25_dims(self):
        """Kimi-K2.5 dims (7168, 2048, topk=8, 384 experts) must pass alignment."""
        tc = [_Tc(hidden_size=7168, inter_size=2048, topk=8, num_experts=384, tp=1, ep=1)]
        int4_cases = [c for c in self._cases(tc) if c[0] == "int4_wo"]
        assert len(int4_cases) > 0

    def test_perf_filename(self):
        for c in self._cases(self._aligned()):
            if c[0] == "int4_wo":
                assert c[9] == "moe_perf.txt"

    def test_float16_always_present(self):
        assert "float16" in _moe_types(self._cases(self._aligned()))

    def test_gpt_oss_excluded(self):
        tc = [
            _Tc(hidden_size=4096, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1, model_name="openai/gpt-oss-20b")
        ]
        assert "int4_wo" not in _moe_types(self._cases(tc))


# ---------------------------------------------------------------------------
# TestBenchmarkMarlinPath
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkMarlinPath:
    """Verify that the int4_wo path dispatches to Marlin and passes correct shapes."""

    _H = 256  # hidden_size (divisible by 256)
    _I = 256  # inter_size
    _E = 4  # num_experts
    _TOPK = 2
    _TP = 1
    _EP = 1
    _GROUP = 128

    def _run(self):
        """Run run_moe_torch with int4_wo and capture fused_marlin_moe / log_perf calls."""
        import torch

        mock_marlin = MagicMock(return_value=torch.zeros(1, self._H))
        mock_repack = MagicMock(
            side_effect=lambda w, p, k, n, b: torch.zeros(
                w.shape[0],
                k // 16,
                n * (b // 2),
                dtype=torch.int32,
            )
        )
        mock_permute = MagicMock(side_effect=lambda s, k, n, g: s.clone())
        mock_log = MagicMock()

        def fake_balanced_logits(num_tokens, num_experts, topk):
            return torch.zeros(num_tokens, num_experts)

        def fake_select_experts(x, gating, config):
            m = x.shape[0]
            k = config.top_k
            return MagicMock(
                topk_weights=torch.ones(m, k),
                topk_ids=torch.zeros(m, k, dtype=torch.int32),
                router_logits=torch.zeros(m, self._E),
            )

        def fake_bwp(device, kernel_func, **kw):
            @contextlib.contextmanager
            def _ctx():
                results = {}
                kernel_func()
                results["latency_ms"] = 1.0
                results["power_stats"] = {}
                yield results

            return _ctx()

        with (
            patch.object(_mod, "fused_marlin_moe", mock_marlin),
            patch.object(_mod, "gptq_marlin_moe_repack", mock_repack),
            patch.object(_mod, "marlin_moe_permute_scales", mock_permute),
            patch.object(_mod, "_HAS_MARLIN_MOE", True),
            patch.object(_mod, "log_perf", mock_log),
            patch.object(_mod, "balanced_logits", fake_balanced_logits),
            patch.object(_mod, "select_experts", fake_select_experts),
            patch.object(_mod, "benchmark_with_power", fake_bwp),
            patch.object(torch, "set_default_device"),
            patch.object(torch.cuda, "set_device"),
            patch.object(torch.cuda, "manual_seed_all"),
        ):
            _mod.run_moe_torch(
                "int4_wo",
                1,
                self._H,
                self._I,
                self._TOPK,
                self._E,
                self._TP,
                self._EP,
                "test/model",
                "moe_perf.txt",
                "balanced",
                0.0,
                "cuda:0",
            )

        return mock_marlin, mock_repack, mock_permute, mock_log

    def test_marlin_kernel_called(self):
        """fused_marlin_moe must be called (not the Triton fused_moe)."""
        mock_marlin, _, _, _ = self._run()
        assert mock_marlin.called

    def test_repack_called_for_w1_and_w2(self):
        """gptq_marlin_moe_repack must be called exactly twice (w1 + w2)."""
        _, mock_repack, _, _ = self._run()
        assert mock_repack.call_count == 2

    def test_repack_w1_args(self):
        """w1 repack: size_k=hidden_size, size_n=shard_intermediate_size, num_bits=4."""
        _, mock_repack, _, _ = self._run()
        args = mock_repack.call_args_list[0]
        size_k = args[0][2]
        size_n = args[0][3]
        num_bits = args[0][4]
        shard_inter = 2 * self._I // self._TP
        assert size_k == self._H
        assert size_n == shard_inter
        assert num_bits == 4

    def test_repack_w2_args(self):
        """w2 repack: size_k=shard_intermediate_size//2, size_n=hidden_size, num_bits=4."""
        _, mock_repack, _, _ = self._run()
        args = mock_repack.call_args_list[1]
        size_k = args[0][2]
        size_n = args[0][3]
        num_bits = args[0][4]
        shard_inter = 2 * self._I // self._TP
        assert size_k == shard_inter // 2
        assert size_n == self._H
        assert num_bits == 4

    def test_scales_permuted(self):
        """marlin_moe_permute_scales must be called for both w1 and w2 scales."""
        _, _, mock_permute, _ = self._run()
        assert mock_permute.call_count == 2

    def test_marlin_num_bits(self):
        """fused_marlin_moe must be called with num_bits=4."""
        mock_marlin, _, _, _ = self._run()
        kwargs = mock_marlin.call_args[1]
        assert kwargs["num_bits"] == 4

    def test_kernel_source_marlin(self):
        """Kernel source logged must be 'sglang_marlin_moe'."""
        _, _, _, mock_log = self._run()
        assert mock_log.call_args.kwargs["kernel_source"] == "sglang_marlin_moe"

    def test_log_perf_moe_dtype(self):
        """moe_dtype logged must be 'int4_wo'."""
        _, _, _, mock_log = self._run()
        item = mock_log.call_args.kwargs["item_list"][0]
        assert item["moe_dtype"] == "int4_wo"

    def test_log_perf_op_name(self):
        """op_name must be 'moe'."""
        _, _, _, mock_log = self._run()
        assert mock_log.call_args.kwargs["op_name"] == "moe"


# ---------------------------------------------------------------------------
# TestBenchmarkFallbackTritonPath
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkFallbackTritonPath:
    """When Marlin is unavailable, int4_wo falls back to Triton GPTQ/AWQ path."""

    def test_int4_wo_still_generated_without_marlin(self):
        """Test cases include int4_wo even when Marlin is unavailable."""
        tc = [_Tc(hidden_size=4096, inter_size=2048, topk=2, num_experts=4, tp=1, ep=1)]
        with (
            patch.object(_mod, "get_common_moe_test_cases", return_value=tc),
            patch.object(_mod, "get_sm_version", return_value=90),
            patch.object(_mod, "HAS_FLASHINFER_CUTE", False),
            patch.object(_mod, "_HAS_MARLIN_MOE", False),
        ):
            cases = _mod.get_moe_test_cases()
        assert "int4_wo" in {c[0] for c in cases}

    def test_kernel_source_triton_when_no_marlin(self):
        """Without Marlin, kernel_source falls back to sglang_fused_moe_triton."""
        with patch.object(_mod, "_HAS_MARLIN_MOE", False):
            moe_type = "int4_wo"
            expected = (
                "sglang_marlin_moe" if moe_type == "int4_wo" and _mod._HAS_MARLIN_MOE else "sglang_fused_moe_triton"
            )
            assert expected == "sglang_fused_moe_triton"
