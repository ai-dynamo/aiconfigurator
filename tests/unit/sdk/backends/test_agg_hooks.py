# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-backend tests for overridable agg-pipeline hooks on BaseBackend."""

import pytest

from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend

pytestmark = pytest.mark.unit

_ALL_BACKENDS = [SGLANGBackend, TRTLLMBackend, VLLMBackend]
_PIPELINE_DRAIN_BACKENDS = [SGLANGBackend, TRTLLMBackend]


# ---------------------------------------------------------------------------
# _mix_step_efficiency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_cls", _ALL_BACKENDS)
def test_mix_step_efficiency_pure_prefill_is_one(backend_cls) -> None:
    # gen_tokens=0: no decode tokens in the forward pass — no correction needed.
    assert backend_cls()._mix_step_efficiency(ctx_tokens=4096, gen_tokens=0) == 1.0


@pytest.mark.parametrize("backend_cls", _ALL_BACKENDS)
def test_mix_step_efficiency_all_backends_return_one(backend_cls) -> None:
    # All backends currently return 1.0: TRT-LLM and SGLang inherit the base
    # default; VLLMBackend explicitly overrides because the power-law formula is
    # inapplicable in the max_num_partial_prefills=1 (tiny gen_frac) regime.
    assert backend_cls()._mix_step_efficiency(ctx_tokens=6144, gen_tokens=3) == 1.0


# ---------------------------------------------------------------------------
# _tpot_mix_steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_cls", _PIPELINE_DRAIN_BACKENDS)
def test_tpot_mix_steps_pipeline_drain_backends_apply_correction(backend_cls) -> None:
    # TRT-LLM and SGLang restore the empirical 3-step pipeline-drain correction.
    assert backend_cls()._tpot_mix_steps(10) == 7
    assert backend_cls()._tpot_mix_steps(3) == 1  # clamped at 1
    assert backend_cls()._tpot_mix_steps(1) == 1  # clamped at 1


def test_tpot_mix_steps_vllm_no_correction() -> None:
    # VLLMBackend inherits the base default: no pipeline-drain correction.
    assert VLLMBackend()._tpot_mix_steps(10) == 10
    assert VLLMBackend()._tpot_mix_steps(1) == 1


# ---------------------------------------------------------------------------
# _ttft_queuing_factor
# ---------------------------------------------------------------------------


def test_ttft_queuing_factor_vllm_b1_no_queuing() -> None:
    # Single request: no queue, factor must be exactly 1.0.
    assert VLLMBackend()._ttft_queuing_factor(b=1, steps_to_finish_ctx=1.0) == 1.0


def test_ttft_queuing_factor_vllm_grows_with_b() -> None:
    # Factor is strictly increasing with b (more concurrency → more queuing).
    backend = VLLMBackend()
    factors = [backend._ttft_queuing_factor(b=b, steps_to_finish_ctx=float(b)) for b in [1, 4, 16, 64, 256]]
    assert factors == sorted(factors)


def test_ttft_queuing_factor_vllm_caps_at_two() -> None:
    # Saturates at 2.0 (steady-state 2T limit) — never exceeds it.
    backend = VLLMBackend()
    assert backend._ttft_queuing_factor(b=256, steps_to_finish_ctx=256.0) == pytest.approx(2.0)
    assert backend._ttft_queuing_factor(b=1024, steps_to_finish_ctx=1024.0) == pytest.approx(2.0)


def test_ttft_queuing_factor_vllm_specific_values() -> None:
    # Spot-check the log₂(b)/8 formula at representative batch sizes.
    import math

    backend = VLLMBackend()
    for b in [2, 4, 8, 16, 32, 64]:
        expected = min(1.0 + math.log2(b) / 8.0, 2.0)
        assert backend._ttft_queuing_factor(b=b, steps_to_finish_ctx=float(b)) == pytest.approx(expected)


@pytest.mark.parametrize("backend_cls", [SGLANGBackend, TRTLLMBackend])
def test_ttft_queuing_factor_non_vllm_uses_legacy_formula(backend_cls) -> None:
    # Non-vLLM backends use the base-class legacy heuristic (unchanged).
    backend = backend_cls()
    steps = 10.0
    expected = min(2 + (steps - 3) / 2 / 10, 4)
    assert backend._ttft_queuing_factor(b=8, steps_to_finish_ctx=steps) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _throughput_cap
# ---------------------------------------------------------------------------


def test_throughput_cap_vllm_little_law_limits() -> None:
    # Cap activates when step-throughput exceeds what request latency sustains.
    backend = VLLMBackend()
    # request_latency = ttft + tpot*(osl-1) = 100 + 5*29 = 245 ms
    # little's law cap = b*(osl-1)*1000/latency = 8*29*1000/245 ≈ 946 tok/s
    cap = backend._throughput_cap(step_throughput=2000.0, ttft=100.0, tpot=5.0, b=8, osl=30)
    assert cap == pytest.approx(8 * 29 * 1000 / 245, rel=1e-4)


def test_throughput_cap_vllm_does_not_increase() -> None:
    # Cap never increases step-throughput above the raw estimate.
    backend = VLLMBackend()
    step = 500.0
    result = backend._throughput_cap(step_throughput=step, ttft=100.0, tpot=5.0, b=8, osl=30)
    assert result <= step


def test_throughput_cap_vllm_passthrough_when_below_limit() -> None:
    # When step-throughput is already below the Little's Law limit, return unchanged.
    backend = VLLMBackend()
    step = 10.0  # well below any realistic limit
    result = backend._throughput_cap(step_throughput=step, ttft=100.0, tpot=5.0, b=8, osl=30)
    assert result == pytest.approx(step)


def test_throughput_cap_non_vllm_identity() -> None:
    # Non-vLLM backends inherit the base default: identity (no cap).
    for backend_cls in [SGLANGBackend, TRTLLMBackend]:
        backend = backend_cls()
        assert backend._throughput_cap(step_throughput=999.0, ttft=100.0, tpot=5.0, b=8, osl=30) == 999.0
