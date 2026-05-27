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
