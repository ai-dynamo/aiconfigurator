# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prefill_queueing_ttft_factor: Kingman (G/G/1) TTFT pre-correction."""

import pytest

from aiconfigurator.sdk.picking import (
    _AUTOSCALE_TTFT_CORRECTION_FACTOR,
    prefill_queueing_ttft_factor,
)

pytestmark = pytest.mark.unit


class TestPrefillQueueingTtftFactor:
    def test_kingman_values(self):
        # Poisson arrivals (ca^2=1), deterministic service: M/D/1
        assert prefill_queueing_ttft_factor(0.0) == 1.0
        assert prefill_queueing_ttft_factor(0.5) == pytest.approx(1.5)
        assert prefill_queueing_ttft_factor(0.9) == pytest.approx(5.5)
        # variable service time adds its cv^2 to the numerator
        assert prefill_queueing_ttft_factor(0.5, service_cv2=1.0) == pytest.approx(2.0)
        # router-regularized arrivals shrink the wait: high utilization with
        # smoothed input lands near the fleet-observed ~2x, not 5.5x
        assert prefill_queueing_ttft_factor(0.9, arrival_cv2=0.18) == pytest.approx(1.81)
        assert prefill_queueing_ttft_factor(0.9, arrival_cv2=0.25) == pytest.approx(2.125)

    def test_default_constant_unchanged_and_within_regularized_regime(self):
        # the module default stays the legacy 1.8 (no behavior change); the
        # formula reproduces it under regularized arrivals at high utilization
        assert _AUTOSCALE_TTFT_CORRECTION_FACTOR == 1.8
        lo = prefill_queueing_ttft_factor(0.9, arrival_cv2=0.15)
        hi = prefill_queueing_ttft_factor(0.9, arrival_cv2=0.25)
        assert lo <= _AUTOSCALE_TTFT_CORRECTION_FACTOR <= hi

    def test_rejects_out_of_domain(self):
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(1.0)
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(-0.1)
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(0.5, service_cv2=-1.0)
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(0.5, service_cv2=float("nan"))
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(0.5, service_cv2=float("inf"))
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(0.5, arrival_cv2=-0.1)
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(0.5, arrival_cv2=float("nan"))
