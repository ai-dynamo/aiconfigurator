# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""prefill_queueing_ttft_factor: derived M/G/1 TTFT pre-correction."""

import pytest

from aiconfigurator.sdk.picking import (
    _AUTOSCALE_TTFT_CORRECTION_FACTOR,
    _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
    prefill_queueing_ttft_factor,
)

pytestmark = pytest.mark.unit


class TestPrefillQueueingTtftFactor:
    def test_pollaczek_khinchine_values(self):
        assert prefill_queueing_ttft_factor(0.0) == 1.0
        assert prefill_queueing_ttft_factor(0.5) == pytest.approx(1.5)
        assert prefill_queueing_ttft_factor(0.9) == pytest.approx(5.5)
        # variable service time adds the (1 + cv^2)/2 PK term
        assert prefill_queueing_ttft_factor(0.5, service_cv2=1.0) == pytest.approx(2.0)

    def test_default_constant_is_derived_at_design_utilization(self):
        assert (
            pytest.approx(prefill_queueing_ttft_factor(_RATE_MATCHING_PREFILL_DEGRADATION_FACTOR))
            == _AUTOSCALE_TTFT_CORRECTION_FACTOR
        )

    def test_rejects_out_of_domain(self):
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(1.0)
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(-0.1)
        with pytest.raises(ValueError):
            prefill_queueing_ttft_factor(0.5, service_cv2=-1.0)
