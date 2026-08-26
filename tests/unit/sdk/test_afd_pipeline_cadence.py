# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which pipeline cadence ``_pipeline_tcycle`` may use at a given occupancy.

Both overlapped models need at least two in-flight microbatches -- K=3 declares
N_min=3 and K=2 declares N_min=2 -- but only the optimistic branch enforced a
threshold, and its fallback landed on conservative, a cadence ``mb=1`` cannot
sustain either. The result was a silently under-reported decode step.
"""

from __future__ import annotations

import pytest

from aiconfigurator.sdk.config import AFDConfig
from aiconfigurator.sdk.inference_session import AFDInferenceSession

pytestmark = pytest.mark.unit


def _cfg(**cfg_kwargs) -> AFDConfig:
    """Minimal valid AFDConfig; ``gpus_per_node`` has no usable default."""
    base = {"n_a_nodes": 2, "n_f_nodes": 1, "gpus_per_node": 8, "tp_a": 2, "a_batch_size": 4}
    base.update(cfg_kwargs)
    return AFDConfig(**base)


def _session(**cfg_kwargs) -> AFDInferenceSession:
    """Bare session carrying only the AFD config the pipeline model reads."""
    session = AFDInferenceSession.__new__(AFDInferenceSession)
    session._afd_config = _cfg(**cfg_kwargs)
    return session


class TestSingleMicrobatchCannotOverlap:
    """``num_microbatches < 2`` must use the serial cadence.

    Both overlapped models need at least two in-flight microbatches: K=3
    declares N_min=3 and K=2 declares N_min=2. Only the optimistic branch ever
    enforced a threshold, and its fallback landed on conservative -- so mb=1
    was handed ``max(t_a + t_a2f, t_f + t_f2a)``, a cadence it cannot sustain,
    silently under-reporting decode-step latency.

    With one microbatch in flight layer i+1's A input *is* layer i's F output,
    so the pools strictly alternate. There is no intra-layer slack either:
    every A-side op sits before the dispatch or after the combine.
    """

    _T_A = 1.0
    _T_F = 3.0  # asymmetric, so serial and conservative are far apart
    _HALF_C = 0.25

    def _tcycle(self, *, num_microbatches, pipeline_model):
        session = _session(
            num_microbatches=num_microbatches,
            pipeline_model=pipeline_model,
        )
        return session._pipeline_tcycle(self._T_A, self._T_F, self._HALF_C, self._HALF_C)

    @property
    def _serial(self):
        return self._T_A + self._T_F + 2 * self._HALF_C

    @property
    def _conservative(self):
        return max(self._T_A + self._HALF_C, self._T_F + self._HALF_C)

    @pytest.mark.parametrize("pipeline_model", ["optimistic", "conservative", "serial"])
    def test_mb1_is_serial_for_every_requested_model(self, pipeline_model):
        """The rule overrides the requested model, not just optimistic."""
        t_cycle, comm_hidden = self._tcycle(num_microbatches=1, pipeline_model=pipeline_model)
        assert t_cycle == pytest.approx(self._serial)
        assert comm_hidden is False

    def test_mb1_no_longer_reports_the_conservative_cadence(self):
        """Regression guard: the two cadences must stay distinguishable here."""
        assert self._serial != pytest.approx(self._conservative)
        t_cycle, _ = self._tcycle(num_microbatches=1, pipeline_model="optimistic")
        assert t_cycle != pytest.approx(self._conservative)

    def test_mb2_still_reaches_an_overlapped_cadence(self):
        """mb=2 is unaffected -- only mb<2 changes."""
        t_cycle, _ = self._tcycle(num_microbatches=2, pipeline_model="conservative")
        assert t_cycle == pytest.approx(self._conservative)
        assert t_cycle < self._serial

    def test_zero_or_none_microbatches_are_normalized_then_serialized(self):
        """``num_microbatches`` is floored at 1 before the check.

        ``AFDConfig`` rejects values below 1, so this only exercises the
        session's own ``or 1`` normalization -- but it must not let a falsy
        value slip into an overlapped cadence.
        """
        session = _session(num_microbatches=2, pipeline_model="optimistic")
        session._afd_config.num_microbatches = None
        t_cycle, comm_hidden = session._pipeline_tcycle(self._T_A, self._T_F, self._HALF_C, self._HALF_C)
        assert t_cycle == pytest.approx(self._serial)
        assert comm_hidden is False

    def test_mb1_serial_matches_an_explicit_serial_request(self):
        implicit, _ = self._tcycle(num_microbatches=1, pipeline_model="optimistic")
        explicit, _ = self._tcycle(num_microbatches=4, pipeline_model="serial")
        assert implicit == pytest.approx(explicit)

    def test_mb1_does_not_emit_the_fallback_warning(self, caplog):
        """mb=1 never reaches the optimistic branch, so no warning fires.

        The warning describes a demotion to conservative; at mb=1 that is not
        what happens, so emitting it would be misleading.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger="aiconfigurator"):
            self._tcycle(num_microbatches=1, pipeline_model="optimistic")
        assert not [r for r in caplog.records if "optimistic pipeline" in r.getMessage()]
