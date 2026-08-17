# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAFD-aligned AFD knobs: F-side calibration and comm-hiding tolerance.

Two knobs ported from the FastAFD-informed work, both defaulting to values
that leave existing behavior alone:

* ``f_latency_scale`` multiplies every F-side contribution so the predicted
  ``T_e`` can be calibrated against a specific FFN runtime -- FastAFD's fused
  MegaMoE kernel measured 42-44% lower decode-step latency than the
  separate-stage path the stock per-op data reflects. It is a calibration
  knob, not a physical model.
* ``comm_hiding_tolerance`` stops the strict K=3 occupancy bound from
  demoting ``mb=2`` to the blocking pipeline when the cross-pool round trip
  is negligible against compute. FastAFD measured mb=2 as sufficient to hide
  both directions on NVL72.
"""

from __future__ import annotations

from types import SimpleNamespace

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


class TestFLatencyScaleValidation:
    @pytest.mark.parametrize("scale", [0.3, 0.45, 1.0, 2.0])
    def test_positive_scales_accepted(self, scale):
        assert _cfg(f_latency_scale=scale).f_latency_scale == scale

    @pytest.mark.parametrize("scale", [0.0, -0.5])
    def test_non_positive_scale_rejected(self, scale):
        with pytest.raises(ValueError, match="f_latency_scale"):
            _cfg(f_latency_scale=scale)

    def test_default_is_no_calibration(self):
        assert _cfg().f_latency_scale == 1.0


class TestCommHidingToleranceValidation:
    @pytest.mark.parametrize("tol", [0.0, 0.1, 1.0])
    def test_non_negative_accepted(self, tol):
        assert _cfg(comm_hiding_tolerance=tol).comm_hiding_tolerance == tol

    def test_negative_rejected(self):
        with pytest.raises(ValueError, match="comm_hiding_tolerance"):
            _cfg(comm_hiding_tolerance=-0.1)

    def test_default_matches_ported_value(self):
        assert _cfg().comm_hiding_tolerance == 0.1


class TestRouterOnAttnConfig:
    def test_default_keeps_router_on_ffn(self):
        assert _cfg().router_on_attn is False

    def test_can_be_enabled(self):
        assert _cfg(router_on_attn=True).router_on_attn is True


class TestFLatencyScaleScalesOnlyTheFSide:
    """White-box check on the two places the scale is applied.

    ``_integrate_decode_phase`` scales the F compute sum and its per-op dict;
    the caller scales the F-node AG/RS contribution. A / cross-pool / combine
    terms must be untouched, and ``1.0`` must be indistinguishable from not
    setting the knob at all.
    """

    def _integrate(self, f_scale, *, f_step=0.6, a_step=0.2):
        """Drive the decode integrator with stubbed per-pool sums."""
        session = _session(num_microbatches=2, pipeline_model="serial")
        session._nextn = 0
        session._a_database = object()
        session._f_database = object()

        calls = []

        def fake_sum_latency(ops, **kwargs):
            # ops identity tells the two pools apart.
            calls.append(kwargs["database"])
            if ops == "attn":
                return a_step, {"attn_op": a_step}
            return f_step, {"ffn_op": f_step}

        session._sum_latency = fake_sum_latency
        partition_a = SimpleNamespace(attn_ops="attn")
        partition_f = SimpleNamespace(ffn_ops="ffn")

        return session._integrate_decode_phase(
            a_partition=partition_a,
            f_partition=partition_f,
            a_model=SimpleNamespace(),
            f_model=SimpleNamespace(),
            runtime_config=SimpleNamespace(isl=1000, osl=4, prefix=0),
            isl=1000,
            osl=4,
            a_batch_size=8,
            b_batch_size=8,
            num_layers=1,
            brk_t_a_per_layer=0.0,
            brk_t_f_per_layer=0.0,
            t_a2f_layer=0.05,
            t_f2a_layer=0.05,
            f_scale=f_scale,
        )

    def test_f_side_scales_and_a_side_does_not(self):
        base_a, base_f, *_rest = self._integrate(1.0)
        scaled_a, scaled_f, *_rest2 = self._integrate(0.45)

        assert scaled_a == pytest.approx(base_a), "A side must not scale"
        assert scaled_f == pytest.approx(base_f * 0.45), "F side must scale by exactly the factor"

    def test_per_op_dict_scales_with_the_f_side(self):
        *_head, base_a_ops, base_f_ops, _hidden = self._integrate(1.0)
        *_head2, scaled_a_ops, scaled_f_ops, _hidden2 = self._integrate(0.45)

        assert scaled_a_ops == base_a_ops
        assert scaled_f_ops["ffn_op"] == pytest.approx(base_f_ops["ffn_op"] * 0.45)

    def test_scale_one_is_identical_to_the_default(self):
        """``1.0`` must take the untouched code path, value-for-value."""
        explicit = self._integrate(1.0)
        session = _session(num_microbatches=2, pipeline_model="serial")
        assert session._afd_config.f_latency_scale == 1.0
        again = self._integrate(1.0)
        assert explicit[:4] == again[:4]

    @pytest.mark.parametrize("scale", [0.3, 0.45, 0.75, 2.0])
    def test_scaling_is_linear_in_the_factor(self, scale):
        _a, base_f, *_ = self._integrate(1.0)
        _a2, scaled_f, *_ = self._integrate(scale)
        assert scaled_f == pytest.approx(base_f * scale)

    def test_ag_rs_contribution_is_scaled_by_the_caller(self):
        """The F-node AG/RS term is pre-scaled and must not be double-scaled.

        ``_integrate_decode_phase`` adds ``brk_t_f_per_layer`` *after* scaling
        the compute sum, so passing an already-scaled break value must land in
        ``t_f_layer`` unmultiplied.
        """
        session = _session(num_microbatches=2, pipeline_model="serial")
        session._nextn = 0
        session._a_database = object()
        session._f_database = object()
        session._sum_latency = lambda ops, **kw: (0.0, {}) if ops == "attn" else (0.0, {})

        _a, t_f, *_ = session._integrate_decode_phase(
            a_partition=SimpleNamespace(attn_ops="attn"),
            f_partition=SimpleNamespace(ffn_ops="ffn"),
            a_model=SimpleNamespace(),
            f_model=SimpleNamespace(),
            runtime_config=SimpleNamespace(isl=1000, osl=4, prefix=0),
            isl=1000,
            osl=4,
            a_batch_size=8,
            b_batch_size=8,
            num_layers=1,
            brk_t_a_per_layer=0.0,
            brk_t_f_per_layer=0.09,  # already scaled by the caller
            t_a2f_layer=0.0,
            t_f2a_layer=0.0,
            f_scale=0.45,
        )
        assert t_f == pytest.approx(0.09), "pre-scaled AG/RS term must not be scaled again"


class TestCommHidingTolerancePipeline:
    """``_pipeline_tcycle`` is the only consumer of the tolerance."""

    # t_a=1.0, t_f=1.0 so max(t_a, t_f) == 1.0 and t_c is directly the
    # fraction under test.
    _T_A = 1.0
    _T_F = 1.0

    def _tcycle(self, *, t_c_total: float, num_microbatches: int, tolerance: float):
        session = _session(
            num_microbatches=num_microbatches,
            pipeline_model="optimistic",
            comm_hiding_tolerance=tolerance,
        )
        half = t_c_total / 2.0
        return session._pipeline_tcycle(self._T_A, self._T_F, half, half)

    def test_negligible_round_trip_keeps_k3_at_mb2(self):
        """t_c = 5% of compute, mb=2: overlapped cycle survives."""
        t_cycle, comm_hidden = self._tcycle(t_c_total=0.05, num_microbatches=2, tolerance=0.1)
        assert t_cycle == pytest.approx(max(self._T_A, self._T_F))
        assert comm_hidden is True

    def test_large_round_trip_still_degrades_at_mb2(self):
        """t_c = 50% of compute, well past the tolerance: K=2 blocking."""
        t_cycle, comm_hidden = self._tcycle(t_c_total=0.5, num_microbatches=2, tolerance=0.1)
        assert t_cycle == pytest.approx(self._T_A + 0.25)
        assert comm_hidden is False

    def test_zero_tolerance_restores_the_strict_bound(self):
        t_cycle, comm_hidden = self._tcycle(t_c_total=0.05, num_microbatches=2, tolerance=0.0)
        assert t_cycle == pytest.approx(self._T_A + 0.025)
        assert comm_hidden is False

    def test_tolerance_does_not_rescue_mb1(self):
        """The waiver requires at least two in-flight microbatches.

        A single microbatch cannot overlap anything, so no tolerance should
        let it claim the K=3 cadence.
        """
        t_cycle, comm_hidden = self._tcycle(t_c_total=0.01, num_microbatches=1, tolerance=1.0)
        assert comm_hidden is False
        assert t_cycle == pytest.approx(self._T_A + 0.005)

    def test_mb3_is_unaffected_by_the_tolerance(self):
        """mb=3 already satisfies the strict bound at small t_c."""
        for tolerance in (0.0, 0.1):
            t_cycle, comm_hidden = self._tcycle(t_c_total=0.05, num_microbatches=3, tolerance=tolerance)
            assert t_cycle == pytest.approx(max(self._T_A, self._T_F))
            assert comm_hidden is True

    def test_conservative_and_serial_models_ignore_the_tolerance(self):
        half = 0.025
        conservative = _session(
            num_microbatches=2, pipeline_model="conservative", comm_hiding_tolerance=1.0
        )._pipeline_tcycle(self._T_A, self._T_F, half, half)
        serial = _session(num_microbatches=2, pipeline_model="serial", comm_hiding_tolerance=1.0)._pipeline_tcycle(
            self._T_A, self._T_F, half, half
        )

        assert conservative == (pytest.approx(self._T_A + half), False)
        assert serial == (pytest.approx(self._T_A + self._T_F + 2 * half), False)

    def test_comm_bound_case_reports_not_hidden_even_when_waived(self):
        """Waiving the occupancy bound is not the same as hiding comm.

        With t_c above max(t_a, t_f) the network sets the cycle, so
        ``comm_hidden`` must stay False even though the K=3 cadence is kept.
        """
        session = _session(num_microbatches=2, pipeline_model="optimistic", comm_hiding_tolerance=10.0)
        t_cycle, comm_hidden = session._pipeline_tcycle(0.1, 0.1, 0.5, 0.5)
        assert t_cycle == pytest.approx(1.0)  # t_c dominates
        assert comm_hidden is False

    def test_fallback_warning_is_emitted_once_per_session(self, caplog):
        """The fallback warning must not scale with the candidate count.

        ``_pipeline_tcycle`` runs once per decode stride per candidate, and
        mb=2 + optimistic is now enumerated instead of pruned -- so an
        un-deduped warning floods the log (measured: tens of MB in minutes on
        a 7k-candidate sweep). Dedupe is per session, so a fresh session warns
        again.
        """
        import logging

        session = _session(num_microbatches=2, pipeline_model="optimistic", comm_hiding_tolerance=0.0)
        with caplog.at_level(logging.WARNING, logger="aiconfigurator"):
            for _ in range(50):
                session._pipeline_tcycle(1.0, 1.0, 0.1, 0.1)
        first = [r for r in caplog.records if "optimistic pipeline" in r.getMessage()]
        assert len(first) == 1, f"expected one warning, got {len(first)}"

        caplog.clear()
        fresh = _session(num_microbatches=2, pipeline_model="optimistic", comm_hiding_tolerance=0.0)
        with caplog.at_level(logging.WARNING, logger="aiconfigurator"):
            fresh._pipeline_tcycle(1.0, 1.0, 0.1, 0.1)
        assert len([r for r in caplog.records if "optimistic pipeline" in r.getMessage()]) == 1


class TestTaskWiring:
    """Task fields feed ``sweep_afd`` under their unprefixed names."""

    def _task(self, **overrides):
        from aiconfigurator.sdk.task_v2 import Task

        base = {
            "serving_mode": "afd",
            "model_path": "Qwen/Qwen3-32B",
            "system_name": "h200_sxm",
            "backend_name": "trtllm",
            "total_gpus": 32,
        }
        base.update(overrides)
        return Task(**base)

    def test_defaults(self):
        task = self._task()
        assert task.afd_router_on_attn is False
        assert task.afd_f_latency_scale == 1.0
        assert task.afd_comm_hiding_tolerance == 0.1

    def test_overrides_are_stored(self):
        task = self._task(
            afd_router_on_attn=True,
            afd_f_latency_scale=0.45,
            afd_comm_hiding_tolerance=0.0,
        )
        assert task.afd_router_on_attn is True
        assert task.afd_f_latency_scale == 0.45
        assert task.afd_comm_hiding_tolerance == 0.0

    def test_knobs_reach_sweep_afd_under_unprefixed_names(self, monkeypatch):
        """``sweep_afd_kwargs`` strips the ``afd_`` prefix for ``sweep_afd``.

        Asserting on the built kwargs (rather than a full run) keeps the test
        away from perf-database availability while still catching a dropped
        or misnamed passthrough.
        """
        task = self._task(
            afd_router_on_attn=True,
            afd_f_latency_scale=0.45,
            afd_comm_hiding_tolerance=0.0,
        )
        kwargs = task.sweep_afd_kwargs(database=object())

        assert kwargs["router_on_attn"] is True
        assert kwargs["f_latency_scale"] == 0.45
        assert kwargs["comm_hiding_tolerance"] == 0.0

    def test_max_af_ratio_defaults_to_uncapped(self):
        assert self._task().afd_max_af_ratio is None

    @pytest.mark.parametrize("bad", [0, -1.0])
    def test_max_af_ratio_rejects_non_positive(self, bad):
        """Rejected at construction, not at ``validate()``.

        ``_resolve_afd_search`` runs in ``__post_init__`` and calls
        ``build_afd_parallel_lists``, which validates the cap while reading
        the search config -- so the failure surfaces before ``validate()``
        is ever reached.
        """
        with pytest.raises(ValueError, match=r"max_af_ratio must be > 0 when set"):
            self._task(afd_max_af_ratio=bad)

    def test_max_af_ratio_rejected_by_validate_when_topology_is_pinned(self):
        """Pinned topologies skip enumeration, so ``validate()`` is the guard.

        With the topology pinned ``build_afd_parallel_lists`` is never called,
        which would otherwise leave the field unchecked.
        """
        task = self._task(afd_n_a_nodes=2, afd_n_f_nodes=1, afd_tp_a=8, afd_max_af_ratio=-1.0)
        with pytest.raises(ValueError, match="afd_max_af_ratio"):
            task.validate()

    def test_max_af_ratio_accepts_a_positive_cap(self):
        task = self._task(afd_max_af_ratio=4.0)
        task.validate()
        assert task.afd_max_af_ratio == 4.0
