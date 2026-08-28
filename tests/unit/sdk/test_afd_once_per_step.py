# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Once-per-step ops must not be amortized into the AFD per-layer cadence.

``embedding`` and ``logits_gemm`` are built with ``scale_factor=1`` -- they run
once per step, not once per layer. Folding them into the per-layer average both
raises the basis ``_pipeline_tcycle`` maxes over (which can flip which pool
looks like the bottleneck) and charges a one-shot cost inside a loop that runs
``num_layers`` times.

The classification is deliberately duck-typed with a bare ``except``: the
composite families (``Fallback`` / ``Overlap``) carry no ``scale_factor`` and
the pyo3 getter *raises* rather than returning ``None``. That branch is the
whole reason a ``getattr`` default is not enough, so it is tested explicitly.
"""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import config as cfgmod
from aiconfigurator.sdk.afd_partition import build_afd_ops_partition
from aiconfigurator.sdk.inference_session import (
    AFDInferenceSession,
    _is_once_per_step_op,
    _split_once_per_step,
)
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.unit


class _RaisingScaleFactor:
    """Stands in for the ``Fallback`` / ``Overlap`` composites.

    Their pyo3 ``_scale_factor`` getter raises ``TypeError: op family carries no
    scale_factor``; a plain ``getattr(op, "_scale_factor", None)`` would let
    that escape to the caller instead of falling back to per-layer.
    """

    _name = "composite_body"

    @property
    def _scale_factor(self):
        raise TypeError("op family carries no scale_factor")


class _Scaled:
    def __init__(self, name, scale):
        self._name = name
        self._scale_factor = scale


class TestClassification:
    def test_scale_factor_one_is_once_per_step(self):
        assert _is_once_per_step_op(_Scaled("embedding", 1.0), 61) is True

    def test_scale_factor_num_layers_is_per_layer(self):
        assert _is_once_per_step_op(_Scaled("add_norm", 61.0), 61) is False

    def test_composite_that_raises_reads_as_per_layer(self):
        """The branch a ``getattr`` default cannot cover."""
        assert _is_once_per_step_op(_RaisingScaleFactor(), 61) is False

    def test_op_without_the_attribute_reads_as_per_layer(self):
        assert _is_once_per_step_op(object(), 61) is False

    @pytest.mark.parametrize("num_layers", [0, 1])
    def test_single_layer_model_has_nothing_to_amortize(self, num_layers):
        """With one layer the distinction is moot and the arithmetic collapses
        to its pre-split form, so nothing may be pulled out."""
        assert _is_once_per_step_op(_Scaled("embedding", 1.0), num_layers) is False

    def test_split_partitions_without_dropping_ops(self):
        ops = [
            _Scaled("embedding", 1.0),
            _Scaled("add_norm", 61.0),
            _RaisingScaleFactor(),
            _Scaled("logits", 1.0),
        ]
        per_layer, once = _split_once_per_step(ops, 61)
        assert [o._name for o in once] == ["embedding", "logits"]
        assert [o._name for o in per_layer] == ["add_norm", "composite_body"]
        assert len(per_layer) + len(once) == len(ops)


class TestRealModelClassification:
    """Against a real model graph, not a hand-built list."""

    @pytest.mark.parametrize(
        "model_path,moe_ep",
        [("deepseek-ai/DeepSeek-V3", 8), ("Qwen/Qwen3-32B", 1)],
    )
    def test_embedding_and_logits_are_the_once_per_step_ops(self, model_path, moe_ep):
        mc = cfgmod.ModelConfig(tp_size=8, moe_tp_size=1 if moe_ep > 1 else 8, moe_ep_size=moe_ep)
        model = get_model(model_path, mc, "sglang")
        num_layers = int(getattr(model, "_num_layers", 1))
        assert num_layers > 1, "fixture model must be multi-layer for this to mean anything"

        partition = build_afd_ops_partition(model, phase="generation")
        _, once = _split_once_per_step(partition.attn_ops, num_layers)
        names = sorted(getattr(op, "_name", "") for op in once)
        assert names == ["generation_embedding", "generation_logits_gemm"]

    def test_ffn_side_has_no_once_per_step_op(self):
        """The F pool runs only per-layer bodies, so the split must be a no-op
        there -- a false positive would silently move MoE cost out of the
        pipeline."""
        mc = cfgmod.ModelConfig(tp_size=8, moe_tp_size=1, moe_ep_size=8)
        model = get_model("deepseek-ai/DeepSeek-V3", mc, "sglang")
        num_layers = int(getattr(model, "_num_layers", 1))
        partition = build_afd_ops_partition(model, phase="generation")
        per_layer, once = _split_once_per_step(partition.ffn_ops, num_layers)
        assert once == []
        assert len(per_layer) == len(partition.ffn_ops)


class TestOncePerStepStaysOutOfTheCycle:
    """``t_once`` must land on the step, never inside the per-layer max."""

    @staticmethod
    def _session():
        afd = cfgmod.AFDConfig(
            n_a_nodes=1,
            n_f_nodes=1,
            gpus_per_node=8,
            tp_a=2,
            f_moe_ep_size=8,
            a_batch_size=64,
            num_microbatches=2,
            pipeline_model="conservative",
        )
        return AFDInferenceSession.__new__(AFDInferenceSession), afd

    def test_t_once_is_added_to_the_step_not_the_cycle(self):
        session, afd = self._session()
        session._afd_config = afd
        session._warned_optimistic_fallback = False

        base_step, base_cycle, _ = session._pipeline_global_step_latency(1.0, 2.0, 0.1, 0.1, num_layers=8)
        with_once, with_cycle, _ = session._pipeline_global_step_latency(1.0, 2.0, 0.1, 0.1, num_layers=8, t_once=5.0)

        # The cadence the pipeline maxes over is untouched...
        assert with_cycle == pytest.approx(base_cycle)
        # ...and the one-shot cost is charged exactly once.
        assert with_once == pytest.approx(base_step + 5.0)

    def test_a_huge_once_cost_cannot_change_the_cycle(self):
        """The regression this guards: a large once-per-step cost used to raise
        the per-layer basis and could flip the bottleneck verdict."""
        session, afd = self._session()
        session._afd_config = afd
        session._warned_optimistic_fallback = False

        _, cycle_small, _ = session._pipeline_global_step_latency(1.0, 2.0, 0.1, 0.1, num_layers=61, t_once=0.0)
        _, cycle_huge, _ = session._pipeline_global_step_latency(1.0, 2.0, 0.1, 0.1, num_layers=61, t_once=1000.0)
        assert cycle_small == pytest.approx(cycle_huge)

    def test_zero_once_cost_is_bit_identical_to_omitting_it(self):
        session, afd = self._session()
        session._afd_config = afd
        session._warned_optimistic_fallback = False

        omitted = session._pipeline_global_step_latency(1.0, 2.0, 0.1, 0.1, num_layers=8)
        explicit_zero = session._pipeline_global_step_latency(1.0, 2.0, 0.1, 0.1, num_layers=8, t_once=0.0)
        assert omitted == explicit_zero


class TestResultSchemaSurfacesTheSplit:
    """The split has to be visible in a result row, not only inside the phase
    dict, or a sweep CSV cannot tell a re-priced row from an old one."""

    def test_key_is_in_the_single_source_of_truth(self):
        """``_PHASE_SCALAR_KEYS`` feeds the un-prefixed picker, the prefixed
        writer, and the CLI/CSV consumers -- registering the key there is what
        keeps the three from drifting apart."""
        assert "t_once_per_step" in AFDInferenceSession._PHASE_SCALAR_KEYS

    def test_phase_scalars_carries_it_through(self):
        metrics = dict.fromkeys(AFDInferenceSession._PHASE_SCALAR_KEYS, 1.0)
        metrics["comm_hidden"] = True
        metrics["t_once_per_step"] = 0.25
        out = AFDInferenceSession._phase_scalars(metrics)
        assert out["t_once_per_step"] == pytest.approx(0.25)

    def test_absent_phase_reports_nan_not_zero(self):
        """A phase that did not run must not look like a phase whose
        once-per-step cost happened to be zero."""
        out = AFDInferenceSession._phase_scalars(None)
        assert out["t_once_per_step"] != out["t_once_per_step"]  # NaN
