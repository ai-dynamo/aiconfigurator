# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR-2 review follow-up guards on ``build_moe_block_ops`` and the hybrid rename loop.

The behavior-frozen hardening batch (none of these may move a number):

- ``gpus_per_node`` has no safe default: the large-EP emission must refuse to
  price cross-node all-to-all off a guessed node width (every production
  large-EP caller injects it from the system spec); the fused path never needs
  the value, so omitting it there stays legal.
- ``dispatch_quant_mode`` threads the fused MoEDispatch quant mode through the
  builder call instead of post-mutating the returned ops (the hybrid family's
  legacy quant-agnostic dispatches); the unset default forwards ``quant_mode``
  unchanged — the existing fused goldens pin that.
- the hybrid layer-type rename loop asserts the builder returned
  phase-canonical names before slicing ``len(phase)`` characters off them.
"""

from __future__ import annotations

import pytest

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common, config
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.models.blocks.moe import MoEBlockShape, build_moe_block_ops

pytestmark = pytest.mark.unit


def _shape():
    return MoEBlockShape(
        hidden_size=1024, moe_inter_size=512, topk=4, num_experts=64, num_shared_experts=0, num_moe_layers=10
    )


def _cfg(**overrides):
    kwargs = dict(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        attention_dp_size=8,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
    )
    kwargs.update(overrides)
    return config.ModelConfig(**kwargs)


def _build(cfg, **overrides):
    kwargs = dict(scale_factor=10, backend_name="sglang", inference_phase="context")
    kwargs.update(overrides)
    return build_moe_block_ops("context", _shape(), cfg, cfg.moe_quant_mode, "uniform", **kwargs)


def _dispatches(op_list):
    return [op for op in op_list if isinstance(op, ops.MoEDispatch)]


class TestGpusPerNodeGuard:
    def test_large_ep_without_gpus_per_node_raises(self):
        cfg = _cfg(moe_comm_backend={"context": "deepep_ht"})
        with pytest.raises(ValueError, match="gpus_per_node"):
            _build(cfg)

    def test_large_ep_with_explicit_none_raises_too(self):
        cfg = _cfg(moe_comm_backend={"context": "deepep_ht"})
        with pytest.raises(ValueError, match="gpus_per_node"):
            _build(cfg, gpus_per_node=None)

    def test_fused_path_never_needs_it(self):
        built = _build(_cfg())
        assert [op._name for op in built] == [
            "context_router_gemm",
            "context_moe_pre_dispatch",
            "context_moe",
            "context_moe_post_dispatch",
        ]


class TestDispatchQuantMode:
    def test_default_forwards_quant_mode(self):
        built = _build(_cfg())
        pre, post = _dispatches(built)
        assert pre._quant_mode is common.MoEQuantMode.bfloat16
        assert post._quant_mode is common.MoEQuantMode.bfloat16

    def test_explicit_none_builds_quant_agnostic_dispatches(self):
        """``None`` is the hybrid family's legacy value: dispatches carry no
        quant mode (previously restored by post-mutating ``op._quant_mode``)."""
        built = _build(_cfg(), dispatch_quant_mode=None)
        pre, post = _dispatches(built)
        assert pre._quant_mode is None
        assert post._quant_mode is None

    def test_override_reaches_dispatches_only(self):
        built = _build(_cfg(), dispatch_quant_mode=common.MoEQuantMode.fp8)
        pre, post = _dispatches(built)
        assert pre._quant_mode is common.MoEQuantMode.fp8
        assert post._quant_mode is common.MoEQuantMode.fp8
        (moe,) = [op for op in built if isinstance(op, ops.MoE)]
        assert moe._quant_mode is common.MoEQuantMode.bfloat16  # the block quant mode is untouched

    def test_hybrid_dispatches_stay_quant_agnostic_via_the_param(self, monkeypatch):
        """The hybrid family now passes ``dispatch_quant_mode=None`` through its
        builder call instead of resetting the returned ops."""
        from aiconfigurator_core.sdk.models import hybrid_moe

        calls = []
        original = hybrid_moe.build_moe_block_ops

        def probe(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(hybrid_moe, "build_moe_block_ops", probe)
        model_config = config.ModelConfig(tp_size=8, pp_size=1, moe_tp_size=1, moe_ep_size=8, attention_dp_size=1)
        model = get_model("XiaomiMiMo/MiMo-V2-Flash", model_config, "sglang")
        assert calls and all(kw["dispatch_quant_mode"] is None for kw in calls)
        dispatches = _dispatches(model.context_ops) + _dispatches(model.generation_ops)
        assert dispatches and all(op._quant_mode is None for op in dispatches)


class TestHybridRenameLoopSelfDefense:
    def test_non_phase_canonical_builder_name_asserts(self, monkeypatch):
        """A returned op name that does not start with the phase would be
        silently mangled by the ``len(phase)`` slice; the rename loop must
        assert instead."""
        from aiconfigurator_core.sdk.models import hybrid_moe

        def bogus_builder(*args, **kwargs):
            return [ops.ElementWise("bogus_marker", 1, 8, 8, 0.8)]

        monkeypatch.setattr(hybrid_moe, "build_moe_block_ops", bogus_builder)
        model_config = config.ModelConfig(tp_size=8, pp_size=1, moe_tp_size=1, moe_ep_size=8, attention_dp_size=1)
        with pytest.raises(AssertionError, match="does not start with its phase"):
            get_model("XiaomiMiMo/MiMo-V2-Flash", model_config, "sglang")
