# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``build_moe_block_ops`` and the ``@register_moe_block`` registry.

Fused-graph equivalence pins the builder's default (fused/small-EP) emission
against what the legacy model classes hand-wire today:

- ``MOEModel.__init__`` (models/moe.py) is the oracle for the router GEMM +
  MoEDispatch/MoE/MoEDispatch span, sliced from a real Qwen3-235B model's
  ``context_ops``/``generation_ops`` by op-name pattern.
- ``DeepSeekModel.__init__`` (models/deepseek.py) is the oracle for the
  shared-expert gate_up/act/ffn2 triplet (context list, and generation via
  ``OverlapOp._group_b``).

``scale_factor`` is deliberately caller-supplied: legacy classes scale MoE ops
by their OWN layer count (DeepSeek uses all 61 layers, not the 58 MoE-true
``shape.num_moe_layers``), so the tests pass ``model._num_layers`` (x mtp for
generation) and pin that the builder does not re-derive it from the shape.
"""

import re

import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config, models
from aiconfigurator.sdk.models.blocks import MoEBlockShape, build_moe_block_ops, register_moe_block
from aiconfigurator.sdk.models.blocks import moe as moe_blocks
from aiconfigurator.sdk.models.helpers import _get_model_info

pytestmark = pytest.mark.unit

QWEN3 = "Qwen/Qwen3-235B-A22B"
DSR1 = "deepseek-ai/DeepSeek-R1"

# Legacy op names that belong to the routed-MoE span (router + dispatch + moe).
_MOE_BLOCK_NAME_RE = re.compile(r"router_gemm|moe")


def _routed_slice(op_list):
    """The router/dispatch/MoE span of a legacy op list (excludes shared experts)."""
    return [op for op in op_list if _MOE_BLOCK_NAME_RE.search(op._name)]


def _shared_slice(op_list):
    return [op for op in op_list if "_shared_" in op._name]


def _assert_ops_equivalent(built, legacy):
    """Same op classes in order, same names, same constructor-derived state.

    The engine wire form (``_spec_json``) carries every constructor-derived
    field (hidden/topk/num_experts/moe_tp/moe_ep/quant/distribution/adp/
    attn_cp_size/seq_split/...), so any kwarg the legacy site passed and the
    builder dropped (or vice versa) fails here. Classes compare by NAME:
    composite getters re-wrap children as the Rust base classes.
    """
    assert [op._name for op in built] == [op._name for op in legacy]
    for got, want in zip(built, legacy, strict=True):
        assert type(got).__name__ == type(want).__name__, got._name
        assert got._spec_json() == want._spec_json(), got._name


def _shape_for(model_path):
    return MoEBlockShape.from_model_info(_get_model_info(model_path))


class TestFusedGraphEquivalence:
    """Builder output ≡ MOEModel's hand-wired MoE block for Qwen3-235B."""

    def _build_qwen3(self, backend_name, **cfg_kwargs):
        model_config = config.ModelConfig(**cfg_kwargs)
        model = models.get_model(QWEN3, model_config, backend_name)
        return model, model_config

    def test_context_matches_legacy_moe_model(self):
        model, cfg = self._build_qwen3("trtllm", tp_size=2, moe_tp_size=1, moe_ep_size=2, attention_dp_size=1)
        built = build_moe_block_ops(
            "context",
            _shape_for(QWEN3),
            cfg,
            cfg.moe_quant_mode,
            "power_law_1.2",  # model-owned alpha string (MOEModel's _power_law_alpha)
            scale_factor=model._num_layers,
            backend_name="trtllm",
            inference_phase="context",
            attn_cp_size=cfg.cp_size,
        )
        _assert_ops_equivalent(built, _routed_slice(model.context_ops))

    @pytest.mark.parametrize("nextn", [0, 2])
    def test_generation_matches_legacy_moe_model(self, nextn):
        model, cfg = self._build_qwen3(
            "trtllm", tp_size=2, moe_tp_size=1, moe_ep_size=2, attention_dp_size=1, nextn=nextn
        )
        built = build_moe_block_ops(
            "generation",
            _shape_for(QWEN3),
            cfg,
            cfg.moe_quant_mode,
            "power_law_1.2",
            scale_factor=model._num_layers * model._mtp_scale_factor,
            backend_name="trtllm",
            inference_phase="generation",
            attn_cp_size=cfg.cp_size,
        )
        _assert_ops_equivalent(built, _routed_slice(model.generation_ops))

    def test_sglang_cp2_context_carries_cp_kwargs(self):
        """Context CP: router GEMM gets seq_split, dispatches get attn_cp_size."""
        model, cfg = self._build_qwen3(
            "sglang", tp_size=1, moe_tp_size=1, moe_ep_size=2, attention_dp_size=1, cp_size=2
        )
        built = build_moe_block_ops(
            "context",
            _shape_for(QWEN3),
            cfg,
            cfg.moe_quant_mode,
            "power_law_1.2",
            scale_factor=model._num_layers,
            backend_name="sglang",
            inference_phase="context",
            attn_cp_size=cfg.cp_size,
        )
        _assert_ops_equivalent(built, _routed_slice(model.context_ops))
        router_gemm, pre_dispatch, _, post_dispatch = built
        assert router_gemm._seq_split == 2
        assert pre_dispatch._attn_cp_size == 2
        assert post_dispatch._attn_cp_size == 2

    def test_sglang_cp2_generation_ignores_cp_kwargs(self):
        """Generation is not CP-modeled: legacy passes neither seq_split nor attn_cp_size."""
        model, cfg = self._build_qwen3(
            "sglang", tp_size=1, moe_tp_size=1, moe_ep_size=2, attention_dp_size=1, cp_size=2
        )
        built = build_moe_block_ops(
            "generation",
            _shape_for(QWEN3),
            cfg,
            cfg.moe_quant_mode,
            "power_law_1.2",
            scale_factor=model._num_layers * model._mtp_scale_factor,
            backend_name="sglang",
            inference_phase="generation",
            attn_cp_size=cfg.cp_size,
        )
        _assert_ops_equivalent(built, _routed_slice(model.generation_ops))
        router_gemm, pre_dispatch, _, post_dispatch = built
        assert router_gemm._seq_split == 1
        assert pre_dispatch._attn_cp_size == 1
        assert post_dispatch._attn_cp_size == 1


class TestSharedExpertsTriplet:
    """Builder's shared-expert triplet ≡ DeepSeekModel's hand-wired gate_up/act/ffn2."""

    def _build_deepseek(self, nextn=0):
        model_config = config.ModelConfig(tp_size=4, moe_tp_size=1, moe_ep_size=4, attention_dp_size=1, nextn=nextn)
        model = models.get_model(DSR1, model_config, "trtllm")
        return model, model_config

    def test_context_block_matches_deepseek_slices(self):
        model, cfg = self._build_deepseek()
        # Legacy scales by ALL 61 layers (dense-as-MoE approximation), not the
        # 58 MoE-true shape.num_moe_layers — scale_factor is caller-supplied.
        shape = _shape_for(DSR1)
        assert model._num_layers == 61
        assert shape.num_moe_layers == 58
        built = build_moe_block_ops(
            "context",
            shape,
            cfg,
            cfg.moe_quant_mode,
            "power_law_1.01",  # DeepSeekModel's _power_law_alpha
            scale_factor=model._num_layers,
            backend_name="trtllm",
            inference_phase="context",
            attn_cp_size=cfg.cp_size,
        )
        # Emission order: router first, then shared experts, then dispatch/moe/dispatch.
        assert [op._name for op in built] == [
            "context_router_gemm",
            "context_shared_gate_up_gemm",
            "context_shared_act_gate",
            "context_shared_ffn2_gemm",
            "context_moe_pre_dispatch",
            "context_moe",
            "context_moe_post_dispatch",
        ]
        _assert_ops_equivalent(_shared_slice(built), _shared_slice(model.context_ops))
        _assert_ops_equivalent(_routed_slice(built), _routed_slice(model.context_ops))
        assert built[0]._scale_factor == 61

    def test_generation_shared_triplet_matches_deepseek_overlap_group_b(self):
        model, cfg = self._build_deepseek(nextn=2)
        overlap = [op for op in model.generation_ops if isinstance(op, ops.OverlapOp)]
        assert len(overlap) == 1
        legacy_shared = _shared_slice(overlap[0]._group_b)
        assert len(legacy_shared) == 3
        built = build_moe_block_ops(
            "generation",
            _shape_for(DSR1),
            cfg,
            cfg.moe_quant_mode,
            "power_law_1.01",
            scale_factor=model._num_layers * model._mtp_scale_factor,
            backend_name="trtllm",
            inference_phase="generation",
            attn_cp_size=cfg.cp_size,
        )
        _assert_ops_equivalent(_shared_slice(built), legacy_shared)


def _toy_shape(num_shared_experts=0):
    return MoEBlockShape(
        hidden_size=1024,
        moe_inter_size=512,
        topk=4,
        num_experts=64,
        num_shared_experts=num_shared_experts,
        num_moe_layers=10,
    )


def _toy_cfg(system=None):
    model_config = config.ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        attention_dp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
    )
    if system is not None:
        model_config.system = system
    return model_config


def _build_toy(cfg, prefix="context", backend_name="sglang", **overrides):
    kwargs = {
        "scale_factor": 10,
        "backend_name": backend_name,
        "inference_phase": prefix,
        "attn_cp_size": 1,
        "gpus_per_node": 8,
    }
    kwargs.update(overrides)
    return build_moe_block_ops(prefix, _toy_shape(), cfg, cfg.moe_quant_mode, "uniform", **kwargs)


_FUSED_CONTEXT_NAMES = [
    "context_router_gemm",
    "context_moe_pre_dispatch",
    "context_moe",
    "context_moe_post_dispatch",
]


@pytest.fixture
def moe_block_registry():
    """Snapshot/restore the module-level registry: tests must not leak registrations."""
    snapshot = dict(moe_blocks._MOE_BLOCK_REGISTRY)
    try:
        yield moe_blocks._MOE_BLOCK_REGISTRY
    finally:
        moe_blocks._MOE_BLOCK_REGISTRY.clear()
        moe_blocks._MOE_BLOCK_REGISTRY.update(snapshot)


class TestRegisterMoeBlock:
    """G3 registry: most-specific-wins selection with a ``default`` continuation."""

    def test_registry_ships_exactly_the_deepseek_sglang_variants(self):
        # Task 5 registers the two A3 DeepSeek router-fidelity variants at
        # import time; anything else here means some test leaked a
        # registration past its fixture.
        assert set(moe_blocks._MOE_BLOCK_REGISTRY) == {
            ("DEEPSEEK", "sglang", "*"),
            ("DEEPSEEKV32", "sglang", "*"),
        }

    def test_duplicate_registration_raises(self, moe_block_registry):
        @register_moe_block(family="TOYFAM", framework="sglang")
        def _first(default, **ctx):
            return default()

        with pytest.raises(ValueError, match=r"\('TOYFAM', 'sglang', '\*'\)"):

            @register_moe_block(family="TOYFAM", framework="sglang")
            def _second(default, **ctx):
                return default()

    def test_most_specific_wins_across_three_specificities(self, moe_block_registry):
        @register_moe_block(family="TOYFAM")
        def _family_only(default, **ctx):
            return ["family"]

        @register_moe_block(family="TOYFAM", framework="sglang")
        def _family_framework(default, **ctx):
            return ["family_framework"]

        @register_moe_block(family="TOYFAM", framework="sglang", system="toy_system")
        def _fully_exact(default, **ctx):
            return ["fully_exact"]

        cfg = _toy_cfg(system="toy_system")
        assert _build_toy(cfg, backend_name="sglang", model_family="TOYFAM") == ["fully_exact"]

        cfg_other_system = _toy_cfg(system="other_system")
        assert _build_toy(cfg_other_system, backend_name="sglang", model_family="TOYFAM") == ["family_framework"]

        assert _build_toy(_toy_cfg(), backend_name="trtllm", model_family="TOYFAM") == ["family"]

    def test_exact_family_beats_exact_framework_and_system(self, moe_block_registry):
        """Left-to-right priority: family > framework > system (exact beats wildcard per position)."""

        @register_moe_block(family="TOYFAM")
        def _family_only(default, **ctx):
            return ["family"]

        @register_moe_block(framework="sglang", system="toy_system")
        def _framework_system(default, **ctx):
            return ["framework_system"]

        cfg = _toy_cfg(system="toy_system")
        assert _build_toy(cfg, backend_name="sglang", model_family="TOYFAM") == ["family"]

    def test_exact_framework_beats_exact_system(self, moe_block_registry):
        @register_moe_block(framework="sglang")
        def _framework_only(default, **ctx):
            return ["framework"]

        @register_moe_block(system="toy_system")
        def _system_only(default, **ctx):
            return ["system"]

        cfg = _toy_cfg(system="toy_system")
        assert _build_toy(cfg, backend_name="sglang", model_family="TOYFAM") == ["framework"]

    def test_unmatched_family_falls_to_default(self, moe_block_registry):
        @register_moe_block(family="OTHERFAM", framework="sglang")
        def _other_family(default, **ctx):
            return ["other_family"]

        result = _build_toy(_toy_cfg(), backend_name="sglang", model_family="TOYFAM")
        assert [op._name for op in result] == _FUSED_CONTEXT_NAMES

    def test_default_family_matches_only_wildcards(self, moe_block_registry):
        @register_moe_block(family="TOYFAM")
        def _family_only(default, **ctx):
            return ["family"]

        result = _build_toy(_toy_cfg(), backend_name="sglang")
        assert [op._name for op in result] == _FUSED_CONTEXT_NAMES

    def test_variant_composes_default_plus_one(self, moe_block_registry):
        marker = ops.ElementWise("context_variant_marker", 1, 8, 8, 0.8)

        @register_moe_block(family="TOYFAM")
        def _append_marker(default, **ctx):
            return default() + [marker]

        result = _build_toy(_toy_cfg(), model_family="TOYFAM")
        assert [op._name for op in result] == _FUSED_CONTEXT_NAMES + ["context_variant_marker"]
        assert result[-1] is marker

    def test_default_continuation_returns_fresh_ops(self, moe_block_registry):
        calls = []

        @register_moe_block(family="TOYFAM")
        def _call_default_twice(default, **ctx):
            calls.append(default())
            calls.append(default())
            return calls[0]

        _build_toy(_toy_cfg(), model_family="TOYFAM")
        first, second = calls
        assert [op._name for op in first] == [op._name for op in second]
        assert all(a is not b for a, b in zip(first, second, strict=True))

    def test_variant_receives_full_ctx(self, moe_block_registry):
        captured = {}

        @register_moe_block(family="TOYFAM")
        def _capture(default, **ctx):
            captured.update(ctx)
            return default()

        shape = _toy_shape()
        cfg = _toy_cfg()
        build_moe_block_ops(
            "generation",
            shape,
            cfg,
            cfg.moe_quant_mode,
            "uniform",
            scale_factor=12.5,
            backend_name="vllm",
            inference_phase="generation",
            model_family="TOYFAM",
            attn_cp_size=2,
            gpus_per_node=4,
            shared_gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        )
        assert set(captured) == {
            "prefix",
            "shape",
            "cfg",
            "quant_mode",
            "workload_distribution",
            "scale_factor",
            "backend_name",
            "inference_phase",
            "model_family",
            "attn_cp_size",
            "gpus_per_node",
            "shared_gemm_quant_mode",
        }
        assert captured["prefix"] == "generation"
        assert captured["shape"] is shape
        assert captured["cfg"] is cfg
        assert captured["quant_mode"] is cfg.moe_quant_mode
        assert captured["workload_distribution"] == "uniform"
        assert captured["scale_factor"] == 12.5
        assert captured["backend_name"] == "vllm"
        assert captured["inference_phase"] == "generation"
        assert captured["model_family"] == "TOYFAM"
        assert captured["attn_cp_size"] == 2
        assert captured["gpus_per_node"] == 4
        assert captured["shared_gemm_quant_mode"] is common.GEMMQuantMode.bfloat16

    def test_decorator_returns_function_unchanged(self, moe_block_registry):
        def variant(default, **ctx):
            return default()

        assert register_moe_block(family="TOYFAM")(variant) is variant
        assert moe_block_registry[("TOYFAM", "*", "*")] is variant


class TestLargeEPSeam:
    """cfg.moe_comm_backend drives the large-EP branch (emission pinned in
    test_moe_block_builder_large_ep.py; this smoke pins the seam itself)."""

    def test_comm_backend_for_phase_emits_large_ep_ops(self):
        cfg = _toy_cfg()
        cfg.moe_ep_size = 8
        cfg.attention_dp_size = 8
        cfg.moe_comm_backend = {"context": "deepep_ht"}
        result = _build_toy(cfg, prefix="context")
        assert [op._name for op in result] == [
            "context_router_gemm",
            "context_moe_dispatch",
            "context_moe",
            "context_moe_combine",
        ]
        assert isinstance(result[1], ops.MoEAllToAll)
        assert isinstance(result[2], ops.MoEExpertCompute)
        assert isinstance(result[3], ops.MoEAllToAll)

    def test_comm_backend_other_phase_falls_back_to_fused(self):
        cfg = _toy_cfg()
        cfg.moe_comm_backend = {"context": "deepep_ht"}
        result = _build_toy(cfg, prefix="generation")
        assert [op._name for op in result] == [name.replace("context", "generation") for name in _FUSED_CONTEXT_NAMES]


class TestPrefixPhaseConsistency:
    """``prefix`` is derived from ``inference_phase``; a mismatch is a caller bug."""

    def test_mismatched_prefix_and_phase_asserts(self):
        cfg = _toy_cfg()
        with pytest.raises(AssertionError, match="prefix"):
            build_moe_block_ops(
                "context",
                _toy_shape(),
                cfg,
                cfg.moe_quant_mode,
                "uniform",
                scale_factor=10,
                backend_name="sglang",
                inference_phase="generation",
            )
