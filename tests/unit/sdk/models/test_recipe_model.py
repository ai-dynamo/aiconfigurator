# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RecipeModel construction contract (GLM-5.2 pilot recipes).

These tests pin the recipe->op-graph mapping WITHOUT a perf database: layer-kind
decomposition must be numerically equivalent to the hand model's fraction
amortization (per-kind scale factors x full/skip fractions reproduce
21 full / 57 skip of 78 layers), the tolerated-divergence MoE policy must
trigger on traced evidence only, and coverage gaps must fail loud
(RecipeGapError), never fall back silently.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import RecipeGapError, RecipeModel, get_recipe_model
from aiconfigurator.sdk.models.recipe import RECIPES_DIR

pytestmark = pytest.mark.unit

_GLM = "zai-org/GLM-5.2"
_GLM_FP8 = "zai-org/GLM-5.2-FP8"


def _mc(tp: int = 1) -> config.ModelConfig:
    return config.ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1)


def _ops_by_prefix(ops_list, prefix):
    return [op for op in ops_list if op._name.startswith(prefix)]


def test_glm52_layer_kind_decomposition_matches_hand_amortization():
    model = get_recipe_model(_GLM, _mc(), "sglang")
    attn = _ops_by_prefix(model.context_ops, "context_attention[")
    # 3 traced kinds; scale factors must cover all 78 layers
    assert sorted(op._name for op in attn) == [
        "context_attention[full_indexer_dense]",
        "context_attention[full_indexer_moe]",
        "context_attention[shared_indexer_moe]",
    ]
    assert sum(op._scale_factor for op in attn) == 78
    # full/skip split from the layer-kind taxonomy == hand model's 21/57 of 78
    full_layers = sum(op._scale_factor * op._full_frac for op in attn)
    assert full_layers == 21


def test_glm52_moe_decompose_policy_triggers_on_traced_fusion():
    model = get_recipe_model(_GLM, _mc(), "sglang")
    # policy fires because traced experts (257) > router width (256)
    assert any("TOLERATED" in n for n in model.mapping_notes)
    moe = _ops_by_prefix(model.context_ops, "context_moe[")
    assert moe and all(op._num_experts == 256 and op._topk == 8 for op in moe)
    # decomposition adds the shared-expert FFN back, decode mirrors the overlap
    assert _ops_by_prefix(model.context_ops, "context_shared_gate_up_gemm[")
    assert _ops_by_prefix(model.generation_ops, "generation_moe_overlap[")
    # dense head layers are modeled (3 layers the hand model misses)
    dense = _ops_by_prefix(model.context_ops, "context_dense_gate_up_gemm[")
    assert [op._scale_factor for op in dense] == [3]


def test_glm52_faithful_policy_keeps_traced_shape():
    model = get_recipe_model(_GLM, _mc(), "sglang", moe_policy="faithful")
    assert any("FAITHFUL" in n for n in model.mapping_notes)
    moe = _ops_by_prefix(model.context_ops, "context_moe[")
    assert moe and all(op._num_experts == 257 and op._topk == 9 for op in moe)
    assert not _ops_by_prefix(model.context_ops, "context_shared_gate_up_gemm[")
    assert not _ops_by_prefix(model.generation_ops, "generation_moe_overlap[")


def test_fp8_identity_and_dsa_gemm_override():
    model = get_recipe_model(_GLM_FP8, _mc(), "sglang")
    attn = _ops_by_prefix(model.context_ops, "context_attention[")
    assert all(op._kvcache_quant_mode == common.KVCacheQuantMode.fp8 for op in attn)
    # traced projections are fp8_block (deep_gemm kernels in the trace)
    assert all(op._gemm_quant_mode == common.GEMMQuantMode.fp8_block for op in attn)

    compat = get_recipe_model(_GLM_FP8, _mc(), "sglang",
                              dsa_gemm_override=common.GEMMQuantMode.bfloat16)
    attn = _ops_by_prefix(compat.context_ops, "context_attention[")
    assert all(op._gemm_quant_mode == common.GEMMQuantMode.bfloat16 for op in attn)
    assert any("OVERRIDDEN" in n for n in compat.mapping_notes)


def test_untraced_layer_kind_fails_loud():
    recipe = yaml.safe_load((RECIPES_DIR / "zai-org--GLM-5.2.recipe.yaml").read_text())
    broken = copy.deepcopy(recipe)
    del broken["layer_kinds"]["full_indexer_dense"]
    from aiconfigurator.sdk.models.helpers import _get_model_info

    model_info = dict(_get_model_info(_GLM))
    model_info["model_path"] = _GLM
    model_info["model_family"] = "DEEPSEEKV32"
    mc = _mc()
    mc.resolve_moe_parallelism()
    with pytest.raises(RecipeGapError, match="never traced"):
        RecipeModel(broken, model_info, mc)


def test_missing_recipe_fails_loud():
    with pytest.raises(RecipeGapError, match="no recipe"):
        get_recipe_model("meta-llama/Llama-3.1-8B", _mc(), "sglang")


def test_engine_cache_identity_differs_from_hand_model():
    """The engine-handle cache assumes the op graph is a pure function of
    (model_path, ModelConfig). A recipe model shares the checkpoint path with
    the registry class but builds a different graph — without a distinct
    engine identity the second model silently answers from the first one's
    compiled handle (observed live before the op_graph_identity key)."""
    from types import SimpleNamespace

    from aiconfigurator.sdk.models import get_model
    from aiconfigurator.sdk.rust_engine_step import _engine_config_json

    fake_db = SimpleNamespace(system="b200_sxm", backend="sglang", version="0.5.14",
                              systems_root="")
    hand = get_model(_GLM, _mc(), "sglang")
    rec = get_recipe_model(_GLM, _mc(), "sglang")
    key_hand = _engine_config_json(hand, fake_db)
    key_rec = _engine_config_json(rec, fake_db)
    assert key_hand != key_rec
    assert "recipe:" in key_rec
    # two recipe variants must also get distinct handles
    key_faithful = _engine_config_json(
        get_recipe_model(_GLM, _mc(), "sglang", moe_policy="faithful"), fake_db)
    assert key_faithful != key_rec
