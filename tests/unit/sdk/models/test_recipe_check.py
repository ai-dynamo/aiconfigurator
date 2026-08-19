# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe-check drift detection against the GLM-5.2 pilot recipes.

Recipes are trace-extracted REFERENCE artifacts (repo ``recipes/``); models
keep being built from config interpretation. These tests pin the checker's
contract: it must (a) confirm what the hand model gets right (attention
coverage, skip-indexer fraction, kv identity), (b) recognize the fused-shared-
expert decomposition as the TOLERATED divergence it is, and (c) CATCH the real
drift the GLM-5.2 pilot found — the dense head layers the hand model does not
model, and the FP8 DSA gemm-key mismatch. If a hand-model fix lands, the
expected statuses below flip to MATCH and this file is the place that notices.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiconfigurator.sdk import config
from aiconfigurator.sdk.models import check_model_against_recipe, get_model

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[4]
_RECIPES = _REPO / "recipes"
_GLM = "zai-org/GLM-5.2"
_GLM_FP8 = "zai-org/GLM-5.2-FP8"


def _mc(tp: int = 1) -> config.ModelConfig:
    return config.ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1)


def _by_area(report):
    out = {}
    for f in report.findings:
        out.setdefault(f.area, []).append(f)
    return out


def test_glm52_hand_model_check_confirms_and_catches():
    model = get_model(_GLM, _mc(), "sglang")
    report = check_model_against_recipe(model, _RECIPES / "zai-org--GLM-5.2.recipe.yaml")
    areas = _by_area(report)

    # (a) confirmations: attention coverage + skip amortization + kv identity
    attn = {f.detail.split(":")[0]: f.status for f in areas["attention/dsa"]}
    assert attn["layer coverage"] == "MATCH"
    assert attn["full-indexer layers"] == "MATCH"
    assert attn["kv cache identity"] == "MATCH"

    # (b) fused shared expert -> the tolerated decomposition, not a failure
    moe = areas["mlp/fused_moe"]
    assert any(f.status == "TOLERATED" and "shared-expert" in f.detail for f in moe)

    # (c) real drift caught: the hand model counts all 78 layers as MoE and
    # models no dense-MLP head (pilot finding). If this starts passing, the
    # hand model was fixed — update the expectation and celebrate.
    assert any(f.status == "DIVERGENT" and "MoE layer coverage" in f.detail for f in moe)
    dense = areas["mlp/dense"]
    assert dense and dense[0].status == "DIVERGENT"
    assert not report.ok


def test_glm52_fp8_check_catches_dsa_gemm_key_mismatch():
    model = get_model(_GLM_FP8, _mc(), "sglang")
    report = check_model_against_recipe(model, _RECIPES / "zai-org--GLM-5.2-FP8.recipe.yaml")
    areas = _by_area(report)
    gemm = [f for f in areas["attention/dsa"] if f.detail.startswith("DSA module gemm key")]
    # traced projections run fp8_block deep_gemm kernels; the hand model keys
    # the module tables with bfloat16 (pilot finding, needs collector provenance)
    assert gemm and gemm[0].status == "DIVERGENT"
    # kv identity is fp8 on both sides
    kv = [f for f in areas["attention/dsa"] if f.detail.startswith("kv cache identity")]
    assert kv and kv[0].status == "MATCH"


def test_unclaimed_block_is_reported_not_skipped():
    import copy

    import yaml

    recipe = yaml.safe_load((_RECIPES / "zai-org--GLM-5.2.recipe.yaml").read_text())
    broken = copy.deepcopy(recipe)
    for phase in broken["layer_kinds"]["shared_indexer_moe"].values():
        for o in phase["layer_ops"]:
            o["op"] = o["op"].replace("DeepseekSparseAttnBackend", "SomeNewAttnBackend")
    model = get_model(_GLM, _mc(), "sglang")
    report = check_model_against_recipe(model, broken)
    assert any(f.status == "UNCHECKED" and "self_attn" in f.area for f in report.findings)


def test_engine_cache_identity_honors_op_graph_discriminator():
    """The engine-handle cache assumes the op graph is a pure function of
    (model_path, ModelConfig). Two structurally different models sharing a
    checkpoint path silently answered from one compiled handle (observed live
    during the recipe pilot); ``engine_identity_extra`` must widen the key."""
    from types import SimpleNamespace

    from aiconfigurator.sdk.rust_engine_step import _engine_config_json

    fake_db = SimpleNamespace(system="b200_sxm", backend="sglang", version="0.5.14", systems_root="")
    base = get_model(_GLM, _mc(), "sglang")
    variant = get_model(_GLM, _mc(), "sglang")
    variant.engine_identity_extra = "shadow-variant:xyz"
    assert _engine_config_json(base, fake_db) != _engine_config_json(variant, fake_db)
