# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-facts contract: the hf→config conversion, its declared approximations,
and both check directions, pinned on the GLM-5.2 pilot evidence.

Facts record TRUE structure (GLM-5.2: 3 dense-head + 75 MoE of 78); deliberate
simplifications are declared approximation rules (dense head modeled as MoE,
fused shared expert modeled decomposed — owner decisions with measured impact
bounds), so checks report APPROX for them and reserve DIVERGENT for real
drift. The FP8 DSA quant-key mismatch stays pinned DIVERGENT until collector
provenance settles it; if a fix lands, that expectation flips and this file is
the place that notices.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import (
    assemble_model_facts,
    check_facts_against_dryrun,
    check_model_against_facts,
    get_model,
    resolve_model_quant_modes,
    summarize_dryruns,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[4]
_DRYRUN = _REPO / "references" / "dryrun"
_GLM = "zai-org/GLM-5.2"
_GLM_FP8 = "zai-org/GLM-5.2-FP8"
_GLM_NVFP4 = "nvidia/GLM-5.2-NVFP4"


def _mc(tp: int = 1) -> config.ModelConfig:
    return config.ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1)


def _summary(model: str) -> Path:
    return _DRYRUN / f"{model.replace('/', '--')}.yaml"


def _statuses(report, area_prefix):
    return {f.status for f in report.findings if f.area.startswith(area_prefix)}


def test_assemble_glm52_facts_structure_and_approximations():
    facts = assemble_model_facts(_GLM, _mc(), "sglang", system_name="b200_sxm")
    assert facts.layer_kinds == {
        "full_indexer_dense": 3, "full_indexer_moe": 18, "shared_indexer_moe": 57}
    assert facts.dense_head_layers == 3 and facts.moe_layers == 75
    assert facts.moe == {"num_routed": 256, "topk": 8, "n_shared": 1, "inter": 2048}
    assert facts.quant["kvcache"] == common.KVCacheQuantMode.bfloat16
    assert facts.branch_params["index_topk"] == 2048
    # both owner-declared simplifications apply to this model
    assert set(facts.approximations) == {"dense_head_as_moe", "fused_shared_expert_decomposed"}


def test_resolve_is_system_aware_single_choke_point():
    """nvfp4 keeps native FP4 on Blackwell but must be remapped on Hopper —
    previously each caller's duty (and missing entirely on the Mocker path)."""
    mc_b200, mc_h200 = _mc(), _mc()
    resolve_model_quant_modes(mc_b200, _GLM_NVFP4, "sglang", system_name="b200_sxm")
    resolve_model_quant_modes(mc_h200, _GLM_NVFP4, "sglang", system_name="h200_sxm")
    assert mc_b200.moe_quant_mode == common.MoEQuantMode.nvfp4
    assert mc_h200.moe_quant_mode != common.MoEQuantMode.nvfp4  # no FP4 TCs on Hopper


def test_compile_engine_applies_system_aware_resolution(monkeypatch):
    """The embedded (Rust build_aic_engine / Mocker) path must run the same
    resolution as CLI/task — it previously ran none."""
    import aiconfigurator_core.sdk.engine as engine
    from aiconfigurator_core.sdk.models import facts as facts_mod

    calls = []
    monkeypatch.setattr(
        facts_mod, "resolve_model_quant_modes",
        lambda mc, mp, be, system_name=None, **kw: calls.append((mp, be, system_name)))
    engine.compile_engine(_GLM, "b200_sxm", "sglang", backend_version="0.5.14")
    assert calls == [(_GLM, "sglang", "b200_sxm")]


def test_facts_vs_dryrun_glm52():
    facts = assemble_model_facts(_GLM, _mc(), "sglang", system_name="b200_sxm")
    report = check_facts_against_dryrun(facts, _summary(_GLM))
    # all three config-derived kinds have evidence; kv and quant agree
    assert not any(f.status == "UNCHECKED" and f.area.startswith("coverage") for f in report.findings)
    assert _statuses(report, "identity/kv") == {"MATCH"}
    assert _statuses(report, "quant/") == {"MATCH"}
    # runtime fuses the shared expert -> recognized as the declared approximation
    assert _statuses(report, "moe/") == {"APPROX"}
    assert report.ok


def test_facts_vs_dryrun_missing_kind_is_surfaced():
    import yaml

    facts = assemble_model_facts(_GLM, _mc(), "sglang", system_name="b200_sxm")
    partial = yaml.safe_load(_summary(_GLM).read_text())
    del partial["layer_kinds"]["full_indexer_dense"]
    report = check_facts_against_dryrun(facts, partial)
    assert any(f.status == "UNCHECKED" and "full_indexer_dense" in f.detail for f in report.findings)


def test_summarize_dryruns_distills_raw_probe_records():
    """The summary format has one owner: this distiller. Pin what it keeps
    (quant classes, runtime shapes, branch evidence) and what it drops
    (op sequences, kernel timings, weight tables)."""
    raw = {
        "model_path": "/work/dummy_models/glm/Fake-1B__moe",
        "sglang_version": "0.5.16", "tp": 1,
        "server_args_resolved": {"kv_cache_dtype": "fp8_e4m3"},
        "model_config": {"branch_params": {"index_topk": 2048, "index_topk_freq": 4}},
        "quant_methods_by_module": {
            "model.layers.0.self_attn.o_proj": "Fp8LinearMethod",
            "model.layers.1.self_attn.o_proj": "Fp8LinearMethod",
            "model.layers.0.mlp.experts": "Fp8MoEMethod",
            "model.embed_tokens": "UnquantizedEmbeddingMethod",  # non-layer: dropped
        },
        "weights": {"model.layers.0.mlp.experts.w13_weight": "float8_e4m3fn[9, 512, 128]",
                    "model.layers.0.mlp.gate.weight": "bfloat16[8, 128]"},
        "phases": {
            "prefill:b2_isl32": {"ops": [
                {"span": "AIC::attn::X.forward_extend::m", "depth": 1, "kernels": {"void fa3_kernel<a>": {}}}]},
            "prefill:b1_isl4096": {"ops": [
                {"span": "AIC::attn::X.forward_extend::m", "depth": 1, "kernels": {"void sparse_kernel<b>": {}}}]},
            "decode:b2_isl32": {"ops": [
                {"span": "AIC::moe::f", "depth": 1, "in": ["topk_output.topk_ids=int32[2, 3]"]}]},
        },
    }
    s = summarize_dryruns([raw])
    kind = s["layer_kinds"]["moe"]
    assert kind["quant_by_module"] == {"mlp.experts": "Fp8MoEMethod",
                                       "self_attn.o_proj": "Fp8LinearMethod"}
    assert kind["moe_runtime"] == {"num_experts": 9, "inter": 256, "router_width": 8, "topk": 3}
    assert kind["prefill_branch"]["threshold_candidates"] == {"index_topk": 2048}
    assert kind["prefill_branch"]["isl32"].startswith("fa3_kernel")
    assert s["kv_cache_dtype"] == "fp8_e4m3"
    assert "weights" not in s and "phases" not in s  # raw bulk stays in the archive


def test_hand_model_vs_facts_glm52():
    facts = assemble_model_facts(_GLM, _mc(), "sglang", system_name="b200_sxm")
    model = get_model(_GLM, _mc(), "sglang")
    report = check_model_against_facts(model, facts)
    by_area = {f.area: f.status for f in report.findings}
    assert by_area["identity/kv"] == "MATCH"
    assert by_area["attention/coverage"] == "MATCH"
    assert by_area["attention/skip_fraction"] == "MATCH"  # 21 full of 78
    # dense head modeled as MoE: the declared approximation, not a failure
    assert by_area["moe/coverage"] == "APPROX"
    assert by_area["moe/shape"] == "MATCH"  # (256, 8, 2048) both sides
    assert report.ok


def test_hand_model_vs_facts_fp8_pins_quant_key_drift():
    facts = assemble_model_facts(_GLM_FP8, _mc(), "sglang", system_name="b200_sxm")
    model = get_model(_GLM_FP8, _mc(), "sglang")
    report = check_model_against_facts(model, facts)
    by_area = {f.area: f.status for f in report.findings}
    # traced/checkpoint projections are fp8_block; the model keys DSA tables
    # with bf16 (pilot finding F8, unresolved pending collector provenance)
    assert by_area["attention/quant_key"] == "DIVERGENT"
    assert by_area["identity/kv"] == "MATCH"


def test_engine_cache_identity_honors_op_graph_discriminator():
    """The engine-handle cache assumes the op graph is a pure function of
    (model_path, ModelConfig). Two structurally different models sharing a
    checkpoint path silently answered from one compiled handle (observed live
    during the pilot); ``engine_identity_extra`` must widen the key."""
    from types import SimpleNamespace

    from aiconfigurator.sdk.rust_engine_step import _engine_config_json

    fake_db = SimpleNamespace(system="b200_sxm", backend="sglang", version="0.5.14", systems_root="")
    base = get_model(_GLM, _mc(), "sglang")
    variant = get_model(_GLM, _mc(), "sglang")
    variant.engine_identity_extra = "shadow-variant:xyz"
    assert _engine_config_json(base, fake_db) != _engine_config_json(variant, fake_db)
