# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-facts check inside get_model + system-aware resolution.

The GLM-5.2 pilot showed the failure surface of model building is the
hf→config conversion feeding get_model, so the build now (a) optionally
applies the system-aware quant remaps itself (``system_name=``; previously
each caller's duty and missing entirely on the compile_engine / Dynamo Mocker
path), and (b) compares its derivations against upstream-produced dry-run
facts (``model_facts/*.yaml``), logging a warning on divergence — never
failing, warning once per model. The framework fusing the shared expert into
the routed experts (runtime 257/topk-9 vs config 256/8/+1) is the known,
deliberate modeling decomposition and must NOT warn.
"""

from __future__ import annotations

import logging

import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model, model_facts_divergences
from aiconfigurator.sdk.models.helpers import _get_model_info, _load_model_facts

pytestmark = pytest.mark.unit

_GLM = "zai-org/GLM-5.2"
_GLM_FP8 = "zai-org/GLM-5.2-FP8"
_GLM_NVFP4 = "nvidia/GLM-5.2-NVFP4"


def _mc(tp: int = 1) -> config.ModelConfig:
    return config.ModelConfig(tp_size=tp, moe_tp_size=tp, moe_ep_size=1)


def _divergences(model_path: str) -> list[str]:
    mc = _mc()
    model = get_model(model_path, mc, "sglang")
    return model_facts_divergences(model_path, mc, dict(_get_model_info(model_path)))


def test_glm52_build_matches_upstream_facts():
    # kv identity, attention-projection quant, and the MoE shape (fused shared
    # expert recognized as the deliberate decomposition) all agree
    assert _divergences(_GLM) == []
    assert _divergences(_GLM_FP8) == []


def test_facts_check_warns_on_real_divergence(monkeypatch, caplog):
    """Tampered facts (as if the framework changed identity under us) must
    surface as a warning during get_model — and only once per model."""
    facts = dict(_load_model_facts(_GLM) or {})
    assert facts, "packaged facts file for GLM-5.2 must exist"
    facts["kv_cache_dtype"] = "fp8_e4m3"  # framework 'switched' kv identity
    import aiconfigurator_core.sdk.models.helpers as helpers

    monkeypatch.setattr(helpers, "_load_model_facts", lambda _mp: facts)
    monkeypatch.setattr(helpers, "_FACTS_WARNED", set())
    with caplog.at_level(logging.WARNING):
        get_model(_GLM, _mc(), "sglang")
        get_model(_GLM, _mc(), "sglang")  # second build: no duplicate warning
    hits = [r for r in caplog.records if "model facts divergence" in r.message]
    assert len(hits) == 1 and "kv cache" in hits[0].getMessage()


def test_fused_shared_expert_runtime_shape_does_not_warn():
    facts = _load_model_facts(_GLM)
    rt = next(ev["moe_runtime"] for ev in facts["layer_kinds"].values() if "moe_runtime" in ev)
    # the packaged facts really do carry the fused runtime shape...
    assert (rt["num_experts"], rt["topk"]) == (257, 9)
    # ...and the check treats it as the known decomposition of config (256, 8, +1)
    assert _divergences(_GLM) == []


def test_get_model_system_name_applies_quant_remaps():
    """nvfp4 keeps native FP4 on Blackwell but must be remapped on Hopper —
    previously each caller's duty, now one optional arg on the build itself."""
    mc_b200, mc_h200 = _mc(), _mc()
    get_model(_GLM_NVFP4, mc_b200, "sglang", system_name="b200_sxm")
    get_model(_GLM_NVFP4, mc_h200, "sglang", system_name="h200_sxm")
    assert mc_b200.moe_quant_mode == common.MoEQuantMode.nvfp4
    assert mc_h200.moe_quant_mode != common.MoEQuantMode.nvfp4  # no FP4 TCs on Hopper


def test_compile_engine_passes_system_name(monkeypatch):
    """The embedded (Rust build_aic_engine / Mocker) path must get the same
    system-aware resolution as CLI/task — it previously ran none."""
    import aiconfigurator_core.sdk.engine as engine

    seen = {}
    real_get_model = engine.get_model

    def spy(model_path, model_config, backend_name, system_name=None):
        seen["system_name"] = system_name
        return real_get_model(model_path, model_config, backend_name, system_name=system_name)

    monkeypatch.setattr(engine, "get_model", spy)
    engine.compile_engine(_GLM, "b200_sxm", "sglang", backend_version="0.5.14")
    assert seen["system_name"] == "b200_sxm"


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
