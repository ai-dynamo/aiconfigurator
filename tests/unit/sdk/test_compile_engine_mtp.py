# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled aic-core engines carry MTP compute depth, not acceptance."""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import engine

pytestmark = pytest.mark.unit


def test_compile_engine_applies_nextn_compute_cost_only(monkeypatch):
    captured = {}

    def _capture_spec(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", _capture_spec)
    monkeypatch.setattr(engine, "_maybe_load_database", lambda *a, **k: None)
    monkeypatch.setattr(engine.aiconfigurator_core, "engine_spec_bincode_from_json", lambda s: b"")

    engine.compile_engine(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        nextn=1,
    )

    model = captured["model"]
    assert model._nextn == 1
    assert not hasattr(model, "_nextn_accepted")
    assert model._mtp_scale_factor == pytest.approx((model._num_layers + 1) / model._num_layers)
    assert captured["kwargs"]["nextn"] == 1
    assert "nextn_accepted" not in captured["kwargs"]


def test_compile_engine_rejects_removed_nextn_accepted_parameter():
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'nextn_accepted'"):
        engine.compile_engine(
            "Qwen/Qwen3-32B",
            "h200_sxm",
            "trtllm",
            nextn=1,
            nextn_accepted=0.7,
        )
