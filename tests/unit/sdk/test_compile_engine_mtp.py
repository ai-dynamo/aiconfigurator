# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""compile_engine must apply MTP to the model it builds.

``compile_engine`` (the flat-kwargs entry the Rust ``AicEngineBuilder`` path
calls) previously forwarded ``nextn``/``nextn_accepted`` only into the spec
JSON; the model itself was built with ``nextn=0``, so the walked op lists
carried no ``1/(1+nextn_accepted)*(L+nextn)/L`` generation scale and compiled
engines ignored the MTP benefit entirely.
"""

from __future__ import annotations

import pytest

from aiconfigurator.sdk import engine

pytestmark = pytest.mark.unit


def test_compile_engine_applies_mtp_to_model(monkeypatch):
    captured = {}

    def _capture_spec(model, **kwargs):
        captured["model"] = model
        captured["nextn"] = kwargs["nextn"]
        captured["nextn_accepted"] = kwargs["nextn_accepted"]
        return "{}"

    monkeypatch.setattr(engine, "build_engine_spec_json", _capture_spec)
    monkeypatch.setattr(engine, "_maybe_load_database", lambda *a, **k: None)
    monkeypatch.setattr(engine.aiconfigurator_core, "engine_spec_bincode_from_json", lambda s: b"")

    engine.compile_engine(
        "Qwen/Qwen3-32B",
        "h200_sxm",
        "trtllm",
        nextn=1,
        nextn_accepted=0.85,
    )

    model = captured["model"]
    assert model._nextn == 1
    assert model._nextn_accepted == 0.85
    assert model._mtp_scale_factor != 1.0
    assert captured["nextn"] == 1 and captured["nextn_accepted"] == 0.85
