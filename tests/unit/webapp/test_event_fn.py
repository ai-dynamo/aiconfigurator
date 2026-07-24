# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Webapp event handler regression tests.

Currently a thin test of the family-keyed MoE-control visibility path
(`EventFn.update_model_related_components`). The webapp branch on the
selected model's family to decide which MoE-related UI controls render.
Adding a new MoE family without wiring it into this branch produces a
silent regression in the deployed webapp: the family is registered in
the SDK and selectable in the model dropdown, but its MoE controls are
hidden — which leads to wrong default precision and memory-fit failure.

These tests exist to make that class of regression impossible to ship
silently for the model families we explicitly own. Each test owns
exactly one family.
"""

import ast
import inspect

import pytest

pytestmark = pytest.mark.unit

# Webapp pulls in gradio at module-import time; tests skip cleanly when gradio
# is absent (e.g. in minimal lint envs). The webapp itself requires gradio at
# runtime, so this is purely a test-environment shim.
pytest.importorskip("gradio")

from aiconfigurator.webapp.events import event_fn
from aiconfigurator.webapp.events.event_fn import EventFn


def test_model_config_calls_do_not_receive_acceptance_assumption():
    """The webapp keeps acceptance above the aic-core ModelConfig boundary."""
    tree = ast.parse(inspect.getsource(event_fn))
    model_config_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ModelConfig"
    ]

    assert model_config_calls
    assert all("nextn_accepted" not in {keyword.arg for keyword in call.keywords} for call in model_config_calls)


class TestUpdateModelRelatedComponents:
    """Regression guards for the family-keyed MoE-control visibility branch."""

    def test_gemma4mix_shows_moe_controls(self):
        """`google/gemma-4-26B-A4B` resolves to the GEMMA4MIX family and is MoE,
        so every MoE-related control (nextn, nextn_accepted, enable_wideep,
        moe_quant_mode, moe_tp_size, moe_ep_size, dp_size) must render visible.
        """
        result = EventFn.update_model_related_components("google/gemma-4-26B-A4B")
        assert len(result) == 7, "expected 7 gr.update objects (nextn + 6 MoE controls)"
        for i, update in enumerate(result):
            visible = update.get("visible") if hasattr(update, "get") else getattr(update, "visible", None)
            assert visible is True, f"component {i} hidden for GEMMA4MIX — DYN-3044-style regression"
