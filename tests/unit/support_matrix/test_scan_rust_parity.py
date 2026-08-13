# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.cli import api as cli_api
from tools.support_matrix.scan_rust_parity import Entry, _run_probe, probe_entry
from tools.support_matrix.support_matrix import TestConstraints

pytestmark = pytest.mark.unit


def _visual_constraints() -> TestConstraints:
    return TestConstraints(
        total_gpus=32,
        isl=256,
        osl=256,
        prefix=128,
        ttft=2000.0,
        tpot=50.0,
        image_height=672,
        image_width=960,
        num_images=1,
    )


def _gemma_entry() -> Entry:
    return Entry(
        model="google/gemma-4-26B-A4B",
        architecture="Gemma4ForConditionalGeneration",
        system="b200_sxm",
        backend="trtllm",
        version="1.3.0rc20",
        mode="agg",
        baseline_status="PASS",
    )


def test_probe_passes_visual_constraints_to_estimator(monkeypatch):
    captured = {}

    def fake_cli_estimate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ttft=12.0, tpot=3.0)

    monkeypatch.setattr(cli_api, "cli_estimate", fake_cli_estimate)

    assert _run_probe(_gemma_entry(), _visual_constraints(), backend_label="rust") == (12.0, 3.0, None)
    assert captured["engine_step_backend"] == "rust"
    assert captured["image_height"] == 672
    assert captured["image_width"] == 960
    assert captured["num_images"] == 1


def test_probe_record_identifies_visual_shape(monkeypatch):
    from tools.support_matrix import scan_rust_parity

    monkeypatch.setattr(scan_rust_parity, "_get_test_constraints", lambda _model: _visual_constraints())
    monkeypatch.setattr(scan_rust_parity, "_run_probe", lambda *_args, **_kwargs: (12.0, 3.0, None))

    record = probe_entry(_gemma_entry())

    assert "image_height=672,image_width=960,num_images=1" in record.probe_shape
