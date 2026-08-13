# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.cli import api as cli_api
from tools.support_matrix.scan_rust_parity import (
    PROBE_STATUS_ENCODER_EVIDENCE_ERROR,
    Entry,
    ProbeRecord,
    _bucket_probe,
    _connect,
    _probe_shape,
    _retire_stale_visual_probe_results,
    _run_probe,
    init_db,
    pending_entries_for_probe,
    probe_entry,
    seed_entries,
    write_probe_record,
)
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
        return SimpleNamespace(
            ttft=12.0,
            tpot=3.0,
            raw={"encoder_latency": 1.25, "encoder_memory": 0.5},
        )

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


def test_visual_probe_fails_closed_without_encoder_evidence(monkeypatch):
    from tools.support_matrix import scan_rust_parity

    monkeypatch.setattr(
        cli_api,
        "cli_estimate",
        lambda **_kwargs: SimpleNamespace(ttft=12.0, tpot=3.0, raw={}),
    )
    monkeypatch.setattr(scan_rust_parity, "_get_test_constraints", lambda _model: _visual_constraints())

    record = probe_entry(_gemma_entry())

    assert record.status == PROBE_STATUS_ENCODER_EVIDENCE_ERROR
    assert _bucket_probe(record.status) == "REGRESSION"
    assert "ENCODER_NOT_EXERCISED:" in record.python_err
    assert "ENCODER_NOT_EXERCISED:" in record.rust_err


def test_resumed_visual_probe_retires_text_only_result(tmp_path):
    db_path = tmp_path / "scan.sqlite"
    entry = _gemma_entry()
    init_db(db_path)
    seed_entries(db_path, [entry])
    with _connect(db_path) as conn:
        write_probe_record(
            conn,
            ProbeRecord(
                entry_key=entry.key,
                probe_shape="isl=256,osl=256,prefix=128,total_gpus=32",
                python_ttft_ms=12.0,
                python_tpot_ms=3.0,
                rust_ttft_ms=12.0,
                rust_tpot_ms=3.0,
                ttft_drift_pct=0.0,
                tpot_drift_pct=0.0,
                python_err=None,
                rust_err=None,
                status="PASS",
                duration_ms=1.0,
                completed_at="2026-08-13T00:00:00+00:00",
            ),
        )

    assert _retire_stale_visual_probe_results(db_path, [entry]) == 1
    assert pending_entries_for_probe(db_path) == {entry.key}


def test_resumed_visual_probe_keeps_current_workload_result(tmp_path):
    db_path = tmp_path / "scan.sqlite"
    entry = _gemma_entry()
    init_db(db_path)
    seed_entries(db_path, [entry])
    with _connect(db_path) as conn:
        write_probe_record(
            conn,
            ProbeRecord(
                entry_key=entry.key,
                probe_shape=_probe_shape(_visual_constraints()),
                python_ttft_ms=12.0,
                python_tpot_ms=3.0,
                rust_ttft_ms=12.0,
                rust_tpot_ms=3.0,
                ttft_drift_pct=0.0,
                tpot_drift_pct=0.0,
                python_err=None,
                rust_err=None,
                status="PASS",
                duration_ms=1.0,
                completed_at="2026-08-13T00:00:00+00:00",
            ),
        )

    assert _retire_stale_visual_probe_results(db_path, [entry]) == 0
    assert pending_entries_for_probe(db_path) == set()
