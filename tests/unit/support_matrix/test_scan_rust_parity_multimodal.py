# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from types import SimpleNamespace

import pandas as pd
import pytest

from tools.support_matrix.scan_rust_parity import (
    PARETO_STATUS_REGRESSION,
    PARETO_STATUS_STRICT_PASS,
    PROBE_STATUS_ENCODER_EVIDENCE_ERROR,
    Entry,
    ProbeRecord,
    _bucket_probe,
    _connect,
    _run_probe,
    cmd_report,
    init_db,
    load_entries,
    pareto_entry,
    probe_entry,
    seed_entries,
    write_probe_record,
)
from tools.support_matrix.support_matrix import (
    STATUS_PASS,
    SUPPORT_MATRIX_HEADER_WITH_SOURCE,
    SUPPORT_MATRIX_IMAGE_WORKLOAD,
    EncoderCoverage,
    SupportMatrix,
    TestConstraints,
)

pytestmark = pytest.mark.unit


def _entry() -> Entry:
    return Entry(
        model="Qwen/Qwen3-VL-8B-Instruct",
        architecture="Qwen3VLForConditionalGeneration",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        mode="agg",
        baseline_status=STATUS_PASS,
        image_workload=SUPPORT_MATRIX_IMAGE_WORKLOAD,
    )


def test_legacy_parity_entry_infers_canonical_image_workload(tmp_path):
    csv_path = tmp_path / "b200_sxm.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUPPORT_MATRIX_HEADER_WITH_SOURCE)
        writer.writerow(
            [
                "Qwen/Qwen3-VL-8B-Instruct",
                "Qwen3VLForConditionalGeneration",
                "b200_sxm",
                "vllm",
                "0.24.0",
                "agg",
                STATUS_PASS,
                "",
                "uv run aiconfigurator cli default --database-mode SILICON",
                "silicon",
            ]
        )

    [entry] = load_entries(tmp_path)

    assert entry.image_workload == SUPPORT_MATRIX_IMAGE_WORKLOAD
    assert entry.key.endswith("image=1024x1024x1")


def test_legacy_encoder_unsupported_pass_is_skipped_without_blocking_scan(tmp_path, caplog):
    csv_path = tmp_path / "b200_sxm.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUPPORT_MATRIX_HEADER_WITH_SOURCE)
        writer.writerow(
            [
                "Qwen/Qwen3.5-27B",
                "Qwen3_5ForConditionalGeneration",
                "b200_sxm",
                "vllm",
                "0.24.0",
                "agg",
                STATUS_PASS,
                "",
                "uv run aiconfigurator cli default --database-mode SILICON",
                "silicon",
            ]
        )
        writer.writerow(
            [
                "Qwen/Qwen3-8B",
                "Qwen3ForCausalLM",
                "b200_sxm",
                "vllm",
                "0.24.0",
                "agg",
                STATUS_PASS,
                "",
                "uv run aiconfigurator cli default --database-mode SILICON",
                "silicon",
            ]
        )

    entries = load_entries(tmp_path)

    assert [entry.model for entry in entries] == ["Qwen/Qwen3-8B"]
    assert entries[0].image_workload is None
    assert "text backbones were not parity-certified" in caplog.text


def test_text_only_parity_key_remains_backward_compatible():
    entry = Entry(
        model="Qwen/Qwen3-8B",
        architecture="Qwen3ForCausalLM",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        mode="agg",
        baseline_status=STATUS_PASS,
    )

    assert entry.key == "Qwen/Qwen3-8B|b200_sxm|vllm|0.24.0|agg"


def test_image_entry_retires_superseded_text_only_sqlite_result(tmp_path):
    db_path = tmp_path / "scan.sqlite"
    legacy_entry = Entry(
        model=_entry().model,
        architecture=_entry().architecture,
        system=_entry().system,
        backend=_entry().backend,
        version=_entry().version,
        mode=_entry().mode,
        baseline_status=STATUS_PASS,
    )
    init_db(db_path)
    seed_entries(db_path, [legacy_entry])
    with _connect(db_path) as conn:
        write_probe_record(
            conn,
            ProbeRecord(
                entry_key=legacy_entry.key,
                probe_shape="legacy-text-only",
                python_ttft_ms=1.0,
                python_tpot_ms=1.0,
                rust_ttft_ms=1.0,
                rust_tpot_ms=1.0,
                ttft_drift_pct=0.0,
                tpot_drift_pct=0.0,
                python_err=None,
                rust_err=None,
                status="PASS",
                duration_ms=1.0,
                completed_at="2026-01-01T00:00:00+00:00",
            ),
        )

    seed_entries(db_path, [_entry()])

    with _connect(db_path) as conn:
        assert conn.execute("SELECT entry_key FROM entries").fetchall() == [(_entry().key,)]
        assert conn.execute("SELECT entry_key FROM probe_results").fetchall() == []


def test_resumed_scan_retires_skipped_encoder_unsupported_result(tmp_path):
    db_path = tmp_path / "scan.sqlite"
    legacy_entry = Entry(
        model="Qwen/Qwen3.5-27B",
        architecture="Qwen3_5ForConditionalGeneration",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        mode="agg",
        baseline_status=STATUS_PASS,
    )
    init_db(db_path)
    seed_entries(db_path, [legacy_entry])
    with _connect(db_path) as conn:
        write_probe_record(
            conn,
            ProbeRecord(
                entry_key=legacy_entry.key,
                probe_shape="legacy-text-only",
                python_ttft_ms=1.0,
                python_tpot_ms=1.0,
                rust_ttft_ms=1.0,
                rust_tpot_ms=1.0,
                ttft_drift_pct=0.0,
                tpot_drift_pct=0.0,
                python_err=None,
                rust_err=None,
                status="PASS",
                duration_ms=1.0,
                completed_at="2026-01-01T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO pareto_results
                (entry_key, comparison_outcome, completed_at)
            VALUES (?, ?, ?)
            """,
            (legacy_entry.key, PARETO_STATUS_STRICT_PASS, "2026-01-01T00:00:00+00:00"),
        )

    # ``load_entries`` now skips this baseline row, so a resumed scan seeds
    # no replacement entry. Seeding must still retire its historical results.
    seed_entries(db_path, [])

    with _connect(db_path) as conn:
        assert conn.execute("SELECT entry_key FROM entries").fetchall() == []
        assert conn.execute("SELECT entry_key FROM probe_results").fetchall() == []
        assert conn.execute("SELECT entry_key FROM pareto_results").fetchall() == []


def test_resumed_scan_fails_closed_when_encoder_coverage_is_unresolvable(tmp_path, monkeypatch):
    db_path = tmp_path / "scan.sqlite"
    unsupported_entry = Entry(
        model="Qwen/Qwen3.5-27B",
        architecture="Qwen3_5ForConditionalGeneration",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        mode="agg",
        baseline_status=STATUS_PASS,
    )
    unresolvable_entry = Entry(
        model="example/unresolvable",
        architecture="ExampleForCausalLM",
        system="b200_sxm",
        backend="vllm",
        version="0.24.0",
        mode="agg",
        baseline_status=STATUS_PASS,
    )
    init_db(db_path)
    seed_entries(db_path, [unsupported_entry, unresolvable_entry])
    with _connect(db_path) as conn:
        for entry in (unsupported_entry, unresolvable_entry):
            write_probe_record(
                conn,
                ProbeRecord(
                    entry_key=entry.key,
                    probe_shape="legacy-text-only",
                    python_ttft_ms=1.0,
                    python_tpot_ms=1.0,
                    rust_ttft_ms=1.0,
                    rust_tpot_ms=1.0,
                    ttft_drift_pct=0.0,
                    tpot_drift_pct=0.0,
                    python_err=None,
                    rust_err=None,
                    status="PASS",
                    duration_ms=1.0,
                    completed_at="2026-01-01T00:00:00+00:00",
                ),
            )
            conn.execute(
                """
                INSERT INTO pareto_results
                    (entry_key, comparison_outcome, completed_at)
                VALUES (?, ?, ?)
                """,
                (entry.key, PARETO_STATUS_STRICT_PASS, "2026-01-01T00:00:00+00:00"),
            )

    def resolve_coverage(model):
        if model == unsupported_entry.model:
            return EncoderCoverage(True, False, unsupported_entry.architecture)
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr("tools.support_matrix.scan_rust_parity._get_encoder_coverage", resolve_coverage)

    with pytest.raises(RuntimeError, match="metadata unavailable"):
        seed_entries(db_path, [])

    # The first entry and its results are retired before the second lookup
    # fails. Closing the failed transaction must roll every mutation back.
    with _connect(db_path) as conn:
        expected_keys = [(unsupported_entry.key,), (unresolvable_entry.key,)]
        assert conn.execute("SELECT entry_key FROM entries ORDER BY rowid").fetchall() == expected_keys
        assert conn.execute("SELECT entry_key FROM probe_results ORDER BY rowid").fetchall() == expected_keys
        assert conn.execute("SELECT entry_key FROM pareto_results ORDER BY rowid").fetchall() == expected_keys


def test_parity_probe_passes_image_arguments(monkeypatch):
    calls = []

    def fake_estimate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            ttft=10.0,
            tpot=2.0,
            raw={"encoder_latency": 1.25, "encoder_memory": 0.5},
        )

    monkeypatch.setattr("aiconfigurator.cli.api.cli_estimate", fake_estimate)

    ttft, tpot, error = _run_probe(
        _entry(),
        TestConstraints(total_gpus=4, isl=256, osl=256, prefix=128, ttft=1500.0, tpot=50.0),
        "python",
    )

    assert (ttft, tpot, error) == (10.0, 2.0, None)
    assert calls[0]["image_height"] == 1024
    assert calls[0]["image_width"] == 1024
    assert calls[0]["num_images"] == 1


def test_parity_probe_fails_when_both_engines_omit_encoder_evidence(monkeypatch):
    monkeypatch.setattr(
        "aiconfigurator.cli.api.cli_estimate",
        lambda **_kwargs: SimpleNamespace(ttft=10.0, tpot=2.0, raw={}),
    )
    monkeypatch.setattr(
        "tools.support_matrix.scan_rust_parity._get_test_constraints",
        lambda _model: TestConstraints(4, 256, 256, 128, 1500.0, 50.0),
    )

    record = probe_entry(_entry())

    assert record.status == PROBE_STATUS_ENCODER_EVIDENCE_ERROR
    assert _bucket_probe(record.status) == "REGRESSION"
    assert "ENCODER_NOT_EXERCISED:" in record.python_err
    assert "ENCODER_NOT_EXERCISED:" in record.rust_err


def test_parity_report_includes_encoder_evidence_regression_details(tmp_path, capsys):
    db_path = tmp_path / "scan.sqlite"
    init_db(db_path)
    seed_entries(db_path, [_entry()])
    with _connect(db_path) as conn:
        write_probe_record(
            conn,
            ProbeRecord(
                entry_key=_entry().key,
                probe_shape="canonical-image",
                python_ttft_ms=None,
                python_tpot_ms=None,
                rust_ttft_ms=None,
                rust_tpot_ms=None,
                ttft_drift_pct=None,
                tpot_drift_pct=None,
                python_err="RuntimeError: ENCODER_NOT_EXERCISED: missing encoder_latency",
                rust_err="RuntimeError: ENCODER_NOT_EXERCISED: missing encoder_latency",
                status=PROBE_STATUS_ENCODER_EVIDENCE_ERROR,
                duration_ms=1.0,
                completed_at="2026-01-01T00:00:00+00:00",
            ),
        )

    assert cmd_report(SimpleNamespace(db_path=str(db_path), top=20, csv=None)) == 0

    report = capsys.readouterr().out
    assert "[ENCODER_EVIDENCE_ERROR]" in report
    assert "ENCODER_NOT_EXERCISED: missing encoder_latency" in report


def test_parity_pareto_runs_both_engines_with_encoder_evidence(monkeypatch):
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "encoder_latency": [1.25],
                "encoder_memory": [0.5],
                "request_rate": [1.0],
            }
        )

    monkeypatch.setattr("tools.support_matrix.scan_rust_parity._get_worker_matrix", lambda: None)
    monkeypatch.setattr(
        "tools.support_matrix.scan_rust_parity._get_test_constraints",
        lambda _model: TestConstraints(4, 256, 256, 128, 1500.0, 50.0),
    )
    monkeypatch.setattr(SupportMatrix, "_run_mode", staticmethod(fake_run_mode))

    record = pareto_entry(_entry())

    assert record.comparison_outcome == PARETO_STATUS_STRICT_PASS
    assert [call["engine_step_backend"] for call in calls] == ["python", "rust"]
    assert all(call["image_workload"] == SUPPORT_MATRIX_IMAGE_WORKLOAD for call in calls)


def test_parity_pareto_missing_encoder_evidence_is_regression(monkeypatch):
    monkeypatch.setattr("tools.support_matrix.scan_rust_parity._get_worker_matrix", lambda: None)
    monkeypatch.setattr(
        "tools.support_matrix.scan_rust_parity._get_test_constraints",
        lambda _model: TestConstraints(4, 256, 256, 128, 1500.0, 50.0),
    )
    monkeypatch.setattr(
        SupportMatrix,
        "_run_mode",
        staticmethod(lambda **_kwargs: pd.DataFrame({"request_rate": [1.0]})),
    )

    record = pareto_entry(_entry())

    assert record.comparison_outcome == PARETO_STATUS_REGRESSION
    assert "ENCODER_NOT_EXERCISED:" in record.error_msg
