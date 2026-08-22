# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat as stat_module

import pytest

from collector.fpm_forward.trtllm_campaign import run_trtllm_campaign
from collector.fpm_forward.trtllm_state import (
    TrtllmCoordinate,
    TrtllmLedger,
    TrtllmManifest,
    TrtllmRuntimeLimits,
)

pytestmark = pytest.mark.unit

REQUIRED_IBS_FIELDS = (
    "numContextRequests",
    "numCtxTokens",
    "numCtxKvTokens",
    "numGenRequests",
    "numGenKvTokens",
    "numQueuedContextRequests",
    "numQueuedCtxTokens",
    "numQueuedGenRequests",
    "numQueuedGenKvTokens",
    "numPausedRequests",
    "numPausedKvTokens",
)


def _runtime_limits() -> TrtllmRuntimeLimits:
    return TrtllmRuntimeLimits(
        max_seq_len=128,
        max_num_requests=4,
        max_batch_size=4,
        max_num_tokens=64,
        kv_cache_max_num_blocks=64,
        kv_cache_tokens_per_block=8,
        decode_cuda_graph_batch_sizes=(1, 2, 4),
    )


def _prefill_coordinate(*, total_tokens: int = 16) -> TrtllmCoordinate:
    return TrtllmCoordinate(
        workload_kind="prefill",
        batch_size=2,
        total_prefill_tokens=total_tokens,
        total_kv_read_tokens=0,
    )


def _manifest(*coordinates: TrtllmCoordinate) -> TrtllmManifest:
    return TrtllmManifest.build(
        campaign_id="kimi-k2.5-tep8",
        timing_rank_count=2,
        runtime_limits=_runtime_limits(),
        coordinates=coordinates or (_prefill_coordinate(),),
    )


def _rank_stats(coordinate: TrtllmCoordinate, *, iteration: int = 7) -> list[dict[str, object]]:
    ibs = dict.fromkeys(REQUIRED_IBS_FIELDS, 0)
    if coordinate.workload_kind == "prefill":
        ibs.update(
            numContextRequests=coordinate.batch_size,
            numCtxTokens=coordinate.total_prefill_tokens,
            numCtxKvTokens=coordinate.total_kv_read_tokens,
        )
    else:
        ibs.update(
            numGenRequests=coordinate.batch_size,
            numGenKvTokens=coordinate.total_kv_read_tokens,
        )
    return [
        {
            "rank": rank,
            "iter": iteration,
            "gpuForwardTimeMS": 1.0 + rank,
            "inflightBatchingStats": dict(ibs),
        }
        for rank in range(2)
    ]


def test_campaign_measures_pending_points_and_records_the_candidate_before_acceptance(tmp_path):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    evidence_path = tmp_path / "raw" / "evidence.jsonl"

    outcome = run_trtllm_campaign(
        ledger,
        evidence_path=evidence_path,
        attempt_id="engine-1",
        measure_point=lambda point, _recorder: _rank_stats(point),
    )

    assert outcome.status == "completed"
    assert outcome.measured_point_ids == (coordinate.point_id,)
    assert outcome.remaining_point_ids == ()
    assert tuple(measurement.point_id for measurement in ledger.accepted_measurements) == (coordinate.point_id,)
    assert json.loads(evidence_path.read_text()) == {
        "schema_name": "aic_trtllm_fpm_raw_evidence",
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "campaign_id": manifest.campaign_id,
        "point_id": coordinate.point_id,
        "coordinate": coordinate.to_dict(),
        "attempt_id": "engine-1",
        "record_kind": "measurement_candidate",
        "payload": {"rank_stats": _rank_stats(coordinate)},
    }


def test_callback_failure_is_recorded_durably_and_leaves_the_point_pending(tmp_path):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    evidence_path = tmp_path / "raw.jsonl"

    def fail_after_streaming(point, recorder):
        recorder.append("runtime_poll", {"point": point.point_id})
        raise RuntimeError("engine disconnected")

    with pytest.raises(RuntimeError, match="engine disconnected"):
        run_trtllm_campaign(
            ledger,
            evidence_path=evidence_path,
            attempt_id="engine-1",
            measure_point=fail_after_streaming,
        )

    records = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert [record["record_kind"] for record in records] == ["runtime_poll", "measurement_error"]
    assert records[-1]["payload"] == {
        "error_type": "RuntimeError",
        "message": "engine disconnected",
    }
    assert ledger.pending_coordinates == (coordinate,)
    assert ledger.accepted_measurements == ()


def test_forced_stop_reopen_and_resume_never_remeasures_an_accepted_coordinate(tmp_path):
    first = _prefill_coordinate(total_tokens=16)
    second = _prefill_coordinate(total_tokens=32)
    manifest = _manifest(first, second)
    ledger_root = tmp_path / "ledger"
    evidence_path = tmp_path / "raw.jsonl"
    ledger = TrtllmLedger.open(
        ledger_root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    stop_requested = False
    first_attempt_calls = []

    def measure_then_stop(coordinate, _recorder):
        nonlocal stop_requested
        first_attempt_calls.append(coordinate.point_id)
        stop_requested = True
        return _rank_stats(coordinate)

    first_outcome = run_trtllm_campaign(
        ledger,
        evidence_path=evidence_path,
        attempt_id="engine-1",
        measure_point=measure_then_stop,
        should_stop=lambda: stop_requested,
    )

    assert first_outcome.status == "stopped"
    assert first_outcome.measured_point_ids == (first.point_id,)
    assert first_outcome.remaining_point_ids == (second.point_id,)
    assert first_attempt_calls == [first.point_id]

    reopened = TrtllmLedger.open(
        ledger_root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    resumed_calls = []
    resumed_outcome = run_trtllm_campaign(
        reopened,
        evidence_path=evidence_path,
        attempt_id="engine-2",
        measure_point=lambda coordinate, _recorder: (
            resumed_calls.append(coordinate.point_id) or _rank_stats(coordinate, iteration=8)
        ),
    )

    assert resumed_outcome.status == "completed"
    assert resumed_outcome.measured_point_ids == (second.point_id,)
    assert resumed_calls == [second.point_id]
    assert [measurement.point_id for measurement in reopened.accepted_measurements] == [
        first.point_id,
        second.point_id,
    ]
    assert [measurement.attempt_id for measurement in reopened.accepted_measurements] == ["engine-1", "engine-2"]
    records = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert [record["point_id"] for record in records] == [first.point_id, second.point_id]
    assert [record["attempt_id"] for record in records] == ["engine-1", "engine-2"]


def test_completion_wins_when_stop_is_requested_during_the_final_point(tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    stop_requested = False

    def measure_then_stop(point, _recorder):
        nonlocal stop_requested
        stop_requested = True
        return _rank_stats(point)

    outcome = run_trtllm_campaign(
        ledger,
        evidence_path=tmp_path / "raw.jsonl",
        attempt_id="engine-1",
        measure_point=measure_then_stop,
        should_stop=lambda: stop_requested,
    )

    assert outcome.status == "completed"
    assert outcome.measured_point_ids == (coordinate.point_id,)
    assert outcome.remaining_point_ids == ()
    assert tuple(measurement.point_id for measurement in ledger.accepted_measurements) == (coordinate.point_id,)


def test_initial_stop_returns_without_measuring_or_materializing_evidence(tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    evidence_path = tmp_path / "raw" / "evidence.jsonl"
    calls = []

    outcome = run_trtllm_campaign(
        ledger,
        evidence_path=evidence_path,
        attempt_id="engine-1",
        measure_point=lambda point, _recorder: calls.append(point) or _rank_stats(point),
        should_stop=lambda: True,
    )

    assert outcome.status == "stopped"
    assert outcome.measured_point_ids == ()
    assert outcome.remaining_point_ids == (coordinate.point_id,)
    assert calls == []
    assert not evidence_path.exists()


def test_measurement_candidate_file_and_parent_are_fsynced_before_ledger_accept(monkeypatch, tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    events = []
    real_fsync = os.fsync
    real_accept = TrtllmLedger.accept

    def record_fsync(file_descriptor):
        descriptor_kind = "directory" if stat_module.S_ISDIR(os.fstat(file_descriptor).st_mode) else "file"
        events.append(f"{descriptor_kind} fsync")
        real_fsync(file_descriptor)

    def record_accept(self, point, *, attempt_id, rank_stats):
        events.append("accept")
        return real_accept(self, point, attempt_id=attempt_id, rank_stats=rank_stats)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(TrtllmLedger, "accept", record_accept)

    run_trtllm_campaign(
        ledger,
        evidence_path=tmp_path / "raw.jsonl",
        attempt_id="engine-1",
        measure_point=lambda point, _recorder: _rank_stats(point),
    )

    assert events[:3] == ["file fsync", "directory fsync", "accept"]


@pytest.mark.parametrize("attempt_id", ["", "   ", 7, None])
def test_invalid_attempt_identity_fails_before_measurement_or_raw_materialization(tmp_path, attempt_id):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    evidence_path = tmp_path / "raw.jsonl"
    calls = []

    with pytest.raises((TypeError, ValueError), match="attempt_id"):
        run_trtllm_campaign(
            ledger,
            evidence_path=evidence_path,
            attempt_id=attempt_id,
            measure_point=lambda point, _recorder: calls.append(point) or _rank_stats(point),
        )

    assert calls == []
    assert not evidence_path.exists()
    assert ledger.pending_coordinates == (coordinate,)


def test_existing_raw_log_bound_to_another_manifest_is_rejected_before_measurement(tmp_path):
    coordinate = _prefill_coordinate()
    first_manifest = _manifest(coordinate)
    evidence_path = tmp_path / "raw.jsonl"
    first_ledger = TrtllmLedger.open(
        tmp_path / "first-ledger",
        manifest=first_manifest,
        current_runtime_limits=_runtime_limits(),
    )
    run_trtllm_campaign(
        first_ledger,
        evidence_path=evidence_path,
        attempt_id="engine-1",
        measure_point=lambda point, _recorder: _rank_stats(point),
    )

    other_manifest = TrtllmManifest.build(
        campaign_id="other-campaign",
        timing_rank_count=2,
        runtime_limits=_runtime_limits(),
        coordinates=(coordinate,),
    )
    other_ledger = TrtllmLedger.open(
        tmp_path / "other-ledger",
        manifest=other_manifest,
        current_runtime_limits=_runtime_limits(),
    )
    calls = []

    with pytest.raises(ValueError, match="manifest identity"):
        run_trtllm_campaign(
            other_ledger,
            evidence_path=evidence_path,
            attempt_id="engine-2",
            measure_point=lambda point, _recorder: calls.append(point) or _rank_stats(point),
        )

    assert calls == []
    assert other_ledger.accepted_measurements == ()


def test_acceptance_failure_after_streamed_raw_leaves_the_point_replayable_without_retry(tmp_path):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    ledger_root = tmp_path / "ledger"
    evidence_path = tmp_path / "raw.jsonl"
    ledger = TrtllmLedger.open(
        ledger_root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    calls = []

    def return_incomplete_rank_stats(point, recorder):
        calls.append(point.point_id)
        recorder.append("runtime_poll", {"iter": 7})
        return _rank_stats(point)[:1]

    with pytest.raises(ValueError, match="exactly 2 rows"):
        run_trtllm_campaign(
            ledger,
            evidence_path=evidence_path,
            attempt_id="engine-1",
            measure_point=return_incomplete_rank_stats,
        )

    assert calls == [coordinate.point_id]
    assert ledger.pending_coordinates == (coordinate,)
    assert ledger.accepted_measurements == ()
    first_records = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert [record["record_kind"] for record in first_records] == ["runtime_poll", "measurement_candidate"]
    assert first_records[-1]["payload"]["rank_stats"] == _rank_stats(coordinate)[:1]

    reopened = TrtllmLedger.open(
        ledger_root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    outcome = run_trtllm_campaign(
        reopened,
        evidence_path=evidence_path,
        attempt_id="engine-2",
        measure_point=lambda point, _recorder: _rank_stats(point, iteration=8),
    )

    assert outcome.status == "completed"
    assert [measurement.attempt_id for measurement in reopened.accepted_measurements] == ["engine-2"]
    all_records = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert [record["attempt_id"] for record in all_records] == ["engine-1", "engine-1", "engine-2"]


@pytest.mark.parametrize("record_kind", ["measurement_candidate", "measurement_error"])
def test_callback_cannot_emit_campaign_owned_record_kinds(tmp_path, record_kind):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    evidence_path = tmp_path / "raw.jsonl"

    def emit_reserved_kind(_point, recorder):
        recorder.append(record_kind, {"forged": True})
        return _rank_stats(coordinate)

    with pytest.raises(ValueError, match="reserved"):
        run_trtllm_campaign(
            ledger,
            evidence_path=evidence_path,
            attempt_id="engine-1",
            measure_point=emit_reserved_kind,
        )

    records = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert [record["record_kind"] for record in records] == ["measurement_error"]
    assert ledger.pending_coordinates == (coordinate,)


def test_streamed_payload_is_detached_and_every_envelope_uses_canonical_json(tmp_path):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    evidence_path = tmp_path / "raw.jsonl"
    payload = {"z": [1], "a": {"value": True}}

    def stream_then_mutate(point, recorder):
        recorder.append("runtime_poll", payload)
        payload["z"].append(2)
        return _rank_stats(point)

    run_trtllm_campaign(
        ledger,
        evidence_path=evidence_path,
        attempt_id="engine-1",
        measure_point=stream_then_mutate,
    )

    raw_bytes = evidence_path.read_bytes()
    assert raw_bytes.endswith(b"\n")
    serialized_records = raw_bytes.decode().splitlines()
    records = [json.loads(line) for line in serialized_records]
    assert records[0]["payload"] == {"a": {"value": True}, "z": [1]}
    assert all(
        line == json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for line, record in zip(serialized_records, records, strict=True)
    )
    assert all(record["manifest_sha256"] == manifest.sha256 for record in records)
    assert all(record["campaign_id"] == manifest.campaign_id for record in records)
    assert all(record["point_id"] == coordinate.point_id for record in records)
    assert all(record["coordinate"] == coordinate.to_dict() for record in records)
    assert list(tmp_path.rglob("*.jsonl")) == [evidence_path]


def test_every_streaming_append_fsyncs_the_file_before_returning(monkeypatch, tmp_path):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    file_fsync_count = 0
    real_fsync = os.fsync

    def record_fsync(file_descriptor):
        nonlocal file_fsync_count
        if not stat_module.S_ISDIR(os.fstat(file_descriptor).st_mode):
            file_fsync_count += 1
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)

    def stream_twice(point, recorder):
        recorder.append("request", {"sequence": 1})
        assert file_fsync_count == 1
        recorder.append("runtime_poll", {"sequence": 2})
        assert file_fsync_count == 2
        return _rank_stats(point)

    run_trtllm_campaign(
        ledger,
        evidence_path=tmp_path / "raw.jsonl",
        attempt_id="engine-1",
        measure_point=stream_twice,
    )

    assert file_fsync_count == 4  # two streamed rows, candidate, then the accepted ledger file


@pytest.mark.parametrize("bad_payload", [{"value": float("nan")}, {"value": object()}])
def test_streaming_serialization_failure_writes_only_the_error_record(tmp_path, bad_payload):
    coordinate = _prefill_coordinate()
    ledger = TrtllmLedger.open(
        tmp_path / "ledger",
        manifest=_manifest(coordinate),
        current_runtime_limits=_runtime_limits(),
    )
    evidence_path = tmp_path / "raw.jsonl"

    def stream_invalid_payload(_point, recorder):
        recorder.append("runtime_poll", bad_payload)
        return _rank_stats(coordinate)

    with pytest.raises((TypeError, ValueError)):
        run_trtllm_campaign(
            ledger,
            evidence_path=evidence_path,
            attempt_id="engine-1",
            measure_point=stream_invalid_payload,
        )

    records = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    assert [record["record_kind"] for record in records] == ["measurement_error"]
    assert records[0]["payload"]["error_type"] in {"TypeError", "ValueError"}
    assert ledger.pending_coordinates == (coordinate,)


@pytest.mark.parametrize("tamper", ["coordinate", "unknown_key", "noncanonical", "truncated"])
def test_existing_raw_log_tampering_is_rejected_even_when_the_ledger_is_complete(tmp_path, tamper):
    coordinate = _prefill_coordinate()
    manifest = _manifest(coordinate)
    ledger_root = tmp_path / "ledger"
    evidence_path = tmp_path / "raw.jsonl"
    ledger = TrtllmLedger.open(
        ledger_root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    run_trtllm_campaign(
        ledger,
        evidence_path=evidence_path,
        attempt_id="engine-1",
        measure_point=lambda point, _recorder: _rank_stats(point),
    )
    record = json.loads(evidence_path.read_text())
    if tamper == "coordinate":
        record["coordinate"]["total_prefill_tokens"] += 1
        evidence_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    elif tamper == "unknown_key":
        record["unexpected"] = True
        evidence_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    elif tamper == "noncanonical":
        evidence_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    else:
        evidence_path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")))

    reopened = TrtllmLedger.open(
        ledger_root,
        manifest=manifest,
        current_runtime_limits=_runtime_limits(),
    )
    calls = []

    with pytest.raises((TypeError, ValueError)):
        run_trtllm_campaign(
            reopened,
            evidence_path=evidence_path,
            attempt_id="engine-2",
            measure_point=lambda point, _recorder: calls.append(point) or _rank_stats(point),
        )

    assert calls == []
