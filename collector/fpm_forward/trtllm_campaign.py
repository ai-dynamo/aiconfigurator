# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure, resumable campaign loop for TRT-LLM FPM measurements."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .trtllm_state import TrtllmCoordinate, TrtllmLedger, TrtllmManifest

_RAW_SCHEMA_NAME = "aic_trtllm_fpm_raw_evidence"
_RAW_SCHEMA_VERSION = 1
_CAMPAIGN_RECORD_KINDS = frozenset({"measurement_candidate", "measurement_error"})
_RAW_ENVELOPE_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "manifest_sha256",
        "campaign_id",
        "point_id",
        "coordinate",
        "attempt_id",
        "record_kind",
        "payload",
    }
)

RankStats = Sequence[Mapping[str, object]]


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _never_stop() -> bool:
    return False


@dataclass(frozen=True, slots=True)
class TrtllmCampaignOutcome:
    status: Literal["completed", "stopped"]
    measured_point_ids: tuple[str, ...]
    remaining_point_ids: tuple[str, ...]


class _RawEvidenceLog:
    def __init__(self, path: str | Path, manifest: TrtllmManifest) -> None:
        self._path = Path(path)
        self._manifest = manifest
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._materialized = self._path.exists()
        if self._materialized:
            self._validate_existing()

    def _validate_existing(self) -> None:
        if not self._path.is_file():
            raise ValueError(f"TRT-LLM raw evidence path is not a file: {self._path}")
        coordinates_by_id = {coordinate.point_id: coordinate for coordinate in self._manifest.coordinates}
        with self._path.open("rb") as evidence_file:
            for line_number, raw_line in enumerate(evidence_file, start=1):
                if not raw_line.endswith(b"\n"):
                    raise ValueError(f"TRT-LLM raw evidence line {line_number} is not newline terminated")
                try:
                    serialized = raw_line[:-1].decode("utf-8")
                    envelope = json.loads(serialized)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"cannot parse TRT-LLM raw evidence line {line_number}: {error}") from error
                if not isinstance(envelope, dict):
                    raise TypeError(f"TRT-LLM raw evidence line {line_number} must be a mapping")
                actual_keys = set(envelope)
                if actual_keys != _RAW_ENVELOPE_KEYS:
                    raise ValueError(
                        f"TRT-LLM raw evidence line {line_number} keys differ: "
                        f"missing={sorted(_RAW_ENVELOPE_KEYS - actual_keys)}, "
                        f"unknown={sorted(actual_keys - _RAW_ENVELOPE_KEYS)}"
                    )
                if serialized != _canonical_json(envelope):
                    raise ValueError(f"TRT-LLM raw evidence line {line_number} is not canonical JSON")
                if envelope["schema_name"] != _RAW_SCHEMA_NAME:
                    raise ValueError(f"TRT-LLM raw evidence line {line_number} has an unsupported schema")
                schema_version = envelope["schema_version"]
                if (
                    not isinstance(schema_version, int)
                    or isinstance(schema_version, bool)
                    or schema_version != _RAW_SCHEMA_VERSION
                ):
                    raise ValueError(f"TRT-LLM raw evidence line {line_number} has an unsupported schema")
                if (
                    envelope["manifest_sha256"] != self._manifest.sha256
                    or envelope["campaign_id"] != self._manifest.campaign_id
                ):
                    raise ValueError(f"TRT-LLM raw evidence line {line_number} has the wrong manifest identity")
                point_id = envelope["point_id"]
                coordinate = coordinates_by_id.get(point_id)
                if coordinate is None or envelope["coordinate"] != coordinate.to_dict():
                    raise ValueError(f"TRT-LLM raw evidence line {line_number} has the wrong point identity")
                attempt_id = envelope["attempt_id"]
                if not isinstance(attempt_id, str) or not attempt_id.strip():
                    raise ValueError(f"TRT-LLM raw evidence line {line_number} has an invalid attempt_id")
                record_kind = envelope["record_kind"]
                if not isinstance(record_kind, str) or not record_kind.strip():
                    raise ValueError(f"TRT-LLM raw evidence line {line_number} has an invalid record_kind")

    def append(
        self,
        *,
        coordinate: TrtllmCoordinate,
        attempt_id: str,
        record_kind: str,
        payload: object,
    ) -> object:
        envelope = {
            "schema_name": _RAW_SCHEMA_NAME,
            "schema_version": _RAW_SCHEMA_VERSION,
            "manifest_sha256": self._manifest.sha256,
            "campaign_id": self._manifest.campaign_id,
            "point_id": coordinate.point_id,
            "coordinate": coordinate.to_dict(),
            "attempt_id": attempt_id,
            "record_kind": record_kind,
            "payload": payload,
        }
        serialized = _canonical_json(envelope)
        detached_envelope = json.loads(serialized)
        with self._path.open("ab") as evidence_file:
            evidence_file.write(serialized.encode("utf-8") + b"\n")
            evidence_file.flush()
            os.fsync(evidence_file.fileno())
        if not self._materialized:
            directory_fd = os.open(self._path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._materialized = True
        return detached_envelope["payload"]


class TrtllmEvidenceRecorder:
    """Append raw evidence already bound to one campaign coordinate."""

    def __init__(
        self,
        evidence_log: _RawEvidenceLog,
        coordinate: TrtllmCoordinate,
        attempt_id: str,
    ) -> None:
        self._evidence_log = evidence_log
        self._coordinate = coordinate
        self._attempt_id = attempt_id

    def append(self, record_kind: str, payload: object) -> None:
        if not isinstance(record_kind, str):
            raise TypeError(f"record_kind must be a string, got {record_kind!r}")
        if not record_kind.strip():
            raise ValueError("record_kind must not be empty")
        if record_kind in _CAMPAIGN_RECORD_KINDS:
            raise ValueError(f"record_kind is reserved for the campaign loop: {record_kind}")
        self._evidence_log.append(
            coordinate=self._coordinate,
            attempt_id=self._attempt_id,
            record_kind=record_kind,
            payload=payload,
        )


def run_trtllm_campaign(
    ledger: TrtllmLedger,
    *,
    evidence_path: str | Path,
    attempt_id: str,
    measure_point: Callable[[TrtllmCoordinate, TrtllmEvidenceRecorder], RankStats],
    should_stop: Callable[[], bool] = _never_stop,
) -> TrtllmCampaignOutcome:
    """Measure pending coordinates in manifest order and accept each durable candidate."""

    if not isinstance(attempt_id, str):
        raise TypeError(f"attempt_id must be a string, got {attempt_id!r}")
    if not attempt_id.strip():
        raise ValueError("attempt_id must not be empty")
    evidence_log = _RawEvidenceLog(evidence_path, ledger.manifest)
    measured_point_ids: list[str] = []
    while True:
        pending_coordinates = ledger.pending_coordinates
        if not pending_coordinates:
            return TrtllmCampaignOutcome(
                status="completed",
                measured_point_ids=tuple(measured_point_ids),
                remaining_point_ids=(),
            )
        if should_stop():
            return TrtllmCampaignOutcome(
                status="stopped",
                measured_point_ids=tuple(measured_point_ids),
                remaining_point_ids=tuple(point.point_id for point in pending_coordinates),
            )

        coordinate = pending_coordinates[0]
        recorder = TrtllmEvidenceRecorder(evidence_log, coordinate, attempt_id)
        try:
            rank_stats = measure_point(coordinate, recorder)
        except Exception as error:
            evidence_log.append(
                coordinate=coordinate,
                attempt_id=attempt_id,
                record_kind="measurement_error",
                payload={"error_type": type(error).__name__, "message": str(error)},
            )
            raise
        detached_candidate = evidence_log.append(
            coordinate=coordinate,
            attempt_id=attempt_id,
            record_kind="measurement_candidate",
            payload={"rank_stats": rank_stats},
        )
        ledger.accept(
            coordinate,
            attempt_id=attempt_id,
            rank_stats=detached_candidate["rank_stats"],
        )
        measured_point_ids.append(coordinate.point_id)
