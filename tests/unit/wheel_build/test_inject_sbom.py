# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from wheel_build import inject_sbom

pytestmark = pytest.mark.unit

WHEEL_NAME = "demo-1.2.3-py3-none-any.whl"
DIST_INFO = "demo-1.2.3.dist-info"
METADATA_BODY = (
    b"Metadata-Version: 2.1\nName: demo\nVersion: 1.2.3\nSummary: test wheel\n\nLong description goes here.\n"
)
WHEEL_BODY = b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
PACKAGE_FILE = "demo/__init__.py"
PACKAGE_BODY = b"VERSION = '1.2.3'\n"


def _record_entry(arc: str, data: bytes) -> tuple[str, str, str]:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return arc, f"sha256={encoded}", str(len(data))


def _format_record(entries: list[tuple[str, str, str]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for entry in entries:
        writer.writerow(entry)
    return buf.getvalue().encode("utf-8")


def _build_minimal_wheel(path: Path) -> None:
    record_entries = [
        _record_entry(PACKAGE_FILE, PACKAGE_BODY),
        _record_entry(f"{DIST_INFO}/METADATA", METADATA_BODY),
        _record_entry(f"{DIST_INFO}/WHEEL", WHEEL_BODY),
        (f"{DIST_INFO}/RECORD", "", ""),
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr(PACKAGE_FILE, PACKAGE_BODY)
        wheel.writestr(f"{DIST_INFO}/METADATA", METADATA_BODY)
        wheel.writestr(f"{DIST_INFO}/WHEEL", WHEEL_BODY)
        wheel.writestr(f"{DIST_INFO}/RECORD", _format_record(record_entries))


def _valid_sbom() -> bytes:
    return json.dumps(
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "components": [
                {"type": "library", "name": "serde", "version": "1.0.0"},
            ],
        }
    ).encode("utf-8")


@pytest.fixture
def wheel_and_sbom(tmp_path: Path) -> tuple[Path, Path]:
    wheel = tmp_path / WHEEL_NAME
    _build_minimal_wheel(wheel)
    sbom = tmp_path / "bom.cdx.json"
    sbom.write_bytes(_valid_sbom())
    return wheel, sbom


def _read_arc(wheel: Path, arc: str) -> bytes:
    with zipfile.ZipFile(wheel, "r") as zf:
        return zf.read(arc)


def _record_rows(wheel: Path) -> list[list[str]]:
    record = _read_arc(wheel, f"{DIST_INFO}/RECORD").decode("utf-8")
    return list(csv.reader(record.splitlines()))


def test_injects_sbom_into_dist_info(wheel_and_sbom: tuple[Path, Path], tmp_path: Path) -> None:
    wheel, sbom = wheel_and_sbom
    out = inject_sbom.inject(wheel, sbom, tmp_path)

    assert out == tmp_path / WHEEL_NAME
    sbom_arc = f"{DIST_INFO}/sboms/cyclonedx.json"
    assert _read_arc(out, sbom_arc) == _valid_sbom()

    metadata = _read_arc(out, f"{DIST_INFO}/METADATA").decode("utf-8")
    sbom_lines = [line for line in metadata.splitlines() if line.startswith("Sbom-File:")]
    assert sbom_lines == ["Sbom-File: sboms/cyclonedx.json"]


def test_record_hashes_match_file_bytes(wheel_and_sbom: tuple[Path, Path], tmp_path: Path) -> None:
    wheel, sbom = wheel_and_sbom
    out = inject_sbom.inject(wheel, sbom, tmp_path)

    rows = _record_rows(out)
    record_self = f"{DIST_INFO}/RECORD"
    by_name = {row[0]: row for row in rows}

    assert by_name[record_self] == [record_self, "", ""]
    for arc, hash_field, size_field in rows:
        if arc == record_self:
            continue
        data = _read_arc(out, arc)
        expected_digest = hashlib.sha256(data).digest()
        expected_b64 = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
        assert hash_field == f"sha256={expected_b64}", arc
        assert size_field == str(len(data)), arc


def test_record_lists_the_injected_sbom(wheel_and_sbom: tuple[Path, Path], tmp_path: Path) -> None:
    wheel, sbom = wheel_and_sbom
    out = inject_sbom.inject(wheel, sbom, tmp_path)

    rows = _record_rows(out)
    arcs = [row[0] for row in rows]
    assert f"{DIST_INFO}/sboms/cyclonedx.json" in arcs
    assert arcs[-1] == f"{DIST_INFO}/RECORD"


def test_idempotent_on_second_run(wheel_and_sbom: tuple[Path, Path], tmp_path: Path) -> None:
    wheel, sbom = wheel_and_sbom
    first = inject_sbom.inject(wheel, sbom, tmp_path)
    metadata_after_first = _read_arc(first, f"{DIST_INFO}/METADATA")

    second = inject_sbom.inject(first, sbom, tmp_path)
    metadata_after_second = _read_arc(second, f"{DIST_INFO}/METADATA")

    assert metadata_after_first == metadata_after_second
    assert metadata_after_second.decode("utf-8").count("Sbom-File:") == 1
    rows = _record_rows(second)
    sbom_rows = [row for row in rows if row[0] == f"{DIST_INFO}/sboms/cyclonedx.json"]
    assert len(sbom_rows) == 1


def test_rejects_non_cyclonedx_json(tmp_path: Path) -> None:
    wheel = tmp_path / WHEEL_NAME
    _build_minimal_wheel(wheel)
    bad_sbom = tmp_path / "not-cyclonedx.json"
    bad_sbom.write_text(json.dumps({"bomFormat": "SPDX", "specVersion": "2.3"}))

    with pytest.raises(inject_sbom.InjectError, match="bomFormat"):
        inject_sbom.inject(wheel, bad_sbom, tmp_path / "out")


def test_rejects_malformed_json(tmp_path: Path) -> None:
    wheel = tmp_path / WHEEL_NAME
    _build_minimal_wheel(wheel)
    bad_sbom = tmp_path / "bad.json"
    bad_sbom.write_text("{not valid json")

    with pytest.raises(inject_sbom.InjectError, match="not valid JSON"):
        inject_sbom.inject(wheel, bad_sbom, tmp_path / "out")


def test_rejects_missing_specversion(tmp_path: Path) -> None:
    wheel = tmp_path / WHEEL_NAME
    _build_minimal_wheel(wheel)
    bad_sbom = tmp_path / "missing-specversion.json"
    bad_sbom.write_text(json.dumps({"bomFormat": "CycloneDX"}))

    with pytest.raises(inject_sbom.InjectError, match="specVersion"):
        inject_sbom.inject(wheel, bad_sbom, tmp_path / "out")
