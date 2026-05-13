# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inject a CycloneDX SBOM into an existing wheel per PEP 770.

Reads a wheel and a CycloneDX JSON document, writes a new wheel that:
- includes the SBOM at ``<dist-info>/sboms/cyclonedx.json``,
- has ``Sbom-File: sboms/cyclonedx.json`` added to ``<dist-info>/METADATA``,
- has a regenerated ``<dist-info>/RECORD`` per PEP 376 (sha256 / urlsafe
  base64 / no padding).

The operation is idempotent: re-running on a wheel that already has the
header and the SBOM replaces the SBOM bytes and leaves METADATA unchanged.

Usage:
    python tools/wheel_build/inject_sbom.py <wheel> --sbom <cyclonedx.json> [--out DIR]
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from email import message_from_bytes
from email.generator import BytesGenerator
from pathlib import Path

SBOM_ARCNAME = "sboms/cyclonedx.json"  # relative to .dist-info/
SBOM_HEADER = "Sbom-File"
RECORD_NAME = "RECORD"
METADATA_NAME = "METADATA"


class InjectError(RuntimeError):
    """Raised when SBOM injection fails for a recoverable reason."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inject a CycloneDX SBOM into a wheel per PEP 770.",
    )
    parser.add_argument("wheel", help="Path to the wheel to modify.")
    parser.add_argument(
        "--sbom",
        required=True,
        help="Path to the CycloneDX JSON document to embed.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for the rewritten wheel (default: alongside the input).",
    )
    args = parser.parse_args(argv)

    wheel_path = Path(args.wheel)
    sbom_path = Path(args.sbom)
    out_dir = Path(args.out) if args.out else wheel_path.parent

    try:
        inject(wheel_path, sbom_path, out_dir)
    except InjectError as exc:
        print(f"inject_sbom: {exc}", file=sys.stderr)
        return 1
    return 0


def inject(wheel_path: Path, sbom_path: Path, out_dir: Path) -> Path:
    """Inject ``sbom_path`` into ``wheel_path``; return the output wheel path."""
    if not wheel_path.is_file():
        raise InjectError(f"wheel not found: {wheel_path}")
    if not sbom_path.is_file():
        raise InjectError(f"SBOM not found: {sbom_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sbom_bytes = sbom_path.read_bytes()
    _validate_cyclonedx(sbom_bytes, sbom_path)

    with zipfile.ZipFile(wheel_path, "r") as src:
        dist_info = _find_dist_info(src.namelist(), wheel_path)
        sbom_arc = f"{dist_info}/{SBOM_ARCNAME}"
        metadata_arc = f"{dist_info}/{METADATA_NAME}"
        record_arc = f"{dist_info}/{RECORD_NAME}"

        original_metadata = src.read(metadata_arc)
        updated_metadata = _ensure_sbom_header(original_metadata, SBOM_ARCNAME)

        new_entries: dict[str, bytes] = {
            sbom_arc: sbom_bytes,
            metadata_arc: updated_metadata,
        }
        ordered_names = _ordered_output_names(src.namelist(), new_entries, record_arc)

        out_path = out_dir / wheel_path.name
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=wheel_path.stem + ".",
            suffix=".whl.tmp",
            dir=out_dir,
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as dst:
                record_entries: list[tuple[str, str, str]] = []
                for arc in ordered_names:
                    if arc == record_arc:
                        continue
                    if arc in new_entries:
                        data = new_entries[arc]
                    else:
                        data = src.read(arc)
                    dst.writestr(arc, data)
                    record_entries.append((arc, *_record_hash(data)))
                record_entries.append((record_arc, "", ""))
                dst.writestr(record_arc, _format_record(record_entries))
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        os.replace(tmp_path, out_path)
        return out_path


def _validate_cyclonedx(sbom_bytes: bytes, sbom_path: Path) -> None:
    try:
        data = json.loads(sbom_bytes)
    except json.JSONDecodeError as exc:
        raise InjectError(f"{sbom_path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise InjectError(f"{sbom_path}: top-level JSON value is not an object")
    if data.get("bomFormat") != "CycloneDX":
        raise InjectError(f"{sbom_path}: bomFormat is not 'CycloneDX'")
    if not data.get("specVersion"):
        raise InjectError(f"{sbom_path}: specVersion is missing or empty")


def _find_dist_info(names: list[str], wheel_path: Path) -> str:
    pattern = re.compile(r"^([^/]+\.dist-info)/")
    candidates = {m.group(1) for n in names if (m := pattern.match(n))}
    if not candidates:
        raise InjectError(f"{wheel_path}: no .dist-info directory found")
    if len(candidates) > 1:
        raise InjectError(f"{wheel_path}: multiple .dist-info dirs: {sorted(candidates)}")
    return candidates.pop()


def _ensure_sbom_header(metadata_bytes: bytes, sbom_relpath: str) -> bytes:
    msg = message_from_bytes(metadata_bytes)
    if sbom_relpath in (msg.get_all(SBOM_HEADER) or []):
        return metadata_bytes
    msg.add_header(SBOM_HEADER, sbom_relpath)
    buf = io.BytesIO()
    BytesGenerator(buf, mangle_from_=False, maxheaderlen=0).flatten(msg)
    return buf.getvalue()


def _ordered_output_names(
    original: list[str],
    new_entries: dict[str, bytes],
    record_arc: str,
) -> list[str]:
    """Preserve original order; append any new arcnames not already present.

    RECORD always lands last so it can hash everything that came before it.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for name in original:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    for name in new_entries:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    if record_arc in ordered:
        ordered = [n for n in ordered if n != record_arc] + [record_arc]
    return ordered


def _record_hash(data: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}", str(len(data))


def _format_record(entries: list[tuple[str, str, str]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    for entry in entries:
        writer.writerow(entry)
    return buf.getvalue().encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
