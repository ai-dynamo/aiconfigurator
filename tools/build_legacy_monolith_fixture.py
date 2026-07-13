# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a minimal legacy wheel for deterministic package-migration tests.

AIConfigurator 0.9 owned both upper-layer and core-layer paths in the
``aiconfigurator`` distribution. The 0.10 split transfers the latter paths to
``aiconfigurator-core``. This fixture deliberately records representative paths
from both layers so the supported uninstall-before-install migration can be
tested without downloading the historical release artifact.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

VERSION = "0.9.0"
DIST_INFO = f"aiconfigurator-{VERSION}.dist-info"
WHEEL_NAME = f"aiconfigurator-{VERSION}-py3-none-any.whl"

PAYLOAD_SOURCES = {
    "aiconfigurator/__init__.py": "src/aiconfigurator/__init__.py",
    "aiconfigurator/sdk/common.py": "src/aiconfigurator/sdk/common.py",
    "aiconfigurator/systems/h100_sxm.yaml": "aic-core/src/aiconfigurator_core/systems/h100_sxm.yaml",
    "aiconfigurator/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json": (
        "aic-core/src/aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json"
    ),
    "aiconfigurator_core/__init__.py": "aic-core/src/aiconfigurator_core/__init__.py",
}

METADATA = f"""Metadata-Version: 2.4
Name: aiconfigurator
Version: {VERSION}
Summary: Deterministic legacy monolith fixture for package migration tests
Requires-Python: >=3.10
"""

WHEEL = """Wheel-Version: 1.0
Generator: aiconfigurator legacy migration fixture
Root-Is-Purelib: true
Tag: py3-none-any
"""


def _record_row(name: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return name, f"sha256={digest}", str(len(data))


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def build_fixture(repo_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = output_dir / WHEEL_NAME
    members = {
        wheel_name: (repo_root / source_name).read_bytes() for wheel_name, source_name in PAYLOAD_SOURCES.items()
    }
    members[f"{DIST_INFO}/METADATA"] = METADATA.encode()
    members[f"{DIST_INFO}/WHEEL"] = WHEEL.encode()

    record_rows = [_record_row(name, data) for name, data in sorted(members.items())]
    record_rows.append((f"{DIST_INFO}/RECORD", "", ""))
    record_buffer = io.StringIO(newline="")
    csv.writer(record_buffer, lineterminator="\n").writerows(record_rows)
    members[f"{DIST_INFO}/RECORD"] = record_buffer.getvalue().encode()

    with zipfile.ZipFile(wheel_path, "w") as archive:
        for name, data in sorted(members.items()):
            _write_member(archive, name, data)
    return wheel_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(build_fixture(args.repo_root.resolve(), args.output_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
