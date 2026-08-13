# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

VERIFY_RELEASE_WHEELS = Path(__file__).resolve().parents[3] / "tools" / "verify_release_wheels.py"


@pytest.fixture
def verifier():
    spec = importlib.util.spec_from_file_location("verify_release_wheels", VERIFY_RELEASE_WHEELS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spica_scan_covers_all_archive_member_types(verifier):
    names = {
        "aiconfigurator/__init__.py",
        "spica/",
        "spica/native.so",
        "spica/data/model.bin",
    }

    assert verifier._spica_entries(names) == [
        "spica/",
        "spica/data/model.bin",
        "spica/native.so",
    ]


def test_release_verifier_rejects_stale_spica_archive_member(verifier, monkeypatch, tmp_path):
    wheel = tmp_path / "aiconfigurator-1.2.0-py3-none-any.whl"
    required = {
        "aiconfigurator/__init__.py",
        "aiconfigurator/cli/main.py",
        "aiconfigurator/generator/api.py",
        "aiconfigurator/logging_utils.py",
        "aiconfigurator/sdk/_compat.py",
        "aiconfigurator/sdk/config_adapter/__init__.py",
        "aiconfigurator/sdk/config_adapter/schemas/estimate-request-v1.schema.json",
        "aiconfigurator/sdk/engine.py",
        "aiconfigurator/sdk/task_v2.py",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in required:
            archive.writestr(name, "")
        archive.writestr(
            "aiconfigurator-1.2.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: aiconfigurator\nVersion: 1.2.0\n",
        )
        archive.writestr("spica/data/model.bin", b"stale")

    # main() resolves both wheel paths before validating the upper wheel. The
    # core placeholder is never opened because stale Spica content fails first.
    (tmp_path / "aiconfigurator_core-1.2.0-py3-none-any.whl").touch()
    monkeypatch.setattr(verifier, "_source_payloads", lambda: (set(), set()))
    monkeypatch.setattr(sys, "argv", ["verify_release_wheels.py", str(tmp_path)])

    with pytest.raises(RuntimeError, match=r"removed Spica payload.*spica/data/model\.bin"):
        verifier.main()


def test_infra_scan_rejects_gap_skill_tool_dataset_report_and_web_payloads(verifier):
    names = {
        ".agents/skills/adapt-server-config/SKILL.md",
        "aiconfigurator/datasets/private.json",
        "aiconfigurator/gap_analysis/pipeline.py",
        "aiconfigurator/reports/gap.html",
        "aiconfigurator/skills/helper.py",
        "aiconfigurator/tools/import.py",
        "datasets/source.csv",
        "reports/generated.json",
        "tools/private_helper.py",
        "vendor/datasets/source.csv",
        "vendor/reports/generated.json",
        "vendor/web/dashboard.js",
        "vendor/webapp/dashboard.js",
        "webapp/dashboard.js",
    }

    assert verifier._infra_entries(names) == sorted(names)


def test_config_adapter_readme_remains_repository_only(verifier):
    upper, _ = verifier._source_payloads()

    assert "aiconfigurator/sdk/config_adapter/README.md" not in upper
    assert "aiconfigurator/sdk/config_adapter/schemas/estimate-request-v1.schema.json" in upper


def test_rust_crate_package_rejects_upper_payload(verifier, monkeypatch):
    result = subprocess.CompletedProcess(
        args=["cargo"],
        returncode=0,
        stdout="Cargo.toml\nsrc/lib.rs\naiconfigurator/sdk/config_adapter/api.py\n",
        stderr="",
    )
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: result)

    with pytest.raises(RuntimeError, match="config_adapter"):
        verifier._verify_rust_crate_package()


@pytest.mark.parametrize("root", ["datasets", "reports"])
def test_rust_crate_package_rejects_infra_roots(verifier, monkeypatch, root):
    result = subprocess.CompletedProcess(
        args=["cargo"],
        returncode=0,
        stdout=f"Cargo.toml\nsrc/lib.rs\n{root}/private.json\n",
        stderr="",
    )
    monkeypatch.setattr(verifier.subprocess, "run", lambda *args, **kwargs: result)

    with pytest.raises(RuntimeError, match=root):
        verifier._verify_rust_crate_package()
