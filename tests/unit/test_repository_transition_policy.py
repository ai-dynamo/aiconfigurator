# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SUCCESSOR_URL = "https://github.com/ai-dynamo/aisimulate"
RELEASES_URL = f"{SUCCESSOR_URL}/releases"
MIGRATION_URL = "https://github.com/ai-dynamo/aiconfigurator/blob/main/docs/aisimulate_migration.md"
FIX_SCOPE = "bug, security, and migration-blocking fixes"
pytestmark = pytest.mark.unit


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_contributor_entry_points_enforce_the_transition_scope() -> None:
    for path in (
        "README.md",
        "CONTRIBUTING.md",
        ".github/pull_request_template.md",
        ".github/ISSUE_TEMPLATE/support_matrix_request.md",
        "docs/index.html",
    ):
        content = " ".join(line.removeprefix("> ").strip() for line in _read(path).splitlines())
        assert SUCCESSOR_URL in content, path
        assert FIX_SCOPE in content, path

    issue_config = yaml.safe_load(_read(".github/ISSUE_TEMPLATE/config.yml"))
    assert issue_config["blank_issues_enabled"] is False
    contact_links = {link["name"]: link["url"] for link in issue_config["contact_links"]}
    assert SUCCESSOR_URL in contact_links["AISimulate features and new coverage"]
    assert contact_links["Report a security vulnerability"].endswith("/security/policy")

    maintenance_form = yaml.safe_load(_read(".github/ISSUE_TEMPLATE/aic_maintenance_report.yml"))
    assert SUCCESSOR_URL in maintenance_form["body"][0]["attributes"]["value"]
    scope = next(field for field in maintenance_form["body"] if field.get("id") == "maintenance_scope")
    assert scope["attributes"]["options"] == [
        "Regression in previously supported AIC behavior",
        "Blocker to migration from AIC to AISimulate",
    ]


def test_migration_install_is_gated_on_stable_artifact_publication() -> None:
    migration_guide = _read("docs/aisimulate_migration.md")
    assert "only after stable" in migration_guide
    assert "development releases do not satisfy this publication gate" in migration_guide
    assert RELEASES_URL in migration_guide

    readme = " ".join(_read("README.md").splitlines())
    assert (
        "After the stable AISimulate 0.12 artifacts are published on PyPI and the corresponding release "
        f"is announced in [AISimulate releases]({RELEASES_URL}), migrate the installed distribution"
    ) in readme


def test_python_package_metadata_names_the_successor() -> None:
    for path in ("pyproject.toml", "aic-core/pyproject.toml"):
        metadata = tomllib.loads(_read(path))["project"]
        assert "active development has moved to AISimulate" in metadata["description"]
        assert metadata["urls"]["AISimulate"] == SUCCESSOR_URL
        assert metadata["urls"]["Migration"] == MIGRATION_URL


def test_rust_crate_metadata_and_readme_name_the_successor() -> None:
    crate_root = "aic-core/rust/aiconfigurator-core"
    package = tomllib.loads(_read(f"{crate_root}/Cargo.toml"))["package"]
    assert "active development has moved to AISimulate" in package["description"]
    assert package["homepage"] == SUCCESSOR_URL
    assert SUCCESSOR_URL in _read(f"{crate_root}/README.md")


def test_cutover_checklist_covers_non_code_repository_actions() -> None:
    checklist = _read(".github/AIC_MIGRATION_CHECKLIST.md")
    for required_action in (
        "Lead the AIC 0.12 release notes",
        "Update the AIC GitHub description",
        "Pin transition DEP issue",
        "Archive the AIC repository",
    ):
        assert required_action in checklist
