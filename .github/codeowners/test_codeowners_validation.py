# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed input validation for the CODEOWNERS toolchain."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import min_approvers
from codeowners_match import compute_resolution


class TestMinApproverInputs:
    @pytest.mark.parametrize(
        "rosters",
        [
            {"runtime": {"alice"}},
            {"runtime": {"alice"}, "docs": set()},
        ],
    )
    def test_incomplete_rosters_fail_closed(self, rosters: dict[str, set[str]]) -> None:
        with pytest.raises(SystemExit, match="incomplete team rosters for: docs"):
            min_approvers.validate_rosters({"runtime", "docs"}, rosters)

    def test_complete_rosters_are_accepted(self) -> None:
        min_approvers.validate_rosters(
            {"runtime", "docs"},
            {"runtime": {"alice"}, "docs": {"bob"}},
        )

    def test_online_roster_rejects_other_org(self) -> None:
        with pytest.raises(SystemExit, match="use a complete offline roster"):
            min_approvers.required_team_slugs({"@acme/runtime", "@other/docs"}, org="acme")

    def test_roster_tsv_reports_malformed_line(self, tmp_path) -> None:
        roster = tmp_path / "rosters.tsv"
        roster.write_text("runtime\talice\textra\n")
        with pytest.raises(SystemExit, match=r"rosters\.tsv:1: expected exactly"):
            min_approvers.load_rosters_tsv(roster)

    def test_corpus_loader_skips_blank_lines(self, tmp_path) -> None:
        corpus = tmp_path / "prs.jsonl"
        corpus.write_text('\n{"n": 1, "files": ["src/a.py"]}\n   \n')
        assert min_approvers.load_corpus(corpus) == [{"n": 1, "files": ["src/a.py"]}]


class TestSpecValidation:
    @staticmethod
    def _spec() -> dict:
        return {
            "meta": {"catch_all": "@acme/contributors"},
            "areas": [
                {
                    "label": "runtime",
                    "github_team": "@acme/runtime",
                    "path_globs": ["src/"],
                },
                {
                    "label": "docs",
                    "github_team": "@acme/docs",
                    "path_globs": ["docs/"],
                },
            ],
            "shared": [],
            "classify": {"keyword_rules": [], "filetype_rules": []},
        }

    @pytest.mark.parametrize("label", ["", "   "])
    def test_rejects_empty_area_label(self, label: str) -> None:
        spec = self._spec()
        spec["areas"][0]["label"] = label
        with pytest.raises(SystemExit, match="label must be non-empty"):
            compute_resolution(spec, [])

    def test_rejects_duplicate_area_labels(self) -> None:
        spec = self._spec()
        spec["areas"][1]["label"] = "runtime"
        with pytest.raises(SystemExit, match="duplicate area label"):
            compute_resolution(spec, [])

    @pytest.mark.parametrize(
        ("target", "value"),
        [("catch_all", "contributors"), ("github_team", "@acme/a @acme/b")],
    )
    def test_rejects_invalid_single_owner(self, target: str, value: str) -> None:
        spec = self._spec()
        if target == "catch_all":
            spec["meta"][target] = value
        else:
            spec["areas"][0][target] = value
        with pytest.raises(SystemExit, match="one GitHub owner token"):
            compute_resolution(spec, [])

    def test_rejects_empty_rule_owners(self) -> None:
        spec = self._spec()
        spec["shared"] = [{"glob": "src/", "owners": []}]
        with pytest.raises(SystemExit, match="owners must be a non-empty list"):
            compute_resolution(spec, ["src/main.py"])

    def test_rejects_unknown_rule_owner(self) -> None:
        spec = self._spec()
        spec["shared"] = [{"glob": "src/", "owners": ["runntime"]}]
        with pytest.raises(SystemExit, match="one GitHub owner token"):
            compute_resolution(spec, ["src/main.py"])

    def test_accepts_known_labels_and_raw_github_owners(self) -> None:
        spec = self._spec()
        spec["shared"] = [{"glob": "src/", "owners": ["runtime", "@acme/security"]}]
        model = compute_resolution(spec, ["src/main.py"])
        assert model.shared == spec["shared"]

    @pytest.mark.parametrize("legacy_config", ["top-level", "filetype"])
    def test_rejects_removed_advisory_config(self, legacy_config: str) -> None:
        spec = self._spec()
        if legacy_config == "top-level":
            spec["advisory"] = [{"glob": "src/", "owners": ["docs"]}]
        else:
            spec["classify"]["filetype_rules"] = [
                {"pattern": "*.md", "coowner": "docs", "advisory": True},
            ]
        with pytest.raises(SystemExit, match="advisory was removed"):
            compute_resolution(spec, ["src/README.md"])

    def test_removed_advisory_cli_flags_are_not_advertised(self) -> None:
        scripts_and_flags = [
            ("emit_codeowners.py", "--advisory-out"),
            ("who_owns.py", "--advisory"),
        ]
        for script, removed_flag in scripts_and_flags:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name(script)), "--help"],
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert removed_flag not in result.stdout

    @pytest.mark.parametrize(
        ("kind", "rule"),
        [
            ("keyword", {"match": "runtime", "area": "missing"}),
            ("keyword", {"match": "runtime", "coowner": "missing"}),
            ("filetype", {"pattern": "*.py", "coowner": "missing"}),
        ],
    )
    def test_rejects_unknown_classification_labels(self, kind: str, rule: dict) -> None:
        spec = self._spec()
        spec["classify"][f"{kind}_rules"] = [rule]
        with pytest.raises(SystemExit, match="must name a declared area"):
            compute_resolution(spec, ["src/main.py"])
