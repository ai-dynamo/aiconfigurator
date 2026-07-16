# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/perf_database/evidence_check.py.

Exercises the pure resolver (`resolve_requirements`) against fixture
`changed_ops.py`-shaped manifests and a fixture `evidence_policy.yaml`: one
test per rule in isolation, combined reasons (asserting the UNION of
requirements, never a single "strictest" pick), the empty-changed shortcut,
fail-closed behavior (unknown reason, malformed policy, unresolved evidence
system, malformed manifest), and determinism (byte-identical reruns).

One integration test runs the real `changed_ops.py` against the real repo
(base=HEAD head=HEAD, which reports nothing changed) and feeds that manifest
through the real `collector/evidence_policy.yaml`, asserting the "no evidence
required" contract end to end.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "perf_database" / "evidence_check.py"
CHANGED_OPS_MODULE_PATH = REPO_ROOT / "tools" / "perf_database" / "changed_ops.py"
REAL_POLICY_PATH = REPO_ROOT / "collector" / "evidence_policy.yaml"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module(MODULE_PATH, "evidence_check")


@pytest.fixture
def changed_ops_mod():
    return _load_module(CHANGED_OPS_MODULE_PATH, "changed_ops")


# --------------------------------------------------------------------------
# fixture policy + manifest builders
# --------------------------------------------------------------------------

POLICY_YAML = """schema_version: 1
thresholds:
  parquet_diff_median_pct: 5
evidence_systems:
  hopper: h200_sxm
  blackwell: b200_sxm
rules:
  pin_version:
    requirement: recollect_or_declared_reuse
  collector_code:
    requirement: before_after_diff
  case_plan:
    requirement: collect_new_cases_only
exceptions_file: evidence_exceptions.yaml
"""


def _manifest_yaml(entries: list[dict]) -> str:
    return yaml.safe_dump({"changed": entries, "unchanged": []}, sort_keys=False)


def _entry(framework="sglang", family="attention", reasons=("pin_version",), tables=None, systems=None) -> dict:
    return {
        "framework": framework,
        "family": family,
        "reasons": list(reasons),
        "tables": tables or ["context_attention_perf", "generation_attention_perf"],
        "systems": systems or ["h200_sxm", "b200_sxm", "gb200"],
        "action": "recollect",
    }


@pytest.fixture
def policy_path(tmp_path):
    path = tmp_path / "evidence_policy.yaml"
    path.write_text(POLICY_YAML)
    return path


def _write_manifest(tmp_path, entries: list[dict]) -> Path:
    path = tmp_path / "changed_ops.yaml"
    path.write_text(_manifest_yaml(entries))
    return path


def _resolve(mod, tmp_path, policy_path, entries: list[dict]) -> list[dict]:
    manifest_path = _write_manifest(tmp_path, entries)
    policy = mod.load_policy(policy_path)
    changed = mod.load_manifest(manifest_path)
    return mod.resolve_requirements(policy, changed)


# --------------------------------------------------------------------------
# each rule alone
# --------------------------------------------------------------------------


class TestRuleAlone:
    def test_pin_version(self, mod, tmp_path, policy_path):
        [item] = _resolve(mod, tmp_path, policy_path, [_entry(reasons=("pin_version",))])
        assert item["framework"] == "sglang"
        assert item["family"] == "attention"
        assert item["reasons"] == ["pin_version"]
        [req] = item["requirements"]
        assert req["type"] == "recollect_or_declared_reuse"
        assert req["tables"] == ["context_attention_perf", "generation_attention_perf"]
        assert req["systems"] == ["b200_sxm", "gb200", "h200_sxm"]
        assert req["evidence_systems"] == ["b200_sxm", "h200_sxm"]
        assert "threshold" not in req

    def test_collector_code(self, mod, tmp_path, policy_path):
        entries = [_entry(family="moe", reasons=("collector_code",), tables=["moe_perf"], systems=["h200_sxm"])]
        [item] = _resolve(mod, tmp_path, policy_path, entries)
        [req] = item["requirements"]
        assert req["type"] == "before_after_diff"
        assert req["tables"] == ["moe_perf"]
        assert req["systems"] == ["h200_sxm"]
        assert req["evidence_systems"] == ["b200_sxm", "h200_sxm"]
        assert req["threshold"] == 5

    def test_case_plan(self, mod, tmp_path, policy_path):
        entries = [
            _entry(
                framework="trtllm",
                family="gemm",
                reasons=("case_plan",),
                tables=["gemm_perf"],
                systems=["h200_sxm", "b200_sxm"],
            )
        ]
        [item] = _resolve(mod, tmp_path, policy_path, entries)
        [req] = item["requirements"]
        assert req["type"] == "collect_new_cases_only"
        assert req["tables"] == ["gemm_perf"]
        assert req["systems"] == ["b200_sxm", "h200_sxm"]
        assert req["evidence_systems"] == []
        assert "threshold" not in req


# --------------------------------------------------------------------------
# combined reasons: union, not the "strictest" single pick
# --------------------------------------------------------------------------


class TestCombinedReasons:
    def test_union_of_requirements_is_emitted(self, mod, tmp_path, policy_path):
        entries = [_entry(reasons=("case_plan", "pin_version", "collector_code"))]  # deliberately out of order
        [item] = _resolve(mod, tmp_path, policy_path, entries)
        # canonical order regardless of input order
        assert item["reasons"] == ["pin_version", "collector_code", "case_plan"]
        types = [req["type"] for req in item["requirements"]]
        assert types == ["recollect_or_declared_reuse", "before_after_diff", "collect_new_cases_only"]
        # every reason's requirement is fully present — none dropped or merged
        assert len(item["requirements"]) == 3


# --------------------------------------------------------------------------
# empty changed list
# --------------------------------------------------------------------------


class TestEmptyChanged:
    def test_resolve_returns_empty_list(self, mod, tmp_path, policy_path):
        assert _resolve(mod, tmp_path, policy_path, []) == []

    def test_main_prints_no_evidence_required_and_exits_zero(self, mod, tmp_path, policy_path, capsys):
        manifest_path = _write_manifest(tmp_path, [])
        rc = mod.main(["--manifest", str(manifest_path), "--policy", str(policy_path)])
        assert rc == mod.EXIT_OK == 0
        captured = capsys.readouterr()
        assert "no evidence required" in captured.err
        doc = yaml.safe_load(captured.out)
        assert doc == {"requirements": []}


# --------------------------------------------------------------------------
# fail-closed: unknown reason
# --------------------------------------------------------------------------


class TestUnknownReason:
    def test_load_manifest_raises(self, mod, tmp_path):
        entries = [_entry(reasons=("not_a_real_reason",))]
        manifest_path = _write_manifest(tmp_path, entries)
        with pytest.raises(mod.EvidenceManifestError, match="unknown reason"):
            mod.load_manifest(manifest_path)

    def test_main_exits_one_with_loud_message(self, mod, tmp_path, policy_path, capsys):
        entries = [_entry(reasons=("not_a_real_reason",))]
        manifest_path = _write_manifest(tmp_path, entries)
        rc = mod.main(["--manifest", str(manifest_path), "--policy", str(policy_path)])
        assert rc == mod.EXIT_ERROR == 1
        assert "unknown reason" in capsys.readouterr().err


# --------------------------------------------------------------------------
# fail-closed: malformed policy
# --------------------------------------------------------------------------


class TestMalformedPolicy:
    @pytest.mark.parametrize(
        "broken_yaml",
        [
            "schema_version: 2\nthresholds: {parquet_diff_median_pct: 5}\nevidence_systems: {hopper: h200_sxm}\n"
            "rules: {pin_version: {requirement: r}, collector_code: {requirement: r}, case_plan: {requirement: r}}\n"
            "exceptions_file: e.yaml\n",  # wrong schema_version
            "schema_version: 1\nevidence_systems: {hopper: h200_sxm}\n"
            "rules: {pin_version: {requirement: r}, collector_code: {requirement: r}, case_plan: {requirement: r}}\n"
            "exceptions_file: e.yaml\n",  # missing thresholds
            "schema_version: 1\nthresholds: {parquet_diff_median_pct: 5}\nevidence_systems: {hopper: h200_sxm}\n"
            "rules: {pin_version: {requirement: r}, collector_code: {requirement: r}}\n"  # missing case_plan rule
            "exceptions_file: e.yaml\n",
            "schema_version: 1\nthresholds: {parquet_diff_median_pct: 5}\nevidence_systems: {hopper: h200_sxm}\n"
            "rules: {pin_version: {requirement: r}, collector_code: {requirement: r}, case_plan: {requirement: r}}\n",
            # missing exceptions_file
            "not: a policy at all\n",
        ],
        ids=[
            "wrong-schema-version",
            "missing-thresholds",
            "missing-case-plan-rule",
            "missing-exceptions-file",
            "not-a-policy",
        ],
    )
    def test_load_policy_raises(self, mod, tmp_path, broken_yaml):
        path = tmp_path / "evidence_policy.yaml"
        path.write_text(broken_yaml)
        with pytest.raises(mod.EvidencePolicyError):
            mod.load_policy(path)

    def test_main_exits_one_with_loud_message(self, mod, tmp_path, capsys):
        policy_path = tmp_path / "evidence_policy.yaml"
        policy_path.write_text("not: a policy at all\n")
        manifest_path = _write_manifest(tmp_path, [])
        rc = mod.main(["--manifest", str(manifest_path), "--policy", str(policy_path)])
        assert rc == mod.EXIT_ERROR == 1
        assert "evidence policy" in capsys.readouterr().err


# --------------------------------------------------------------------------
# fail-closed: unresolved evidence_system
# --------------------------------------------------------------------------


class TestUnresolvedEvidenceSystem:
    @pytest.mark.parametrize(
        "evidence_systems_block",
        ["evidence_systems: {}\n", "evidence_systems: {hopper: ''}\n", "evidence_systems: {hopper: null}\n"],
        ids=["empty-mapping", "blank-value", "null-value"],
    )
    def test_load_policy_raises(self, mod, tmp_path, evidence_systems_block):
        policy_yaml = (
            "schema_version: 1\nthresholds: {parquet_diff_median_pct: 5}\n"
            + evidence_systems_block
            + "rules: {pin_version: {requirement: r}, collector_code: {requirement: r}, case_plan: {requirement: r}}\n"
            "exceptions_file: e.yaml\n"
        )
        path = tmp_path / "evidence_policy.yaml"
        path.write_text(policy_yaml)
        with pytest.raises(mod.EvidencePolicyError, match="unresolved evidence_system"):
            mod.load_policy(path)


# --------------------------------------------------------------------------
# fail-closed: malformed manifest
# --------------------------------------------------------------------------


class TestMalformedManifest:
    @pytest.mark.parametrize(
        "raw_manifest",
        ["not: a manifest\n", "changed: not-a-list\n", "changed:\n  - framework: sglang\n"],  # entry missing fields
        ids=["missing-changed-key", "changed-not-a-list", "entry-missing-fields"],
    )
    def test_load_manifest_raises(self, mod, tmp_path, raw_manifest):
        path = tmp_path / "changed_ops.yaml"
        path.write_text(raw_manifest)
        with pytest.raises(mod.EvidenceManifestError):
            mod.load_manifest(path)

    def test_missing_manifest_file_raises(self, mod, tmp_path):
        with pytest.raises(mod.EvidenceManifestError, match="not found"):
            mod.load_manifest(tmp_path / "does-not-exist.yaml")


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


class TestDeterminism:
    def test_resolve_and_render_are_byte_identical_across_runs(self, mod, tmp_path, policy_path):
        entries = [
            _entry(framework="vllm", family="gemm", reasons=("case_plan",), tables=["gemm_perf"], systems=["h200_sxm"]),
            _entry(framework="sglang", family="attention", reasons=("pin_version", "collector_code")),
            _entry(
                framework="sglang", family="moe", reasons=("collector_code",), tables=["moe_perf"], systems=["b200_sxm"]
            ),
        ]
        manifest_path = _write_manifest(tmp_path, entries)
        policy = mod.load_policy(policy_path)

        first_items = mod.resolve_requirements(policy, mod.load_manifest(manifest_path))
        second_items = mod.resolve_requirements(policy, mod.load_manifest(manifest_path))
        assert first_items == second_items

        first_report = mod.render_report(first_items)
        second_report = mod.render_report(second_items)
        assert first_report == second_report

        # sorted by (framework, family) regardless of input order
        assert [(item["framework"], item["family"]) for item in first_items] == [
            ("sglang", "attention"),
            ("sglang", "moe"),
            ("vllm", "gemm"),
        ]

    def test_main_out_file_is_byte_identical_across_runs(self, mod, tmp_path, policy_path):
        manifest_path = _write_manifest(tmp_path, [_entry()])
        out1 = tmp_path / "out1.yaml"
        out2 = tmp_path / "out2.yaml"
        rc1 = mod.main(["--manifest", str(manifest_path), "--policy", str(policy_path), "--out", str(out1)])
        rc2 = mod.main(["--manifest", str(manifest_path), "--policy", str(policy_path), "--out", str(out2)])
        assert rc1 == rc2 == 0
        assert out1.read_bytes() == out2.read_bytes()


# --------------------------------------------------------------------------
# integration: real changed_ops against the real repo (base=HEAD head=HEAD)
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_real_repo_no_change_means_no_evidence_required(mod, changed_ops_mod, tmp_path, capsys):
    changed, _unchanged = changed_ops_mod.compute_changed_ops(REPO_ROOT, "HEAD", "HEAD")
    assert changed == []

    manifest_path = tmp_path / "changed_ops.yaml"
    manifest_path.write_text(changed_ops_mod.render_report(changed, _unchanged))

    rc = mod.main(["--manifest", str(manifest_path), "--policy", str(REAL_POLICY_PATH)])
    assert rc == 0
    assert "no evidence required" in capsys.readouterr().err
