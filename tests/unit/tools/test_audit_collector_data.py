# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/perf_database/audit_collector_data.py.

Builds synthetic family-layout trees (`<system>/<family>/<backend>/<version>/`)
under `tmp_path` and exercises each of the six fail-closed rules (R1-R6,
design §6/§6.5/§8) both green (a fully compliant tree) and red (one named
offender per rule). Uses the REAL op catalog (`collector/op_backend_catalog.yaml`)
for family-placement checks -- same convention as
`tests/unit/tools/test_backend_facts.py` -- so table/family names below
(`gemm_perf`/`gemm`, `context_attention_perf`/`attention`,
`custom_allreduce_perf`/`comm`) are real catalog entries, not fixtures.

Also runs the audit against the REAL data tree
(aic-core/src/aiconfigurator_core/systems/data) -- this is the CI-riding
gate: it must pass on the tree as committed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "perf_database" / "audit_collector_data.py"
REAL_CATALOG = REPO_ROOT / "collector" / "op_backend_catalog.yaml"
REAL_DATA_ROOT = REPO_ROOT / "aic-core" / "src" / "aiconfigurator_core" / "systems" / "data"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("audit_collector_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _touch(root: Path, rel: str, content: bytes = b"PAR1stub") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _collection_meta(tables: dict[str, str], *, legacy: bool = False) -> str:
    lines = ["schema_version: 1"]
    if legacy:
        lines.append("provenance: legacy")
    lines.append("runtime:")
    lines.append("  framework: sglang")
    lines.append('  version: "0.5.14"')
    lines.append("tables:")
    for table, status in tables.items():
        lines.append(f"  {table}:")
        lines.append(f"    status: {status}")
    return "\n".join(lines) + "\n"


def _reuse_yaml(entries: list[tuple[str, str]]) -> str:
    lines = ["schema_version: 1", "reuse:"]
    for table, from_version in entries:
        lines.append(f"  - table: {table}")
        lines.append(f'    from_version: "{from_version}"')
        lines.append("    reason: unit test declaration")
        lines.append("    approved_by: unit-test")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# fully compliant synthetic tree -- every rule green
# ---------------------------------------------------------------------------


def _build_compliant_tree(root: Path) -> None:
    # gemm/sglang: two versions, both fully covered by collection_meta.yaml
    # (0.5.14 non-legacy "complete", 0.5.12 legacy "partial" -- exercises the
    # legacy-tier-satisfies-coverage clause of R1), plus a declared-reuse-only
    # sibling that borrows from 0.5.14.
    _touch(root, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
    _write(root, "h200_sxm/gemm/sglang/0.5.14/collection_meta.yaml", _collection_meta({"gemm_perf": "complete"}))

    _touch(root, "h200_sxm/gemm/sglang/0.5.12/gemm_perf.parquet")
    _write(
        root,
        "h200_sxm/gemm/sglang/0.5.12/collection_meta.yaml",
        _collection_meta({"gemm_perf": "partial"}, legacy=True),
    )

    _write(
        root,
        "h200_sxm/gemm/sglang/0.5.10/reuse.yaml",
        _reuse_yaml([("gemm_perf", "0.5.14")]),
    )

    # attention/trtllm: single complete version.
    _touch(root, "h200_sxm/attention/trtllm/1.3.0rc10/context_attention_perf.parquet")
    _write(
        root,
        "h200_sxm/attention/trtllm/1.3.0rc10/collection_meta.yaml",
        _collection_meta({"context_attention_perf": "complete"}),
    )

    # comm family: real data, NO reuse.yaml anywhere under it (R3).
    _touch(root, "h200_sxm/comm/nccl/2.23/custom_allreduce_perf.parquet")
    _write(
        root,
        "h200_sxm/comm/nccl/2.23/collection_meta.yaml",
        _collection_meta({"custom_allreduce_perf": "complete"}),
    )


def test_compliant_tree_is_all_green(mod, tmp_path):
    _build_compliant_tree(tmp_path)
    results = mod.run_audit(tmp_path, REAL_CATALOG)
    for rule, failures in results.items():
        assert failures == [], f"{rule} unexpectedly failed: {failures}"


def test_compliant_tree_render_report_ok(mod, tmp_path):
    _build_compliant_tree(tmp_path)
    results = mod.run_audit(tmp_path, REAL_CATALOG)
    report, failed = mod.render_report(results)
    assert failed is False
    assert "audit OK" in report
    for rule in mod.RULES:
        assert f"[OK]   {rule}" in report


# ---------------------------------------------------------------------------
# R1: sidecar coverage
# ---------------------------------------------------------------------------


class TestR1SidecarCoverage:
    def test_missing_sidecar_is_named(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        failures = mod.check_r1_sidecar_coverage(tmp_path, mod.iter_version_dirs(tmp_path))
        assert len(failures) == 1
        assert "h200_sxm/gemm/sglang/0.5.14" in failures[0]
        assert "missing collection_meta.yaml" in failures[0]
        assert "gemm_perf" in failures[0]

    def test_uncovered_table_is_named(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/other_perf.parquet")
        _write(
            tmp_path,
            "h200_sxm/gemm/sglang/0.5.14/collection_meta.yaml",
            _collection_meta({"gemm_perf": "complete"}),
        )
        failures = mod.check_r1_sidecar_coverage(tmp_path, mod.iter_version_dirs(tmp_path))
        assert len(failures) == 1
        assert "other_perf" in failures[0]
        assert "gemm_perf" not in failures[0].split("'")[1::2]  # only the uncovered table is named

    def test_dir_with_no_parquet_needs_no_sidecar(self, mod, tmp_path):
        # A declared-reuse-only dir (no parquet of its own) is out of R1's scope.
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml", _reuse_yaml([("gemm_perf", "0.5.14")]))
        failures = mod.check_r1_sidecar_coverage(tmp_path, mod.iter_version_dirs(tmp_path))
        assert failures == []

    def test_legacy_tier_status_only_entry_satisfies_coverage(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.12/gemm_perf.parquet")
        _write(
            tmp_path,
            "h200_sxm/gemm/sglang/0.5.12/collection_meta.yaml",
            _collection_meta({"gemm_perf": "partial"}, legacy=True),
        )
        failures = mod.check_r1_sidecar_coverage(tmp_path, mod.iter_version_dirs(tmp_path))
        assert failures == []

    def test_malformed_sidecar_is_reported_not_raised(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.14/collection_meta.yaml", "not: a: valid: mapping: [")
        failures = mod.check_r1_sidecar_coverage(tmp_path, mod.iter_version_dirs(tmp_path))
        assert len(failures) == 1
        assert "collection_meta.yaml" in failures[0]


# ---------------------------------------------------------------------------
# R2: reuse validity
# ---------------------------------------------------------------------------


class TestR2ReuseValidity:
    def test_donor_version_does_not_exist(self, mod, tmp_path):
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml", _reuse_yaml([("gemm_perf", "9.9.9")]))
        failures = mod.check_r2_reuse_validity(tmp_path, mod.iter_version_dirs(tmp_path))
        assert len(failures) == 1
        assert "9.9.9" in failures[0]
        assert "does not exist" in failures[0]

    def test_donor_version_exists_but_lacks_real_data(self, mod, tmp_path):
        # Donor dir exists but holds no gemm_perf.parquet of its own.
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.12/other_perf.parquet")
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml", _reuse_yaml([("gemm_perf", "0.5.12")]))
        failures = mod.check_r2_reuse_validity(tmp_path, mod.iter_version_dirs(tmp_path))
        assert len(failures) == 1
        assert "0.5.12" in failures[0]
        assert "no real parquet data" in failures[0]

    def test_donor_is_itself_a_declared_reuse_dir_fails(self, mod, tmp_path):
        # 0.5.10 declares from_version=0.5.12, but 0.5.12 has no real data of
        # its own -- only its own reuse.yaml pointing further to 0.5.14. A
        # single-hop check, per design: chaining through a declaration is not
        # a "real donor".
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.12/reuse.yaml", _reuse_yaml([("gemm_perf", "0.5.14")]))
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml", _reuse_yaml([("gemm_perf", "0.5.12")]))
        failures = mod.check_r2_reuse_validity(tmp_path, mod.iter_version_dirs(tmp_path))
        offending = [f for f in failures if "0.5.10" in f]
        assert len(offending) == 1
        assert "not itself be a declared-reuse-only dir" in offending[0]

    def test_valid_donor_passes(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml", _reuse_yaml([("gemm_perf", "0.5.14")]))
        failures = mod.check_r2_reuse_validity(tmp_path, mod.iter_version_dirs(tmp_path))
        assert failures == []

    def test_malformed_reuse_yaml_is_reported_not_raised(self, mod, tmp_path):
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml", "schema_version: 1\nreuses: []\n")
        failures = mod.check_r2_reuse_validity(tmp_path, mod.iter_version_dirs(tmp_path))
        assert len(failures) == 1
        assert "reuse.yaml" in failures[0]

    def test_never_crosses_backend(self, mod, tmp_path):
        # A same-named version under a DIFFERENT backend is never treated as
        # a donor (reuse.yaml carries no backend field, so donors resolve
        # only within the declaring dir's own <family>/<backend> subtree).
        _touch(tmp_path, "h200_sxm/gemm/trtllm/0.5.14/gemm_perf.parquet")
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml", _reuse_yaml([("gemm_perf", "0.5.14")]))
        failures = mod.check_r2_reuse_validity(tmp_path, mod.iter_version_dirs(tmp_path))
        assert len(failures) == 1
        assert "does not exist" in failures[0]


# ---------------------------------------------------------------------------
# R3: comm exclusion (design §6.5 rule 5)
# ---------------------------------------------------------------------------


class TestR3CommExclusion:
    def test_reuse_yaml_under_comm_is_named(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/comm/nccl/2.23/custom_allreduce_perf.parquet")
        _write(tmp_path, "h200_sxm/comm/nccl/2.19/reuse.yaml", _reuse_yaml([("custom_allreduce_perf", "2.23")]))
        failures = mod.check_r3_comm_exclusion(tmp_path)
        assert len(failures) == 1
        assert "h200_sxm/comm/nccl/2.19/reuse.yaml" in failures[0]

    def test_non_comm_reuse_yaml_is_ignored(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _write(tmp_path, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml", _reuse_yaml([("gemm_perf", "0.5.14")]))
        assert mod.check_r3_comm_exclusion(tmp_path) == []

    def test_multiple_comm_offenders_all_named(self, mod, tmp_path):
        _write(tmp_path, "h200_sxm/comm/nccl/2.19/reuse.yaml", _reuse_yaml([("custom_allreduce_perf", "2.23")]))
        _write(tmp_path, "b200_sxm/comm/sglang/0.5.10/reuse.yaml", _reuse_yaml([("custom_allreduce_perf", "0.5.14")]))
        failures = mod.check_r3_comm_exclusion(tmp_path)
        assert len(failures) == 2


# ---------------------------------------------------------------------------
# R4: family placement
# ---------------------------------------------------------------------------


class TestR4FamilyPlacement:
    def test_table_filed_under_wrong_family_is_named(self, mod, tmp_path):
        # gemm_perf belongs to family 'gemm' per the real catalog, filed under 'moe'.
        _touch(tmp_path, "h200_sxm/moe/sglang/0.5.14/gemm_perf.parquet")
        failures = mod.check_r4_family_placement(
            tmp_path, mod.iter_version_dirs(tmp_path), mod.load_family_map(REAL_CATALOG)
        )
        assert len(failures) == 1
        assert "gemm_perf" in failures[0]
        assert "'gemm'" in failures[0]
        assert "'moe'" in failures[0]

    def test_unmapped_table_is_named(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/totally_unknown_op_perf.parquet")
        failures = mod.check_r4_family_placement(
            tmp_path, mod.iter_version_dirs(tmp_path), mod.load_family_map(REAL_CATALOG)
        )
        assert len(failures) == 1
        assert "totally_unknown_op_perf" in failures[0]
        assert "not mapped to any family" in failures[0]

    def test_correctly_placed_table_passes(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        failures = mod.check_r4_family_placement(
            tmp_path, mod.iter_version_dirs(tmp_path), mod.load_family_map(REAL_CATALOG)
        )
        assert failures == []

    def test_missing_catalog_fails_closed(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        failures = mod.check_r4_family_placement(tmp_path, mod.iter_version_dirs(tmp_path), None)
        assert len(failures) == 1
        assert "op catalog not found" in failures[0]


# ---------------------------------------------------------------------------
# R5: identity (manifest v2 resolution)
# ---------------------------------------------------------------------------


class TestR5Identity:
    def test_broken_catalog_surfaces_as_failure(self, mod, tmp_path):
        bad_catalog = tmp_path / "op_backend_catalog.yaml"
        bad_catalog.write_text("schema_version: 1\nfamilies: []\n", encoding="utf-8")
        failures = mod.check_r5_identity(bad_catalog)
        assert len(failures) == 1
        assert "op catalog" in failures[0]

    def test_real_manifest_and_catalog_resolve_clean(self, mod):
        assert mod.check_r5_identity(REAL_CATALOG) == []


# ---------------------------------------------------------------------------
# R6: no legacy markers
# ---------------------------------------------------------------------------


class TestR6NoLegacyMarkers:
    def test_shared_layer_reuse_txt_is_named(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.10/SHARED_LAYER_REUSE.txt")
        failures = mod.check_r6_no_legacy_markers(tmp_path)
        assert len(failures) == 1
        assert "h200_sxm/gemm/sglang/0.5.10/SHARED_LAYER_REUSE.txt" in failures[0]

    def test_incomplete_txt_is_named(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.10/INCOMPLETE.txt")
        failures = mod.check_r6_no_legacy_markers(tmp_path)
        assert len(failures) == 1
        assert "h200_sxm/gemm/sglang/0.5.10/INCOMPLETE.txt" in failures[0]

    def test_both_marker_types_all_named(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.10/SHARED_LAYER_REUSE.txt")
        _touch(tmp_path, "h200_sxm/attention/trtllm/1.0.0/INCOMPLETE.txt")
        failures = mod.check_r6_no_legacy_markers(tmp_path)
        assert len(failures) == 2

    def test_clean_tree_passes(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        assert mod.check_r6_no_legacy_markers(tmp_path) == []


# ---------------------------------------------------------------------------
# orchestration: run_audit / render_report / main
# ---------------------------------------------------------------------------


class TestOrchestration:
    def test_run_audit_missing_data_root_fails_closed(self, mod, tmp_path):
        results = mod.run_audit(tmp_path / "does-not-exist", REAL_CATALOG)
        assert results["R1"] and "does not exist" in results["R1"][0]
        assert results["R2"] and "does not exist" in results["R2"][0]
        assert results["R6"] and "does not exist" in results["R6"][0]
        # R5 is independent of data_root -- still evaluated against the real
        # manifest/registries via the (valid) catalog passed in.
        assert results["R5"] == []

    def test_render_report_groups_failures_by_rule(self, mod, tmp_path):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.10/SHARED_LAYER_REUSE.txt")
        results = mod.run_audit(tmp_path, REAL_CATALOG)
        report, failed = mod.render_report(results)
        assert failed is True
        assert "[FAIL] R6" in report
        assert "collector data audit FAILED" in report

    def test_main_exit_code_nonzero_on_failure(self, mod, tmp_path, capsys):
        _touch(tmp_path, "h200_sxm/gemm/sglang/0.5.10/SHARED_LAYER_REUSE.txt")
        rc = mod.main(["--data-root", str(tmp_path), "--catalog", str(REAL_CATALOG)])
        assert rc == 1
        assert "FAILED" in capsys.readouterr().out

    def test_main_exit_code_zero_on_success(self, mod, tmp_path, capsys):
        _build_compliant_tree(tmp_path)
        rc = mod.main(["--data-root", str(tmp_path), "--catalog", str(REAL_CATALOG)])
        assert rc == 0
        assert "audit OK" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the real tree -- the CI-riding gate
# ---------------------------------------------------------------------------


def test_real_tree_passes_the_audit(mod):
    """This is the gate collector-audit.yml runs on every PR: the committed
    data tree, as-is, must satisfy every rule.
    """
    results = mod.run_audit(REAL_DATA_ROOT, REAL_CATALOG)
    report, failed = mod.render_report(results)
    assert not failed, report
