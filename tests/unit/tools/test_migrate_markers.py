# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/perf_database/migrate_markers.py.

Builds synthetic family-layout trees (`<system>/<family>/<backend>/<version>/`)
inside a real `git init` repo under tmp_path (the script shells out to `git
add`/`git rm`), and exercises: donor-table derivation for
`SHARED_LAYER_REUSE.txt` -> `reuse.yaml`, legacy-sidecar synthesis for
`INCOMPLETE.txt` -> `collection_meta.yaml`, the AIC-1502 backfill of
`collection_meta.yaml` for parquet-holding dirs that never had ANY marker,
both fail-closed abort paths, determinism/idempotency, the plan/execute/verify
CLI modes, and — per the design's schema lock — that every generated file
round-trips through the REAL loader parser
(`aiconfigurator_core.sdk.perf_database._parse_reuse_yaml` /
`_load_collection_meta_yaml`), not a reimplementation of it.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from aiconfigurator_core.sdk.perf_database import (
    _collection_meta_has_partial_table,
    _load_collection_meta_yaml,
    _parse_reuse_yaml,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "perf_database" / "migrate_markers.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("migrate_markers", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    """A real git repo rooted at tmp_path; the synthetic tree is written directly under it."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "migrate-test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "migrate-test@example.com"], cwd=tmp_path, check=True)
    return tmp_path


def _touch(root: Path, rel: str, content: bytes = b"stub-data") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _track(root: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


# --- version ordering -----------------------------------------------------------------


class TestVersionSortKey:
    def test_newest_wins_among_parseable(self, mod):
        versions = ["0.5.12", "0.5.14", "0.5.9"]
        assert max(versions, key=mod._version_sort_key) == "0.5.14"

    def test_release_candidates_ordered_correctly(self, mod):
        versions = ["1.3.0rc10", "1.3.0rc15", "1.0.0"]
        assert max(versions, key=mod._version_sort_key) == "1.3.0rc15"

    def test_calendar_and_short_versions_parse(self, mod):
        versions = ["2.23", "2021.17.2", "2.27.5"]
        assert max(versions, key=mod._version_sort_key) == "2021.17.2"

    def test_unparseable_version_does_not_crash_and_sorts_last(self, mod):
        versions = ["not-a-version", "0.5.12"]
        assert max(versions, key=mod._version_sort_key) == "0.5.12"

    def test_unparseable_versions_are_deterministic_among_themselves(self, mod):
        # No parseable version present: falls back to string tie-break, not a crash.
        versions = ["zeta", "alpha"]
        assert max(versions, key=mod._version_sort_key) == "zeta"


# --- donor_table_map -------------------------------------------------------------------


class TestDonorTableMap:
    def test_single_donor(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        vdir = repo / "h200_sxm" / "gemm" / "sglang" / "0.5.12"
        vdir.mkdir(parents=True)
        assert mod.donor_table_map(vdir) == {"gemm_perf": "0.5.14"}

    def test_newest_sibling_wins_when_multiple_hold_same_table(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        vdir = repo / "h200_sxm" / "gemm" / "sglang" / "0.5.12"
        vdir.mkdir(parents=True)
        assert mod.donor_table_map(vdir) == {"gemm_perf": "0.5.14"}

    def test_per_table_independent_donor(self, mod, repo):
        # context_attention_perf only ever collected at 0.5.9; generation at 0.5.14.
        _touch(repo, "h200_sxm/attention/sglang/0.5.9/context_attention_perf.parquet")
        _touch(repo, "h200_sxm/attention/sglang/0.5.14/generation_attention_perf.parquet")
        vdir = repo / "h200_sxm" / "attention" / "sglang" / "0.5.12"
        vdir.mkdir(parents=True)
        assert mod.donor_table_map(vdir) == {
            "context_attention_perf": "0.5.9",
            "generation_attention_perf": "0.5.14",
        }

    def test_excludes_own_version(self, mod, repo):
        vdir = repo / "h200_sxm" / "gemm" / "sglang" / "0.5.12"
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/gemm_perf.parquet")
        assert mod.donor_table_map(vdir) == {}

    def test_ignores_marker_and_sidecar_files_in_siblings(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/SHARED_LAYER_REUSE.txt")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.10/reuse.yaml")
        vdir = repo / "h200_sxm" / "gemm" / "sglang" / "0.5.12"
        vdir.mkdir(parents=True)
        assert mod.donor_table_map(vdir) == {}

    def test_no_siblings_at_all(self, mod, repo):
        vdir = repo / "h200_sxm" / "gemm" / "sglang" / "0.5.12"
        vdir.mkdir(parents=True)
        assert mod.donor_table_map(vdir) == {}


# --- scan_tree: SHARED_LAYER_REUSE.txt -> reuse.yaml -----------------------------------


class TestScanSharedLayerReuse:
    def test_generates_one_entry_per_donor_table(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        [action] = scan.reuse_actions
        assert action.src == Path("h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        assert action.dst == Path("h200_sxm/gemm/sglang/0.5.12/reuse.yaml")
        assert action.entries == (mod.ReuseEntry(table="gemm_perf", from_version="0.5.14"),)

    def test_marker_dir_with_own_data_still_gets_sibling_donor_entries(self, mod, repo):
        # Mirrors the real tree's l40s dirs: SHARED marker coexists with own data.
        _touch(repo, "l40s/gemm/sglang/0.5.12/gemm_perf.parquet")
        _touch(repo, "l40s/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "l40s/gemm/sglang/0.5.14/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        [action] = scan.reuse_actions
        assert action.entries == (mod.ReuseEntry(table="gemm_perf", from_version="0.5.14"),)

    def test_no_donor_aborts_and_lists_offender(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.10/SHARED_LAYER_REUSE.txt")  # sibling is marker-only too
        _track(repo)
        with pytest.raises(mod.MigrationError, match="no donor table"):
            mod.scan_tree(repo)

    def test_no_donor_lists_multiple_offenders_sorted(self, mod, repo):
        _touch(repo, "a100_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "h200_sxm/moe/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        with pytest.raises(mod.MigrationError) as exc_info:
            mod.scan_tree(repo)
        message = str(exc_info.value)
        assert "a100_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt" in message
        assert "h200_sxm/moe/sglang/0.5.12/SHARED_LAYER_REUSE.txt" in message

    def test_does_not_mutate_on_abort(self, mod, repo):
        target = _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        with pytest.raises(mod.MigrationError):
            mod.scan_tree(repo)
        assert target.exists()


# --- scan_tree: comm family exclusion (design §6.5 rule 5) -----------------------------


class TestScanCommFamilyExclusion:
    def test_comm_marker_is_deleted_without_reuse_yaml_even_with_donor(self, mod, repo):
        # A real donor exists (0.5.10 holds custom_allreduce_perf) -- proves the
        # exclusion is unconditional, not merely a side effect of "no donor found".
        _touch(repo, "h200_sxm/comm/sglang/0.5.10/custom_allreduce_perf.parquet")
        _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        assert scan.reuse_actions == []
        [action] = scan.comm_exclusion_deletions
        assert action.src == Path("h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")

    def test_comm_marker_with_no_donor_at_all_does_not_abort(self, mod, repo):
        # Would trip the "no donor table" fail-closed abort for any other family;
        # comm bypasses donor lookup entirely so this must not raise.
        _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)  # no MigrationError
        assert scan.reuse_actions == []
        [action] = scan.comm_exclusion_deletions
        assert action.src == Path("h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")

    def test_logs_exclusion_reason(self, mod, repo, caplog):
        _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        with caplog.at_level(logging.INFO, logger="migrate_markers"):
            mod.scan_tree(repo)
        assert any(
            "comm family excluded from sibling reuse (design §6.5 rule 5); marker dropped without declaration"
            in record.message
            and "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt" in record.message
            for record in caplog.records
        )

    def test_non_comm_family_unaffected_in_same_scan(self, mod, repo):
        # One tree, one comm-family marker and one gemm-family marker: only comm is
        # excluded; gemm still converts normally via the donor-based path.
        _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        assert len(scan.comm_exclusion_deletions) == 1
        assert scan.comm_exclusion_deletions[0].src == Path("h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        [action] = scan.reuse_actions
        assert action.src == Path("h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        assert action.entries == (mod.ReuseEntry(table="gemm_perf", from_version="0.5.14"),)

    def test_family_named_similarly_is_not_treated_as_comm(self, mod, repo):
        # Exact-name match only: a family literally named "commfoo" is unrelated to
        # "comm" and must go through the normal donor-based conversion path.
        _touch(repo, "h200_sxm/commfoo/sglang/0.5.14/x_perf.parquet")
        _touch(repo, "h200_sxm/commfoo/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        assert scan.comm_exclusion_deletions == []
        [action] = scan.reuse_actions
        assert action.entries == (mod.ReuseEntry(table="x_perf", from_version="0.5.14"),)

    def test_does_not_mutate_on_scan(self, mod, repo):
        target = _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        mod.scan_tree(repo)  # scan only plans; execute_plan performs the deletion
        assert target.exists()


# --- scan_tree: INCOMPLETE.txt -> collection_meta.yaml ----------------------------------


class TestScanIncomplete:
    def test_has_data_generates_partial_sidecar_for_every_own_table(self, mod, repo):
        _touch(repo, "gb300/attention/vllm/0.14.0/context_attention_perf.parquet")
        _touch(repo, "gb300/attention/vllm/0.14.0/generation_attention_perf.parquet")
        _touch(repo, "gb300/attention/vllm/0.14.0/INCOMPLETE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        [action] = scan.sidecar_actions
        assert action.src == Path("gb300/attention/vllm/0.14.0/INCOMPLETE.txt")
        assert action.dst == Path("gb300/attention/vllm/0.14.0/collection_meta.yaml")
        assert action.framework == "vllm"
        assert action.version == "0.14.0"
        assert action.tables == ("context_attention_perf", "generation_attention_perf")

    def test_marker_only_incomplete_aborts(self, mod, repo):
        """Hypothetical — no marker-only INCOMPLETE.txt dir exists in the real tree
        today (PR 2 already dropped marker-only dirs during the family-layout
        migration); this exercises the fail-closed path if that assumption ever
        breaks, instead of silently synthesizing an empty sidecar."""
        _touch(repo, "h200_sxm/moe/vllm/0.0.0test/INCOMPLETE.txt")
        _track(repo)
        with pytest.raises(mod.MigrationError, match="marker-only"):
            mod.scan_tree(repo)

    def test_does_not_mutate_on_abort(self, mod, repo):
        target = _touch(repo, "h200_sxm/moe/vllm/0.0.0test/INCOMPLETE.txt")
        _track(repo)
        with pytest.raises(mod.MigrationError):
            mod.scan_tree(repo)
        assert target.exists()


# --- scan_tree: backfill (AIC-1502) -----------------------------------------------------


class TestScanBackfill:
    def test_parquet_dir_with_no_sidecar_gets_backfilled(self, mod, repo):
        _touch(repo, "h100_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        assert scan.reuse_actions == []
        assert scan.sidecar_actions == []
        [action] = scan.backfill_actions
        assert action.dst == Path("h100_sxm/gemm/sglang/0.5.9/collection_meta.yaml")
        assert action.framework == "sglang"
        assert action.version == "0.5.9"
        assert action.tables == ("gemm_perf",)

    def test_covers_every_parquet_stem_in_the_dir(self, mod, repo):
        _touch(repo, "h100_sxm/attention/vllm/0.14.0/context_attention_perf.parquet")
        _touch(repo, "h100_sxm/attention/vllm/0.14.0/generation_attention_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        [action] = scan.backfill_actions
        assert action.tables == ("context_attention_perf", "generation_attention_perf")

    def test_existing_sidecar_is_not_overwritten_or_re_planned(self, mod, repo):
        # Mirrors the real tree's 7 INCOMPLETE-derived sidecars: already migrated,
        # status: partial, and must never be touched by the backfill pass.
        _touch(repo, "gb300/moe/vllm/0.14.0/moe_perf.parquet")
        _touch(
            repo,
            "gb300/moe/vllm/0.14.0/collection_meta.yaml",
            b"schema_version: 1\nprovenance: legacy\nruntime:\n  framework: vllm\n  version: '0.14.0'\n"
            b"tables:\n  moe_perf:\n    status: partial\n",
        )
        _track(repo)
        scan = mod.scan_tree(repo)
        assert scan.backfill_actions == []
        meta_path = repo / "gb300" / "moe" / "vllm" / "0.14.0" / "collection_meta.yaml"
        meta = _load_collection_meta_yaml(str(meta_path))
        assert meta["tables"]["moe_perf"]["status"] == "partial"

    def test_dir_with_incomplete_marker_is_not_also_backfilled(self, mod, repo):
        # The INCOMPLETE.txt path (TestScanIncomplete) already produces a
        # SidecarAction for this dir; it must not ALSO show up as a backfill
        # action (that would double-plan the same destination path).
        _touch(repo, "gb300/attention/vllm/0.14.0/context_attention_perf.parquet")
        _touch(repo, "gb300/attention/vllm/0.14.0/INCOMPLETE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        assert len(scan.sidecar_actions) == 1
        assert scan.backfill_actions == []

    def test_reuse_only_dir_with_no_parquet_is_skipped(self, mod, repo):
        # Declared-reuse-only dir: reuse.yaml present, no own parquet data.
        # Nothing to describe -- not a backfill candidate. (The donor dir,
        # 0.5.14, legitimately holds real parquet and DOES get backfilled --
        # that's a separate, correct action, not what this test is about.)
        _touch(
            repo,
            "h100_sxm/gemm/sglang/0.5.9/reuse.yaml",
            b"schema_version: 1\nreuse:\n- table: gemm_perf\n  from_version: '0.5.14'\n"
            b"  reason: r\n  approved_by: yimingl\n",
        )
        _touch(repo, "h100_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        assert Path("h100_sxm/gemm/sglang/0.5.9/collection_meta.yaml") not in {a.dst for a in scan.backfill_actions}

    def test_dir_with_own_parquet_and_reuse_yaml_still_gets_backfilled(self, mod, repo):
        # A dir can hold BOTH its own parquet data AND a reuse.yaml declaring
        # other tables' donors (the l40s pattern); its own table still needs a
        # sidecar for R1 coverage.
        _touch(repo, "l40s/gemm/sglang/0.5.12/gemm_perf.parquet")
        _touch(
            repo,
            "l40s/gemm/sglang/0.5.12/reuse.yaml",
            b"schema_version: 1\nreuse:\n- table: moe_perf\n  from_version: '0.5.14'\n"
            b"  reason: r\n  approved_by: yimingl\n",
        )
        _track(repo)
        scan = mod.scan_tree(repo)
        [action] = scan.backfill_actions
        assert action.tables == ("gemm_perf",)

    def test_multiple_backfills_sorted_by_destination(self, mod, repo):
        _touch(repo, "h200_sxm/moe/sglang/0.5.12/moe_perf.parquet")
        _touch(repo, "a100_sxm/gemm/sglang/0.5.12/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        assert [a.dst for a in scan.backfill_actions] == sorted(a.dst for a in scan.backfill_actions)

    def test_does_not_mutate_on_scan(self, mod, repo):
        target = _touch(repo, "h100_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _track(repo)
        mod.scan_tree(repo)
        assert target.exists()
        assert not (target.parent / "collection_meta.yaml").exists()


# --- determinism -------------------------------------------------------------------------


class TestDeterminism:
    def test_rescan_is_byte_identical(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "gb300/moe/vllm/0.14.0/moe_perf.parquet")
        _touch(repo, "gb300/moe/vllm/0.14.0/INCOMPLETE.txt")
        _track(repo)
        scan1 = mod.scan_tree(repo)
        scan2 = mod.scan_tree(repo)
        assert mod.render_plan(scan1) == mod.render_plan(scan2)

    def test_plan_lines_are_sorted(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "a100_sxm/moe/sglang/0.5.14/moe_perf.parquet")
        _touch(repo, "a100_sxm/moe/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        lines = [ln for ln in mod.render_plan(scan) if not ln.startswith("#")]
        assert lines == sorted(lines)


# --- render_reuse_yaml / render_collection_meta_yaml: real-loader round trip -------------


class TestRenderRoundTripsThroughRealLoader:
    def test_reuse_yaml_parses_through_real_loader(self, mod):
        entries = (
            mod.ReuseEntry(table="gemm_perf", from_version="2.23"),  # ambiguous-as-float on purpose
            mod.ReuseEntry(table="moe_perf", from_version="0.5.14"),
        )
        text = mod.render_reuse_yaml(entries)
        # yaml.safe_load is comment-blind, so pin the SPDX header explicitly
        # (the copyright CI check requires it on every committed sidecar).
        assert text.startswith("# SPDX-FileCopyrightText:")
        raw = yaml.safe_load(text)
        assert raw["schema_version"] == 1
        assert isinstance(raw["reuse"][0]["from_version"], str)

    def test_reuse_yaml_written_to_disk_parses_via_parse_reuse_yaml(self, mod, tmp_path):
        entries = (mod.ReuseEntry(table="gemm_perf", from_version="0.5.14"),)
        path = tmp_path / "reuse.yaml"
        path.write_text(mod.render_reuse_yaml(entries), encoding="utf-8")
        parsed = _parse_reuse_yaml(str(path))
        assert parsed == {
            "entries": [
                {
                    "table": "gemm_perf",
                    "from_version": "0.5.14",
                    "reason": mod.SHARED_REUSE_REASON,
                    "approved_by": "yimingl",
                }
            ]
        }

    def test_collection_meta_yaml_written_to_disk_parses_via_loader_and_is_partial(self, mod, tmp_path):
        action = mod.SidecarAction(
            src=Path("x/INCOMPLETE.txt"),
            dst=Path("x/collection_meta.yaml"),
            framework="vllm",
            version="0.14.0",
            tables=("moe_perf",),
        )
        path = tmp_path / "collection_meta.yaml"
        rendered = mod.render_collection_meta_yaml(action)
        # Covers the shared _render_legacy_collection_meta_yaml path (backfill included).
        assert rendered.startswith("# SPDX-FileCopyrightText:")
        path.write_text(rendered, encoding="utf-8")
        meta = _load_collection_meta_yaml(str(path))
        assert meta["provenance"] == "legacy"
        assert meta["runtime"] == {"framework": "vllm", "version": "0.14.0"}
        assert meta["tables"]["moe_perf"]["status"] == "partial"
        assert _collection_meta_has_partial_table(meta) is True

    def test_backfill_collection_meta_yaml_written_to_disk_parses_via_loader_and_is_complete(self, mod, tmp_path):
        action = mod.BackfillAction(
            dst=Path("x/collection_meta.yaml"),
            framework="sglang",
            version="0.5.9",
            tables=("gemm_perf",),
        )
        path = tmp_path / "collection_meta.yaml"
        path.write_text(mod.render_backfill_collection_meta_yaml(action), encoding="utf-8")
        meta = _load_collection_meta_yaml(str(path))
        assert meta["provenance"] == "legacy"
        assert meta["runtime"] == {"framework": "sglang", "version": "0.5.9"}
        assert meta["tables"]["gemm_perf"]["status"] == "complete"
        # Backfilled dirs were always served as complete -- never partial.
        assert _collection_meta_has_partial_table(meta) is False


# --- execute_plan: real git add / git rm -------------------------------------------------


class TestExecutePlan:
    def test_execute_writes_reuse_yaml_and_deletes_marker(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)

        reuse_path = repo / "h200_sxm" / "gemm" / "sglang" / "0.5.12" / "reuse.yaml"
        assert reuse_path.is_file()
        assert not (repo / "h200_sxm" / "gemm" / "sglang" / "0.5.12" / "SHARED_LAYER_REUSE.txt").exists()
        parsed = _parse_reuse_yaml(str(reuse_path))
        assert parsed["entries"][0]["table"] == "gemm_perf"

    def test_execute_writes_collection_meta_yaml_and_deletes_marker(self, mod, repo):
        _touch(repo, "gb300/moe/vllm/0.14.0/moe_perf.parquet")
        _touch(repo, "gb300/moe/vllm/0.14.0/INCOMPLETE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)

        meta_path = repo / "gb300" / "moe" / "vllm" / "0.14.0" / "collection_meta.yaml"
        assert meta_path.is_file()
        assert not (repo / "gb300" / "moe" / "vllm" / "0.14.0" / "INCOMPLETE.txt").exists()
        meta = _load_collection_meta_yaml(str(meta_path))
        assert _collection_meta_has_partial_table(meta)

    def test_original_data_table_untouched(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet", b"gemm-bytes")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)
        donor = repo / "h200_sxm" / "gemm" / "sglang" / "0.5.14" / "gemm_perf.parquet"
        assert donor.read_bytes() == b"gemm-bytes"

    def test_execute_deletes_comm_marker_without_creating_reuse_yaml(self, mod, repo):
        _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)
        vdir = repo / "h200_sxm" / "comm" / "sglang" / "0.5.12"
        assert not (vdir / "SHARED_LAYER_REUSE.txt").exists()
        assert not (vdir / "reuse.yaml").exists()

    def test_execute_writes_backfill_sidecar_with_no_source_marker_to_delete(self, mod, repo):
        _touch(repo, "h100_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)

        meta_path = repo / "h100_sxm" / "gemm" / "sglang" / "0.5.9" / "collection_meta.yaml"
        assert meta_path.is_file()
        meta = _load_collection_meta_yaml(str(meta_path))
        assert meta["provenance"] == "legacy"
        assert meta["tables"]["gemm_perf"]["status"] == "complete"
        assert not _collection_meta_has_partial_table(meta)

    def test_execute_backfill_leaves_existing_incomplete_sidecar_untouched(self, mod, repo):
        _touch(repo, "gb300/attention/vllm/0.14.0/context_attention_perf.parquet")
        _touch(repo, "gb300/attention/vllm/0.14.0/INCOMPLETE.txt")
        _touch(repo, "h100_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)

        incomplete_meta = _load_collection_meta_yaml(
            str(repo / "gb300" / "attention" / "vllm" / "0.14.0" / "collection_meta.yaml")
        )
        assert incomplete_meta["tables"]["context_attention_perf"]["status"] == "partial"
        backfill_meta = _load_collection_meta_yaml(
            str(repo / "h100_sxm" / "gemm" / "sglang" / "0.5.9" / "collection_meta.yaml")
        )
        assert backfill_meta["tables"]["gemm_perf"]["status"] == "complete"


# --- verify_tree ---------------------------------------------------------------------------


class TestVerify:
    def test_verify_passes_after_execute(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "gb300/moe/vllm/0.14.0/moe_perf.parquet")
        _touch(repo, "gb300/moe/vllm/0.14.0/INCOMPLETE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)
        assert mod.verify_tree(repo) == []

    def test_verify_fails_on_stray_marker(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)
        assert mod.verify_tree(repo) == []

        _touch(repo, "a100_sxm/moe/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        errors = mod.verify_tree(repo)
        assert any("legacy marker" in e for e in errors)

    def test_verify_fails_on_malformed_reuse_yaml(self, mod, repo):
        # Hand-planted, never emitted by the tool itself: missing the required 'reuse' key.
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/reuse.yaml", b"schema_version: 1\n")
        _track(repo)
        errors = mod.verify_tree(repo)
        assert any("missing required top-level 'reuse' key" in e for e in errors)

    def test_verify_fails_on_dangling_donor_reference(self, mod, repo):
        _touch(
            repo,
            "h200_sxm/gemm/sglang/0.5.12/reuse.yaml",
            b"schema_version: 1\nreuse:\n- table: gemm_perf\n  from_version: '9.9.9'\n"
            b"  reason: r\n  approved_by: yimingl\n",
        )
        _track(repo)
        errors = mod.verify_tree(repo)
        assert any("from_version target does not exist" in e for e in errors)

    def test_verify_passes_on_legacy_sidecar_with_only_complete_tables(self, mod, repo):
        # AIC-1502: a status:complete-only legacy sidecar (the backfill shape) is
        # now a legitimate, fully-covered dir -- not an error. Supersedes the old
        # "legacy implies >=1 partial table" invariant, which the backfill breaks
        # by design (these dirs were never incomplete).
        _touch(repo, "gb300/moe/vllm/0.14.0/moe_perf.parquet")
        _touch(
            repo,
            "gb300/moe/vllm/0.14.0/collection_meta.yaml",
            b"schema_version: 1\nprovenance: legacy\nruntime:\n  framework: vllm\n  version: '0.14.0'\n"
            b"tables:\n  moe_perf:\n    status: complete\n",
        )
        _track(repo)
        assert mod.verify_tree(repo) == []

    def test_verify_fails_on_parquet_dir_missing_sidecar_entirely(self, mod, repo):
        _touch(repo, "h100_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _track(repo)
        errors = mod.verify_tree(repo)
        assert any("missing collection_meta.yaml" in e for e in errors)

    def test_verify_fails_on_sidecar_missing_coverage_for_one_table(self, mod, repo):
        _touch(repo, "gb300/moe/vllm/0.14.0/moe_perf.parquet")
        _touch(repo, "gb300/moe/vllm/0.14.0/router_perf.parquet")
        _touch(
            repo,
            "gb300/moe/vllm/0.14.0/collection_meta.yaml",
            b"schema_version: 1\nprovenance: legacy\nruntime:\n  framework: vllm\n  version: '0.14.0'\n"
            b"tables:\n  moe_perf:\n    status: complete\n",
        )
        _track(repo)
        errors = mod.verify_tree(repo)
        assert any("does not cover parquet table(s)" in e and "router_perf" in e for e in errors)

    def test_verify_passes_on_reuse_only_dir_with_no_parquet_and_no_sidecar(self, mod, repo):
        _touch(
            repo,
            "h100_sxm/gemm/sglang/0.5.9/reuse.yaml",
            b"schema_version: 1\nreuse:\n- table: gemm_perf\n  from_version: '0.5.14'\n"
            b"  reason: r\n  approved_by: yimingl\n",
        )
        _touch(repo, "h100_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)
        assert mod.verify_tree(repo) == []

    def test_verify_passes_after_backfill_execute(self, mod, repo):
        _touch(repo, "h100_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _touch(repo, "gb300/attention/vllm/0.14.0/context_attention_perf.parquet")
        _touch(repo, "gb300/attention/vllm/0.14.0/INCOMPLETE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)
        assert mod.verify_tree(repo) == []

    def test_verify_passes_after_comm_exclusion_execute(self, mod, repo):
        _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)
        assert mod.verify_tree(repo) == []


# --- idempotency ---------------------------------------------------------------------------


class TestIdempotency:
    def test_plan_empty_and_verify_clean_after_execute(self, mod, repo):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "gb300/moe/vllm/0.14.0/moe_perf.parquet")
        _touch(repo, "gb300/moe/vllm/0.14.0/INCOMPLETE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)

        rescan = mod.scan_tree(repo)
        assert rescan.reuse_actions == []
        assert rescan.sidecar_actions == []
        assert rescan.backfill_actions == []
        assert mod.verify_tree(repo) == []

    def test_tree_with_no_markers_and_existing_sidecar_has_empty_plan(self, mod, repo):
        # A dir with no legacy markers and an already-present sidecar (e.g. from a
        # prior backfill run) proposes nothing on any pass -- true idempotency,
        # as distinct from TestScanBackfill's "no markers, no sidecar yet" case.
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(
            repo,
            "h200_sxm/gemm/sglang/0.5.14/collection_meta.yaml",
            b"schema_version: 1\nprovenance: legacy\nruntime:\n  framework: sglang\n  version: '0.5.14'\n"
            b"tables:\n  gemm_perf:\n    status: complete\n",
        )
        _track(repo)
        scan = mod.scan_tree(repo)
        assert scan.reuse_actions == []
        assert scan.sidecar_actions == []
        assert scan.backfill_actions == []
        assert mod.verify_tree(repo) == []

    def test_backfill_idempotent_after_execute(self, mod, repo):
        _touch(repo, "h100_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)

        rescan = mod.scan_tree(repo)
        assert rescan.backfill_actions == []
        assert mod.verify_tree(repo) == []

    def test_comm_exclusion_idempotent_after_execute(self, mod, repo):
        _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_tree(repo)
        mod.execute_plan(repo, scan)

        rescan = mod.scan_tree(repo)
        assert rescan.reuse_actions == []
        assert rescan.sidecar_actions == []
        assert rescan.comm_exclusion_deletions == []


# --- CLI ---------------------------------------------------------------------------------------


class TestCli:
    def test_plan_execute_verify_round_trip(self, mod, repo, capsys):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet")
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _touch(repo, "gb300/moe/vllm/0.14.0/moe_perf.parquet")
        _touch(repo, "gb300/moe/vllm/0.14.0/INCOMPLETE.txt")
        _track(repo)

        rc = mod.main(["--data-root", str(repo)])
        assert rc == 0
        plan_out = capsys.readouterr().out
        assert "git add h200_sxm/gemm/sglang/0.5.12/reuse.yaml" in plan_out
        assert "git rm h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt" in plan_out
        assert "shared_reuse_conversions=1 incomplete_sidecars=1 marker_deletions=2" in plan_out

        rc = mod.main(["--data-root", str(repo), "--execute"])
        assert rc == 0
        exec_out = capsys.readouterr().out
        assert "executed 1 reuse.yaml conversion(s), 1 collection_meta.yaml sidecar(s)" in exec_out

        rc = mod.main(["--data-root", str(repo), "--verify"])
        assert rc == 0
        assert "VERIFY OK" in capsys.readouterr().out

        # Idempotent: plan on the now-migrated tree is empty (no add/rm lines, summary only).
        rc = mod.main(["--data-root", str(repo)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "git add" not in out
        assert "marker_deletions=0" in out

    def test_cli_abort_on_no_donor_does_not_mutate(self, mod, repo, capsys):
        target = _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        rc = mod.main(["--data-root", str(repo), "--execute"])
        assert rc == 1
        assert "ABORT" in capsys.readouterr().err
        assert target.exists()

    def test_cli_missing_data_root_aborts_cleanly(self, mod, tmp_path, capsys):
        missing = tmp_path / "does-not-exist"
        rc = mod.main(["--data-root", str(missing)])
        assert rc == 1
        err = capsys.readouterr().err
        assert f"ABORT: data root not found: {missing.resolve()}" in err

    def test_cli_verify_fail_prints_and_returns_nonzero(self, mod, repo, capsys):
        _touch(repo, "h200_sxm/gemm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        rc = mod.main(["--data-root", str(repo), "--verify"])
        assert rc == 1
        assert "VERIFY FAIL" in capsys.readouterr().err

    def test_marker_only_incomplete_aborts_with_clear_reason(self, mod, repo):
        _touch(repo, "h200_sxm/moe/vllm/0.0.0test/INCOMPLETE.txt")
        _track(repo)
        with pytest.raises(mod.MigrationError, match="marker-only"):
            mod.scan_tree(repo)

    def test_plan_includes_comm_exclusion_line_and_summary_counts(self, mod, repo, capsys):
        _touch(repo, "h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt")
        _track(repo)
        rc = mod.main(["--data-root", str(repo)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "git rm h200_sxm/comm/sglang/0.5.12/SHARED_LAYER_REUSE.txt  # comm family excluded" in out
        assert "git add" not in out  # no reuse.yaml ever staged for a comm marker
        assert "comm_exclusions=1" in out
        assert "marker_deletions=1" in out
        assert "legacy_backfills=0" in out

    def test_plan_execute_verify_round_trip_for_backfill_only_tree(self, mod, repo, capsys):
        _touch(repo, "h100_sxm/gemm/sglang/0.5.9/gemm_perf.parquet")
        _track(repo)

        rc = mod.main(["--data-root", str(repo)])
        assert rc == 0
        plan_out = capsys.readouterr().out
        assert "git add h100_sxm/gemm/sglang/0.5.9/collection_meta.yaml" in plan_out
        assert "legacy backfill sidecar" in plan_out
        assert "legacy_backfills=1" in plan_out

        rc = mod.main(["--data-root", str(repo), "--verify"])
        assert rc == 1  # not yet executed: parquet dir still has no sidecar
        assert "VERIFY FAIL" in capsys.readouterr().err

        rc = mod.main(["--data-root", str(repo), "--execute"])
        assert rc == 0
        assert "1 legacy backfill sidecar(s)" in capsys.readouterr().out

        rc = mod.main(["--data-root", str(repo), "--verify"])
        assert rc == 0
        assert "VERIFY OK" in capsys.readouterr().out

        rc = mod.main(["--data-root", str(repo)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "git add" not in out
        assert "legacy_backfills=0" in out
