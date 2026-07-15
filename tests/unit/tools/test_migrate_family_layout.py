# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/perf_database/migrate_family_layout.py.

Builds synthetic legacy-layout trees inside a real `git init` repo under
tmp_path (the script shells out to `git mv`/`git rm`/`git add`), and exercises
each rule (catalog lookup, table move, SHARED_LAYER_REUSE.txt replication
scope, INCOMPLETE.txt handling, empty-dir cleanup) plus the plan/execute/
verify CLI modes and the fail-closed abort paths.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "perf_database" / "migrate_family_layout.py"
CATALOG_PATH = REPO_ROOT / "collector" / "op_backend_catalog.yaml"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("migrate_family_layout", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def family_map(mod):
    return mod.load_family_map(CATALOG_PATH)


@pytest.fixture
def family_set(family_map):
    return set(family_map.values())


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


# --- Rule 1: family_for_table / load_family_map --------------------------------


class TestFamilyForTable:
    def test_known_table_from_real_catalog(self, mod, family_map):
        assert mod.family_for_table("gemm_perf", family_map) == "gemm"
        assert mod.family_for_table("nccl_perf", family_map) == "comm"
        assert mod.family_for_table("oneccl_perf", family_map) == "comm"

    def test_unknown_table_raises(self, mod, family_map):
        with pytest.raises(mod.MigrationError, match="mystery_perf"):
            mod.family_for_table("mystery_perf", family_map)

    def test_load_family_map_covers_op_files(self, mod):
        loaded = mod.load_family_map(CATALOG_PATH)
        assert loaded["moe_perf"] == "moe"
        assert loaded["context_attention_perf"] == "attention"


# --- Rule 2: plan_table_move (normal + pseudo-backend) --------------------------


class TestPlanTableMove:
    def test_normal_backend(self, mod, family_map):
        mv = mod.plan_table_move("h200_sxm", "trtllm", "1.3.0rc15", "gemm_perf.parquet", family_map)
        assert mv.src == Path("h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet")
        assert mv.dst == Path("h200_sxm/gemm/trtllm/1.3.0rc15/gemm_perf.parquet")

    def test_pseudo_backend_nccl(self, mod, family_map):
        mv = mod.plan_table_move("h200_sxm", "nccl", "2.23", "nccl_perf.parquet", family_map)
        assert mv.dst == Path("h200_sxm/comm/nccl/2.23/nccl_perf.parquet")

    def test_pseudo_backend_oneccl(self, mod, family_map):
        mv = mod.plan_table_move("b60", "oneccl", "2021.11", "oneccl_perf.parquet", family_map)
        assert mv.dst == Path("b60/comm/oneccl/2021.11/oneccl_perf.parquet")

    def test_unknown_table_raises(self, mod, family_map):
        with pytest.raises(mod.MigrationError):
            mod.plan_table_move("h200_sxm", "trtllm", "1.0.0", "mystery_perf.parquet", family_map)


# --- Rule 3: shared_layer_reuse_scope --------------------------------------------


class TestSharedLayerReuseScope:
    def test_scope_excludes_own_version(self, mod):
        index = {"v1": {"gemm"}, "v2": {"moe"}}
        assert mod.shared_layer_reuse_scope(index, "v2") == {"gemm"}
        assert mod.shared_layer_reuse_scope(index, "v1") == {"moe"}

    def test_scope_unions_multiple_other_versions(self, mod):
        index = {"v1": {"gemm"}, "v2": {"moe"}, "v3": set()}
        assert mod.shared_layer_reuse_scope(index, "v3") == {"gemm", "moe"}

    def test_scope_empty_when_no_sibling_versions(self, mod):
        assert mod.shared_layer_reuse_scope({"v1": set()}, "v1") == set()


# --- Rule 4: incomplete_targets --------------------------------------------------


class TestIncompleteTargets:
    def test_targets_from_own_data(self, mod, family_map):
        assert mod.incomplete_targets(["gemm_perf", "moe_perf"], family_map) == {"gemm", "moe"}

    def test_marker_only_has_no_targets(self, mod, family_map):
        assert mod.incomplete_targets([], family_map) == set()


# --- Rule 5: cleanup_empty_dirs ---------------------------------------------------


class TestCleanupEmptyDirs:
    def test_removes_empty_version_and_backend_dirs(self, mod, tmp_path):
        vdir = tmp_path / "h200_sxm" / "trtllm" / "1.3.0"
        vdir.mkdir(parents=True)
        removed = mod.cleanup_empty_dirs([vdir])
        assert vdir in removed
        assert not vdir.exists()
        assert not vdir.parent.exists()  # backend dir also emptied
        assert (tmp_path / "h200_sxm").exists()  # system dir left alone

    def test_leaves_non_empty_dirs(self, mod, tmp_path):
        vdir = tmp_path / "h200_sxm" / "trtllm" / "1.3.0"
        vdir.mkdir(parents=True)
        (vdir / "leftover.parquet").write_bytes(b"x")
        removed = mod.cleanup_empty_dirs([vdir])
        assert removed == []
        assert vdir.exists()


# --- scan_legacy_tree / render_plan ----------------------------------------------


class TestScanAndPlan:
    def test_normal_move_plan_line(self, mod, family_map, family_set, repo):
        _touch(repo, "h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        assert scan.moves == [
            mod.TableMove(
                src=Path("h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet"),
                dst=Path("h200_sxm/gemm/trtllm/1.3.0rc15/gemm_perf.parquet"),
            )
        ]
        assert scan.manifest == {"gemm_perf.parquet": 1}
        lines = mod.render_plan(scan)
        assert lines == [
            "git mv h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet h200_sxm/gemm/trtllm/1.3.0rc15/gemm_perf.parquet",
            "# manifest: gemm_perf.parquet=1",
        ]

    def test_pseudo_backend_move(self, mod, family_map, family_set, repo):
        _touch(repo, "h200_sxm/nccl/2.23/nccl_perf.parquet")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        assert scan.moves[0].dst == Path("h200_sxm/comm/nccl/2.23/nccl_perf.parquet")

    def test_plan_is_deterministic_and_sorted(self, mod, family_map, family_set, repo):
        _touch(repo, "h200_sxm/trtllm/1.3.0rc15/moe_perf.parquet")
        _touch(repo, "h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet")
        _touch(repo, "b200_sxm/sglang/0.5.12/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        lines = mod.render_plan(scan)
        move_lines = [ln for ln in lines if ln.startswith("git mv")]
        assert move_lines == sorted(move_lines)
        # Re-scanning must produce byte-identical output — no iteration-order flakiness.
        scan2 = mod.scan_legacy_tree(repo, family_map, family_set)
        assert mod.render_plan(scan2) == lines


# --- Rule 3 marker replication: SHARED_LAYER_REUSE.txt ---------------------------


class TestSharedLayerReuseMarker:
    def test_replicates_into_other_version_families_excluding_own(self, mod, family_map, family_set, repo):
        # a100_sxm/trtllm: v1.0.0 has real gemm data; v1.3.0rc15 is a reuse slot that
        # also happens to carry its own moe data. Scope must come from v1.0.0 only.
        _touch(repo, "a100_sxm/trtllm/1.0.0/gemm_perf.parquet")
        _touch(repo, "a100_sxm/trtllm/1.3.0rc15/moe_perf.parquet")
        _touch(repo, "a100_sxm/trtllm/1.3.0rc15/SHARED_LAYER_REUSE.txt", b"lfs-pointer")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        [action] = [a for a in scan.marker_actions if a.marker == mod.SHARED_LAYER_REUSE]
        assert action.src == Path("a100_sxm/trtllm/1.3.0rc15/SHARED_LAYER_REUSE.txt")
        assert action.targets == (Path("a100_sxm/gemm/trtllm/1.3.0rc15/SHARED_LAYER_REUSE.txt"),)

    def test_marker_only_dropped_when_no_sibling_versions(self, mod, family_map, family_set, repo, caplog):
        _touch(repo, "l40s/vllm/0.22.0/SHARED_LAYER_REUSE.txt")
        _track(repo)
        with caplog.at_level(logging.WARNING):
            scan = mod.scan_legacy_tree(repo, family_map, family_set)
        [action] = scan.marker_actions
        assert action.targets == ()
        assert "dropped" in action.reason
        assert any("dropping marker-only" in rec.message for rec in caplog.records)


# --- Rule 4 marker replication: INCOMPLETE.txt ------------------------------------


class TestIncompleteMarker:
    def test_replicates_into_own_data_families(self, mod, family_map, family_set, repo):
        _touch(repo, "rtx_pro_6000_server/vllm/0.14.0/moe_perf.parquet")
        _touch(repo, "rtx_pro_6000_server/vllm/0.14.0/INCOMPLETE.txt", b"Partial collector bring-up data only.")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        [action] = [a for a in scan.marker_actions if a.marker == mod.INCOMPLETE]
        assert action.targets == (Path("rtx_pro_6000_server/moe/vllm/0.14.0/INCOMPLETE.txt"),)

    def test_marker_only_dropped_and_logged(self, mod, family_map, family_set, repo, caplog):
        _touch(repo, "gb300/vllm/0.14.0/INCOMPLETE.txt", b"Partial collector bring-up data only.")
        _track(repo)
        with caplog.at_level(logging.WARNING):
            scan = mod.scan_legacy_tree(repo, family_map, family_set)
        [action] = scan.marker_actions
        assert action.marker == mod.INCOMPLETE
        assert action.targets == ()
        assert "dropped" in action.reason
        assert any("dropping marker-only" in rec.message for rec in caplog.records)


# --- execute_plan: real git mv / git add / git rm ---------------------------------


class TestExecutePlan:
    def test_execute_moves_files_and_cleans_up_empty_dirs(self, mod, family_map, family_set, repo):
        _touch(repo, "h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet", b"gemm-bytes")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        mod.execute_plan(repo, scan)
        dst = repo / "h200_sxm" / "gemm" / "trtllm" / "1.3.0rc15" / "gemm_perf.parquet"
        assert dst.read_bytes() == b"gemm-bytes"
        assert not (repo / "h200_sxm" / "trtllm").exists()

    def test_execute_replicates_and_drops_marker(self, mod, family_map, family_set, repo):
        _touch(repo, "a100_sxm/trtllm/1.0.0/gemm_perf.parquet")
        _touch(repo, "a100_sxm/trtllm/1.3.0rc15/SHARED_LAYER_REUSE.txt", b"lfs-pointer")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        mod.execute_plan(repo, scan)
        replicated = repo / "a100_sxm" / "gemm" / "trtllm" / "1.3.0rc15" / "SHARED_LAYER_REUSE.txt"
        assert replicated.read_bytes() == b"lfs-pointer"
        assert not (repo / "a100_sxm" / "trtllm" / "1.3.0rc15").exists()

    def test_execute_drops_marker_only_dir_entirely(self, mod, family_map, family_set, repo):
        _touch(repo, "l40s/vllm/0.22.0/SHARED_LAYER_REUSE.txt")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        mod.execute_plan(repo, scan)
        assert not (repo / "l40s" / "vllm").exists()


# --- verify_tree -------------------------------------------------------------------


class TestVerify:
    def test_verify_passes_after_execute(self, mod, family_map, family_set, repo):
        _touch(repo, "h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet")
        _touch(repo, "h200_sxm/nccl/2.23/nccl_perf.parquet")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        mod.execute_plan(repo, scan)
        assert mod.verify_tree(repo, family_map, family_set) == []

    def test_verify_fails_on_stray_legacy_parquet(self, mod, family_map, family_set, repo):
        _touch(repo, "h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        mod.execute_plan(repo, scan)
        assert mod.verify_tree(repo, family_map, family_set) == []

        # Plant a stray depth-3 (legacy-shaped) parquet on top of the migrated tree.
        _touch(repo, "h200_sxm/sglang/0.5.12/gemm_perf.parquet")
        _track(repo)
        errors = mod.verify_tree(repo, family_map, family_set)
        assert errors
        assert any("legacy-shaped" in e for e in errors)


# --- Idempotency: plan mode on an already-migrated tree ----------------------------


class TestIdempotency:
    def test_plan_is_empty_on_already_migrated_tree(self, mod, family_map, family_set, repo):
        _touch(repo, "h200_sxm/gemm/trtllm/1.3.0rc15/gemm_perf.parquet")
        _touch(repo, "h200_sxm/comm/nccl/2.23/nccl_perf.parquet")
        _track(repo)
        scan = mod.scan_legacy_tree(repo, family_map, family_set)
        assert scan.moves == []
        assert scan.marker_actions == []
        assert mod.render_plan(scan) == []
        assert mod.verify_tree(repo, family_map, family_set) == []


# --- Fail-closed aborts --------------------------------------------------------------


class TestFailClosed:
    def test_unknown_table_aborts_without_mutating(self, mod, family_map, family_set, repo):
        target = _touch(repo, "h200_sxm/trtllm/1.3.0rc15/mystery_perf.parquet")
        _track(repo)
        with pytest.raises(mod.MigrationError, match="mystery_perf"):
            mod.scan_legacy_tree(repo, family_map, family_set)
        assert target.exists()

    def test_unexpected_first_level_dir_aborts_listing_offender(self, mod, family_map, family_set, repo):
        _touch(repo, "h200_sxm/weirdbackend/1.0/gemm_perf.parquet")
        _track(repo)
        with pytest.raises(mod.MigrationError, match="weirdbackend"):
            mod.scan_legacy_tree(repo, family_map, family_set)

    def test_cli_abort_returns_nonzero_and_does_not_mutate(self, mod, repo, capsys):
        target = _touch(repo, "h200_sxm/trtllm/1.3.0rc15/mystery_perf.parquet")
        _track(repo)
        rc = mod.main(["--data-root", str(repo), "--execute"])
        assert rc == 1
        assert "ABORT" in capsys.readouterr().err
        assert target.exists()


# --- CLI round trip ------------------------------------------------------------------


class TestCliRoundTrip:
    def test_plan_execute_verify(self, mod, repo, capsys):
        _touch(repo, "h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet")
        _touch(repo, "h200_sxm/nccl/2.23/nccl_perf.parquet")
        _touch(repo, "a100_sxm/trtllm/1.0.0/gemm_perf.parquet")
        _touch(repo, "a100_sxm/trtllm/1.3.0rc15/SHARED_LAYER_REUSE.txt")
        _track(repo)

        rc = mod.main(["--data-root", str(repo)])
        assert rc == 0
        plan_out = capsys.readouterr().out
        assert "git mv h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet" in plan_out

        rc = mod.main(["--data-root", str(repo), "--execute"])
        assert rc == 0
        capsys.readouterr()

        rc = mod.main(["--data-root", str(repo), "--verify"])
        assert rc == 0
        assert "VERIFY OK" in capsys.readouterr().out

    def test_verify_red_on_stray_file_via_cli(self, mod, repo, capsys):
        _touch(repo, "h200_sxm/trtllm/1.3.0rc15/gemm_perf.parquet")
        _track(repo)
        assert mod.main(["--data-root", str(repo), "--execute"]) == 0
        capsys.readouterr()

        _touch(repo, "h200_sxm/sglang/0.5.12/gemm_perf.parquet")
        _track(repo)
        rc = mod.main(["--data-root", str(repo), "--verify"])
        assert rc == 1
        assert "VERIFY FAIL" in capsys.readouterr().err
