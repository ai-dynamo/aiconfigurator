# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/perf_database/changed_ops.py.

Builds a minimal fixture git repo (tmp_path) shaped like the real
`collector/` + `aic-core/.../systems/data` tree at two commits ("base" and
"head"), and exercises the three GPU-free change signals (pin_version,
collector_code, case_plan), the `systems`/`tables` derivation, determinism,
and the design-§8 output schema. One test also runs against the REAL repo
with base=HEAD head=HEAD, which must report everything unchanged.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "perf_database" / "changed_ops.py"


@pytest.fixture
def mod():
    spec = importlib.util.spec_from_file_location("changed_ops", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "changed-ops-test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "changed-ops-test@example.com"], cwd=tmp_path, check=True)
    return tmp_path


# --------------------------------------------------------------------------
# Fixture tree: a minimal sglang framework with two families (gemm, attention)
# sharing SHARED_CORE, each with its own module + case-input yaml.
# --------------------------------------------------------------------------

REGISTRY_TYPES_PY = """from enum import Enum


class PerfFile(str, Enum):
    def __str__(self) -> str:
        return self.value

    GEMM = "gemm_perf.txt"
    CONTEXT_ATTENTION = "context_attention_perf.txt"
"""

CATALOG_YAML = """schema_version: 1
families:
  - family: gemm
    op_files: [gemm_perf]
  - family: attention
    op_files: [context_attention_perf]
"""

CLOSURES_YAML = """collector.sglang.collect_gemm:
  - collector/cases/base_ops/gemm.yaml
collector.sglang.collect_attn:
  - collector/cases/base_ops/attention.yaml
"""

REGISTRY_PY = """from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.sglang.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="attention_context",
        module="collector.sglang.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention",
        perf_filename=PerfFile.CONTEXT_ATTENTION,
    ),
]
"""

# A second sibling registry list the real vllm registry also has
# (REGISTRY_XPU): must never be picked up by the parser.
REGISTRY_PY_WITH_INACTIVE_SIBLING = (
    REGISTRY_PY
    + """
REGISTRY_INACTIVE: list[OpEntry] = [
    OpEntry(
        op="ghost",
        module="collector.sglang.collect_ghost",
        get_func="get_ghost_test_cases",
        run_func="run_ghost",
        perf_filename=PerfFile.GEMM,
    ),
]
"""
)


def _manifest_yaml(*, version: str = "0.5.14", digest: str = "0" * 64, family_override: str | None = None) -> str:
    families_block = ""
    if family_override:
        families_block = f"""    families:
      {family_override}:
        version: "{version}"
        images:
          default: "example/sglang:v{version}@sha256:{digest}"
"""
    return f"""schema_version: 2
frameworks:
  sglang:
    source_repo: "https://example.com/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "example/sglang:v0.5.14@sha256:{"0" * 64}"
{families_block}"""


def _default_files() -> dict[str, str]:
    return {
        "collector/registry_types.py": REGISTRY_TYPES_PY,
        "collector/framework_manifest.yaml": _manifest_yaml(),
        "collector/op_backend_catalog.yaml": CATALOG_YAML,
        "collector/hash_closures.yaml": CLOSURES_YAML,
        "collector/sglang/registry.py": REGISTRY_PY,
        "collector/helper.py": "# helper v1\n",
        "collector/case_generator.py": "# case_generator v1\n",
        "collector/model_cases.py": "# model_cases v1\n",
        "collector/capabilities.py": "# capabilities v1\n",
        "collector/version_resolver.py": "# version_resolver v1\n",
        "collector/sglang/collect_gemm.py": "# collect_gemm v1\n",
        "collector/sglang/collect_attn.py": "# collect_attn v1\n",
        "collector/cases/base_ops/gemm.yaml": "# gemm cases v1\n",
        "collector/cases/base_ops/attention.yaml": "# attention cases v1\n",
    }


def _write_tree(repo: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _commit_all(repo: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    # --allow-empty: some tests intentionally re-write identical content (no-op
    # head commit) to exercise the "no change" path without git refusing the commit.
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", message], cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _base_and_head(repo: Path, *, head_overrides: dict[str, str]) -> tuple[str, str]:
    """Commit the default tree as base, then apply overrides and commit as head."""
    _write_tree(repo, _default_files())
    base_sha = _commit_all(repo, "base")
    files = _default_files()
    files.update(head_overrides)
    _write_tree(repo, files)
    head_sha = _commit_all(repo, "head")
    return base_sha, head_sha


def _diff_for(diffs: list, framework: str, family: str):
    [match] = [d for d in diffs if d.framework == framework and d.family == family]
    return match


# --------------------------------------------------------------------------
# signal 1: pin_version
# --------------------------------------------------------------------------


class TestPinVersion:
    def test_family_override_pin_bump_isolates_that_family(self, mod, repo):
        base_sha, head_sha = _base_and_head(
            repo,
            head_overrides={
                "collector/framework_manifest.yaml": _manifest_yaml(
                    version="0.5.15", digest="1" * 64, family_override="gemm"
                )
            },
        )
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert _diff_for(changed, "sglang", "gemm").reasons == ("pin_version",)
        assert _diff_for(unchanged, "sglang", "attention").reasons == ()

    def test_digest_only_change_is_pin_version(self, mod, repo):
        # Same version string, only the image (which carries the digest) differs.
        base_sha, head_sha = _base_and_head(
            repo,
            head_overrides={
                "collector/framework_manifest.yaml": _manifest_yaml(
                    version="0.5.14", digest="2" * 64, family_override="gemm"
                )
            },
        )
        changed, _unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert _diff_for(changed, "sglang", "gemm").reasons == ("pin_version",)


# --------------------------------------------------------------------------
# signal 2: collector_code
# --------------------------------------------------------------------------


class TestCollectorCode:
    def test_module_file_edit_isolates_that_family(self, mod, repo):
        base_sha, head_sha = _base_and_head(
            repo, head_overrides={"collector/sglang/collect_gemm.py": "# collect_gemm v2 (changed)\n"}
        )
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert _diff_for(changed, "sglang", "gemm").reasons == ("collector_code",)
        assert _diff_for(unchanged, "sglang", "attention").reasons == ()

    def test_shared_core_edit_changes_every_family_for_the_framework(self, mod, repo):
        base_sha, head_sha = _base_and_head(repo, head_overrides={"collector/helper.py": "# helper v2 (changed)\n"})
        changed, _unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert _diff_for(changed, "sglang", "gemm").reasons == ("collector_code",)
        assert _diff_for(changed, "sglang", "attention").reasons == ("collector_code",)

    def test_inactive_sibling_registry_list_is_never_parsed(self, mod, repo):
        # collector/vllm/registry.py has a real REGISTRY_XPU sibling list that
        # importlib.import_module(...).REGISTRY never reads; this proves the
        # ast-based parser matches that semantics instead of walking the file.
        _write_tree(repo, _default_files() | {"collector/sglang/registry.py": REGISTRY_PY_WITH_INACTIVE_SIBLING})
        base_sha = _commit_all(repo, "base")
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, base_sha)
        families = {d.family for d in (*changed, *unchanged) if d.framework == "sglang"}
        assert families == {"gemm", "attention"}


# --------------------------------------------------------------------------
# signal 3: case_plan
# --------------------------------------------------------------------------


class TestCasePlan:
    def test_case_yaml_edit_triggers_case_plan(self, mod, repo):
        base_sha, head_sha = _base_and_head(
            repo, head_overrides={"collector/cases/base_ops/gemm.yaml": "# gemm cases v2 (changed)\n"}
        )
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        gemm = _diff_for(changed, "sglang", "gemm")
        # A case-input file is also an authored hash_closures.yaml extra for its
        # module, so collector_hash legitimately changes alongside it (design
        # §5: "collector_hash covers ... the family's cases/base_ops/*.yaml") —
        # both reasons are honest, not a bug in either signal.
        assert set(gemm.reasons) == {"case_plan", "collector_code"}
        assert _diff_for(unchanged, "sglang", "attention").reasons == ()

    def test_shared_case_generator_edit_triggers_case_plan_for_every_family(self, mod, repo):
        base_sha, head_sha = _base_and_head(
            repo, head_overrides={"collector/case_generator.py": "# case_generator v2 (changed)\n"}
        )
        changed, _unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert "case_plan" in _diff_for(changed, "sglang", "gemm").reasons
        assert "case_plan" in _diff_for(changed, "sglang", "attention").reasons


# --------------------------------------------------------------------------
# combined reasons
# --------------------------------------------------------------------------


class TestCombinedReasons:
    def test_pin_code_and_case_all_change_together(self, mod, repo):
        base_sha, head_sha = _base_and_head(
            repo,
            head_overrides={
                "collector/framework_manifest.yaml": _manifest_yaml(
                    version="0.5.15", digest="3" * 64, family_override="gemm"
                ),
                "collector/sglang/collect_gemm.py": "# collect_gemm v2 (changed)\n",
                "collector/cases/base_ops/gemm.yaml": "# gemm cases v2 (changed)\n",
            },
        )
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        gemm = _diff_for(changed, "sglang", "gemm")
        assert gemm.reasons == ("pin_version", "collector_code", "case_plan")
        assert _diff_for(unchanged, "sglang", "attention").reasons == ()


# --------------------------------------------------------------------------
# no-change
# --------------------------------------------------------------------------


class TestNoChange:
    def test_identical_trees_are_all_unchanged(self, mod, repo):
        _write_tree(repo, _default_files())
        base_sha = _commit_all(repo, "base")
        _write_tree(repo, _default_files())  # re-written identically; no new blob content
        head_sha = _commit_all(repo, "head (no-op)")
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert changed == []
        assert {(d.framework, d.family) for d in unchanged} == {("sglang", "gemm"), ("sglang", "attention")}

    def test_same_rev_twice_is_all_unchanged(self, mod, repo):
        _write_tree(repo, _default_files())
        base_sha = _commit_all(repo, "base")
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, base_sha)
        assert changed == []
        assert len(unchanged) == 2


# --------------------------------------------------------------------------
# tables / systems derivation
# --------------------------------------------------------------------------


class TestTablesAndSystems:
    def test_systems_scoped_by_family_and_backend_not_just_table_presence(self, mod, repo):
        prefix = mod.DATA_PREFIX
        files = _default_files()
        files[f"{prefix}/h200_sxm/gemm/sglang/0.5.14/gemm_perf.parquet"] = "x"
        files[f"{prefix}/b200_sxm/attention/sglang/0.5.14/context_attention_perf.parquet"] = "x"
        # Unrelated: another family entirely — must never leak into gemm's systems.
        files[f"{prefix}/l40s/attention/trtllm/9.9.9/context_attention_perf.parquet"] = "x"
        _write_tree(repo, files)
        base_sha = _commit_all(repo, "base")
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, base_sha)
        gemm = _diff_for(unchanged, "sglang", "gemm")
        attention = _diff_for(unchanged, "sglang", "attention")
        assert gemm.tables == ("gemm_perf",)
        assert gemm.systems == ("h200_sxm",)
        assert attention.tables == ("context_attention_perf",)
        assert attention.systems == ("b200_sxm",)

    def test_no_data_means_no_systems(self, mod, repo):
        _write_tree(repo, _default_files())
        base_sha = _commit_all(repo, "base")
        _changed, unchanged = mod.compute_changed_ops(repo, base_sha, base_sha)
        assert _diff_for(unchanged, "sglang", "gemm").systems == ()


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


class TestDeterminism:
    def test_render_report_is_byte_identical_across_runs(self, mod, repo):
        base_sha, head_sha = _base_and_head(
            repo, head_overrides={"collector/sglang/collect_gemm.py": "# collect_gemm v2 (changed)\n"}
        )
        changed1, unchanged1 = mod.compute_changed_ops(repo, base_sha, head_sha)
        changed2, unchanged2 = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert mod.render_report(changed1, unchanged1) == mod.render_report(changed2, unchanged2)

    def test_output_order_is_sorted_by_framework_then_family(self, mod, repo):
        _write_tree(repo, _default_files())
        base_sha = _commit_all(repo, "base")
        _changed, unchanged = mod.compute_changed_ops(repo, base_sha, base_sha)
        pairs = [(d.framework, d.family) for d in unchanged]
        assert pairs == sorted(pairs)


# --------------------------------------------------------------------------
# rendering: design §8's locked schema
# --------------------------------------------------------------------------


class TestRenderSchema:
    def test_changed_entry_schema(self, mod, repo):
        base_sha, head_sha = _base_and_head(
            repo, head_overrides={"collector/sglang/collect_gemm.py": "# collect_gemm v2 (changed)\n"}
        )
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        doc = yaml.safe_load(mod.render_report(changed, unchanged))
        [gemm] = [e for e in doc["changed"] if e["family"] == "gemm"]
        assert list(gemm.keys()) == ["framework", "family", "reasons", "tables", "systems", "action"]
        assert gemm["framework"] == "sglang"
        assert gemm["reasons"] == ["collector_code"]
        assert gemm["action"] == "recollect"

    def test_unchanged_entry_schema(self, mod, repo):
        _write_tree(repo, _default_files())
        base_sha = _commit_all(repo, "base")
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, base_sha)
        doc = yaml.safe_load(mod.render_report(changed, unchanged))
        [gemm] = [e for e in doc["unchanged"] if e["family"] == "gemm"]
        assert list(gemm.keys()) == ["framework", "family", "tables", "systems"]

    def test_top_level_keys_are_changed_and_unchanged(self, mod, repo):
        _write_tree(repo, _default_files())
        base_sha = _commit_all(repo, "base")
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, base_sha)
        doc = yaml.safe_load(mod.render_report(changed, unchanged))
        assert set(doc.keys()) == {"changed", "unchanged"}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestCli:
    def test_main_writes_to_out_file(self, mod, repo, monkeypatch):
        base_sha, head_sha = _base_and_head(
            repo, head_overrides={"collector/sglang/collect_gemm.py": "# collect_gemm v2 (changed)\n"}
        )
        monkeypatch.setattr(mod, "REPO_ROOT", repo)
        out_path = repo / "out" / "manifest.yaml"
        rc = mod.main(["--base", base_sha, "--head", head_sha, "--out", str(out_path)])
        assert rc == 0
        doc = yaml.safe_load(out_path.read_text())
        assert len(doc["changed"]) == 1

    def test_main_default_stdout(self, mod, repo, monkeypatch, capsys):
        base_sha, head_sha = _base_and_head(repo, head_overrides={})
        monkeypatch.setattr(mod, "REPO_ROOT", repo)
        rc = mod.main(["--base", base_sha, "--head", head_sha])
        assert rc == 0
        doc = yaml.safe_load(capsys.readouterr().out)
        assert doc["changed"] == []


# --------------------------------------------------------------------------
# real-repo integration smoke test
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_real_repo_base_equals_head_is_all_unchanged(mod):
    changed, unchanged = mod.compute_changed_ops(REPO_ROOT, "HEAD", "HEAD")
    assert changed == []
    assert len(unchanged) > 0
    frameworks = {d.framework for d in unchanged}
    assert frameworks <= {"sglang", "trtllm", "vllm", "wideep_sglang", "wideep_trtllm"}
