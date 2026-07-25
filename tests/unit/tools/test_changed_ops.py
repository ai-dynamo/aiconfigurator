# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/perf_database/changed_ops.py.

Builds a minimal fixture git repo (tmp_path) shaped like the real
`collector/` + `aic-core/.../systems/data` tree at two commits ("base" and
"head"), and exercises the three GPU-free change signals (pin_version,
collector_code, case_plan), the `systems`/`tables` derivation, determinism,
and the design-§8 output schema. One test also runs against the REAL repo
with base=HEAD head=HEAD, which must report everything unchanged.

Also covers: VersionRoute-versioned registry entries failing closed
(NotImplementedError, FIX 1), SHARED_CORE being read per-revision from
`collector/provenance.py` rather than imported live (FIX 2), a pre-Collector-V3
`--base` exiting loudly on a dedicated code (FIX 2), and two frameworks
sharing one physical `data_backend` directory disambiguating `systems` by
table stem (FIX 4).
"""

from __future__ import annotations

import importlib.util
import shutil
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

# collector/provenance.py's SHARED_CORE, mirrored exactly so the fixture's
# collector_hash construction matches the real module's (changed_ops.py now
# parses this per-revision via `ast` rather than importing the live module).
PROVENANCE_PY = """SHARED_CORE = (
    "collector/helper.py",
    "collector/case_generator.py",
    "collector/model_cases.py",
    "collector/capabilities.py",
    "collector/version_resolver.py",
)
"""

# Same as PROVENANCE_PY but SHARED_CORE names one extra file — used to prove
# SHARED_CORE is read from THIS revision's collector/provenance.py, not
# imported from the live checkout (FIX 2 / revision purity).
PROVENANCE_PY_WITH_EXTRA_SHARED_FILE = """SHARED_CORE = (
    "collector/helper.py",
    "collector/case_generator.py",
    "collector/model_cases.py",
    "collector/capabilities.py",
    "collector/version_resolver.py",
    "collector/extra_shared.py",
)
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

# A registry whose REGISTRY list contains a VersionRoute-versioned OpEntry
# (no literal `module=`) — the shape _parse_registry_entries does not support
# and must fail closed on (FIX 1) rather than silently skip.
REGISTRY_PY_WITH_VERSION_ROUTE = """from collector.registry_types import OpEntry, PerfFile, VersionRoute

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="x",
        get_func="get_x_test_cases",
        run_func="run_x",
        perf_filename=PerfFile.GEMM,
        versions=(VersionRoute(min_version="1.0", module="m"),),
    ),
]
"""


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
        "collector/provenance.py": PROVENANCE_PY,
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

    def test_shared_core_is_read_per_revision_not_from_live_checkout(self, mod, repo):
        # collector/provenance.py's SHARED_CORE at HEAD names one extra file
        # not present at BASE: collector_hash must reflect that revision's
        # own SHARED_CORE (ast-parsed via `git show`), never the currently
        # checked-out collector.provenance module's constant.
        base_sha, head_sha = _base_and_head(
            repo,
            head_overrides={
                "collector/provenance.py": PROVENANCE_PY_WITH_EXTRA_SHARED_FILE,
                "collector/extra_shared.py": "# new shared file\n",
            },
        )
        changed, _unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert _diff_for(changed, "sglang", "gemm").reasons == ("collector_code",)
        assert _diff_for(changed, "sglang", "attention").reasons == ("collector_code",)


# --------------------------------------------------------------------------
# registry parsing: unsupported shapes (FIX 1)
# --------------------------------------------------------------------------


class TestVersionRouteUnsupported:
    def test_version_route_entry_raises_not_implemented(self, mod, repo):
        _write_tree(repo, _default_files() | {"collector/sglang/registry.py": REGISTRY_PY_WITH_VERSION_ROUTE})
        base_sha = _commit_all(repo, "base")
        with pytest.raises(NotImplementedError, match="VersionRoute"):
            mod.compute_changed_ops(repo, base_sha, base_sha)


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
# FIX 4: two frameworks sharing one physical backend_dir (trtllm +
# wideep_trtllm both resolve `data_backend: trtllm`) with disjoint table
# stems in the same family — each framework's `systems`/`tables` must reflect
# only ITS OWN tables, never the sibling framework's.
# --------------------------------------------------------------------------

_SHARED_BACKEND_DIR_MANIFEST_YAML = """schema_version: 2
frameworks:
  trtllm:
    default:
      version: "1.3.0"
      images:
        default: "example/trtllm:v1.3.0@sha256:{}"
  wideep_trtllm:
    collector_dir: "collector/wideep_trtllm"
    data_backend: trtllm
    default:
      version: "1.3.0"
      images:
        default: "example/trtllm:v1.3.0@sha256:{}"
""".format("1" * 64, "1" * 64)

_SHARED_BACKEND_DIR_CATALOG_YAML = """schema_version: 1
families:
  - family: moe
    op_files: [moe_perf, wideep_moe_perf]
"""

_SHARED_BACKEND_DIR_CLOSURES_YAML = """collector.trtllm.collect_moe: []
collector.wideep_trtllm.collect_moe: []
"""

_TRTLLM_REGISTRY_PY = """from collector.registry_types import OpEntry

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="moe",
        module="collector.trtllm.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe",
        perf_filename="moe_perf.txt",
    ),
]
"""

_WIDEEP_TRTLLM_REGISTRY_PY = """from collector.registry_types import OpEntry

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="moe",
        module="collector.wideep_trtllm.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe",
        perf_filename="wideep_moe_perf.txt",
    ),
]
"""


class TestSharedBackendDirDisambiguation:
    def test_stock_and_wideep_frameworks_disambiguate_by_table_stem(self, mod, repo):
        files = {
            "collector/registry_types.py": REGISTRY_TYPES_PY,
            "collector/provenance.py": PROVENANCE_PY,
            "collector/framework_manifest.yaml": _SHARED_BACKEND_DIR_MANIFEST_YAML,
            "collector/op_backend_catalog.yaml": _SHARED_BACKEND_DIR_CATALOG_YAML,
            "collector/hash_closures.yaml": _SHARED_BACKEND_DIR_CLOSURES_YAML,
            "collector/trtllm/registry.py": _TRTLLM_REGISTRY_PY,
            "collector/wideep_trtllm/registry.py": _WIDEEP_TRTLLM_REGISTRY_PY,
            "collector/helper.py": "# helper v1\n",
            "collector/case_generator.py": "# case_generator v1\n",
            "collector/model_cases.py": "# model_cases v1\n",
            "collector/capabilities.py": "# capabilities v1\n",
            "collector/version_resolver.py": "# version_resolver v1\n",
            "collector/trtllm/collect_moe.py": "# collect_moe v1\n",
            "collector/wideep_trtllm/collect_moe.py": "# wideep collect_moe v1\n",
            f"{mod.DATA_PREFIX}/h100_sxm/moe/trtllm/1.3.0/moe_perf.parquet": "x",
            f"{mod.DATA_PREFIX}/gb200_nvl72/moe/trtllm/1.3.0/wideep_moe_perf.parquet": "x",
        }
        _write_tree(repo, files)
        base_sha = _commit_all(repo, "shared-backend-dir base")
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, base_sha)
        assert changed == []
        trtllm_moe = _diff_for(unchanged, "trtllm", "moe")
        wideep_moe = _diff_for(unchanged, "wideep_trtllm", "moe")
        assert trtllm_moe.tables == ("moe_perf",)
        assert trtllm_moe.systems == ("h100_sxm",)
        assert wideep_moe.tables == ("wideep_moe_perf",)
        assert wideep_moe.systems == ("gb200_nvl72",)


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
# FIX 2: pre-Collector-V3 baseline handling. `--base` predating
# collector/provenance.py, collector/hash_closures.yaml, or
# collector/op_backend_catalog.yaml has no provenance metadata to diff
# against — the tool must exit loudly with a dedicated exit code, not crash
# on a missing-file git error or (worse) silently produce a wrong manifest.
# --------------------------------------------------------------------------


def _pre_v3_files(*, omit: str) -> dict[str, str]:
    """The default fixture tree with one Collector-V3 metadata file removed —
    simulates a --base revision that predates Collector V3.
    """
    return {rel: content for rel, content in _default_files().items() if rel != omit}


class TestPreV3Baseline:
    @pytest.mark.parametrize(
        "omitted_path",
        ["collector/provenance.py", "collector/hash_closures.yaml", "collector/op_backend_catalog.yaml"],
    )
    def test_compute_changed_ops_raises_when_base_is_missing_any_v3_metadata_file(self, mod, repo, omitted_path):
        _write_tree(repo, _pre_v3_files(omit=omitted_path))
        base_sha = _commit_all(repo, "pre-v3 base")
        _write_tree(repo, _default_files())
        head_sha = _commit_all(repo, "head (v3)")
        with pytest.raises(mod.PreCollectorV3BaselineError, match="predates Collector V3"):
            mod.compute_changed_ops(repo, base_sha, head_sha)

    def test_main_exits_with_dedicated_code_and_loud_message(self, mod, repo, monkeypatch, capsys):
        _write_tree(repo, _pre_v3_files(omit="collector/provenance.py"))
        base_sha = _commit_all(repo, "pre-v3 base")
        _write_tree(repo, _default_files())
        head_sha = _commit_all(repo, "head (v3)")
        monkeypatch.setattr(mod, "REPO_ROOT", repo)
        rc = mod.main(["--base", base_sha, "--head", head_sha])
        assert rc == mod.EXIT_PRE_V3_BASELINE == 3
        err = capsys.readouterr().err
        assert "predates Collector V3 provenance metadata" in err

    def test_main_exits_with_dedicated_code_when_head_predates_v3(self, mod, repo, monkeypatch, capsys):
        """A pre-V3 --head must map to the same documented exit code 3 (naming
        head), not escape as a raw CalledProcessError from `git show`."""
        _write_tree(repo, _default_files())
        base_sha = _commit_all(repo, "base (v3)")
        for rel in _default_files():
            if rel not in _pre_v3_files(omit="collector/provenance.py"):
                (repo / rel).unlink()
        head_sha = _commit_all(repo, "pre-v3 head")
        monkeypatch.setattr(mod, "REPO_ROOT", repo)
        rc = mod.main(["--base", base_sha, "--head", head_sha])
        assert rc == mod.EXIT_PRE_V3_BASELINE == 3
        err = capsys.readouterr().err
        assert "head revision predates Collector V3 provenance metadata" in err

    def test_both_revisions_having_v3_metadata_is_unaffected(self, mod, repo):
        base_sha, head_sha = _base_and_head(
            repo, head_overrides={"collector/sglang/collect_gemm.py": "# collect_gemm v2 (changed)\n"}
        )
        changed, unchanged = mod.compute_changed_ops(repo, base_sha, head_sha)
        assert _diff_for(changed, "sglang", "gemm").reasons == ("collector_code",)
        assert _diff_for(unchanged, "sglang", "attention").reasons == ()


# --------------------------------------------------------------------------
# real-repo integration smoke test
# --------------------------------------------------------------------------


def _repo_root_is_git_checkout() -> bool:
    if shutil.which("git") is None:
        return False
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


@pytest.mark.unit
@pytest.mark.skipif(
    not _repo_root_is_git_checkout(),
    reason="needs the real repo checkout (the CI test image COPYs sources without .git)",
)
def test_real_repo_base_equals_head_is_all_unchanged(mod):
    changed, unchanged = mod.compute_changed_ops(REPO_ROOT, "HEAD", "HEAD")
    assert changed == []
    assert len(unchanged) > 0
    frameworks = {d.framework for d in unchanged}
    assert frameworks <= {"sglang", "trtllm", "vllm", "wideep_sglang", "wideep_trtllm"}
