# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""collection_meta.yaml provenance writer tests (Collector V3 design §5)."""

from pathlib import Path

import pytest
import yaml

from collector import provenance

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
HASH_CLOSURES_PATH = REPO_ROOT / "collector" / "hash_closures.yaml"


# --------------------------------------------------------------------------
# Fixture "repo" builder: a minimal file tree shaped like the real repo, so
# collector_hash's SHARED_CORE + closure-extra resolution runs against real
# on-disk files without depending on the actual (large) collector package.
# --------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_fake_repo(root: Path, *, module_body: str = "op A", extra_body: str = "sweep A") -> None:
    for shared_file in provenance.SHARED_CORE:
        _write(root / shared_file, f"# {shared_file}\n")
    _write(root / "collector" / "sglang" / "collect_gemm.py", f"# {module_body}\n")
    _write(root / "collector" / "cases" / "base_ops" / "gemm.yaml", f"# {extra_body}\n")
    _write(root / "collector" / "cases" / "models" / "LlamaForCausalLM_cases.yaml", "# model cases\n")


FAKE_CLOSURES = {
    "collector.sglang.collect_gemm": [
        "collector/cases/base_ops/gemm.yaml",
        "__model_cases__",
    ],
}


# --------------------------------------------------------------------------
# load_closures: fail-closed coverage
# --------------------------------------------------------------------------


def test_load_closures_fails_closed_on_missing_registry_module(tmp_path):
    # Only ever true if at least one real registry module exists, which it does.
    real_module = sorted(provenance.enumerate_registry_modules())[0]
    incomplete = tmp_path / "hash_closures.yaml"
    # Deliberately omit every real registry module.
    incomplete.write_text("some.other.module: []\n", encoding="utf-8")

    with pytest.raises(KeyError, match=real_module.split(".")[-1]):
        provenance.load_closures(incomplete)


def test_load_closures_rejects_non_mapping_top_level(tmp_path):
    bad = tmp_path / "hash_closures.yaml"
    bad.write_text("- foo\n- bar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        provenance.load_closures(bad)


def test_load_closures_rejects_non_list_extras(tmp_path):
    bad = tmp_path / "hash_closures.yaml"
    bad.write_text("collector.sglang.collect_gemm: not-a-list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list of strings"):
        provenance.load_closures(bad)


def test_hash_closures_yaml_covers_every_registry_module():
    # The fail-closed CI gate (design §5, mirrors test_manifest_resolution's
    # test_real_manifest_resolves_every_registry_op): every module across all
    # five registries has a hash_closures.yaml entry.
    closures = provenance.load_closures(HASH_CLOSURES_PATH)
    missing = provenance.enumerate_registry_modules() - closures.keys()
    assert missing == set()


def test_hash_closures_yaml_has_no_stale_entries():
    # Entries for modules no registry references anymore would be silently
    # wrong (blast-radius honesty cuts both ways) — keep the file exact.
    closures = provenance.load_closures(HASH_CLOSURES_PATH)
    stale = closures.keys() - provenance.enumerate_registry_modules()
    assert stale == set()


# --------------------------------------------------------------------------
# collector_hash: fail-closed, deterministic, content-sensitive, rebase-stable
# --------------------------------------------------------------------------


def test_collector_hash_fails_closed_for_unknown_module(tmp_path):
    _build_fake_repo(tmp_path)
    with pytest.raises(KeyError, match="no hash_closures"):
        provenance.collector_hash("collector.sglang.not_registered", tmp_path, FAKE_CLOSURES)


def test_collector_hash_deterministic(tmp_path):
    _build_fake_repo(tmp_path)
    h1 = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    h2 = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_collector_hash_changes_when_module_file_changes(tmp_path):
    _build_fake_repo(tmp_path, module_body="op A")
    before = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    _build_fake_repo(tmp_path, module_body="op A changed")
    after = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    assert before != after


def test_collector_hash_changes_when_closure_extra_changes(tmp_path):
    _build_fake_repo(tmp_path, extra_body="sweep A")
    before = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    _build_fake_repo(tmp_path, extra_body="sweep A changed")
    after = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    assert before != after


def test_collector_hash_changes_when_shared_core_file_changes(tmp_path):
    _build_fake_repo(tmp_path)
    before = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    _write(tmp_path / "collector" / "helper.py", "# helper.py changed\n")
    after = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    assert before != after


def test_collector_hash_changes_when_model_cases_group_changes(tmp_path):
    _build_fake_repo(tmp_path)
    before = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    _write(tmp_path / "collector" / "cases" / "models" / "LlamaForCausalLM_cases.yaml", "# changed\n")
    after = provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)
    assert before != after


def test_collector_hash_missing_closure_file_raises(tmp_path):
    _build_fake_repo(tmp_path)
    (tmp_path / "collector" / "cases" / "base_ops" / "gemm.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        provenance.collector_hash("collector.sglang.collect_gemm", tmp_path, FAKE_CLOSURES)


def test_collector_hash_stable_across_repo_relocation(tmp_path_factory):
    # Rebase-stability: content hash must not depend on the absolute path the
    # repo happens to live at.
    root_a = tmp_path_factory.mktemp("repo_a")
    root_b = tmp_path_factory.mktemp("repo_b_somewhere_else")
    _build_fake_repo(root_a)
    _build_fake_repo(root_b)

    hash_a = provenance.collector_hash("collector.sglang.collect_gemm", root_a, FAKE_CLOSURES)
    hash_b = provenance.collector_hash("collector.sglang.collect_gemm", root_b, FAKE_CLOSURES)
    assert hash_a == hash_b


# --------------------------------------------------------------------------
# case_plan_hash
# --------------------------------------------------------------------------


def test_case_plan_hash_deterministic():
    assert provenance.case_plan_hash(["a", "b", "c"]) == provenance.case_plan_hash(["a", "b", "c"])


def test_case_plan_hash_order_independent():
    assert provenance.case_plan_hash(["c", "a", "b"]) == provenance.case_plan_hash(["a", "b", "c"])


def test_case_plan_hash_deduplicates():
    assert provenance.case_plan_hash(["a", "a", "b"]) == provenance.case_plan_hash(["a", "b"])


def test_case_plan_hash_changes_with_different_set():
    assert provenance.case_plan_hash(["a", "b"]) != provenance.case_plan_hash(["a", "b", "c"])


def test_case_plan_hash_empty_set_is_stable():
    assert provenance.case_plan_hash([]) == provenance.case_plan_hash([])
    assert provenance.case_plan_hash([]).startswith("sha256:")


# --------------------------------------------------------------------------
# derive_table_status — table-driven
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unresolved_failed_count", "had_module_failure", "expected"),
    [
        (0, False, provenance.STATUS_COMPLETE),
        (1, False, provenance.STATUS_PARTIAL),
        (0, True, provenance.STATUS_PARTIAL),
        (3, True, provenance.STATUS_PARTIAL),
    ],
    ids=["all-passed", "unresolved-failures", "module-collection-failure", "both"],
)
def test_derive_table_status(unresolved_failed_count, had_module_failure, expected):
    assert (
        provenance.derive_table_status(
            unresolved_failed_count=unresolved_failed_count,
            had_module_failure=had_module_failure,
        )
        == expected
    )


# --------------------------------------------------------------------------
# write_collection_meta: design §5 schema rendering
# --------------------------------------------------------------------------


RUNTIME_META = {
    "framework": "sglang",
    "version": "0.5.14",
    "image": "lmsysorg/sglang:v0.5.14",
    "image_digest": "sha256:" + "0" * 64,
}

TABLES = {
    "moe_perf": {
        "collector_ref": "0b077da5",
        "collector_hash": "sha256:" + "1" * 64,
        "case_plan_hash": "sha256:" + "2" * 64,
        "collected_at": "2026-07-20",
        "rows": 12345,
        "status": "complete",
    },
    "gemm_perf": {
        "collector_ref": "0b077da5",
        "collector_hash": "sha256:" + "3" * 64,
        "case_plan_hash": "sha256:" + "4" * 64,
        "collected_at": "2026-07-20",
        "rows": 42,
        "status": "partial",
    },
}


def test_write_collection_meta_returns_flat_sidecar_path(tmp_path):
    meta_path = provenance.write_collection_meta(tmp_path, RUNTIME_META, TABLES)
    assert meta_path == tmp_path / "collection_meta.yaml"
    assert meta_path.exists()


def test_write_collection_meta_schema_matches_design_5(tmp_path):
    meta_path = provenance.write_collection_meta(tmp_path, RUNTIME_META, TABLES)
    doc = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    assert doc["schema_version"] == 1
    assert doc["runtime"] == {
        "framework": "sglang",
        "version": "0.5.14",
        "image": "lmsysorg/sglang:v0.5.14",
        "image_digest": "sha256:" + "0" * 64,
    }
    assert set(doc["tables"]) == {"moe_perf", "gemm_perf"}
    for table_name, expected in TABLES.items():
        assert doc["tables"][table_name] == {
            "collector_ref": expected["collector_ref"],
            "collector_hash": expected["collector_hash"],
            "case_plan_hash": expected["case_plan_hash"],
            "collected_at": expected["collected_at"],
            "rows": expected["rows"],
            "status": expected["status"],
        }


def test_write_collection_meta_omits_absent_optional_runtime_fields(tmp_path):
    runtime_meta = {"framework": "wideep_sglang", "version": "0.5.10", "image": "deepseek-v4-blackwell"}
    meta_path = provenance.write_collection_meta(tmp_path, runtime_meta, TABLES)
    doc = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert "image_digest" not in doc["runtime"]


def test_write_collection_meta_deterministic_key_order(tmp_path):
    meta_path = provenance.write_collection_meta(tmp_path, RUNTIME_META, TABLES)
    text = meta_path.read_text(encoding="utf-8")
    assert text.startswith("# SPDX-FileCopyrightText:")
    lines = [line for line in text.splitlines() if line and not line.startswith((" ", "#"))]
    assert lines == ["schema_version: 1", "runtime:", "tables:"]

    all_lines = text.splitlines()
    runtime_start = all_lines.index("runtime:") + 1
    runtime_lines = all_lines[runtime_start : runtime_start + 4]
    assert [line.split(":")[0].strip() for line in runtime_lines] == ["framework", "version", "image", "image_digest"]

    # tables render in sorted (not insertion) order.
    tables_section = text.split("tables:", 1)[1]
    table_names_in_order = [
        line.strip().rstrip(":")
        for line in tables_section.splitlines()
        if line.startswith("  ") and not line.startswith("   ")
    ]
    assert table_names_in_order == sorted(TABLES)

    # field order within a table entry follows the design's declared order,
    # regardless of the insertion order of the input dict.
    shuffled_table = {
        "status": "complete",
        "rows": 1,
        "collected_at": "2026-07-20",
        "case_plan_hash": "sha256:" + "9" * 64,
        "collector_hash": "sha256:" + "8" * 64,
        "collector_ref": "deadbeef",
    }
    meta_path2 = provenance.write_collection_meta(tmp_path / "other", RUNTIME_META, {"only_perf": shuffled_table})
    text2 = meta_path2.read_text(encoding="utf-8")
    field_lines = text2.split("only_perf:", 1)[1].splitlines()[1:7]
    assert [line.split(":")[0].strip() for line in field_lines] == [
        "collector_ref",
        "collector_hash",
        "case_plan_hash",
        "collected_at",
        "rows",
        "status",
    ]


def test_write_collection_meta_creates_out_dir(tmp_path):
    out_dir = tmp_path / "nested" / "not_yet_created"
    meta_path = provenance.write_collection_meta(out_dir, RUNTIME_META, TABLES)
    assert meta_path.exists()
    assert meta_path.parent == out_dir
