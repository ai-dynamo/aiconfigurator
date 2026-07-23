# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-op runtime resolution tests (Collector V3 spec §4)."""

import pytest

from collector.framework_manifest import resolve_op_runtime, validate_resolution

pytestmark = pytest.mark.unit

DIGEST = "@sha256:" + "0" * 64

MANIFEST_NO_OVERRIDES = f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{DIGEST}"
"""

MANIFEST_WITH_OVERRIDE = f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{DIGEST}"
    families:
      gemm:
        version: "0.5.15"
        images:
          default: "lmsysorg/sglang:v0.5.15{DIGEST}"
"""

CATALOG = """
schema_version: 1
families:
  - family: gemm
    op_files: [gemm_perf]
  - family: attention
    op_files: [context_attention_perf, generation_attention_perf]
"""


@pytest.fixture
def paths(tmp_path):
    def _write(manifest_text, catalog_text=None):
        manifest = tmp_path / "framework_manifest.yaml"
        manifest.write_text(manifest_text, encoding="utf-8")
        catalog = tmp_path / "op_backend_catalog.yaml"
        if catalog_text is not None:
            catalog.write_text(catalog_text, encoding="utf-8")
        return manifest, catalog

    return _write


def test_default_resolution_without_catalog(paths):
    manifest, catalog = paths(MANIFEST_NO_OVERRIDES)
    runtime = resolve_op_runtime("sglang", "gemm", manifest_path=manifest, catalog_path=catalog)
    assert runtime.version == "0.5.14"
    assert runtime.family is None  # no catalog -> no family identity yet


def test_family_override_wins_with_catalog(paths):
    manifest, catalog = paths(MANIFEST_WITH_OVERRIDE, CATALOG)
    gemm = resolve_op_runtime("sglang", "gemm", manifest_path=manifest, catalog_path=catalog)
    attn = resolve_op_runtime("sglang", "attention_context", manifest_path=manifest, catalog_path=catalog)
    assert (gemm.family, gemm.version) == ("gemm", "0.5.15")
    assert (attn.family, attn.version) == ("attention", "0.5.14")


def test_overrides_without_catalog_fail_closed(paths):
    manifest, catalog = paths(MANIFEST_WITH_OVERRIDE)  # no catalog file
    with pytest.raises(LookupError, match="op catalog is missing"):
        resolve_op_runtime("sglang", "gemm", manifest_path=manifest, catalog_path=catalog)


def test_table_missing_from_catalog_fails_closed(paths):
    manifest, catalog = paths(
        MANIFEST_NO_OVERRIDES,
        "schema_version: 1\nfamilies:\n  - family: gemm\n    op_files: [gemm_perf]\n",
    )
    with pytest.raises(LookupError, match="has no family"):
        resolve_op_runtime("sglang", "attention_context", manifest_path=manifest, catalog_path=catalog)


def test_unknown_op_is_a_hard_error(paths):
    manifest, catalog = paths(MANIFEST_NO_OVERRIDES)
    with pytest.raises(KeyError, match="no op 'not_an_op'"):
        resolve_op_runtime("sglang", "not_an_op", manifest_path=manifest, catalog_path=catalog)


def test_real_manifest_resolves_every_registry_op():
    # The fail-closed CI gate (spec §4): every op in every registry resolves to
    # exactly one pinned runtime with the committed manifest (+ catalog, once
    # PR #1345 lands). Runs in CI via the unit suite.
    assert validate_resolution() == []


def test_validator_reports_missing_framework_entry(tmp_path):
    digest = "@sha256:" + "0" * 64
    manifest = tmp_path / "framework_manifest.yaml"
    manifest.write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest}"
""",
        encoding="utf-8",
    )
    errors = validate_resolution(manifest_path=manifest, catalog_path=tmp_path / "op_backend_catalog.yaml")
    assert any("trtllm" in error for error in errors)
    assert any("wideep_sglang" in error for error in errors)


def test_validator_reports_non_mapping_catalog(paths):
    manifest, catalog = paths(MANIFEST_NO_OVERRIDES, "- foo\n- bar\n")
    errors = validate_resolution(manifest_path=manifest, catalog_path=catalog)
    assert len(errors) == 1
    assert errors[0].startswith("op catalog: ")


def test_validator_reports_malformed_catalog_yaml(paths):
    manifest, catalog = paths(MANIFEST_NO_OVERRIDES, "families: [\n")
    errors = validate_resolution(manifest_path=manifest, catalog_path=catalog)
    assert len(errors) == 1
    assert errors[0].startswith("op catalog: ")


def test_validator_reports_registry_import_failure(monkeypatch):
    import collector.framework_manifest as fm

    monkeypatch.setitem(fm._REGISTRY_MODULES, "sglang", "collector.nonexistent_registry")
    errors = fm.validate_resolution()
    assert any(error.startswith("sglang: registry import failed") for error in errors)
