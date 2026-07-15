# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-op runtime resolution tests (Collector V3 spec §4)."""

import pytest

from collector.framework_manifest import resolve_op_runtime

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
