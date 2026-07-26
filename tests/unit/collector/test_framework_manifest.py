# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for collector framework version/image manifest."""

from pathlib import Path

import pytest

from collector.framework_manifest import get_collector_runtime, require_collector_runtime
from collector.sglang.registry import REGISTRY as SGLANG_REGISTRY
from collector.trtllm.registry import REGISTRY as TRTLLM_REGISTRY
from collector.vllm.registry import REGISTRY as VLLM_REGISTRY
from collector.wideep.sglang.registry import REGISTRY as WIDEEP_SGLANG_REGISTRY
from collector.wideep.trtllm.registry import REGISTRY as WIDEEP_TRTLLM_REGISTRY

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_ROOT = REPO_ROOT / "collector"


def test_manifest_exposes_current_framework_versions_and_images():
    sglang = get_collector_runtime("sglang")
    trtllm = get_collector_runtime("trtllm")
    vllm = get_collector_runtime("vllm")

    assert sglang.version == "0.5.14"
    assert sglang.image().startswith("lmsysorg/sglang:v0.5.14@sha256:")
    assert sglang.image("cu130").startswith("lmsysorg/sglang:v0.5.14-cu130@sha256:")
    assert trtllm.version == "1.3.0rc10"
    assert trtllm.image().startswith("nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10@sha256:")
    assert vllm.version == "0.24.0"
    assert vllm.image().startswith("vllm/vllm-openai:v0.24.0@sha256:")
    assert vllm.image("cu129").startswith("vllm/vllm-openai:v0.24.0-cu129@sha256:")
    # Unknown variants intentionally fall back to the pinned default image.
    assert vllm.image("cu130") == vllm.image()


def test_active_cuda_vllm_collectors_are_exactly_pinned_to_manifest_version():
    expected = f'__compat__ = "vllm=={get_collector_runtime("vllm").version}"'
    assert all(not entry.versions for entry in VLLM_REGISTRY)

    for module in sorted({entry.module for entry in VLLM_REGISTRY}):
        source = (REPO_ROOT / f"{module.replace('.', '/')}.py").read_text(encoding="utf-8")
        declarations = [line.strip() for line in source.splitlines() if line.startswith("__compat__")]
        assert declarations == [expected], module


def test_wideep_runtime_stays_independent_from_default_framework_runtime():
    wideep_sglang = get_collector_runtime("sglang", workload="wideep")
    assert wideep_sglang.version == "0.5.10"
    assert wideep_sglang.version != get_collector_runtime("sglang").version
    assert wideep_sglang.collector_dir == "collector/wideep/sglang"
    assert "deepseek-v4" in wideep_sglang.image()


def test_wideep_entries_are_flattened_peer_frameworks():
    # workload="wideep" is the compatibility spelling for manifest key wideep_<fw>
    via_workload = get_collector_runtime("sglang", workload="wideep")
    direct = get_collector_runtime("wideep_sglang")
    assert via_workload == direct
    assert direct.framework == "wideep_sglang"
    assert direct.data_backend == "sglang"
    assert direct.collector_dir == "collector/wideep/sglang"
    # wideep inherits the base framework's source_repo unless overridden
    assert direct.source_repo == get_collector_runtime("sglang").source_repo


def test_public_images_must_be_digest_pinned(tmp_path):
    manifest = tmp_path / "framework_manifest.yaml"
    manifest.write_text(
        """
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="digest-pinned"):
        get_collector_runtime("sglang", path=manifest)


def test_wideep_entry_missing_base_framework_is_rejected(tmp_path):
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
  wideep_sglang:
    collector_dir: "collector/wideep/sglang"
    data_backend: "sglang"
    default:
      version: "0.5.10"
      images:
        default: "deepseek-v4-blackwell"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_framework"):
        get_collector_runtime("wideep_sglang", path=manifest)


def test_wideep_entry_missing_data_backend_is_rejected(tmp_path):
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
  wideep_sglang:
    base_framework: sglang
    collector_dir: "collector/wideep/sglang"
    default:
      version: "0.5.10"
      images:
        default: "deepseek-v4-blackwell"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="data_backend"):
        get_collector_runtime("wideep_sglang", path=manifest)


WIDEEP_OPS = {entry.op for entry in WIDEEP_SGLANG_REGISTRY}


@pytest.mark.parametrize(
    ("installed_version", "requested_ops", "workload", "version"),
    [
        ("0.5.14+cu130", set(), "default", "0.5.14"),
        ("0.5.10", {"wideep_moe"}, "wideep", "0.5.10"),
    ],
)
def test_runtime_selection_accepts_only_the_matching_pin(installed_version, requested_ops, workload, version):
    runtime = require_collector_runtime("sglang", installed_version, requested_ops=requested_ops, wideep_ops=WIDEEP_OPS)
    assert (runtime.workload, runtime.version) == (workload, version)


@pytest.mark.parametrize(
    ("installed_version", "requested_ops", "match"),
    [
        ("0.5.13", {"gemm"}, r"stock collector requires exactly 0\.5\.14"),
        ("0.5.14rc1", {"gemm"}, r"stock collector requires exactly 0\.5\.14"),
        ("0.5.14.post1", {"gemm"}, r"stock collector requires exactly 0\.5\.14"),
        ("0.5.14", {"wideep_moe"}, r"WideEP collector requires exactly 0\.5\.10"),
        ("0.5.14", {"gemm", "wideep_moe"}, r"0\.5\.14 != 0\.5\.10.*separate containers"),
    ],
)
def test_runtime_selection_rejects_mismatched_or_mixed_pins(installed_version, requested_ops, match):
    with pytest.raises(RuntimeError, match=match):
        require_collector_runtime("sglang", installed_version, requested_ops=requested_ops, wideep_ops=WIDEEP_OPS)


def test_unknown_requested_op_fails_with_key_error():
    with pytest.raises(KeyError, match=r"has no op\(s\): \['not_a_real_op'\]"):
        require_collector_runtime("sglang", "0.5.14", requested_ops={"not_a_real_op"}, wideep_ops=set())


def test_typo_mixed_with_real_op_fails_closed():
    # A typo must not be silently dropped just because another requested op is valid.
    with pytest.raises(KeyError, match=r"has no op\(s\): \['not_a_real_op'\]"):
        require_collector_runtime("sglang", "0.5.14", requested_ops={"gemm", "not_a_real_op"}, wideep_ops=set())


def test_wideep_registry_entries_are_separate_from_stock_backend_registries():
    sglang_modules = {entry.op: entry.module for entry in SGLANG_REGISTRY}
    trtllm_modules = {entry.op: entry.module for entry in TRTLLM_REGISTRY}
    wideep_sglang_modules = {entry.op: entry.module for entry in WIDEEP_SGLANG_REGISTRY}
    wideep_trtllm_modules = {entry.op: entry.module for entry in WIDEEP_TRTLLM_REGISTRY}

    assert "wideep_mla_context" not in sglang_modules
    assert "wideep_mla_generation" not in sglang_modules
    assert "wideep_moe" not in sglang_modules
    assert "trtllm_moe_wideep" not in trtllm_modules
    assert "wideep_mla_context" not in wideep_sglang_modules
    assert "wideep_mla_generation" not in wideep_sglang_modules
    assert wideep_sglang_modules["wideep_moe"].startswith("collector.wideep.sglang.")
    assert wideep_trtllm_modules["trtllm_moe_wideep"].startswith("collector.wideep.trtllm.")


def test_deepep_collectors_live_under_wideep_namespace():
    assert (COLLECTOR_ROOT / "wideep" / "sglang" / "collect_deepep_moe.py").exists()
    assert (COLLECTOR_ROOT / "wideep" / "sglang" / "deepep" / "extract_data.py").exists()
    assert (COLLECTOR_ROOT / "wideep" / "trtllm" / "collect_moe_compute.py").exists()

    assert not (COLLECTOR_ROOT / "deep_collector").exists()
    assert not (COLLECTOR_ROOT / "sglang" / "collect_wideep_deepep_moe.py").exists()
    assert not (COLLECTOR_ROOT / "trtllm" / "collect_wideep_moe_compute.py").exists()


def test_family_overrides_split_ops_across_runtimes(tmp_path):
    digest = "@sha256:" + "0" * 64
    (tmp_path / "framework_manifest.yaml").write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest}"
    families:
      gemm:
        version: "0.5.15"
        images:
          default: "lmsysorg/sglang:v0.5.15{digest}"
""",
        encoding="utf-8",
    )
    (tmp_path / "op_backend_catalog.yaml").write_text(
        """
schema_version: 1
families:
  - family: gemm
    op_files: [gemm_perf]
  - family: attention
    op_files: [context_attention_perf, generation_attention_perf]
""",
        encoding="utf-8",
    )
    # One container cannot serve two pins: fail closed with the op->version split.
    with pytest.raises(RuntimeError, match="multiple runtime versions"):
        require_collector_runtime(
            "sglang",
            "0.5.14",
            requested_ops={"gemm", "attention_context"},
            wideep_ops=set(),
            path=tmp_path / "framework_manifest.yaml",
            catalog_path=tmp_path / "op_backend_catalog.yaml",
        )
    # A single-family request against the matching container succeeds.
    runtime = require_collector_runtime(
        "sglang",
        "0.5.15",
        requested_ops={"gemm"},
        wideep_ops=set(),
        path=tmp_path / "framework_manifest.yaml",
        catalog_path=tmp_path / "op_backend_catalog.yaml",
    )
    assert (runtime.family, runtime.version) == ("gemm", "0.5.15")


def test_family_override_same_version_different_image_is_rejected(tmp_path):
    digest_a = "@sha256:" + "a" * 64
    digest_b = "@sha256:" + "b" * 64
    (tmp_path / "framework_manifest.yaml").write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest_a}"
    families:
      gemm:
        version: "0.5.14"
        images:
          default: "lmsysorg/sglang:v0.5.14-gemm{digest_b}"
""",
        encoding="utf-8",
    )
    (tmp_path / "op_backend_catalog.yaml").write_text(
        """
schema_version: 1
families:
  - family: gemm
    op_files: [gemm_perf]
  - family: attention
    op_files: [context_attention_perf, generation_attention_perf]
""",
        encoding="utf-8",
    )
    # Runtime identity is (version, images), not version alone: the same package
    # version pinned to two different images is still two containers, so a mixed
    # request must fail closed with the op->runtime split instead of letting
    # registry order pick one image silently.
    with pytest.raises(RuntimeError) as excinfo:
        require_collector_runtime(
            "sglang",
            "0.5.14",
            requested_ops={"gemm", "attention_context"},
            wideep_ops=set(),
            path=tmp_path / "framework_manifest.yaml",
            catalog_path=tmp_path / "op_backend_catalog.yaml",
        )
    message = str(excinfo.value)
    assert "same runtime version but different images" in message
    assert f"gemm→0.5.14 [default=lmsysorg/sglang:v0.5.14-gemm{digest_b}]" in message
    assert f"attention_context→0.5.14 [default=lmsysorg/sglang:v0.5.14{digest_a}]" in message
    # Each image group alone is still a valid single-container request.
    runtime = require_collector_runtime(
        "sglang",
        "0.5.14",
        requested_ops={"gemm"},
        wideep_ops=set(),
        path=tmp_path / "framework_manifest.yaml",
        catalog_path=tmp_path / "op_backend_catalog.yaml",
    )
    assert (runtime.family, runtime.version) == ("gemm", "0.5.14")
    assert runtime.image() == f"lmsysorg/sglang:v0.5.14-gemm{digest_b}"


def test_stock_and_wideep_same_version_different_image_is_rejected(tmp_path):
    digest_a = "@sha256:" + "a" * 64
    digest_b = "@sha256:" + "b" * 64
    (tmp_path / "framework_manifest.yaml").write_text(
        f"""
schema_version: 2
frameworks:
  sglang:
    source_repo: "https://github.com/sgl-project/sglang.git"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14{digest_a}"
  wideep_sglang:
    base_framework: sglang
    collector_dir: "collector/wideep/sglang"
    data_backend: "sglang"
    default:
      version: "0.5.14"
      images:
        default: "lmsysorg/sglang:v0.5.14-wideep{digest_b}"
""",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError) as excinfo:
        require_collector_runtime(
            "sglang",
            "0.5.14",
            requested_ops={"gemm", "wideep_moe"},
            wideep_ops={"wideep_moe"},
            path=tmp_path / "framework_manifest.yaml",
        )
    message = str(excinfo.value)
    assert "different images for the same runtime version" in message
    assert f"lmsysorg/sglang:v0.5.14{digest_a}" in message
    assert f"lmsysorg/sglang:v0.5.14-wideep{digest_b}" in message
    assert "separate containers" in message
