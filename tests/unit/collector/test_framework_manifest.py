# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for collector framework version/image manifest."""

import copy
from pathlib import Path

import pytest

from collector.framework_manifest import get_collector_runtime, load_manifest, validate_manifest
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

    assert sglang.version == "0.5.10"
    assert sglang.image() == "lmsysorg/sglang:v0.5.10"
    assert sglang.image("cu130") == "lmsysorg/sglang:v0.5.10-cu130"
    assert trtllm.version == "1.3.0rc10"
    assert trtllm.image() == "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10"
    assert vllm.version == "0.24.0"
    assert vllm.image() == "vllm/vllm-openai:v0.24.0"
    assert vllm.image("cu129") == "vllm/vllm-openai:v0.24.0-cu129"
    # vLLM 0.24.0 has no separately published cu130 tag. Unknown variants
    # intentionally fall back to the pinned default image.
    assert vllm.image("cu130") == "vllm/vllm-openai:v0.24.0"


def test_active_cuda_vllm_collectors_are_exactly_pinned_to_manifest_version():
    expected = f'__compat__ = "vllm=={get_collector_runtime("vllm").version}"'
    assert all(not entry.versions for entry in VLLM_REGISTRY)

    for module in sorted({entry.module for entry in VLLM_REGISTRY}):
        source = (REPO_ROOT / f"{module.replace('.', '/')}.py").read_text(encoding="utf-8")
        declarations = [line.strip() for line in source.splitlines() if line.startswith("__compat__")]
        assert declarations == [expected], module


def test_wideep_versions_stay_aligned_with_default_framework_versions():
    manifest = load_manifest()

    for framework, wideep_spec in manifest["wideep"].items():
        assert wideep_spec["version"] == manifest["frameworks"][framework]["version"]

    wideep_sglang = get_collector_runtime("sglang", workload="wideep")
    assert wideep_sglang.version == get_collector_runtime("sglang").version
    assert wideep_sglang.collector_dir == "collector/wideep/sglang"
    assert "deepseek-v4" in wideep_sglang.image()


def test_manifest_validation_rejects_wideep_version_drift():
    manifest = copy.deepcopy(load_manifest())
    manifest["wideep"]["sglang"]["version"] = "0.5.9"

    with pytest.raises(ValueError, match=r"wideep\.sglang\.version must match"):
        validate_manifest(manifest)


def test_wideep_registry_entries_are_separate_from_stock_backend_registries():
    sglang_modules = {entry.op: entry.module for entry in SGLANG_REGISTRY}
    trtllm_modules = {entry.op: entry.module for entry in TRTLLM_REGISTRY}
    wideep_sglang_modules = {entry.op: entry.module for entry in WIDEEP_SGLANG_REGISTRY}
    wideep_trtllm_modules = {entry.op: entry.module for entry in WIDEEP_TRTLLM_REGISTRY}

    assert "wideep_mla_context" not in sglang_modules
    assert "wideep_mla_generation" not in sglang_modules
    assert "wideep_moe" not in sglang_modules
    assert "trtllm_moe_wideep" not in trtllm_modules
    assert wideep_sglang_modules["wideep_mla_context"].startswith("collector.wideep.sglang.")
    assert wideep_sglang_modules["wideep_mla_generation"].startswith("collector.wideep.sglang.")
    assert wideep_sglang_modules["wideep_moe"].startswith("collector.wideep.sglang.")
    assert wideep_trtllm_modules["trtllm_moe_wideep"].startswith("collector.wideep.trtllm.")


def test_deepep_collectors_live_under_wideep_namespace():
    assert (COLLECTOR_ROOT / "wideep" / "sglang" / "collect_deepep_moe.py").exists()
    assert (COLLECTOR_ROOT / "wideep" / "sglang" / "deepep" / "extract_data.py").exists()
    assert (COLLECTOR_ROOT / "wideep" / "trtllm" / "collect_moe_compute.py").exists()

    assert not (COLLECTOR_ROOT / "deep_collector").exists()
    assert not (COLLECTOR_ROOT / "sglang" / "collect_wideep_deepep_moe.py").exists()
    assert not (COLLECTOR_ROOT / "trtllm" / "collect_wideep_moe_compute.py").exists()
