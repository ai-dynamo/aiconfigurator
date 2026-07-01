# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for collector framework version/image manifest."""

from pathlib import Path

import pytest

from collector.framework_manifest import get_collector_runtime, require_collector_runtime
from collector.sglang.registry import REGISTRY as SGLANG_REGISTRY
from collector.trtllm.registry import REGISTRY as TRTLLM_REGISTRY
from collector.wideep.sglang.registry import REGISTRY as WIDEEP_SGLANG_REGISTRY
from collector.wideep.trtllm.registry import REGISTRY as WIDEEP_TRTLLM_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLECTOR_ROOT = REPO_ROOT / "collector"


def test_manifest_exposes_current_framework_versions_and_images():
    sglang = get_collector_runtime("sglang")
    trtllm = get_collector_runtime("trtllm")
    vllm = get_collector_runtime("vllm")

    assert sglang.version == "0.5.14"
    assert sglang.image() == "lmsysorg/sglang:v0.5.14"
    assert sglang.image("cu130") == "lmsysorg/sglang:v0.5.14-cu130"
    assert trtllm.version == "1.3.0rc10"
    assert trtllm.image() == "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc10"
    assert vllm.version == "0.19.0"
    assert vllm.image("cu130") == "vllm/vllm-openai:v0.19.0-cu130"


def test_wideep_runtime_stays_independent_from_default_framework_runtime():
    wideep_sglang = get_collector_runtime("sglang", workload="wideep")
    assert wideep_sglang.version == "0.5.10"
    assert wideep_sglang.version != get_collector_runtime("sglang").version
    assert wideep_sglang.collector_dir == "collector/wideep/sglang"
    assert "deepseek-v4" in wideep_sglang.image()


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
