# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import compatibility contract between the AIC and AIC Core wheels."""

from __future__ import annotations

import importlib
import importlib.resources
import sys
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

CORE_SDK_LEAF_MODULES = [
    "afd_partition",
    "backends.base_backend",
    "backends.factory",
    "backends.sglang_backend",
    "backends.trtllm_backend",
    "backends.vllm_backend",
    "common",
    "config",
    "config_builders",
    "engine",
    "errors",
    "inference_summary",
    "memory",
    "models.base",
    "models.deepseek",
    "models.deepseek_v32",
    "models.deepseek_v4",
    "models.gemma4",
    "models.gpt",
    "models.helpers",
    "models.hybrid_moe",
    "models.llama",
    "models.minimax_m3",
    "models.moe",
    "models.nemotron_h",
    "models.nemotron_nas",
    "models.qwen35",
    "models.qwen3vl",
    "models.vit_ops",
    "operations.afd_transfer",
    "operations.attention",
    "operations.base",
    "operations.communication",
    "operations.dsa",
    "operations.dsv4",
    "operations.elementwise",
    "operations.embedding",
    "operations.gemm",
    "operations.mamba",
    "operations.mla",
    "operations.moe",
    "operations.msa",
    "operations.overlap",
    "operations.util_empirical",
    "perf_database",
    "perf_interp.config",
    "perf_interp.engine",
    "performance_result",
    "rust_engine_step",
    "step_estimate",
    "system_spec",
    "utils",
]


def _discover_python_leaves(root: object, prefix: str = "") -> set[str]:
    """Return import suffixes for every non-package Python module below root."""
    modules: set[str] = set()
    for child in root.iterdir():
        if child.name == "__pycache__":
            continue
        if child.is_dir():
            modules.update(_discover_python_leaves(child, f"{prefix}{child.name}."))
        elif child.name.endswith(".py") and child.name != "__init__.py":
            modules.add(f"{prefix}{child.name.removesuffix('.py')}")
    return modules


def test_import_contract_covers_every_core_sdk_leaf() -> None:
    """A new core SDK module must add a legacy wrapper and contract case."""
    core_sdk_root = importlib.resources.files("aiconfigurator_core.sdk")

    assert set(CORE_SDK_LEAF_MODULES) == _discover_python_leaves(core_sdk_root)


@pytest.mark.parametrize("module_suffix", CORE_SDK_LEAF_MODULES)
def test_legacy_leaf_module_is_canonical_module(module_suffix: str) -> None:
    """Every compatibility leaf must share caches and private module state."""
    legacy_name = f"aiconfigurator.sdk.{module_suffix}"
    canonical_name = f"aiconfigurator_core.sdk.{module_suffix}"

    legacy_module = importlib.import_module(legacy_name)
    canonical_module = importlib.import_module(canonical_name)

    assert legacy_module is canonical_module
    assert sys.modules[legacy_name] is sys.modules[canonical_name]


@pytest.mark.parametrize("package_suffix", ["models", "operations", "perf_interp"])
def test_legacy_package_reexports_canonical_public_surface(package_suffix: str) -> None:
    """Package facades preserve child wrappers and export canonical objects."""
    legacy_package = importlib.import_module(f"aiconfigurator.sdk.{package_suffix}")
    canonical_package = importlib.import_module(f"aiconfigurator_core.sdk.{package_suffix}")

    assert legacy_package.__all__ == canonical_package.__all__
    for public_name in canonical_package.__all__:
        assert getattr(legacy_package, public_name) is getattr(canonical_package, public_name)


def test_models_package_delegates_private_registry() -> None:
    """Private registry access sees the canonical registry, not a copied one."""
    legacy_models = importlib.import_module("aiconfigurator.sdk.models")
    canonical_models = importlib.import_module("aiconfigurator_core.sdk.models")

    assert legacy_models._MODEL_REGISTRY is canonical_models._MODEL_REGISTRY


@pytest.mark.parametrize(
    ("package_suffix", "attribute"),
    [
        ("models", "_get_model_info"),
        ("operations", "clear_all_op_caches"),
    ],
)
def test_legacy_package_patch_updates_canonical_package(package_suffix: str, attribute: str) -> None:
    """Patching a legacy package attribute must affect canonical code."""
    canonical_package = importlib.import_module(f"aiconfigurator_core.sdk.{package_suffix}")

    with patch(f"aiconfigurator.sdk.{package_suffix}.{attribute}") as mocked:
        assert getattr(canonical_package, attribute) is mocked

    assert getattr(canonical_package, attribute) is not mocked


def test_representative_from_imports_return_canonical_objects() -> None:
    """The user-facing from-import form remains backward compatible."""
    from aiconfigurator.sdk.config import ModelConfig as LegacyModelConfig
    from aiconfigurator.sdk.models import GPTModel as LegacyGPTModel
    from aiconfigurator.sdk.operations import GEMM as LEGACY_GEMM
    from aiconfigurator_core.sdk.config import ModelConfig
    from aiconfigurator_core.sdk.models import GPTModel
    from aiconfigurator_core.sdk.operations import GEMM

    assert LegacyModelConfig is ModelConfig
    assert LegacyGPTModel is GPTModel
    assert LEGACY_GEMM is GEMM
