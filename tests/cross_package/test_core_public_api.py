# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the standalone core wheel's supported Python facade."""

from __future__ import annotations

import importlib.resources
import inspect
import subprocess
import sys

import aiconfigurator_core
import aiconfigurator_core.sdk as sdk
from aiconfigurator_core.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator_core.sdk.engine import EngineHandle, compile_engine
from aiconfigurator_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

EXPECTED_FACADE = {
    "EngineHandle",
    "ModelConfig",
    "RuntimeConfig",
    "RustForwardPassPerfModel",
    "compile_engine",
    "estimate_kv_cache",
    "estimate_num_gpu_blocks",
}


def test_sdk_facade_import_is_lazy_in_a_fresh_interpreter() -> None:
    script = """
import sys

import aiconfigurator_core.sdk

protected_modules = {
    "aiconfigurator_core.sdk.engine",
    "aiconfigurator_core.sdk.memory",
    "aiconfigurator_core.sdk.rust_engine_step",
}
loaded_modules = protected_modules.intersection(sys.modules)
assert not loaded_modules, f"SDK facade eagerly loaded: {sorted(loaded_modules)}"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_sdk_facade_exports_the_canonical_objects() -> None:
    assert set(sdk.__all__) == EXPECTED_FACADE
    assert sdk.EngineHandle is EngineHandle
    assert sdk.ModelConfig is ModelConfig
    assert sdk.RuntimeConfig is RuntimeConfig
    assert sdk.RustForwardPassPerfModel is RustForwardPassPerfModel
    assert sdk.compile_engine is compile_engine
    assert sdk.estimate_kv_cache is estimate_kv_cache
    assert sdk.estimate_num_gpu_blocks is estimate_num_gpu_blocks


def test_native_and_ergonomic_fpm_classes_are_deliberately_distinct() -> None:
    assert aiconfigurator_core.RustForwardPassPerfModel is not RustForwardPassPerfModel
    assert RustForwardPassPerfModel.__module__ == "aiconfigurator_core.sdk.rust_engine_step"


def test_stable_function_signatures() -> None:
    assert str(inspect.signature(compile_engine)) == (
        "(model_path: 'str', system: 'str', backend: 'str', backend_version: 'str | None' = None, *, "
        "tp_size: 'int' = 1, pp_size: 'int' = 1, attention_dp_size: 'int' = 1, "
        "moe_tp_size: 'int | None' = None, moe_ep_size: 'int | None' = None, "
        "gemm_quant_mode: 'str | None' = None, moe_quant_mode: 'str | None' = None, "
        "kvcache_quant_mode: 'str | None' = None, fmha_quant_mode: 'str | None' = None, "
        "comm_quant_mode: 'str | None' = None, nextn: 'int' = 0, "
        "kv_block_size: 'int | None' = None, "
        "systems_path: 'str | None' = None) -> 'bytes'"
    )
    assert "scheduler_block_size" in inspect.signature(estimate_num_gpu_blocks).parameters
    assert "memory_fraction_kind" in inspect.signature(estimate_kv_cache).parameters


def test_distribution_carries_typing_contract() -> None:
    root = importlib.resources.files("aiconfigurator_core")
    assert (root / "py.typed").is_file()
    assert (root / "_aiconfigurator_core.pyi").is_file()
