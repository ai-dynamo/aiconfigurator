# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test installed upper/core package layers outside the source checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import sys
from pathlib import Path

# Running this file directly prepends ``tools/`` to sys.path. Remove that path
# so installed-package checks cannot accidentally resolve repository helpers.
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _TOOLS_DIR]


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_distribution_files(name: str, required: tuple[str, ...]) -> None:
    distribution = importlib.metadata.distribution(name)
    files = {str(path): path for path in distribution.files or ()}
    missing = [path for path in required if path not in files or not distribution.locate_file(files[path]).is_file()]
    if missing:
        raise RuntimeError(f"distribution {name!r} is missing installed files: {missing}")


def _forbid_module(name: str) -> None:
    if importlib.util.find_spec(name) is not None:
        raise RuntimeError(f"module {name!r} belongs to an uninstalled layer")


def _verify_core(*, exercise_engine: bool) -> str:
    core_version = _distribution_version("aiconfigurator-core")
    if core_version is None:
        raise RuntimeError("aiconfigurator-core distribution is not installed")
    _require_distribution_files(
        "aiconfigurator-core",
        (
            "aiconfigurator_core/__init__.py",
            "aiconfigurator_core/_aiconfigurator_core.pyi",
            "aiconfigurator_core/model_configs/meta-llama--Meta-Llama-3.1-8B_config.json",
            "aiconfigurator_core/py.typed",
            "aiconfigurator_core/sdk/__init__.py",
            "aiconfigurator_core/sdk/engine.py",
            "aiconfigurator_core/sdk/memory.py",
            "aiconfigurator_core/systems/h100_sxm.yaml",
        ),
    )

    core = importlib.import_module("aiconfigurator_core")
    if core._build_smoke() != 1:
        raise RuntimeError("native core extension returned an unexpected schema version")

    sdk = importlib.import_module("aiconfigurator_core.sdk")
    protected_sdk_modules = {
        "aiconfigurator_core.sdk.engine",
        "aiconfigurator_core.sdk.memory",
        "aiconfigurator_core.sdk.rust_engine_step",
    }
    eagerly_loaded_modules = protected_sdk_modules.intersection(sys.modules)
    if eagerly_loaded_modules:
        raise RuntimeError(f"aiconfigurator_core.sdk eagerly loaded modules: {sorted(eagerly_loaded_modules)}")

    expected_facade = {
        "EngineHandle",
        "ModelConfig",
        "RuntimeConfig",
        "RustForwardPassPerfModel",
        "compile_engine",
        "estimate_kv_cache",
        "estimate_num_gpu_blocks",
    }
    if set(sdk.__all__) != expected_facade:
        raise RuntimeError(f"unexpected aiconfigurator_core.sdk facade: {sdk.__all__!r}")
    for public_name in expected_facade:
        if getattr(sdk, public_name, None) is None:
            raise RuntimeError(f"aiconfigurator_core.sdk is missing {public_name}")

    for module in (
        "aiconfigurator_core.sdk.engine",
        "aiconfigurator_core.sdk.memory",
        "aiconfigurator_core.sdk.perf_database",
    ):
        importlib.import_module(module)

    resources = importlib.resources.files("aiconfigurator_core")
    required_resources = (
        resources / "model_configs" / "meta-llama--Meta-Llama-3.1-8B_config.json",
        resources / "systems" / "h100_sxm.yaml",
        resources / "systems" / "data" / "b200_sxm" / "gemm" / "vllm" / "0.19.0" / "gemm_perf.parquet",
        resources / "systems" / "data" / "l40s" / "gemm" / "vllm" / "0.22.0" / "reuse.yaml",
        resources / "systems" / "data" / "b200_sxm" / "gemm" / "vllm" / "0.19.0" / "collection_meta.yaml",
    )
    missing = [str(path) for path in required_resources if not path.is_file()]
    if missing:
        raise RuntimeError(f"standalone core is missing bundled resources: {missing}")

    if exercise_engine:
        from aiconfigurator_core.sdk.engine import EngineHandle

        engine = EngineHandle.compile(
            "MiniMaxAI/MiniMax-M2.5",
            "b200_sxm",
            "vllm",
            backend_version="0.19.0",
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
        )
        prefill_ms = engine.predict_prefill_latency(1, 1024, 0)
        decode_ms = engine.predict_decode_latency(1, 1024, 2)
        if not (prefill_ms > 0 and decode_ms > 0):
            raise RuntimeError(f"standalone core produced invalid latencies: {prefill_ms=}, {decode_ms=}")

    return core_version


def _verify_upper(*, import_runtime: bool) -> str:
    aic_version = _distribution_version("aiconfigurator")
    if aic_version is None:
        raise RuntimeError("aiconfigurator distribution is not installed")
    _require_distribution_files(
        "aiconfigurator",
        (
            "aiconfigurator/cli/main.py",
            "aiconfigurator/generator/api.py",
            "aiconfigurator/sdk/_compat.py",
            "aiconfigurator/sdk/engine.py",
            "aiconfigurator/sdk/memory.py",
            "aiconfigurator/sdk/task_v2.py",
            "aiconfigurator/webapp/main.py",
        ),
    )
    if import_runtime:
        for module in ("aiconfigurator.cli.main", "aiconfigurator.generator.api"):
            importlib.import_module(module)
    _forbid_module("spica")
    return aic_version


def _verify_legacy_sdk_compatibility() -> None:
    """Verify representative legacy aliases and the upper-owned Task API."""
    for module_name, public_name in (
        ("engine", "EngineHandle"),
        ("memory", "estimate_kv_cache"),
    ):
        canonical = importlib.import_module(f"aiconfigurator_core.sdk.{module_name}")
        legacy = importlib.import_module(f"aiconfigurator.sdk.{module_name}")
        if legacy is not canonical:
            raise RuntimeError(
                f"aiconfigurator.sdk.{module_name} is not the canonical aiconfigurator_core.sdk.{module_name} module"
            )
        if getattr(legacy, public_name) is not getattr(canonical, public_name):
            raise RuntimeError(f"legacy {public_name} is not the canonical core object")

    task_module = importlib.import_module("aiconfigurator.sdk.task_v2")
    if task_module.Task.__module__ != "aiconfigurator.sdk.task_v2":
        raise RuntimeError("Task must remain implemented by the upper aiconfigurator package")
    if importlib.util.find_spec("aiconfigurator_core.sdk.task_v2") is not None:
        raise RuntimeError("Task must not be shipped by the standalone core package")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("core", "full", "upper"), required=True)
    parser.add_argument("--exercise-engine", action="store_true")
    args = parser.parse_args()

    if args.expect == "core":
        core_version = _verify_core(exercise_engine=args.exercise_engine)
        if _distribution_version("aiconfigurator") is not None:
            raise RuntimeError("core-only install unexpectedly contains the aiconfigurator distribution")
        for module in ("aiconfigurator", "spica"):
            _forbid_module(module)
        print(
            f"Verified standalone aiconfigurator-core {core_version}, including canonical "
            "aiconfigurator_core.sdk imports, resources, and native extension"
        )
        return 0

    if args.expect == "full":
        core_version = _verify_core(exercise_engine=args.exercise_engine)
        aic_version = _verify_upper(import_runtime=True)
        if core_version != aic_version:
            raise RuntimeError(f"upper/core version mismatch: {aic_version=} {core_version=}")
        _verify_legacy_sdk_compatibility()
        print(
            f"Verified full aiconfigurator {aic_version} with standalone core, legacy SDK aliases, "
            "and upper-owned aiconfigurator.sdk.task_v2.Task"
        )
        return 0

    aic_version = _verify_upper(import_runtime=False)
    if _distribution_version("aiconfigurator-core") is not None:
        raise RuntimeError("upper-only install unexpectedly contains aiconfigurator-core metadata")
    _forbid_module("aiconfigurator_core")
    if importlib.util.find_spec("aiconfigurator.sdk.engine") is None:
        raise RuntimeError("upper-only install is missing the legacy aiconfigurator.sdk.engine wrapper")
    print(
        f"Verified upper-only aiconfigurator payload {aic_version}; legacy SDK wrappers remain installed "
        "and await the intentionally absent core dependency"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
