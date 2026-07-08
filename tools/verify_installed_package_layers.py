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

# Running this file directly prepends ``tools/`` to sys.path. That directory
# contains helper scripts under ``tools/spica/``, which must not be mistaken for
# the separately packaged top-level ``spica`` namespace during isolation checks.
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


def _distribution_file_names(name: str) -> set[str]:
    distribution = importlib.metadata.distribution(name)
    return {str(path).replace("\\", "/") for path in distribution.files or ()}


def _require_distribution_sdk_mirrors(name: str) -> None:
    """Require the core distribution to own complete files under both SDK names."""
    distribution = importlib.metadata.distribution(name)
    files = {str(path).replace("\\", "/"): path for path in distribution.files or ()}
    prefixes = ("aiconfigurator/sdk/", "aiconfigurator_core/sdk/")
    trees: list[dict[str, object]] = []

    for prefix in prefixes:
        tree = {
            filename.removeprefix(prefix): package_path
            for filename, package_path in files.items()
            if filename.startswith(prefix) and filename.endswith(".py")
        }
        if not tree:
            raise RuntimeError(f"distribution {name!r} owns no Python SDK files under {prefix}")
        missing = [
            f"{prefix}{relative}"
            for relative, package_path in tree.items()
            if not distribution.locate_file(package_path).is_file()
        ]
        if missing:
            raise RuntimeError(f"distribution {name!r} is missing installed SDK files: {missing}")
        trees.append(tree)

    if set(trees[0]) != set(trees[1]):
        canonical_only = sorted(set(trees[0]) - set(trees[1]))
        core_only = sorted(set(trees[1]) - set(trees[0]))
        raise RuntimeError(
            f"distribution {name!r} has different SDK leaf sets: "
            f"aiconfigurator-only={canonical_only}, aiconfigurator_core-only={core_only}"
        )


def _forbid_distribution_prefixes(name: str, prefixes: tuple[str, ...]) -> None:
    misplaced = sorted(path for path in _distribution_file_names(name) if path.startswith(prefixes))
    if misplaced:
        raise RuntimeError(f"distribution {name!r} must not own files under {prefixes}: {misplaced}")


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
            "aiconfigurator/sdk/common.py",
            "aiconfigurator/sdk/engine.py",
            "aiconfigurator/sdk/memory.py",
            "aiconfigurator/sdk/task_v2.py",
            "aiconfigurator_core/sdk/common.py",
            "aiconfigurator_core/sdk/engine.py",
            "aiconfigurator_core/sdk/memory.py",
            "aiconfigurator_core/sdk/task_v2.py",
        ),
    )
    _require_distribution_sdk_mirrors("aiconfigurator-core")

    core = importlib.import_module("aiconfigurator_core")
    if core._build_smoke() != 1:
        raise RuntimeError("native core extension returned an unexpected schema version")

    # The wheel materializes the symlinked source under two import names. Both
    # APIs must work, but Python may create distinct module and class objects.
    # Deliberately do not impose an identity contract between the namespaces.
    for namespace in ("aiconfigurator.sdk", "aiconfigurator_core.sdk"):
        engine_module = importlib.import_module(f"{namespace}.engine")
        memory_module = importlib.import_module(f"{namespace}.memory")
        importlib.import_module(f"{namespace}.pareto_analysis")
        task_module = importlib.import_module(f"{namespace}.task_v2")
        for module, public_name in (
            (engine_module, "EngineHandle"),
            (memory_module, "estimate_kv_cache"),
            (task_module, "Task"),
        ):
            if not hasattr(module, public_name):
                raise RuntimeError(f"{module.__name__} is missing public API {public_name}")

    resources = importlib.resources.files("aiconfigurator")
    required_resources = (
        resources / "model_configs" / "meta-llama--Meta-Llama-3.1-8B_config.json",
        resources / "systems" / "h100_sxm.yaml",
        resources / "systems" / "data" / "b200_sxm" / "vllm" / "0.19.0" / "gemm_perf.parquet",
    )
    missing = [str(path) for path in required_resources if not path.is_file()]
    if missing:
        raise RuntimeError(f"standalone core is missing bundled resources: {missing}")

    if exercise_engine:
        from aiconfigurator.sdk.engine import EngineHandle

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
            "aiconfigurator/webapp/main.py",
            "spica/config.py",
        ),
    )
    _forbid_distribution_prefixes(
        "aiconfigurator",
        ("aiconfigurator/sdk/", "aiconfigurator_core/sdk/"),
    )
    if import_runtime:
        for module in ("aiconfigurator.cli.main", "aiconfigurator.generator.api", "spica.config"):
            importlib.import_module(module)
    return aic_version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("core", "full", "upper"), required=True)
    parser.add_argument("--exercise-engine", action="store_true")
    args = parser.parse_args()

    if args.expect == "core":
        core_version = _verify_core(exercise_engine=args.exercise_engine)
        if _distribution_version("aiconfigurator") is not None:
            raise RuntimeError("core-only install unexpectedly contains the aiconfigurator distribution")
        for module in ("aiconfigurator.cli", "aiconfigurator.generator", "aiconfigurator.webapp", "spica"):
            _forbid_module(module)
        print(
            f"Verified standalone aiconfigurator-core {core_version}, including "
            "from aiconfigurator_core.sdk.task_v2 import Task"
        )
        return 0

    if args.expect == "full":
        core_version = _verify_core(exercise_engine=args.exercise_engine)
        aic_version = _verify_upper(import_runtime=True)
        if core_version != aic_version:
            raise RuntimeError(f"upper/core version mismatch: {aic_version=} {core_version=}")
        print(
            f"Verified full aiconfigurator {aic_version} with standalone core, including "
            "from aiconfigurator.sdk.task_v2 import Task"
        )
        return 0

    aic_version = _verify_upper(import_runtime=False)
    if _distribution_version("aiconfigurator-core") is not None:
        raise RuntimeError("upper-only install unexpectedly contains aiconfigurator-core metadata")
    for module in ("aiconfigurator.sdk", "aiconfigurator_core"):
        _forbid_module(module)
    print(f"Verified upper-only aiconfigurator payload {aic_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
