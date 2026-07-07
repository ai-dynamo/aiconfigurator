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
            "aiconfigurator/sdk/task_v2.py",
            "aiconfigurator_core/sdk/__init__.py",
            "aiconfigurator_core/sdk/task_v2.py",
        ),
    )

    core = importlib.import_module("aiconfigurator_core")
    if core._build_smoke() != 1:
        raise RuntimeError("native core extension returned an unexpected schema version")

    core_namespaced_task = importlib.import_module("aiconfigurator_core.sdk.task_v2").Task
    canonical_task = importlib.import_module("aiconfigurator.sdk.task_v2").Task
    if core_namespaced_task is not canonical_task:
        raise RuntimeError("aiconfigurator_core.sdk.task_v2.Task is not the canonical SDK Task class")

    for module in (
        "aiconfigurator.sdk.engine",
        "aiconfigurator.sdk.memory",
        "aiconfigurator.sdk.pareto_analysis",
    ):
        importlib.import_module(module)

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
