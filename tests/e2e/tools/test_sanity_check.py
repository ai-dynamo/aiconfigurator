# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess as sp
import sys

import pytest

from aiconfigurator.sdk.perf_database import (
    get_latest_database_version,
    get_supported_databases,
)

pytestmark = [pytest.mark.e2e, pytest.mark.build]

SANITY_CHECK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../tools/sanity_check"))

# Always-validated combo: keeps one loader + query smoke path (strict
# provenance, warm_all_op_data, bf16 coverage gate) alive on PRs that touch
# no perf data at all.
SENTINEL_COMBO = ("h200_sxm", "trtllm")

# CI scopes this suite to the PR's diff via AIC_SANITY_TARGETS (build-test.yml):
#   unset -> full matrix (local/manual runs keep the historical behavior)
#   "all" -> full matrix (sanity-check tooling or the DB loader changed)
#   ""    -> sentinel combo only (no perf data changed)
#   newline-separated changed paths under .../systems/ -> sentinel + the
#     (system, backend) combos those paths belong to


def _combos_from_changed_paths(changed_paths, supported):
    combos = set()
    for path in changed_paths:
        parts = [p for p in path.strip().split("/") if p]
        if "systems" not in parts:
            continue
        rest = parts[parts.index("systems") + 1 :]
        if not rest:
            continue
        if rest[0] != "data":
            # systems/<system>.yaml spec edits feed SOL math for every
            # backend of that system.
            system = rest[0].removesuffix(".yaml")
            if rest[0].endswith(".yaml") and system in supported:
                combos.update((system, backend) for backend in supported[system])
            continue
        if len(rest) < 2:
            continue
        system = rest[1]
        backends = supported.get(system)
        if not backends:
            continue
        backend = next((p for p in rest[2:] if p in backends), None)
        if backend is not None:
            combos.add((system, backend))
        else:
            # Shared data with no backend path component (e.g. comm/nccl)
            # feeds every backend of the system.
            combos.update((system, b) for b in backends)
    return combos


def _selected_combos(supported):
    targets = os.environ.get("AIC_SANITY_TARGETS")
    if targets is None or targets.strip() == "all":
        return {(system, backend) for system, backends in supported.items() for backend in backends}
    combos = _combos_from_changed_paths(targets.splitlines(), supported)
    combos.add(SENTINEL_COMBO)
    return combos


def _supported_system_backend_latest():
    """(system, backend, latest_version) for each selected system+backend combo."""
    supported = get_supported_databases()
    result = []
    for system, backend in sorted(_selected_combos(supported)):
        if backend not in supported.get(system, {}):
            continue
        fail_ok = system in ["b60"]  # xpu
        version = get_latest_database_version(system, backend)
        if version is not None:
            result.append((system, backend, version, fail_ok))
    return result


@pytest.mark.parametrize(
    "system,backend,version,fail_ok",
    _supported_system_backend_latest(),
)
def test_validate_database(system, backend, version, fail_ok):
    """
    Test that validate_database.ipynb runs successfully for the latest
    backend version of each system+backend combination that AIC supports.
    """
    env = {
        **os.environ,
        "AIC_VALIDATE_SYSTEM": system,
        "AIC_VALIDATE_BACKEND": backend,
        "AIC_VALIDATE_VERSION": version,
        "MPLBACKEND": "agg",
    }
    try:
        result = sp.run(
            [sys.executable, "-c", "import import_ipynb; import validate_database"],
            cwd=SANITY_CHECK_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except sp.TimeoutExpired:
        error_message = f"validate_database timed out (300s) for {system}/{backend}/{version}"
        if fail_ok:
            pytest.xfail(error_message)
        pytest.fail(error_message)
    success = result.returncode == 0
    error_message = (
        f"validate_database failed for {system}/{backend}/{version}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    if fail_ok and not success:
        pytest.xfail(error_message)

    assert success, error_message
