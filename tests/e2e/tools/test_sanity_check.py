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


def _supported_system_backend_latest():
    """(system, backend, latest_version) for each system+backend that AIC supports."""
    supported = get_supported_databases()
    result = []
    for system, backends in sorted(supported.items()):
        fail_ok = system in ["b60"]  # xpu
        for backend in sorted(backends.keys()):
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
