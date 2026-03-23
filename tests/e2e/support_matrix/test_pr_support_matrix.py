# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight support-matrix smoke test for PR validation.

Runs the same logic as the daily ``generate_support_matrix`` workflow but only
for a curated subset of representative models that cover the major architecture
families (dense, MoE, DeepSeek MLA, hybrid Mamba, etc.).

For every (model, system, backend, version) combination the test:
  1. Queries ``cli_support`` to decide if the combo is expected to work.
  2. If unsupported → the test is skipped.
  3. If supported   → runs the full ``SupportMatrix.run_single_test`` pipeline
     and **fails** the test when the result disagrees with the matrix.
"""

from __future__ import annotations

from functools import cache

import pytest

from aiconfigurator.cli.api import cli_support
from aiconfigurator.sdk.common import BackendName
from aiconfigurator.sdk.perf_database import get_latest_database_version

pytestmark = [pytest.mark.e2e, pytest.mark.build, pytest.mark.support_matrix]

# Representative models — one per major architecture family.
PR_MODELS: list[str] = [
    "nvidia/DeepSeek-V3.1-NVFP4",
    "meta-llama/Meta-Llama-3.1-8B",
    "MiniMaxAI/MiniMax-M2.5",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "Qwen/Qwen3-235B-A22B",
    "openai/gpt-oss-20b",
]

PR_SYSTEMS: list[str] = ["h100_sxm", "b200_sxm"]
PR_BACKENDS: list[str] = sorted(b.value for b in BackendName)


@cache
def _latest_version(system: str, backend: str) -> str | None:
    return get_latest_database_version(system=system, backend=backend)


def _build_param_grid() -> list[pytest.param]:
    """Build a flat list of pytest params for every valid (model, system, backend, version) combo."""
    params: list[pytest.param] = []
    for model in PR_MODELS:
        short_model = model.rsplit("/", 1)[-1]
        for system in PR_SYSTEMS:
            for backend in PR_BACKENDS:
                version = _latest_version(system, backend)
                if version is None:
                    continue
                params.append(
                    pytest.param(
                        model,
                        system,
                        backend,
                        version,
                        id=f"{short_model}-{system}-{backend}-v{version}",
                    )
                )
    return params


@pytest.mark.parametrize("model, system, backend, version", _build_param_grid())
def test_pr_support_matrix(model: str, system: str, backend: str, version: str):
    """Validate that supported model/system/backend combos still produce results."""
    agg_supported, disagg_supported = cli_support(model, system, backend=backend, backend_version=version)

    if not agg_supported and not disagg_supported:
        pytest.skip(f"Not supported: {model} on {system}/{backend} v{version}")

    from tools.support_matrix.support_matrix import SupportMatrix

    sm = SupportMatrix()
    success_dict, error_dict = sm.run_single_test(
        model=model,
        system=system,
        backend=backend,
        version=version,
    )

    failures: list[str] = []
    for mode, expected in [("agg", agg_supported), ("disagg", disagg_supported)]:
        if expected and not success_dict[mode]:
            error_msg = error_dict[mode] or "no error message captured"
            failures.append(f"  {mode}: expected PASS but got FAIL — {error_msg}")

    if failures:
        detail = "\n".join(failures)
        pytest.fail(f"Support matrix regression for {model} on {system}/{backend} v{version}:\n{detail}")
