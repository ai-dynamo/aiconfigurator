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
  4. For supported modes, compares Rust engine-step output against Python.
"""

from __future__ import annotations

from functools import cache

import pytest

from aiconfigurator.cli.api import cli_support
from aiconfigurator.sdk.perf_database import get_latest_database_version

pytestmark = [pytest.mark.e2e, pytest.mark.build, pytest.mark.support_matrix]

# Representative model/system/backend cases. Keep this intentionally small:
# the full cross product belongs in the scheduled support-matrix workflow,
# while PR e2e has a 30-minute job timeout shared with the rest of the suite.
# Add cases only when the whole e2e job still fits that budget.
PR_CASES: list[tuple[str, str, str]] = [
    ("nvidia/DeepSeek-V3.1-NVFP4", "b200_sxm", "trtllm"),
    ("meta-llama/Meta-Llama-3.1-8B", "b200_sxm", "sglang"),
    ("MiniMaxAI/MiniMax-M2.5", "b200_sxm", "vllm"),
    ("openai/gpt-oss-20b", "h100_sxm", "trtllm"),
]


@cache
def _latest_version(system: str, backend: str) -> str | None:
    return get_latest_database_version(system=system, backend=backend)


def _build_param_grid() -> list[pytest.param]:
    """Build pytest params for the curated PR smoke cases."""
    params: list[pytest.param] = []
    for model, system, backend in PR_CASES:
        short_model = model.rsplit("/", 1)[-1]
        version = _latest_version(system, backend)
        if version is None:
            params.append(
                pytest.param(
                    model,
                    system,
                    backend,
                    "",
                    id=f"{short_model}-{system}-{backend}-no-database",
                )
            )
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
    """Validate that supported model/system/backend combos produce Python/Rust comparable results."""
    if not version:
        pytest.fail(f"No latest database version found for {system=}, {backend=}")

    agg_supported, disagg_supported = cli_support(model, system, backend=backend, backend_version=version)

    if not agg_supported and not disagg_supported:
        pytest.skip(f"Not supported: {model} on {system}/{backend} v{version}")

    from tools.support_matrix.support_matrix import SupportMatrix

    success_dict, error_dict = SupportMatrix.run_single_test(
        model=model,
        system=system,
        backend=backend,
        version=version,
        compare_engine_step_backends=True,
    )

    failures: list[str] = []
    for mode, expected in [("agg", agg_supported), ("disagg", disagg_supported)]:
        if expected and not success_dict[mode]:
            error_msg = error_dict[mode] or "no error message captured"
            failures.append(f"  {mode}: expected PASS but got FAIL — {error_msg}")

    if failures:
        detail = "\n".join(failures)
        pytest.fail(f"Support matrix regression for {model} on {system}/{backend} v{version}:\n{detail}")
