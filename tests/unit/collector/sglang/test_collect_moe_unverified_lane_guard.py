# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proves ``_raise_if_unverified_moe_lane`` (collect_moe.py) fires exactly
for the three lanes (int4_wo, w4a16_mxfp4, w4a8_mxfp4_mxfp8) whose dispatch
citations were never re-verified against sglang 0.5.17, and only when the
installed version isn't 0.5.14 -- never for Qwen3.8-Max's actual collected
lanes (bfloat16/fp8_block/nvfp4), at either version (code review, 2026-08-18,
AIC-1762 Task 4c/4d follow-up).

AST-extracts the function the same way ``test_collect_gdn_contract.py``'s
``TestResolveFlashinferGdnDecode`` and this directory's other collector
function tests do; ``pkg_resources.get_distribution`` is stubbed (no
sys.modules injection needed here -- the guard reads the installed version
through ``pkg_resources``, not through an import-time capability probe like
the runner-backend pin), and the real ``collector.version_resolver.
_check_compat`` is used unmocked so the test exercises the actual version
grammar, not a re-implementation of it.
"""

import ast
import types
from pathlib import Path

import pytest

from collector.version_resolver import _check_compat

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "sglang" / "collect_moe.py"

UNVERIFIED_LANES = ["int4_wo", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"]
QWEN38MAX_LANES = ["bfloat16", "fp8_block", "nvfp4"]


def _load_guard(installed_version: str):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_raise_if_unverified_moe_lane"
    )
    fake_distribution = types.SimpleNamespace(version=installed_version)
    fake_pkg_resources = types.SimpleNamespace(get_distribution=lambda _name: fake_distribution)
    loaded = {"pkg_resources": fake_pkg_resources, "_check_compat": _check_compat}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded["_raise_if_unverified_moe_lane"]


@pytest.mark.parametrize("moe_type", UNVERIFIED_LANES)
def test_guard_fires_at_0517_for_unverified_lanes(moe_type):
    guard = _load_guard("0.5.17")

    with pytest.raises(RuntimeError, match=moe_type):
        guard(moe_type)


@pytest.mark.parametrize("moe_type", UNVERIFIED_LANES)
def test_guard_stays_silent_at_0514_for_unverified_lanes(moe_type):
    guard = _load_guard("0.5.14")

    guard(moe_type)  # must not raise


@pytest.mark.parametrize("installed_version", ["0.5.14", "0.5.17"])
@pytest.mark.parametrize("moe_type", QWEN38MAX_LANES)
def test_guard_never_fires_for_qwen38max_collected_lanes(moe_type, installed_version):
    """The three lanes this bump actually verified must never be gated by
    this guard, at either pinned version -- a bug that widened the guard's
    scope would silently break Qwen3.8-Max collection."""
    guard = _load_guard(installed_version)

    guard(moe_type)  # must not raise
