# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the MNNVL dtype gate in the all-reduce collectors (issue #1416).

TRT-LLM `AllReduce(mapping, dtype)` only builds the MNNVL path when dtype is
passed at construction; dtype=None silently falls back to the heuristic path.
Static (ast) checks because the collectors need tensorrt_llm + GPUs to import.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit

_COLLECTOR = REPO_ROOT / "collector" / "network" / "collect_all_reduce.py"
_SLURM_WORKER = REPO_ROOT / "collector" / "network" / "slurm" / "collect_allreduce.py"


def _allreduce_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # slurm/collect_allreduce.py: AllReduce(...)
        if isinstance(func, ast.Name) and func.id == "AllReduce":
            calls.append(node)
        # collector/network/collect_all_reduce.py: trtllm_mods["AllReduce"](...)
        if isinstance(func, ast.Subscript) and isinstance(func.slice, ast.Constant) and func.slice.value == "AllReduce":
            calls.append(node)
    return calls


def _has_dtype_kw(call: ast.Call) -> bool:
    dtypes = [kw for kw in call.keywords if kw.arg == "dtype"]
    # dtype=None would keep the old broken behaviour; require a real value.
    return bool(dtypes) and not (isinstance(dtypes[0].value, ast.Constant) and dtypes[0].value.value is None)


def test_network_collector_passes_dtype_to_allreduce():
    calls = _allreduce_calls(_COLLECTOR)
    assert calls, f"no AllReduce construction found in {_COLLECTOR}"
    assert all(_has_dtype_kw(c) for c in calls), (
        "Every AllReduce(...) construction must pass dtype=<torch dtype> so "
        "MNNVL can engage on multi-node TP sweeps (issue #1416); "
        "dtype=None silently disables it."
    )


def test_slurm_worker_passes_dtype_to_allreduce():
    calls = _allreduce_calls(_SLURM_WORKER)
    assert calls, f"no AllReduce construction found in {_SLURM_WORKER}"
    assert all(_has_dtype_kw(c) for c in calls), (
        "slurm/collect_allreduce.py must pass dtype=torch.bfloat16 to "
        "AllReduce(...) so the slurm worker enables MNNVL on multi-node "
        "TP sweeps (issue #1416)."
    )
