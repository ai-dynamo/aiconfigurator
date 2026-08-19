# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the Slurm launcher <-> worker rank-env contract (issue #1416).

`srun` exports the per-task global rank as ``SLURM_PROCID`` (and the
node-local id as ``SLURM_LOCALID``); it does not define a per-task ``RANK``.
The worker must therefore resolve its rank from ``RANK`` with a
``SLURM_PROCID`` fallback, and the launchers must not synthesize a fake
``RANK`` that would shadow the real one. Static (ast) checks because the
worker needs torch + tensorrt_llm + GPUs to import.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit

_WORKER = REPO_ROOT / "collector" / "network" / "slurm" / "collect_allreduce.py"
_LAUNCHERS = sorted((REPO_ROOT / "collector" / "network" / "slurm").glob("slurm_custom_ar_*gpu.sh"))


def _module_assignments(path: Path, name: str) -> list[ast.Assign]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]


def _environ_keys(node: ast.AST) -> list[str]:
    """String constants used to index/get os.environ within an AST subtree."""
    keys = []
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            s = child.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str):
                keys.append(s.value)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "get":
            keys.extend(arg.value for arg in child.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str))
    return keys


def test_worker_rank_resolves_rank_then_slurm_procid():
    assigns = _module_assignments(_WORKER, "rank")
    assert assigns, f"no module-level `rank = ...` assignment found in {_WORKER}"
    keys = _environ_keys(assigns[0].value)
    assert "RANK" in keys and "SLURM_PROCID" in keys, (
        "slurm/collect_allreduce.py must resolve its rank from RANK with an "
        "SLURM_PROCID fallback: srun exports SLURM_PROCID per task and never a "
        f"per-task RANK. Found env keys in the `rank` assignment: {keys}"
    )


def test_worker_rank_errors_when_neither_var_is_set():
    assigns = _module_assignments(_WORKER, "rank")
    assert assigns
    # A subscript (os.environ["SLURM_PROCID"]) raises KeyError instead of
    # silently collapsing every task to rank 0 (which would make every rank
    # the perf-row writer).
    assert any(isinstance(child, ast.Subscript) for child in ast.walk(assigns[0].value)), (
        "the `rank` resolution must raise (not default to 0) when neither RANK nor SLURM_PROCID is present"
    )


def test_launchers_rely_on_slurm_exported_rank():
    assert _LAUNCHERS, "no slurm_custom_ar_*gpu.sh launchers found"
    for launcher in _LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert "collect_allreduce.py" in text, f"{launcher.name} must invoke the slurm worker"
        assert "RANK=" not in text, (
            f"{launcher.name} must not synthesize RANK -- the worker's "
            "RANK-then-SLURM_PROCID resolution is the contract"
        )
