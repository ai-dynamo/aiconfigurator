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
_NETWORK_COLLECTOR = REPO_ROOT / "collector" / "network" / "collect_all_reduce.py"
_LAUNCHERS = [
    REPO_ROOT / "collector" / "network" / "slurm" / f"slurm_custom_ar_{world_size}gpu.sh"
    for world_size in (2, 4, 8, 16)
]


def _module_assignments(path: Path, name: str) -> list[ast.Assign]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]


def _load_pure_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("source", [_WORKER, _NETWORK_COLLECTOR])
def test_rank_resolver_prefers_rank_then_slurm_procid_and_fails_closed(source):
    resolve_rank = _load_pure_function(source, "_resolve_rank")

    assert resolve_rank({"RANK": "7", "SLURM_PROCID": "3"}) == 7
    assert resolve_rank({"SLURM_PROCID": "3"}) == 3
    with pytest.raises(KeyError, match="SLURM_PROCID"):
        resolve_rank({})


def test_worker_uses_the_rank_resolver():
    assigns = _module_assignments(_WORKER, "rank")
    assert assigns, f"no module-level `rank = ...` assignment found in {_WORKER}"
    resolution = assigns[0].value
    assert isinstance(resolution, ast.Call)
    assert isinstance(resolution.func, ast.Name) and resolution.func.id == "_resolve_rank"


def test_network_collector_uses_the_rank_resolver():
    tree = ast.parse(_NETWORK_COLLECTOR.read_text(encoding="utf-8"))
    benchmark = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "allreduce_benchmark"
    )
    calls = [
        node
        for node in ast.walk(benchmark)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_resolve_rank"
    ]
    assert calls, "allreduce_benchmark must resolve every Slurm backend rank through _resolve_rank"


def test_launchers_rely_on_slurm_exported_rank():
    assert _LAUNCHERS, "no slurm_custom_ar_*gpu.sh launchers found"
    for launcher in _LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert "collect_allreduce.py" in text, f"{launcher.name} must invoke the slurm worker"
        assert "RANK=" not in text, (
            f"{launcher.name} must not synthesize RANK -- the worker's "
            "RANK-then-SLURM_PROCID resolution is the contract"
        )


def test_launchers_bind_one_gpu_per_task():
    for launcher in _LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert "#SBATCH --gpus-per-node=" in text
        assert "#SBATCH --gpus " not in text
        assert "--gpus-per-task=1" in text
        assert "--gpu-bind=single:1" in text
