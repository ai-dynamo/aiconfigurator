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
_LAUNCHERS = [
    REPO_ROOT / "collector" / "network" / "slurm" / f"slurm_custom_ar_{world_size}gpu.sh" for world_size in (8, 16)
]


def _module_assignments(path: Path, name: str) -> list[ast.Assign]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _is_environ_get(node: ast.AST, key: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and _is_os_environ(node.func.value)
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == key
        and not node.keywords
    )


def _is_environ_subscript(node: ast.AST, key: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and _is_os_environ(node.value)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key
    )


def test_worker_rank_resolves_rank_then_slurm_procid():
    assigns = _module_assignments(_WORKER, "rank")
    assert assigns, f"no module-level `rank = ...` assignment found in {_WORKER}"
    conversion = assigns[0].value
    assert (
        isinstance(conversion, ast.Call)
        and isinstance(conversion.func, ast.Name)
        and conversion.func.id == "int"
        and len(conversion.args) == 1
        and not conversion.keywords
    )
    resolution = conversion.args[0]
    assert isinstance(resolution, ast.BoolOp) and isinstance(resolution.op, ast.Or)
    assert len(resolution.values) == 2
    assert _is_environ_get(resolution.values[0], "RANK")
    assert _is_environ_subscript(resolution.values[1], "SLURM_PROCID"), (
        "slurm/collect_allreduce.py must resolve its rank from RANK with an "
        "SLURM_PROCID fallback: srun exports SLURM_PROCID per task and never a "
        "per-task RANK."
    )


def test_worker_rank_errors_when_neither_var_is_set():
    assigns = _module_assignments(_WORKER, "rank")
    assert assigns
    conversion = assigns[0].value
    assert isinstance(conversion, ast.Call) and len(conversion.args) == 1
    resolution = conversion.args[0]
    assert isinstance(resolution, ast.BoolOp) and len(resolution.values) == 2
    fallback = resolution.values[1]
    # os.environ["SLURM_PROCID"] raises KeyError instead of
    # silently collapsing every task to rank 0 (which would make every rank
    # the perf-row writer).
    assert _is_environ_subscript(fallback, "SLURM_PROCID"), (
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


def test_launchers_bind_one_gpu_per_task_and_match_worker_device_indexing():
    worker = _WORKER.read_text(encoding="utf-8")
    assert "visible_device_count = torch.cuda.device_count()" in worker
    assert "if visible_device_count != 1:" in worker
    assert "device_index = 0" in worker
    assert "torch.cuda.set_device(device_index)" in worker
    assert "cudart.cudaSetDevice(device_index)" in worker
    assert "torch.cuda.set_device(local_rank)" not in worker

    for launcher in _LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert "#SBATCH --gpus-per-node=4" in text
        assert "#SBATCH --gpus " not in text
        assert "--gpus-per-task=1" in text
        assert "--gpu-bind=single:1" in text


def test_launchers_use_rc20_generic_paths_without_model_specific_state():
    for launcher in _LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        assert "trtllm_aarch64_release_v1.3.0rc20.sqsh" in text
        assert "/path/to/aiconfigurator:/workspace/aiconfigurator" in text
        assert "/workspace/aiconfigurator/collector/network/slurm/collect_allreduce.py" in text
        assert "/kimi" not in text
        assert "TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED" not in text
