# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard the MNNVL dtype gate in the all-reduce collectors (issue #1416).

TRT-LLM `AllReduce(mapping, dtype)` only builds the MNNVL path when dtype is
passed at construction; dtype=None silently falls back to the heuristic path.
Static (ast) checks because the collectors need tensorrt_llm + GPUs to import.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit

_COLLECTOR = REPO_ROOT / "collector" / "network" / "collect_all_reduce.py"
_SLURM_WORKER = REPO_ROOT / "collector" / "network" / "slurm" / "collect_allreduce.py"
_GB300_TRTLLM_DATA = (
    REPO_ROOT
    / "aic-core"
    / "src"
    / "aiconfigurator_core"
    / "systems"
    / "data"
    / "gb300"
    / "comm"
    / "trtllm"
    / "1.3.0rc20"
    / "custom_allreduce_perf.parquet"
)


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


def _load_pure_function(path: Path, name: str):
    """Compile one dependency-free function without importing the GPU collector."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


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


def test_trtllm_kernel_source_records_actual_mnnvl_variant():
    kernel_source = _load_pure_function(_COLLECTOR, "_trtllm_mnnvl_kernel_source")
    allreduce = SimpleNamespace(mnnvl_allreduce=object())
    ops = SimpleNamespace(_MNNVL_ONE_SHOT_THRESHOLD_BYTES=1024)

    one_shot_tensor = SimpleNamespace(numel=lambda: 64, element_size=lambda: 2)
    assert kernel_source(allreduce, one_shot_tensor, 8, ops) == "TRTLLM_MNNVL_oneshot"

    two_shot_tensor = SimpleNamespace(numel=lambda: 65, element_size=lambda: 2)
    assert kernel_source(allreduce, two_shot_tensor, 8, ops) == "TRTLLM_MNNVL_twoshot"


def test_trtllm_kernel_source_rejects_unobservable_fallback_or_api_drift():
    kernel_source = _load_pure_function(_COLLECTOR, "_trtllm_mnnvl_kernel_source")
    tensor = SimpleNamespace(numel=lambda: 64, element_size=lambda: 2)
    ops = SimpleNamespace(_MNNVL_ONE_SHOT_THRESHOLD_BYTES=1024)

    with pytest.raises(RuntimeError, match="regular fallback"):
        kernel_source(SimpleNamespace(mnnvl_allreduce=None), tensor, 8, ops)
    with pytest.raises(RuntimeError, match="variant threshold"):
        kernel_source(SimpleNamespace(mnnvl_allreduce=object()), tensor, 8, SimpleNamespace())


def test_gb300_trtllm_multinode_rows_record_mnnvl_variant():
    import pyarrow.parquet as pq

    rows = pq.read_table(
        _GB300_TRTLLM_DATA,
        columns=["num_gpus", "message_size", "kernel_source"],
    ).to_pylist()
    assert len(rows) == 92
    assert len({(row["num_gpus"], row["message_size"]) for row in rows}) == len(rows)

    for row in rows:
        tp_size = row["num_gpus"]
        if tp_size < 8:
            assert row["kernel_source"] == "TRTLLM"
            continue
        aggregate_bytes = row["message_size"] * 2 * tp_size
        variant = "oneshot" if aggregate_bytes <= 64 * 1024 * 8 * 2 else "twoshot"
        assert row["kernel_source"] == f"TRTLLM_MNNVL_{variant}"
