# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation-time memory-feasibility filter of the vLLM MLA module collector.

The filter is the one sanctioned in-collector case filter
(.claude/rules/collector/layer_permissions.md): size vs capacity, applied
inside get_generation_test_cases, drops counted and logged. These tests load
the real function bodies from source (vLLM itself is not importable in CI)
and pin the two properties that matter:

- a case whose KV lower bound exceeds the device budget is never queued, and
  the drop is logged in the canonical format;
- on >= 90 GB devices (H20/B200/RTX PRO 6000) the filter keeps the full
  declared sweep, i.e. it cannot change SM90/SM100/SM120 behavior.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MLA_MODULE_SOURCE = ROOT / "collector/vllm/collect_mla_module.py"


def _load_function(path: Path, name: str, namespace: dict):
    """Compile the real function body without importing unavailable vLLM."""
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _load_assignment(path: Path, name: str):
    """Read a module-level constant from source."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    expr = ast.fix_missing_locations(ast.Expression(node.value))
                    return eval(compile(expr, str(path), "eval"), {})  # noqa: S307 — our own source
    raise AssertionError(f"{name} not found in {path}")


L40S_TOTAL_MEMORY = 47_669_248_000  # 44.39 GiB usable, as reported by torch
H20_TOTAL_MEMORY = 102_600_000_000  # ~96 GB class devices (H20 / RTX PRO 6000)

SWEEP = SimpleNamespace(
    inner_sweep_head_counts=[128],
    generation_batch_sizes=[256, 512, 1024],
    generation_sequence_lengths=[16384, 32768, 65536, 131072],
    generation_max_tokens=2**25,
)


def _build_get_generation_test_cases(total_memory, combos):
    namespace = {
        "get_mla_module_sweep_spec": lambda backend: SWEEP,
        "_get_precision_combos": lambda phase, attn_type: combos,
        "_device_total_memory_bytes": lambda: total_memory,
        "_MLA_KV_ENTRY_ELEMS": _load_assignment(MLA_MODULE_SOURCE, "_MLA_KV_ENTRY_ELEMS"),
        "_MEMORY_BUDGET_SAFETY_FACTOR": _load_assignment(MLA_MODULE_SOURCE, "_MEMORY_BUDGET_SAFETY_FACTOR"),
    }
    _load_function(MLA_MODULE_SOURCE, "_generation_kv_footprint_bytes", namespace)
    return _load_function(MLA_MODULE_SOURCE, "get_generation_test_cases", namespace)


BF16_COMBO = [("bfloat16", "bfloat16", "bfloat16")]


def test_memory_filter_drops_infeasible_cases_and_logs(capsys):
    get_cases = _build_get_generation_test_cases(L40S_TOTAL_MEMORY, BF16_COMBO)
    cases = get_cases("mla")
    shapes = {(s, b) for s, b, *_ in cases}
    # 2 x 33.5M tokens x 576 x 2B = 77.3 GB cannot fit on a 46 GB device...
    assert (32768, 1024) not in shapes
    assert (65536, 512) not in shapes
    assert (131072, 256) not in shapes
    # ...while the 16.7M-token points (2 x 19.3 GB) stay queued.
    assert (16384, 1024) in shapes
    assert (65536, 256) in shapes
    out = capsys.readouterr().out
    assert "mla_generation_module: dropped 3/" in out
    assert "(memory budget, device=44GiB)" in out


def test_memory_filter_uses_dsa_op_name_in_log(capsys):
    get_cases = _build_get_generation_test_cases(L40S_TOTAL_MEMORY, BF16_COMBO)
    get_cases("dsa")
    assert "dsa_generation_module: dropped" in capsys.readouterr().out


def test_memory_filter_keeps_full_sweep_on_90gb_class_devices(capsys):
    get_cases = _build_get_generation_test_cases(H20_TOTAL_MEMORY, BF16_COMBO)
    cases = get_cases("mla")
    # Every token-feasible point of the declared sweep must stay queued:
    # the filter cannot change H20/B200/RTX PRO 6000 collection behavior.
    expected = sum(
        1
        for b in SWEEP.generation_batch_sizes
        for s in SWEEP.generation_sequence_lengths
        if b * s <= SWEEP.generation_max_tokens
    )
    assert len(cases) == expected
    assert (32768, 1024) in {(s, b) for s, b, *_ in cases}
    assert "dropped" not in capsys.readouterr().out


def test_memory_filter_inactive_without_cuda(capsys):
    get_cases = _build_get_generation_test_cases(None, BF16_COMBO)
    cases = get_cases("mla")
    assert (32768, 1024) in {(s, b) for s, b, *_ in cases}
    assert "dropped" not in capsys.readouterr().out


def test_memory_filter_fp8_kv_halves_the_bound(capsys):
    fp8_combo = [("bfloat16", "fp8", "bfloat16")]
    get_cases = _build_get_generation_test_cases(L40S_TOTAL_MEMORY, fp8_combo)
    cases = get_cases("mla")
    # 2 x 33.5M x 576 x 1B = 38.7 GB fits the 42.9 GB budget.
    assert (32768, 1024) in {(s, b) for s, b, *_ in cases}
    assert "dropped" not in capsys.readouterr().out
