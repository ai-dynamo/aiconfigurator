# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]


def _literal_kernel_sources(module_name: str) -> set[str]:
    module_path = ROOT / Path(*module_name.split(".")).with_suffix(".py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    labels: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "kernel_source":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                labels.add(keyword.value.value)
            elif isinstance(keyword.value, ast.Name) and keyword.value.id in constants:
                labels.add(constants[keyword.value.id])
    return labels


def test_every_active_sglang_wideep_producer_label_has_verified_backend_facts():
    registry = importlib.import_module("collector.wideep.sglang.registry").REGISTRY
    mappings = yaml.safe_load((ROOT / "collector/kernel_source_backends.yaml").read_text())["mappings"]
    producer_labels = {entry.module: _literal_kernel_sources(entry.module) for entry in registry}
    assert producer_labels
    assert all(producer_labels.values())

    uncovered = {}
    for module_name, labels in producer_labels.items():
        for label in labels:
            matches = [
                row for row in mappings if row.get("framework") == "sglang" and row.get("kernel_source") == label
            ]
            if not matches or any(row.get("backend") == "unverified" for row in matches):
                uncovered.setdefault(module_name, []).append(label)
    assert uncovered == {}

    facts = yaml.safe_load((ROOT / "collector/op_backend_facts.yaml").read_text())["ops"]
    moe_rows = [
        row
        for op in facts
        if op["op_file"] in {"wideep_context_moe_perf", "wideep_generation_moe_perf"}
        for row in op["facts"]
        if row["framework"] == "sglang" and row["backends"] == ["deepep"]
    ]
    assert moe_rows
    # The committed 0.5.6/0.5.10 data predates the producer-label cleanup and
    # therefore keeps the historical ``deepepmoe`` evidence key.  Both that
    # key and the active ``deepep_moe`` key map to the same verified backend.
    assert {tuple(row["kernel_sources"]) for row in moe_rows} == {("deepepmoe",)}
