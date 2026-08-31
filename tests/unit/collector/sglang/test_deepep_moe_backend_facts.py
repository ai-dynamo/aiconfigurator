# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]


def test_sglang_deepep_moe_facts_are_covered_by_kernel_registry():
    mappings = yaml.safe_load((ROOT / "collector/kernel_source_backends.yaml").read_text())["mappings"]
    matches = [row for row in mappings if row.get("framework") == "sglang" and row.get("kernel_source") == "deepep_moe"]
    assert matches == [
        {
            "framework": "sglang",
            "kernel_source": "deepep_moe",
            "backend": "deepep",
            "source": "collector/wideep/sglang/collect_deepep_moe.py (DeepEP-MoE compute path)",
        }
    ]

    facts = yaml.safe_load((ROOT / "collector/op_backend_facts.yaml").read_text())["ops"]
    moe_rows = [
        row
        for op in facts
        if op["op_file"] in {"wideep_context_moe_perf", "wideep_generation_moe_perf"}
        for row in op["facts"]
        if row["framework"] == "sglang" and row["backends"] == ["deepep"]
    ]
    assert moe_rows
    assert {tuple(row["kernel_sources"]) for row in moe_rows} == {("deepep_moe",)}
