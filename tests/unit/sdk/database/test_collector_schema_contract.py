# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-sided schema contract for the retained ``moe_a2a`` table.

The collector writes the table and the compiled engine consumes it through
``fetch_table_view``. Each side pins the header independently; this module is
the SDK-side half. The collector-side twins are:

- ``tests/unit/collector/test_collect_moe_a2a.py::MOE_A2A_HEADER``
- ``tests/unit/collector/test_collect_trtllm_alltoall.py::MOE_A2A_HEADER``

This file MUST NOT import anything from ``collector`` (module-boundary rule:
SDK tests do not reach into the collector). The twin literals are verified by
reading the twin test files as text; a drifting header breaks one side's pin
before it can silently break the cross-module contract.

Each test writes ONE synthetic row under the frozen header (column order
preserved), round-trips it through the ENGINE table view, and asserts the
nested key plus the unit convention. The view's column readers look up the
frozen header's column names directly, so a renamed or dropped column fails
at the fold — a stronger pin than the retired loaders' ``row.get`` defaults.
The ``moe_a2a`` writer records MICROSECONDS and the view converts them to
milliseconds.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
from aiconfigurator_core.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]

# Copied verbatim from the collector-side writer pins:
# tests/unit/collector/test_collect_moe_a2a.py::MOE_A2A_HEADER (sglang DeepEP
# writer) and tests/unit/collector/test_collect_trtllm_alltoall.py::
# MOE_A2A_HEADER (trtllm NVLink alltoall writer) — both collectors emit this
# exact header. No ``power`` column by design: per-phase power needs
# winning-config re-runs (a hardware measurement-method design), and
# log_perf's header-from-first-row property makes partial power columns
# unrepresentable; the loader tolerates absence.
MOE_A2A_HEADER = (
    "framework,version,device,op_name,kernel_source,"
    "comm_backend,phase,comm_dtype,ep_size,node_num,hidden_size,topk,num_experts,"
    "num_tokens,sms,transmit_us,notify_us,latency"
)

# The collector twin files, with the literal each must pin. Paths are relative
# to the repo root; existence itself is part of the contract (Tasks 2-5 landed
# the writers and their pins).
_TWIN_PINS = {
    "tests/unit/collector/test_collect_moe_a2a.py": ("MOE_A2A_HEADER", MOE_A2A_HEADER),
    "tests/unit/collector/test_collect_trtllm_alltoall.py": ("MOE_A2A_HEADER", MOE_A2A_HEADER),
}


def _view_db_over_row(tmp_path, header: str, row: dict, filename: str) -> PerfDatabase:
    """Write one row to parquet with the header's exact columns and order,
    inside a minimal systems tree the engine view can serve."""
    columns = header.split(",")
    assert set(row) == set(columns), "synthetic row must cover the frozen header exactly"
    root = tmp_path / "systems"
    root.mkdir(exist_ok=True)
    (root / "h100_sxm.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data/h100_sxm",
                "gpu": {
                    "sm_version": 90,
                    "mem_bw": 4_800_000_000_000.0,
                    "mem_bw_empirical_scaling_factor": 0.8,
                    "mem_empirical_constant_latency": 0.000003,
                    "bfloat16_tc_flops": 989_000_000_000_000.0,
                    "fp8_tc_flops": 1_978_000_000_000_000.0,
                },
                "node": {
                    "num_gpus_per_node": 8,
                    "inter_node_bw": 50_000_000_000.0,
                    "intra_node_bw": 450_000_000_000.0,
                    "p2p_latency": 0.00001,
                },
                "misc": {"nccl_version": "2.26.2"},
            }
        ),
        encoding="utf-8",
    )
    path = root / "data/h100_sxm/moe_comm/sglang/0.5.10" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=columns).to_parquet(path, index=False)
    # These hand-written rows test schema decoding, not collector provenance.
    return PerfDatabase("h100_sxm", "sglang", "0.5.10", str(root), database_mode="HYBRID", strict_provenance=False)


def test_moe_a2a_header_row_loads_with_us_to_ms_conversion(tmp_path):
    # One 850 us DeepEP HT dispatch measurement: ep_size 8 on 4-GPU nodes.
    row = {
        "framework": "SGLang",
        "version": "0.5.10",
        "device": "NVIDIA GB200",
        "op_name": "moe_a2a",
        "kernel_source": "deepep_ht",
        "comm_backend": "deepep_ht",
        "phase": "dispatch",
        "comm_dtype": "default",
        "ep_size": 8,
        "node_num": 2,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "num_tokens": 4096,
        "sms": 24,
        "transmit_us": 800.0,
        "notify_us": 50.0,
        "latency": 850.0,  # MICROSECONDS — the moe_a2a writer convention
    }
    db = _view_db_over_row(tmp_path, MOE_A2A_HEADER, row, "moe_a2a_perf.parquet")

    data = fetch_table_view(db, "_moe_a2a_data")

    # 10-part nested key: [comm_backend][phase][comm_dtype][ep_size][node_num]
    # [hidden_size][topk][num_experts][sms][num_tokens].
    leaf = data["deepep_ht"]["dispatch"]["default"][8][2][7168][8][256][24][4096]
    assert leaf["latency"] == pytest.approx(0.850)  # us -> ms at load
    assert leaf["power"] == 0.0  # no power column in the frozen header
    assert leaf["energy"] == 0.0
    assert set(leaf.keys()) == {"latency", "power", "energy"}


def test_collector_side_twin_pins_exist_and_freeze_the_same_literals():
    # Text-level check only — this SDK test may not import collector modules.
    # Each twin must contain this file's literal verbatim, so a header change
    # on either side fails one pin before data can drift across the boundary.
    for relative_path, (symbol_name, literal) in _TWIN_PINS.items():
        twin = REPO_ROOT / relative_path
        assert twin.is_file(), f"missing collector-side twin {relative_path}"
        # The frozen literal is a parenthesized implicit concatenation in the
        # twins; compare the named assignment after the parser folds that
        # concatenation instead of accepting the literal under a stale name.
        assignments = _string_assignments(twin.read_text(), str(twin))
        assert assignments.get(symbol_name) == literal, (
            f"{relative_path}::{symbol_name} does not pin the shared header literal"
        )


def _string_assignments(source: str, filename: str) -> dict[str, str]:
    """Top-level named string assignments, implicit concatenation folded."""
    import ast

    assignments = {}
    for node in ast.parse(source, filename=filename).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value.value
    return assignments
