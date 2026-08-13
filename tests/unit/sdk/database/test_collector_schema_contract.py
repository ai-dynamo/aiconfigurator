# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-sided schema contract: collector-written headers -> SDK loaders (D1).

The ``moe_a2a_perf`` and ``moe_expert_compute_perf`` tables are produced by collectors and
consumed by ``aiconfigurator_core.sdk.operations.moe_comm``. Each side pins the
CSV header independently; this module is the SDK-side half. The collector-side
twins (which pin the same literals against the actual writers) are:

- ``tests/unit/collector/test_collect_moe_a2a.py::MOE_A2A_HEADER``
- ``tests/unit/collector/test_collect_trtllm_alltoall.py::MOE_A2A_HEADER``
- ``tests/unit/collector/sglang/test_collect_moe_ep.py::MOE_EP_HEADER``
- ``tests/unit/collector/trtllm/test_collect_moe_ep.py::MOE_EP_HEADER``
- ``tests/unit/collector/test_vllm_collect_moe_ep.py::MOE_EP_HEADER``

This file MUST NOT import anything from ``collector`` (module-boundary rule:
SDK tests do not reach into the collector). The twin literals are verified by
reading the twin test files as text; a drifting header breaks one side's pin
before it can silently break the cross-module contract.

Each test writes ONE synthetic row under the frozen header via pandas,
round-trips it through the real loader, and asserts the nested key plus the
unit convention. The two tables deliberately disagree on raw latency units —
``moe_a2a`` records MICROSECONDS (the loader divides by 1000), ``moe_ep``
records MILLISECONDS (stored raw) — and that sibling divergence is exactly
what these tests keep visible (see "MoE table units and caveats" in
``collector/README.md``).
"""

from pathlib import Path

import pandas as pd
import pytest

from aiconfigurator_core.sdk.common import MoEQuantMode
from aiconfigurator_core.sdk.operations.moe_comm import load_moe_a2a_data, load_moe_expert_compute_data

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

# Copied verbatim from the collector-side writer pins:
# tests/unit/collector/sglang/test_collect_moe_ep.py::MOE_EP_HEADER, repeated
# verbatim by tests/unit/collector/trtllm/test_collect_moe_ep.py and
# tests/unit/collector/test_vllm_collect_moe_ep.py — all three moe_ep writers
# emit this exact header.
MOE_EXPERT_COMPUTE_HEADER = (
    "framework,version,device,op_name,kernel_source,"
    "moe_dtype,distribution,inference_phase,num_tokens,hidden_size,inter_size,"
    "topk,num_experts,num_slots,moe_tp_size,moe_ep_size,latency"
)

# The collector twin files, with the literal each must pin. Paths are relative
# to the repo root; existence itself is part of the contract (Tasks 2-5 landed
# the writers and their pins).
_TWIN_PINS = {
    "tests/unit/collector/test_collect_moe_a2a.py": ("MOE_A2A_HEADER", MOE_A2A_HEADER),
    "tests/unit/collector/test_collect_trtllm_alltoall.py": ("MOE_A2A_HEADER", MOE_A2A_HEADER),
    "tests/unit/collector/sglang/test_collect_moe_ep.py": ("MOE_EP_HEADER", MOE_EXPERT_COMPUTE_HEADER),
    "tests/unit/collector/trtllm/test_collect_moe_ep.py": ("MOE_EP_HEADER", MOE_EXPERT_COMPUTE_HEADER),
    "tests/unit/collector/test_vllm_collect_moe_ep.py": ("MOE_EP_HEADER", MOE_EXPERT_COMPUTE_HEADER),
}


def _write_row_under_header(tmp_path, header: str, row: dict, filename: str) -> str:
    """Write one row to parquet with the header's exact columns and order."""
    columns = header.split(",")
    assert set(row) == set(columns), "synthetic row must cover the frozen header exactly"
    path = tmp_path / filename
    pd.DataFrame([row], columns=columns).to_parquet(path, index=False)
    return str(path)


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
    path = _write_row_under_header(tmp_path, MOE_A2A_HEADER, row, "moe_a2a_perf.parquet")

    data = load_moe_a2a_data([(path, None)])

    # 10-part nested key: [comm_backend][phase][comm_dtype][ep_size][node_num]
    # [hidden_size][topk][num_experts][sms][num_tokens].
    leaf = data["deepep_ht"]["dispatch"]["default"][8][2][7168][8][256][24][4096]
    assert leaf["latency"] == pytest.approx(0.850)  # us -> ms at load
    assert leaf["power"] == 0.0  # no power column in the frozen header
    assert leaf["energy"] == 0.0
    assert set(leaf.keys()) == {"latency", "power", "energy"}


def test_moe_ep_header_row_loads_with_ms_stored_raw(tmp_path):
    # One 0.25 ms generation-phase EP MoE measurement — the moe_ep writer
    # records MILLISECONDS, the opposite of its moe_a2a sibling above.
    row = {
        "framework": "SGLang",
        "version": "0.5.10",
        "device": "NVIDIA GB200",
        "op_name": "moe_ep",
        "kernel_source": "deepep_moe",
        "moe_dtype": "fp8_block",
        "distribution": "power_law_1.01",
        "inference_phase": "generation",
        "num_tokens": 128,
        "hidden_size": 7168,
        "inter_size": 2048,
        "topk": 8,
        "num_experts": 256,
        "num_slots": 288,
        "moe_tp_size": 1,
        "moe_ep_size": 16,
        "latency": 0.25,  # MILLISECONDS — the moe_ep writer convention
    }
    path = _write_row_under_header(tmp_path, MOE_EXPERT_COMPUTE_HEADER, row, "moe_expert_compute_perf.parquet")

    data = load_moe_expert_compute_data([(path, None)])

    # 12-part nested key: [kernel_source][quant][distribution][inference_phase]
    # [topk][num_experts][num_slots][hidden_size][inter_size][moe_tp_size]
    # [moe_ep_size][num_tokens]; moe_dtype becomes a MoEQuantMode enum key.
    leaf = data["deepep_moe"][MoEQuantMode.fp8_block]["power_law_1.01"]["generation"][8][256][288][7168][2048][1][16][
        128
    ]
    assert leaf["latency"] == 0.25  # stored raw — no /1000
    assert leaf["power"] == 0.0
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
